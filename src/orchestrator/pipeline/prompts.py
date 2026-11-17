# Orchestrator prompt building. Prompts are in Traditional Chinese (Taiwan)
# since that's the language users write in and replies must come back in --
# see llm/text_norm.py for the OpenCC safety net applied to every output.
#
# Every prompt is dedented before it's sent: the f-strings below are indented
# to match the code, and that indentation would otherwise reach the model as
# leading spaces on every line.
from textwrap import dedent

from chat_history import history_to_chat_messages as _history_to_chat_messages
from llm.text_norm import ZH_TW_REPLY_INSTRUCTION

ASSISTANT_NAME = "融資業務助理"

# One line of identity shared by the routing and reply prompts.
IDENTITY = f"你是「{ASSISTANT_NAME}」，一套供汽車融資公司業務與授信人員內部使用的多代理助理系統。"

# How each route is presented to users when the assistant describes itself.
CAPABILITY_LABELS = {"rag": "法規與內規查詢", "credit_lookup": "客戶往來查詢", "loan_calc": "貸款試算"}

ROLE_VIEWS = {
    "sales": "業務：可查看客戶基本資料、合約與繳款狀態，以及近12個月是否有逾期紀錄（僅有／無）；"
             "月付金額、未償餘額、逾期明細與內部註記需洽授信人員。",
    "credit": "授信：可查看客戶基本資料、合約、月付金額、未償餘額、逾期次數與各期明細，以及內部註記。",
}


def build_assistant_profile(tool_descriptions, caller) -> str:

    """Who the assistant is and what it can do in THIS chat -- built once per
    chat from the agents actually connected (ToolRouter.describe_tools_for_prompt)
    and the caller's role, and placed at the top of every reply prompt, so a
    question like 「你能做什麼？」 is answered about this system rather than
    about a generic chatbot."""

    capabilities = []
    for tool in tool_descriptions:
        label = CAPABILITY_LABELS.get(tool["name"])
        if label:
            capabilities.append(f"- {label}：{tool['description']}（例如：「{tool['example_user_message']}」）")
    capabilities.append("- 一般對話：問候、說明本系統的功能與使用方式。")

    role = (caller or {}).get("role")
    role_line = ROLE_VIEWS.get(role, "目前使用者的身分未知，客戶資料查詢可能不會提供資料。")
    user_id = (caller or {}).get("user_id", "")

    capability_block = "\n".join(capabilities)
    return dedent(f"""\
        {IDENTITY}

        你目前能提供的功能（只有這些）：
        {{capabilities}}

        目前使用者：{user_id}，{role_line}

        其他說明：
        - 結束對話時，可以選擇儲存這次對話的摘要並下載。
        - 系統只提供資料摘要，不做授信核准或拒絕的判斷；法規回覆僅供參考，不構成法律意見。
        - 客戶資料為展示用的虛構資料，身分證字號與電話均已遮罩。

        回覆原則：
        - 使用者詢問你能做什麼、如何使用時，只根據上方功能清單說明，並提供一兩個可以直接提問的例子。
          不要聲稱能做清單以外的事，例如撰寫文章或信件、翻譯、寫程式、規劃行程。
        - 閒聊（例如問候、請你講個簡短的笑話）或簡單的常識問題，請直接簡短回應，再用一句話提醒使用者本系統能協助的範圍。
        - 較大的、與融資業務無關的工作（例如撰寫信件、翻譯、寫程式、規劃行程），請禮貌說明這不在本系統的服務範圍，並介紹可以協助的事項。
        """).replace("{capabilities}", capability_block)


def build_tool_decision_prompt(user_message, history_messages, tool_descriptions):

    # First stage prompt: Tool Route
    # JSON shape enforced by guided decoding (see schemas/tool_schema.py),
    # so this prompt only needs to cover tool semantics, not output formatting.
    # tool_descriptions comes from ToolRouter.describe_tools_for_prompt() --
    # a list of {"name", "description", "example_user_message"} dicts, one
    # per available tool. "default"'s is hand-written (no AgentCard to pull
    # from); every remote agent's comes straight from its own discovered
    # AgentCard, so this prompt can't drift out of sync with what that agent
    # actually advertises. args is always {} in every example now -- no tool
    # reads tool_call.args anymore (remote agents just get the raw user
    # message as A2A message text; "default" ignores its args entirely).
    tools_block = "\n\n".join(
        f"{i}. {tool['name']}\n{tool['description']}\nargs: {{}}"
        for i, tool in enumerate(tool_descriptions, start=1)
    )

    examples_block = "\n\n".join(
        f'使用者: {tool["example_user_message"]}\n輸出:\n{{"tool": "{tool["name"]}", "action": "run", "args": {{}}}}'
        for tool in tool_descriptions
    )

    # The blocks are inserted after dedent(), so their own lines (which start
    # at column 0) don't stop the template's indentation from being removed.
    SYSTEM_PROMPT = dedent(f"""\
        {IDENTITY}
        你現在的任務是擔任協調者(orchestrator)：根據使用者的訊息，從系統中選出最適合處理這項工作的工具。
        連接到其他代理的工具，請依使用者訊息的意圖正確分派。
        -----------------------
        可用工具
        -----------------------

        {{tools}}

        -----------------------
        範例
        -----------------------

        {{examples}}
        """).replace("{tools}", tools_block).replace("{examples}", examples_block)

    # Have the system prompt and user message split, avoiding using a single, excessivly long prompt that may cause unexpected behaviors.
    # Prior turns (if any) are spliced in between as real chat turns, so the
    # model sees actual history instead of a summary folded into the system prompt.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *_history_to_chat_messages(history_messages),
        {
            "role": "user",
            "content": user_message,
        }
    ]

    return messages

def build_response_prompt(user_message, history_messages, tool_call, tool_result, assistant_profile=""):

    # Second-stage prompt: Plain-text reply. Only used for results that need
    # composing -- relay_verbatim results are returned as-is by
    # OrchestratorLLM.generate_response without an LLM call.
    # assistant_profile (build_assistant_profile) goes first, so the model
    # always knows what system it is speaking for.
    if tool_call.tool == "default":
        # The default route looks nothing up -- its "result" is a
        # placeholder, so it's left out rather than shown to the model.
        task = dedent("""\
            這是一般對話，沒有查詢任何法規或客戶資料。
            請依照上方的身分與回覆原則，直接以自然、簡潔的方式回覆使用者。""")
    else:
        task = dedent("""\
            你已經把使用者的訊息交給系統中對應的功能處理，下方是它的結果。
            請根據這個結果，給使用者一個簡潔且有幫助的回覆。""") + f"\n\n結果:\n{tool_result['output']}"

    SYSTEM_PROMPT = "\n\n".join(part for part in (
        assistant_profile.strip(),
        task,
        "不要提及工具名稱、JSON 或任何內部分派細節。" + ZH_TW_REPLY_INSTRUCTION,
    ) if part)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *_history_to_chat_messages(history_messages),
        {
            "role": "user",
            "content": user_message,
        }
    ]

    return messages


# Third-stage prompt: decides what's worth remembering from this turn.
# tool_result is a ToolResult (see tools/base.py): {"output": str, "relay_verbatim": bool}.
def build_mem_update_prompt(user_message, tool_call, tool_result, final_response):

    result_text = tool_result["output"]

    # Known soft spot: this string comparison assumes the RAG agent is
    # always configured under the name "rag" (orchestrator/config.py's
    # REMOTE_AGENTS default). If an operator renames it there, this branch
    # silently stops firing -- not fixed here, since the tool name is no
    # longer a closed enum this file could validate against.
    if tool_call.tool == "rag":
        # RAG's raw answer is the ground truth for this turn -- prefer it over
        # final_response, which may have paraphrased or trimmed it. No more
        # string-splitting needed: tool_result["output"] is already just the
        # answer text, since relay_verbatim is carried as its own field now.
        result_section = dedent("""\
            本輪使用了 RAG 工具。請將記憶筆記寫成兩行並加上標籤:
            問題: <簡潔重述使用者的問題>
            答案: <濃縮下方 RAG 回答中的關鍵事實>

            RAG 原始回答（逐字，請以此為準，而非最終回覆）:
            """) + result_text
    else:
        result_section = f"使用的工具: {tool_call.tool}\n工具結果: {result_text}"

    SYSTEM_PROMPT = dedent(f"""\
        你是這段與代理系統持續對話的記憶管理者。
        請寫一則簡潔的記憶筆記（2-4 句，若下方有指示則使用「問題/答案」格式），
        摘要本輪使用者問了什麼、得到了什麼資訊或做了什麼決定。
        這則筆記之後會由你自己在後續回合讀回，因此請優先記錄具體的事實、名稱、決定與檢索到的答案，
        而不是模糊的描述。不要重述這些指示。只輸出筆記內容，不要加標題。
        {ZH_TW_REPLY_INSTRUCTION}

        {{result}}

        協調者本輪給使用者的最終回覆: {{final}}
        """).replace("{result}", result_section).replace("{final}", final_response)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        }
    ]

    return messages


# Fourth-stage prompt: names the saved memory file at end-of-session ("bye" + "y").
def build_filename_prompt(memory_content):

    # The slug stays in English on purpose: it becomes a file name, and
    # memory_manager's sanitizer only keeps ASCII letters/digits.
    SYSTEM_PROMPT = dedent("""\
        你要為一份已儲存的對話記憶檔命名，方便日後查閱。
        請根據下方的對話記憶，輸出一個簡短的英文主題代稱(slug)：2-5 個英文單字，
        全部小寫、單字之間以底線連接，不含標點符號與副檔名，
        且不要包含 "chat" 或 "points" 這兩個字（系統會自動加上後綴）。
        只輸出代稱本身，不要輸出其他任何內容。

        對話記憶:
        """) + memory_content

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "請現在提供代稱。",
        }
    ]

    return messages
