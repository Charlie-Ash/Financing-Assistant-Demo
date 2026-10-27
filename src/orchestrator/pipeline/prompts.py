# Orchestrator prompt building. Prompts are in Traditional Chinese (Taiwan)
# since that's the language users write in and replies must come back in --
# see llm/text_norm.py for the OpenCC safety net applied to every output.
from chat_history import history_to_chat_messages as _history_to_chat_messages
from llm.text_norm import ZH_TW_REPLY_INSTRUCTION


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
        f"""使用者: {tool['example_user_message']}
        輸出:
        {{
            "tool": "{tool['name']}",
            "action": "run",
            "args": {{}}
        }}"""
        for tool in tool_descriptions
    )

    SYSTEM_PROMPT = f"""
        你是一個多代理系統的協調者(orchestrator)。
        你的任務是根據使用者的訊息，從系統中選出最適合處理這項工作的工具。
        連接到其他代理的工具，請依使用者訊息的意圖正確分派。
        -----------------------
        可用工具
        -----------------------

        {tools_block}

        -----------------------
        範例
        -----------------------

        {examples_block}
    """

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

def build_response_prompt(user_message, history_messages, tool_call, tool_result):

    # Second-stage prompt: Plain-text reply. Only used for results that need
    # composing -- relay_verbatim results are returned as-is by
    # OrchestratorLLM.generate_response without an LLM call.
    instruction = "請根據下方的工具結果，給使用者一個簡潔且有幫助的回覆。"

    SYSTEM_PROMPT = f"""
        你是同一位協調者，現在要直接回覆使用者。
        你已經把使用者的訊息分派給「{tool_call.tool}」工具，而它已產生結果。
        {instruction}
        不要提及工具名稱、JSON 或任何內部分派細節。
        {ZH_TW_REPLY_INSTRUCTION}

        工具結果:
        {tool_result["output"]}
    """

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
        result_section = f"""
        本輪使用了 RAG 工具。請將記憶筆記寫成兩行並加上標籤:
        問題: <簡潔重述使用者的問題>
        答案: <濃縮下方 RAG 回答中的關鍵事實>

        RAG 原始回答（逐字，請以此為準，而非最終回覆）:
        {result_text}
        """
    else:
        result_section = f"""
        使用的工具: {tool_call.tool}
        工具結果: {result_text}
        """

    SYSTEM_PROMPT = f"""
        你是這段與代理系統持續對話的記憶管理者。
        請寫一則簡潔的記憶筆記（2-4 句，若下方有指示則使用「問題/答案」格式），
        摘要本輪使用者問了什麼、得到了什麼資訊或做了什麼決定。
        這則筆記之後會由你自己在後續回合讀回，因此請優先記錄具體的事實、名稱、決定與檢索到的答案，
        而不是模糊的描述。不要重述這些指示。只輸出筆記內容，不要加標題。
        {ZH_TW_REPLY_INSTRUCTION}

        {result_section}

        協調者本輪給使用者的最終回覆: {final_response}
    """

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
    SYSTEM_PROMPT = f"""
        你要為一份已儲存的對話記憶檔命名，方便日後查閱。
        請根據下方的對話記憶，輸出一個簡短的英文主題代稱(slug)：2-5 個英文單字，
        全部小寫、單字之間以底線連接，不含標點符號與副檔名，
        且不要包含 "chat" 或 "points" 這兩個字（系統會自動加上後綴）。
        只輸出代稱本身，不要輸出其他任何內容。

        對話記憶:
        {memory_content}
    """

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
