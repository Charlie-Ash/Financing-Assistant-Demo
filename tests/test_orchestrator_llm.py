# The REAL OrchestratorLLM's reply stage, driven by FakeChatLLM: a verbatim
# result (the regulation agent's cited answer) must reach the user exactly
# as the agent wrote it, without a second LLM pass that could trim its
# source list or disclaimer; other results are still composed by the LLM.
from schemas.tool_call import ToolCall
from orchestrator.pipeline.llm import OrchestratorLLM

from fake_dependencies import FakeChatLLM

CITED_ANSWER = "依《消費者保護法》第 21 條……\n\n參考來源：\n- 《消費者保護法》第 21 條\n\n以上僅供參考。"


def test_verbatim_result_is_returned_unchanged_without_an_llm_call():

    chat = FakeChatLLM("should not be used")
    llm = OrchestratorLLM(tool_schema={}, chat_llm=chat)

    reply = llm.generate_response(
        "分期契約要寫什麼？", [], ToolCall(tool="rag", action="run", args={}),
        {"output": CITED_ANSWER, "relay_verbatim": True},
    )

    assert reply == CITED_ANSWER
    assert chat.calls == []


def test_non_verbatim_result_is_composed_by_the_llm():

    chat = FakeChatLLM("  已幫您記下。  ")
    llm = OrchestratorLLM(tool_schema={}, chat_llm=chat)

    reply = llm.generate_response(
        "幫我記下", [], ToolCall(tool="actions", action="run", args={}),
        {"output": "Note saved.", "relay_verbatim": False},
    )

    assert reply == "已幫您記下。"
    assert "Note saved." in chat.calls[0]["messages"][0]["content"]
