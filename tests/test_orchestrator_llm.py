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

    chat = FakeChatLLM("  抱歉，目前無法確認您的身分，無法提供客戶資料。  ")
    llm = OrchestratorLLM(tool_schema={}, chat_llm=chat)

    reply = llm.generate_response(
        "王大明有沒有逾期？", [], ToolCall(tool="credit_lookup", action="run", args={}),
        {"output": "無法確認查詢者的身分或權限。", "relay_verbatim": False},
    )

    assert reply == "抱歉，目前無法確認您的身分，無法提供客戶資料。"
    assert "無法確認查詢者的身分或權限。" in chat.calls[0]["messages"][0]["content"]
