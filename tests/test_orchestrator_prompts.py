# The orchestrator's reply prompt must say who the assistant is and what this
# chat can actually do -- without that, a question like 「你能協助我甚麼？」 on
# the general-chat route was answered as a generic chatbot (writing,
# translation, coding...). Also checks every prompt reaches the model
# without the source code's indentation.
from orchestrator.agents.tool_router import DEFAULT_TOOL_DESCRIPTION
from orchestrator.pipeline.llm import OrchestratorLLM
from orchestrator.pipeline.prompts import (
    build_assistant_profile,
    build_filename_prompt,
    build_mem_update_prompt,
    build_response_prompt,
    build_tool_decision_prompt,
)
from schemas.tool_call import ToolCall

from fake_dependencies import FakeChatLLM

RAG = {"name": "rag", "description": "查詢法規與公司內部作業規範並回答問題。", "example_user_message": "提前清償要付違約金嗎？"}
CREDIT = {"name": "credit_lookup", "description": "查詢特定客戶的往來紀錄。", "example_user_message": "王大明有沒有逾期紀錄？"}
SALES = {"user_id": "wang01", "role": "sales"}
DEFAULT_CALL = ToolCall(tool="default", action="run", args={})
DEFAULT_RESULT = {"output": "Default tool has completed.", "relay_verbatim": False}


def test_profile_lists_the_connected_agents_with_examples_and_general_chat():

    profile = build_assistant_profile([DEFAULT_TOOL_DESCRIPTION, RAG, CREDIT], SALES)

    assert "融資業務助理" in profile
    assert "法規與內規查詢" in profile and "提前清償要付違約金嗎？" in profile
    assert "客戶往來查詢" in profile and "王大明有沒有逾期紀錄？" in profile
    assert "一般對話" in profile


def test_profile_does_not_claim_an_agent_that_is_down():

    profile = build_assistant_profile([DEFAULT_TOOL_DESCRIPTION, RAG], SALES)

    assert "法規與內規查詢" in profile
    assert "客戶往來查詢" not in profile


def test_profile_describes_what_the_callers_role_can_see():

    sales = build_assistant_profile([RAG, CREDIT], SALES)
    credit = build_assistant_profile([RAG, CREDIT], {"user_id": "lin02", "role": "credit"})

    assert "業務：" in sales and "需洽授信人員" in sales
    assert "授信：" in credit and "內部註記" in credit
    assert "wang01" in sales and "lin02" in credit


def test_profile_states_the_boundaries_and_the_off_topic_rule():

    profile = build_assistant_profile([RAG, CREDIT], SALES)

    assert "不做授信核准或拒絕" in profile
    assert "不構成法律意見" in profile
    assert "不要聲稱能做清單以外的事" in profile
    assert "簡短回應" in profile and "笑話" in profile and "不在本系統的服務範圍" in profile


def test_general_chat_reply_prompt_has_the_profile_and_no_placeholder_result():

    profile = build_assistant_profile([RAG, CREDIT], SALES)
    system = build_response_prompt("請問你能協助我甚麼?", [], DEFAULT_CALL, DEFAULT_RESULT, profile)[0]["content"]

    assert system.startswith("你是「融資業務助理」")
    assert "這是一般對話" in system
    assert "Default tool has completed." not in system


def test_other_results_are_still_shown_to_the_reply_stage():

    system = build_response_prompt(
        "王大明有沒有逾期？", [], ToolCall(tool="credit_lookup", action="run", args={}),
        {"output": "無法確認查詢者的身分或權限。", "relay_verbatim": False}, "PROFILE",
    )[0]["content"]

    assert system.startswith("PROFILE")
    assert "無法確認查詢者的身分或權限。" in system


def test_orchestrator_llm_sends_its_profile_with_the_reply_call():

    chat = FakeChatLLM("你好！我可以協助查詢法規與客戶往來紀錄。")
    llm = OrchestratorLLM(tool_schema={}, chat_llm=chat, assistant_profile="我是融資業務助理。")

    llm.generate_response("你好", [], DEFAULT_CALL, DEFAULT_RESULT)

    assert chat.calls[0]["messages"][0]["content"].startswith("我是融資業務助理。")


def test_routing_prompt_carries_the_identity_and_the_updated_default_route():

    system = build_tool_decision_prompt("你能做什麼？", [], [DEFAULT_TOOL_DESCRIPTION, RAG, CREDIT])[0]["content"]

    assert system.startswith("你是「融資業務助理」")
    assert "詢問本系統能做什麼" in system
    assert "記錄筆記" not in system


def test_no_prompt_reaches_the_model_with_code_indentation():

    profile = build_assistant_profile([DEFAULT_TOOL_DESCRIPTION, RAG, CREDIT], SALES)
    prompts = [
        build_tool_decision_prompt("你好", [], [DEFAULT_TOOL_DESCRIPTION, RAG, CREDIT])[0]["content"],
        build_response_prompt("你好", [], DEFAULT_CALL, DEFAULT_RESULT, profile)[0]["content"],
        build_mem_update_prompt("你好", DEFAULT_CALL, DEFAULT_RESULT, "你好！")[0]["content"],
        build_mem_update_prompt("違約金？", ToolCall(tool="rag", action="run", args={}), {"output": "依第2條…", "relay_verbatim": True}, "依第2條…")[0]["content"],
        build_filename_prompt("Summary to question 1: …")[0]["content"],
    ]

    for prompt in prompts:
        assert not any(line.startswith("    ") for line in prompt.splitlines()), prompt
