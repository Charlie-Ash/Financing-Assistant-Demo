# The REAL Loan Calculator subgraph with FakeChatLLM standing in for the
# extraction call. The key property: a quote is only ever computed from
# numbers the user actually wrote -- anything missing is asked for, anything
# the model invents is ignored, and every default is listed as an assumption.
import json

from langchain_core.messages import AIMessage, HumanMessage

from finance.plans import load_plans
from schemas.tool_call import ToolCall
from tools.loanCalcTool.a2a.a2a_server import build_agent_card
from tools.loanCalcTool.calculator.graph import build_loan_calc_subgraph
from tools.loanCalcTool.calculator.request_schema import LOAN_CALC_REQUEST_SCHEMA

from fake_dependencies import FakeChatLLM

EMPTY = {"vehicle_price": 0, "down_payment_amount": 0, "down_payment_percent": 0, "loan_amount": 0,
         "term_months": 0, "annual_rate_percent": 0, "residual_percent": 0}


def _extraction(**values) -> str:
    return json.dumps({**EMPTY, **values})


def _ask(question, llm, history=()):
    subgraph = build_loan_calc_subgraph(llm, load_plans())
    result = subgraph.invoke({
        "tool_call": ToolCall(tool="loan_calc", action="run", args={"request": question}),
        "messages": [*history, HumanMessage(content=question)],
        "caller": {"user_id": "wang01", "role": "sales"},
    })
    return result["tool_result"]


def test_balloon_quote_is_computed_by_code_with_its_assumptions_listed():

    llm = FakeChatLLM(_extraction(vehicle_price=1_200_000, down_payment_percent=20, term_months=48))
    result = _ask("RAV4 120萬，頭期兩成，殘值型48期，月付多少？", llm)

    output = result["output"]
    assert result["relay_verbatim"] is True
    assert output.startswith("【貸款試算】殘值型車貸")
    assert "貸款金額 NT$960,000，期數 48 期，年利率 3.5%" in output
    assert "每月應付 **NT$14,464**" in output
    assert "殘值 NT$360,000" in output
    assert "未指定利率，依「殘值型車貸」方案以 3.5% 試算" in output
    assert "未指定殘值" in output
    assert "總費用年百分率 約" in output
    assert llm.calls[0]["json_schema"] == LOAN_CALC_REQUEST_SCHEMA
    assert len(llm.calls) == 1  # extraction only -- no LLM writes the reply


def test_an_invented_price_is_ignored_and_the_user_is_asked():

    # The model "helpfully" fills in a typical RAV4 price and term.
    llm = FakeChatLLM(_extraction(vehicle_price=1_200_000, term_months=60))
    result = _ask("RAV4 月付多少？", llm)

    output = result["output"]
    assert "還需要以下資訊" in output
    assert "貸款金額，或車價" in output and "期數" in output
    assert "每月應付" not in output and "1,200,000" not in output


def test_a_missing_term_is_asked_for_and_known_values_are_echoed():

    llm = FakeChatLLM(_extraction(vehicle_price=1_200_000, down_payment_percent=20))
    output = _ask("車價120萬，頭期兩成，月付多少？", llm)["output"]

    assert "期數（「一般車貸」可選 12～84 期）" in output
    assert "目前已知：車價 NT$1,200,000、頭期款 20%" in output
    assert "每月應付" not in output


def test_default_plan_and_full_financing_are_stated_as_assumptions():

    llm = FakeChatLLM(_extraction(loan_amount=600_000, term_months=36))
    output = _ask("借60萬，分36期，月付多少？", llm)["output"]

    assert output.startswith("【貸款試算】一般車貸｜本息平均攤還")
    assert "未指定方案，依「一般車貸」" in output
    assert "未指定利率，依「一般車貸」方案以 3.88% 試算" in output


def test_zero_rate_plan_shows_the_fee_driven_apr():

    llm = FakeChatLLM(_extraction(loan_amount=600_000, term_months=24))
    output = _ask("0利率24期，借60萬，每月繳多少？", llm)["output"]

    assert "【貸款試算】0利率專案" in output
    assert "每月應付 **NT$25,000**" in output
    assert "總費用年百分率 約 1.29%" in output
    assert "未指定利率" not in output  # 「0利率」 states the rate


def test_equal_principal_is_used_when_asked_for():

    llm = FakeChatLLM(_extraction(loan_amount=600_000, term_months=36))
    output = _ask("借60萬分36期，用本金平均攤還的話每月多少？", llm)["output"]

    assert "本金平均攤還" in output and "首期應付" in output and "逐月遞減" in output


def test_out_of_range_and_inconsistent_inputs_stop_the_calculation():

    term = _ask("殘值型 100 期，車價120萬", FakeChatLLM(_extraction(vehicle_price=1_200_000, term_months=100)))["output"]
    assert "可選期數為 36／48／60 期" in term and "每月應付" not in term

    conflict = _ask("車價120萬，頭期30萬也就是兩成，分60期",
                    FakeChatLLM(_extraction(vehicle_price=1_200_000, down_payment_amount=300_000, down_payment_percent=20, term_months=60)))["output"]
    assert "不一致" in conflict and "每月應付" not in conflict

    too_much_down = _ask("車價50萬，頭期60萬，分36期",
                         FakeChatLLM(_extraction(vehicle_price=500_000, down_payment_amount=600_000, term_months=36)))["output"]
    assert "無需貸款" in too_much_down

    rate = _ask("借60萬，36期，利率25%", FakeChatLLM(_extraction(loan_amount=600_000, term_months=36, annual_rate_percent=25)))["output"]
    assert "超出試算範圍" in rate


def test_follow_up_reuses_numbers_the_user_stated_earlier():

    history = [HumanMessage(content="車價120萬，頭期兩成，分48期，月付多少？"), AIMessage(content="【貸款試算】…每月應付 **NT$21,xxx**")]
    llm = FakeChatLLM(_extraction(vehicle_price=1_200_000, down_payment_percent=20, term_months=60))
    output = _ask("那改成60期呢？", llm, history)["output"]

    assert "期數 60 期" in output and "貸款金額 NT$960,000" in output


def test_numbers_from_the_assistants_own_replies_do_not_count():

    # 21,000 only appears in an earlier reply, never in what the user wrote.
    history = [HumanMessage(content="分48期月付多少？"), AIMessage(content="每月應付 NT$21,000")]
    output = _ask("那貸款金額呢？", FakeChatLLM(_extraction(loan_amount=21_000, term_months=48)), history)["output"]

    assert "還需要以下資訊" in output


def test_agent_card_describes_the_calculator():

    card = build_agent_card("localhost", 8003)

    assert card.name == "Loan Calculator Agent"
    assert card.skills[0].id == "quote_car_loan"
    assert "提前清償金額請改用客戶查詢" in card.skills[0].description
