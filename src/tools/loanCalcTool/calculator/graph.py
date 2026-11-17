# Builds the Loan Calculator tool's own SUBGRAPH:
#
#   extract  (LLM: the numbers the user stated, as JSON -- its only job)
#     -> resolve  (code: ground every number, pick the plan, ask for anything
#                  missing, apply stated defaults, reject bad inputs)
#     -> calculate (code: finance/loan_calc.py, then render the reply)
#
# Like the other agents' subgraphs, the LLM is passed in so the wiring can be
# tested with a fake (see tests/test_loan_calc_agent.py). The reply goes to
# the user verbatim: every number in it comes from code.
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from chat_history import history_to_chat_messages
from finance.loan_calc import quote
from finance.plans import Plan
from llm.sampling import SamplingConfig
from tools.loanCalcTool.calculator.render import render_quote
from tools.loanCalcTool.calculator.request_schema import LOAN_CALC_REQUEST_SCHEMA, LoanCalcRequest
from tools.loanCalcTool.calculator.resolve import resolve_request
from tools.loanCalcTool.calculator.state import LoanCalcSubgraphState

if TYPE_CHECKING:
    from llm.client import ChatLLM

EXTRACT_PROMPT = (
    "你是貸款試算的資料擷取器。請從使用者的問題（以及先前的對話）中，找出使用者明確提到的數字，填入對應欄位。\n"
    "沒有提到的欄位一律填 0，絕對不要自行推測或填入常見數值（例如一般車價、常見利率或期數）。\n"
    "- vehicle_price：車價（元）\n"
    "- down_payment_amount：頭期款金額（元）；down_payment_percent：頭期款比例（%，例如兩成 = 20）\n"
    "- loan_amount：貸款金額（元）\n"
    "- term_months：期數（月；若說幾年，換算為月）\n"
    "- annual_rate_percent：年利率（%）\n"
    "- residual_percent：殘值比例（%）\n"
    "金額請換算為元，例如 120萬 = 1200000。若使用者說「改成…」，以新的數值取代，其餘沿用先前對話中使用者提到的數字。"
)

EXTRACT_SAMPLING = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=200)


def build_loan_calc_subgraph(llm: "ChatLLM", plans: dict[str, Plan]):

    graph = StateGraph(LoanCalcSubgraphState)

    def done(output: str, verbatim: bool = True) -> dict:
        return {"tool_result": {"output": output, "relay_verbatim": verbatim}}

    # Node: the LLM reads the numbers out of the question (and, for a
    # follow-up like 「改成 60 期」, out of the earlier conversation).
    def extract(state: LoanCalcSubgraphState) -> dict:

        request = state["tool_call"].args.get("request", "")
        history_messages = state.get("messages", [])[:-1]
        messages = [
            {"role": "system", "content": EXTRACT_PROMPT},
            *history_to_chat_messages(history_messages),
            {"role": "user", "content": request},
        ]
        try:
            extracted = LoanCalcRequest.model_validate_json(llm.chat(messages, EXTRACT_SAMPLING, json_schema=LOAN_CALC_REQUEST_SCHEMA))
        except Exception as e:
            return {"request": request, **done(f"Loan calculator error: could not read the request ({e})", verbatim=False)}
        return {"request": request, "extracted": extracted}

    # Node: plain-Python checks. Ends the run with a question or an
    # explanation whenever the inputs aren't complete and consistent.
    def resolve(state: LoanCalcSubgraphState) -> dict:

        # Only the user's own words count as a source of numbers -- earlier
        # replies contain computed figures the user never stated.
        earlier = [m.content for m in state.get("messages", [])[:-1] if isinstance(m, HumanMessage)]
        user_texts = [state["request"], *reversed(earlier)]
        resolution = resolve_request(state["extracted"], user_texts, plans)
        if resolution.message:
            return {"resolution": resolution, **done(resolution.message)}
        return {"resolution": resolution}

    # Node: the arithmetic and the reply, both code.
    def calculate(state: LoanCalcSubgraphState) -> dict:

        r = state["resolution"]
        result = quote(r.loan_amount, r.annual_rate_percent / 100, r.term_months, r.method, r.balloon, r.fees)
        return done(render_quote(r, result))

    def route(next_node: str):
        return lambda state: "done" if state.get("tool_result") else next_node

    graph.add_node("extract", extract)
    graph.add_node("resolve", resolve)
    graph.add_node("calculate", calculate)

    graph.add_edge(START, "extract")
    graph.add_conditional_edges("extract", route("resolve"), {"done": END, "resolve": "resolve"})
    graph.add_conditional_edges("resolve", route("calculate"), {"done": END, "calculate": "calculate"})
    graph.add_edge("calculate", END)

    return graph.compile()
