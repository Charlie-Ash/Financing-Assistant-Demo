# Builds the Credit Lookup tool's own SUBGRAPH. Mirrors ragTool/rag/graph.py's
# build_rag_subgraph(): takes the data source, LLM and audit sink as
# parameters instead of constructing them itself, so the wiring can be tested
# with a temporary database and a fake LLM (see tests/test_credit_lookup_agent.py).
#
#   check_caller -> extract_request -> resolve_customer -> lookup -> summarise
#        |               |                  |
#        +-> denied      +-> no customer    +-> not found / several matches
#
# The LLM is used twice and never sees raw data: once to read WHO the
# question is about (structured output), once to summarise the already
# role-filtered and masked facts. Every exit writes one audit line.
from datetime import date
from typing import TYPE_CHECKING, Callable

from langgraph.graph import StateGraph, START, END

from chat_history import history_to_chat_messages
from llm.sampling import SamplingConfig
from llm.text_norm import ZH_TW_REPLY_INSTRUCTION
from tools.creditLookupTool.credit_lookup.policy import (
    DISCLAIMER,
    build_view,
    is_valid_caller,
    render_candidates,
    render_facts,
    scope_line,
)
from tools.creditLookupTool.credit_lookup.request_schema import CREDIT_LOOKUP_REQUEST_SCHEMA, CreditLookupRequest
from tools.creditLookupTool.credit_lookup.state import CreditLookupSubgraphState

# Imported only for type names in string-quoted annotations.
if TYPE_CHECKING:
    from llm.client import ChatLLM
    from tools.creditLookupTool.credit_lookup.datasource import CreditDataSource

EXTRACT_PROMPT = (
    "你是客戶資料查詢系統的解析器。請從使用者的問題中找出要查詢的客戶：\n"
    "- customer_name：客戶姓名，照使用者的寫法填入，沒有則填空字串。\n"
    "- customer_id：客戶編號（C 加四位數字，例如 C0003），沒有則填空字串。\n"
    "- focus：問題重點，contracts（合約）、payments（繳款、逾期）、payoff（提前清償要付多少）或 overview（整體往來）。\n"
    "若問題用「他」、「她」、「這位客戶」等指稱，請從先前的對話中找出所指的客戶。"
)

SUMMARY_PROMPT = (
    "你是授信與業務人員的客戶資料助理。下方是已依查詢者權限過濾、並遮罩個人資料後的客戶資料。"
    "請用二到四句自然、通順的話摘要與使用者問題相關的重點，不要照抄欄位名稱或格式"
    "（完整資料會另外附在摘要下方）。只能使用下方資料中的內容，不得推測或補充；"
    "資料中沒有的項目（例如被權限隱藏的欄位）請說明需洽授信人員，不要猜測。"
    "不得建議或判斷是否核准、拒絕授信。"
    + ZH_TW_REPLY_INSTRUCTION
)

FOCUS_LABELS = {"overview": "整體往來", "contracts": "合約", "payments": "繳款與逾期", "payoff": "提前清償試算"}

EXTRACT_SAMPLING = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=200)
SUMMARY_SAMPLING = SamplingConfig(temperature=0.2, top_p=0.9, max_tokens=300)


def months_before(day: date, months: int) -> date:

    month_index = day.month - 1 - months
    return date(day.year + month_index // 12, month_index % 12 + 1, min(day.day, 28))


def build_credit_lookup_subgraph(
    datasource: "CreditDataSource",
    llm: "ChatLLM",
    audit: Callable[..., None],
    payment_months: int = 12,
):
    """Assemble and compile the Credit Lookup subgraph (see the module
    comment). audit(caller, question, outcome, customer_ids) is called exactly
    once per run, e.g. functools.partial(audit.append_audit, path)."""

    graph = StateGraph(CreditLookupSubgraphState)

    # Ends the run with a reply and its audit line. Results the user should
    # see exactly as written are relayed verbatim; error-ish outcomes are left
    # for the orchestrator to phrase.
    def finish(state: CreditLookupSubgraphState, outcome: str, output: str, customer_ids=(), verbatim=True) -> dict:

        audit(state.get("caller"), state.get("request", ""), outcome, list(customer_ids))
        return {"tool_result": {"output": output, "relay_verbatim": verbatim}}

    # Node: refuse before reading anything if we don't know who is asking.
    def check_caller(state: CreditLookupSubgraphState) -> dict:

        request = state["tool_call"].args.get("request", "")
        if not is_valid_caller(state.get("caller")):
            return {"request": request, **finish(
                {**state, "request": request}, "denied",
                "無法確認查詢者的身分或權限，基於個人資料保護，不提供客戶資料。", verbatim=False,
            )}
        return {"request": request}

    # Node: the LLM reads WHO and WHAT from the question (and, for 「他」,
    # from the recent conversation). Structured output only -- no data yet.
    def extract_request(state: CreditLookupSubgraphState) -> dict:

        history_messages = state.get("messages", [])[:-1]
        messages = [
            {"role": "system", "content": EXTRACT_PROMPT},
            *history_to_chat_messages(history_messages),
            {"role": "user", "content": state["request"]},
        ]
        try:
            parsed = CreditLookupRequest.model_validate_json(llm.chat(messages, EXTRACT_SAMPLING, json_schema=CREDIT_LOOKUP_REQUEST_SCHEMA))
        except Exception as e:
            return finish(state, "error", f"Credit lookup error: could not read the request ({e})", verbatim=False)

        if not (parsed.customer_name.strip() or parsed.customer_id.strip()):
            return finish(state, "no_customer", "請提供要查詢的客戶姓名或客戶編號（例如 C0003）。")

        return {"customer_name": parsed.customer_name.strip(), "customer_id": parsed.customer_id.strip(), "focus": parsed.focus}

    # Node: fixed-query lookup. An id beats a name; a name shared by several
    # customers returns a masked list and no data.
    def resolve_customer(state: CreditLookupSubgraphState) -> dict:

        try:
            matches = datasource.find_customers(name=state["customer_name"], customer_id=state["customer_id"])
        except Exception as e:
            return finish(state, "error", f"Credit lookup error: database unavailable ({e})", verbatim=False)

        if not matches:
            who = state["customer_id"] or state["customer_name"]
            return finish(state, "not_found", f"查無客戶「{who}」的往來資料，請確認姓名或客戶編號。")
        if len(matches) > 1:
            return finish(state, "ambiguous", render_candidates(matches), [m["customer_id"] for m in matches])

        return {"customer": matches[0]}

    # Node: fetch, then filter and mask for the caller's role -- in Python,
    # before the LLM sees anything.
    def lookup(state: CreditLookupSubgraphState) -> dict:

        customer = state["customer"]
        try:
            as_of = datasource.as_of_date()
            contracts = datasource.get_contracts(customer["customer_id"])
            payments = datasource.get_payments(customer["customer_id"], months_before(as_of, payment_months))
        except Exception as e:
            return finish(state, "error", f"Credit lookup error: database unavailable ({e})", verbatim=False)

        return {"view": build_view(
            customer, contracts, payments, as_of, state["caller"]["role"], payment_months,
            include_payoff=state.get("focus") == "payoff",
        )}

    # Node: the LLM summarises the masked facts; the reply itself is built by
    # code around that summary, so the facts are always exact.
    def summarise(state: CreditLookupSubgraphState) -> dict:

        view = state["view"]
        facts = render_facts(view)
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"查詢重點：{FOCUS_LABELS.get(state['focus'], '整體往來')}\n使用者問題：{state['request']}\n\n{facts}"},
        ]
        try:
            summary = llm.chat(messages, SUMMARY_SAMPLING).strip()
        except Exception:
            summary = ""  # the exact facts below are still worth returning

        parts = [summary, facts, scope_line(view["role"], view["as_of"]), DISCLAIMER]
        return finish(state, "ok", "\n\n".join(p for p in parts if p), [view["customer"]["customer_id"]])

    def route(next_node: str) -> Callable[[CreditLookupSubgraphState], str]:
        # Every node either sets tool_result (done) or hands on to the next.
        return lambda state: "done" if state.get("tool_result") else next_node

    graph.add_node("check_caller", check_caller)
    graph.add_node("extract_request", extract_request)
    graph.add_node("resolve_customer", resolve_customer)
    graph.add_node("lookup", lookup)
    graph.add_node("summarise", summarise)

    graph.add_edge(START, "check_caller")
    for node, next_node in [("check_caller", "extract_request"), ("extract_request", "resolve_customer"),
                            ("resolve_customer", "lookup"), ("lookup", "summarise")]:
        graph.add_conditional_edges(node, route(next_node), {"done": END, next_node: next_node})
    graph.add_edge("summarise", END)

    return graph.compile()
