# Runs the REAL Credit Lookup subgraph against a real SQLite database built
# by the real seed_db on a temp path, with FakeChatLLM standing in for the
# LLM (its first reply is the extraction JSON, its second the summary) and a
# list standing in for the audit log. Covers every exit: denied, no
# customer, not found, ambiguous, and role-filtered success.
import json
import sqlite3
from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from schemas.tool_call import ToolCall
from tools.creditLookupTool.credit_lookup.datasource import SqliteCreditDataSource
from tools.creditLookupTool.credit_lookup.graph import build_credit_lookup_subgraph
from tools.creditLookupTool.credit_lookup.request_schema import CREDIT_LOOKUP_REQUEST_SCHEMA
from tools.creditLookupTool.seed_db import build_database, is_valid_national_id

from fake_dependencies import FakeChatLLM

SALES = {"user_id": "wang01", "role": "sales"}
CREDIT = {"user_id": "lin02", "role": "credit"}


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):

    path = tmp_path_factory.mktemp("credit") / "credit_demo.db"
    build_database(path, as_of=date(2026, 10, 21))
    return path


def _extraction(name="", customer_id="", focus="overview") -> str:
    return json.dumps({"customer_name": name, "customer_id": customer_id, "focus": focus}, ensure_ascii=False)


def _run(db_path, question, caller, llm, history=()):

    audit = []
    subgraph = build_credit_lookup_subgraph(
        SqliteCreditDataSource(db_path), llm,
        lambda caller, question, outcome, ids: audit.append({"caller": caller, "outcome": outcome, "ids": ids}),
    )
    result = subgraph.invoke({
        "tool_call": ToolCall(tool="credit_lookup", action="run", args={"request": question}),
        "messages": [*history, HumanMessage(content=question)],
        "caller": caller,
    })
    return result["tool_result"], audit


def _raw(db_path, customer_id):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT national_id, phone FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()


def test_seeded_database_has_the_demo_customers_and_only_fake_ids(db_path):

    with sqlite3.connect(db_path) as conn:
        names = [row[0] for row in conn.execute("SELECT name FROM customers WHERE customer_id <= 'C0005' ORDER BY customer_id")]
        ids = [row[0] for row in conn.execute("SELECT national_id FROM customers")]
    assert names == ["王大明", "陳美玲", "林志強", "王志明", "王志明"]
    assert not any(is_valid_national_id(i) for i in ids)


def test_missing_caller_is_denied_without_touching_the_llm(db_path):

    llm = FakeChatLLM(_extraction("王大明"))
    result, audit = _run(db_path, "王大明有沒有逾期紀錄？", None, llm)

    assert "不提供客戶資料" in result["output"]
    assert llm.calls == []
    assert audit[0]["outcome"] == "denied"


def test_question_without_a_customer_asks_for_one(db_path):

    result, audit = _run(db_path, "有沒有人逾期？", SALES, FakeChatLLM(_extraction()))

    assert "客戶姓名或客戶編號" in result["output"]
    assert audit[0]["outcome"] == "no_customer"


def test_unknown_customer_is_reported_not_found(db_path):

    result, audit = _run(db_path, "張無忌的合約", SALES, FakeChatLLM(_extraction("張無忌")))

    assert "查無客戶「張無忌」" in result["output"]
    assert audit[0]["outcome"] == "not_found"


def test_shared_name_returns_a_masked_candidate_list_and_no_data(db_path):

    llm = FakeChatLLM(_extraction("王志明"))
    result, audit = _run(db_path, "王志明的合約狀況", SALES, llm)

    output = result["output"]
    assert "C0004" in output and "C0005" in output
    for customer_id in ("C0004", "C0005"):
        national_id, _ = _raw(db_path, customer_id)
        assert national_id not in output and national_id[0] + "******" + national_id[-3:] in output
    assert "合約】" not in output  # no record data
    assert len(llm.calls) == 1  # extraction only, no summary
    assert audit[0] == {"caller": SALES, "outcome": "ambiguous", "ids": ["C0004", "C0005"]}


def test_sales_lookup_shows_only_the_late_flag_and_masked_ids(db_path):

    llm = FakeChatLLM([_extraction("林志強", focus="payments"), "林志強目前有逾期紀錄。"])
    result, audit = _run(db_path, "林志強有沒有逾期？", SALES, llm)

    output = result["output"]
    national_id, phone = _raw(db_path, "C0003")
    assert output.startswith("林志強目前有逾期紀錄。")
    assert "近12個月有逾期紀錄：有" in output and "「業務」權限" in output
    assert "未償餘額" not in output and "內部註記" not in output and "逾期次數" not in output
    assert national_id not in output and phone not in output
    assert result["relay_verbatim"] is True
    assert audit[0] == {"caller": SALES, "outcome": "ok", "ids": ["C0003"]}


def test_credit_lookup_shows_detail_and_risk_note(db_path):

    llm = FakeChatLLM([_extraction(customer_id="C0003", focus="payments"), "摘要"])
    result, _ = _run(db_path, "C0003 的繳款狀況", CREDIT, llm)

    output = result["output"]
    assert "未償餘額 NT$" in output and "逾期次數：3" in output and "目前逾期中：是" in output
    assert "收入減少" in output and "「授信」權限" in output
    assert _raw(db_path, "C0003")[0] not in output


def test_the_llm_never_receives_an_unmasked_id_or_phone(db_path):

    llm = FakeChatLLM([_extraction("陳美玲"), "摘要"])
    _run(db_path, "陳美玲的往來狀況", CREDIT, llm)

    national_id, phone = _raw(db_path, "C0002")
    prompts = json.dumps([call["messages"] for call in llm.calls], ensure_ascii=False)
    assert national_id not in prompts and phone not in prompts
    assert llm.calls[0]["json_schema"] == CREDIT_LOOKUP_REQUEST_SCHEMA


def test_pronoun_resolution_gets_the_earlier_conversation(db_path):

    history = [HumanMessage(content="王大明有沒有逾期紀錄？"), AIMessage(content="王大明近12個月無逾期紀錄。")]
    llm = FakeChatLLM([_extraction("王大明", focus="contracts"), "摘要"])
    _run(db_path, "那他的合約呢？", SALES, llm, history)

    extraction_prompt = [m["content"] for m in llm.calls[0]["messages"]]
    assert "王大明有沒有逾期紀錄？" in extraction_prompt and extraction_prompt[-1] == "那他的合約呢？"


def test_summary_failure_still_returns_the_exact_facts(db_path):

    class SummaryFails(FakeChatLLM):
        def chat(self, messages, sampling, json_schema=None):
            if json_schema is None:
                raise RuntimeError("LLM down")
            return super().chat(messages, sampling, json_schema)

    result, audit = _run(db_path, "王大明的合約", SALES, SummaryFails(_extraction("王大明")))

    assert result["output"].startswith("【客戶資料】王大明（C0001）")
    assert audit[0]["outcome"] == "ok"
