# The Credit Lookup agent's access rules, as pure functions: masking, what
# each role may see, and the deterministic late-payment facts. Nothing here
# touches a database or an LLM.
from datetime import date

from tools.creditLookupTool.credit_lookup.policy import (
    build_view,
    compute_late_facts,
    is_valid_caller,
    mask_national_id,
    mask_phone,
    render_facts,
)
from tools.creditLookupTool.seed_db import is_valid_national_id

AS_OF = date(2026, 10, 21)

CUSTOMER = {
    "customer_id": "C0003", "name": "林志強", "national_id": "E164839275", "phone": "0933221100",
    "birth_date": "1978-11-02", "customer_since": "2019-09-15", "branch": "高雄",
    "risk_note": "近兩期未繳，客戶表示收入減少。",
}
CONTRACTS = [{
    "contract_id": "K00003", "customer_id": "C0003", "product": "車貸", "vehicle": "Hiace",
    "principal": 1_100_000, "monthly_payment": 20_508, "term_months": 60, "start_date": "2024-09-06",
    "status": "逾期中", "outstanding_balance": 696_667, "paid_periods": 22,
}]
PAYMENTS = [
    {"contract_id": "K00003", "period": 18, "due_date": "2026-04-06", "paid_date": "2026-04-12", "days_late": 6},
    {"contract_id": "K00003", "period": 22, "due_date": "2026-08-06", "paid_date": "2026-08-06", "days_late": 0},
    {"contract_id": "K00003", "period": 23, "due_date": "2026-09-06", "paid_date": None, "days_late": 45},
]


def test_national_id_and_phone_are_masked():

    assert mask_national_id("E164839275") == "E******275"
    assert mask_phone("0933221100") == "0933-***-100"


def test_caller_needs_a_known_role_and_a_user_id():

    assert is_valid_caller({"user_id": "lin02", "role": "credit"})
    assert not is_valid_caller(None)
    assert not is_valid_caller({"user_id": "lin02", "role": "admin"})
    assert not is_valid_caller({"user_id": " ", "role": "sales"})


def test_late_facts_are_computed_from_payment_rows():

    facts = compute_late_facts(PAYMENTS, AS_OF)

    assert facts["late_count"] == 2
    assert facts["max_days_late"] == 45
    assert facts["currently_overdue"] is True
    assert facts["overdue_days"] == (AS_OF - date(2026, 9, 6)).days


def test_sales_view_hides_amounts_payment_detail_and_risk_note():

    view = build_view(CUSTOMER, CONTRACTS, PAYMENTS, AS_OF, "sales", 12)
    facts = render_facts(view)

    assert view["late"] == {"has_late_record": True}
    assert "outstanding_balance" not in view["contracts"][0]
    assert "risk_note" not in view["customer"]
    assert "近12個月有逾期紀錄：有" in facts
    for hidden in ("696,667", "20,508", "收入減少", "45 天", "E164839275", "0933221100"):
        assert hidden not in facts


def test_credit_view_shows_detail_but_still_masks_identifiers():

    facts = render_facts(build_view(CUSTOMER, CONTRACTS, PAYMENTS, AS_OF, "credit", 12))

    for shown in ("未償餘額 NT$696,667", "月付 NT$20,508", "逾期次數：2", "最長逾期：45 天", "收入減少", "E******275"):
        assert shown in facts
    assert "E164839275" not in facts and "0933221100" not in facts


def test_seeded_ids_never_pass_the_official_checksum():

    # A known-valid example ID, to prove the checker itself works.
    assert is_valid_national_id("A123456789")
