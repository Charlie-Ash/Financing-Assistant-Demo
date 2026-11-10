# finance/loan_calc.py and finance/numbers.py against hand-checkable numbers.
# This is the arithmetic every quote and payoff in the system comes from, so
# it's tested without any agent, LLM or database.
from datetime import date

import pytest

from finance.loan_calc import (
    BALLOON,
    EQUAL_PRINCIPAL,
    annual_percentage_rate,
    level_payment,
    months_between,
    payoff_quote,
    prepayment_fee_rate,
    quote,
    remaining_balance,
)
from finance.numbers import is_mentioned, mentioned_numbers
from finance.plans import load_plans, product_terms


def test_equal_installment_matches_the_standard_formula():

    # 750,000 at 4.5% over 60 months: the payment the demo DB used to store.
    assert round(level_payment(750_000, 0.045, 60)) == 13_982
    q = quote(750_000, 0.045, 60)
    assert round(q.first_payment) == round(q.last_payment) == 13_982
    assert q.schedule[-1].balance == pytest.approx(0, abs=1e-6)
    assert q.total_paid == pytest.approx(q.principal + q.total_interest)


def test_zero_rate_splits_the_principal_evenly():

    q = quote(600_000, 0.0, 24)

    assert q.first_payment == pytest.approx(25_000)
    assert q.total_interest == pytest.approx(0)


def test_equal_principal_payments_decrease_and_add_up():

    q = quote(600_000, 0.0388, 36, EQUAL_PRINCIPAL)

    assert q.first_payment > q.last_payment
    assert all(row.principal == pytest.approx(600_000 / 36) for row in q.schedule)
    assert q.total_paid == pytest.approx(600_000 + q.total_interest)


def test_balloon_pays_the_residual_with_the_last_instalment():

    # RAV4 1,200,000, 20% down -> 960,000; 30% residual of the price = 360,000.
    q = quote(960_000, 0.035, 48, BALLOON, balloon=360_000)

    assert round(q.first_payment) == 14_464
    assert q.last_payment == pytest.approx(q.first_payment + 360_000, abs=1)
    # The final payment clears what's left (the residual plus that month's
    # regular principal) together with one month's interest on it.
    assert q.last_payment == pytest.approx(q.schedule[-2].balance * (1 + 0.035 / 12), abs=1e-6)
    assert q.schedule[-1].balance == pytest.approx(0, abs=1e-6)


def test_remaining_balance_is_amortised_not_straight_line():

    assert remaining_balance(750_000, 0.045, 60, 0) == 750_000
    assert remaining_balance(750_000, 0.045, 60, 60) == 0
    # Straight-line would say 500,000; interest-front-loaded amortisation leaves more.
    assert round(remaining_balance(750_000, 0.045, 60, 20)) == 518_464


def test_apr_equals_the_rate_without_fees_and_exceeds_it_with_fees():

    no_fees = quote(960_000, 0.035, 48)
    assert no_fees.apr == pytest.approx(0.035, abs=1e-6)

    zero_rate_with_fee = quote(600_000, 0.0, 24, fees=8_000)
    assert zero_rate_with_fee.apr > 0.01  # a "0%" plan still costs ~1.3% a year
    assert annual_percentage_rate(100, [50, 50]) == 0


def test_prepayment_fee_tiers_switch_at_12_and_24_months():

    assert prepayment_fee_rate(0) == prepayment_fee_rate(11) == 0.02
    assert prepayment_fee_rate(12) == prepayment_fee_rate(23) == 0.01
    assert prepayment_fee_rate(24) == prepayment_fee_rate(60) == 0.0
    assert months_between(date(2025, 3, 10), date(2026, 11, 10)) == 20
    assert months_between(date(2025, 3, 10), date(2026, 11, 9)) == 19


def test_payoff_adds_overdue_instalments_and_the_fee():

    p = payoff_quote(500_000, 40_000, months_since_start=18)

    assert p.fee == pytest.approx(5_000)
    assert p.total == pytest.approx(545_000)


def test_numbers_are_read_in_the_forms_people_write_them():

    assert {1_200_000, 20, 48} <= mentioned_numbers(["RAV4 120萬，頭期兩成，殘值型48期，月付多少？"])
    assert {1_200_000, 48} <= mentioned_numbers(["一百二十萬分四十八期"])
    assert {1_200_000, 60, 3.5} <= mentioned_numbers(["借1,200,000，5年，利率3.5%"])
    assert 3.5 in mentioned_numbers(["三點五趴"])
    assert 15_000 in mentioned_numbers(["一萬五千元"])
    # The Chinese comma separates clauses -- it must not glue numbers together.
    assert 12_000_005 not in mentioned_numbers(["借1,200,000，5年"])


def test_a_number_nobody_wrote_is_not_mentioned():

    mentioned = mentioned_numbers(["RAV4 月付多少？"])

    assert not is_mentioned(1_200_000, mentioned)
    assert not is_mentioned(60, mentioned)


def test_demo_plans_and_product_terms_load():

    plans = load_plans()

    assert plans["殘值型車貸"].allows_term(48) and not plans["殘值型車貸"].allows_term(50)
    assert plans["一般車貸"].allows_term(84) and not plans["一般車貸"].allows_term(90)
    assert product_terms("殘值型車貸")["residual_percent"] == 30
    assert product_terms("租賃")["annual_rate_percent"] == 4.2
