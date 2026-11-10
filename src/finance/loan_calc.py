# Loan arithmetic for the whole project: plain Python, no LLM, no I/O. Used
# by the Loan Calculator agent for "what-if" quotes, by the Credit Lookup
# agent for payoff quotes on existing contracts, and by the Credit Lookup
# seed script so the demo database's payments and balances agree with what
# the calculator says.
#
# Conventions: annual_rate is a fraction (0.035 = 3.5%), the monthly rate is
# annual_rate / 12, amounts are NT$ kept as floats until rendering.
from dataclasses import dataclass
from datetime import date

EQUAL_INSTALLMENT = "equal_installment"   # 本息平均攤還
EQUAL_PRINCIPAL = "equal_principal"       # 本金平均攤還
BALLOON = "balloon"                       # 殘值型：月付本息平均攤還，期末另付殘值
METHODS = (EQUAL_INSTALLMENT, EQUAL_PRINCIPAL, BALLOON)

METHOD_LABELS = {EQUAL_INSTALLMENT: "本息平均攤還", EQUAL_PRINCIPAL: "本金平均攤還", BALLOON: "殘值型（期末付殘值）"}

# Early-repayment fee tiers from the demo policy 《提前清償作業規範》第2條 (v2,
# data/regulations/policies/提前清償作業規範_v2.md): within 12 months of the
# contract 2%, from 12 to under 24 months 1%, from 24 months none. Kept in
# sync with that file by hand -- the citation travels with every quote.
PREPAYMENT_FEE_TIERS = ((12, 0.02), (24, 0.01))
PREPAYMENT_FEE_SOURCE = "《【示範用虛構文件】提前清償作業規範》第2條（第2版）"


@dataclass(frozen=True)
class ScheduleRow:

    period: int
    payment: float
    interest: float
    principal: float
    balance: float


@dataclass(frozen=True)
class LoanQuote:

    principal: float
    annual_rate: float
    months: int
    method: str
    balloon: float
    fees: float
    schedule: tuple[ScheduleRow, ...]

    @property
    def first_payment(self) -> float:
        """The usual monthly amount for level-payment plans (a balloon plan's
        first payment excludes the balloon); the largest one for equal principal."""
        return self.schedule[0].payment

    @property
    def last_payment(self) -> float:
        return self.schedule[-1].payment

    @property
    def total_paid(self) -> float:
        return sum(row.payment for row in self.schedule)

    @property
    def total_interest(self) -> float:
        return sum(row.interest for row in self.schedule)

    @property
    def apr(self) -> float:
        """總費用年百分率 (annual, fraction), counting upfront fees."""
        return annual_percentage_rate(self.principal - self.fees, [row.payment for row in self.schedule])


def level_payment(principal: float, annual_rate: float, months: int, balloon: float = 0.0) -> float:

    """Equal monthly payment that repays `principal` over `months`, leaving
    `balloon` to be paid on top of the last payment.
    M = (P - B/(1+r)^n) * r / (1 - (1+r)^-n); with r = 0, M = (P - B) / n."""

    if months <= 0:
        raise ValueError("months must be positive")
    r = annual_rate / 12
    if r == 0:
        return (principal - balloon) / months
    return (principal - balloon / (1 + r) ** months) * r / (1 - (1 + r) ** -months)


def schedule(principal: float, annual_rate: float, months: int, method: str = EQUAL_INSTALLMENT, balloon: float = 0.0) -> list[ScheduleRow]:

    """Month-by-month repayment table. For BALLOON the last row includes the
    balloon; every method ends at a balance of 0."""

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method != BALLOON:
        balloon = 0.0

    r = annual_rate / 12
    payment = level_payment(principal, annual_rate, months, balloon) if method != EQUAL_PRINCIPAL else 0.0
    balance = principal
    rows = []

    for period in range(1, months + 1):
        interest = balance * r
        if method == EQUAL_PRINCIPAL:
            principal_part = principal / months
        else:
            principal_part = payment - interest
        if period == months:
            principal_part = balance  # absorbs rounding drift and, for BALLOON, the balloon itself
        amount = principal_part + interest
        balance -= principal_part
        rows.append(ScheduleRow(period, amount, interest, principal_part, max(balance, 0.0)))

    return rows


def quote(principal: float, annual_rate: float, months: int, method: str = EQUAL_INSTALLMENT, balloon: float = 0.0, fees: float = 0.0) -> LoanQuote:

    if principal <= 0:
        raise ValueError("principal must be positive")
    rows = schedule(principal, annual_rate, months, method, balloon)
    return LoanQuote(principal, annual_rate, months, method, balloon if method == BALLOON else 0.0, fees, tuple(rows))


def remaining_balance(principal: float, annual_rate: float, months: int, periods_paid: int, method: str = EQUAL_INSTALLMENT, balloon: float = 0.0) -> float:

    """Principal still owed after `periods_paid` scheduled payments."""

    if periods_paid <= 0:
        return principal
    if periods_paid >= months:
        return 0.0
    return schedule(principal, annual_rate, months, method, balloon)[periods_paid - 1].balance


def annual_percentage_rate(net_amount: float, payments: list[float]) -> float:

    """總費用年百分率, simplified: the monthly rate i at which the payments'
    present value equals the money actually received (loan minus upfront
    fees), times 12. Found by bisection; 0 when payments don't exceed it."""

    if net_amount <= 0 or sum(payments) <= net_amount:
        return 0.0

    def present_value(i: float) -> float:
        return sum(p / (1 + i) ** k for k, p in enumerate(payments, start=1))

    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2
        if present_value(mid) > net_amount:
            low = mid
        else:
            high = mid
    return (low + high) / 2 * 12


def months_between(start: date, end: date) -> int:

    """Whole months from start to end (e.g. 2025-03-10 -> 2026-11-10 is 20)."""

    months = (end.year - start.year) * 12 + end.month - start.month
    if end.day < start.day:
        months -= 1
    return max(months, 0)


def prepayment_fee_rate(months_since_start: int) -> float:

    for limit, rate in PREPAYMENT_FEE_TIERS:
        if months_since_start < limit:
            return rate
    return 0.0


@dataclass(frozen=True)
class PayoffQuote:

    remaining_principal: float
    overdue_amount: float
    months_since_start: int
    fee_rate: float

    @property
    def fee(self) -> float:
        return self.remaining_principal * self.fee_rate

    @property
    def total(self) -> float:
        return self.remaining_principal + self.overdue_amount + self.fee


def payoff_quote(remaining_principal: float, overdue_amount: float, months_since_start: int) -> PayoffQuote:

    """Full early repayment today: principal not yet due, plus instalments
    already due but unpaid, plus the policy fee on that principal. Late
    interest on overdue instalments isn't modelled."""

    return PayoffQuote(remaining_principal, overdue_amount, months_since_start, prepayment_fee_rate(months_since_start))
