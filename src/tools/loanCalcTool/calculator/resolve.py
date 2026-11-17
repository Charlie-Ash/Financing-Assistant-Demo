# Turns what the LLM extracted into calculator inputs -- or into a question
# back to the user. Plain Python, run before any arithmetic:
#
#   1. ground   -- every number must trace back to something the user wrote
#                  (finance/numbers.py); anything else is treated as not given
#   2. plan     -- chosen from the user's own words (殘值 / 0利率 / ...), never
#                  guessed by the model; default 一般車貸, stated as such
#   3. required -- a loan amount (or a car price) and a term; if missing, ask
#   4. defaults -- rate, residual, fees from the plan, each listed as 假設
#   5. checks   -- impossible or conflicting values stop the calculation
from dataclasses import dataclass, field

from finance.loan_calc import BALLOON, EQUAL_INSTALLMENT, EQUAL_PRINCIPAL
from finance.numbers import is_mentioned, mentioned_numbers
from finance.plans import DEFAULT_PLAN, Plan
from tools.loanCalcTool.calculator.request_schema import LoanCalcRequest

MIN_LOAN, MAX_LOAN = 10_000, 50_000_000
MAX_RATE_PERCENT = 20

# Checked in order; the first plan whose keyword the user wrote wins.
PLAN_KEYWORDS = [
    ("殘值型車貸", ("殘值",)),
    ("0利率專案", ("0利率", "零利率", "0%利率", "0 利率", "無息", "免息")),
    ("一般車貸", ("一般車貸", "一般方案")),
]
EQUAL_PRINCIPAL_KEYWORDS = ("本金平均", "本金攤還", "本金均攤")


@dataclass
class Resolution:

    # Set when the calculation can't run: the reply asking for / explaining
    # what's needed. Otherwise None and the inputs below are complete.
    message: str | None = None
    plan: Plan | None = None
    method: str = EQUAL_INSTALLMENT
    vehicle_price: float = 0
    down_payment: float = 0
    loan_amount: float = 0
    term_months: int = 0
    annual_rate_percent: float = 0
    balloon: float = 0
    fees: float = 0
    assumptions: list[str] = field(default_factory=list)


def _money(value: float) -> str:

    return f"NT${round(value):,}"


def _find_plan(texts_newest_first: list[str], plans: dict[str, Plan]) -> Plan | None:

    for text in texts_newest_first:
        compact = text.replace(" ", "")
        for name, keywords in PLAN_KEYWORDS:
            if name in plans and any(k.replace(" ", "") in compact for k in keywords):
                return plans[name]
    return None


def _ask(missing: list[str], known: list[str], plan: Plan) -> Resolution:

    lines = ["要進行試算，還需要以下資訊："]
    lines += [f"- {item}" for item in missing]
    if known:
        lines.append(f"目前已知：{'、'.join(known)}")
    lines.append(f"例如可以這樣問：「車價 120 萬，頭期兩成，{plan.name}分 {plan.terms[0] if plan.terms else 60} 期，月付多少？」")
    return Resolution(message="\n".join(lines), plan=plan)


def resolve_request(extracted: LoanCalcRequest, user_texts_newest_first: list[str], plans: dict[str, Plan]) -> Resolution:

    """user_texts_newest_first: the current question first, then earlier user
    messages -- the only text numbers may come from (never the assistant's
    own replies, which contain computed numbers)."""

    mentioned = mentioned_numbers(user_texts_newest_first)

    def grounded(value: float) -> float:
        return value if value > 0 and is_mentioned(value, mentioned) else 0

    price = grounded(extracted.vehicle_price)
    dp_amount = grounded(extracted.down_payment_amount)
    dp_percent = grounded(extracted.down_payment_percent)
    loan = grounded(extracted.loan_amount)
    term = int(grounded(extracted.term_months))
    rate = grounded(extracted.annual_rate_percent)
    residual_percent = grounded(extracted.residual_percent)

    named_plan = _find_plan(user_texts_newest_first, plans)
    plan = named_plan or plans[DEFAULT_PLAN]
    texts = "".join(user_texts_newest_first)
    if plan.method == BALLOON:
        method = BALLOON
    elif any(k in texts for k in EQUAL_PRINCIPAL_KEYWORDS):
        method = EQUAL_PRINCIPAL
    else:
        method = plan.method

    # --- required inputs: ask for whatever is missing, compute nothing ---
    known = []
    if price:
        known.append(f"車價 {_money(price)}")
    if dp_amount:
        known.append(f"頭期款 {_money(dp_amount)}")
    if dp_percent:
        known.append(f"頭期款 {dp_percent:g}%")
    if loan:
        known.append(f"貸款金額 {_money(loan)}")
    if term:
        known.append(f"{term} 期")
    if rate:
        known.append(f"年利率 {rate:g}%")

    missing = []
    if not loan and not price:
        missing.append("貸款金額，或車價（與頭期款）")
    if not term:
        missing.append(f"期數（「{plan.name}」可選 {plan.describe_terms()}）")
    if missing:
        return _ask(missing, known, plan)

    # --- consistency checks: stop instead of calculating with bad inputs ---
    def stop(message: str) -> Resolution:
        return Resolution(message=message, plan=plan)

    if dp_amount and dp_percent and price and abs(dp_amount - price * dp_percent / 100) > price * 0.01:
        return stop(f"頭期款金額 {_money(dp_amount)} 與比例 {dp_percent:g}% 不一致（車價 {_money(price)} 的 {dp_percent:g}% 為 "
                    f"{_money(price * dp_percent / 100)}），請確認要以哪一個為準。")

    assumptions = []
    if not named_plan:
        assumptions.append(f"未指定方案，依「{plan.name}」（{'本金平均攤還' if method == EQUAL_PRINCIPAL else '本息平均攤還'}）試算。")

    if loan:
        down_payment = price - loan if price else 0
        if price and (dp_amount or dp_percent):
            expected = price - (dp_amount or price * dp_percent / 100)
            if abs(expected - loan) > price * 0.01:
                return stop(f"貸款金額 {_money(loan)} 與車價扣除頭期款後的 {_money(expected)} 不一致，請確認。")
    else:
        if dp_amount:
            down_payment = dp_amount
        elif dp_percent:
            down_payment = price * dp_percent / 100
        else:
            down_payment = 0
            assumptions.append("未指定頭期款，以全額貸款試算。")
        loan = price - down_payment

    if down_payment < 0:
        return stop(f"貸款金額 {_money(loan)} 高於車價 {_money(price)}，請確認金額。")
    if price and down_payment >= price:
        return stop(f"頭期款 {_money(down_payment)} 不低於車價 {_money(price)}，無需貸款，請確認金額。")
    if not MIN_LOAN <= loan <= MAX_LOAN:
        return stop(f"貸款金額 {_money(loan)} 超出試算範圍（{_money(MIN_LOAN)}～{_money(MAX_LOAN)}），請確認金額。")
    if not plan.allows_term(term):
        return stop(f"「{plan.name}」可選期數為 {plan.describe_terms()}，無法以 {term} 期試算。")

    if rate:
        if rate > MAX_RATE_PERCENT:
            return stop(f"年利率 {rate:g}% 超出試算範圍（0～{MAX_RATE_PERCENT}%），請確認。")
    else:
        rate = plan.annual_rate_percent
        # Naming the 0% plan (「0利率」) *is* stating the rate -- not an assumption.
        if not (named_plan and rate == 0):
            assumptions.append(f"未指定利率，依「{plan.name}」方案以 {rate:g}% 試算。")

    balloon = 0
    if method == BALLOON:
        base, base_label = (price, "車價") if price else (loan, "貸款金額")
        if residual_percent:
            if residual_percent >= 100:
                return stop(f"殘值比例 {residual_percent:g}% 不合理，請確認。")
        else:
            residual_percent = plan.residual_percent
            assumptions.append(f"未指定殘值，依「{plan.name}」方案以{base_label} {residual_percent:g}% 試算。")
        if not price:
            assumptions.append("未提供車價，殘值以貸款金額計算。")
        balloon = base * residual_percent / 100
        if balloon >= loan:
            return stop(f"殘值 {_money(balloon)} 不低於貸款金額 {_money(loan)}，請確認殘值比例。")

    assumptions.append(f"開辦費依「{plan.name}」方案以 {_money(plan.fees)} 計入總費用年百分率。")

    return Resolution(
        plan=plan, method=method, vehicle_price=price, down_payment=down_payment, loan_amount=loan,
        term_months=term, annual_rate_percent=rate, balloon=balloon, fees=plan.fees, assumptions=assumptions,
    )
