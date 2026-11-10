# The Loan Calculator's reply, built entirely by code from the resolved
# inputs and finance/loan_calc.py's quote -- no LLM touches any number.
from finance.loan_calc import BALLOON, EQUAL_PRINCIPAL, METHOD_LABELS, LoanQuote
from tools.loanCalcTool.calculator.resolve import Resolution

DISCLAIMER = "試算結果僅供參考，實際條件以核貸結果為準；方案為示範用虛構資料。"

# How many schedule rows to show from the start (the last row is always shown).
SCHEDULE_HEAD = 3


def _money(value: float) -> str:

    return f"NT${round(value):,}"


def render_quote(r: Resolution, quote: LoanQuote) -> str:

    lines = [f"【貸款試算】{r.plan.name}｜{METHOD_LABELS[r.method]}"]

    if r.vehicle_price:
        dp_percent = r.down_payment / r.vehicle_price * 100
        lines.append(f"- 車價 {_money(r.vehicle_price)}，頭期款 {_money(r.down_payment)}（{dp_percent:.0f}%）")
    lines.append(f"- 貸款金額 {_money(r.loan_amount)}，期數 {r.term_months} 期，年利率 {r.annual_rate_percent:g}%")

    last = quote.schedule[-1]
    if r.method == BALLOON:
        lines.append(f"- 殘值 {_money(r.balloon)}，於第 {r.term_months} 期與當期款項一併支付")
        lines.append(f"- 每月應付 **{_money(quote.first_payment)}**；第 {r.term_months} 期應付 {_money(last.payment)}（含殘值）")
    elif r.method == EQUAL_PRINCIPAL:
        lines.append(f"- 首期應付 **{_money(quote.first_payment)}**，之後逐月遞減，末期應付 {_money(last.payment)}")
    else:
        lines.append(f"- 每月應付 **{_money(quote.first_payment)}**")

    lines.append(f"- 總利息 {_money(quote.total_interest)}，總還款金額 {_money(quote.total_paid)}（不含開辦費）")
    lines.append(f"- 開辦費 {_money(r.fees)}，總費用年百分率 約 {quote.apr * 100:.2f}%（含開辦費，簡化估算）")

    lines.append("【還款明細（節錄）】")
    rows = list(quote.schedule[:SCHEDULE_HEAD])
    for row in rows:
        lines.append(_schedule_line(row))
    if quote.months > SCHEDULE_HEAD:
        if quote.months > SCHEDULE_HEAD + 1:
            lines.append("- …")
        lines.append(_schedule_line(last))

    if r.assumptions:
        lines.append("【試算假設】")
        lines += [f"- {a}" for a in r.assumptions]

    lines.append(DISCLAIMER)
    return "\n".join(lines)


def _schedule_line(row) -> str:

    return (f"- 第 {row.period} 期：應付 {_money(row.payment)}（利息 {_money(row.interest)}／本金 {_money(row.principal)}），"
            f"剩餘本金 {_money(row.balance)}")
