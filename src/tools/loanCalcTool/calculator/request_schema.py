# Structured-output schema for the Loan Calculator agent's one LLM call (see
# graph.py's extract node): the numbers the user stated, nothing else. Every
# field is required and 0 means "not mentioned" -- the plan and repayment
# method are decided in code from the user's own words (resolve.py), and any
# number the model fills in that the user never wrote is discarded there.
from pydantic import BaseModel


class LoanCalcRequest(BaseModel):

    vehicle_price: float          # 車價（元）
    down_payment_amount: float    # 頭期款金額（元）
    down_payment_percent: float   # 頭期款比例（%，兩成 = 20）
    loan_amount: float            # 貸款金額（元）
    term_months: int              # 期數（月）
    annual_rate_percent: float    # 年利率（%）
    residual_percent: float       # 殘值比例（%）


LOAN_CALC_REQUEST_SCHEMA = LoanCalcRequest.model_json_schema()
