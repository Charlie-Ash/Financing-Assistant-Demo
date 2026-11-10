# The demo financing plans (data/products/financing_plans.json): default
# rate, method, allowed terms, fees and residual for each plan. Used by the
# Loan Calculator agent to fill in what a user didn't specify (always shown
# as an assumption) and by the Credit Lookup seed script to price contracts.
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from finance.loan_calc import BALLOON, EQUAL_INSTALLMENT

PLANS_PATH = Path(__file__).resolve().parents[2] / "data" / "products" / "financing_plans.json"

DEFAULT_PLAN = "一般車貸"


@dataclass(frozen=True)
class Plan:

    name: str
    method: str
    annual_rate_percent: float
    fees: float
    residual_percent: float
    terms: tuple[int, ...] = ()
    term_min: int = 0
    term_max: int = 0

    def allows_term(self, months: int) -> bool:
        if self.terms:
            return months in self.terms
        return self.term_min <= months <= self.term_max

    def describe_terms(self) -> str:
        if self.terms:
            return "／".join(str(t) for t in self.terms) + " 期"
        return f"{self.term_min}～{self.term_max} 期"


@lru_cache(maxsize=None)
def load_plans(path: Path = PLANS_PATH) -> dict[str, Plan]:

    raw = json.loads(path.read_text(encoding="utf-8"))["plans"]
    return {
        name: Plan(
            name=name,
            method=spec["method"],
            annual_rate_percent=spec["annual_rate_percent"],
            fees=spec["fees"],
            residual_percent=spec.get("residual_percent", 0),
            terms=tuple(spec.get("terms", ())),
            term_min=spec.get("term_min", 0),
            term_max=spec.get("term_max", 0),
        )
        for name, spec in raw.items()
    }


# How the demo database prices each contract product (see
# tools/creditLookupTool/seed_db.py). 租賃 has no plan in the catalogue, so it
# uses its own rate with equal instalments.
PRODUCT_TERMS = {
    "車貸": {"plan": "一般車貸"},
    "殘值型車貸": {"plan": "殘值型車貸"},
    "租賃": {"annual_rate_percent": 4.2, "method": EQUAL_INSTALLMENT, "residual_percent": 0, "fees": 3000},
}


def product_terms(product: str) -> dict:

    """(annual_rate_percent, method, residual_percent, fees) for a contract product."""

    spec = PRODUCT_TERMS[product]
    if "plan" in spec:
        plan = load_plans()[spec["plan"]]
        return {"annual_rate_percent": plan.annual_rate_percent, "method": plan.method,
                "residual_percent": plan.residual_percent, "fees": plan.fees}
    return dict(spec)


def product_method(product: str) -> str:

    return BALLOON if product == "殘值型車貸" else EQUAL_INSTALLMENT
