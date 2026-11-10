# Builds the Credit Lookup agent's SYNTHETIC customer database: SQLite, from
# a fixed random seed, so every machine gets the same customers. Nothing here
# is real -- names are drawn from small built-in lists, and national IDs are
# random digits in the A123456789 shape that deliberately don't pass the
# official checksum, so they can't collide with a real person's ID.
#
#   cd src && python -m tools.creditLookupTool.seed_db            # (re)build
#
# The agent also runs build_database() itself on first start if the file is
# missing (see credit_lookup/tool_class.py). "Today" for every relative date
# (payment due dates, the 12-month window) is the as_of date stored in the
# meta table, so the data stays self-consistent however long ago it was built.
import random
import sqlite3
import string
from datetime import date, timedelta
from pathlib import Path

from finance.loan_calc import remaining_balance, schedule
from finance.plans import product_terms
from tools.creditLookupTool.config import DB_PATH

# Bump when the tables change: CreditLookupTool rebuilds an older database
# automatically (it's synthetic, so nothing is lost; the audit log is a
# separate file). v2 added each contract's rate, residual (balloon) and fees,
# and computes payments/balances with finance/loan_calc.py.
SCHEMA_VERSION = 2

SEED = 20261021
RANDOM_CUSTOMERS = 25

SURNAMES = ["陳", "林", "黃", "張", "李", "王", "吳", "劉", "蔡", "楊", "許", "鄭", "謝", "郭", "洪"]
GIVEN_NAMES = ["家豪", "雅婷", "志偉", "淑芬", "俊傑", "怡君", "建宏", "佩珊", "冠宇", "欣怡", "宗翰", "雅雯", "承恩", "詩涵", "柏翰"]
BRANCHES = ["台北", "新北", "桃園", "台中", "台南", "高雄"]
# (model, typical financed amount in NT$)
VEHICLES = [("RAV4", 1_000_000), ("Corolla Cross", 750_000), ("Altis", 650_000), ("Yaris", 500_000), ("Camry", 1_200_000), ("Hiace", 1_100_000)]
PRODUCTS = ["車貸", "殘值型車貸", "租賃"]
RISK_NOTES = ["", "", "", "", "曾申請展延一次，已依約恢復正常繳款。", "收入來源為自營，建議覆核財力證明。"]

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY, name TEXT NOT NULL, national_id TEXT NOT NULL,
    phone TEXT NOT NULL, birth_date TEXT NOT NULL, customer_since TEXT NOT NULL,
    branch TEXT NOT NULL, risk_note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE contracts (
    contract_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    product TEXT NOT NULL, vehicle TEXT NOT NULL, principal INTEGER NOT NULL,
    monthly_payment INTEGER NOT NULL, term_months INTEGER NOT NULL, start_date TEXT NOT NULL,
    status TEXT NOT NULL, outstanding_balance INTEGER NOT NULL,
    annual_rate REAL NOT NULL, residual_value INTEGER NOT NULL, fees INTEGER NOT NULL
);
CREATE TABLE payments (
    contract_id TEXT NOT NULL REFERENCES contracts(contract_id), period INTEGER NOT NULL,
    due_date TEXT NOT NULL, paid_date TEXT, amount_due INTEGER NOT NULL,
    amount_paid INTEGER NOT NULL, days_late INTEGER NOT NULL,
    PRIMARY KEY (contract_id, period)
);
"""


def add_months(day: date, months: int) -> date:

    month_index = day.month - 1 + months
    return date(day.year + month_index // 12, month_index % 12 + 1, min(day.day, 28))


# Letter codes of the Taiwanese national ID checksum.
_ID_LETTER_CODES = dict(zip("ABCDEFGHJKLMNPQRSTUVXYWZIO", range(10, 36)))


def is_valid_national_id(national_id: str) -> bool:

    code = _ID_LETTER_CODES[national_id[0]]
    digits = [int(d) for d in national_id[1:]]
    total = code // 10 + (code % 10) * 9 + sum(d * w for d, w in zip(digits, [8, 7, 6, 5, 4, 3, 2, 1, 1]))
    return total % 10 == 0


def _never_valid(national_id: str) -> str:

    """Nudge the last digit if needed so the ID fails the official checksum --
    a fake ID that can't belong to a real person."""

    if is_valid_national_id(national_id):
        national_id = national_id[:-1] + str((int(national_id[-1]) + 1) % 10)
    return national_id


def _fake_national_id(rng: random.Random) -> str:

    return _never_valid(rng.choice(string.ascii_uppercase) + rng.choice("12") + "".join(rng.choice(string.digits) for _ in range(8)))


def _fake_phone(rng: random.Random) -> str:

    return "09" + "".join(rng.choice(string.digits) for _ in range(8))


def _contract_rows(contract_id, customer_id, product, vehicle, principal, term, start, as_of, late_days_by_period, unpaid_periods):

    """One contract plus its payment rows up to as_of. late_days_by_period /
    unpaid_periods script the repayment behaviour (period -> days late).
    Rate, method, residual and fees come from the product's demo plan, and
    every amount from finance/loan_calc.py -- the same arithmetic the Loan
    Calculator agent quotes with."""

    terms = product_terms(product)
    annual_rate = terms["annual_rate_percent"] / 100
    balloon = round(principal * terms["residual_percent"] / 100)
    rows = schedule(principal, annual_rate, term, terms["method"], balloon)
    monthly = round(rows[0].payment)
    payments, paid_periods = [], 0

    for row in rows:
        period = row.period
        due = add_months(start, period)
        if due > as_of:
            break
        amount = round(row.payment)  # the last row of a balloon contract includes the balloon
        if period in unpaid_periods:
            payments.append((contract_id, period, due.isoformat(), None, amount, 0, (as_of - due).days))
            continue
        days_late = late_days_by_period.get(period, 0)
        paid = due + timedelta(days=days_late)
        payments.append((contract_id, period, due.isoformat(), paid.isoformat(), amount, amount, days_late))
        paid_periods += 1

    finished = paid_periods == term
    if unpaid_periods:
        status = "逾期中"
    elif finished:
        status = "已結清"
    else:
        status = "繳款中"

    # Principal still owed: the amortised balance after the instalments
    # actually paid (unpaid overdue instalments still count as owed).
    outstanding = 0 if finished else round(remaining_balance(principal, annual_rate, term, paid_periods, terms["method"], balloon))
    contract = (contract_id, customer_id, product, vehicle, principal, monthly, term, start.isoformat(), status, outstanding,
                annual_rate, balloon, terms["fees"])
    return contract, payments


def is_current(db_path: Path) -> bool:

    """True if db_path exists and was built with this SCHEMA_VERSION."""

    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return row is not None and row[0] == str(SCHEMA_VERSION)


def _demo_customers(as_of: date):

    """Hand-written customers the README and evals refer to by name."""

    # Starts two years and ~15 days ago, so its two unpaid instalments are
    # ~45 and ~15 days overdue.
    overdue_start = add_months(as_of, -24) - timedelta(days=15)
    return [
        # (customer, [(contract args...)])
        (("C0001", "王大明", "F129384756", "0912345678", "1985-03-12", "2021-05-10", "台北", ""),
         [("K00001", "車貸", "Corolla Cross", 750_000, 60, add_months(as_of, -20), {}, set())]),
        (("C0002", "陳美玲", "B287465123", "0928765432", "1990-07-21", "2020-02-03", "台中", ""),
         [("K00002", "殘值型車貸", "RAV4", 1_000_000, 48, add_months(as_of, -30), {25: 8, 29: 15}, set())]),
        (("C0003", "林志強", "E164839275", "0933221100", "1978-11-02", "2019-09-15", "高雄",
          "近兩期未繳，已電話聯繫兩次未接；客戶表示收入減少。"),
         [("K00003", "車貸", "Hiace", 1_100_000, 60, overdue_start, {18: 6}, {23, 24})]),
        (("C0004", "王志明", "A174628391", "0955123456", "1982-01-30", "2022-08-01", "台北", ""),
         [("K00004", "車貸", "Yaris", 500_000, 36, add_months(as_of, -12), {}, set())]),
        (("C0005", "王志明", "L192837465", "0966789012", "1995-12-09", "2023-03-20", "台中", ""),
         [("K00005", "租賃", "Altis", 650_000, 36, add_months(as_of, -8), {}, set())]),
    ]


def build_database(db_path: Path = DB_PATH, as_of: date | None = None) -> dict:

    """(Re)build the database at db_path; returns row counts."""

    as_of = as_of or date.today()
    rng = random.Random(SEED)
    customers, contracts, payments = [], [], []

    for customer, contract_specs in _demo_customers(as_of):
        customer = (*customer[:2], _never_valid(customer[2]), *customer[3:])
        customers.append(customer)
        for contract_id, product, vehicle, principal, term, start, late, unpaid in contract_specs:
            contract, rows = _contract_rows(contract_id, customer[0], product, vehicle, principal, term, start, as_of, late, unpaid)
            contracts.append(contract)
            payments.extend(rows)

    contract_number = len(contracts)
    for i in range(RANDOM_CUSTOMERS):
        customer_id = f"C{len(customers) + 1:04d}"
        birth = date(rng.randint(1965, 2000), rng.randint(1, 12), rng.randint(1, 28))
        since = add_months(as_of, -rng.randint(12, 72))
        customers.append((
            customer_id, rng.choice(SURNAMES) + rng.choice(GIVEN_NAMES), _fake_national_id(rng), _fake_phone(rng),
            birth.isoformat(), since.isoformat(), rng.choice(BRANCHES), rng.choice(RISK_NOTES),
        ))

        # Most customers pay on time; roughly a quarter have a few late
        # payments, and a few are currently overdue.
        behaviour = rng.random()
        for _ in range(rng.choice([1, 1, 1, 2])):
            contract_number += 1
            vehicle, price = rng.choice(VEHICLES)
            term = rng.choice([36, 48, 60])
            start = add_months(as_of, -rng.randint(3, 70))
            elapsed = max(1, (as_of.year - start.year) * 12 + as_of.month - start.month)
            late = {}
            if behaviour > 0.75:
                late = {rng.randint(1, min(elapsed, term)): rng.randint(3, 30) for _ in range(rng.randint(1, 3))}
            unpaid = {min(elapsed, term)} if behaviour > 0.93 and elapsed < term else set()
            contract, rows = _contract_rows(
                f"K{contract_number:05d}", customer_id, rng.choice(PRODUCTS), vehicle,
                round(price * rng.uniform(0.7, 1.1), -3), term, start, as_of, late, unpaid,
            )
            contracts.append(contract)
            payments.extend(rows)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO meta VALUES ('as_of_date', ?)", (as_of.isoformat(),))
        conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?)", customers)
        conn.executemany("INSERT INTO contracts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", contracts)
        conn.executemany("INSERT INTO payments VALUES (?, ?, ?, ?, ?, ?, ?)", payments)
    conn.close()

    return {"as_of": as_of.isoformat(), "customers": len(customers), "contracts": len(contracts), "payments": len(payments)}


def main():

    counts = build_database()
    print(f"Synthetic credit database built at {DB_PATH}")
    print(f"  as of {counts['as_of']}: {counts['customers']} customers, {counts['contracts']} contracts, {counts['payments']} payment rows")


if __name__ == "__main__":
    main()
