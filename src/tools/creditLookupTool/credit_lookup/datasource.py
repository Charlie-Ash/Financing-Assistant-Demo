# The Credit Lookup agent's only way into customer data: a small, fixed set
# of query functions behind a protocol. The LLM never writes SQL -- it only
# extracts WHO is being asked about (see graph.py), and these functions do
# the fetching with fixed, parameterised queries. Swapping the demo SQLite
# file for a real CRM / core loan system means writing another class with
# the same four methods; nothing else in the agent changes.
import sqlite3
from datetime import date
from pathlib import Path
from typing import Protocol


class CreditDataSource(Protocol):

    def as_of_date(self) -> date: ...

    def find_customers(self, name: str = "", customer_id: str = "") -> list[dict]: ...

    def get_contracts(self, customer_id: str) -> list[dict]: ...

    def get_payments(self, customer_id: str, since: date) -> list[dict]: ...


class SqliteCreditDataSource:

    """Read-only access to the synthetic database built by seed_db.py."""

    def __init__(self, db_path: Path):

        self.db_path = db_path

    # A short-lived, read-only connection per call: the A2A server may run
    # graph nodes on worker threads, and sqlite connections aren't shareable
    # across threads by default.
    def _query(self, sql: str, params: tuple = ()) -> list[dict]:

        conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(sql, params)]
        finally:
            conn.close()

    # "Today" for this dataset: the date it was generated (see seed_db.py),
    # so the 12-month window is stable however old the file is.
    def as_of_date(self) -> date:

        return date.fromisoformat(self._query("SELECT value FROM meta WHERE key = 'as_of_date'")[0]["value"])

    def find_customers(self, name: str = "", customer_id: str = "") -> list[dict]:

        if customer_id:
            return self._query("SELECT * FROM customers WHERE customer_id = ?", (customer_id.strip().upper(),))
        if name:
            return self._query("SELECT * FROM customers WHERE name = ? ORDER BY customer_id", (name.strip(),))
        return []

    def get_contracts(self, customer_id: str) -> list[dict]:

        # due_periods / overdue_amount: instalments that have fallen due so
        # far, and the total of those still unpaid -- what a payoff quote
        # needs on top of the scheduled balance.
        return self._query(
            """SELECT c.*,
                      (SELECT COUNT(*) FROM payments p
                       WHERE p.contract_id = c.contract_id AND p.paid_date IS NOT NULL) AS paid_periods,
                      (SELECT COUNT(*) FROM payments p WHERE p.contract_id = c.contract_id) AS due_periods,
                      (SELECT COALESCE(SUM(p.amount_due), 0) FROM payments p
                       WHERE p.contract_id = c.contract_id AND p.paid_date IS NULL) AS overdue_amount
               FROM contracts c WHERE c.customer_id = ? ORDER BY c.start_date""",
            (customer_id,),
        )

    def get_payments(self, customer_id: str, since: date) -> list[dict]:

        return self._query(
            """SELECT p.* FROM payments p JOIN contracts c USING (contract_id)
               WHERE c.customer_id = ? AND p.due_date >= ? ORDER BY p.due_date""",
            (customer_id, since.isoformat()),
        )
