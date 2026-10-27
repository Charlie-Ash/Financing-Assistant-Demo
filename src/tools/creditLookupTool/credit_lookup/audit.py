# Append-only audit trail: one JSON line per lookup request, recording who
# asked (user id + role), what they asked, the outcome, and which customers
# were returned -- as required by the demo policy 《個人資料處理作業規範》
# 第6條. Only masked/identifying-by-id data is written; never raw national
# IDs or phone numbers.
import json
from datetime import datetime
from pathlib import Path


def append_audit(path: Path, caller: dict | None, question: str, outcome: str, customer_ids: list[str]) -> None:

    caller = caller or {}
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "user_id": caller.get("user_id", ""),
        "role": caller.get("role", ""),
        "question": question,
        "outcome": outcome,
        "customer_ids": customer_ids,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
