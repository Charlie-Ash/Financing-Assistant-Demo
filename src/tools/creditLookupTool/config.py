# Shared constants for the Credit Lookup tool. Read once at import by
# CreditLookupSettings (pydantic-settings): real env vars >
# src/tools/creditLookupTool/.env > defaults below, validated at startup. See
# .env.example next to this file for every setting.
#
# No model/VRAM settings live here: this agent generates text through the
# shared LLM server or the Gemini API (LLM_PROVIDER, inherited from
# llm/settings.LLMClientSettings), and this process needs no GPU at all.
from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from llm.settings import LLMClientSettings

# Resolved relative to this file rather than the process's CWD (same pattern
# as ragTool/config.py).
CREDITLOOKUPTOOL_DIR = Path(__file__).resolve().parent
AGENTSYSTEM_ROOT = CREDITLOOKUPTOOL_DIR.parent.parent.parent


class CreditLookupSettings(LLMClientSettings):

    model_config = SettingsConfigDict(env_file=str(CREDITLOOKUPTOOL_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    # The synthetic customer database (built by `python -m
    # tools.creditLookupTool.seed_db`, or automatically on first start) and
    # the append-only audit log of every lookup. Relative paths are resolved
    # against the project root.
    credit_lookup_db_path: Path = Path("data") / "credit" / "credit_demo.db"
    credit_lookup_audit_log_path: Path = Path("data") / "credit" / "audit_log.jsonl"

    # How far back payment history is summarised.
    credit_lookup_payment_months: int = Field(default=12, gt=0)

    # Standalone A2A server bind address (see a2a/a2a_server.py). Port
    # follows RAG's 8001.
    credit_lookup_a2a_host: str = "0.0.0.0"
    credit_lookup_a2a_port: int = Field(default=8002, gt=0, le=65535)


def _from_root(path: Path) -> Path:

    return path if path.is_absolute() else AGENTSYSTEM_ROOT / path


settings = CreditLookupSettings()

DB_PATH = _from_root(settings.credit_lookup_db_path)
AUDIT_LOG_PATH = _from_root(settings.credit_lookup_audit_log_path)
PAYMENT_MONTHS = settings.credit_lookup_payment_months

CREDIT_LOOKUP_A2A_HOST = settings.credit_lookup_a2a_host
CREDIT_LOOKUP_A2A_PORT = settings.credit_lookup_a2a_port
