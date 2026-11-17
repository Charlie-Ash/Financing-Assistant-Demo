# Shared constants for the Loan Calculator tool. Read once at import by
# LoanCalcSettings (pydantic-settings): real env vars >
# src/tools/loanCalcTool/.env > defaults below, validated at startup. See
# .env.example next to this file for every setting.
#
# This agent loads no model and needs no GPU: the LLM (shared server or
# Gemini, LLM_PROVIDER) only reads the numbers out of a question, and all
# arithmetic is done by finance/loan_calc.py.
from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from llm.settings import LLMClientSettings

LOANCALCTOOL_DIR = Path(__file__).resolve().parent


class LoanCalcSettings(LLMClientSettings):

    model_config = SettingsConfigDict(env_file=str(LOANCALCTOOL_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    # Standalone A2A server bind address (see a2a/a2a_server.py). Port
    # follows Credit Lookup's 8002.
    loan_calc_a2a_host: str = "0.0.0.0"
    loan_calc_a2a_port: int = Field(default=8003, gt=0, le=65535)


settings = LoanCalcSettings()

LOAN_CALC_A2A_HOST = settings.loan_calc_a2a_host
LOAN_CALC_A2A_PORT = settings.loan_calc_a2a_port
