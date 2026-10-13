# Shared constants for the Actions tool. Read once at import by
# ActionsSettings (pydantic-settings): real env vars >
# src/tools/actionsTool/.env > defaults below, validated at startup. See
# .env.example next to this file for every setting.
#
# No model/VRAM settings live here any more: this agent generates text
# through the shared LLM server (src/llm/), so which model it uses is
# decided there, and this process needs no GPU at all.
from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from llm.settings import LLMClientSettings

# Resolved relative to this file rather than the process's CWD (same pattern
# as ragTool/config.py).
ACTIONSTOOL_DIR = Path(__file__).resolve().parent
AGENTSYSTEM_ROOT = ACTIONSTOOL_DIR.parent.parent.parent

# Notes are written into the project's shared top-level data/ folder, in
# their own subfolder.
NOTES_DIR = AGENTSYSTEM_ROOT / "data" / "notes"


class ActionsSettings(LLMClientSettings):

    model_config = SettingsConfigDict(env_file=str(ACTIONSTOOL_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    # Standalone A2A server bind address (see a2a/a2a_server.py). Port
    # follows RAG's 8001.
    actions_a2a_host: str = "0.0.0.0"
    actions_a2a_port: int = Field(default=8002, gt=0, le=65535)


settings = ActionsSettings()

ACTIONS_A2A_HOST = settings.actions_a2a_host
ACTIONS_A2A_PORT = settings.actions_a2a_port
