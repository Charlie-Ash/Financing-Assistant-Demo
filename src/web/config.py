# Settings for the web interface server (python -m web.server). Read once at
# import by WebSettings (pydantic-settings): real env vars > src/web/.env >
# defaults below, validated at startup. See .env.example next to this file.
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
AGENTSYSTEM_ROOT = WEB_DIR.parent.parent

# Where the orchestrator saves end-of-chat summaries (memory_manager.finalize_and_save)
# -- the only folder the download endpoint will serve files from.
SUMMARY_DIR = AGENTSYSTEM_ROOT / "data"


class WebSettings(BaseSettings):

    model_config = SettingsConfigDict(env_file=str(WEB_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    # 127.0.0.1 on purpose: the system has no login yet, so only this machine
    # (or an SSH tunnel / VS Code port forward into it) can open the page.
    # Binding to 0.0.0.0 would expose customer lookups to the whole network.
    web_host: str = "127.0.0.1"
    web_port: int = Field(default=8080, gt=0, le=65535)


settings = WebSettings()

WEB_HOST = settings.web_host
WEB_PORT = settings.web_port
