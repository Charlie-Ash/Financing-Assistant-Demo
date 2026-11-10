# Orchestrator-side config: which remote A2A agents to discover at startup,
# how each one's results should be relayed, and how to reach the LLM (the
# shared local server or the Gemini API, see llm/settings.py). Read once at import by OrchestratorSettings (pydantic-settings):
# real env vars > src/orchestrator/.env > defaults below, validated before
# anything else runs -- a bad value fails at startup naming the field. See
# .env.example next to this file for every setting.
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from llm.settings import LLMClientSettings


@dataclass
class RemoteAgentConfig:
    """One entry from REMOTE_AGENTS: a short routing name, the agent's base
    URL, and whether a *successful* result from this agent should be relayed
    to the user as-is (True) or composed into a reply by generate_response
    (False). A failed/errored call always forces relay_verbatim=False
    regardless of this default -- see orchestrator/agents/remote_agent.py."""

    name: str
    url: str
    relay_verbatim_on_success: bool


_CONFIG_DIR = Path(__file__).resolve().parent
_AGENTSYSTEM_ROOT = _CONFIG_DIR.parent.parent


class OrchestratorSettings(LLMClientSettings):

    model_config = SettingsConfigDict(env_file=str(_CONFIG_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    # "name@url,name@url,..." -- adding/removing an agent is a one-line
    # .env edit, not a code change. Matches the ports the standalone servers
    # default to (RAG_A2A_PORT=8001, CREDIT_LOOKUP_A2A_PORT=8002, LOAN_CALC_A2A_PORT=8003) -- "localhost"
    # here on purpose, not "0.0.0.0": that's a bind address for the servers,
    # not something a client should connect to.
    remote_agents: str = "rag@http://localhost:8001,credit_lookup@http://localhost:8002,loan_calc@http://localhost:8003"

    # Comma-separated agent names whose successful results are relayed
    # verbatim. Anyone not listed is composed into a reply, the safer
    # default for an agent whose output style isn't known in advance.
    remote_agents_verbatim: str = "rag,credit_lookup,loan_calc"

    # Where the persistent LangGraph checkpointer (AsyncSqliteSaver, see
    # orchestrator.py) writes conversation state, so it survives an
    # orchestrator process restart. Same data/-directory convention as
    # memory_manager.py's CHAT_LOG_DIR.
    checkpoint_db_path: Path = _AGENTSYSTEM_ROOT / "data" / "checkpoints" / "checkpoints.db"

    # How long the orchestrator's A2A client waits for a remote agent's
    # response before reporting an error ToolResult, instead of httpx's
    # stock 5s -- a RAG/Credit Lookup turn does retrieval plus LLM calls, which
    # comfortably exceeds 5s (see 2026-08-26 findings).
    remote_agent_timeout_seconds: float = Field(default=180.0, gt=0)


def _parse_remote_agents(raw: str, verbatim_raw: str) -> list[RemoteAgentConfig]:

    verbatim_names = {name.strip() for name in verbatim_raw.split(",") if name.strip()}

    agents = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, _, url = entry.partition("@")
        agents.append(RemoteAgentConfig(
            name=name.strip(),
            url=url.strip(),
            relay_verbatim_on_success=name.strip() in verbatim_names,
        ))

    return agents


settings = OrchestratorSettings()

REMOTE_AGENTS = _parse_remote_agents(settings.remote_agents, settings.remote_agents_verbatim)
CHECKPOINT_DB_PATH = settings.checkpoint_db_path
REMOTE_AGENT_TIMEOUT_SECONDS = settings.remote_agent_timeout_seconds
