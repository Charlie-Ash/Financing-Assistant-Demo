# Orchestrator-side config: which remote A2A agents to discover at startup,
# how each one's results should be relayed, and this process's own vLLM
# engine settings. Backed by pydantic-settings (env vars, then this agent's
# own .env file, then the field default below) instead of raw os.environ.get,
# so a bad value is rejected at startup instead of surfacing later -- see
# OrchestratorSettings below. Same pattern used in tools/ragTool/config.py /
# tools/actionsTool/config.py.

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


# "name@url,name@url,..." -- e.g. "rag@http://localhost:8001,actions@http://localhost:8002".
# Adding/removing an agent is a one-line env var edit, not a code change.
# Matches the ports the standalone servers already default to
# (tools/ragTool/config.py's RAG_A2A_PORT=8001, tools/actionsTool/config.py's
# ACTIONS_A2A_PORT=8002) -- "localhost" here on purpose, not "0.0.0.0": that's
# a bind address for the servers, not something a client should connect to.
_DEFAULT_REMOTE_AGENTS = "rag@http://localhost:8001,actions@http://localhost:8002"

# Comma-separated list of agent names (from the entries above) whose
# successful results should be relayed verbatim. Anyone not listed defaults
# to relay_verbatim_on_success=False (composed into a reply), the safer
# default for an agent whose output style isn't known in advance.
_DEFAULT_REMOTE_AGENTS_VERBATIM = "rag"


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


# Same data/-directory convention as memory_manager.py's CHAT_LOG_DIR, computed
# independently here rather than imported from there (this module has no
# other dependency on memory_manager.py, and both files already compute
# their own AGENTSYSTEM_ROOT the same way).
_CONFIG_DIR = Path(__file__).resolve().parent
_AGENTSYSTEM_ROOT = _CONFIG_DIR.parent.parent
_DEFAULT_CHECKPOINT_DB_PATH = _AGENTSYSTEM_ROOT / "data" / "checkpoints" / "checkpoints.db"


class OrchestratorSettings(BaseSettings):
    """Every env-configurable setting for the orchestrator process, loaded
    from real environment variables first, then this agent's own .env file
    (orchestrator/.env, see .env.example), then the defaults below -- and
    validated at process startup rather than deep inside vLLM later. Field
    names are lowercase versions of the actual env var names below (matched
    case-insensitively by pydantic-settings), so no explicit alias is needed."""

    model_config = SettingsConfigDict(
        env_file=str(_CONFIG_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    remote_agents: str = _DEFAULT_REMOTE_AGENTS
    remote_agents_verbatim: str = _DEFAULT_REMOTE_AGENTS_VERBATIM
    checkpoint_db_path: Path = _DEFAULT_CHECKPOINT_DB_PATH

    # How long the orchestrator's A2A client waits for a remote agent's
    # response before giving up and reporting an error ToolResult, instead of
    # inheriting httpx's stock 5-second default. A RAG/Actions turn does up
    # to two sequential LLM calls plus retrieval, and a freshly started agent
    # server's first real call has been observed taking ~128s (one-time
    # chat-template warm-up cost) -- both comfortably exceed 5s, which was
    # silently truncating real, in-progress calls into false timeout errors
    # (see 2026-08-26 findings).
    remote_agent_timeout_seconds: float = Field(default=180.0, gt=0)

    # This process's own vLLM engine -- previously hardcoded/read ad hoc in
    # orchestrator/pipeline/llm.py instead of living here with everything
    # else; moved in now so swapping to a Traditional Chinese model later is
    # a config change, not a code change.
    orchestrator_llm_model: str = "google/gemma-4-E4B-it"
    orchestrator_gpu_memory_utilization: float = Field(default=0.6, gt=0, le=1)
    orchestrator_max_model_len: int = Field(default=4096, gt=0)


settings = OrchestratorSettings()

# Backward-compatible module-level constants -- every existing
# "from orchestrator import config; config.REMOTE_AGENTS"-style call site
# elsewhere in the codebase keeps working unchanged.
REMOTE_AGENTS = _parse_remote_agents(settings.remote_agents, settings.remote_agents_verbatim)
CHECKPOINT_DB_PATH = settings.checkpoint_db_path
REMOTE_AGENT_TIMEOUT_SECONDS = settings.remote_agent_timeout_seconds
ORCHESTRATOR_LLM_MODEL = settings.orchestrator_llm_model
ORCHESTRATOR_GPU_MEMORY_UTILIZATION = settings.orchestrator_gpu_memory_utilization
ORCHESTRATOR_MAX_MODEL_LEN = settings.orchestrator_max_model_len
