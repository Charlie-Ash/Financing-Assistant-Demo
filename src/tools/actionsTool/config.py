# Shared constants for the Actions tool.

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved relative to this file rather than the process's CWD (same pattern
# as ragTool/config.py).
ACTIONSTOOL_DIR = Path(__file__).resolve().parent
AGENTSYSTEM_ROOT = ACTIONSTOOL_DIR.parent.parent.parent

# Notes are written into the project's shared top-level data/ folder, in
# their own subfolder (moved verbatim from the old notesTool/config.py).
NOTES_DIR = AGENTSYSTEM_ROOT / "data" / "notes"


class ActionsSettings(BaseSettings):
    """Every env-configurable setting for the Actions agent process, loaded
    from real environment variables first, then this agent's own .env file
    (tools/actionsTool/.env, see .env.example), then the defaults below --
    validated at process startup rather than deep inside vLLM later. Field
    names are lowercase versions of the actual env var names below (matched
    case-insensitively by pydantic-settings), so no explicit alias is needed."""

    model_config = SettingsConfigDict(
        env_file=str(ACTIONSTOOL_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # Small (1B) instruct model -- this agent's job (write a short note from
    # a user request, optionally using recent chat history) doesn't need
    # anywhere near the capacity RAG's or the orchestrator's models have.
    # This engine runs in its own standalone A2A server process (see
    # a2a/a2a_server.py); it is not sharing a process with the orchestrator
    # or RAG. In practice this project runs Actions in its own serialized
    # session (Actions + orchestrator only, RAG's server stopped) rather than
    # concurrently with both other engines -- see A2A_DIAGNOSIS.md, which
    # found the default 0.1 (~3.2 GiB) too tight even in complete isolation
    # once fixed per-process overhead (CUDA context, torch.compile cache,
    # CUDA graphs) is accounted for, and A2A_DIAGNOSIS_FIX_PLAN.md for the
    # measured numbers and current default.
    actions_llm_model: str = "google/gemma-3-1b-it"
    actions_gpu_memory_utilization: float = Field(default=0.1, gt=0, le=1)
    # Note content generation is short -- no need for the 4096 the other two
    # engines use.
    actions_max_model_len: int = Field(default=2048, gt=0)

    # Standalone A2A server bind address (see a2a/a2a_server.py). Port
    # follows RAG's 8001.
    actions_a2a_host: str = "0.0.0.0"
    actions_a2a_port: int = Field(default=8002, gt=0, le=65535)


settings = ActionsSettings()

# Backward-compatible module-level constants -- every existing
# "from tools.actionsTool.config import LLM_MODEL"-style call site elsewhere
# in the codebase keeps working unchanged.
LLM_MODEL = settings.actions_llm_model
GPU_MEMORY_UTILIZATION = settings.actions_gpu_memory_utilization
MAX_MODEL_LEN = settings.actions_max_model_len
ACTIONS_A2A_HOST = settings.actions_a2a_host
ACTIONS_A2A_PORT = settings.actions_a2a_port
