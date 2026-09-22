# Shared constants for the RAG tool.

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved relative to this file rather than the process's CWD.
RAGTOOL_DIR = Path(__file__).resolve().parent
AGENTSYSTEM_ROOT = RAGTOOL_DIR.parent.parent.parent

# Qdrant's on-disk store lives inside this tool's own folder.
QDRANT_DB_PATH = str(RAGTOOL_DIR / "qdrant_db")

# Documents to ingest live in the project's shared top-level data/ folder.
DATA_DIR = AGENTSYSTEM_ROOT / "data"


class RagSettings(BaseSettings):
    """Every env-configurable setting for the RAG agent process, loaded from
    real environment variables first, then this agent's own .env file
    (tools/ragTool/.env, see .env.example), then the defaults below --
    validated at process startup rather than deep inside vLLM later. Field
    names are lowercase versions of the actual env var names below (matched
    case-insensitively by pydantic-settings), so no explicit alias is needed."""

    model_config = SettingsConfigDict(
        env_file=str(RAGTOOL_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # Both previously hardcoded with no env override -- exposed now since a
    # Traditional Chinese RAG setup needs to swap both (embedding model,
    # possibly a differently-named collection for a rebuilt index).
    rag_embedding_model: str = "BAAI/bge-m3"
    rag_collection_name: str = "my_docs"

    # Small, official Google QAT-quantized checkpoint (~7.3 GiB VRAM). A
    # different size class from the orchestrator's E4B-it model, so the two
    # can coexist on one GPU without competing for memory.
    rag_llm_model: str = "google/gemma-4-E2B-it-qat-w4a16-ct"
    # This engine runs in its own standalone A2A server process (see
    # a2a/a2a_server.py), not sharing a process with the orchestrator's
    # larger gemma-4-E4B-it engine -- so 0.3 is no longer budgeted against a
    # shared single-process total, it's this process's own fraction of the
    # whole device. On a 32 GiB card, 0.3 nominally gives ~9.6 GiB (this
    # model's ~7.3 GiB weights + ~2.3 GiB headroom), but measured steady-state
    # usage as a separate process runs higher once fixed per-process overhead
    # (CUDA context, torch.compile cache, CUDA graphs) is included -- see
    # A2A_DIAGNOSIS.md for the real numbers and the serialized-session run
    # procedure (RAG + orchestrator together; Actions started separately)
    # this project uses to fit on one GPU.
    rag_gpu_memory_utilization: float = Field(default=0.3, gt=0, le=1)
    rag_max_model_len: int = Field(default=4096, gt=0)

    # Standalone A2A server bind address (see a2a_server.py).
    rag_a2a_host: str = "0.0.0.0"
    rag_a2a_port: int = Field(default=8001, gt=0, le=65535)


settings = RagSettings()

# Backward-compatible module-level constants -- every existing
# "from tools.ragTool.config import LLM_MODEL"-style call site elsewhere in
# the codebase keeps working unchanged.
EMBEDDING_MODEL = settings.rag_embedding_model
COLLECTION_NAME = settings.rag_collection_name
LLM_MODEL = settings.rag_llm_model
GPU_MEMORY_UTILIZATION = settings.rag_gpu_memory_utilization
MAX_MODEL_LEN = settings.rag_max_model_len
RAG_A2A_HOST = settings.rag_a2a_host
RAG_A2A_PORT = settings.rag_a2a_port
