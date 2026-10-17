# Shared constants for the RAG tool. Read once at import by RagSettings
# (pydantic-settings): real env vars > src/tools/ragTool/.env > defaults
# below, validated at startup. See .env.example next to this file for every
# setting.
#
# No generation-model/VRAM settings live here any more: this agent
# generates text through the shared LLM server or the Gemini API
# (LLM_PROVIDER, inherited from llm/settings.LLMClientSettings). The
# embedding and reranker models below still run inside this process.
import re
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from llm.settings import LLMClientSettings

# Resolved relative to this file rather than the process's CWD.
RAGTOOL_DIR = Path(__file__).resolve().parent
AGENTSYSTEM_ROOT = RAGTOOL_DIR.parent.parent.parent

# Qdrant's on-disk store lives inside this tool's own folder.
QDRANT_DB_PATH = str(RAGTOOL_DIR / "qdrant_db")

# Documents to ingest live in the project's shared top-level data/ folder.
DATA_DIR = AGENTSYSTEM_ROOT / "data"


class RagSettings(LLMClientSettings):

    model_config = SettingsConfigDict(env_file=str(RAGTOOL_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    rag_collection_name: str = "my_docs"
    rag_embedding_model: str = "BAAI/bge-m3"
    # Second-stage cross-encoder: re-scores the top rag_retrieve_top_k
    # vector hits against the query and keeps the best rag_rerank_top_n.
    rag_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Where the embedding + reranker models run. "cuda" costs ~2.7 GiB
    # (both models in fp16 plus this process's CUDA context) and is already
    # counted in the shared server's VRAM budget; "cpu" frees that at the
    # cost of slower ingestion and a few seconds of reranking per query.
    rag_embed_device: Literal["cpu", "cuda"] = "cuda"
    rag_retrieve_top_k: int = Field(default=20, gt=0)
    rag_rerank_top_n: int = Field(default=4, gt=0)

    # Standalone A2A server bind address (see a2a/a2a_server.py).
    rag_a2a_host: str = "0.0.0.0"
    rag_a2a_port: int = Field(default=8001, gt=0, le=65535)


settings = RagSettings()

EMBEDDING_MODEL = settings.rag_embedding_model
RERANKER_MODEL = settings.rag_reranker_model
EMBED_DEVICE = settings.rag_embed_device
RETRIEVE_TOP_K = settings.rag_retrieve_top_k
RERANK_TOP_N = settings.rag_rerank_top_n

# The embedding model is part of the collection name, so switching models
# builds a fresh collection instead of silently querying vectors from a
# different embedding space (e.g. "my_docs__bge-m3").
COLLECTION_NAME = (
    f"{settings.rag_collection_name}__"
    f"{re.sub(r'[^A-Za-z0-9._-]+', '-', EMBEDDING_MODEL.split('/')[-1])}"
)

RAG_A2A_HOST = settings.rag_a2a_host
RAG_A2A_PORT = settings.rag_a2a_port
