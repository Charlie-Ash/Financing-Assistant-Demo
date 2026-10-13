# Settings for both sides of the shared LLM server:
#   - LLMClientSettings: what every agent needs to *reach* the server. Not
#     used directly -- each agent's own config.py subclasses it, so the same
#     LLM_BASE_URL/LLM_SERVED_NAME/... fields are read from that agent's own
#     .env (letting one agent point somewhere else without affecting others).
#   - LLMServerSettings: what llm/serve.py needs to *start* the server, read
#     from src/llm/.env.
# Priority for both (pydantic-settings): real env vars > .env file > default.
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm.profiles import DEFAULT_PROFILE, MODEL_PROFILES

LLM_DIR = Path(__file__).resolve().parent

# The name the server registers its model under (--served-model-name) and
# the name every client requests. Stable across profile swaps, so agents
# never need to know which Hugging Face model is actually behind it.
DEFAULT_SERVED_NAME = "hotai-llm"


class LLMClientSettings(BaseSettings):

    model_config = SettingsConfigDict(extra="ignore")

    llm_base_url: str = "http://localhost:8000/v1"
    llm_served_name: str = DEFAULT_SERVED_NAME
    # Per-request timeout. Generous: under load the shared engine batches
    # requests from every agent, and a cold first request compiles grammars.
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    # How long an agent waits at startup for the server to come up (model
    # download/load can take minutes) before failing with a clear message.
    llm_startup_wait_seconds: float = Field(default=300.0, ge=0)


class LLMServerSettings(BaseSettings):

    model_config = SettingsConfigDict(env_file=str(LLM_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    llm_profile: str = DEFAULT_PROFILE
    llm_served_name: str = DEFAULT_SERVED_NAME
    llm_host: str = "0.0.0.0"
    llm_port: int = Field(default=8000, gt=0, le=65535)
    # Whole-card fraction for weights + CUDA graphs + KV cache. ~1.3 GiB of
    # fixed per-process overhead (CUDA context etc.) sits on top of this, and
    # the RAG process's embedding/reranker models (~2.7 GiB on CUDA) need
    # room too -- 0.77 leaves ~3 GiB margin on a 31.85 GiB RTX 5090 with the
    # default profile. See the VRAM budget in PROJECT_NOTES.
    llm_gpu_memory_utilization: float = Field(default=0.77, gt=0, le=1)
    # Overrides the profile's own max_model_len when set.
    llm_max_model_len: int | None = Field(default=None, gt=0)

    @field_validator("llm_profile")
    @classmethod
    def _known_profile(cls, value: str) -> str:

        if value not in MODEL_PROFILES:
            raise ValueError(f"unknown profile {value!r}; choose one of {sorted(MODEL_PROFILES)}")
        return value
