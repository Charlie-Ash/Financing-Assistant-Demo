# Settings for both sides of the shared LLM server:
#   - LLMClientSettings: what every agent needs to *reach* an LLM -- either
#     the shared local vLLM server or Google's Gemini API (LLM_PROVIDER).
#     Not used directly -- each agent's own config.py subclasses it, so the
#     same fields are read from that agent's own .env (letting one agent use
#     Gemini while another stays on the local server).
#   - LLMServerSettings: what llm/serve.py needs to *start* the server, read
#     from src/llm/.env.
# Priority for both (pydantic-settings): real env vars > .env file > default.
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm.profiles import DEFAULT_PROFILE, MODEL_PROFILES

LLM_DIR = Path(__file__).resolve().parent

# The name the server registers its model under (--served-model-name) and
# the name every client requests. Stable across profile swaps, so agents
# never need to know which Hugging Face model is actually behind it.
DEFAULT_SERVED_NAME = "hotai-llm"

# Gemini's OpenAI-compatible endpoint: the same openai client ChatLLM uses
# for vLLM talks to it, just with a real API key.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Cheapest current stable model that routed 9/9 test prompts correctly
# (2026-10-17). It rejects reasoning_effort="none", hence "minimal" below.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


class LLMClientSettings(BaseSettings):

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    # "vllm" = the shared local server (python -m llm.serve); "gemini" =
    # Google's hosted API. Chosen per agent, in that agent's own .env.
    llm_provider: Literal["vllm", "gemini"] = "vllm"

    # --- local vLLM server (llm_provider="vllm") ---
    llm_base_url: str = "http://localhost:8000/v1"
    llm_served_name: str = DEFAULT_SERVED_NAME

    # --- Gemini API (llm_provider="gemini") ---
    # Separate fields rather than reusing llm_base_url/llm_served_name, so a
    # .env set up for Gemini is ignored by branches that don't know about it.
    llm_gemini_model: str = DEFAULT_GEMINI_MODEL
    llm_gemini_base_url: str = GEMINI_OPENAI_BASE_URL
    # How much the model may "think" before answering. Thinking tokens are
    # billed as output and add latency, so keep it as low as the model
    # accepts: gemini-3.5-flash-lite needs "minimal", while 3.1-flash-lite
    # and 3.5-flash accept "none".
    llm_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "minimal"
    # Read from GEMINI_API_KEY (or LLM_API_KEY). SecretStr keeps it out of
    # reprs, logs and tracebacks.
    llm_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("llm_api_key", "gemini_api_key"),
    )

    # How the Simplified -> Traditional safety net rewrites this agent's
    # output (see llm/text_norm.py): "s2twp" also converts mainland
    # vocabulary; "chars" only fixes non-Traditional characters and never
    # alters valid Traditional text -- for agents that quote sources verbatim.
    llm_traditional_mode: Literal["s2twp", "chars"] = "s2twp"

    # Per-request timeout. Generous: under load the shared engine batches
    # requests from every agent, and a cold first request compiles grammars.
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    # How long an agent waits at startup for the server to come up (model
    # download/load can take minutes) before failing with a clear message.
    llm_startup_wait_seconds: float = Field(default=300.0, ge=0)

    @model_validator(mode="after")
    def _gemini_needs_api_key(self) -> "LLMClientSettings":

        if self.llm_provider == "gemini" and not (self.llm_api_key and self.llm_api_key.get_secret_value().strip()):
            raise ValueError(
                "LLM_PROVIDER=gemini needs an API key: set GEMINI_API_KEY in this "
                "agent's .env (or as an environment variable)."
            )
        return self


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
