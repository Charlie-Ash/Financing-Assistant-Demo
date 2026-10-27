# The pydantic-settings classes must reject bad values at construction --
# i.e. at process startup -- naming the offending field, instead of letting
# a malformed value surface later somewhere deep inside vLLM or uvicorn.
# _env_file=None keeps a developer's local .env out of these tests.
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm.settings import LLMServerSettings
from orchestrator.config import OrchestratorSettings
from tools.creditLookupTool.config import CreditLookupSettings
from tools.ragTool.config import RagSettings


def test_env_vars_are_read_and_coerced(monkeypatch):

    monkeypatch.setenv("RAG_A2A_PORT", "9001")
    monkeypatch.setenv("LLM_BASE_URL", "http://gpu-box:8000/v1")

    settings = RagSettings(_env_file=None)

    assert settings.rag_a2a_port == 9001
    assert settings.llm_base_url == "http://gpu-box:8000/v1"


def test_rag_data_dir_is_read_as_a_path(monkeypatch):

    monkeypatch.setenv("RAG_DATA_DIR", "data/other_docs")

    settings = RagSettings(_env_file=None)

    assert settings.rag_data_dir == Path("data/other_docs")


def test_only_the_regulation_agent_defaults_to_the_character_only_safety_net():

    assert RagSettings(_env_file=None).llm_traditional_mode == "chars"
    assert OrchestratorSettings(_env_file=None).llm_traditional_mode == "s2twp"
    assert CreditLookupSettings(_env_file=None).llm_traditional_mode == "s2twp"


def test_non_integer_port_fails_naming_the_field(monkeypatch):

    monkeypatch.setenv("CREDIT_LOOKUP_A2A_PORT", "notaport")

    with pytest.raises(ValidationError, match="credit_lookup_a2a_port"):
        CreditLookupSettings(_env_file=None)


def test_out_of_range_gpu_fraction_fails_naming_the_field(monkeypatch):

    monkeypatch.setenv("LLM_GPU_MEMORY_UTILIZATION", "1.4")

    with pytest.raises(ValidationError, match="llm_gpu_memory_utilization"):
        LLMServerSettings(_env_file=None)


def test_unknown_profile_lists_the_valid_ones():

    with pytest.raises(ValidationError, match="qwen3-14b-fp8"):
        LLMServerSettings(_env_file=None, llm_profile="gemma-4")


def test_embed_device_only_accepts_cpu_or_cuda(monkeypatch):

    monkeypatch.setenv("RAG_EMBED_DEVICE", "gpu")

    with pytest.raises(ValidationError, match="rag_embed_device"):
        RagSettings(_env_file=None)


def test_gemini_provider_without_api_key_fails_naming_the_key(monkeypatch):

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        OrchestratorSettings(_env_file=None)


def test_unknown_provider_is_rejected(monkeypatch):

    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(ValidationError, match="llm_provider"):
        RagSettings(_env_file=None)


def test_gemini_api_key_env_var_is_read_and_kept_secret(monkeypatch):

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.secret-value")

    settings = CreditLookupSettings(_env_file=None)

    assert settings.llm_api_key.get_secret_value() == "AQ.secret-value"
    assert "AQ.secret-value" not in repr(settings)


def test_non_positive_timeout_is_rejected(monkeypatch):

    monkeypatch.setenv("REMOTE_AGENT_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValidationError, match="remote_agent_timeout_seconds"):
        OrchestratorSettings(_env_file=None)
