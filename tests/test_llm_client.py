# GPU-free tests for the shared LLM layer (src/llm/): ChatLLM is driven
# against an httpx.MockTransport standing in for the vLLM server's
# OpenAI-compatible API, so what's asserted is the real HTTP request the
# client would send -- not a mocked method call.
import json

import httpx
import pytest

from llm.client import ChatLLM, LLMServerUnavailable
from llm.profiles import MODEL_PROFILES
from llm.sampling import SamplingConfig
from llm.serve import build_serve_command
from llm.settings import LLMClientSettings, LLMServerSettings
from llm.text_norm import cjk_char_count, simplified_leakage_chars, to_traditional

SAMPLING = SamplingConfig(temperature=0.2, top_p=0.9, max_tokens=64, repetition_penalty=1.1)


def _chat_llm(handler, normalize_traditional=True) -> ChatLLM:

    return ChatLLM(
        base_url="http://llm.test/v1",
        served_name="hotai-llm",
        normalize_traditional=normalize_traditional,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _completion(content: str) -> httpx.Response:

    return httpx.Response(200, json={
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "hotai-llm",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    })


def test_chat_sends_served_name_sampling_and_json_schema():

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return _completion('{"tool": "rag", "action": "run", "args": {}}')

    schema = {"type": "object", "properties": {"tool": {"enum": ["default", "rag"]}}}
    text = _chat_llm(handler).chat([{"role": "user", "content": "hi"}], SAMPLING, json_schema=schema)

    body = captured["body"]
    assert captured["path"] == "/v1/chat/completions"
    assert body["model"] == "hotai-llm"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9
    assert body["max_tokens"] == 64
    assert body["repetition_penalty"] == 1.1
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert text == '{"tool": "rag", "action": "run", "args": {}}'


def test_chat_without_schema_sends_no_response_format():

    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _completion("ok")

    _chat_llm(handler).chat([{"role": "user", "content": "hi"}], SAMPLING)

    assert "response_format" not in captured["body"]


def test_chat_normalizes_simplified_to_traditional():

    llm = _chat_llm(lambda request: _completion("这个软件的信息"))

    assert llm.chat([{"role": "user", "content": "hi"}], SAMPLING) == "這個軟體的資訊"


def test_chat_can_return_raw_text_for_leakage_measurement():

    llm = _chat_llm(lambda request: _completion("这个软件"), normalize_traditional=False)

    assert llm.chat([{"role": "user", "content": "hi"}], SAMPLING) == "这个软件"


def test_wait_until_ready_returns_once_served_model_is_listed():

    def handler(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"object": "list", "data": [{"id": "hotai-llm", "object": "model", "created": 0, "owned_by": "vllm"}]})

    _chat_llm(handler).wait_until_ready(max_wait_seconds=0)


def test_wait_until_ready_rejects_wrong_served_name():

    def handler(request):
        return httpx.Response(200, json={"object": "list", "data": [{"id": "other-model", "object": "model", "created": 0, "owned_by": "vllm"}]})

    with pytest.raises(LLMServerUnavailable, match="LLM_SERVED_NAME"):
        _chat_llm(handler).wait_until_ready(max_wait_seconds=0)


def test_wait_until_ready_fails_with_actionable_message_when_unreachable():

    def handler(request):
        raise httpx.ConnectError("connection refused")

    llm = _chat_llm(handler)
    llm.client = llm.client.with_options(max_retries=0)

    with pytest.raises(LLMServerUnavailable, match="python -m llm.serve"):
        llm.wait_until_ready(max_wait_seconds=0)


def _gemini_llm(handler, reasoning_effort="minimal") -> ChatLLM:

    llm = ChatLLM(
        base_url="https://gemini.test/v1beta/openai/",
        served_name="gemini-3.5-flash-lite",
        provider="gemini",
        api_key="AQ.test-key",
        reasoning_effort=reasoning_effort,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    # No real backoff sleeps in tests.
    llm.client = llm.client.with_options(max_retries=0)
    return llm


def _model_list(*ids: str) -> httpx.Response:

    return httpx.Response(200, json={"object": "list", "data": [
        {"id": model_id, "object": "model", "created": 0, "owned_by": "google"} for model_id in ids
    ]})


def test_gemini_chat_sends_key_and_reasoning_effort_but_no_repetition_penalty():

    captured = {}

    def handler(request):
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return _completion("好的")

    _gemini_llm(handler).chat([{"role": "user", "content": "hi"}], SAMPLING)

    body = captured["body"]
    assert captured["auth"] == "Bearer AQ.test-key"
    assert body["model"] == "gemini-3.5-flash-lite"
    assert body["reasoning_effort"] == "minimal"
    assert "repetition_penalty" not in body


def test_gemini_wait_until_ready_accepts_models_prefix():

    _gemini_llm(lambda request: _model_list("models/gemini-3.5-flash", "models/gemini-3.5-flash-lite")).wait_until_ready(0)


def test_gemini_wait_until_ready_names_the_model_setting_when_unavailable():

    with pytest.raises(LLMServerUnavailable, match="LLM_GEMINI_MODEL"):
        _gemini_llm(lambda request: _model_list("models/gemini-3.5-flash")).wait_until_ready(0)


def test_gemini_rejected_key_fails_fast_naming_the_key_setting():

    def handler(request):
        return httpx.Response(401, json={"error": {"code": 401, "message": "API key not valid", "status": "UNAUTHENTICATED"}})

    with pytest.raises(LLMServerUnavailable, match="GEMINI_API_KEY"):
        _gemini_llm(handler).wait_until_ready(max_wait_seconds=300)


def test_gemini_rate_limit_becomes_a_friendly_error():

    def handler(request):
        return httpx.Response(429, json={"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})

    with pytest.raises(LLMServerUnavailable, match="429"):
        _gemini_llm(handler).chat([{"role": "user", "content": "hi"}], SAMPLING)


def test_gemini_overload_becomes_a_friendly_error():

    def handler(request):
        return httpx.Response(503, json={"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})

    with pytest.raises(LLMServerUnavailable, match="LLM_GEMINI_MODEL"):
        _gemini_llm(handler).chat([{"role": "user", "content": "hi"}], SAMPLING)


def test_from_settings_picks_gemini_model_url_and_key():

    settings = LLMClientSettings(
        _env_file=None, llm_provider="gemini", llm_api_key="AQ.k",
        llm_gemini_model="gemini-3.5-flash", llm_reasoning_effort="none",
    )
    llm = ChatLLM.from_settings(settings)

    assert llm.provider == "gemini"
    assert llm.served_name == "gemini-3.5-flash"
    assert llm.reasoning_effort == "none"
    assert str(llm.client.base_url).startswith("https://generativelanguage.googleapis.com/v1beta/openai")
    assert llm.client.api_key == "AQ.k"


def test_from_settings_defaults_to_local_vllm_server():

    llm = ChatLLM.from_settings(LLMClientSettings(_env_file=None))

    assert llm.provider == "vllm"
    assert llm.served_name == "hotai-llm"
    assert llm.client.api_key == "EMPTY"


def test_chars_mode_never_alters_valid_traditional_text():

    # Official 動產擔保交易法 wording: s2twp would rewrite 占 -> 佔 and 致 -> 緻.
    law_text = "債務人不履行契約或抵押物被出賣，致有害於抵押權之行使者，抵押權人得占有抵押物。"

    assert to_traditional(law_text, "chars") == law_text
    assert to_traditional(law_text) != law_text  # the default mode does alter it
    assert to_traditional("动产担保交易法第17条", "chars") == "動產擔保交易法第17條"


def test_traditional_mode_is_applied_to_replies():

    llm = _chat_llm(lambda request: _completion("抵押權人得占有抵押物"))
    llm.traditional_mode = "chars"

    assert llm.chat([{"role": "user", "content": "hi"}], SAMPLING) == "抵押權人得占有抵押物"


def test_leakage_is_zero_for_traditional_text_and_counts_simplified_chars():

    assert simplified_leakage_chars("這個軟體的資訊 OK") == 0
    # 这->這, 个->個, 软件->軟體 (2), 信息->資訊 (2); 的 is shared
    assert simplified_leakage_chars("这个软件的信息") == 6
    assert cjk_char_count("這個 app 很好") == 4


def test_serve_command_for_default_profile():

    settings = LLMServerSettings(_env_file=None)
    command = build_serve_command(settings, MODEL_PROFILES[settings.llm_profile])

    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3-14B-FP8"]
    assert command[command.index("--served-model-name") + 1] == "hotai-llm"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.77"
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8"
    assert json.loads(command[command.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": False}
    assert "--quantization" not in command


def test_serve_command_quantizes_on_the_fly_and_honours_max_len_override():

    settings = LLMServerSettings(_env_file=None, llm_profile="llama3-taiwan-8b-fp8", llm_max_model_len=4096)
    command = build_serve_command(settings, MODEL_PROFILES[settings.llm_profile])

    assert command[command.index("--quantization") + 1] == "fp8"
    assert command[command.index("--max-model-len") + 1] == "4096"
    assert "--default-chat-template-kwargs" not in command
