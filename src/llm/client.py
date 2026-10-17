# What every agent uses to generate text: a thin wrapper over an
# OpenAI-compatible chat API -- either the shared local vLLM server or
# Google's Gemini API (LLM_PROVIDER, see llm/settings.py). Replaces each
# agent's own in-process vllm.LLM engine -- the server applies the model's
# chat template itself, so callers just pass plain role/content message lists.
import time

from openai import APIError, APIStatusError, InternalServerError, OpenAI, RateLimitError

from llm.sampling import SamplingConfig
from llm.settings import LLMClientSettings
from llm.text_norm import to_traditional

# The hosted API answers 429 (quota) and 503 (model overloaded) far more
# often than a local server does; the openai client retries both with
# exponential backoff, so give it a few more attempts than its default 2.
GEMINI_MAX_RETRIES = 4


class LLMServerUnavailable(RuntimeError):
    pass


class ChatLLM:

    def __init__(
        self,
        base_url: str,
        served_name: str,
        timeout_seconds: float = 120.0,
        normalize_traditional: bool = True,
        http_client=None,
        provider: str = "vllm",
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ):

        self.base_url = base_url
        # The model name sent with every request: the vLLM server's served
        # name, or the Gemini model id.
        self.served_name = served_name
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        # Off only for measuring a model's raw Simplified leakage (see
        # evals/run_routing_eval.py) -- every agent keeps it on.
        self.normalize_traditional = normalize_traditional

        # vLLM ignores the API key unless started with --api-key, but the
        # client library requires one. http_client is injectable so tests
        # can swap in an httpx.MockTransport instead of a real server.
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            timeout=timeout_seconds,
            http_client=http_client,
            max_retries=GEMINI_MAX_RETRIES if provider == "gemini" else 2,
        )

    @classmethod
    def from_settings(cls, settings: LLMClientSettings, normalize_traditional: bool = True) -> "ChatLLM":

        if settings.llm_provider == "gemini":
            return cls(
                settings.llm_gemini_base_url,
                settings.llm_gemini_model,
                settings.llm_timeout_seconds,
                normalize_traditional,
                provider="gemini",
                api_key=settings.llm_api_key.get_secret_value(),
                reasoning_effort=settings.llm_reasoning_effort,
            )

        return cls(settings.llm_base_url, settings.llm_served_name, settings.llm_timeout_seconds, normalize_traditional)

    # Blocks until the server lists this client's served model, so an agent
    # started before (or alongside) llm/serve.py simply waits for the model
    # to finish loading instead of failing its first real request. Fails
    # with an actionable message rather than a bare connection error --
    # same spirit as tools/gpu_utils.check_gpu_memory().
    def wait_until_ready(self, max_wait_seconds: float, poll_interval_seconds: float = 2.0) -> None:

        if self.provider == "gemini":
            self._check_gemini_model()
            return

        deadline = time.monotonic() + max_wait_seconds
        announced = False

        while True:

            try:
                served = [model.id for model in self.client.models.list()]
            except Exception as e:
                last_error = e
            else:
                if self.served_name in served:
                    return
                raise LLMServerUnavailable(
                    f"LLM server at {self.base_url} is up but serves {served}, not "
                    f"{self.served_name!r}. LLM_SERVED_NAME must match between "
                    f"src/llm/.env and this agent's .env."
                )

            if time.monotonic() >= deadline:
                raise LLMServerUnavailable(
                    f"LLM server not reachable at {self.base_url} after waiting "
                    f"{max_wait_seconds:.0f}s ({last_error}). Start it first with "
                    f"`python -m llm.serve` from src/, or check LLM_BASE_URL in "
                    f"this agent's .env."
                )

            if not announced:
                print(f"Waiting for LLM server at {self.base_url} ...")
                announced = True

            time.sleep(poll_interval_seconds)

    # The hosted API is either up or not -- there's no model to wait for --
    # so one models.list() call checks the key and the model id, and fails
    # immediately with a message naming the setting to fix.
    def _check_gemini_model(self) -> None:

        try:
            # Gemini lists ids as "models/<id>"; requests take the bare id.
            available = [model.id.removeprefix("models/") for model in self.client.models.list()]
        except APIStatusError as e:
            if e.status_code in (400, 401, 403):
                raise LLMServerUnavailable(
                    f"Gemini API rejected the API key (HTTP {e.status_code}). Check "
                    f"GEMINI_API_KEY in this agent's .env."
                ) from e
            raise LLMServerUnavailable(f"Gemini API error while listing models: {e}") from e
        except APIError as e:
            raise LLMServerUnavailable(
                f"Gemini API not reachable at {self.base_url} ({e}). Check the internet connection."
            ) from e

        if self.served_name not in available:
            raise LLMServerUnavailable(
                f"Gemini model {self.served_name!r} isn't available to this API key. "
                f"Set LLM_GEMINI_MODEL in this agent's .env to a model listed in "
                f"Google AI Studio (e.g. gemini-3.5-flash-lite)."
            )

    # One chat completion. json_schema (a JSON Schema dict) turns on guided
    # decoding (vLLM) / structured output (Gemini), so the returned text
    # parses against it.
    def chat(self, messages: list[dict], sampling: SamplingConfig, json_schema: dict | None = None) -> str:

        request = {
            "model": self.served_name,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
        }

        if self.provider == "gemini":
            # Gemini has no repetition_penalty. reasoning_effort goes through
            # extra_body so it works on any openai client version.
            if self.reasoning_effort:
                request["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        else:
            # vLLM-specific sampling param, not part of the OpenAI schema.
            request["extra_body"] = {"repetition_penalty": sampling.repetition_penalty}

        if json_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema},
            }

        try:
            response = self.client.chat.completions.create(**request)
        except RateLimitError as e:
            if self.provider != "gemini":
                raise
            raise LLMServerUnavailable(
                f"Gemini rate limit or quota reached for {self.served_name!r} (HTTP 429), even "
                f"after retries. Free-tier requests-per-minute/day limits are shown in Google "
                f"AI Studio; slow down, wait for the reset, or enable billing."
            ) from e
        except InternalServerError as e:
            if self.provider != "gemini":
                raise
            raise LLMServerUnavailable(
                f"Gemini model {self.served_name!r} is overloaded or failing (HTTP {e.status_code}), "
                f"even after retries. Try again shortly, or set LLM_GEMINI_MODEL to another model."
            ) from e

        text = response.choices[0].message.content or ""

        return to_traditional(text) if self.normalize_traditional else text
