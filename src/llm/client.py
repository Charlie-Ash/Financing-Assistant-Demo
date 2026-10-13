# What every agent uses to generate text: a thin wrapper over the shared
# vLLM server's OpenAI-compatible chat API. Replaces each agent's own
# in-process vllm.LLM engine -- the server applies the model's chat template
# itself, so callers just pass plain role/content message lists.
import time

from openai import OpenAI

from llm.sampling import SamplingConfig
from llm.settings import LLMClientSettings
from llm.text_norm import to_traditional


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
    ):

        self.base_url = base_url
        self.served_name = served_name
        # Off only for measuring a model's raw Simplified leakage (see
        # evals/run_routing_eval.py) -- every agent keeps it on.
        self.normalize_traditional = normalize_traditional

        # vLLM ignores the API key unless started with --api-key, but the
        # client library requires one. http_client is injectable so tests
        # can swap in an httpx.MockTransport instead of a real server.
        self.client = OpenAI(
            base_url=base_url, api_key="EMPTY", timeout=timeout_seconds, http_client=http_client,
        )

    @classmethod
    def from_settings(cls, settings: LLMClientSettings) -> "ChatLLM":

        return cls(settings.llm_base_url, settings.llm_served_name, settings.llm_timeout_seconds)

    # Blocks until the server lists this client's served model, so an agent
    # started before (or alongside) llm/serve.py simply waits for the model
    # to finish loading instead of failing its first real request. Fails
    # with an actionable message rather than a bare connection error --
    # same spirit as tools/gpu_utils.check_gpu_memory().
    def wait_until_ready(self, max_wait_seconds: float, poll_interval_seconds: float = 2.0) -> None:

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

    # One chat completion. json_schema (a JSON Schema dict) turns on guided
    # decoding, so the returned text is guaranteed to parse against it --
    # the same guarantee vLLM's in-process StructuredOutputsParams gave.
    def chat(self, messages: list[dict], sampling: SamplingConfig, json_schema: dict | None = None) -> str:

        request = {
            "model": self.served_name,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            # vLLM-specific sampling param, not part of the OpenAI schema.
            "extra_body": {"repetition_penalty": sampling.repetition_penalty},
        }

        if json_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema},
            }

        response = self.client.chat.completions.create(**request)
        text = response.choices[0].message.content or ""

        return to_traditional(text) if self.normalize_traditional else text
