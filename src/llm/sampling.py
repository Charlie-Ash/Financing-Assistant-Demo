# Per-call generation settings. Replaces vllm.SamplingParams in agent code,
# so no agent process needs vllm (or a GPU) installed to generate text --
# only the shared server (llm/serve.py) does.
from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingConfig:

    temperature: float
    top_p: float
    max_tokens: int
    repetition_penalty: float = 1.0
