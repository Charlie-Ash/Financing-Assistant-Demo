# Model profiles the shared vLLM server (llm/serve.py) can be started with.
# Swapping the model every agent uses means picking a different profile here
# (LLM_PROFILE in src/llm/.env, or `python -m llm.serve --profile ...`) and
# restarting the server -- agents never see the Hugging Face model id, only
# the stable served name (settings.DEFAULT_SERVED_NAME), so none of them
# need a code change or even a restart.
#
# VRAM notes are for the RTX 5090 (31.85 GiB total). The old three-engine
# setup measured -4 to -5 GiB short when all agents ran at once (see
# "Current Process/Experiment Data/Data_collection.md", Experiment 4); one
# shared FP8 engine pays weights + fixed per-process overhead exactly once.
from dataclasses import dataclass


# Qwen3 is a hybrid reasoning model: by default its chat template lets it
# emit a <think> block before answering. Routing (guided JSON) and the short
# generations this system does gain nothing from that except latency and
# wasted max_tokens, so it's switched off server-side for every request.
_QWEN3_NO_THINKING = ("--default-chat-template-kwargs", '{"enable_thinking": false}')


@dataclass(frozen=True)
class ModelProfile:

    hf_id: str
    max_model_len: int
    # None = serve the checkpoint as-is (already-quantized checkpoints like
    # Qwen's official *-FP8 repos carry their own quantization config).
    # "fp8" = quantize a bf16 checkpoint to FP8 on the fly at load time.
    quantization: str | None = None
    # "fp8" halves KV cache per token vs. bf16 -- Blackwell supports it natively.
    kv_cache_dtype: str = "fp8"
    extra_serve_args: tuple[str, ...] = ()


MODEL_PROFILES: dict[str, ModelProfile] = {
    # Default. ~15.5 GiB weights; strongest routing/structured-output option
    # that still leaves ~7.5 GiB KV cache at gpu_memory_utilization=0.77.
    "qwen3-14b-fp8": ModelProfile(
        hf_id="Qwen/Qwen3-14B-FP8",
        max_model_len=16384,
        extra_serve_args=_QWEN3_NO_THINKING,
    ),
    # Lighter fallback, ~8.8 GiB weights -- frees ~6.7 GiB vs. the default.
    "qwen3-8b-fp8": ModelProfile(
        hf_id="Qwen/Qwen3-8B-FP8",
        max_model_len=16384,
        extra_serve_args=_QWEN3_NO_THINKING,
    ),
    # Taiwan-tuned challenger (Traditional Chinese / Taiwan corpora),
    # ~8.5 GiB once quantized to FP8 at load. Native context is 8k.
    "llama3-taiwan-8b-fp8": ModelProfile(
        hf_id="yentinglin/Llama-3-Taiwan-8B-Instruct",
        max_model_len=8192,
        quantization="fp8",
    ),
}

DEFAULT_PROFILE = "qwen3-14b-fp8"
