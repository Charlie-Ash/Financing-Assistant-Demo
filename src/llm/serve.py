# Starts the ONE shared vLLM server every agent generates text through.
# Run from inside src/, before any agent:
#
#   python -m llm.serve                          # profile from src/llm/.env (default qwen3-14b-fp8)
#   python -m llm.serve --profile qwen3-8b-fp8   # one-off override, e.g. for evals
#   python -m llm.serve --dry-run                # print the vllm command only
#
# Then start the RAG/Credit Lookup A2A servers and main.py in any order -- each
# waits for this server to finish loading (llm/client.ChatLLM.wait_until_ready).
import argparse
import os
import shlex
import shutil
import sys

from llm.profiles import MODEL_PROFILES, ModelProfile
from llm.settings import LLMServerSettings


def build_serve_command(settings: LLMServerSettings, profile: ModelProfile) -> list[str]:

    command = [
        "vllm", "serve", profile.hf_id,
        "--served-model-name", settings.llm_served_name,
        "--host", settings.llm_host,
        "--port", str(settings.llm_port),
        "--gpu-memory-utilization", str(settings.llm_gpu_memory_utilization),
        "--max-model-len", str(settings.llm_max_model_len or profile.max_model_len),
        "--kv-cache-dtype", profile.kv_cache_dtype,
    ]

    if profile.quantization:
        command += ["--quantization", profile.quantization]

    return command + list(profile.extra_serve_args)


def main():

    parser = argparse.ArgumentParser(description="Start the shared vLLM server.")
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), help="overrides LLM_PROFILE")
    parser.add_argument("--dry-run", action="store_true", help="print the command without running it")
    args = parser.parse_args()

    settings = LLMServerSettings(**({"llm_profile": args.profile} if args.profile else {}))
    profile = MODEL_PROFILES[settings.llm_profile]
    command = build_serve_command(settings, profile)

    print(f"Profile: {settings.llm_profile}")
    print(" ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        return

    if shutil.which("vllm") is None:
        sys.exit("`vllm` not found on PATH -- activate the venv that has vllm installed.")

    # Imported here, not at module top, so --dry-run (and tests of
    # build_serve_command) work on a machine without torch/CUDA.
    from tools.gpu_utils import check_gpu_memory
    check_gpu_memory(settings.llm_gpu_memory_utilization)

    # Replace this process with vllm's, so Ctrl+C / signals go straight to it.
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
