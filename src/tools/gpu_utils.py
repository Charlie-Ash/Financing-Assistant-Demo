# GPU pre-flight check, run by llm/serve.py before starting the shared vLLM
# server -- the only process in this system that loads an LLM engine.
import torch


def check_gpu_memory(gpu_memory_utilization: float) -> None:
    """Fail fast with an actionable message instead of vLLM's bare ValueError.

    vLLM's own pre-flight check just reports free vs. required VRAM with no
    guidance on why memory might already be missing.
    """

    if not torch.cuda.is_available():
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024 ** 3)
    total_gib = total_bytes / (1024 ** 3)
    required_gib = total_gib * gpu_memory_utilization

    print(f"GPU memory: {free_gib:.1f} GiB free / {total_gib:.1f} GiB total "
          f"(vLLM will request ~{required_gib:.1f} GiB)")

    if free_gib < required_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB free, but gpu_memory_utilization="
            f"{gpu_memory_utilization} needs ~{required_gib:.1f} GiB on a "
            f"{total_gib:.1f} GiB device. Likely causes: a stale vLLM worker "
            f"process from a previous crashed run still holding the CUDA "
            f"context (check `nvidia-smi` and kill leftover python "
            f"processes), another application using the GPU (e.g. an old "
            f"Docker container), or a second LLM server already running. "
            f"Lower LLM_GPU_MEMORY_UTILIZATION in src/llm/.env, pick a "
            f"smaller model with `--profile`, or free up VRAM before retrying."
        )
