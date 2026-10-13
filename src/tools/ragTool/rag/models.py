# Loads the two models that still run inside the RAG process: the
# embedding model (query + ingestion) and the reranker. Generation is not
# here -- it goes through the shared LLM server (src/llm/).
#
# On CUDA both are cast to fp16 (BAAI's own recommended use_fp16=True for
# bge-m3/bge-reranker-v2-m3): ~1.1 GiB each instead of ~2.2 GiB in fp32,
# which is what the shared server's VRAM budget assumes.
import torch
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from tools.ragTool.config import EMBED_DEVICE, EMBEDDING_MODEL, RERANK_TOP_N, RERANKER_MODEL


def _half_on_cuda(model) -> None:

    if EMBED_DEVICE != "cuda":
        return

    # sentence-transformers' CrossEncoder is itself an nn.Module from v4 on;
    # older versions wrap the HF model in a .model attribute instead.
    module = model if isinstance(model, torch.nn.Module) else model.model
    module.half()


def load_embedding_model() -> HuggingFaceEmbedding:

    try:
        embedding_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            trust_remote_code=True,
            device=EMBED_DEVICE,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{EMBEDDING_MODEL}': {e}") from e

    _half_on_cuda(embedding_model._model)
    return embedding_model


def load_reranker() -> SentenceTransformerRerank:

    try:
        reranker = SentenceTransformerRerank(model=RERANKER_MODEL, top_n=RERANK_TOP_N, device=EMBED_DEVICE)
    except Exception as e:
        raise RuntimeError(f"Failed to load reranker '{RERANKER_MODEL}': {e}") from e

    _half_on_cuda(reranker._model)
    return reranker
