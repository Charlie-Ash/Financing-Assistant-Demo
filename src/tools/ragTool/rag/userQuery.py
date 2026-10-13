from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from tools.ragTool.config import RERANK_TOP_N, RETRIEVE_TOP_K


def user_query(index: VectorStoreIndex, input_string: str, reranker: Optional[BaseNodePostprocessor] = None) -> list[str]:

    # Retriever: a wide first pass (RETRIEVE_TOP_K candidates) for the
    # reranker below to narrow down; without one, fetch only as many chunks
    # as would have survived reranking, so the prompt size stays the same
    top_k = RETRIEVE_TOP_K if reranker is not None else RERANK_TOP_N
    retriever = index.as_retriever(similarity_top_k=top_k)

    # "retriever" is already connected to "QdrantVectorStore". This'll automatically embed the input, go to the DB and find relevant chunks
    results_node = retriever.retrieve(input_string)

    # Second stage: the cross-encoder re-scores each (query, chunk) pair
    # jointly and keeps only its top_n -- more precise than vector
    # similarity alone, which matters for dense contract/loan-term text
    if reranker is not None:
        results_node = reranker.postprocess_nodes(results_node, query_str=input_string)

    # Tranfer relevant text into a list
    # Note: returning chunk size is already determined as 512 during ingestion
    results_text_list = [node.text for node in results_node]

    return results_text_list
