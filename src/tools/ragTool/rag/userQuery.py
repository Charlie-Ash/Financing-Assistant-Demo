from dataclasses import dataclass
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from tools.ragTool.config import RERANK_TOP_N, RETRIEVE_TOP_K


@dataclass(frozen=True)
class RetrievedChunk:

    text: str
    # Short source name used inline, e.g. 《消費者保護法》第 22 條
    citation: str
    # Version/date context shown next to it, e.g. 第二章 消費者權益；修正日期 2015-06-17
    detail: str = ""
    url: str = ""

    @property
    def label(self) -> str:

        return f"{self.citation}（{self.detail}）" if self.detail else self.citation


# Builds the citation from the metadata dataIngestion.py attached to each
# chunk -- laws, policies and plain documents are labelled differently.
def chunk_from_metadata(text: str, metadata: dict) -> RetrievedChunk:

    title = metadata.get("doc_title", "未知來源")
    article_no = metadata.get("article_no", "")
    source_type = metadata.get("source_type")

    if source_type == "law":
        details = [metadata.get("heading", ""), f"修正日期 {metadata['modified_date']}" if metadata.get("modified_date") else "", metadata.get("effective_note", "")]
        return RetrievedChunk(text, f"《{title}》{article_no}", "；".join(d for d in details if d), metadata.get("url", ""))

    if source_type == "policy":
        details = [metadata.get("heading", ""), f"第{metadata['version']}版" if metadata.get("version") else "", f"生效日 {metadata['effective_date']}" if metadata.get("effective_date") else ""]
        return RetrievedChunk(text, f"《{title}》{article_no}", "；".join(d for d in details if d), metadata.get("file_name", ""))

    return RetrievedChunk(text, f"《{title}》", "", metadata.get("file_name", ""))


def user_query(index: VectorStoreIndex, input_string: str, reranker: Optional[BaseNodePostprocessor] = None) -> list[RetrievedChunk]:

    # Retriever: a wide first pass (RETRIEVE_TOP_K candidates) for the
    # reranker below to narrow down; without one, fetch only as many chunks
    # as would have survived reranking, so the prompt size stays the same
    top_k = RETRIEVE_TOP_K if reranker is not None else RERANK_TOP_N
    retriever = index.as_retriever(similarity_top_k=top_k)

    # "retriever" is already connected to "QdrantVectorStore". This'll automatically embed the input, go to the DB and find relevant chunks
    results_node = retriever.retrieve(input_string)

    # Second stage: the cross-encoder re-scores each (query, chunk) pair
    # jointly and keeps only its top_n -- more precise than vector
    # similarity alone, which matters for dense legal text
    if reranker is not None:
        results_node = reranker.postprocess_nodes(results_node, query_str=input_string)

    return [chunk_from_metadata(node.text, getattr(node, "metadata", None) or {}) for node in results_node]
