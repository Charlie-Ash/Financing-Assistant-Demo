# Builds the RAG tool's own SUBGRAPH. Mirrors orchestrator/pipeline/graph.py's
# build_graph(llm, tool_router): takes the already-loaded index/llm/sampling/
# reranker as parameters instead of constructing them itself, so this module's
# wiring can be tested without a GPU or LLM server (see tests/fake_dependencies.py).
from typing import TYPE_CHECKING, Optional

from langgraph.graph import StateGraph, START, END

from llm.text_norm import ZH_TW_REPLY_INSTRUCTION
from tools.ragTool.rag.state import RAGSubgraphState
from tools.ragTool.rag.userQuery import RetrievedChunk, user_query

# Imported only for type names in string-quoted annotations on
# build_rag_subgraph, so importing this module stays lightweight
if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llm.client import ChatLLM
    from llm.sampling import SamplingConfig

SYS_PROMPT = (
    "你是一位金融法規與公司內部規範的查詢助理。請只根據下方標示【來源】的資料回答使用者的問題。"
    "每一項說明都要註明出處，格式為《名稱》第X條，例如《消費者保護法》第 22 條。"
    "若資料不足以回答，請直接說明查無相關規定，不要自行推測或補充資料以外的內容。"
    "請只說明規定內容，不要對個案做出確定的法律判斷。"
    + ZH_TW_REPLY_INSTRUCTION
)

# Appended in code after every answer, so the traceability doesn't depend
# on the model remembering to include it.
DISCLAIMER = "以上僅供參考，實際適用請依主管機關公告及法務意見為準。"


def format_sources(chunks: list["RetrievedChunk"]) -> str:

    """The deterministic 參考來源 list: one line per distinct source, in rank order."""

    lines, seen = [], set()
    for chunk in chunks:
        if chunk.label in seen:  # a long article split into several chunks
            continue
        seen.add(chunk.label)
        lines.append(f"- {chunk.label}" + (f" {chunk.url}" if chunk.url else ""))

    return "參考來源：\n" + "\n".join(lines)


def build_rag_subgraph(
    index: "VectorStoreIndex",
    llm: "ChatLLM",
    sampling: "SamplingConfig",
    reranker: Optional["BaseNodePostprocessor"] = None,
):
    """Assemble and compile the RAG tool's subgraph: extract_query -> retrieve ->
    generate_answer, with an empty-query short-circuit straight to END.

    Compiled with no explicit checkpointer, so it inherits the parent
    orchestrator graph's MemorySaver when run as a node there (LangGraph's
    default subgraph behaviour). This subgraph's own internal state
    (query/retrieved_chunks) is recomputed fresh every call and never needs to
    survive across turns, so inheriting costs a little unnecessary
    serialization -- but it's what keeps this a real, introspectable subgraph:
    compiling with checkpointer=False makes LangGraph's find_subgraph_pregel()
    (langgraph/pregel/_utils.py) treat the node as if it weren't a subgraph at
    all, hiding it from get_graph(xray=True)/get_subgraphs() and anything
    built on those (e.g. LangGraph Studio's subgraph view). Every field here
    is a plain str/list or a small frozen dataclass (RetrievedChunk), so the
    serialization cost is cheap; keep it that way if this subgraph's state
    ever grows.
    """

    graph = StateGraph(RAGSubgraphState)

    # Node: pull the query out of this turn's tool_call.args. If it's missing,
    # also set the same error tool_result the rest of the graph would otherwise
    # have had to special-case downstream -- route_after_extract below sends
    # this straight to END instead of attempting retrieval/generation.
    def extract_query(state: RAGSubgraphState) -> dict:

        query = state["tool_call"].args.get("query", "")

        if not query:
            return {
                "query": query,
                "tool_result": {"output": "RAG tool error: no query was provided.", "relay_verbatim": False},
            }

        return {"query": query}

    # Routing function for the conditional edge below: decides whether there's
    # a query worth retrieving/generating for at all.
    def route_after_extract(state: RAGSubgraphState) -> str:

        return "has_query" if state["query"] else "empty_query"

    # Node: fetch the top matching chunks for this turn's query.
    def retrieve(state: RAGSubgraphState) -> dict:

        retrieved_chunks = user_query(index, state["query"], reranker)
        return {"retrieved_chunks": retrieved_chunks}

    # Node: turn the retrieved chunks + query into a generated answer.
    def generate_answer(state: RAGSubgraphState) -> dict:

        chunks = state["retrieved_chunks"]
        context_block = "\n\n".join(
            f"【來源 {i}】{chunk.label}\n{chunk.text}" for i, chunk in enumerate(chunks, start=1)
        )

        user_message = f"{SYS_PROMPT}\n\n{context_block}\n\n使用者問題:\n{state['query']}"
        messages = [{"role": "user", "content": user_message}]

        try:
            answer = llm.chat(messages, sampling).strip()
        except Exception as e:
            return {"tool_result": {"output": f"RAG tool error: generation failed ({e})", "relay_verbatim": False}}

        if chunks:
            answer = f"{answer}\n\n{format_sources(chunks)}\n\n{DISCLAIMER}"

        # This answer was already fully generated by the RAG tool's own model,
        # so relay_verbatim=True tells the response-generation step to show it
        # as-is rather than paraphrase/summarize it further.
        return {"tool_result": {"output": answer, "relay_verbatim": True}}

    graph.add_node("extract_query", extract_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "extract_query")
    graph.add_conditional_edges(
        "extract_query",
        route_after_extract,
        {
            "has_query": "retrieve",
            "empty_query": END,
        },
    )
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
