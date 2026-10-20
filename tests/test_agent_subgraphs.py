# Runs the REAL RAG and Actions subgraphs (not the fake stand-ins the A2A
# tests use), with FakeChatLLM in place of the shared LLM server and small
# fakes for the vector index / reranker / note writer -- so the node logic
# itself (prompt building, reranking, JSON parsing, error paths) is covered
# GPU-free.
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage

from llm.sampling import SamplingConfig
from llm.text_norm import ZH_TW_REPLY_INSTRUCTION
from schemas.tool_call import ToolCall
from tools.actionsTool.actions import graph as actions_graph
from tools.actionsTool.actions.graph import build_actions_subgraph
from tools.actionsTool.actions.note_schema import NOTE_SCHEMA
from tools.ragTool.config import RERANK_TOP_N, RETRIEVE_TOP_K
from tools.ragTool.rag.graph import DISCLAIMER, build_rag_subgraph

from fake_dependencies import FakeChatLLM

SAMPLING = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=64)


@dataclass
class _FakeNode:
    text: str
    metadata: dict


def _law_metadata(article: int) -> dict:
    return {
        "source_type": "law", "doc_title": "消費者保護法", "article_no": f"第 {article} 條",
        "heading": "第 二 章 消費者權益", "modified_date": "2015-06-17", "effective_note": "",
        "url": f"https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=J0170001&flno={article}",
    }


class _FakeIndex:
    """Records the top_k it was asked for and returns that many chunks."""

    def __init__(self):
        self.requested_top_k = None

    def as_retriever(self, similarity_top_k):
        self.requested_top_k = similarity_top_k
        index = self

        class _Retriever:
            def retrieve(self, query):
                return [_FakeNode(f"chunk-{i}", _law_metadata(i)) for i in range(index.requested_top_k)]

        return _Retriever()


class _FakeReranker:
    """Keeps the last two nodes, reversed -- an order vector search wouldn't produce."""

    def __init__(self):
        self.query_str = None

    def postprocess_nodes(self, nodes, query_str):
        self.query_str = query_str
        return list(reversed(nodes))[:2]


def _rag_state(query):
    return {"tool_call": ToolCall(tool="rag", action="run", args={"query": query})}


def test_rag_reranks_wide_retrieval_and_relays_answer_verbatim():

    last = RETRIEVE_TOP_K - 1
    # Cites only the top-ranked of the two chunks that survive reranking.
    answer = f"提前還款違約金為剩餘本金的 1%（《消費者保護法》第 {last} 條）。"
    llm = FakeChatLLM(canned_text=f"  {answer}  ")
    index, reranker = _FakeIndex(), _FakeReranker()

    result = build_rag_subgraph(index, llm, SAMPLING, reranker).invoke(_rag_state("違約金怎麼算？"))

    assert index.requested_top_k == RETRIEVE_TOP_K
    assert reranker.query_str == "違約金怎麼算？"
    assert result["tool_result"]["relay_verbatim"] is True

    # Reranked order, each chunk labelled with its citation and date context.
    prompt = llm.calls[0]["messages"][0]["content"]
    assert f"【來源 1】《消費者保護法》第 {last} 條（第 二 章 消費者權益；修正日期 2015-06-17）\nchunk-{last}" in prompt
    assert f"【來源 2】《消費者保護法》第 {last - 1} 條" in prompt
    assert "chunk-0" not in prompt

    # The answer, then a code-built list of the sources it cites (with links),
    # then the disclaimer. The retrieved-but-uncited chunk is left out.
    output = result["tool_result"]["output"]
    assert output.startswith(f"{answer}\n\n參考來源：\n")
    assert (
        f"- 《消費者保護法》第 {last} 條（第 二 章 消費者權益；修正日期 2015-06-17） "
        f"https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=J0170001&flno={last}"
    ) in output
    assert f"第 {last - 1} 條（" not in output
    assert output.endswith(DISCLAIMER)
    assert ZH_TW_REPLY_INSTRUCTION in prompt
    assert llm.calls[0]["json_schema"] is None


def test_rag_without_reranker_retrieves_only_final_count():

    index = _FakeIndex()

    build_rag_subgraph(index, FakeChatLLM("答案"), SAMPLING).invoke(_rag_state("問題"))

    assert index.requested_top_k == RERANK_TOP_N


def test_rag_generation_failure_becomes_error_result():

    llm = FakeChatLLM(error=RuntimeError("server down"))

    result = build_rag_subgraph(_FakeIndex(), llm, SAMPLING).invoke(_rag_state("問題"))

    assert result["tool_result"]["relay_verbatim"] is False
    assert "server down" in result["tool_result"]["output"]


def test_rag_empty_query_short_circuits_without_calling_llm():

    llm = FakeChatLLM("unused")

    result = build_rag_subgraph(_FakeIndex(), llm, SAMPLING).invoke(_rag_state(""))

    assert "no query" in result["tool_result"]["output"]
    assert llm.calls == []


def test_actions_generates_note_with_schema_and_history(monkeypatch):

    saved = {}

    def fake_save_note(content, file_name):
        saved.update(content=content, file_name=file_name)
        return f"/notes/{file_name}_notes.txt"

    monkeypatch.setattr(actions_graph, "save_note", fake_save_note)

    llm = FakeChatLLM('{"content": "王先生下週二簽約", "file_name": "wang_contract"}')
    state = {
        "tool_call": ToolCall(tool="actions", action="run", args={"request": "記下來"}),
        "messages": [HumanMessage("王先生何時簽約？"), AIMessage("下週二。"), HumanMessage("記下來")],
    }

    result = build_actions_subgraph(llm, SAMPLING).invoke(state)

    assert saved == {"content": "王先生下週二簽約", "file_name": "wang_contract"}
    assert result["tool_result"] == {"output": "Note saved to /notes/wang_contract_notes.txt.", "relay_verbatim": False}

    call = llm.calls[0]
    assert call["json_schema"] == NOTE_SCHEMA
    roles = [m["role"] for m in call["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert call["messages"][-1]["content"] == "記下來"


def test_actions_invalid_json_skips_saving(monkeypatch):

    monkeypatch.setattr(actions_graph, "save_note", lambda *a: (_ for _ in ()).throw(AssertionError("must not save")))

    state = {"tool_call": ToolCall(tool="actions", action="run", args={"request": "記下來"}), "messages": []}

    result = build_actions_subgraph(FakeChatLLM("not json"), SAMPLING).invoke(state)

    assert "generation failed" in result["tool_result"]["output"]
