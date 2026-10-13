# Standalone entrypoint: starts the RAG tool as its own A2A protocol server
# process (own FastAPI/uvicorn, own embedded Qdrant index + embedding and
# reranker models; text generation goes through the shared LLM server --
# see tools/ragTool/rag/tool_class.py). Run as
# `python -m tools.ragTool.a2a.a2a_server` from inside src/, the same way
# main.py is run today -- NOT `python a2a_server.py` directly, which would
# put the wrong directory on sys.path and break this project's
# absolute-import convention.
#
# Only one process may open the embedded Qdrant path (config.py's
# QDRANT_DB_PATH) at a time, so run a single RAG server per checkout.
import uvicorn

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from tools.ragTool.a2a.agent_executor import RAGAgentExecutor
from tools.ragTool.config import RAG_A2A_HOST, RAG_A2A_PORT


# Builds this agent's card: the JSON document A2A clients GET at
# /.well-known/agent-card.json to discover what this agent can do, before
# ever sending it work. Kept as its own function (not inlined in main())
# so it can be constructed and asserted on in tests without needing
# RAGTool()'s Qdrant/model/LLM-server side effects.
def build_agent_card(host: str, port: int) -> AgentCard:

    skill = AgentSkill(
        id="answer_from_company_documents",
        name="Answer questions from company documents",
        description=(
            "針對公司已匯入的文件資料（如產品說明、貸款條款、合約、內部規章）回答問題。"
            "會先檢索最相關的段落，再根據這些段落產生有依據的回答。"
            "凡是答案需要從公司文件中查找的問題，都應使用此工具。"
        ),
        tags=["rag", "documents", "question-answering"],
        examples=[
            "我們的提前還款違約金怎麼計算？",
            "請摘要新進業務的入職檢查清單。",
        ],
    )

    return AgentCard(
        name="RAG Document Agent",
        description="獨立代理：以檢索增強生成(RAG)回答公司文件相關問題。",
        url=f"http://{host}:{port}/",
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        # Deliberately honest: the executor never checks push-notification
        # config and only emits one interim "working" status, so don't
        # advertise more than that.
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        skills=[skill],
    )


def main():

    # Only place in this process that touches Qdrant/the embedding models --
    # builds the vector index + reranker and connects to the shared LLM
    # server (waiting for it to finish loading if needed).
    from tools.ragTool.rag.tool_class import RAGTool

    rag_tool = RAGTool()

    agent_card = build_agent_card(RAG_A2A_HOST, RAG_A2A_PORT)

    request_handler = DefaultRequestHandler(
        agent_executor=RAGAgentExecutor(rag_tool.subgraph),
        task_store=InMemoryTaskStore(),
    )

    app = A2AFastAPIApplication(agent_card=agent_card, http_handler=request_handler)

    # Blocks forever, same shape as main.py's blocking input() loop -- this
    # call never returns while the server is up.
    uvicorn.run(app.build(), host=RAG_A2A_HOST, port=RAG_A2A_PORT)


if __name__ == "__main__":
    main()
