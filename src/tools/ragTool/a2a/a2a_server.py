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
        id="answer_regulation_and_policy_questions",
        name="Answer regulation and internal-policy questions",
        description=(
            "查詢法規與公司內部作業規範並回答問題，答案附上條號出處（《法規名》第X條）。"
            "涵蓋：動產擔保交易法、消費者保護法、個人資料保護法、洗錢防制法，"
            "以及公司內部規範（提前清償、車貸申請文件、逾期帳款處理、個人資料處理）。"
            "凡是詢問法律規定、法條內容、公司規定或作業流程的問題，都應使用此工具。"
        ),
        tags=["rag", "regulation", "policy", "citation"],
        examples=[
            "提前清償要付違約金嗎？",
            "動產擔保交易法對附條件買賣有什麼規定？",
            "個資法規定蒐集個人資料時要告知客戶哪些事項？",
        ],
    )

    return AgentCard(
        name="Regulation & Policy Agent",
        description="獨立代理：檢索法規與公司內部規範，回答問題並附上條號出處。",
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
