# Standalone entrypoint: starts the Credit Lookup tool as its own A2A
# protocol server process (own FastAPI/uvicorn; text generation goes through
# the shared LLM server or the Gemini API -- see
# tools/creditLookupTool/credit_lookup/tool_class.py). Run as
# `python -m tools.creditLookupTool.a2a.a2a_server` from inside src/, the same
# way main.py and RAG's a2a_server.py are run -- NOT `python a2a_server.py`
# directly, which would put the wrong directory on sys.path and break this
# project's absolute-import convention.
import uvicorn

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from tools.creditLookupTool.a2a.agent_executor import CreditLookupAgentExecutor
from tools.creditLookupTool.config import CREDIT_LOOKUP_A2A_HOST, CREDIT_LOOKUP_A2A_PORT


# Builds this agent's card: the JSON document A2A clients GET at
# /.well-known/agent-card.json to discover what this agent can do, before
# ever sending it work. Kept as its own function (not inlined in main())
# so it can be constructed and asserted on in tests without needing
# CreditLookupTool()'s database/LLM-server setup.
def build_agent_card(host: str, port: int) -> AgentCard:

    skill = AgentSkill(
        id="lookup_customer_credit_history",
        name="Look up a customer's history with the company",
        description=(
            "查詢特定客戶與公司的往來紀錄：基本資料、合約狀況、繳款紀錄與逾期情形，並依查詢者權限摘要；"
            "也能試算某位客戶現有合約若提前清償需付多少。"
            "凡是問到某位客戶（以姓名或客戶編號指稱，例如「王大明」或「C0003」）的合約、繳款、逾期、往來狀況或提前清償金額，都應使用此工具。"
            "只提供資料摘要，不做授信核准或拒絕的判斷。"
        ),
        tags=["credit", "customer", "lookup"],
        examples=[
            "王大明有沒有逾期紀錄？",
            "查一下 C0003 的合約狀況",
            "王大明現在提前清償要付多少？",
            "他最近一年繳款正常嗎？",
        ],
    )

    return AgentCard(
        name="Credit Lookup Agent",
        description="獨立代理：依查詢者權限，查詢並摘要客戶的合約、繳款與逾期紀錄（個人資料已遮罩）。",
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

    # Builds the demo database if missing, connects to the LLM backend
    # (waiting for a local server to finish loading if needed) and builds
    # this agent's subgraph.
    from tools.creditLookupTool.credit_lookup.tool_class import CreditLookupTool

    credit_lookup_tool = CreditLookupTool()

    agent_card = build_agent_card(CREDIT_LOOKUP_A2A_HOST, CREDIT_LOOKUP_A2A_PORT)

    request_handler = DefaultRequestHandler(
        agent_executor=CreditLookupAgentExecutor(credit_lookup_tool.subgraph),
        task_store=InMemoryTaskStore(),
    )

    app = A2AFastAPIApplication(agent_card=agent_card, http_handler=request_handler)

    # Blocks forever, same shape as main.py's blocking input() loop -- this
    # call never returns while the server is up.
    uvicorn.run(app.build(), host=CREDIT_LOOKUP_A2A_HOST, port=CREDIT_LOOKUP_A2A_PORT)


if __name__ == "__main__":
    main()
