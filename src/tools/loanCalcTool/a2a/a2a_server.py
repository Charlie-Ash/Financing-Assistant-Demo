# Standalone entrypoint: starts the Loan Calculator tool as its own A2A
# protocol server process (own FastAPI/uvicorn; the LLM only reads numbers
# out of questions -- see tools/loanCalcTool/calculator/graph.py). Run as
# `python -m tools.loanCalcTool.a2a.a2a_server` from inside src/, the same way
# as the other agents.
import uvicorn

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from tools.loanCalcTool.a2a.agent_executor import LoanCalcAgentExecutor
from tools.loanCalcTool.config import LOAN_CALC_A2A_HOST, LOAN_CALC_A2A_PORT


# Builds this agent's card: the JSON document A2A clients GET at
# /.well-known/agent-card.json to discover what this agent can do. Kept as
# its own function so it can be built in tests and evals without the LLM.
def build_agent_card(host: str, port: int) -> AgentCard:

    skill = AgentSkill(
        id="quote_car_loan",
        name="Car loan payment calculator",
        description=(
            "依車價、頭期款、貸款金額、期數與利率，試算車貸的每月應付金額、總利息與總費用年百分率，"
            "支援本息平均攤還、本金平均攤還、殘值型車貸與 0 利率專案。"
            "凡是詢問「月付多少」「分幾期要繳多少」「利息多少」等一般試算問題，都應使用此工具。"
            "只做一般試算，不查詢特定客戶的合約；某位客戶現有合約的提前清償金額請改用客戶查詢。"
        ),
        tags=["loan", "calculator", "installment"],
        examples=[
            "車價 120 萬，頭期兩成，分 60 期，月付多少？",
            "RAV4 120萬，頭期兩成，殘值型48期，月付多少？",
            "借 60 萬，0 利率 24 期，每月繳多少？",
        ],
    )

    return AgentCard(
        name="Loan Calculator Agent",
        description="獨立代理：以固定公式試算車貸月付金、總利息與總費用年百分率（數字皆由程式計算）。",
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

    # Loads the demo plans and connects to the LLM backend (waiting for a
    # local server to finish loading if needed).
    from tools.loanCalcTool.calculator.tool_class import LoanCalcTool

    loan_calc_tool = LoanCalcTool()

    agent_card = build_agent_card(LOAN_CALC_A2A_HOST, LOAN_CALC_A2A_PORT)

    request_handler = DefaultRequestHandler(
        agent_executor=LoanCalcAgentExecutor(loan_calc_tool.subgraph),
        task_store=InMemoryTaskStore(),
    )

    app = A2AFastAPIApplication(agent_card=agent_card, http_handler=request_handler)

    # Blocks forever, same shape as main.py's blocking input() loop.
    uvicorn.run(app.build(), host=LOAN_CALC_A2A_HOST, port=LOAN_CALC_A2A_PORT)


if __name__ == "__main__":
    main()
