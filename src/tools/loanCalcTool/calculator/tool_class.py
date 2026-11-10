# Loan Calculator operations: the demo financing plans and the LLM client
# (shared server or Gemini, see src/llm/) -- no model, no GPU, no database.
from finance.plans import load_plans
from llm.client import ChatLLM
from tools.loanCalcTool import config
from tools.loanCalcTool.calculator.graph import build_loan_calc_subgraph


class LoanCalcTool():

    def __init__(self):

        print("Loan Calculator tool initialized.")

        self.plans = load_plans()
        self.llm = ChatLLM.from_settings(config.settings)
        self.llm.wait_until_ready(config.settings.llm_startup_wait_seconds)

        # Invoked by this package's standalone A2A server (a2a/a2a_server.py);
        # the orchestrator reaches it only over the network.
        self.subgraph = build_loan_calc_subgraph(self.llm, self.plans)
