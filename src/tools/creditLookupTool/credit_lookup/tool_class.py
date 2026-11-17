# Credit Lookup operations: the synthetic SQLite customer database (built on
# first start if missing) and text generation through the shared LLM server
# or the Gemini API (src/llm/) -- this process loads no model and needs no GPU.
from functools import partial

from llm.client import ChatLLM
from tools.creditLookupTool import config
from tools.creditLookupTool.credit_lookup.audit import append_audit
from tools.creditLookupTool.credit_lookup.datasource import SqliteCreditDataSource
from tools.creditLookupTool.credit_lookup.graph import build_credit_lookup_subgraph
from tools.creditLookupTool.seed_db import SCHEMA_VERSION, build_database, is_current


class CreditLookupTool():

    def __init__(self):

        print("Credit Lookup tool initialized.")

        # Synthetic data, so an outdated or missing database is simply rebuilt
        # (the audit log is a separate file and is kept).
        if not is_current(config.DB_PATH):
            counts = build_database(config.DB_PATH)
            print(f"Built synthetic customer database (schema v{SCHEMA_VERSION}) at {config.DB_PATH}: "
                  f"{counts['customers']} customers, {counts['contracts']} contracts.")

        self.datasource = SqliteCreditDataSource(config.DB_PATH)

        self.llm = ChatLLM.from_settings(config.settings)
        self.llm.wait_until_ready(config.settings.llm_startup_wait_seconds)

        # CreditLookupTool's per-turn logic is a compiled LangGraph subgraph
        # (see graph.py in this package), invoked by this package's own
        # standalone A2A server (a2a/a2a_server.py) -- the orchestrator
        # reaches this agent only over the network, via
        # orchestrator/agents/remote_agent.py.
        self.subgraph = build_credit_lookup_subgraph(
            self.datasource, self.llm, partial(append_audit, config.AUDIT_LOG_PATH), config.PAYMENT_MONTHS,
        )
