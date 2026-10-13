# Actions (note-writing) operations: text generation through the shared LLM
# server (src/llm/) -- this process loads no model and needs no GPU.
from llm.client import ChatLLM
from llm.sampling import SamplingConfig
from tools.actionsTool import config
from tools.actionsTool.actions.graph import build_actions_subgraph


class ActionsTool():

    def __init__(self):

        print("Actions tool initialized.")

        self.llm = ChatLLM.from_settings(config.settings)
        self.llm.wait_until_ready(config.settings.llm_startup_wait_seconds)

        self.sampling = SamplingConfig(
            temperature=0.3,  # some randomness for natural note phrasing, but still fairly literal
            top_p=0.9,  # top_p; nucleus sampling
            max_tokens=300,  # notes are meant to stay short
            repetition_penalty=1.1,  # Penalty to apply if tokens continue repeating.
        )

        # ActionsTool's per-turn logic is a compiled LangGraph subgraph (see
        # graph.py in this package), invoked by this package's own standalone
        # A2A server (a2a/a2a_server.py) -- the orchestrator reaches this
        # agent only over the network, via orchestrator/agents/remote_agent.py.
        self.subgraph = build_actions_subgraph(self.llm, self.sampling)
