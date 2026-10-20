# State schema for the Credit Lookup tool's own SUBGRAPH (see graph.py in
# this package), invoked by this package's standalone A2A server
# (a2a/agent_executor.py). tool_call/tool_result/messages share their names
# with OrchestratorState's fields (see orchestrator/pipeline/state.py);
# "messages" and "caller" are reconstructed by the executor from the
# metadata the orchestrator attaches to each A2A message -- recent chat
# history (to resolve 「他」 / 「那位客戶」) and who is asking (user id + role).
# Everything else is private working state: LangGraph never merges
# undeclared fields back to a caller.
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from schemas.tool_call import ToolCall
from tools.base import ToolResult


class CreditLookupSubgraphState(TypedDict):

    # Shared with OrchestratorState: set by the parent's decide_tool node.
    tool_call: ToolCall

    # Chat history before this turn plus this turn's message, read by
    # extract_request to resolve references to earlier turns.
    messages: Annotated[list[AnyMessage], add_messages]

    # Who is asking: {"user_id": ..., "role": "sales" | "credit"}.
    caller: dict

    # Private: the raw question pulled out of tool_call.args.
    request: str

    # Private: extract_request's structured reading of the question.
    customer_name: str
    customer_id: str
    focus: str

    # Private: resolve_customer's single match, then the role-filtered,
    # masked view built from it (see policy.build_view).
    customer: dict
    view: dict

    # Shared with OrchestratorState: read by the parent's generate_response node.
    tool_result: ToolResult
