# State schema for the Loan Calculator tool's own SUBGRAPH (see graph.py in
# this package), invoked by this package's standalone A2A server
# (a2a/agent_executor.py). "messages" (recent chat history + this turn) and
# "caller" are reconstructed by the executor from the A2A message metadata;
# history is what lets 「那改成 60 期呢？」 reuse the earlier numbers.
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from schemas.tool_call import ToolCall
from tools.base import ToolResult
from tools.loanCalcTool.calculator.request_schema import LoanCalcRequest
from tools.loanCalcTool.calculator.resolve import Resolution


class LoanCalcSubgraphState(TypedDict):

    tool_call: ToolCall
    messages: Annotated[list[AnyMessage], add_messages]
    caller: dict

    # Private: the question, the LLM's reading of it, and the checked inputs.
    request: str
    extracted: LoanCalcRequest
    resolution: Resolution

    tool_result: ToolResult
