# Bridges the A2A protocol server to the Loan Calculator tool's own compiled
# subgraph (see tools/loanCalcTool/calculator/graph.py). Each incoming A2A
# message becomes one ToolCall(tool="loan_calc", action="run",
# args={"request": ...}). Recent chat history (so 「那改成 60 期呢？」 can reuse
# the earlier numbers) and the caller arrive in the message's metadata, the
# same way they do for the Credit Lookup agent.
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from langchain_core.messages import HumanMessage

from chat_history import chat_messages_to_history
from schemas.tool_call import ToolCall


class LoanCalcAgentExecutor(AgentExecutor):

    def __init__(self, subgraph):

        # subgraph: the compiled graph from build_loan_calc_subgraph() --
        # either the real LoanCalcTool().subgraph (a2a_server.py) or a fake
        # one built by tests/fake_dependencies.py's
        # build_fake_loan_calc_subgraph() (tests).
        self.subgraph = subgraph

    # Runs one turn of the Loan Calculator subgraph for an incoming A2A message
    # and reports the result back as a completed (or failed) task.
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:

        request = context.get_user_input()

        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # One cheap, honest progress signal before the (potentially slow,
        # LLM-backed) invoke below -- same reasoning as RAGAgentExecutor.
        await updater.start_work()

        tool_call = ToolCall(tool="loan_calc", action="run", args={"request": request})

        # Reconstruct what the orchestrator attached to this message (see
        # module comment above); [] / None if nothing was sent.
        metadata = (context.message.metadata or {}) if context.message else {}
        history_messages = chat_messages_to_history(metadata.get("history", []))

        try:
            result = await self.subgraph.ainvoke({
                "tool_call": tool_call,
                "messages": [*history_messages, HumanMessage(content=request)],
                "caller": metadata.get("caller"),
            })
        except Exception as e:
            await updater.failed(
                new_agent_text_message(f"Loan Calculator agent failed: {e}", task.context_id, task.id)
            )
            return

        # tool_result's relay_verbatim isn't surfaced in task state here --
        # even a ToolResult with relay_verbatim=False (e.g. a request the
        # LLM failed to read) still completes the task, same as how the
        # orchestrator's generate_response node treats it as content to
        # compose around rather than a failure.
        tool_result = result["tool_result"]

        await updater.add_artifact(
            [Part(root=TextPart(text=tool_result["output"]))],
            name="loan_calc_result",
        )
        await updater.complete()

    # No cancellation support: a single subgraph.ainvoke() call has no
    # partial-progress state worth preserving.
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:

        raise ServerError(error=UnsupportedOperationError())
