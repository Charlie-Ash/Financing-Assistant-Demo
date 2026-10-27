# Bridges the A2A protocol server to the Credit Lookup tool's own compiled
# subgraph (see tools/creditLookupTool/credit_lookup/graph.py). Each incoming
# A2A message becomes exactly one ToolCall(tool="credit_lookup",
# action="run", args={"request": ...}), built here from the raw message text
# -- the orchestrator only ever reaches this agent over the network (see
# orchestrator/agents/remote_agent.py). This executor runs the subgraph as a
# standalone top-level graph (tools/creditLookupTool/a2a/a2a_server.py), so
# it has no checkpointer/cross-call memory of its own.
#
# Two things cross the process boundary in the message's `metadata`, since
# no Python object can: a recent slice of chat history (so 「他」 can be
# resolved to a customer named earlier) and the caller -- {"user_id",
# "role"} -- whose role decides which fields the lookup may return. Both are
# attached by the orchestrator's call_remote_agent and reconstructed here
# into the subgraph's "messages" and "caller" state keys. A message without
# a caller gets no customer data (the subgraph's check_caller node).
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from langchain_core.messages import HumanMessage

from chat_history import chat_messages_to_history
from schemas.tool_call import ToolCall


class CreditLookupAgentExecutor(AgentExecutor):

    def __init__(self, subgraph):

        # subgraph: the compiled graph from build_credit_lookup_subgraph() --
        # either the real CreditLookupTool().subgraph (a2a_server.py) or a
        # fake one built by tests/fake_dependencies.py's
        # build_fake_credit_lookup_subgraph() (tests).
        self.subgraph = subgraph

    # Runs one turn of the Credit Lookup subgraph for an incoming A2A message
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

        tool_call = ToolCall(tool="credit_lookup", action="run", args={"request": request})

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
                new_agent_text_message(f"Credit Lookup agent failed: {e}", task.context_id, task.id)
            )
            return

        # tool_result's relay_verbatim isn't surfaced in task state here --
        # even a ToolResult with relay_verbatim=False (e.g. a denied or
        # failed lookup) still completes the task, same as how the
        # orchestrator's generate_response node treats it as content to
        # compose around rather than a failure.
        tool_result = result["tool_result"]

        await updater.add_artifact(
            [Part(root=TextPart(text=tool_result["output"]))],
            name="credit_lookup_result",
        )
        await updater.complete()

    # No cancellation support: a single subgraph.ainvoke() call has no
    # partial-progress state worth preserving.
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:

        raise ServerError(error=UnsupportedOperationError())
