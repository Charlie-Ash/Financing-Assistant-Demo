# Drives CreditLookupAgentExecutor through the real a2a-sdk request-handling stack
# (DefaultRequestHandler -> TaskManager -> ResultAggregator ->
# InMemoryTaskStore), using build_fake_credit_lookup_subgraph() in place of the
# real GPU-backed subgraph -- a2a-sdk itself needs no GPU, so this is real
# (non-mocked) framework coverage of the executor's wiring, just with a fake
# Credit Lookup subgraph underneath it. See fake_dependencies.py for why the fake
# is an actually-compiled StateGraph rather than a bare stub function.
import uuid

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    TaskState,
    TextPart,
)
from a2a.utils.errors import ServerError

from langgraph.graph import StateGraph, START, END

from tools.creditLookupTool.a2a.agent_executor import CreditLookupAgentExecutor
from tools.creditLookupTool.credit_lookup.state import CreditLookupSubgraphState

from fake_dependencies import build_fake_credit_lookup_subgraph


def _send_params(text: str, history: list[dict] | None = None, caller: dict | None = None) -> MessageSendParams:

    metadata = {key: value for key, value in (("history", history), ("caller", caller)) if value}
    return MessageSendParams(
        message=Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            message_id=str(uuid.uuid4()),
            metadata=metadata or None,
        )
    )


def _build_handler(
    canned_output="credit-lookup-output", relay_verbatim=False, received_args_sink=None, received_messages_sink=None,
    received_caller_sink=None,
):

    subgraph = build_fake_credit_lookup_subgraph(
        canned_output=canned_output,
        relay_verbatim=relay_verbatim,
        received_args_sink=received_args_sink,
        received_messages_sink=received_messages_sink,
        received_caller_sink=received_caller_sink,
    )
    executor = CreditLookupAgentExecutor(subgraph)
    return DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())


async def test_message_send_completes_and_returns_the_subgraphs_answer():

    handler = _build_handler(canned_output="王大明（C0001）近12個月有逾期紀錄：無")
    result = await handler.on_message_send(_send_params("王大明有沒有逾期紀錄？"))

    assert result.status.state == TaskState.completed
    assert len(result.artifacts) == 1
    assert result.artifacts[0].parts[0].root.text == "王大明（C0001）近12個月有逾期紀錄：無"


async def test_message_send_passes_the_users_text_through_as_the_credit_lookup_request():

    received_args_sink = {"args": None}
    handler = _build_handler(received_args_sink=received_args_sink)

    await handler.on_message_send(_send_params("查一下 C0003 的合約"))

    assert received_args_sink["args"] == {"request": "查一下 C0003 的合約"}


async def test_message_send_passes_history_metadata_through_to_the_subgraph_as_messages():

    received_messages_sink = {"messages": None}
    handler = _build_handler(received_messages_sink=received_messages_sink)

    history = [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "canned reply"},
    ]
    await handler.on_message_send(_send_params("他最近有逾期嗎？", history=history))

    received = received_messages_sink["messages"]
    assert [(type(m).__name__, m.content) for m in received] == [
        ("HumanMessage", "first message"),
        ("AIMessage", "canned reply"),
        ("HumanMessage", "他最近有逾期嗎？"),
    ]


async def test_message_send_passes_caller_metadata_through_to_the_subgraph():

    received_caller_sink = {"caller": "unset"}
    handler = _build_handler(received_caller_sink=received_caller_sink)

    await handler.on_message_send(_send_params("王大明有沒有逾期紀錄？", caller={"user_id": "lin02", "role": "credit"}))

    assert received_caller_sink["caller"] == {"user_id": "lin02", "role": "credit"}


async def test_message_send_without_caller_metadata_passes_none():
    # The subgraph then refuses to return customer data (see check_caller).

    received_caller_sink = {"caller": "unset"}
    handler = _build_handler(received_caller_sink=received_caller_sink)

    await handler.on_message_send(_send_params("王大明有沒有逾期紀錄？"))

    assert received_caller_sink["caller"] is None


async def test_message_send_with_no_history_metadata_passes_only_the_current_turn():

    received_messages_sink = {"messages": None}
    handler = _build_handler(received_messages_sink=received_messages_sink)

    await handler.on_message_send(_send_params("a message with no history"))

    received = received_messages_sink["messages"]
    assert [(type(m).__name__, m.content) for m in received] == [("HumanMessage", "a message with no history")]


async def test_two_calls_are_independent_of_each_other():
    # No checkpointer is attached when the subgraph is invoked as a
    # standalone top-level graph (see graph.py's build_credit_lookup_subgraph()
    # docstring), so two separate calls should each get their own fresh
    # completed task rather than colliding on shared state.

    handler = _build_handler(canned_output="saved")

    result_1 = await handler.on_message_send(_send_params("查詢一"))
    result_2 = await handler.on_message_send(_send_params("查詢二"))

    assert result_1.id != result_2.id
    assert result_1.status.state == TaskState.completed
    assert result_2.status.state == TaskState.completed


async def test_subgraph_exception_fails_the_task_instead_of_raising():

    def broken_run(state: CreditLookupSubgraphState) -> dict:
        raise RuntimeError("boom")

    sub = StateGraph(CreditLookupSubgraphState)
    sub.add_node("broken_run", broken_run)
    sub.add_edge(START, "broken_run")
    sub.add_edge("broken_run", END)
    broken_subgraph = sub.compile()

    executor = CreditLookupAgentExecutor(broken_subgraph)
    context = RequestContext(request=_send_params("this will blow up"))
    queue = EventQueue()

    await executor.execute(context, queue)

    # Calling execute() directly (bypassing DefaultRequestHandler's
    # consumer, which is what normally closes the queue on a final event)
    # means the queue is never auto-closed here -- drain events until the
    # TaskStatusUpdateEvent marked final=True instead of waiting on
    # is_closed(). Events: task creation, the "working" status, then the
    # terminal "failed" status.
    final_status = None
    for _ in range(10):
        event = await queue.dequeue_event()
        queue.task_done()
        if hasattr(event, "final") and event.final:
            final_status = event
            break

    assert final_status is not None
    assert final_status.status.state == TaskState.failed


async def test_cancel_is_unsupported():

    executor = CreditLookupAgentExecutor(build_fake_credit_lookup_subgraph())
    context = RequestContext(request=_send_params("cancel me"))
    queue = EventQueue()

    try:
        await executor.cancel(context, queue)
        assert False, "expected cancel() to raise"
    except ServerError:
        pass
