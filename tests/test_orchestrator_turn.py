# Orchestrator.run_turn -- what the web interface calls -- returns the reply
# and the tool that answered; run_orchestrator (the CLI's call) still returns
# just the reply. Built on the same fakes as the graph tests, no GPU/LLM.
from contextlib import AsyncExitStack

from schemas.tool_call import ToolCall
from orchestrator.orchestrator import Orchestrator
from orchestrator.pipeline.graph import build_graph

from fake_dependencies import FakeOrchestratorLLM, FakeToolRouter


async def _orchestrator(tool_name: str) -> Orchestrator:

    llm = FakeOrchestratorLLM(next_tool_call=ToolCall(tool=tool_name, action="run", args={}))
    tool_router = await FakeToolRouter.create()
    return Orchestrator(
        tool_router, llm, build_graph(llm, tool_router), "turn-thread", AsyncExitStack(),
        {"user_id": "wang01", "role": "sales"},
    )


async def test_run_turn_reports_which_agent_answered():

    orchestrator = await _orchestrator("credit_lookup")

    turn = await orchestrator.run_turn("王大明有沒有逾期？")

    assert turn == {"reply": "canned reply about credit_lookup-output", "tool": "credit_lookup"}


async def test_run_orchestrator_still_returns_just_the_reply():

    orchestrator = await _orchestrator("default")

    assert await orchestrator.run_orchestrator("你好") == "canned reply about default-output"


async def test_connected_agents_lists_the_discovered_remote_agents():

    orchestrator = await _orchestrator("default")

    assert orchestrator.connected_agents() == ["rag", "credit_lookup"]
