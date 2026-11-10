# Quick live check of the Gemini backend: API key, model id, one Traditional
# Chinese chat, and three real routing decisions (one per tool) through the
# orchestrator's own prompt and schema. 5 small requests -- free-tier safe.
#
#   # from the repo root; GEMINI_API_KEY in src/orchestrator/.env:
#   python evals/check_gemini.py
#   python evals/check_gemini.py --model gemini-3.5-flash --effort none
#
# Model and reasoning effort default to src/orchestrator/.env
# (LLM_GEMINI_MODEL / LLM_REASONING_EFFORT); LLM_PROVIDER there doesn't
# need to be set yet. Run it as a file, not piped into `python -`: Windows
# PowerShell 5.1 pipes text as ASCII and turns Chinese into "?".
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm.client import ChatLLM
from llm.sampling import SamplingConfig
from orchestrator.agents.tool_router import DEFAULT_TOOL_DESCRIPTION, describe_agent_card
from orchestrator.config import OrchestratorSettings
from orchestrator.pipeline.llm import OrchestratorLLM
from schemas.tool_schema import build_tool_schema
from tools.creditLookupTool.a2a.a2a_server import build_agent_card as build_credit_lookup_agent_card
from tools.loanCalcTool.a2a.a2a_server import build_agent_card as build_loan_calc_agent_card
from tools.ragTool.a2a.a2a_server import build_agent_card as build_rag_agent_card

ROUTING_PROMPTS = [
    ("你好，今天可以幫我做什麼？", "default"),
    ("提前清償要付違約金嗎？", "rag"),
    ("王大明有沒有逾期紀錄？", "credit_lookup"),
    ("車價120萬，頭期兩成，分60期，月付多少？", "loan_calc"),
]


def main():

    parser = argparse.ArgumentParser(description="Live check of the Gemini LLM backend.")
    parser.add_argument("--model", help="overrides LLM_GEMINI_MODEL")
    parser.add_argument("--effort", choices=["none", "minimal", "low", "medium", "high"], help="overrides LLM_REASONING_EFFORT")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    args = parser.parse_args()

    overrides = {"llm_provider": "gemini"}
    if args.model:
        overrides["llm_gemini_model"] = args.model
    if args.effort:
        overrides["llm_reasoning_effort"] = args.effort

    chat_llm = ChatLLM.from_settings(OrchestratorSettings(**overrides))
    print(f"Model: {chat_llm.served_name} | reasoning_effort: {chat_llm.reasoning_effort}")

    chat_llm.wait_until_ready(max_wait_seconds=0)
    print("API key and model id: OK")

    start = time.perf_counter()
    reply = chat_llm.chat(
        [{"role": "user", "content": "請用繁體中文一句話介紹車貸。"}],
        SamplingConfig(temperature=0.4, top_p=0.95, max_tokens=400),
    )
    print(f"\nChat ({time.perf_counter() - start:.1f}s): {reply}")

    tool_descriptions = [
        DEFAULT_TOOL_DESCRIPTION,
        describe_agent_card("rag", build_rag_agent_card("localhost", 0)),
        describe_agent_card("credit_lookup", build_credit_lookup_agent_card("localhost", 0)),
        describe_agent_card("loan_calc", build_loan_calc_agent_card("localhost", 0)),
    ]
    llm = OrchestratorLLM(build_tool_schema([tool["name"] for tool in tool_descriptions]), chat_llm=chat_llm)

    failures = 0
    print("\nRouting:")
    for question, expected in ROUTING_PROMPTS:

        time.sleep(args.delay)
        try:
            call = llm.tool_decision(question, [], tool_descriptions)
            ok = call.tool == expected
            print(f"  {'OK      ' if ok else 'MISROUTE'} {question} -> {call.tool} (expected {expected})")
        except Exception as e:
            ok = False
            print(f"  FAILED   {question} -> {type(e).__name__}: {e}")
        failures += not ok

    print(f"\n{len(ROUTING_PROMPTS) - failures}/{len(ROUTING_PROMPTS)} routing checks passed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
