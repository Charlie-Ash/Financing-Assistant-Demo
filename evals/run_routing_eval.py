# Model-selection eval: routing accuracy + native Traditional Chinese quality
# of whatever model the orchestrator is configured to use.
#
#   # on the GPU box, from the repo root, with the LLM server running:
#   python evals/run_routing_eval.py --label qwen3-14b-fp8
#
# Repeat per candidate by restarting the server with another profile
# (`python -m llm.serve --profile qwen3-8b-fp8` from src/) -- nothing else
# needs restarting, and no RAG/Actions servers are needed at all: the
# routing prompt is built from their agent cards locally.
#
# With LLM_PROVIDER=gemini in src/orchestrator/.env no server is needed;
# pick the model with LLM_GEMINI_MODEL, and add --delay to stay under the
# free tier's requests-per-minute limit:
#   python evals/run_routing_eval.py --label gemini-3.5-flash-lite --delay 5
#
# Metrics:
#   - routing accuracy, overall and per category (evals/routing_zh.jsonl,
#     the Experiment 2 question set translated to Traditional Chinese)
#   - schema-valid rate (guided decoding should make this 100%)
#   - tool-decision latency
#   - Simplified leakage: the production response stage is run on the
#     clear-default questions with OpenCC normalization OFF, and every
#     character s2twp would have had to change is counted -- i.e. how much
#     the safety net is doing for this model.
import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm.client import ChatLLM
from llm.text_norm import cjk_char_count, simplified_leakage_chars
from orchestrator import config
from orchestrator.agents.tool_router import DEFAULT_TOOL_DESCRIPTION, describe_agent_card
from orchestrator.pipeline.llm import OrchestratorLLM
from schemas.tool_call import ToolCall
from schemas.tool_schema import build_tool_schema
from tools.actionsTool.a2a.a2a_server import build_agent_card as build_actions_agent_card
from tools.ragTool.a2a.a2a_server import build_agent_card as build_rag_agent_card

EVALS_DIR = Path(__file__).resolve().parent

# What DefaultTool.run() returns -- fed to the response stage for the
# leakage measurement, same as a real default-routed turn.
DEFAULT_TOOL_RESULT = {"output": "Default tool has completed.", "relay_verbatim": False}


def load_jsonl(path: Path) -> list[dict]:

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():

    parser = argparse.ArgumentParser(description="Routing accuracy + Traditional Chinese leakage eval.")
    parser.add_argument("--label", required=True, help="name for this run, e.g. the served profile")
    parser.add_argument("--questions", type=Path, default=EVALS_DIR / "routing_zh.jsonl")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to wait between LLM calls (API rate limits)")
    args = parser.parse_args()

    questions = load_jsonl(args.questions)

    tool_descriptions = [
        DEFAULT_TOOL_DESCRIPTION,
        describe_agent_card("rag", build_rag_agent_card("localhost", 0)),
        describe_agent_card("actions", build_actions_agent_card("localhost", 0)),
    ]
    tool_schema = build_tool_schema([tool["name"] for tool in tool_descriptions])

    chat_llm = ChatLLM.from_settings(config.settings, normalize_traditional=False)
    chat_llm.wait_until_ready(config.settings.llm_startup_wait_seconds)
    llm = OrchestratorLLM(tool_schema, chat_llm=chat_llm)

    routing_rows = []
    for q in questions:

        start = time.perf_counter()
        try:
            predicted, error = llm.tool_decision(q["question"], [], tool_descriptions).tool, None
        except Exception as e:
            predicted, error = None, str(e)
        latency = time.perf_counter() - start

        routing_rows.append({
            **q,
            "predicted": predicted,
            "correct": predicted == q["expected"],
            "schema_valid": error is None,
            "error": error,
            "latency_s": round(latency, 3),
        })
        time.sleep(args.delay)

    leakage_rows = []
    for q in questions:

        if q["expected"] != "default":
            continue

        default_call = ToolCall(tool="default", action="run", args={})
        response = llm.generate_response(q["question"], [], default_call, DEFAULT_TOOL_RESULT)
        leakage_rows.append({
            "id": q["id"],
            "response": response,
            "cjk_chars": cjk_char_count(response),
            "leaked_chars": simplified_leakage_chars(response),
        })
        time.sleep(args.delay)

    categories = sorted({row["category"] for row in routing_rows})
    latencies = [row["latency_s"] for row in routing_rows]
    total_cjk = sum(row["cjk_chars"] for row in leakage_rows)
    total_leaked = sum(row["leaked_chars"] for row in leakage_rows)

    summary = {
        "label": args.label,
        "provider": chat_llm.provider,
        "model": chat_llm.served_name,
        "routing_accuracy": sum(row["correct"] for row in routing_rows) / len(routing_rows),
        "accuracy_by_category": {
            category: (
                sum(row["correct"] for row in routing_rows if row["category"] == category)
                / sum(1 for row in routing_rows if row["category"] == category)
            )
            for category in categories
        },
        "schema_valid_rate": sum(row["schema_valid"] for row in routing_rows) / len(routing_rows),
        "decision_latency_s": {"mean": statistics.mean(latencies), "median": statistics.median(latencies)},
        "responses_with_leakage": sum(row["leaked_chars"] > 0 for row in leakage_rows) / len(leakage_rows),
        "leaked_char_rate": total_leaked / total_cjk if total_cjk else 0.0,
    }

    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"routing_{args.label}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "routing": routing_rows, "leakage": leakage_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    for row in routing_rows:
        if not row["correct"]:
            print(f"MISROUTE {row['id']}: expected {row['expected']}, got {row['predicted']} -- {row['question']}")
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
