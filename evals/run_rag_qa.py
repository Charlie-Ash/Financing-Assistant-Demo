# RAG answer-quality run: sends every question in a QA set to the live RAG
# agent over A2A (exactly as the orchestrator would) and writes a CSV with
# blank score columns for grading against the project's rubric:
#   Faithfulness 0-2 (is every claim supported by the source document?)
#   Relevance    0-2 (does it actually answer the question?)
#   -> combined 0-4 per question.
#
#   # RAG server + LLM server running; from the repo root:
#   python evals/run_rag_qa.py --label qwen3-14b-fp8
#
# If the RAG agent uses Gemini (LLM_PROVIDER=gemini in its .env), add
# --delay to stay under the free tier's requests-per-minute limit.
#
# The QA set (evals/rag_qa_zh.jsonl): 20 questions grounded in the ingested
# laws and demo policies, each with a reference answer and its source
# article -- including one superseded-version check (qa-16) and one question
# the documents can't answer (qa-20), which should get "查無相關規定".
import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from orchestrator import config
from orchestrator.agents.remote_agent import call_remote_agent, discover_remote_agent

EVALS_DIR = Path(__file__).resolve().parent


async def run(label: str, qa_path: Path, agent_name: str, delay: float) -> Path:

    qa_items = [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    agent_config = next((cfg for cfg in config.REMOTE_AGENTS if cfg.name == agent_name), None)
    if agent_config is None:
        sys.exit(f"No agent named {agent_name!r} in REMOTE_AGENTS ({config.settings.remote_agents}).")

    agent = await discover_remote_agent(agent_config)
    if agent is None:
        sys.exit(f"RAG agent unreachable at {agent_config.url} -- start it first.")

    results_dir = EVALS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"rag_qa_{label}_{datetime.now():%Y%m%d_%H%M%S}.csv"

    # utf-8-sig so Excel opens the Chinese text correctly
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:

        writer = csv.writer(f)
        writer.writerow(["id", "question", "reference_answer", "source_doc", "answer", "faithfulness_0_2", "relevance_0_2"])

        for item in qa_items:
            result = await call_remote_agent(agent, item["question"])
            print(f"{item['id']}: {result['output'][:80]}")
            writer.writerow([
                item["id"], item["question"], item.get("reference_answer", ""), item.get("source_doc", ""),
                result["output"], "", "",
            ])
            await asyncio.sleep(delay)

    return out_path


def main():

    parser = argparse.ArgumentParser(description="Collect RAG answers for rubric scoring.")
    parser.add_argument("--label", required=True, help="name for this run, e.g. the served profile")
    parser.add_argument("--qa", type=Path, default=EVALS_DIR / "rag_qa_zh.jsonl")
    parser.add_argument("--agent", default="rag", help="routing name of the RAG agent in REMOTE_AGENTS")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to wait between questions (API rate limits)")
    args = parser.parse_args()

    out_path = asyncio.run(run(args.label, args.qa, args.agent, args.delay))
    print(f"\nAnswers written to {out_path} -- fill in the two score columns.")


if __name__ == "__main__":
    main()
