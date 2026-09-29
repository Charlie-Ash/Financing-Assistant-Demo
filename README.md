# A2ASystem

A locally-hosted, multi-agent knowledge assistant built on **Google's Agent-to-Agent (A2A) protocol** and **LangGraph**, running entirely on a single consumer GPU via **vLLM**. It is being adapted into a **Traditional Chinese (Taiwan)** sales and financing assistant.

## About

A2ASystem is an orchestrator ("Secretary") that routes each user message to the right specialist agent using vLLM guided/structured decoding, rather than a hand-parsed prompt:

- **RAG Agent** — answers questions from a company's own documents (PDF ingestion → vector search → grounded answer).
- **Actions Agent** — turns a request into a saved note on disk.

Each agent is its own long-running process, and the orchestrator talks to them over the network using the A2A protocol rather than calling them as local functions. All of them generate text through **one shared vLLM server** (see [Architecture](#architecture)).

This repository is also the codebase behind an ongoing case-study research paper (submitted to **ICS2026**) on incrementally promoting a local multi-agent assistant onto a real networked agent protocol on constrained, single-GPU consumer hardware, and measuring what that promotion actually costs in latency, routing accuracy, and VRAM. See [Project evolution & branches](#project-evolution--branches) below.

## Architecture

One vLLM server process holds the GPU's LLM, and three independent agent processes call it over its OpenAI-compatible HTTP API:

| Process | Role | Entry point | Default address |
|---|---|---|---|
| **LLM server** | The only process that loads an LLM: one Chinese-capable FP8 model, served as `hotai-llm` | `src/llm/serve.py` | `http://0.0.0.0:8000/v1` |
| **Orchestrator** | User-facing REPL; A2A **client**; decides which agent to call and composes the final reply | `src/main.py` | (local, stdin/stdout) |
| **RAG Agent** | A2A **server**; answers from ingested documents | `src/tools/ragTool/a2a/a2a_server.py` | `http://0.0.0.0:8001` |
| **Actions Agent** | A2A **server**; writes notes to disk | `src/tools/actionsTool/a2a/a2a_server.py` | `http://0.0.0.0:8002` |

Internally, every agent is its own compiled **LangGraph** graph:

- **Orchestrator** — `decide_tool` (guided-decoding tool routing) → route to the local `default` tool or a discovered remote agent → `generate_response` → `update_memory`.
- **RAG Agent** — `extract_query` → `retrieve` (Qdrant vector search for 20 candidates, then `bge-reranker-v2-m3` keeps the best 4) → `generate_answer`.
- **Actions Agent** — `extract_request` → `generate_content` (guided-decoding note content + filename) → `save_note`.

At startup, the orchestrator discovers each configured remote agent's `AgentCard` over HTTP; an agent that's unreachable is simply excluded from routing for that session rather than causing a crash. Since the Actions agent needs conversation context (e.g. "note down your previous answer") but lives in a different process, the orchestrator attaches a slice of recent chat history to the outgoing A2A message's metadata, and the remote agent reconstructs it on its side.

### One shared LLM server

The agents never see the Hugging Face model ID. They only request the served name `hotai-llm`, so swapping models means restarting the server with a different profile from [src/llm/profiles.py](src/llm/profiles.py); no agent needs a change or a restart:

| Profile | Model | Weights | Role |
|---|---|---|---|
| `qwen3-14b-fp8` (default) | `Qwen/Qwen3-14B-FP8` | ~15.5 GiB | Strongest routing / structured output that still fits |
| `qwen3-8b-fp8` | `Qwen/Qwen3-8B-FP8` | ~8.8 GiB | Lighter fallback |
| `llama3-taiwan-8b-fp8` | `yentinglin/Llama-3-Taiwan-8B-Instruct`, FP8 at load | ~8.5 GiB | Taiwan-tuned challenger |

Every prompt asks for Traditional Chinese, and every output passes through an OpenCC `s2twp` safety net, which converts Simplified characters and mainland vocabulary to Taiwan usage.

This replaces the earlier design, where each agent loaded its own vLLM engine. That design measured 4–5 GiB short of fitting all three agents on one RTX 5090, because every process paid for its own weights, KV cache and roughly 1 GiB of fixed CUDA overhead.

**VRAM budget** (RTX 5090, 31.85 GiB, default profile):

| Item | GiB |
|---|---|
| Qwen3-14B-FP8 weights | ~15.5 |
| CUDA graphs + activations | ~1.5 |
| KV cache (fp8, ~95k tokens) | ~7.5 |
| **Server fraction, `LLM_GPU_MEMORY_UTILIZATION=0.77`** (sum of the rows above) | **~24.5** |
| Fixed per-process overhead outside the fraction | ~1.3 |
| RAG process: bge-m3 + reranker in fp16 on CUDA | ~2.7 |
| Desktop | ~0.2 |
| **Total / margin** | **~28.7 / +3.1** |

The orchestrator, the Actions agent and any future agent add about 0 GiB. If the margin is tight, set `RAG_EMBED_DEVICE=cpu` (frees 2.7 GiB) or serve `qwen3-8b-fp8` (frees about 6.7 GiB). These figures are estimates from earlier measurements; confirm them with `nvidia-smi` on the box.

### State and memory

Conversation state is checkpointed across turns with LangGraph's `AsyncSqliteSaver` (`data/checkpoints/checkpoints.db`), so the orchestrator can restart without losing the current conversation. A separate, LLM-summarized memory log (`data/chat_log/chat_log.md`) is also kept per session.

## Project structure

```
A2ASystem/
├── README.md
├── pytest.ini                    # pythonpath=src, asyncio_mode=auto
├── requirements.txt
├── evals/                        # model-selection evals (run on the GPU box, see Choosing the model)
│   ├── routing_zh.jsonl          # Experiment 2 routing questions, in Traditional Chinese
│   ├── run_routing_eval.py       # routing accuracy + Simplified-leakage per served model
│   ├── run_rag_qa.py             # collects RAG answers into a CSV for rubric scoring
│   └── rag_qa_zh.example.jsonl   # template for the company-specific RAG QA set
├── data/                         # runtime data (mostly generated at run time)
│   └── Test_PDF.pdf              # sample document for RAG ingestion
│   # created at runtime: chat_log/, checkpoints/, notes/
├── src/
│   ├── main.py                   # CLI entry point (Orchestrator REPL)
│   ├── chat_history.py           # shared LangChain-messages <-> wire-format helpers
│   ├── llm/                      # the shared LLM layer
│   │   ├── serve.py              # starts the one vLLM server (python -m llm.serve)
│   │   ├── profiles.py           # model profiles (Hugging Face ID, quantization, context length)
│   │   ├── settings.py           # server settings + the LLM-client settings every agent inherits
│   │   ├── client.py             # ChatLLM: what every agent calls to generate text
│   │   ├── sampling.py           # SamplingConfig (per-call generation settings)
│   │   ├── text_norm.py          # OpenCC Simplified -> Traditional safety net
│   │   └── .env.example          # copy to .env in this folder to override defaults (see Configuration)
│   ├── orchestrator/              # the Orchestrator agent (A2A client)
│   │   ├── orchestrator.py       # session/thread lifecycle, run_orchestrator()
│   │   ├── config.py             # REMOTE_AGENTS list, checkpoint DB path, LLM server address
│   │   ├── .env.example          # copy to .env in this folder to override defaults (see Configuration)
│   │   ├── memory_manager.py     # chat_log.md + thread_id.txt file I/O
│   │   ├── agents/
│   │   │   ├── remote_agent.py   # A2A client: discover + call remote agents
│   │   │   └── tool_router.py    # local "default" tool + discovered remote agents
│   │   └── pipeline/
│   │       ├── graph.py          # orchestrator's LangGraph StateGraph
│   │       ├── state.py          # OrchestratorState
│   │       ├── llm.py            # OrchestratorLLM (routing, reply, memory calls via ChatLLM)
│   │       └── prompts.py        # prompt builders
│   ├── schemas/
│   │   ├── tool_call.py          # ToolCall pydantic model
│   │   └── tool_schema.py        # guided-decoding schema builder
│   └── tools/
│       ├── base.py               # Tool protocol + ToolResult contract
│       ├── gpu_utils.py          # VRAM pre-flight check (run by llm/serve.py)
│       ├── defaultTool/          # trivial in-process fallback tool
│       ├── ragTool/              # standalone RAG A2A agent (own process)
│       │   ├── config.py
│       │   ├── .env.example      # copy to .env in this folder to override defaults (see Configuration)
│       │   ├── rag/              # dataIngestion, userQuery, models (embedding + reranker), graph, state, tool_class
│       │   └── a2a/              # a2a_server.py, agent_executor.py
│       └── actionsTool/          # standalone Actions A2A agent (own process)
│           ├── config.py
│           ├── .env.example      # copy to .env in this folder to override defaults (see Configuration)
│           ├── actions/          # noteWriter, note_schema, graph, state, tool_class
│           └── a2a/              # a2a_server.py, agent_executor.py
└── tests/
    ├── fake_dependencies.py      # GPU-free, network-free fakes for testing
    ├── test_*.py                 # A2A servers, executors, graph structure/flow, checkpointer,
    │                             # LLM client, settings validation, real agent subgraphs
    └── graph_visualization/      # standalone scripts (not pytest) to render graphs to PNG
```

## Project evolution & branches

This repository documents three incremental phases of the same system, each a deliberate stepping stone toward the next — the subject of the accompanying ICS2026 paper.

| Phase | Branch | Summary |
|---|---|---|
| **System I** | [`guided-decoding-system`](../../tree/guided-decoding-system) | The original hand-rolled, single-process orchestrator: vLLM guided/structured decoding drives tool routing directly, with a flat markdown file as memory. |
| **System II** | [`lang-graph-intergration-system`](../../tree/lang-graph-intergration-system) | The same control flow re-expressed as an explicit LangGraph `StateGraph` (supervisor pattern). RAG is promoted to a compiled subgraph and a checkpointer adds real graph-state chat history — still a single process. |
| **System III** | `main` (this branch) | RAG and a new Actions tool are promoted out-of-process into standalone agents speaking the A2A protocol. The orchestrator becomes a genuine A2A client with cross-process memory and persistent checkpointing. |

Each phase keeps the same underlying scenario (route a user request, answer from documents, or save a note) so the cost of each architectural step — latency, routing accuracy, GPU memory — can be measured and compared directly.

## Environment & requirements

- **Python**: developed against Python 3.14.
- **GPU**: an NVIDIA CUDA-capable GPU is required for the LLM server only; the agents need no GPU to generate text (the RAG agent can optionally run its embedding and reranker models on CUDA). Developed on a single RTX 5090 (~32 GiB VRAM). `LLM_GPU_MEMORY_UTILIZATION` (see [Configuration](#configuration)) sets how much of the card the server reserves, and `src/tools/gpu_utils.py` runs a pre-flight check before it loads. All agents can run at the same time; see the VRAM budget under [Architecture](#architecture).
- **Models used by default**: `Qwen/Qwen3-14B-FP8` for every agent (via the shared server), `BAAI/bge-m3` embeddings and `BAAI/bge-reranker-v2-m3` reranking for RAG.
- **vLLM on Windows**: vLLM won't build on native Windows. It is only needed on the Linux machine that runs `python -m llm.serve`; comment it out of `requirements.txt` elsewhere, since the agents and tests don't import it.
- **Key dependencies** (see `requirements.txt`): `vllm` (server only), `openai` (agents' client for the server), `opencc-python-reimplemented`, `langgraph` + `langgraph-checkpoint-sqlite`, `langchain-core`, `a2a-sdk[http-server]`, `uvicorn`, `llama-index` + `qdrant-client` (RAG), `pymupdf`, `sentence-transformers`, `pydantic-settings` (per-agent config, see [Configuration](#configuration)), `pytest` + `pytest-asyncio`.

## Setup

```bash
git clone https://github.com/Charlie-Ash/Financing-Assistant-Demo.git
cd Financing-Assistant-Demo
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

# CUDA-enabled torch is installed separately (see requirements.txt):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Everything above gets the code and dependencies installed with sensible defaults already built in — you don't strictly need to touch any config to run the project. See [Configuration](#configuration) below if you want to change a model, a port, a VRAM budget, or anything else.

## Configuration

The LLM server and each of the three agents (Orchestrator, RAG, Actions) read their own settings independently, in this order of priority: **a real OS environment variable** you've exported, then **that agent's own `.env` file**, then **the built-in default** baked into its `config.py`. You never have to set anything for the project to run — the built-in defaults are what's been used throughout development — but if you want to change a model, a port, or a GPU memory budget, a `.env` file is the easiest way.

**How to set one up:** each agent ships a `.env.example` right next to its `config.py`, listing every setting it accepts with its default already filled in (just commented out). Copy it to `.env` **in that same folder**, then uncomment and edit only the lines you actually want to change:

```bash
# do this for whichever agent(s) you want to customize — none are required
cp src/llm/.env.example src/llm/.env
cp src/orchestrator/.env.example src/orchestrator/.env
cp src/tools/ragTool/.env.example src/tools/ragTool/.env
cp src/tools/actionsTool/.env.example src/tools/actionsTool/.env
```

A few things worth knowing:
- **Each agent has its own `.env`, not one shared file.** This matches how they actually run — three independent processes — so RAG's `.env` has no effect on the orchestrator or Actions, and vice versa.
- **`.env` files are never committed.** They're covered by `.gitignore`; only the `.env.example` templates are checked in. If you put a real API key or path in a `.env`, it stays local to your machine.
- **A bad value fails immediately, not mysteriously later.** Every setting is type- and range-checked at process startup (e.g. a GPU memory fraction must be between 0 and 1, a port must be a valid port number). If you typo a value, you'll get a clear error naming the exact setting and the exact problem the moment you try to run that agent — not a confusing crash from inside vLLM several seconds in.
- **A real environment variable always wins** over anything in `.env`, which always wins over the built-in default — so you can still do a one-off override (e.g. `RAG_A2A_PORT=8005 python -m tools.ragTool.a2a.a2a_server`) without touching any file.

**Every available setting**, grouped by which agent's `.env` it belongs in:

| Agent | Variable | Default | Purpose |
|---|---|---|---|
| Orchestrator | `REMOTE_AGENTS` | `rag@http://localhost:8001,actions@http://localhost:8002` | Which A2A agents the orchestrator discovers, as `name@url` pairs |
| Orchestrator | `REMOTE_AGENTS_VERBATIM` | `rag` | Which agents' successful results are relayed to the user as-is instead of composed into a reply |
| Orchestrator | `CHECKPOINT_DB_PATH` | `data/checkpoints/checkpoints.db` | Where the persistent LangGraph checkpointer writes conversation state |
| Orchestrator | `REMOTE_AGENT_TIMEOUT_SECONDS` | `180` | How long the orchestrator waits for a remote agent's A2A response before giving up |
| LLM server | `LLM_PROFILE` | `qwen3-14b-fp8` | Which model to serve (a key in `src/llm/profiles.py`) |
| LLM server | `LLM_GPU_MEMORY_UTILIZATION` | `0.77` | Fraction of VRAM the server reserves for weights, CUDA graphs and KV cache |
| LLM server | `LLM_MAX_MODEL_LEN` | (profile's own) | Overrides the profile's context window |
| LLM server | `LLM_HOST` / `LLM_PORT` | `0.0.0.0` / `8000` | Server bind address |
| LLM server + every agent | `LLM_SERVED_NAME` | `hotai-llm` | Name the model is served and requested under; must match on both sides |
| Every agent | `LLM_BASE_URL` | `http://localhost:8000/v1` | Where the agent reaches the LLM server |
| Every agent | `LLM_TIMEOUT_SECONDS` | `120` | Per-request timeout for LLM calls |
| Every agent | `LLM_STARTUP_WAIT_SECONDS` | `300` | How long the agent waits at startup for the server to finish loading |
| RAG | `RAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model used to index and search documents |
| RAG | `RAG_COLLECTION_NAME` | `my_docs` | Base Qdrant collection name; the embedding model is appended (`my_docs__bge-m3`), so switching models rebuilds the index |
| RAG | `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder that re-scores retrieved chunks |
| RAG | `RAG_EMBED_DEVICE` | `cuda` | Where the embedding and reranker run (`cuda` or `cpu`) |
| RAG | `RAG_RETRIEVE_TOP_K` / `RAG_RERANK_TOP_N` | `20` / `4` | Vector-search candidates, and how many survive reranking into the prompt |
| RAG | `RAG_A2A_HOST` / `RAG_A2A_PORT` | `0.0.0.0` / `8001` | RAG agent server bind address |
| Actions | `ACTIONS_A2A_HOST` / `ACTIONS_A2A_PORT` | `0.0.0.0` / `8002` | Actions agent server bind address |

## Running the project

Start the LLM server, then the two agent servers, then the orchestrator. Each agent waits for the LLM server to finish loading (up to `LLM_STARTUP_WAIT_SECONDS`), so the agents can be started straight away. Agent discovery happens once, at orchestrator startup, and an agent that isn't reachable yet is simply skipped for that session.

```bash
# Terminal 0 — shared LLM server (http://0.0.0.0:8000/v1)
cd src
python -m llm.serve                    # or: python -m llm.serve --profile qwen3-8b-fp8

# Terminal 1 — RAG agent (http://0.0.0.0:8001)
cd src
python -m tools.ragTool.a2a.a2a_server

# Terminal 2 — Actions agent (http://0.0.0.0:8002)
cd src
python -m tools.actionsTool.a2a.a2a_server

# Terminal 3 — Orchestrator (REPL)
cd src
python main.py
```

Type `bye` in the orchestrator REPL to exit (you'll be asked whether to save a summarized memory of the session).

Want to change a model, a port, or a GPU memory budget first? See [Configuration](#configuration) above.

## Choosing the model (evals)

Pick the default model by measurement, not by spec sheet. With the LLM server running a given profile:

```bash
python evals/run_routing_eval.py --label qwen3-14b-fp8    # from the repo root
```

It reports three things:
- routing accuracy on the Experiment 2 question set, translated to Traditional Chinese, overall and per category;
- schema-valid rate;
- **Simplified leakage**: how much the OpenCC safety net had to fix for that model's replies.

Restart the server with each profile (`python -m llm.serve --profile ...`) and rerun. The agents don't need restarting.

`evals/run_rag_qa.py` sends a QA set to the running RAG agent and writes a CSV to score against the rubric (Faithfulness 0–2 + Relevance 0–2). Before running it, copy `evals/rag_qa_zh.example.jsonl` to `evals/rag_qa_zh.jsonl` and fill in 20–30 questions grounded in the ingested documents. Results go to `evals/results/`, which is git-ignored.

## Testing

```bash
pytest
```

Tests run without a GPU, an LLM server, or real network access, using fakes (`tests/fake_dependencies.py`) and in-process ASGI transports for the A2A servers.

Two standalone scripts (not part of the pytest suite — run directly) render each agent's compiled LangGraph to a PNG for inspection:

```bash
python tests/graph_visualization/orchestrator_graph_viz.py
python tests/graph_visualization/rag_graph_viz.py
```
