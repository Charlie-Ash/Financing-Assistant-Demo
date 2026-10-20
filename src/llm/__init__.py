# Shared LLM layer. Every agent (orchestrator, RAG, Credit Lookup, and any future
# Phase 4 agent) talks to ONE vLLM server process over its OpenAI-compatible
# HTTP API, instead of each loading its own in-process vLLM engine. See
# llm/serve.py (starts that server) and llm/client.py (what agents call).
