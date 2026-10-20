# Regulation agent operations: embedded Qdrant index + reranker in this process, text
# generation through the shared LLM server (src/llm/).
import qdrant_client

from llm.client import ChatLLM
from llm.sampling import SamplingConfig
from tools.ragTool import config
from tools.ragTool.rag.dataIngestion import build_vector_index, load_vector_index
from tools.ragTool.rag.graph import build_rag_subgraph
from tools.ragTool.rag.models import load_reranker


class RAGTool():

    def __init__(self):

        print("RAG tool initialized.")

        # Check if vector DB already exists
        try:
            db_client = qdrant_client.QdrantClient(
                path=config.QDRANT_DB_PATH
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open Qdrant DB at {config.QDRANT_DB_PATH}: {e}") from e

        if not db_client.collection_exists(config.COLLECTION_NAME):  # Build another DB

            print(f"No vector database found for collection '{config.COLLECTION_NAME}'.")
            print("Building vector database...")

            db_client.close()
            self.index, self.db_client = build_vector_index()

            print("Embedding complete.")

        else:

            print("Vector database already exist.")
            print("Loading vector database...")

            db_client.close()
            self.index, self.db_client = load_vector_index()

            print("Loading completed.")

        # Cross-encoder reranker, loaded once and reused for every query.
        self.reranker = load_reranker()

        self.llm = ChatLLM.from_settings(config.settings)
        self.llm.wait_until_ready(config.settings.llm_startup_wait_seconds)

        self.sampling = SamplingConfig(
            temperature=0.2,  # low: legal answers should stay close to the sources
            top_p=0.95,  # top_p; nucleus sampling
            max_tokens=512,  # Max tokens outputted
            repetition_penalty=1.1,  # Penalty to apply if tokens continue repeating.
        )

        # RAGTool's per-turn logic is a compiled LangGraph subgraph (see
        # graph.py in this package), invoked by this package's own standalone
        # A2A server (a2a/a2a_server.py) -- the orchestrator reaches this
        # agent only over the network, via orchestrator/agents/remote_agent.py.
        self.subgraph = build_rag_subgraph(self.index, self.llm, self.sampling, self.reranker)
