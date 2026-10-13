# Builds the Actions tool's own SUBGRAPH. Mirrors orchestrator/pipeline/graph.py's
# build_graph(llm, tool_router) and ragTool/rag/graph.py's
# build_rag_subgraph(): takes the already-built llm/sampling as parameters
# instead of constructing them itself, so this module's wiring can be tested
# without an LLM server (see tests/fake_dependencies.py).
from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, START, END

from chat_history import history_to_chat_messages
from llm.text_norm import ZH_TW_REPLY_INSTRUCTION
from tools.actionsTool.actions.note_schema import NOTE_SCHEMA, NoteContent
from tools.actionsTool.actions.noteWriter import save_note
from tools.actionsTool.actions.state import ActionsSubgraphState

# Imported only for "ChatLLM"/"SamplingConfig" to be resolved as types for
# string-quoted annotations on build_actions_subgraph.
if TYPE_CHECKING:
    from llm.client import ChatLLM
    from llm.sampling import SamplingConfig

# The file_name slug stays ASCII on purpose: it becomes a file on disk
# (see noteWriter.py's sanitizer), while the note content itself is Chinese.
SYS_PROMPT = (
    "你是一位筆記助理。請根據下方使用者的要求，撰寫應該儲存的筆記內容(content)，"
    "並為筆記選一個簡短、可作為檔名的英文代稱(file_name)：全部小寫英文、"
    "單字之間以底線連接，不含副檔名、空白或標點符號。"
    "若要求提到先前對話中的內容（例如「把你剛才的回答記下來」），請參考對話紀錄來判斷。"
    + ZH_TW_REPLY_INSTRUCTION
)


def build_actions_subgraph(llm: "ChatLLM", sampling: "SamplingConfig"):
    """Assemble and compile the Actions tool's subgraph: extract_request ->
    generate_content -> save_note, with an empty-request short-circuit
    straight to END and a generation-failure short-circuit that skips
    save_note.

    Compiled with no explicit checkpointer, same reasoning as RAG's
    build_rag_subgraph(): inherits the parent orchestrator graph's
    MemorySaver when run as a node there. Unlike RAG, this subgraph declares
    "messages" as a shared state key (see state.py) specifically so it
    inherits that checkpointed history too, not just the checkpointer
    object -- generate_content needs it to resolve references to earlier
    turns.
    """

    graph = StateGraph(ActionsSubgraphState)

    # Node: pull the request out of this turn's tool_call.args. If it's
    # missing, also set the same error tool_result the rest of the graph
    # would otherwise have had to special-case downstream -- route_after_extract
    # below sends this straight to END instead of attempting generation.
    def extract_request(state: ActionsSubgraphState) -> dict:

        request = state["tool_call"].args.get("request", "")

        if not request:
            return {
                "request": request,
                "tool_result": {"output": "Actions tool error: no request was provided.", "relay_verbatim": False},
            }

        return {"request": request}

    def route_after_extract(state: ActionsSubgraphState) -> str:

        return "has_request" if state["request"] else "empty_request"

    # Node: this agent's own LLM writes the note's content and file_name
    # from the raw request, using recent chat history (turns before this
    # one -- state["messages"] already includes this turn's just-appended
    # HumanMessage, same convention orchestrator/pipeline/graph.py's
    # generate_response uses) so it can resolve references like "note down
    # your previous answer". Via this package's standalone A2A server,
    # "messages" is reconstructed from history the orchestrator explicitly
    # attached to the incoming A2A message (see a2a/agent_executor.py) --
    # this node itself doesn't need to know or care where "messages" came
    # from, in-process node or reconstructed-from-the-wire alike.
    def generate_content(state: ActionsSubgraphState) -> dict:

        history_messages = state.get("messages", [])[:-1]

        messages = [
            {"role": "system", "content": SYS_PROMPT},
            *history_to_chat_messages(history_messages),
            {"role": "user", "content": state["request"]},
        ]

        try:
            # json_schema forces output to match NoteContent's schema
            note = NoteContent.model_validate_json(llm.chat(messages, sampling, json_schema=NOTE_SCHEMA))
        except Exception as e:
            return {"tool_result": {"output": f"Actions tool error: generation failed ({e})", "relay_verbatim": False}}

        return {"generated_content": note.content, "generated_file_name": note.file_name}

    # Routing function: generate_content sets tool_result itself on failure
    # instead of raising, so route on whether generation actually produced
    # content rather than on an exception.
    def route_after_generate(state: ActionsSubgraphState) -> str:

        return "success" if state.get("generated_content") else "failed"

    # Node: write the generated content to disk via the shared noteWriter
    # (moved from the old notesTool -- same sanitization/disambiguation logic).
    def save_note_node(state: ActionsSubgraphState) -> dict:

        try:
            saved_path = save_note(state["generated_content"], state["generated_file_name"])
        except Exception as e:
            return {"tool_result": {"output": f"Actions tool error: failed to save note ({e})", "relay_verbatim": False}}

        return {"tool_result": {"output": f"Note saved to {saved_path}.", "relay_verbatim": False}}

    graph.add_node("extract_request", extract_request)
    graph.add_node("generate_content", generate_content)
    graph.add_node("save_note", save_note_node)

    graph.add_edge(START, "extract_request")
    graph.add_conditional_edges(
        "extract_request",
        route_after_extract,
        {
            "has_request": "generate_content",
            "empty_request": END,
        },
    )
    graph.add_conditional_edges(
        "generate_content",
        route_after_generate,
        {
            "success": "save_note",
            "failed": END,
        },
    )
    graph.add_edge("save_note", END)

    return graph.compile()
