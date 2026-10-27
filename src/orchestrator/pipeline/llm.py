# Interacts first-hand with the language model
from llm.client import ChatLLM
from llm.sampling import SamplingConfig
from orchestrator import config
from orchestrator.pipeline.prompts import (
    build_tool_decision_prompt,
    build_response_prompt,
    build_mem_update_prompt,
    build_filename_prompt,
)
from orchestrator import memory_manager
from schemas.tool_call import ToolCall

class OrchestratorLLM():

    # tool_schema is built by schemas.tool_schema.build_tool_schema() from
    # whatever tool names ToolRouter actually discovered/configured at
    # startup -- passed in rather than imported as a module-level constant,
    # since that set isn't known until ToolRouter.discover() has run.
    # chat_llm is injectable (evals pass one with Traditional-Chinese
    # normalization off); by default it's built from orchestrator/config.py
    # and pointed at the shared LLM server (see src/llm/).
    def __init__(self, tool_schema: dict, chat_llm: ChatLLM | None = None):

        if chat_llm is None:
            chat_llm = ChatLLM.from_settings(config.settings)
            chat_llm.wait_until_ready(config.settings.llm_startup_wait_seconds)

        self.llm = chat_llm
        self.tool_schema = tool_schema

        self.decision_sampling = SamplingConfig(
            temperature=0,  # temperture: randomness
            top_p=1.0,  # top_p; nucleus sampling
            max_tokens=512,  # Max tokens outputted
            repetition_penalty=1.1,  # Penalty to apply if tokens continue repeating.
        )

        self.response_sampling = SamplingConfig(
            temperature=0.4,  # some randomness allowed; this stage writes free text, not JSON
            top_p=0.95,  # top_p; nucleus sampling
            max_tokens=512,  # Max tokens outputted
            repetition_penalty=1.1,  # Penalty to apply if tokens continue repeating.
        )

        self.mem_update_sampling = SamplingConfig(
            temperature=0.2,  # low randomness; this stage writes a short factual summary
            top_p=0.9,  # top_p; nucleus sampling
            max_tokens=180,  # memory notes are meant to stay short
            repetition_penalty=1.1,  # Penalty to apply if tokens continue repeating.
        )

        self.filename_sampling = SamplingConfig(
            temperature=0.0,  # deterministic; this stage outputs a short filename slug
            top_p=1.0,  # top_p; nucleus sampling
            max_tokens=24,  # a 2-5 word slug needs very few tokens
            repetition_penalty=1.1,  # Penalty to apply if tokens continue repeating.
        )

    # Phase 1 Orchestrotor LLM usage: Tool decision
    # tool_descriptions comes from ToolRouter.describe_tools_for_prompt() --
    # see build_tool_decision_prompt for the shape.
    def tool_decision(self, user_message, history_messages, tool_descriptions) -> ToolCall:

        messages = build_tool_decision_prompt(user_message, history_messages, tool_descriptions)

        # json_schema forces output to match ToolCall's schema, restricted to the tools actually discovered
        raw_text = self.llm.chat(messages, self.decision_sampling, json_schema=self.tool_schema)
        print("\nRAW MODEL OUTPUT:\n", raw_text)  # Maybe can remove this in the future

        # Guided decoding guarantees schema-conformant JSON, so parsing/validation collapses into this one call
        return ToolCall.model_validate_json(raw_text)

    # Phase 2 Orchestrotor LLM usage: Response Generation
    def generate_response(self, user_message, history_messages, tool_call: ToolCall, tool_result) -> str:

        # tool_result is a ToolResult (see tools/base.py); every tool always
        # returns one, so no None-handling is needed here anymore.
        # A verbatim result (e.g. the regulation agent's cited answer) is
        # already the reply: passing it through the LLM again could only
        # trim or reword it -- including its source list and disclaimer.
        if tool_result["relay_verbatim"]:
            return tool_result["output"]

        messages = build_response_prompt(user_message, history_messages, tool_call, tool_result)

        return self.llm.chat(messages, self.response_sampling).strip()

    # Phase 3 Orchestrotor LLM usage: Conversation memory update
    def orchestrator_mem_update(self, user_message, tool_call: ToolCall, tool_result, final_response) -> None:

        try:
            question_number = memory_manager.count_existing_entries() + 1

            messages = build_mem_update_prompt(user_message, tool_call, tool_result, final_response)

            note_text = self.llm.chat(messages, self.mem_update_sampling).strip() or "(no memory note generated for this turn)"

            memory_manager.append_entry(question_number, note_text)

        except Exception as e:
            # A memory-write hiccup should never take down an otherwise-successful turn.
            print(f"Warning: failed to update conversation memory ({e}).")

    # Phase 4 Orchestrotor LLM usage: Chat memory filename decision (used only at "bye" + "y")
    def decide_chat_memory_filename(self) -> str:

        memory_content = memory_manager.read_chat_log()

        try:
            messages = build_filename_prompt(memory_content)

            return self.llm.chat(messages, self.filename_sampling).strip()

        except Exception as e:
            # memory_manager's own sanitizer falls back to a timestamp-based name
            # for an empty slug, so failing here still results in a saved file.
            print(f"Warning: failed to generate a memory filename ({e}); using a default name.")
            return ""
