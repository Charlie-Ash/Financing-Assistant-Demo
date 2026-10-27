# The routing schema must constrain both "tool" and "action" with enum:
# Gemini's structured output ignores pydantic's "const" (it returned
# action="respond" in testing) but respects enum, while vLLM honours both.
from schemas.tool_call import ToolCall
from schemas.tool_schema import build_tool_schema


def test_tool_schema_constrains_tool_and_action_with_enum():

    schema = build_tool_schema(["default", "rag", "credit_lookup"])

    assert schema["properties"]["tool"]["enum"] == ["default", "rag", "credit_lookup"]
    assert schema["properties"]["action"]["enum"] == ["run"]
    assert schema["properties"]["action"]["const"] == "run"


def test_schema_conformant_output_still_validates():

    call = ToolCall.model_validate_json('{"tool": "rag", "action": "run", "args": {}}')

    assert call.tool == "rag"
