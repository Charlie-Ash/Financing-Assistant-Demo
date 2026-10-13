# Structured-output schema for the Actions agent's own content-generation
# call (see graph.py's generate_content). Guided decoding (the shared vLLM
# server's json_schema response_format) forces schema-conformant JSON, the
# same mechanism schemas/tool_schema.py's schema uses for the orchestrator's
# routing call.
from pydantic import BaseModel


class NoteContent(BaseModel):

    content: str
    file_name: str


NOTE_SCHEMA = NoteContent.model_json_schema()
