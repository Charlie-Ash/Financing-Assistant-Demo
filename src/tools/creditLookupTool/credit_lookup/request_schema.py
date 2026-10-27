# Structured-output schema for the Credit Lookup agent's one extraction call
# (see graph.py's extract_request): the LLM only works out WHO is being asked
# about and WHAT aspect -- the lookup itself is done by fixed queries in
# datasource.py. Every field is a plain required string (empty when absent)
# rather than an optional/nullable one, and the focus choices come out as a
# JSON-schema "enum" -- both forms Gemini's structured output enforces (it
# ignores "const", see schemas/tool_schema.py).
from typing import Literal

from pydantic import BaseModel


class CreditLookupRequest(BaseModel):

    # Exactly as the user (or earlier conversation) names the customer; "" if none.
    customer_name: str

    # A customer number such as "C0003"; "" if none.
    customer_id: str

    # Which aspect the question is about -- steers the summary, not the data shown.
    focus: Literal["overview", "contracts", "payments"]


CREDIT_LOOKUP_REQUEST_SCHEMA = CreditLookupRequest.model_json_schema()
