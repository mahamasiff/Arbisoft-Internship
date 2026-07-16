import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

DEFAULT_SAVE_PATH = Path("outputs/rag_results.jsonl")


class RAGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    source_context_used: str


def parse_and_validate(raw_json: str) -> RAGResponse:
    """Parse the LLM's raw JSON string and validate it against RAGResponse.

    Raises pydantic.ValidationError if the string isn't valid JSON, or if it's
    valid JSON but doesn't match the schema (missing field, wrong type, or an
    extra/hallucinated field, since RAGResponse forbids unknown fields).
    """
    return RAGResponse.model_validate_json(raw_json)


def save_response(
    query: str,
    embedding_model: str,
    response: RAGResponse,
    path: Path = DEFAULT_SAVE_PATH,
) -> None:
    """Append a validated response as one line to a JSONL log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "embedding_model": embedding_model,
        "answer": response.answer,
        "source_context_used": response.source_context_used,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
