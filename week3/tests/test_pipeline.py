import json

import pytest
from pydantic import ValidationError

from pipeline import RAGResponse, parse_and_validate, save_response

VALID_JSON = json.dumps({
    "answer": "LoRA freezes the pretrained weights and injects trainable rank-decomposition matrices.",
    "source_context_used": "We propose Low-Rank Adaptation, or LoRA...",
})


def test_valid_response_validates():
    result = parse_and_validate(VALID_JSON)
    assert isinstance(result, RAGResponse)
    assert result.answer.startswith("LoRA freezes")
    assert result.source_context_used


def test_missing_required_field_fails():
    bad_json = json.dumps({"answer": "Some answer with no source field."})
    with pytest.raises(ValidationError):
        parse_and_validate(bad_json)


def test_wrong_type_fails():
    bad_json = json.dumps({
        "answer": 42,  # should be a string, not a number
        "source_context_used": "some context",
    })
    with pytest.raises(ValidationError):
        parse_and_validate(bad_json)


def test_extra_hallucinated_field_fails():
    bad_json = json.dumps({
        "answer": "Some answer.",
        "source_context_used": "some context",
        "confidence_score": 0.97,  # hallucinated field not in the schema
    })
    with pytest.raises(ValidationError):
        parse_and_validate(bad_json)


def test_malformed_json_fails():
    with pytest.raises(ValidationError):
        parse_and_validate("{answer: this is not valid JSON")


def test_empty_string_fails():
    with pytest.raises(ValidationError):
        parse_and_validate("")


def test_save_response_writes_valid_jsonl_line(tmp_path):
    result = parse_and_validate(VALID_JSON)
    log_path = tmp_path / "results.jsonl"

    save_response("What is LoRA?", "ollama", result, path=log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["query"] == "What is LoRA?"
    assert record["embedding_model"] == "ollama"
    assert record["answer"] == result.answer
    assert record["source_context_used"] == result.source_context_used
    assert "timestamp" in record


def test_save_response_appends_multiple_lines(tmp_path):
    result = parse_and_validate(VALID_JSON)
    log_path = tmp_path / "results.jsonl"

    save_response("Q1", "ollama", result, path=log_path)
    save_response("Q2", "gemini", result, path=log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["query"] == "Q1"
    assert json.loads(lines[1])["query"] == "Q2"
