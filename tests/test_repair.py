"""Contrato do laço de reparo: orçamento, erro reenviado e fato errado sem nova tentativa."""

import json

import pytest

from structured_extraction_study.extractors.llm import LLMExtractor
from structured_extraction_study.extractors.protocol import Extractor
from structured_extraction_study.llm.client import ScriptedLLMClient
from structured_extraction_study.metrics import match_record
from structured_extraction_study.schemas.order import evaluate_order_dict

VALID = {
    "customer_name": "Ana Souza",
    "email": "ana.souza@example.com",
    "phone": "11988880001",
    "document_id": "12345678909",
    "order_date": "2026-03-15",
    "amount_brl": "1250.00",
    "city": "Campinas",
    "item": "Teclado mecânico",
}


def test_valid_json_stops_on_the_first_attempt() -> None:
    client = ScriptedLLMClient([json.dumps(VALID), "unused"])
    result = LLMExtractor(client, max_attempts=3).extract("Cliente: Ana Souza")
    assert result.schema_valid
    assert result.json_valid
    assert result.attempts == 1
    assert result.errors == ()
    assert result.data == VALID
    assert client.outputs_remaining == 1
    assert result.latency_ms >= 0
    assert "amount_brl" in client.calls[0][0]["content"]
    assert client.calls[0][1]["content"].startswith("Text:\n")


def test_fenced_json_is_accepted_without_repair() -> None:
    fenced = "```json\n" + json.dumps(VALID, ensure_ascii=False) + "\n```"
    client = ScriptedLLMClient([fenced])
    result = LLMExtractor(client, max_attempts=2).extract("texto")
    assert result.schema_valid
    assert result.attempts == 1
    assert result.data == VALID


def test_malformed_json_is_reasked_with_the_decode_error() -> None:
    bad = "{"
    client = ScriptedLLMClient([bad, json.dumps(VALID), "unused"])
    result = LLMExtractor(client, max_attempts=2).extract("texto")
    assert result.schema_valid
    assert result.attempts == 2
    with pytest.raises(json.JSONDecodeError) as exc:
        json.loads(bad)
    follow_up = client.calls[1][-1]["content"]
    assert str(exc.value) in follow_up
    assert client.calls[1][2]["role"] == "assistant"
    assert client.calls[1][2]["content"] == bad
    assert client.outputs_remaining == 1


def test_schema_errors_are_included_verbatim_in_the_repair_prompt() -> None:
    client = ScriptedLLMClient(["{}", json.dumps(VALID)])
    result = LLMExtractor(client, max_attempts=2).extract("texto")
    _, errors = evaluate_order_dict({})
    assert errors
    follow_up = client.calls[1][-1]["content"]
    for error in errors:
        assert error in follow_up
    assert result.schema_valid
    assert result.attempts == 2


def test_extra_property_triggers_one_repair() -> None:
    invalid = dict(VALID)
    invalid["extra"] = "surpresa"
    _, errors = evaluate_order_dict(invalid)
    client = ScriptedLLMClient([json.dumps(invalid), json.dumps(VALID)])
    result = LLMExtractor(client, max_attempts=2).extract("texto")
    assert result.schema_valid
    assert result.attempts == 2
    follow_up = client.calls[1][-1]["content"]
    for error in errors:
        assert error in follow_up


def test_budget_stops_while_output_is_still_invalid() -> None:
    client = ScriptedLLMClient(["nope", "still nope", "should not be used"])
    result = LLMExtractor(client, max_attempts=2).extract("texto")
    assert result.attempts == 2
    assert result.json_valid is False
    assert result.schema_valid is False
    assert result.data is None
    assert len(client.calls) == 2
    assert client.outputs_remaining == 1


def test_schema_valid_wrong_amount_does_not_repair() -> None:
    wrong = dict(VALID)
    wrong["amount_brl"] = "1.00"
    client = ScriptedLLMClient([json.dumps(wrong), "unused"])
    result = LLMExtractor(client, max_attempts=3).extract("texto")
    assert result.schema_valid
    assert result.attempts == 1
    assert client.outputs_remaining == 1
    score = match_record(result.data, VALID)
    fields = {item.field: item for item in score.fields}
    assert score.exact is False
    assert fields["amount_brl"].exact is False
    assert fields["amount_brl"].normalized is False
    assert fields["customer_name"].exact is True


def test_max_attempts_must_be_positive() -> None:
    client = ScriptedLLMClient([])
    with pytest.raises(ValueError, match="max_attempts"):
        LLMExtractor(client, max_attempts=0)
    assert isinstance(LLMExtractor(ScriptedLLMClient([json.dumps(VALID)])), Extractor)


def test_scripted_client_requires_an_output() -> None:
    client = ScriptedLLMClient([])
    with pytest.raises(RuntimeError, match="ran out"):
        client.complete([{"role": "user", "content": "oi"}])
