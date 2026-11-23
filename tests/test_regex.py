"""Contrato do baseline: rótulo explícito, forma de superfície e sem campo inventado."""

import re

from structured_extraction_study.extractors.protocol import Extractor
from structured_extraction_study.extractors.regex import RegexExtractor
from structured_extraction_study.fixtures import load_placeholder_fixture
from structured_extraction_study.metrics import match_record, summarize_record_matches


def test_regex_on_placeholder_matches_gold_after_normalization() -> None:
    extractor = RegexExtractor()
    scores = []
    rows = load_placeholder_fixture()
    for row in rows:
        result = extractor.extract(row["text"])
        assert result.json_valid, row["id"]
        assert result.schema_valid, row["id"]
        assert result.attempts == 1
        assert result.latency_ms >= 0
        score = match_record(result.data, row["gold"])
        for field in score.fields:
            assert field.normalized, (row["id"], field.field, result.data, row["gold"])
        scores.append(score)
    summary = summarize_record_matches(scores)
    assert all(rate == 1.0 for rate in summary.per_field_normalized.values())
    assert summary.record_exact_rate == 0.0


def test_regex_keeps_surface_form_on_the_first_order() -> None:
    row = load_placeholder_fixture()[0]
    data = RegexExtractor().extract(row["text"]).data
    assert data is not None
    assert data["customer_name"] == "Ana Souza"
    assert data["order_date"] == "15/03/2026"
    assert data["amount_brl"] == "R$ 1.250,00"
    assert data["phone"] == "(11) 98888-0001"
    assert data["document_id"] == "123.456.789-09"
    score = match_record(data, row["gold"])
    fields = {item.field: item for item in score.fields}
    assert fields["order_date"].exact is False
    assert fields["order_date"].normalized is True
    assert fields["customer_name"].exact is True


def test_regex_leaves_unlabeled_ticket_fields_null() -> None:
    row = next(item for item in load_placeholder_fixture() if item["id"] == "fx-008")
    data = RegexExtractor().extract(row["text"]).data
    assert data is not None
    assert data["document_id"] is None
    assert data["order_date"] is None
    assert data["amount_brl"] is None
    assert data["item"] == "Atraso na entrega"


def test_regex_does_not_invent_unlabeled_values() -> None:
    text = (
        "Olá, o pedido de ana.souza@example.com atrasou. "
        "O valor combinado foi 10,00 e o telefone 11988880001."
    )
    result = RegexExtractor().extract(text)
    assert result.schema_valid
    assert result.data is not None
    assert all(value is None for value in result.data.values())


def test_regex_extractor_satisfies_the_protocol() -> None:
    assert isinstance(RegexExtractor(), Extractor)
    emails = []
    for row in load_placeholder_fixture():
        emails.extend(re.findall(r"[\w.+-]+@[\w.-]+", row["text"]))
    assert emails
    assert all(email.endswith("@example.com") for email in emails)
