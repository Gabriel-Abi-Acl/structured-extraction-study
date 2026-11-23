"""Contrato das métricas: exato, normalizado, registro inteiro e taxas binárias."""

import pytest

from structured_extraction_study.metrics import (
    match_record,
    schema_valid_rate,
    summarize_record_matches,
    valid_json_rate,
)

GOLD = {
    "customer_name": "Ana Souza",
    "email": "ana.souza@example.com",
    "phone": "11988880001",
    "document_id": "12345678909",
    "order_date": "2026-03-15",
    "amount_brl": "1250.00",
    "city": "Campinas",
    "item": "Teclado mecânico",
}


def test_exact_and_normalized_field_match() -> None:
    predicted = dict(GOLD)
    predicted["order_date"] = "15/03/2026"
    predicted["amount_brl"] = "R$ 1.250,00"
    predicted["customer_name"] = "ana  souza"
    score = match_record(predicted, GOLD)
    fields = {item.field: item for item in score.fields}
    assert score.exact is False
    assert fields["order_date"].exact is False
    assert fields["order_date"].normalized is True
    assert fields["amount_brl"].exact is False
    assert fields["amount_brl"].normalized is True
    assert fields["customer_name"].exact is False
    assert fields["customer_name"].normalized is True
    assert fields["email"].exact is True
    assert match_record(dict(GOLD), GOLD).exact is True


def test_missing_key_is_not_a_null_match() -> None:
    missing = match_record({}, {"city": None})
    present = match_record({"city": None}, {"city": None})
    assert missing.fields[0].exact is False
    assert missing.fields[0].normalized is False
    assert present.fields[0].exact is True
    assert present.fields[0].normalized is True
    assert match_record(None, {"city": "Campinas"}).exact is False


def test_unknown_field_without_normalizer_uses_equality() -> None:
    same = match_record({"note": "Abc"}, {"note": "Abc"}, normalizers={})
    folded = match_record({"note": "abc"}, {"note": "Abc"}, normalizers={})
    assert same.exact is True
    assert folded.fields[0].normalized is False


def test_non_string_currency_does_not_count_as_normalized() -> None:
    score = match_record({"amount_brl": 10}, {"amount_brl": "10.00"})
    assert score.fields[0].exact is False
    assert score.fields[0].normalized is False


def test_summary_rates() -> None:
    first = match_record(dict(GOLD), GOLD)
    second = match_record({"customer_name": "Outra Pessoa"}, {"customer_name": "Ana Souza"})
    summary = summarize_record_matches([first])
    assert summary.record_exact_rate == 1.0
    assert summary.per_field_exact["amount_brl"] == 1.0
    mixed = summarize_record_matches([first, first])
    assert mixed.record_exact_rate == 1.0
    assert second.exact is False
    with pytest.raises(ValueError, match="differ"):
        summarize_record_matches([first, second])


def test_boolean_rates_reject_empty_and_non_booleans() -> None:
    assert valid_json_rate([True, False, True]) == pytest.approx(2 / 3)
    assert schema_valid_rate([True, True]) == 1.0
    with pytest.raises(ValueError, match="empty"):
        valid_json_rate([])
    with pytest.raises(TypeError):
        schema_valid_rate([1, 0])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="at least one"):
        match_record({}, {})
