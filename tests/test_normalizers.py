"""Contrato dos normalizadores: forma canônica estável, sem inventar valor inválido."""

import pytest

from structured_extraction_study.normalize import (
    normalize_currency,
    normalize_date,
    normalize_document_id,
    normalize_email,
    normalize_phone,
    normalize_text,
)


def test_date_accepts_brazilian_separators_and_iso() -> None:
    assert normalize_date("15/03/2026") == "2026-03-15"
    assert normalize_date("02.04.2026") == "2026-04-02"
    assert normalize_date("20-05-2026") == "2026-05-20"
    assert normalize_date("7/7/2026") == "2026-07-07"
    assert normalize_date("2026-01-08") == "2026-01-08"
    assert normalize_date(" 2026-11-30 ") == "2026-11-30"


def test_impossible_date_stays_unchanged() -> None:
    assert normalize_date("31/02/2026") == "31/02/2026"
    assert normalize_date("2026-02-31") == "2026-02-31"
    assert normalize_date("amanhã") == "amanhã"


def test_currency_brazilian_grouping_and_canonical_decimal() -> None:
    assert normalize_currency("R$ 1.250,00") == "1250.00"
    assert normalize_currency("R$ 89,90") == "89.90"
    assert normalize_currency("320,00") == "320.00"
    assert normalize_currency("R$ 2.000,50") == "2000.50"
    assert normalize_currency("R$ 15.999,99") == "15999.99"
    assert normalize_currency("R$ 0,50") == "0.50"
    assert normalize_currency("89.90") == "89.90"
    assert normalize_currency("89.9") == "89.90"
    assert normalize_currency("10") == "10.00"


def test_ambiguous_currency_is_not_rewritten() -> None:
    assert normalize_currency("1.250") == "1.250"
    assert normalize_currency("a combinar") == "a combinar"


def test_document_id_strips_mask_without_checksum() -> None:
    assert normalize_document_id("123.456.789-09") == "12345678909"
    assert normalize_document_id("000.111.222-33") == "00011122233"
    assert normalize_document_id("12345678900") == "12345678900"
    assert normalize_document_id("sem numero") == ""


def test_phone_drops_country_code_only_when_national_length_remains() -> None:
    assert normalize_phone("(11) 98888-0001") == "11988880001"
    assert normalize_phone("+55 31 98888-0003") == "31988880003"
    assert normalize_phone("+55 (85) 98888-0007") == "85988880007"
    assert normalize_phone("(41) 3333-0004") == "4133330004"
    assert normalize_phone("5511988880001") == "11988880001"
    assert normalize_phone("551133330004") == "1133330004"
    assert normalize_phone("55988880001") == "55988880001"


def test_text_and_email_fold_case_and_space() -> None:
    assert normalize_text("  Ana   Souza ") == "ana souza"
    assert normalize_text("Niterói") == "niterói"
    assert normalize_email(" Ana.Souza@Example.com ") == "ana.souza@example.com"


def test_null_is_preserved_and_non_strings_fail() -> None:
    assert normalize_date(None) is None
    assert normalize_currency(None) is None
    assert normalize_phone(None) is None
    assert normalize_document_id(None) is None
    with pytest.raises(TypeError):
        normalize_currency(10)  # type: ignore[arg-type]


def test_canonical_values_are_idempotent() -> None:
    assert normalize_date(normalize_date("15/03/2026")) == "2026-03-15"
    assert normalize_currency(normalize_currency("R$ 1.250,00")) == "1250.00"
    assert normalize_phone(normalize_phone("+55 11 98888-0001")) == "11988880001"
    assert normalize_document_id(normalize_document_id("123.456.789-09")) == "12345678909"
