"""Normalizadores de campo para a métrica de correspondência normalizada.

A forma canônica é estável e idempotente quando a entrada já está canônica.
Valor que não cabe na regra volta sem alteração, para o casamento falhar
em vez de inventar uma data, um valor ou um identificador.
"""

import re
from collections.abc import Callable
from datetime import date
from decimal import Decimal

Normalizer = Callable[[str | None], str | None]

_DATE_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATE_BR = re.compile(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$")
_CURRENCY_PREFIX = re.compile(r"(?i)^r\$\s*")


def normalize_text(value: str | None) -> str | None:
    """Nome, cidade e item: caixa e espaços deixam de separar o mesmo texto."""

    text = _require_text(value)
    if text is None:
        return None
    return re.sub(r"\s+", " ", text).strip().casefold()


def normalize_email(value: str | None) -> str | None:
    """E-mail comparado sem caixa e sem espaço nas pontas."""

    text = _require_text(value)
    if text is None:
        return None
    return text.strip().casefold()


def normalize_phone(value: str | None) -> str | None:
    """Telefone em dígitos. O prefixo 55 sai só quando sobra um número nacional."""

    text = _require_text(value)
    if text is None:
        return None
    digits = re.sub(r"\D", "", text)
    if digits.startswith("55") and len(digits) in {12, 13}:
        return digits[2:]
    return digits


def normalize_document_id(value: str | None) -> str | None:
    """Identificador fictício no formato de CPF: só os dígitos, sem validar DV."""

    text = _require_text(value)
    if text is None:
        return None
    return re.sub(r"\D", "", text)


def normalize_date(value: str | None) -> str | None:
    """Data ISO `YYYY-MM-DD`. Dia inexistente permanece como foi escrito."""

    text = _require_text(value)
    if text is None:
        return None
    stripped = text.strip()
    iso_match = _DATE_ISO.fullmatch(stripped)
    if iso_match:
        return _safe_iso(
            int(iso_match.group(1)),
            int(iso_match.group(2)),
            int(iso_match.group(3)),
            stripped,
        )
    br_match = _DATE_BR.fullmatch(stripped)
    if br_match:
        return _safe_iso(
            int(br_match.group(3)),
            int(br_match.group(2)),
            int(br_match.group(1)),
            stripped,
        )
    return stripped


def normalize_currency(value: str | None) -> str | None:
    """Valor em reais como string com duas casas. Milhar com ponto só vale junto da vírgula."""

    text = _require_text(value)
    if text is None:
        return None
    original = text.strip()
    amount = _CURRENCY_PREFIX.sub("", original).replace(" ", "")
    if not amount:
        return original
    if "," in amount:
        amount = amount.replace(".", "").replace(",", ".")
    if re.fullmatch(r"\d+(?:\.\d{1,2})?", amount):
        quantized = Decimal(amount).quantize(Decimal("0.01"))
        return f"{quantized:.2f}"
    return original


FIELD_NORMALIZERS: dict[str, Normalizer] = {
    "customer_name": normalize_text,
    "email": normalize_email,
    "phone": normalize_phone,
    "document_id": normalize_document_id,
    "order_date": normalize_date,
    "amount_brl": normalize_currency,
    "city": normalize_text,
    "item": normalize_text,
}


def _require_text(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("expected a string or null")
    return value


def _safe_iso(year: int, month: int, day: int, original: str) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return original
