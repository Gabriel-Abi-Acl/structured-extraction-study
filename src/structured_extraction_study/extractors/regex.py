"""Baseline determinístico: só lê rótulos explícitos e deixa o resto nulo.

A captura preserva a forma escrita no texto. A normalização fica na métrica,
para a correspondência exata e a normalizada poderem divergir.
"""

import json
import re
import time

from structured_extraction_study.extractors.parsing import to_result
from structured_extraction_study.result import ExtractionResult
from structured_extraction_study.schemas.order import ORDER_FIELDS

_LABELS: dict[str, str] = {
    "customer_name": r"cliente|nome",
    "email": r"e-?mail",
    "phone": r"telefone",
    "document_id": r"cpf",
    "order_date": r"data",
    "amount_brl": r"valor",
    "city": r"cidade|município|municipio",
    "item": r"item|produto|assunto",
}


def _compile_label(alternation: str) -> re.Pattern[str]:
    return re.compile(
        rf"^(?:{alternation})\s*:\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )


PATTERNS: dict[str, re.Pattern[str]] = {
    field: _compile_label(label) for field, label in _LABELS.items()
}

if set(PATTERNS) != set(ORDER_FIELDS):
    raise RuntimeError("Regex patterns must cover every order field")


class RegexExtractor:
    """Extrai um objeto completo. Campo sem rótulo sai `null` e não é inventado."""

    def extract(self, text: str) -> ExtractionResult:
        """Varre o texto uma vez. `attempts` permanece 1 porque não há reparo."""

        started = time.perf_counter()
        captured = {field: _capture(PATTERNS[field], text) for field in ORDER_FIELDS}
        elapsed_ms = (time.perf_counter() - started) * 1000
        raw_output = json.dumps(captured, ensure_ascii=False)
        return to_result(raw_output, attempts=1, latency_ms=elapsed_ms)


def _capture(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    value = match.group(1).strip()
    if value == "":
        return None
    return value
