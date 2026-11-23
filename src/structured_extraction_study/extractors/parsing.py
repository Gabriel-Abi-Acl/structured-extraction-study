"""Converte a saída bruta em `ExtractionResult` com a mesma régua para regras e LLM."""

import json
import re

from structured_extraction_study.result import ExtractionResult
from structured_extraction_study.schemas.order import evaluate_order_dict

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def strip_json_fence(raw_output: str) -> str:
    """Remove uma cerca markdown que envolva o objeto inteiro. Prosa ao redor fica."""

    match = _FENCE.match(raw_output)
    if match is None:
        return raw_output.strip()
    return match.group(1).strip()


def to_result(raw_output: str, *, attempts: int, latency_ms: float) -> ExtractionResult:
    """Classifica JSON inválido, JSON que foge do esquema e objeto aceito."""

    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if latency_ms < 0:
        raise ValueError("latency_ms must be >= 0")
    try:
        parsed = json.loads(strip_json_fence(raw_output))
    except json.JSONDecodeError as exc:
        return ExtractionResult(
            raw_output=raw_output,
            data=None,
            json_valid=False,
            schema_valid=False,
            errors=(str(exc),),
            attempts=attempts,
            latency_ms=latency_ms,
        )
    schema_valid, errors = evaluate_order_dict(parsed)
    data = dict(parsed) if isinstance(parsed, dict) else None
    return ExtractionResult(
        raw_output=raw_output,
        data=data,
        json_valid=True,
        schema_valid=schema_valid,
        errors=errors,
        attempts=attempts,
        latency_ms=latency_ms,
    )
