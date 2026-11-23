"""Métricas de campo e de registro. Não agregam latência nem custo.

Correspondência exata compara o valor extraído com o ouro.
A normalizada aplica o normalizador dos dois lados.
Campo ausente no previsto não equivale a nulo.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from structured_extraction_study.normalize import FIELD_NORMALIZERS

Normalizer = Callable[[Any], Any]


@dataclass(frozen=True)
class FieldScore:
    """Um campo do ouro contra o valor previsto, quando ele existe."""

    field: str
    exact: bool
    normalized: bool


@dataclass(frozen=True)
class RecordScore:
    """O registro só é exato quando cada campo do ouro é exato."""

    exact: bool
    fields: tuple[FieldScore, ...]


@dataclass(frozen=True)
class MetricsSummary:
    """Taxas no intervalo [0, 1] para um conjunto não vazio de registros."""

    record_exact_rate: float
    per_field_exact: dict[str, float]
    per_field_normalized: dict[str, float]


def match_record(
    predicted: Mapping[str, Any] | None,
    gold: Mapping[str, Any],
    normalizers: Mapping[str, Normalizer] | None = None,
) -> RecordScore:
    """Compara um registro. `predicted=None` marca falha em todos os campos do ouro."""

    if not isinstance(gold, Mapping) or len(gold) == 0:
        raise ValueError("gold must contain at least one field")
    if predicted is not None and not isinstance(predicted, Mapping):
        raise TypeError("predicted must be a mapping or null")
    chosen = FIELD_NORMALIZERS if normalizers is None else normalizers
    scores: list[FieldScore] = []
    for key, gold_value in gold.items():
        if not isinstance(key, str):
            raise TypeError("field names must be strings")
        scores.append(_score_field(key, predicted, gold_value, chosen))
    return RecordScore(
        exact=all(score.exact for score in scores),
        fields=tuple(scores),
    )


def summarize_record_matches(matches: Sequence[RecordScore]) -> MetricsSummary:
    """Agrega correspondência exata do registro e as taxas por campo."""

    if len(matches) == 0:
        raise ValueError("matches must not be empty")
    field_names = tuple(score.field for score in matches[0].fields)
    if not field_names:
        raise ValueError("matches must include fields")
    for match in matches:
        names = tuple(score.field for score in match.fields)
        if names != field_names:
            raise ValueError("RecordScore field sets differ")
    per_field_exact: dict[str, float] = {}
    per_field_normalized: dict[str, float] = {}
    for index, field_name in enumerate(field_names):
        column = [match.fields[index] for match in matches]
        per_field_exact[field_name] = _mean(score.exact for score in column)
        per_field_normalized[field_name] = _mean(score.normalized for score in column)
    return MetricsSummary(
        record_exact_rate=_mean(match.exact for match in matches),
        per_field_exact=per_field_exact,
        per_field_normalized=per_field_normalized,
    )


def valid_json_rate(json_valid_flags: Sequence[bool]) -> float:
    """Fração de saídas que parsearam como JSON."""

    return _fraction_true(json_valid_flags)


def schema_valid_rate(schema_valid_flags: Sequence[bool]) -> float:
    """Fração de saídas aceitas pelo esquema. JSON válido pode falhar aqui."""

    return _fraction_true(schema_valid_flags)


def _score_field(
    key: str,
    predicted: Mapping[str, Any] | None,
    gold_value: Any,
    normalizers: Mapping[str, Normalizer],
) -> FieldScore:
    if predicted is None or key not in predicted:
        return FieldScore(field=key, exact=False, normalized=False)
    predicted_value = predicted[key]
    exact = predicted_value == gold_value
    normalizer = normalizers.get(key)
    if normalizer is None:
        return FieldScore(field=key, exact=exact, normalized=exact)
    try:
        normalized = normalizer(predicted_value) == normalizer(gold_value)
    except (TypeError, ValueError):
        normalized = False
    return FieldScore(field=key, exact=exact, normalized=normalized)


def _fraction_true(flags: Sequence[bool]) -> float:
    if isinstance(flags, (str, bytes)) or not isinstance(flags, Sequence):
        raise TypeError("flags must be a sequence of booleans")
    if len(flags) == 0:
        raise ValueError("flags must not be empty")
    total = 0
    for flag in flags:
        if not isinstance(flag, bool):
            raise TypeError("flags must be booleans")
        total += int(flag)
    return total / len(flags)


def _mean(flags: Iterable[bool]) -> float:
    values = list(flags)
    if not values:
        raise ValueError("flags must not be empty")
    return sum(1 for flag in values if flag) / len(values)
