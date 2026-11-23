"""Harness offline dos três extratores e o comando `evaluate-extraction`.

A medição reusa o casamento de campo já existente. Erro de execução fica de fora
do denominador; JSON inválido continua na taxa. Número de cliente scriptado é
checagem de pipeline, não resultado de modelo.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from structured_extraction_study.config import load_settings
from structured_extraction_study.dataset import (
    EVAL_CATEGORIES,
    SCRIPTED_OUTPUT_KINDS,
    default_eval_texts_path,
    load_eval_texts,
)
from structured_extraction_study.extractors.llm import LLMExtractor
from structured_extraction_study.extractors.protocol import Extractor
from structured_extraction_study.extractors.regex import RegexExtractor
from structured_extraction_study.llm.client import ScriptedLLMClient
from structured_extraction_study.metrics import (
    RecordScore,
    match_record,
    schema_valid_rate,
    summarize_record_matches,
    valid_json_rate,
)
from structured_extraction_study.result import ExtractionResult

LATENCY_TIMING = "offline_wall_clock"
PIPELINE_NOTE = (
    "Números de backend scriptado são checagem de pipeline, não resultado de modelo. "
    "A latência é tempo de parede offline dentro de extract (offline_wall_clock), "
    "não latência de API."
)
SYSTEM_REGEX = "regex"
SYSTEM_LLM = "llm_scripted"
SYSTEM_LLM_REPAIR = "llm_scripted_repair"
SYSTEMS: tuple[str, ...] = (SYSTEM_REGEX, SYSTEM_LLM, SYSTEM_LLM_REPAIR)
SCENARIOS: tuple[str, ...] = (*SCRIPTED_OUTPUT_KINDS, "mix")
MISSING_KEY_MESSAGE = (
    "evaluate-extraction: --allow-live requires an API key in the environment "
    "(OPENAI_API_KEY or ANTHROPIC_API_KEY). No request was sent."
)
LIVE_UNAVAILABLE_MESSAGE = (
    "evaluate-extraction: live evaluation is not implemented. No request was sent."
)


@dataclass(frozen=True)
class EvalObservation:
    """Uma extração já classificada, ou um erro de execução sem métrica."""

    text_id: str
    category: str
    system: str
    error: str | None
    json_valid: bool | None
    schema_valid: bool | None
    attempts: int | None
    latency_ms: float | None
    record: RecordScore | None
    predicted: dict[str, Any] | None
    raw_output: str | None
    gold: dict[str, Any]

    def __post_init__(self) -> None:
        scored = (
            self.json_valid,
            self.schema_valid,
            self.attempts,
            self.latency_ms,
            self.record,
        )
        if self.error is None:
            if any(value is None for value in scored):
                raise ValueError("scored observations need metrics")
            return
        if any(value is not None for value in scored):
            raise ValueError("execution errors are not scored")


def run_evaluation(
    rows: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    repair_attempts: int,
) -> list[EvalObservation]:
    """Roda regex, LLM scriptado sem reparo e LLM scriptado com reparo."""

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    if repair_attempts < 1:
        raise ValueError("repair_attempts must be >= 1")
    if len(rows) == 0:
        raise ValueError("rows must not be empty")
    observations: list[EvalObservation] = []
    for row in rows:
        for system in SYSTEMS:
            observations.append(
                _run_system(row, system, scenario=scenario, repair_attempts=repair_attempts)
            )
    return observations


def observe_extraction(
    row: Mapping[str, Any],
    system: str,
    extractor: Extractor,
) -> EvalObservation:
    """Mede um extrator. Exceção vira erro e não entra na taxa."""

    try:
        result = extractor.extract(str(row["text"]))
    except Exception as exc:
        return _error_observation(row, system, exc)
    return _scored_observation(row, system, result)


def build_summary(
    observations: Sequence[EvalObservation],
    *,
    scenario: str,
    repair_attempts: int,
    dataset: str,
) -> dict[str, Any]:
    """Agrega por sistema e por categoria. A latência fica marcada como offline."""

    grouped = _by_system(observations)
    ordered = [name for name in SYSTEMS if name in grouped]
    ordered.extend(sorted(set(grouped) - set(SYSTEMS)))
    return {
        "dataset": dataset,
        "scenario": scenario,
        "repair_max_attempts": repair_attempts,
        "latency_timing": LATENCY_TIMING,
        "pipeline_check": True,
        "note": PIPELINE_NOTE,
        "text_count": len({item.text_id for item in observations}),
        "systems": ordered,
        "by_system": {name: grouped[name] for name in ordered},
    }


def render_report(summary: Mapping[str, Any]) -> str:
    """Tabelas em Markdown com a mesma agregação de `summary.json`."""

    lines = [
        "# Avaliação offline",
        "",
        str(summary["note"]),
        "",
        f"Dataset: `{summary['dataset']}`.",
        f"Cenário scriptado: `{summary['scenario']}`.",
        (
            "Orçamento de reparo de `llm_scripted_repair`: "
            f"{summary['repair_max_attempts']} (inclui a primeira chamada). "
            "`llm_scripted` faz uma chamada."
        ),
        f"Marcação de latência: `{summary['latency_timing']}`.",
        "",
        "Sistemas:",
        "",
        "- `regex`: baseline `RegexExtractor`. Não usa a saída scriptada.",
        "- `llm_scripted`: `LLMExtractor` com `max_attempts=1` (sem reparo).",
        "- `llm_scripted_repair`: `LLMExtractor` com o orçamento de reparo.",
        "",
        (
            "Erro de execução entra na coluna Erros e sai dos denominadores. "
            "JSON inválido ou fora do esquema permanece nas taxas."
        ),
        "",
    ]
    by_system = summary["by_system"]
    for system in summary["systems"]:
        lines.extend(_system_section(str(system), by_system[system]))
    return "\n".join(lines).rstrip() + "\n"


def write_evaluation(
    output_dir: Path,
    observations: Sequence[EvalObservation],
    summary: Mapping[str, Any],
) -> None:
    """Grava `records.jsonl`, `summary.json` e `report.md`."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records = "".join(
        json.dumps(_record_payload(item), ensure_ascii=False) + "\n" for item in observations
    )
    (output_dir / "records.jsonl").write_text(records, encoding="utf-8")
    encoded = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (output_dir / "summary.json").write_text(encoded, encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")


def resolve_repair_attempts(explicit: int | None, env: Mapping[str, str]) -> int:
    """Usa o valor do comando ou, na ausência, `REPAIR_MAX_ATTEMPTS`."""

    if explicit is None:
        return load_settings(env).repair_max_attempts
    if explicit < 1:
        raise ValueError("repair_attempts must be >= 1")
    return explicit


def refuse_live(env: Mapping[str, str]) -> str:
    """Recusa a chamada ao vivo. Este comando não abre rede em nenhum caso."""

    settings = load_settings(env)
    if settings.openai_api_key == "" and settings.anthropic_api_key == "":
        return MISSING_KEY_MESSAGE
    return LIVE_UNAVAILABLE_MESSAGE


def dataset_label(path: Path) -> str:
    """Nome estável do dataset padrão; outro caminho fica como foi passado."""

    if path.resolve() == default_eval_texts_path().resolve():
        return "data/eval/texts.v1.jsonl"
    return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Comando `evaluate-extraction`. Código 2 fecha a porta ao vivo."""

    parser = _parser()
    args = parser.parse_args(None if argv is None else list(argv))
    if args.allow_live:
        print(refuse_live(_environ()), file=sys.stderr)
        return 2
    try:
        repair_attempts = resolve_repair_attempts(args.repair_attempts, _environ())
    except ValueError as exc:
        print(f"evaluate-extraction: {exc}", file=sys.stderr)
        return 2
    source = args.dataset if args.dataset is not None else default_eval_texts_path()
    try:
        rows = load_eval_texts(source)
        observations = run_evaluation(rows, scenario=args.scenario, repair_attempts=repair_attempts)
        summary = build_summary(
            observations,
            scenario=args.scenario,
            repair_attempts=repair_attempts,
            dataset=dataset_label(source),
        )
        write_evaluation(args.output_dir, observations, summary)
    except (OSError, ValueError) as exc:
        print(f"evaluate-extraction: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(observations)} records to {args.output_dir}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-extraction",
        description=(
            "Compare the regex baseline with a scripted LLM extractor, with and without repair. "
            "Scripted numbers are pipeline checks, not model results."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for records.jsonl, summary.json, and report.md.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Labeled JSONL file. Defaults to data/eval/texts.v1.jsonl.",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="mix",
        help=(
            "Per-line scripted output: valid, malformed, partially_wrong, or mix. "
            "mix returns malformed first and the gold JSON on later repair calls. Default: mix."
        ),
    )
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=None,
        help=(
            "Attempt budget for llm_scripted_repair, including the first call. "
            "Default: REPAIR_MAX_ATTEMPTS, or 2."
        ),
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help=(
            "Request the live LLM path. It is not implemented and never sends a request. "
            "Without OPENAI_API_KEY or ANTHROPIC_API_KEY the command exits with code 2."
        ),
    )
    return parser


def _environ() -> Mapping[str, str]:
    return os.environ


def _run_system(
    row: Mapping[str, Any],
    system: str,
    *,
    scenario: str,
    repair_attempts: int,
) -> EvalObservation:
    try:
        extractor = _build_extractor(
            row, system, scenario=scenario, repair_attempts=repair_attempts
        )
    except (TypeError, ValueError) as exc:
        return _error_observation(row, system, exc)
    return observe_extraction(row, system, extractor)


def _build_extractor(
    row: Mapping[str, Any],
    system: str,
    *,
    scenario: str,
    repair_attempts: int,
) -> Extractor:
    if system == SYSTEM_REGEX:
        return RegexExtractor()
    if system == SYSTEM_LLM:
        calls = 1
    elif system == SYSTEM_LLM_REPAIR:
        calls = repair_attempts
    else:
        raise ValueError(f"unknown system: {system}")
    client = ScriptedLLMClient(_script(row, scenario, calls))
    return LLMExtractor(client, max_attempts=calls)


def _script(row: Mapping[str, Any], scenario: str, calls: int) -> list[str]:
    # `mix` usa as duas strings já rotuladas: inválida na primeira, ouro no reparo.
    # Os outros cenários repetem a saída fixa da linha, sem texto novo.
    scripted = row.get("scripted_outputs")
    if not isinstance(scripted, Mapping):
        raise ValueError("scripted_outputs is required for the scripted LLM systems")
    if scenario == "mix":
        malformed = _output(scripted, "malformed")
        valid = _output(scripted, "valid")
        return [malformed, *([valid] * (calls - 1))]
    return [_output(scripted, scenario)] * calls


def _output(scripted: Mapping[str, Any], kind: str) -> str:
    if kind not in SCRIPTED_OUTPUT_KINDS:
        raise ValueError(f"unknown scenario: {kind}")
    value = scripted.get(kind)
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"scripted_outputs.{kind} must be a string")
    return value


def _scored_observation(
    row: Mapping[str, Any],
    system: str,
    result: ExtractionResult,
) -> EvalObservation:
    gold = dict(row["gold"])
    score = match_record(result.data, gold)
    predicted = None if result.data is None else dict(result.data)
    return EvalObservation(
        text_id=str(row["id"]),
        category=str(row["category"]),
        system=system,
        error=None,
        json_valid=result.json_valid,
        schema_valid=result.schema_valid,
        attempts=result.attempts,
        latency_ms=result.latency_ms,
        record=score,
        predicted=predicted,
        raw_output=result.raw_output,
        gold=gold,
    )


def _error_observation(row: Mapping[str, Any], system: str, exc: BaseException) -> EvalObservation:
    return EvalObservation(
        text_id=str(row["id"]),
        category=str(row["category"]),
        system=system,
        error=f"{type(exc).__name__}: {exc}",
        json_valid=None,
        schema_valid=None,
        attempts=None,
        latency_ms=None,
        record=None,
        predicted=None,
        raw_output=None,
        gold=dict(row["gold"]),
    )


def _by_system(observations: Sequence[EvalObservation]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EvalObservation]] = {}
    for item in observations:
        grouped.setdefault(item.system, []).append(item)
    rendered: dict[str, dict[str, Any]] = {}
    for system, items in grouped.items():
        categories: dict[str, list[EvalObservation]] = {}
        for item in items:
            categories.setdefault(item.category, []).append(item)
        names = [name for name in EVAL_CATEGORIES if name in categories]
        names.extend(sorted(set(categories) - set(EVAL_CATEGORIES)))
        rendered[system] = {
            "overall": _metric_block(items),
            "by_category": {name: _metric_block(categories[name]) for name in names},
        }
    return rendered


def _metric_block(items: Sequence[EvalObservation]) -> dict[str, Any]:
    # Erro de execução não entra na taxa. JSON inválido continua no denominador.
    scored = [item for item in items if item.error is None]
    block: dict[str, Any] = {
        "scored": len(scored),
        "errors": len(items) - len(scored),
        "record_exact_rate": None,
        "per_field_exact": None,
        "per_field_normalized": None,
        "valid_json_rate": None,
        "schema_valid_rate": None,
        "mean_attempts": None,
        "mean_latency_ms": None,
        "latency_timing": LATENCY_TIMING,
    }
    if not scored:
        return block
    matches: list[RecordScore] = []
    json_flags: list[bool] = []
    schema_flags: list[bool] = []
    attempts_total = 0
    latency_total = 0.0
    for item in scored:
        if (
            item.record is None
            or item.json_valid is None
            or item.schema_valid is None
            or item.attempts is None
            or item.latency_ms is None
        ):
            raise ValueError("scored observation is missing metrics")
        matches.append(item.record)
        json_flags.append(item.json_valid)
        schema_flags.append(item.schema_valid)
        attempts_total += item.attempts
        latency_total += item.latency_ms
    field_summary = summarize_record_matches(matches)
    count = len(matches)
    block["record_exact_rate"] = field_summary.record_exact_rate
    block["per_field_exact"] = field_summary.per_field_exact
    block["per_field_normalized"] = field_summary.per_field_normalized
    block["valid_json_rate"] = valid_json_rate(json_flags)
    block["schema_valid_rate"] = schema_valid_rate(schema_flags)
    block["mean_attempts"] = attempts_total / count
    block["mean_latency_ms"] = latency_total / count
    return block


def _record_payload(item: EvalObservation) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": item.text_id,
        "category": item.category,
        "system": item.system,
        "execution_error": item.error,
        "json_valid": item.json_valid,
        "schema_valid": item.schema_valid,
        "attempts": item.attempts,
        "latency_ms": item.latency_ms,
        "latency_timing": LATENCY_TIMING,
        "record_exact": None if item.record is None else item.record.exact,
        "per_field_exact": None,
        "per_field_normalized": None,
        "predicted": item.predicted,
        "raw_output": item.raw_output,
        "gold": item.gold,
    }
    if item.record is not None:
        payload["per_field_exact"] = {score.field: score.exact for score in item.record.fields}
        payload["per_field_normalized"] = {
            score.field: score.normalized for score in item.record.fields
        }
    return payload


def _system_section(system: str, payload: Mapping[str, Any]) -> list[str]:
    return [
        f"## {system}",
        "",
        _rates_table(payload),
        "",
        "### Campos",
        "",
        _fields_table(payload),
        "",
    ]


def _rates_table(payload: Mapping[str, Any]) -> str:
    header = (
        "| Categoria | Avaliados | Erros | Registro exato | JSON válido | "
        "Esquema válido | Tentativas médias | Latência média (ms, offline) |"
    )
    separator = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    rows = [_rate_row("todas", payload["overall"])]
    by_category = payload["by_category"]
    rows.extend(_rate_row(name, by_category[name]) for name in by_category)
    return "\n".join([header, separator, *rows])


def _rate_row(label: str, block: Mapping[str, Any]) -> str:
    cells = [
        label,
        str(block["scored"]),
        str(block["errors"]),
        _fmt(block["record_exact_rate"]),
        _fmt(block["valid_json_rate"]),
        _fmt(block["schema_valid_rate"]),
        _fmt(block["mean_attempts"]),
        _fmt(block["mean_latency_ms"]),
    ]
    return "| " + " | ".join(cells) + " |"


def _fields_table(payload: Mapping[str, Any]) -> str:
    header = "| Categoria | Campo | Exato | Normalizado |"
    separator = "| --- | --- | ---: | ---: |"
    rows = [*_field_rows("todas", payload["overall"])]
    for name, block in payload["by_category"].items():
        rows.extend(_field_rows(name, block))
    return "\n".join([header, separator, *rows])


def _field_rows(label: str, block: Mapping[str, Any]) -> list[str]:
    exact = block["per_field_exact"]
    normalized = block["per_field_normalized"]
    if not isinstance(exact, Mapping) or not isinstance(normalized, Mapping):
        return [f"| {label} | — | — | — |"]
    return [
        f"| {label} | {field} | {_fmt(exact[field])} | {_fmt(normalized[field])} |"
        for field in exact
    ]


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("metric value must be a number or null")
    return f"{float(value):.3f}"
