"""Contrato da agregação: taxas por campo, erro fora do denominador e cenário scriptado."""

import pytest

from structured_extraction_study.dataset import load_eval_texts
from structured_extraction_study.evaluation import (
    EvalObservation,
    build_summary,
    observe_extraction,
    render_report,
    resolve_repair_attempts,
    run_evaluation,
)
from structured_extraction_study.extractors.regex import RegexExtractor
from structured_extraction_study.metrics import FieldScore, RecordScore

_RATES = (
    "| Categoria | Avaliados | Erros | Registro exato | JSON válido | "
    "Esquema válido | Tentativas médias | Latência média (ms, offline) |"
)


def test_summary_reports_exact_and_normalized_rates() -> None:
    row = {
        "id": "local-1",
        "category": "clean_template",
        "text": "Cliente: Ana Souza\nData: 15/03/2026\nValor: R$ 45,90\n",
        "gold": {
            "customer_name": "Ana Souza",
            "order_date": "2026-03-15",
            "amount_brl": "45.90",
        },
    }
    observation = observe_extraction(row, "regex", RegexExtractor())
    summary = _summary([observation])
    block = summary["by_system"]["regex"]["overall"]
    assert block["errors"] == 0
    assert block["scored"] == 1
    assert block["record_exact_rate"] == 0.0
    assert block["per_field_exact"]["customer_name"] == 1.0
    assert block["per_field_exact"]["order_date"] == 0.0
    assert block["per_field_normalized"]["order_date"] == 1.0
    assert block["per_field_exact"]["amount_brl"] == 0.0
    assert block["per_field_normalized"]["amount_brl"] == 1.0
    assert block["valid_json_rate"] == 1.0
    assert block["schema_valid_rate"] == 1.0
    assert block["mean_attempts"] == 1.0
    assert block["mean_latency_ms"] >= 0
    assert block["latency_timing"] == "offline_wall_clock"
    assert summary["latency_timing"] == "offline_wall_clock"
    assert summary["pipeline_check"] is True
    report = render_report(summary)
    assert summary["note"] in report
    assert _RATES in report
    assert "| todas | order_date | 0.000 | 1.000 |" in report
    todas = next(line for line in report.splitlines() if line.startswith("| todas |"))
    assert todas.startswith("| todas | 1 | 0 | 0.000 | 1.000 | 1.000 | 1.000 |")


def test_errors_are_excluded_from_metric_denominators() -> None:
    observations = [
        _scored("t-1", exact=True, attempts=1, latency_ms=10),
        _scored(
            "t-2",
            exact=False,
            json_valid=False,
            schema_valid=False,
            attempts=3,
            latency_ms=30,
        ),
        _errored("t-3"),
    ]
    block = _summary(observations)["by_system"]["regex"]["overall"]
    assert block["scored"] == 2
    assert block["errors"] == 1
    assert block["record_exact_rate"] == 0.5
    assert block["per_field_exact"]["city"] == 0.5
    assert block["per_field_normalized"]["city"] == 1.0
    assert block["valid_json_rate"] == 0.5
    assert block["schema_valid_rate"] == 0.5
    assert block["mean_attempts"] == 2.0
    assert block["mean_latency_ms"] == 20.0


def test_category_rates_do_not_absorb_errors_from_the_same_or_another_category() -> None:
    observations = [
        _scored("t-1", category="clean_template", exact=True, attempts=4, latency_ms=8),
        _errored("t-2", category="clean_template"),
        _scored("t-3", category="free_form", exact=False, attempts=2, latency_ms=4),
    ]
    regex = _summary(observations)["by_system"]["regex"]
    assert regex["overall"]["scored"] == 2
    assert regex["overall"]["errors"] == 1
    assert regex["overall"]["record_exact_rate"] == 0.5
    assert regex["overall"]["mean_attempts"] == 3.0
    clean = regex["by_category"]["clean_template"]
    assert clean["scored"] == 1
    assert clean["errors"] == 1
    assert clean["record_exact_rate"] == 1.0
    assert clean["mean_attempts"] == 4.0
    assert clean["mean_latency_ms"] == 8.0
    free = regex["by_category"]["free_form"]
    assert free["errors"] == 0
    assert free["record_exact_rate"] == 0.0
    assert free["mean_attempts"] == 2.0


def test_all_execution_errors_leave_rates_empty() -> None:
    summary = _summary([_errored("t-1")])
    block = summary["by_system"]["regex"]["overall"]
    assert block["scored"] == 0
    assert block["errors"] == 1
    assert block["record_exact_rate"] is None
    assert block["per_field_exact"] is None
    assert block["per_field_normalized"] is None
    assert block["valid_json_rate"] is None
    assert block["schema_valid_rate"] is None
    assert block["mean_attempts"] is None
    assert block["mean_latency_ms"] is None
    report = render_report(summary)
    assert "| todas | 0 | 1 | — | — | — | — | — |" in report


def test_extractor_exception_is_excluded_from_denominators() -> None:
    row = load_eval_texts()[0]
    failed = observe_extraction(row, "llm_scripted", _Boom())
    assert failed.error == "RuntimeError: boom"
    assert failed.record is None
    assert failed.latency_ms is None
    ok = observe_extraction(row, "regex", RegexExtractor())
    summary = _summary([failed, ok])
    failed_block = summary["by_system"]["llm_scripted"]["overall"]
    assert failed_block["errors"] == 1
    assert failed_block["scored"] == 0
    assert failed_block["valid_json_rate"] is None
    assert failed_block["mean_attempts"] is None
    ok_block = summary["by_system"]["regex"]["overall"]
    assert ok_block["errors"] == 0
    assert ok_block["scored"] == 1
    assert ok_block["schema_valid_rate"] == 1.0


def test_missing_script_is_an_unscored_error_for_the_llm_systems() -> None:
    row = {
        "id": "row-1",
        "category": "clean_template",
        "text": "Cliente: Ana Souza",
        "gold": {"customer_name": "Ana Souza"},
    }
    by_system = {
        item.system: item for item in run_evaluation([row], scenario="valid", repair_attempts=2)
    }
    assert by_system["regex"].error is None
    assert by_system["regex"].record is not None
    assert by_system["regex"].record.exact is True
    assert by_system["llm_scripted"].record is None
    assert by_system["llm_scripted"].error is not None
    assert "scripted_outputs" in by_system["llm_scripted"].error
    assert by_system["llm_scripted_repair"].record is None
    summary = _summary(
        [by_system["regex"], by_system["llm_scripted"], by_system["llm_scripted_repair"]]
    )
    llm = summary["by_system"]["llm_scripted"]["overall"]
    assert llm["errors"] == 1
    assert llm["scored"] == 0
    assert llm["record_exact_rate"] is None
    regex = summary["by_system"]["regex"]["by_category"]["clean_template"]
    assert regex["record_exact_rate"] == 1.0


def test_scenarios_replay_the_fixed_output_of_the_line() -> None:
    row = load_eval_texts()[0]
    scripted = row["scripted_outputs"]

    valid = _index(run_evaluation([row], scenario="valid", repair_attempts=3))
    assert valid["llm_scripted"].raw_output == scripted["valid"]
    assert valid["llm_scripted"].attempts == 1
    assert valid["llm_scripted"].record is not None
    assert valid["llm_scripted"].record.exact is True
    assert valid["llm_scripted_repair"].attempts == 1
    assert valid["llm_scripted_repair"].record is not None
    assert valid["llm_scripted_repair"].record.exact is True

    malformed = _index(run_evaluation([row], scenario="malformed", repair_attempts=3))
    assert malformed["llm_scripted"].raw_output == scripted["malformed"]
    assert malformed["llm_scripted"].json_valid is False
    assert malformed["llm_scripted"].attempts == 1
    assert malformed["llm_scripted"].error is None
    assert malformed["llm_scripted_repair"].raw_output == scripted["malformed"]
    assert malformed["llm_scripted_repair"].json_valid is False
    assert malformed["llm_scripted_repair"].attempts == 3
    assert malformed["llm_scripted_repair"].error is None

    partial = _index(run_evaluation([row], scenario="partially_wrong", repair_attempts=3))
    assert partial["llm_scripted"].raw_output == scripted["partially_wrong"]
    assert partial["llm_scripted"].schema_valid is True
    assert partial["llm_scripted"].record is not None
    assert partial["llm_scripted"].record.exact is False
    assert partial["llm_scripted"].attempts == 1
    assert partial["llm_scripted_repair"].attempts == 1

    mix = _index(run_evaluation([row], scenario="mix", repair_attempts=3))
    assert mix["llm_scripted"].raw_output == scripted["malformed"]
    assert mix["llm_scripted"].attempts == 1
    assert mix["llm_scripted_repair"].raw_output == scripted["valid"]
    assert mix["llm_scripted_repair"].attempts == 2
    assert mix["llm_scripted_repair"].record is not None
    assert mix["llm_scripted_repair"].record.exact is True
    assert mix["regex"].attempts == 1
    assert mix["regex"].schema_valid is True
    assert mix["regex"].error is None


def test_resolve_repair_attempts_prefers_an_explicit_budget() -> None:
    assert resolve_repair_attempts(None, {}) == 2
    assert resolve_repair_attempts(None, {"REPAIR_MAX_ATTEMPTS": "4"}) == 4
    assert resolve_repair_attempts(3, {"REPAIR_MAX_ATTEMPTS": "4"}) == 3
    with pytest.raises(ValueError, match=">= 1"):
        resolve_repair_attempts(0, {})


def _summary(observations: list[EvalObservation]) -> dict[str, object]:
    return build_summary(observations, scenario="valid", repair_attempts=2, dataset="memory")


def _index(observations: list[EvalObservation]) -> dict[str, EvalObservation]:
    return {item.system: item for item in observations}


def _scored(
    text_id: str,
    *,
    category: str = "clean_template",
    exact: bool,
    json_valid: bool = True,
    schema_valid: bool = True,
    attempts: int = 1,
    latency_ms: float = 1.0,
) -> EvalObservation:
    return EvalObservation(
        text_id=text_id,
        category=category,
        system="regex",
        error=None,
        json_valid=json_valid,
        schema_valid=schema_valid,
        attempts=attempts,
        latency_ms=latency_ms,
        record=RecordScore(
            exact=exact,
            fields=(FieldScore("city", exact, True),),
        ),
        predicted={"city": "Santos"},
        raw_output="{}",
        gold={"city": "Santos"},
    )


def _errored(text_id: str, *, category: str = "clean_template") -> EvalObservation:
    return EvalObservation(
        text_id=text_id,
        category=category,
        system="regex",
        error="RuntimeError: boom",
        json_valid=None,
        schema_valid=None,
        attempts=None,
        latency_ms=None,
        record=None,
        predicted=None,
        raw_output=None,
        gold={"city": "Santos"},
    )


class _Boom:
    def extract(self, text: str) -> object:
        raise RuntimeError("boom")
