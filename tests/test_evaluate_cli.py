"""Contrato do comando: dataset real em diretório temporário e porta ao vivo fechada."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from structured_extraction_study.dataset import (
    EVAL_CATEGORIES,
    default_eval_texts_path,
    load_eval_texts,
)
from structured_extraction_study.evaluation import (
    LIVE_UNAVAILABLE_MESSAGE,
    MISSING_KEY_MESSAGE,
    main,
)
from structured_extraction_study.schemas.order import ORDER_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def test_cli_evaluates_the_real_dataset_offline(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    code = main(
        [
            "--output-dir",
            str(output_dir),
            "--repair-attempts",
            "2",
        ]
    )
    assert code == 0
    records = [
        json.loads(line)
        for line in (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    rows = load_eval_texts()
    by_id = {row["id"]: row for row in rows}

    assert summary["dataset"] == "data/eval/texts.v1.jsonl"
    assert summary["scenario"] == "mix"
    assert summary["repair_max_attempts"] == 2
    assert summary["pipeline_check"] is True
    assert summary["latency_timing"] == "offline_wall_clock"
    assert summary["text_count"] == 48
    assert summary["systems"] == ["regex", "llm_scripted", "llm_scripted_repair"]
    assert "checagem de pipeline, não resultado de modelo" in summary["note"]
    assert summary["note"] in report
    assert "`offline_wall_clock`" in report
    assert len(records) == 144

    for record in records:
        assert record["latency_timing"] == "offline_wall_clock"
        assert record["execution_error"] is None
        assert record["latency_ms"] >= 0
        source = by_id[record["id"]]
        assert record["category"] == source["category"]
        assert record["gold"] == source["gold"]
        if record["system"] == "llm_scripted":
            assert record["raw_output"] == source["scripted_outputs"]["malformed"]
            assert record["json_valid"] is False
            assert record["schema_valid"] is False
            assert record["attempts"] == 1
            assert record["record_exact"] is False
            assert record["predicted"] is None
            assert all(flag is False for flag in record["per_field_exact"].values())
        elif record["system"] == "llm_scripted_repair":
            assert record["raw_output"] == source["scripted_outputs"]["valid"]
            assert record["predicted"] == source["gold"]
            assert record["json_valid"] is True
            assert record["schema_valid"] is True
            assert record["attempts"] == 2
            assert record["record_exact"] is True
            assert all(flag is True for flag in record["per_field_exact"].values())
        else:
            assert record["system"] == "regex"
            assert record["attempts"] == 1
            assert record["json_valid"] is True
            assert record["schema_valid"] is True

    for system in summary["systems"]:
        overall = summary["by_system"][system]["overall"]
        assert overall["scored"] == 48
        assert overall["errors"] == 0
        assert overall["latency_timing"] == "offline_wall_clock"
        assert overall["mean_latency_ms"] >= 0
        assert set(overall["per_field_exact"]) == set(ORDER_FIELDS)
        categories = summary["by_system"][system]["by_category"]
        assert list(categories) == list(EVAL_CATEGORIES)
        for category in EVAL_CATEGORIES:
            assert categories[category]["scored"] == 8
            assert categories[category]["errors"] == 0
            assert category in report
        assert f"## {system}" in report

    scripted = summary["by_system"]["llm_scripted"]["overall"]
    repaired = summary["by_system"]["llm_scripted_repair"]["overall"]
    regex = summary["by_system"]["regex"]["overall"]
    assert scripted["valid_json_rate"] == 0.0
    assert scripted["schema_valid_rate"] == 0.0
    assert scripted["record_exact_rate"] == 0.0
    assert scripted["mean_attempts"] == 1.0
    assert repaired["valid_json_rate"] == 1.0
    assert repaired["schema_valid_rate"] == 1.0
    assert repaired["record_exact_rate"] == 1.0
    assert repaired["mean_attempts"] == 2.0
    assert regex["valid_json_rate"] == 1.0
    assert regex["schema_valid_rate"] == 1.0
    assert regex["mean_attempts"] == 1.0
    assert any(
        regex["per_field_normalized"][field] > regex["per_field_exact"][field]
        for field in ORDER_FIELDS
    )
    repair_section = report.split("## llm_scripted_repair", 1)[1]
    assert "| todas | 48 | 0 | 1.000 | 1.000 | 1.000 | 2.000 |" in repair_section
    scripted_section = report.split("## llm_scripted", 1)[1].split("\n## ", 1)[0]
    assert "| todas | 48 | 0 | 0.000 | 0.000 | 0.000 | 1.000 |" in scripted_section
    blob = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            output_dir / "records.jsonl",
            output_dir / "summary.json",
            output_dir / "report.md",
        )
    )
    assert "sk-" not in blob
    assert "OPENAI_API_KEY" not in blob
    assert "api_key" not in blob.casefold()


def test_cli_repair_budget_defaults_to_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REPAIR_MAX_ATTEMPTS", raising=False)
    dataset = tmp_path / "one.jsonl"
    line = default_eval_texts_path().read_text(encoding="utf-8").splitlines()[0]
    dataset.write_text(line + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    code = main(
        [
            "--output-dir",
            str(output_dir),
            "--dataset",
            str(dataset),
            "--scenario",
            "malformed",
        ]
    )
    assert code == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["repair_max_attempts"] == 2
    assert summary["dataset"] == str(dataset)
    assert summary["by_system"]["llm_scripted_repair"]["overall"]["mean_attempts"] == 2.0


def test_repair_attempts_below_one_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--output-dir", str(tmp_path / "out"), "--repair-attempts", "0"])
    assert code == 2
    assert "repair_attempts" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_allow_live_without_api_key_exits_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = main(["--allow-live", "--output-dir", str(tmp_path / "out")])
    assert code == 2
    assert capsys.readouterr().err.strip() == MISSING_KEY_MESSAGE
    assert not (tmp_path / "out").exists()


def test_allow_live_treats_a_blank_key_as_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    code = main(["--allow-live", "--output-dir", str(tmp_path / "out")])
    assert code == 2
    assert capsys.readouterr().err.strip() == MISSING_KEY_MESSAGE
    assert not (tmp_path / "out").exists()


def test_allow_live_with_a_key_does_not_call_the_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    before = _sdk_modules()

    def refuse_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network call")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket, "create_connection", refuse_network)
    code = main(["--allow-live", "--output-dir", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.strip() == LIVE_UNAVAILABLE_MESSAGE
    assert captured.out == ""
    assert not (tmp_path / "out").exists()
    assert _sdk_modules() == before


def test_console_script_help_and_live_gate(tmp_path: Path) -> None:
    help_run = _run_script("--help")
    assert help_run.returncode == 0
    assert "--allow-live" in help_run.stdout
    assert "partially_wrong" in help_run.stdout
    assert "mix" in help_run.stdout

    gate = _run_script("--allow-live", "--output-dir", str(tmp_path / "out"))
    assert gate.returncode == 2
    assert "API key" in gate.stderr
    assert "No request was sent" in gate.stderr
    assert not (tmp_path / "out").exists()


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "LLM_MODEL"):
        env.pop(key, None)
    return subprocess.run(
        ["uv", "run", "--no-sync", "evaluate-extraction", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _sdk_modules() -> set[str]:
    return {name for name in sys.modules if name.split(".", 1)[0] in {"openai", "anthropic"}}
