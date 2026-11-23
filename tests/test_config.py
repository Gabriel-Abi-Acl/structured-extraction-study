"""Contrato da configuração: ambiente explícito, padrão seguro e exemplo sem segredo."""

from pathlib import Path

import pytest

from structured_extraction_study.config import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_blank_env_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-process-env")
    settings = load_settings(
        {
            "LLM_PROVIDER": "  ",
            "LLM_MODEL": "",
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "   ",
            "REPAIR_MAX_ATTEMPTS": "",
        }
    )
    assert settings.llm_provider == "scripted"
    assert settings.llm_model == ""
    assert settings.openai_api_key == ""
    assert settings.anthropic_api_key == ""
    assert settings.repair_max_attempts == 2
    overridden = load_settings({"REPAIR_MAX_ATTEMPTS": " 3 "})
    assert overridden.repair_max_attempts == 3
    assert overridden.openai_api_key == ""


def test_repair_budget_must_be_a_positive_integer() -> None:
    with pytest.raises(ValueError, match="integer"):
        load_settings({"REPAIR_MAX_ATTEMPTS": "duas"})
    with pytest.raises(ValueError, match=">= 1"):
        load_settings({"REPAIR_MAX_ATTEMPTS": "0"})


def test_env_example_keeps_every_value_empty() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    keys = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        assert separator == "="
        assert value.strip() == ""
        keys.append(key)
    assert keys == [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "REPAIR_MAX_ATTEMPTS",
    ]
    assert "sk-" not in "\n".join(lines)
