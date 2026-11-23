"""Contrato do cliente: SDK real só na chamada ao vivo, e chave vazia não dispara import."""

import builtins
import sys

import pytest

from structured_extraction_study.config import load_settings
from structured_extraction_study.llm.client import (
    AnthropicMessagesClient,
    OpenAIChatClient,
    build_live_client,
)


def test_package_import_does_not_load_sdks() -> None:
    import structured_extraction_study as study

    assert study.__version__ == "0.1.0"
    assert "openai" not in sys.modules
    assert "anthropic" not in sys.modules


def test_build_live_client_does_not_import_the_sdk() -> None:
    settings = load_settings(
        {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-test",
            "OPENAI_API_KEY": "test-key",
        }
    )
    client = build_live_client(settings)
    assert isinstance(client, OpenAIChatClient)
    assert "openai" not in sys.modules
    with pytest.raises(ValueError, match="openai or anthropic"):
        build_live_client(load_settings({"LLM_PROVIDER": "scripted", "LLM_MODEL": "x"}))


def test_empty_key_fails_before_the_sdk_is_needed() -> None:
    openai_client = OpenAIChatClient(model="gpt-test", api_key="  ")
    anthropic_client = AnthropicMessagesClient(model="claude-test", api_key="")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is empty"):
        openai_client.complete([{"role": "user", "content": "oi"}])
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is empty"):
        anthropic_client.complete([{"role": "user", "content": "oi"}])
    assert "openai" not in sys.modules
    assert "anthropic" not in sys.modules
    with pytest.raises(ValueError, match="model is required"):
        OpenAIChatClient(model="  ", api_key="test-key")


def test_missing_extra_is_reported_on_the_first_live_call(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name in {"openai", "anthropic"} or name.startswith(("openai.", "anthropic.")):
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="openai extra is not installed"):
        OpenAIChatClient(model="gpt-test", api_key="test-key").complete(
            [{"role": "user", "content": "oi"}]
        )
    with pytest.raises(RuntimeError, match="anthropic extra is not installed"):
        AnthropicMessagesClient(model="claude-test", api_key="test-key").complete(
            [{"role": "user", "content": "oi"}]
        )
