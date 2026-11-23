"""Configuração por variáveis de ambiente, sem leitura de arquivo `.env`.

Chaves ficam só na memória do processo. Este módulo não registra o valor delas.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Parâmetros do experimento. Strings vazias significam 'não configurado'."""

    llm_provider: str
    llm_model: str
    openai_api_key: str
    anthropic_api_key: str
    repair_max_attempts: int


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Lê o ambiente informado ou `os.environ` quando `env` é omitido.

    Valor ausente ou em branco cai no padrão. O orçamento de reparo precisa ser
    um inteiro maior ou igual a 1, porque a primeira tentativa entra na conta.
    """

    source = os.environ if env is None else env
    raw_attempts = _setting(source, "REPAIR_MAX_ATTEMPTS", "2")
    try:
        attempts = int(raw_attempts)
    except ValueError as exc:
        raise ValueError("REPAIR_MAX_ATTEMPTS must be an integer") from exc
    if attempts < 1:
        raise ValueError("REPAIR_MAX_ATTEMPTS must be >= 1")
    return Settings(
        llm_provider=_setting(source, "LLM_PROVIDER", "scripted"),
        llm_model=_setting(source, "LLM_MODEL", ""),
        openai_api_key=_setting(source, "OPENAI_API_KEY", ""),
        anthropic_api_key=_setting(source, "ANTHROPIC_API_KEY", ""),
        repair_max_attempts=attempts,
    )


def _setting(source: Mapping[str, str], key: str, default: str) -> str:
    value = source.get(key)
    if value is None:
        return default
    stripped = value.strip()
    if stripped == "":
        return default
    return stripped
