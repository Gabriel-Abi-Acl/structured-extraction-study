"""Resultado único dos extratores, para métricas comparáveis entre regras e LLM."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionResult:
    """Saída de uma extração, incluindo tentativas e o tempo medido no extrator.

    `raw_output` é o texto produzido (JSON do baseline ou texto do modelo).
    `data` permanece disponível quando o JSON é um objeto, mesmo se o esquema falhar,
    para a métrica de campo não depender da validade.
    """

    raw_output: str
    data: dict[str, object] | None
    json_valid: bool
    schema_valid: bool
    errors: tuple[str, ...]
    attempts: int
    latency_ms: float
