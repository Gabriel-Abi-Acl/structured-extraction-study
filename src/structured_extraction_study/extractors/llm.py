"""Extrator de LLM com validação de esquema e reparo limitado por tentativas.

O orçamento inclui a primeira chamada. Valor errado que ainda valida não gera
reenvio: isso é erro de campo, medido à parte. O cliente é injetado, então o
teste usa o falso scriptado e a chamada real fica fora deste módulo.
"""

import json
import time
from collections.abc import Sequence

from structured_extraction_study.extractors.parsing import to_result
from structured_extraction_study.llm.client import ChatMessage, LLMClient
from structured_extraction_study.result import ExtractionResult
from structured_extraction_study.schemas.order import load_order_schema


class LLMExtractor:
    """Pede um objeto JSON e, se a validação falhar, reenvia o erro até o limite."""

    def __init__(self, client: LLMClient, *, max_attempts: int = 1) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._client = client
        self._max_attempts = max_attempts

    def extract(self, text: str) -> ExtractionResult:
        """Uma tentativa quando `max_attempts` é 1. Acima disso, o erro entra no diálogo."""

        started = time.perf_counter()
        messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": f"Text:\n{text}"},
        ]
        result: ExtractionResult | None = None
        for attempt in range(1, self._max_attempts + 1):
            raw_output = self._client.complete(messages)
            result = to_result(raw_output, attempts=attempt, latency_ms=0.0)
            if result.schema_valid or attempt == self._max_attempts:
                break
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": repair_instruction(result.errors)})
        if result is None:
            raise RuntimeError("LLMExtractor made no attempts")
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ExtractionResult(
            raw_output=result.raw_output,
            data=result.data,
            json_valid=result.json_valid,
            schema_valid=result.schema_valid,
            errors=result.errors,
            attempts=result.attempts,
            latency_ms=elapsed_ms,
        )


def system_prompt() -> str:
    """Instrui o modelo a devolver só o objeto do esquema, com nulo no campo ausente."""

    schema = json.dumps(load_order_schema(), ensure_ascii=False, indent=2)
    return (
        "Extract one JSON object from the user text.\n"
        "Use null when a field is absent. Do not invent values.\n"
        "Reply with the JSON object only, without markdown.\n"
        "The object must validate against this JSON Schema:\n"
        f"{schema}"
    )


def repair_instruction(errors: Sequence[str]) -> str:
    """Monta o reenvio com o texto exato dos erros de validação."""

    if errors:
        lines = "\n".join(f"- {error}" for error in errors)
    else:
        lines = "- The previous output was rejected."
    return (
        "The previous JSON failed validation. "
        "Return one JSON object and nothing else.\n"
        f"Validation errors:\n{lines}"
    )
