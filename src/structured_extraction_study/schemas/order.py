"""Pedido sintético: modelo Pydantic alinhado ao JSON Schema versionado.

Os dois aceitam as mesmas chaves. O arquivo JSON é o que entra no prompt,
para o contrato não depender da serialização do Pydantic.
"""

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaValidationError
from pydantic import BaseModel, ConfigDict, ValidationError

_SCHEMA_PATH = Path(__file__).with_name("order.schema.json")


class OrderRecord(BaseModel):
    """Registro com todas as chaves presentes. Ausência de dado é `null`, não omissão."""

    model_config = ConfigDict(extra="forbid")

    customer_name: str | None
    email: str | None
    phone: str | None
    document_id: str | None
    order_date: str | None
    amount_brl: str | None
    city: str | None
    item: str | None


ORDER_FIELDS: tuple[str, ...] = tuple(OrderRecord.model_fields)


def load_order_schema() -> dict[str, Any]:
    """Devolve uma cópia do JSON Schema de pedido."""

    payload = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("order schema must be a JSON object")
    return payload


def evaluate_order_dict(data: Any) -> tuple[bool, tuple[str, ...]]:
    """Valida um objeto já parseado. Erros do esquema vão para o prompt de reparo."""

    if not isinstance(data, dict):
        return False, ("Expected a JSON object",)
    schema_errors = _schema_errors(data)
    if schema_errors:
        return False, schema_errors
    model_errors = _model_errors(data)
    if model_errors:
        return False, model_errors
    return True, ()


def _schema_errors(data: dict[str, Any]) -> tuple[str, ...]:
    validator = Draft202012Validator(load_order_schema())
    found: list[SchemaValidationError] = sorted(
        validator.iter_errors(data),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.message),
    )
    messages: list[str] = []
    for error in found:
        location = ".".join(str(part) for part in error.absolute_path)
        if location:
            messages.append(f"{location}: {error.message}")
        else:
            messages.append(error.message)
    return tuple(messages)


def _model_errors(data: dict[str, Any]) -> tuple[str, ...]:
    try:
        OrderRecord.model_validate(data)
    except ValidationError as exc:
        messages: list[str] = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"])
            prefix = f"{location}: " if location else ""
            messages.append(f"{prefix}{error['msg']}")
        return tuple(messages)
    return ()
