"""Fixture mínima para travar o baseline. O conjunto v1 entra por `load_eval_texts`."""

import json
from pathlib import Path
from typing import Any


def load_placeholder_fixture() -> list[dict[str, Any]]:
    """Carrega os oito textos sintéticos versionados com o pacote."""

    path = Path(__file__).with_name("placeholder.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("placeholder fixture must be a list")
    return payload
