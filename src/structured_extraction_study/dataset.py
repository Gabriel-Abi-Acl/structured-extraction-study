"""Leitura do dataset rotulado v1 (`data/eval/texts.v1.jsonl`).

Recusa arquivo que quebre o contrato: id repetido, categoria fora do
conjunto, ouro inválido ou valor que o texto não sustenta.
"""

import json
import re
from pathlib import Path
from typing import Any

from structured_extraction_study.extractors.parsing import strip_json_fence
from structured_extraction_study.normalize import FIELD_NORMALIZERS, normalize_text
from structured_extraction_study.schemas.order import ORDER_FIELDS, evaluate_order_dict

EVAL_CATEGORIES: tuple[str, ...] = (
    "clean_template",
    "free_form",
    "varied_formats",
    "missing_fields",
    "distractors",
    "noisy",
)

SCRIPTED_OUTPUT_KINDS: tuple[str, ...] = ("valid", "malformed", "partially_wrong")

_TEXT_FIELDS = frozenset({"customer_name", "city", "item"})
_ROW_KEYS = frozenset({"id", "category", "text", "gold", "scripted_outputs"})
_REQUIRED_KEYS = frozenset({"id", "category", "text", "gold"})

# Barra fica de fora: uma data com "/" não pode ser lida como telefone.
_PHONE_CANDIDATE = re.compile(r"(?<!\d)\+?(?:\d[\s().-]*){9,12}\d(?!\d)")
_DOCUMENT_CANDIDATE = re.compile(r"(?<!\d)(?:\d{3}(?:[.\s]\d{3}){2}[-\s]\d{2}|\d{11})(?!\d)")
_DATE_CANDIDATE = re.compile(r"(?<!\d)(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{4})(?!\d)")
_EMAIL_CANDIDATE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_MASKED_DOCUMENT = re.compile(r"\d{3}(?:[.\s]\d{3}){2}[-\s]\d{2}")
_CURRENCY_CANDIDATES = (
    re.compile(r"(?i)(?<!\w)R\$\s*\d{1,3}(?:\.\d{3})*,\d{2}"),
    re.compile(r"(?i)(?<!\w)R\$\s*\d+,\d{2}"),
    re.compile(r"(?i)(?<!\w)R\$\s*\d+(?!\d)"),
    re.compile(r"(?<!\d)\d{1,3}(?:\.\d{3})+,\d{2}(?!\d)"),
    re.compile(r"(?<!\d)\d+,\d{2}(?!\d)"),
    re.compile(r"(?<!\d)\d+\.\d{2}(?!\d)"),
)
_SPAN_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "email": (_EMAIL_CANDIDATE,),
    "phone": (_PHONE_CANDIDATE,),
    "document_id": (_DOCUMENT_CANDIDATE,),
    "order_date": (_DATE_CANDIDATE,),
    "amount_brl": _CURRENCY_CANDIDATES,
}


def default_eval_texts_path() -> Path:
    """Caminho de `texts.v1.jsonl` a partir da raiz do repositório."""

    return _repo_root() / "data" / "eval" / "texts.v1.jsonl"


def load_eval_texts(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Carrega o JSONL e valida cada linha antes de devolver.

    O valor rotulado precisa aparecer no texto ou sair de um trecho pelo
    normalizador daquele campo. Saída scriptada, quando vem, segue o cliente offline.
    """

    source = Path(path) if path is not None else default_eval_texts_path()
    try:
        raw_text = source.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"eval dataset not found: {source}") from exc
    if raw_text.startswith("\ufeff"):
        raise ValueError("eval dataset must be UTF-8 without a BOM")
    lines = raw_text.splitlines()
    if not lines:
        raise ValueError("eval dataset is empty")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if line.strip() == "":
            raise ValueError(f"line {number}: blank line")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {number}: invalid JSON") from exc
        row = _validate_row(payload, number)
        if row["id"] in seen:
            raise ValueError(f"line {number}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        rows.append(row)
    return rows


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        marker = parent / "pyproject.toml"
        package = parent / "src" / "structured_extraction_study"
        if marker.is_file() and package.is_dir():
            return parent
    raise FileNotFoundError("repository root not found")


def _validate_row(payload: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"line {line_number}: expected a JSON object")
    unknown = set(payload) - _ROW_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"line {line_number}: unknown keys: {names}")
    missing = _REQUIRED_KEYS - set(payload)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"line {line_number}: missing keys: {names}")

    row_id = payload["id"]
    if not isinstance(row_id, str) or row_id.strip() == "" or row_id != row_id.strip():
        raise ValueError(f"line {line_number}: id must be a non-blank string")
    category = payload["category"]
    if category not in EVAL_CATEGORIES:
        raise ValueError(f"line {line_number}: unknown category {category!r}")
    text = payload["text"]
    if not isinstance(text, str) or text.strip() == "":
        raise ValueError(f"line {line_number}: text must be a non-blank string")
    gold = _validate_gold(payload["gold"], line_number)
    for field, value in gold.items():
        if value is None:
            continue
        if not _is_grounded(field, value, text):
            raise ValueError(f"line {line_number}: gold.{field} is not grounded in the text")
    row = {"id": row_id, "category": category, "text": text, "gold": gold}
    if "scripted_outputs" in payload:
        row["scripted_outputs"] = _validate_scripted(payload["scripted_outputs"], gold, line_number)
    return row


def _validate_gold(gold: Any, line_number: int) -> dict[str, str | None]:
    if not isinstance(gold, dict):
        raise ValueError(f"line {line_number}: gold must be an object")
    if set(gold) != set(ORDER_FIELDS):
        raise ValueError(f"line {line_number}: gold keys must match OrderRecord")
    for field in ORDER_FIELDS:
        value = gold[field]
        if value is None:
            continue
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError(f"line {line_number}: gold.{field} must be a non-blank string or null")
    valid, errors = evaluate_order_dict(gold)
    if not valid:
        detail = "; ".join(errors) if errors else "schema rejected the object"
        raise ValueError(f"line {line_number}: gold failed validation: {detail}")
    return {field: gold[field] for field in ORDER_FIELDS}


def _validate_scripted(
    scripted: Any,
    gold: dict[str, str | None],
    line_number: int,
) -> dict[str, str]:
    if not isinstance(scripted, dict):
        raise ValueError(f"line {line_number}: scripted_outputs must be an object")
    if set(scripted) != set(SCRIPTED_OUTPUT_KINDS):
        raise ValueError(
            f"line {line_number}: scripted_outputs keys must be {', '.join(SCRIPTED_OUTPUT_KINDS)}"
        )
    checked: dict[str, str] = {}
    for kind in SCRIPTED_OUTPUT_KINDS:
        raw_output = scripted[kind]
        if not isinstance(raw_output, str) or raw_output.strip() == "":
            raise ValueError(f"line {line_number}: scripted_outputs.{kind} must be a string")
        checked[kind] = raw_output
    _check_scripted_roles(checked, gold, line_number)
    return checked


def _check_scripted_roles(
    scripted: dict[str, str],
    gold: dict[str, str | None],
    line_number: int,
) -> None:
    # O extrator tira uma cerca antes do json.loads. A checagem usa a mesma régua.
    valid_parsed, valid_ok = _parse_scripted(scripted["valid"])
    if not valid_ok or valid_parsed != gold or not evaluate_order_dict(valid_parsed)[0]:
        raise ValueError(f"line {line_number}: scripted_outputs.valid must be the gold JSON")
    _, malformed_ok = _parse_scripted(scripted["malformed"])
    if malformed_ok:
        raise ValueError(f"line {line_number}: scripted_outputs.malformed must not be JSON")
    partial_parsed, partial_ok = _parse_scripted(scripted["partially_wrong"])
    partial_valid = partial_ok and evaluate_order_dict(partial_parsed)[0]
    if not partial_valid:
        raise ValueError(f"line {line_number}: scripted_outputs.partially_wrong must validate")
    if partial_parsed == gold:
        raise ValueError(
            f"line {line_number}: scripted_outputs.partially_wrong must differ from gold"
        )


def _parse_scripted(raw_output: str) -> tuple[Any, bool]:
    try:
        return json.loads(strip_json_fence(raw_output)), True
    except json.JSONDecodeError:
        return None, False


def _is_grounded(field: str, value: str, text: str) -> bool:
    """O rótulo vale se está escrito ou se um trecho normaliza para o mesmo valor."""

    if _literal_in_text(value, text):
        return True
    normalizer = FIELD_NORMALIZERS[field]
    target = normalizer(value)
    if field in _TEXT_FIELDS:
        return _normalized_text_in(target, text)
    if not isinstance(target, str) or target.strip() == "":
        return False
    for span in _spans(field, text):
        if normalizer(span) == target:
            return True
    return False


def _literal_in_text(value: str, text: str) -> bool:
    needle = value.strip()
    if needle == "":
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", text) is not None


def _normalized_text_in(target: str | None, text: str) -> bool:
    if not target:
        return False
    haystack = normalize_text(text)
    if not haystack:
        return False
    return re.search(rf"(?<!\w){re.escape(target)}(?!\w)", haystack) is not None


def _spans(field: str, text: str) -> list[str]:
    patterns = _SPAN_PATTERNS.get(field)
    if patterns is None:
        return []
    kept = _collect_spans(patterns, text)
    if field == "phone":
        # CPF mascarado tem 11 dígitos, mas não é telefone.
        kept = [item for item in kept if _MASKED_DOCUMENT.fullmatch(item[2]) is None]
    if field == "amount_brl":
        # "08.07" de uma data e "55.61" de um telefone não são moeda.
        blocked = _collect_spans((_DATE_CANDIDATE,), text)
        blocked.extend(_collect_spans((_PHONE_CANDIDATE,), text))
        kept = [item for item in kept if not _overlaps(item[0], item[1], blocked)]
    return [span for _, _, span in kept]


def _overlaps(start: int, end: int, others: list[tuple[int, int, str]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end, _ in others)


def _collect_spans(
    patterns: tuple[re.Pattern[str], ...],
    text: str,
) -> list[tuple[int, int, str]]:
    # O trecho mais longo fica. "R$ 1.250,00" não conta também como "250,00".
    found: list[tuple[int, int, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            found.append((match.start(), match.end(), match.group(0)))
    found.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    kept: list[tuple[int, int, str]] = []
    for start, end, span in found:
        overlaps = any(start < kept_end and end > kept_start for kept_start, kept_end, _ in kept)
        if overlaps:
            continue
        kept.append((start, end, span))
    kept.sort(key=lambda item: item[0])
    return kept
