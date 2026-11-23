"""Contrato do dataset v1: 48 textos sintéticos, ouro ancorado e loader estrito."""

import json
import re
from pathlib import Path

import pytest

from structured_extraction_study.dataset import (
    EVAL_CATEGORIES,
    SCRIPTED_OUTPUT_KINDS,
    _spans,
    load_eval_texts,
)
from structured_extraction_study.extractors.parsing import to_result
from structured_extraction_study.normalize import normalize_currency, normalize_date
from structured_extraction_study.schemas.order import ORDER_FIELDS

_TEMPLATE_LABELS = (
    "Cliente:",
    "E-mail:",
    "Telefone:",
    "CPF:",
    "Data:",
    "Valor:",
    "Cidade:",
    "Item:",
)
_FREE_FORM_LABEL = re.compile(
    r"^(?:cliente|nome|e-?mail|telefone|cpf|data|valor|cidade|município|municipio|item|produto|assunto)\s*:",
    re.IGNORECASE,
)
_ABSENT_LABEL = {
    "customer_name": re.compile(r"^(?:cliente|nome)\s*:", re.IGNORECASE),
    "city": re.compile(r"^(?:cidade|município|municipio)\s*:", re.IGNORECASE),
    "item": re.compile(r"^(?:item|produto|assunto)\s*:", re.IGNORECASE),
}


def test_v1_dataset_is_balanced_and_labeled() -> None:
    rows = load_eval_texts()
    assert len(rows) == 48
    assert [row["id"] for row in rows] == [f"ev-{number:03d}" for number in range(1, 49)]
    counts = {category: 0 for category in EVAL_CATEGORIES}
    for row in rows:
        assert row["category"] in counts
        counts[row["category"]] += 1
        assert set(row["gold"]) == set(ORDER_FIELDS)
        assert set(row["scripted_outputs"]) == set(SCRIPTED_OUTPUT_KINDS)
        if row["category"] != "missing_fields":
            assert all(value is not None for value in row["gold"].values())
    assert counts == {category: 8 for category in EVAL_CATEGORIES}


def test_v1_values_are_synthetic_and_scripted_outputs_keep_their_roles() -> None:
    for row in rows():
        for span in _spans("email", row["text"]):
            assert span.casefold().endswith("@example.com")
        email = row["gold"]["email"]
        if email is not None:
            assert email.endswith("@example.com")
        document_id = row["gold"]["document_id"]
        if document_id is not None:
            assert re.fullmatch(r"\d{11}", document_id)
            assert document_id.startswith("390")
        phone = row["gold"]["phone"]
        if phone is not None and document_id is not None:
            assert phone != document_id
        assert "api_key" not in row["text"].casefold()
        assert "sk-" not in row["text"]
        valid = to_result(row["scripted_outputs"]["valid"], attempts=1, latency_ms=0.0)
        assert valid.schema_valid
        assert valid.data == row["gold"]
        malformed = to_result(row["scripted_outputs"]["malformed"], attempts=1, latency_ms=0.0)
        assert malformed.json_valid is False
        partial = to_result(row["scripted_outputs"]["partially_wrong"], attempts=1, latency_ms=0.0)
        assert partial.schema_valid
        assert partial.data != row["gold"]


def test_category_shapes_match_the_documented_split() -> None:
    grouped = {category: [] for category in EVAL_CATEGORIES}
    for row in rows():
        grouped[row["category"]].append(row)

    for row in grouped["clean_template"]:
        lines = row["text"].splitlines()
        assert len(lines) == len(_TEMPLATE_LABELS)
        for line, label in zip(lines, _TEMPLATE_LABELS, strict=True):
            assert line.startswith(label)
        assert row["gold"]["customer_name"] in row["text"]

    for row in grouped["free_form"]:
        assert not any(_FREE_FORM_LABEL.match(line) for line in row["text"].splitlines())

    styles: set[str] = set()
    for row in grouped["varied_formats"]:
        span = _spans("order_date", row["text"])[0]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", span):
            styles.add("iso")
        elif "/" in span:
            styles.add("slash")
        elif "." in span:
            styles.add("dot")
        elif "-" in span:
            styles.add("dash")
    assert styles == {"iso", "slash", "dot", "dash"}

    absent: set[str] = set()
    for row in grouped["missing_fields"]:
        gold = row["gold"]
        assert any(value is None for value in gold.values())
        for field, value in gold.items():
            if value is None:
                absent.add(field)
                label = _ABSENT_LABEL.get(field)
                if label is None:
                    assert _spans(field, row["text"]) == []
                else:
                    assert not any(label.match(line) for line in row["text"].splitlines())
    assert absent == set(ORDER_FIELDS)

    for row in grouped["distractors"]:
        assert _has_competing_date_or_amount(row["text"], row["gold"])

    assert any(
        row["gold"]["customer_name"] not in row["text"]
        or row["gold"]["city"] not in row["text"]
        or row["gold"]["item"] not in row["text"]
        for row in grouped["noisy"]
    )


def test_loader_accepts_values_derived_by_the_normalizers(tmp_path: Path) -> None:
    text = "\n".join(
        [
            "Cliente: Ana  Souza",
            "E-mail: Ana.Souza@Example.com",
            "Telefone: (11) 98800-1001",
            "CPF: 390.111.222-01",
            "Data: 15/03/2026",
            "Valor: R$ 1.250,00",
            "Cidade: Sao  Paulo",
            "Item: Teclado",
            "mecânico",
        ]
    )
    row = _row(
        text=text,
        gold=_gold(
            customer_name="Ana Souza",
            email="ana.souza@example.com",
            phone="11988001001",
            document_id="39011122201",
            order_date="2026-03-15",
            amount_brl="1250.00",
            city="Sao Paulo",
            item="Teclado mecânico",
        ),
    )
    loaded = load_eval_texts(_write(tmp_path, [row]))
    assert loaded[0]["gold"]["amount_brl"] == "1250.00"
    assert loaded[0]["gold"]["order_date"] == "2026-03-15"


def test_loader_rejects_values_the_normalizer_does_not_support(tmp_path: Path) -> None:
    ambiguous = _row(
        text="Valor: 1.250",
        gold=_gold(amount_brl="1250.00"),
    )
    extenso = _row(
        text="Data: 15 de março de 2026",
        gold=_gold(order_date="2026-03-15"),
    )
    accent = _row(
        text="Cidade: Sao Paulo",
        gold=_gold(city="São Paulo"),
    )
    literal = _row(
        text="Data: 15 de março de 2026",
        gold=_gold(order_date="15 de março de 2026"),
    )
    with pytest.raises(ValueError, match="not grounded"):
        load_eval_texts(_write(tmp_path, [ambiguous]))
    with pytest.raises(ValueError, match="not grounded"):
        load_eval_texts(_write(tmp_path, [extenso]))
    with pytest.raises(ValueError, match="not grounded"):
        load_eval_texts(_write(tmp_path, [accent]))
    loaded = load_eval_texts(_write(tmp_path, [literal]))
    assert loaded[0]["gold"]["order_date"] == "15 de março de 2026"


def test_loader_rejects_broken_rows(tmp_path: Path) -> None:
    missing_key = _row()
    del missing_key["text"]
    extra_key = _row()
    extra_key["note"] = "x"
    short_gold = _gold(customer_name="Ana Souza")
    del short_gold["item"]
    extra_gold = _gold(customer_name="Ana Souza")
    extra_gold["extra"] = None
    blank_gold = _gold(customer_name="  ")
    numeric_gold = _gold(customer_name="Ana Souza")
    numeric_gold["amount_brl"] = 10
    ungrounded = _row(
        text="Cliente: Ana Souza",
        gold=_gold(customer_name="Ana Souza", city="Recife"),
    )
    bad_script = _row()
    bad_script["scripted_outputs"] = {
        "valid": json.dumps(_gold(customer_name="Outra Pessoa"), ensure_ascii=False),
        "malformed": "{",
        "partially_wrong": json.dumps(_gold(customer_name="Ana Souza"), ensure_ascii=False),
    }
    same_partial = _row()
    same_partial["scripted_outputs"] = {
        "valid": json.dumps(same_partial["gold"], ensure_ascii=False),
        "malformed": "ainda-json",
        "partially_wrong": json.dumps(same_partial["gold"], ensure_ascii=False),
    }
    json_marked_malformed = _row()
    json_marked_malformed["scripted_outputs"] = {
        "valid": json.dumps(json_marked_malformed["gold"], ensure_ascii=False),
        "malformed": json.dumps(json_marked_malformed["gold"], ensure_ascii=False),
        "partially_wrong": json.dumps(_gold(city="Recife"), ensure_ascii=False),
    }
    cases = [
        (_row(category="template"), "unknown category"),
        (_row(id=" ev-001"), "id must be"),
        (_row(text="  "), "text must be"),
        (missing_key, "missing keys"),
        (extra_key, "unknown keys"),
        (_row(gold=short_gold), "gold keys"),
        (_row(gold=extra_gold), "gold keys"),
        (_row(gold=blank_gold), "non-blank"),
        (_row(gold=numeric_gold), "non-blank string"),
        (ungrounded, "not grounded"),
        (bad_script, "must be the gold JSON"),
        (same_partial, "must differ from gold"),
        (json_marked_malformed, "must not be JSON"),
    ]
    for row, fragment in cases:
        with pytest.raises(ValueError, match=fragment):
            load_eval_texts(_write(tmp_path, [row]))


def test_loader_rejects_duplicate_ids_blank_lines_and_bad_json(tmp_path: Path) -> None:
    path = _write(tmp_path, [_row(), _row(id="ev-002")])
    assert len(load_eval_texts(path)) == 2
    with pytest.raises(ValueError, match="duplicate id"):
        load_eval_texts(_write(tmp_path, [_row(), _row()]))
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank line"):
        load_eval_texts(path)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_eval_texts(path)
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_eval_texts(path)
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_eval_texts(path)
    path.write_text("\ufeff" + json.dumps(_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="BOM"):
        load_eval_texts(path)


def test_scripted_outputs_may_be_omitted_and_nulls_need_no_span(tmp_path: Path) -> None:
    loaded = load_eval_texts(_write(tmp_path, [_row()]))
    assert "scripted_outputs" not in loaded[0]
    empty = load_eval_texts(_write(tmp_path, [_row(text="sem campos neste bilhete", gold=_gold())]))
    assert all(value is None for value in empty[0]["gold"].values())


def rows() -> list[dict]:
    return load_eval_texts()


def _has_competing_date_or_amount(text: str, gold: dict[str, str | None]) -> bool:
    dates = {normalize_date(span) for span in _spans("order_date", text)}
    amounts = {normalize_currency(span) for span in _spans("amount_brl", text)}
    gold_date = normalize_date(gold["order_date"])
    gold_amount = normalize_currency(gold["amount_brl"])
    date_conflict = gold_date in dates and any(item != gold_date for item in dates)
    amount_conflict = gold_amount in amounts and any(item != gold_amount for item in amounts)
    return date_conflict or amount_conflict


def _gold(**overrides: object) -> dict[str, object]:
    gold: dict[str, object] = {field: None for field in ORDER_FIELDS}
    gold.update(overrides)
    return gold


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "ev-001",
        "category": "clean_template",
        "text": "Cliente: Ana Souza\nE-mail: ana.souza@example.com",
        "gold": _gold(customer_name="Ana Souza", email="ana.souza@example.com"),
    }
    row.update(overrides)
    return row


def _write(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "texts.jsonl"
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
