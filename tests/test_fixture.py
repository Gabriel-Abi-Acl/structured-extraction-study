"""Contrato da fixture placeholder: oito textos sintéticos que validam no esquema."""

import re

from jsonschema import Draft202012Validator

from structured_extraction_study.fixtures import load_placeholder_fixture
from structured_extraction_study.schemas.order import (
    ORDER_FIELDS,
    OrderRecord,
    evaluate_order_dict,
    load_order_schema,
)


def test_schema_and_model_share_fields_and_accept_nulls() -> None:
    schema = load_order_schema()
    Draft202012Validator.check_schema(schema)
    assert set(schema["required"]) == set(ORDER_FIELDS)
    assert set(schema["properties"]) == set(ORDER_FIELDS)
    payload = {field: None for field in ORDER_FIELDS}
    payload["customer_name"] = "Ana Souza"
    assert evaluate_order_dict(payload) == (True, ())
    OrderRecord.model_validate(payload)
    missing = dict(payload)
    del missing["item"]
    valid, errors = evaluate_order_dict(missing)
    assert valid is False
    assert any("item" in error for error in errors)
    extra = dict(payload)
    extra["extra"] = 1
    assert evaluate_order_dict(extra)[0] is False
    assert evaluate_order_dict([])[0] is False


def test_placeholder_has_eight_labeled_synthetic_records() -> None:
    rows = load_placeholder_fixture()
    assert len(rows) == 8
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    for row in rows:
        assert row["text"].strip()
        assert set(row["gold"]) == set(ORDER_FIELDS)
        assert evaluate_order_dict(row["gold"])[0] is True
        document_id = row["gold"]["document_id"]
        assert document_id is None or re.fullmatch(r"\d{11}", document_id)
        emails = re.findall(r"[\w.+-]+@[\w.-]+", row["text"])
        assert all(email.endswith("@example.com") for email in emails)
