"""Estudo de extração estruturada em textos curtos em português.

Expõe o contrato mínimo do PR1: esquema, extratores, métricas e configuração.
Os SDKs de provedor não são importados por este módulo.
"""

from structured_extraction_study.config import Settings, load_settings
from structured_extraction_study.extractors.llm import LLMExtractor
from structured_extraction_study.extractors.protocol import Extractor
from structured_extraction_study.extractors.regex import RegexExtractor
from structured_extraction_study.llm.client import (
    AnthropicMessagesClient,
    OpenAIChatClient,
    ScriptedLLMClient,
    build_live_client,
)
from structured_extraction_study.metrics import (
    MetricsSummary,
    RecordScore,
    match_record,
    schema_valid_rate,
    summarize_record_matches,
    valid_json_rate,
)
from structured_extraction_study.result import ExtractionResult
from structured_extraction_study.schemas.order import ORDER_FIELDS, OrderRecord

__version__ = "0.1.0"

__all__ = [
    "ORDER_FIELDS",
    "AnthropicMessagesClient",
    "ExtractionResult",
    "Extractor",
    "LLMExtractor",
    "MetricsSummary",
    "OpenAIChatClient",
    "OrderRecord",
    "RecordScore",
    "RegexExtractor",
    "ScriptedLLMClient",
    "Settings",
    "build_live_client",
    "load_settings",
    "match_record",
    "schema_valid_rate",
    "summarize_record_matches",
    "valid_json_rate",
]
