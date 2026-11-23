"""Protocolo comum para o baseline e para o extrator de modelo."""

from typing import Protocol, runtime_checkable

from structured_extraction_study.result import ExtractionResult


@runtime_checkable
class Extractor(Protocol):
    """Uma chamada `extract` devolve o registro, a validade e o custo de tentativas."""

    def extract(self, text: str) -> ExtractionResult:
        """Extrai um registro do texto. Não levanta por JSON inválido."""
