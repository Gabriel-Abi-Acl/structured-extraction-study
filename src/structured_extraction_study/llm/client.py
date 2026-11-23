"""Cliente agnóstico de provedor, com falso scriptado para a suíte offline.

OpenAI e Anthropic são extras opcionais. O import acontece dentro de `complete`,
depois da checagem de chave vazia, para um teste não abrir rede nem exigir SDK.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from structured_extraction_study.config import Settings

ChatMessage = dict[str, str]


class LLMClient(Protocol):
    """Devolve texto bruto. Quem valida o JSON é o extrator, não o cliente."""

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Uma conclusão de chat. Pode levantar se o script ou a credencial acabar."""


class ScriptedLLMClient:
    """Reproduz saídas em ordem. Serve para JSON válido, malformado e parcialmente errado."""

    def __init__(self, outputs: Sequence[str]) -> None:
        copied: list[str] = []
        for item in outputs:
            if not isinstance(item, str):
                raise TypeError("scripted outputs must be strings")
            copied.append(item)
        self._outputs = copied
        self.calls: list[list[ChatMessage]] = []

    @property
    def outputs_remaining(self) -> int:
        """Quantas respostas scriptadas ainda não foram consumidas."""

        return len(self._outputs)

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Consome a próxima saída e guarda uma cópia das mensagens recebidas."""

        self.calls.append(
            [{key: str(value) for key, value in message.items()} for message in messages]
        )
        if not self._outputs:
            raise RuntimeError("ScriptedLLMClient ran out of scripted outputs")
        return self._outputs.pop(0)


class OpenAIChatClient:
    """Chat Completions. O pacote `openai` é importado na primeira conclusão."""

    def __init__(self, *, model: str, api_key: str) -> None:
        if not model.strip():
            raise ValueError("model is required")
        self._model = model.strip()
        self._api_key = api_key
        self._sdk: Any = None

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Chama o SDK só com chave não vazia. Conteúdo vazio é erro, não JSON."""

        key = self._api_key.strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is empty")
        if self._sdk is None:
            self._sdk = _import_openai()(api_key=key)
        response = self._sdk.chat.completions.create(
            model=self._model,
            messages=[dict(message) for message in messages],
            temperature=0,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or content.strip() == "":
            raise RuntimeError("Empty completion")
        return content


class AnthropicMessagesClient:
    """Messages API. O pacote `anthropic` é importado na primeira conclusão."""

    def __init__(self, *, model: str, api_key: str) -> None:
        if not model.strip():
            raise ValueError("model is required")
        self._model = model.strip()
        self._api_key = api_key
        self._sdk: Any = None

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Separa o papel system, exigido pela API, e junta os blocos de texto."""

        key = self._api_key.strip()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is empty")
        if self._sdk is None:
            self._sdk = _import_anthropic()(api_key=key)
        system_parts: list[str] = []
        chat: list[ChatMessage] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                system_parts.append(content)
                continue
            if role not in {"user", "assistant"}:
                raise RuntimeError(f"Unsupported role: {role}")
            chat.append({"role": role, "content": content})
        if not chat:
            raise RuntimeError("No user or assistant messages")
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": chat,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        response = self._sdk.messages.create(**kwargs)
        texts = [block.text for block in response.content if getattr(block, "text", None)]
        if not texts:
            raise RuntimeError("Empty completion")
        return "".join(texts)


def build_live_client(settings: Settings) -> LLMClient:
    """Monta o cliente do provedor nomeado em `LLM_PROVIDER`. Não chama a API."""

    provider = settings.llm_provider.strip().lower()
    if provider == "openai":
        return OpenAIChatClient(model=settings.llm_model, api_key=settings.openai_api_key)
    if provider == "anthropic":
        return AnthropicMessagesClient(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
        )
    raise ValueError("LLM_PROVIDER must be openai or anthropic to build a live client")


def _import_openai() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai extra is not installed") from exc
    return OpenAI


def _import_anthropic() -> Any:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("The anthropic extra is not installed") from exc
    return Anthropic
