"""OpenAI-based RAG answer generator."""

from __future__ import annotations

from ..errors import DependencyError, GeneratorError
from ..models import QueryResult, RetrievedChunk
from .base import BaseGenerator

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based on the "
    "provided context. If the context does not contain enough information, "
    "say so clearly. Cite the source numbers [1], [2], etc. when referencing "
    "context passages."
)

_DEFAULT_USER_TEMPLATE = (
    "Context:\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"
)


class OpenAIGenerator(BaseGenerator):
    """Generate answers using OpenAI's chat completion API.

    Requires: ``pip install rag-bridge-kit[openai]``

    Parameters
    ----------
    api_key : str
        OpenAI API key.
    model : str
        Chat model to use.
    system_prompt : str | None
        Custom system prompt. Uses a sensible default if None.
    user_template : str | None
        Custom user message template. Must include ``{context}`` and ``{query}``.
    temperature : float
        Sampling temperature.
    max_tokens : int | None
        Maximum tokens in the response.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        system_prompt: str | None = None,
        user_template: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> None:
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyError(
                "openai is required for OpenAIGenerator. "
                "Install with: pip install rag-bridge-kit[openai]"
            ) from exc

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._user_template = user_template or _DEFAULT_USER_TEMPLATE
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
    ) -> QueryResult:
        context = self._build_context(context_chunks)
        user_message = self._user_template.format(context=context, query=query)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            kwargs: dict = {
                "model": self._model,
                "messages": messages,
                "temperature": self._temperature,
            }
            if self._max_tokens is not None:
                kwargs["max_tokens"] = self._max_tokens

            response = self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            answer = choice.message.content or ""

            return QueryResult(
                answer=answer.strip(),
                query=query,
                retrieved_chunks=context_chunks,
                model=self._model,
                provider="openai",
                tokens_input=getattr(response.usage, "prompt_tokens", None),
                tokens_output=getattr(response.usage, "completion_tokens", None),
                raw=response,
            )
        except Exception as exc:
            raise GeneratorError(
                f"OpenAI generation failed: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return f"OpenAIGenerator(model={self._model!r})"
