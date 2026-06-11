"""OpenAI embeddings provider."""

from __future__ import annotations

from ..errors import DependencyError, EmbedderError
from .base import BaseEmbedder

# Model name -> dimensions mapping for known models
_KNOWN_MODELS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    """Generate embeddings using OpenAI's API.

    Requires: ``pip install rag-kit[openai]``

    Parameters
    ----------
    api_key : str
        OpenAI API key.
    model : str
        Embedding model name.
    batch_size : int
        Maximum texts per API call.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
    ) -> None:
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyError(
                "openai is required for OpenAIEmbedder. "
                "Install with: pip install rag-kit[openai]"
            ) from exc

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self._dims = _KNOWN_MODELS.get(model, 1536)

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._model,
                )
                for item in response.data:
                    all_vectors.append(item.embedding)
            except Exception as exc:
                raise EmbedderError(
                    f"OpenAI embedding failed for batch starting at index {i}: {exc}"
                ) from exc

        return all_vectors

    def __repr__(self) -> str:
        return f"OpenAIEmbedder(model={self._model!r}, dims={self._dims})"
