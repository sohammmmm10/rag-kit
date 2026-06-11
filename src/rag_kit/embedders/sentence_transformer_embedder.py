"""SentenceTransformers local embedder (no API needed)."""

from __future__ import annotations

from ..errors import DependencyError, EmbedderError
from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Generate embeddings using a local SentenceTransformers model.

    Requires: ``pip install rag-kit[sentence-transformers]``

    Parameters
    ----------
    model_name : str
        HuggingFace model name (e.g. ``all-MiniLM-L6-v2``).
    device : str | None
        Torch device to use (``cpu``, ``cuda``, ``mps``). Auto-detected if None.
    batch_size : int
        Batch size for encoding.
    """

    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install with: pip install rag-kit[sentence-transformers]"
            ) from exc

        self._model_name = model_name
        self._batch_size = batch_size
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise EmbedderError(
                f"Failed to load SentenceTransformer model '{model_name}': {exc}"
            ) from exc
        self._dims = self._model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        return self._dims  # type: ignore[return-value]

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return [vec.tolist() for vec in embeddings]
        except Exception as exc:
            raise EmbedderError(
                f"SentenceTransformer encoding failed: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"SentenceTransformerEmbedder(model={self._model_name!r}, "
            f"dims={self._dims})"
        )
