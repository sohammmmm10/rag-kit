"""Base class for embedders."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for all embedding providers."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of texts."""

    def embed_one(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text."""
        results = self.embed([text])
        return results[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dimensions={self.dimensions})"
