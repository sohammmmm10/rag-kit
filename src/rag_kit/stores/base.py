"""Base class for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import EmbeddedChunk, RetrievedChunk


class BaseStore(ABC):
    """Abstract base class for all vector stores."""

    @abstractmethod
    def add(self, chunks: list[EmbeddedChunk]) -> int:
        """Add embedded chunks to the store. Returns number of chunks added."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Search for similar chunks and return them ranked by relevance."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks in the store."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all chunks from the store."""

    def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks by their IDs. Returns number deleted."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support deletion."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(count={self.count()})"
