"""Base class for chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for all text chunkers."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""

    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        """Split multiple documents into chunks."""
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
