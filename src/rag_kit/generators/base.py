"""Base class for generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import QueryResult, RetrievedChunk


class BaseGenerator(ABC):
    """Abstract base class for all generators.

    A generator takes a query + retrieved context chunks and produces
    a final answer.
    """

    @abstractmethod
    def generate(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
    ) -> QueryResult:
        """Generate an answer from the query and retrieved context."""

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Helper to build a context string from retrieved chunks."""
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", chunk.metadata.get("filename", "unknown"))
            parts.append(f"[{i}] (source: {source}, score: {chunk.score})\n{chunk.content}")
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
