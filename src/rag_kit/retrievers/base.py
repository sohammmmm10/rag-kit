"""Base class for retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import RetrievedChunk


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
