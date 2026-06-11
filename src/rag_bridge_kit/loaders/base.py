"""Base class for document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Document


class BaseLoader(ABC):
    """Abstract base class for all document loaders."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Load and return a list of documents."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
