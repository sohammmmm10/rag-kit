"""Similarity-based retriever using embedder + vector store."""

from __future__ import annotations

from ..embedders.base import BaseEmbedder
from ..models import RetrievedChunk
from ..stores.base import BaseStore
from .base import BaseRetriever


class SimilarityRetriever(BaseRetriever):
    """Retrieve chunks by embedding the query and searching the vector store.

    Parameters
    ----------
    embedder : BaseEmbedder
        The embedder to convert query text to vectors.
    store : BaseStore
        The vector store to search.
    top_k : int
        Default number of results to return.
    min_score : float
        Minimum similarity score threshold.
    """

    def __init__(
        self,
        *,
        embedder: BaseEmbedder,
        store: BaseStore,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self._top_k
        query_vector = self._embedder.embed_one(query)
        return self._store.search(
            query_vector,
            top_k=k,
            min_score=self._min_score,
        )

    def __repr__(self) -> str:
        return (
            f"SimilarityRetriever(embedder={self._embedder!r}, "
            f"store={self._store!r}, top_k={self._top_k})"
        )
