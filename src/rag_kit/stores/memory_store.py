"""In-memory vector store using cosine similarity."""

from __future__ import annotations

import math

from ..models import Chunk, EmbeddedChunk, RetrievedChunk
from .base import BaseStore


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


class MemoryStore(BaseStore):
    """A simple in-memory vector store.

    Good for testing, prototyping, and small datasets. All data is lost
    when the process exits.
    """

    def __init__(self) -> None:
        self._store: dict[str, EmbeddedChunk] = {}

    def add(self, chunks: list[EmbeddedChunk]) -> int:
        added = 0
        for ec in chunks:
            self._store[ec.chunk_id] = ec
            added += 1
        return added

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        scored: list[tuple[float, EmbeddedChunk]] = []

        for ec in self._store.values():
            score = _cosine_similarity(query_vector, ec.vector)
            if score >= min_score:
                scored.append((score, ec))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RetrievedChunk] = []
        for score, ec in scored[:top_k]:
            results.append(
                RetrievedChunk(
                    chunk=ec.chunk,
                    score=round(score, 6),
                    vector=ec.vector,
                )
            )

        return results

    def count(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()

    def delete(self, chunk_ids: list[str]) -> int:
        deleted = 0
        for cid in chunk_ids:
            if cid in self._store:
                del self._store[cid]
                deleted += 1
        return deleted

    def __repr__(self) -> str:
        return f"MemoryStore(count={self.count()})"
