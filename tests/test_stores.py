"""Tests for rag_kit vector stores."""

import pytest

from rag_kit.models import Chunk, EmbeddedChunk
from rag_kit.stores import MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def sample_chunks() -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            chunk=Chunk(content="Python is great", chunk_id="c1", doc_id="d1", index=0),
            vector=[1.0, 0.0, 0.0],
        ),
        EmbeddedChunk(
            chunk=Chunk(content="Java is popular", chunk_id="c2", doc_id="d1", index=1),
            vector=[0.0, 1.0, 0.0],
        ),
        EmbeddedChunk(
            chunk=Chunk(content="Python programming", chunk_id="c3", doc_id="d2", index=0),
            vector=[0.9, 0.1, 0.0],
        ),
    ]


class TestMemoryStore:
    def test_add_and_count(
        self, store: MemoryStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        added = store.add(sample_chunks)
        assert added == 3
        assert store.count() == 3

    def test_search(
        self, store: MemoryStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        store.add(sample_chunks)
        results = store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # Most similar to [1, 0, 0] should be c1
        assert results[0].chunk.chunk_id == "c1"
        assert results[0].score > 0.9

    def test_search_with_min_score(
        self, store: MemoryStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        store.add(sample_chunks)
        results = store.search([1.0, 0.0, 0.0], top_k=10, min_score=0.5)
        # Only c1 and c3 should pass (both have high similarity to [1,0,0])
        assert all(r.score >= 0.5 for r in results)

    def test_search_empty_store(self, store: MemoryStore) -> None:
        results = store.search([1.0, 0.0], top_k=5)
        assert results == []

    def test_clear(
        self, store: MemoryStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        store.add(sample_chunks)
        assert store.count() == 3
        store.clear()
        assert store.count() == 0

    def test_delete(
        self, store: MemoryStore, sample_chunks: list[EmbeddedChunk]
    ) -> None:
        store.add(sample_chunks)
        deleted = store.delete(["c1", "c2"])
        assert deleted == 2
        assert store.count() == 1

    def test_delete_nonexistent(self, store: MemoryStore) -> None:
        deleted = store.delete(["nonexistent"])
        assert deleted == 0

    def test_repr(self, store: MemoryStore) -> None:
        assert "MemoryStore" in repr(store)
