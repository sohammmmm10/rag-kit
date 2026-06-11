"""Tests for rag_bridge_kit embedders."""

import math

from rag_bridge_kit.embedders import DefaultEmbedder


class TestDefaultEmbedder:
    def test_dimensions(self) -> None:
        embedder = DefaultEmbedder(dims=64)
        assert embedder.dimensions == 64

    def test_embed_single(self) -> None:
        embedder = DefaultEmbedder(dims=32)
        vec = embedder.embed_one("hello")
        assert len(vec) == 32

    def test_embed_batch(self) -> None:
        embedder = DefaultEmbedder(dims=16)
        vecs = embedder.embed(["hello", "world", "test"])
        assert len(vecs) == 3
        assert all(len(v) == 16 for v in vecs)

    def test_deterministic(self) -> None:
        embedder = DefaultEmbedder(dims=16)
        v1 = embedder.embed_one("hello")
        v2 = embedder.embed_one("hello")
        assert v1 == v2

    def test_different_texts_different_vectors(self) -> None:
        embedder = DefaultEmbedder(dims=16)
        v1 = embedder.embed_one("hello")
        v2 = embedder.embed_one("world")
        assert v1 != v2

    def test_unit_normalized(self) -> None:
        embedder = DefaultEmbedder(dims=64)
        vec = embedder.embed_one("some text")
        magnitude = math.sqrt(sum(v * v for v in vec))
        assert abs(magnitude - 1.0) < 0.01

    def test_repr(self) -> None:
        embedder = DefaultEmbedder(dims=128)
        assert "DefaultEmbedder" in repr(embedder)
        assert "128" in repr(embedder)
