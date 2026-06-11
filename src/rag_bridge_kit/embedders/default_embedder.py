"""Default hash-based embedder for local/offline use (no API needed)."""

from __future__ import annotations

import hashlib
import math

from .base import BaseEmbedder


class DefaultEmbedder(BaseEmbedder):
    """A simple hash-based embedder for testing and local development.

    This does NOT produce meaningful semantic embeddings. It generates
    deterministic vectors based on the SHA-256 hash of the input text.
    Use a real embedder (OpenAI, SentenceTransformers) for production.
    """

    def __init__(self, *, dims: int = 128) -> None:
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(text) for text in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Create a deterministic pseudo-embedding from a hash."""
        text_bytes = text.strip().lower().encode("utf-8")
        # Generate enough hash bytes for the requested dimensions
        vectors: list[float] = []
        iteration = 0
        while len(vectors) < self._dims:
            hash_input = f"{iteration}:{text_bytes.hex()}".encode("utf-8")
            digest = hashlib.sha256(hash_input).digest()
            for byte_val in digest:
                if len(vectors) >= self._dims:
                    break
                vectors.append(byte_val / 255.0)
            iteration += 1

        # Normalize to unit vector
        magnitude = math.sqrt(sum(v * v for v in vectors))
        if magnitude > 0:
            vectors = [v / magnitude for v in vectors]

        return [round(v, 6) for v in vectors]

    def __repr__(self) -> str:
        return f"DefaultEmbedder(dims={self._dims})"
