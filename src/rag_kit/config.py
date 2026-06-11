"""Configuration for rag-kit."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _as_int(value: str | None, *, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


def _as_float(value: str | None, *, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


@dataclass
class RagKitSettings:
    """Global settings for rag-kit."""

    # Chunking defaults
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval defaults
    top_k: int = 5
    similarity_threshold: float = 0.0

    # Embedding defaults
    embedding_batch_size: int = 64

    @classmethod
    def from_env(cls) -> "RagKitSettings":
        return cls(
            chunk_size=_as_int(os.getenv("RAGKIT_CHUNK_SIZE"), default=512),
            chunk_overlap=_as_int(os.getenv("RAGKIT_CHUNK_OVERLAP"), default=64),
            top_k=_as_int(os.getenv("RAGKIT_TOP_K"), default=5),
            similarity_threshold=_as_float(
                os.getenv("RAGKIT_SIMILARITY_THRESHOLD"), default=0.0
            ),
            embedding_batch_size=_as_int(
                os.getenv("RAGKIT_EMBEDDING_BATCH_SIZE"), default=64
            ),
        )
