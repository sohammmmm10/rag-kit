"""Core data models for rag-kit."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


@dataclass
class Document:
    """Represents a loaded document before chunking."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            import hashlib

            self.doc_id = hashlib.sha256(
                f"{self.source}:{self.content[:256]}".encode()
            ).hexdigest()[:16]


@dataclass
class Chunk:
    """Represents a chunk of text split from a document."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    doc_id: str = ""
    index: int = 0

    def __post_init__(self) -> None:
        if not self.chunk_id:
            import hashlib

            self.chunk_id = hashlib.sha256(
                f"{self.doc_id}:{self.index}:{self.content[:128]}".encode()
            ).hexdigest()[:16]


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector attached."""

    chunk: Chunk
    vector: list[float]

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.chunk.metadata

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store with a relevance score."""

    chunk: Chunk
    score: float
    vector: list[float] | None = None

    @property
    def content(self) -> str:
        return self.chunk.content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.chunk.metadata


@dataclass
class QueryResult:
    """The final result of a RAG query."""

    answer: str
    query: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    model: str | None = None
    provider: str | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw: Any = None


@dataclass
class IngestStats:
    """Statistics from an ingest operation."""

    documents_loaded: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    chunks_stored: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
