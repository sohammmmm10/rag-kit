"""rag-bridge-kit: Plug-and-play RAG pipeline library for Python."""

__version__ = "0.1.0"

from .config import RagKitSettings
from .errors import (
    ChunkerError,
    DependencyError,
    EmbedderError,
    GeneratorError,
    LoaderError,
    PipelineError,
    RagKitError,
    RetrieverError,
    StoreError,
    ValidationError,
)
from .models import (
    Chunk,
    Document,
    EmbeddedChunk,
    IngestStats,
    QueryResult,
    RetrievedChunk,
)
from .pipeline import RAGPipeline

__all__ = [
    # Pipeline
    "RAGPipeline",
    # Models
    "Document",
    "Chunk",
    "EmbeddedChunk",
    "RetrievedChunk",
    "QueryResult",
    "IngestStats",
    # Config
    "RagKitSettings",
    # Errors
    "RagKitError",
    "LoaderError",
    "ChunkerError",
    "EmbedderError",
    "StoreError",
    "RetrieverError",
    "GeneratorError",
    "PipelineError",
    "ValidationError",
    "DependencyError",
]
