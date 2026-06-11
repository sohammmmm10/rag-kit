"""Custom exceptions for rag-bridge-kit."""


class RagKitError(Exception):
    """Base exception for rag-bridge-kit."""


class LoaderError(RagKitError):
    """Raised when a document loader fails."""


class ChunkerError(RagKitError):
    """Raised when a chunker fails."""


class EmbedderError(RagKitError):
    """Raised when an embedder fails."""


class StoreError(RagKitError):
    """Raised when a vector store operation fails."""


class RetrieverError(RagKitError):
    """Raised when a retriever fails."""


class GeneratorError(RagKitError):
    """Raised when a generator fails."""


class PipelineError(RagKitError):
    """Raised when the RAG pipeline encounters an error."""


class ValidationError(RagKitError):
    """Raised for invalid user input."""


class DependencyError(RagKitError):
    """Raised when an optional dependency is not installed."""
