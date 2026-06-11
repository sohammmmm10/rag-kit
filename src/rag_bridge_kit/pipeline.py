"""RAG Pipeline â€” the main entry point for rag-bridge-kit."""

from __future__ import annotations

import time
from typing import Any

from .chunkers.base import BaseChunker
from .chunkers.fixed_chunker import FixedChunker
from .config import RagKitSettings
from .embedders.base import BaseEmbedder
from .embedders.default_embedder import DefaultEmbedder
from .errors import PipelineError, ValidationError
from .generators.base import BaseGenerator
from .generators.default_generator import DefaultGenerator
from .loaders.base import BaseLoader
from .models import (
    Chunk,
    Document,
    EmbeddedChunk,
    IngestStats,
    QueryResult,
    RetrievedChunk,
)
from .retrievers.base import BaseRetriever
from .retrievers.similarity_retriever import SimilarityRetriever
from .stores.base import BaseStore
from .stores.memory_store import MemoryStore


class RAGPipeline:
    """Plug-and-play RAG pipeline.

    Orchestrates the full flow: Load -> Chunk -> Embed -> Store -> Retrieve -> Generate.

    All components are pluggable. Sensible defaults are provided for
    every stage so you can start with zero configuration.

    Parameters
    ----------
    loader : BaseLoader | None
        Document loader (set later via ``ingest`` if not provided).
    chunker : BaseChunker | None
        Text chunker. Defaults to ``FixedChunker()``.
    embedder : BaseEmbedder | None
        Embedding provider. Defaults to ``DefaultEmbedder()``.
    store : BaseStore | None
        Vector store. Defaults to ``MemoryStore()``.
    retriever : BaseRetriever | None
        Retriever. Auto-created from embedder + store if not provided.
    generator : BaseGenerator | None
        Answer generator. Defaults to ``DefaultGenerator()``.
    settings : RagKitSettings | None
        Global settings.

    Examples
    --------
    Minimal usage::

        from rag_kit import RAGPipeline
        from rag_kit.loaders import TextLoader

        pipeline = RAGPipeline(loader=TextLoader("docs/"))
        pipeline.ingest()
        result = pipeline.query("What is the refund policy?")
        print(result.answer)
    """

    def __init__(
        self,
        *,
        loader: BaseLoader | None = None,
        chunker: BaseChunker | None = None,
        embedder: BaseEmbedder | None = None,
        store: BaseStore | None = None,
        retriever: BaseRetriever | None = None,
        generator: BaseGenerator | None = None,
        settings: RagKitSettings | None = None,
    ) -> None:
        self.settings = settings or RagKitSettings()
        self.loader = loader
        self.chunker = chunker or FixedChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.embedder = embedder or DefaultEmbedder()
        self.store = store or MemoryStore()
        self.retriever = retriever or SimilarityRetriever(
            embedder=self.embedder,
            store=self.store,
            top_k=self.settings.top_k,
            min_score=self.settings.similarity_threshold,
        )
        self.generator = generator or DefaultGenerator()

    # ------------------------------------------------------------------ #
    #  INGEST: Load -> Chunk -> Embed -> Store
    # ------------------------------------------------------------------ #

    def ingest(
        self,
        loader: BaseLoader | None = None,
        *,
        documents: list[Document] | None = None,
    ) -> IngestStats:
        """Run the full ingestion pipeline.

        You can pass a ``loader`` to override the default, or pass
        pre-loaded ``documents`` directly.
        """
        start = time.monotonic()
        stats = IngestStats()

        # Step 1: Load documents
        active_loader = loader or self.loader
        if documents:
            docs = documents
        elif active_loader:
            try:
                docs = active_loader.load()
            except Exception as exc:
                raise PipelineError(f"Loading failed: {exc}") from exc
        else:
            raise ValidationError(
                "No loader or documents provided. Pass a loader to RAGPipeline "
                "or provide documents directly to ingest()."
            )

        stats.documents_loaded = len(docs)
        if not docs:
            stats.duration_seconds = round(time.monotonic() - start, 3)
            return stats

        # Step 2: Chunk
        try:
            chunks = self.chunker.chunk_many(docs)
        except Exception as exc:
            raise PipelineError(f"Chunking failed: {exc}") from exc

        stats.chunks_created = len(chunks)
        if not chunks:
            stats.duration_seconds = round(time.monotonic() - start, 3)
            return stats

        # Step 3: Embed
        try:
            embedded = self._embed_chunks(chunks)
        except Exception as exc:
            raise PipelineError(f"Embedding failed: {exc}") from exc

        stats.embeddings_generated = len(embedded)

        # Step 4: Store
        try:
            added = self.store.add(embedded)
        except Exception as exc:
            raise PipelineError(f"Storing failed: {exc}") from exc

        stats.chunks_stored = added
        stats.duration_seconds = round(time.monotonic() - start, 3)

        return stats

    def ingest_text(self, text: str, *, source: str = "direct_input") -> IngestStats:
        """Convenience method to ingest raw text directly."""
        doc = Document(content=text, source=source)
        return self.ingest(documents=[doc])

    def ingest_texts(
        self, texts: list[str], *, source: str = "direct_input"
    ) -> IngestStats:
        """Convenience method to ingest a list of raw texts."""
        docs = [
            Document(content=text, source=f"{source}_{i}")
            for i, text in enumerate(texts)
        ]
        return self.ingest(documents=docs)

    # ------------------------------------------------------------------ #
    #  QUERY: Retrieve -> Generate
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> QueryResult:
        """Run the full query pipeline: retrieve context + generate answer.

        Parameters
        ----------
        question : str
            The user's question.
        top_k : int | None
            Number of context chunks to retrieve. Uses settings default if None.
        """
        question = question.strip()
        if not question:
            raise ValidationError("Query cannot be empty.")

        k = top_k if top_k is not None else self.settings.top_k

        # Step 1: Retrieve
        try:
            retrieved = self.retriever.retrieve(question, top_k=k)
        except Exception as exc:
            raise PipelineError(f"Retrieval failed: {exc}") from exc

        # Step 2: Generate
        try:
            result = self.generator.generate(question, retrieved)
        except Exception as exc:
            raise PipelineError(f"Generation failed: {exc}") from exc

        return result

    def retrieve(
        self, question: str, *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks without generating an answer."""
        question = question.strip()
        if not question:
            raise ValidationError("Query cannot be empty.")

        k = top_k if top_k is not None else self.settings.top_k
        return self.retriever.retrieve(question, top_k=k)

    # ------------------------------------------------------------------ #
    #  UTILITY
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        return self.store.count()

    def clear(self) -> None:
        """Remove all chunks from the store."""
        self.store.clear()

    def _embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed chunks in batches."""
        batch_size = self.settings.embedding_batch_size
        embedded: list[EmbeddedChunk] = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]
            vectors = self.embedder.embed(texts)

            for chunk, vector in zip(batch, vectors):
                embedded.append(EmbeddedChunk(chunk=chunk, vector=vector))

        return embedded

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(\n"
            f"  chunker={self.chunker!r},\n"
            f"  embedder={self.embedder!r},\n"
            f"  store={self.store!r},\n"
            f"  generator={self.generator!r},\n"
            f")"
        )
