"""Tests for rag_bridge_kit.pipeline (RAGPipeline)."""

import pytest

from rag_bridge_kit import RAGPipeline, Document, ValidationError
from rag_bridge_kit.chunkers import FixedChunker, SentenceChunker
from rag_bridge_kit.embedders import DefaultEmbedder
from rag_bridge_kit.stores import MemoryStore


@pytest.fixture
def pipeline() -> RAGPipeline:
    return RAGPipeline(
        chunker=FixedChunker(chunk_size=100, chunk_overlap=20),
        embedder=DefaultEmbedder(dims=32),
        store=MemoryStore(),
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            content=(
                "Python is a high-level programming language. "
                "It is widely used for web development, data science, and AI. "
                "Python has a simple and readable syntax."
            ),
            source="python.txt",
        ),
        Document(
            content=(
                "Machine learning is a subset of artificial intelligence. "
                "It allows systems to learn from data without being explicitly programmed. "
                "Deep learning is a subset of machine learning."
            ),
            source="ml.txt",
        ),
        Document(
            content=(
                "Retrieval Augmented Generation combines retrieval and generation. "
                "RAG pipelines retrieve relevant documents and use them as context. "
                "This improves the accuracy and factuality of LLM responses."
            ),
            source="rag.txt",
        ),
    ]


class TestRAGPipeline:
    def test_ingest_documents(
        self, pipeline: RAGPipeline, sample_documents: list[Document]
    ) -> None:
        stats = pipeline.ingest(documents=sample_documents)
        assert stats.documents_loaded == 3
        assert stats.chunks_created > 0
        assert stats.chunks_stored > 0
        assert pipeline.count() > 0

    def test_ingest_text(self, pipeline: RAGPipeline) -> None:
        stats = pipeline.ingest_text("This is a simple test document.")
        assert stats.documents_loaded == 1
        assert stats.chunks_stored >= 1

    def test_ingest_texts(self, pipeline: RAGPipeline) -> None:
        stats = pipeline.ingest_texts(["First doc.", "Second doc.", "Third doc."])
        assert stats.documents_loaded == 3
        assert stats.chunks_stored >= 3

    def test_query(
        self, pipeline: RAGPipeline, sample_documents: list[Document]
    ) -> None:
        pipeline.ingest(documents=sample_documents)
        result = pipeline.query("What is Python?")
        assert result.answer
        assert result.query == "What is Python?"
        assert len(result.retrieved_chunks) > 0

    def test_query_empty_raises(self, pipeline: RAGPipeline) -> None:
        with pytest.raises(ValidationError):
            pipeline.query("")

    def test_query_with_top_k(
        self, pipeline: RAGPipeline, sample_documents: list[Document]
    ) -> None:
        pipeline.ingest(documents=sample_documents)
        result = pipeline.query("What is RAG?", top_k=2)
        assert len(result.retrieved_chunks) <= 2

    def test_retrieve_only(
        self, pipeline: RAGPipeline, sample_documents: list[Document]
    ) -> None:
        pipeline.ingest(documents=sample_documents)
        chunks = pipeline.retrieve("machine learning", top_k=3)
        assert len(chunks) <= 3

    def test_clear(
        self, pipeline: RAGPipeline, sample_documents: list[Document]
    ) -> None:
        pipeline.ingest(documents=sample_documents)
        assert pipeline.count() > 0
        pipeline.clear()
        assert pipeline.count() == 0

    def test_no_loader_or_documents_raises(self, pipeline: RAGPipeline) -> None:
        with pytest.raises(ValidationError):
            pipeline.ingest()

    def test_default_pipeline(self) -> None:
        """Pipeline works with all defaults."""
        p = RAGPipeline()
        stats = p.ingest_text("Hello world, this is a test.")
        assert stats.chunks_stored >= 1
        result = p.query("test")
        assert result.answer

    def test_repr(self, pipeline: RAGPipeline) -> None:
        r = repr(pipeline)
        assert "RAGPipeline" in r
        assert "FixedChunker" in r
        assert "DefaultEmbedder" in r

    def test_sentence_chunker_integration(self) -> None:
        p = RAGPipeline(
            chunker=SentenceChunker(max_chunk_size=80, sentence_overlap=1),
            embedder=DefaultEmbedder(dims=16),
        )
        p.ingest_text(
            "Sentence one about AI. Sentence two about ML. "
            "Sentence three about data. Sentence four about Python."
        )
        assert p.count() >= 1
        result = p.query("AI")
        assert result.answer
