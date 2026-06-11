"""Tests for rag_bridge_kit chunkers."""

import pytest

from rag_bridge_kit.chunkers import FixedChunker, RecursiveChunker, SentenceChunker
from rag_bridge_kit.errors import ValidationError
from rag_bridge_kit.models import Document


@pytest.fixture
def sample_doc() -> Document:
    return Document(
        content="The quick brown fox jumps over the lazy dog. "
        "This is a second sentence. And here is a third one. "
        "Finally, the last sentence of the document.",
        source="test.txt",
    )


@pytest.fixture
def long_doc() -> Document:
    return Document(
        content="Word " * 500,
        source="long.txt",
    )


# --- FixedChunker ---

class TestFixedChunker:
    def test_basic_chunking(self, sample_doc: Document) -> None:
        chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(sample_doc)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.content) <= 50
            assert chunk.doc_id == sample_doc.doc_id

    def test_empty_document(self) -> None:
        doc = Document(content="   ", source="empty.txt")
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_small_document(self) -> None:
        doc = Document(content="Small text", source="small.txt")
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Small text"

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(ValidationError):
            FixedChunker(chunk_size=0)

    def test_overlap_exceeds_size(self) -> None:
        with pytest.raises(ValidationError):
            FixedChunker(chunk_size=10, chunk_overlap=10)

    def test_chunk_indices(self, long_doc: Document) -> None:
        chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(long_doc)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))


# --- SentenceChunker ---

class TestSentenceChunker:
    def test_basic_sentence_chunking(self, sample_doc: Document) -> None:
        chunker = SentenceChunker(max_chunk_size=80, sentence_overlap=0)
        chunks = chunker.chunk(sample_doc)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata.get("chunker") == "sentence"

    def test_overlap(self, sample_doc: Document) -> None:
        chunker = SentenceChunker(max_chunk_size=60, sentence_overlap=1)
        chunks = chunker.chunk(sample_doc)
        assert len(chunks) >= 2

    def test_empty_document(self) -> None:
        doc = Document(content="", source="empty.txt")
        chunker = SentenceChunker(max_chunk_size=100)
        chunks = chunker.chunk(doc)
        assert chunks == []


# --- RecursiveChunker ---

class TestRecursiveChunker:
    def test_basic_recursive_chunking(self, sample_doc: Document) -> None:
        chunker = RecursiveChunker(chunk_size=60, chunk_overlap=10)
        chunks = chunker.chunk(sample_doc)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata.get("chunker") == "recursive"

    def test_paragraph_splitting(self) -> None:
        doc = Document(
            content="Paragraph one.\n\nParagraph two.\n\nParagraph three.",
            source="paragraphs.txt",
        )
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1

    def test_empty_document(self) -> None:
        doc = Document(content="   ", source="empty.txt")
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_chunk_many(self) -> None:
        docs = [
            Document(content="First document content.", source="a.txt"),
            Document(content="Second document content.", source="b.txt"),
        ]
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk_many(docs)
        assert len(chunks) >= 2
