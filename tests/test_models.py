"""Tests for rag_bridge_kit.models."""

from rag_bridge_kit.models import Chunk, Document, EmbeddedChunk, RetrievedChunk


def test_document_auto_id() -> None:
    doc = Document(content="Hello world", source="test.txt")
    assert doc.doc_id
    assert len(doc.doc_id) == 16


def test_document_custom_id() -> None:
    doc = Document(content="Hello", doc_id="custom-123")
    assert doc.doc_id == "custom-123"


def test_chunk_auto_id() -> None:
    chunk = Chunk(content="Some chunk", doc_id="doc1", index=0)
    assert chunk.chunk_id
    assert len(chunk.chunk_id) == 16


def test_embedded_chunk_properties() -> None:
    chunk = Chunk(content="test", metadata={"key": "val"}, chunk_id="c1")
    ec = EmbeddedChunk(chunk=chunk, vector=[0.1, 0.2])
    assert ec.content == "test"
    assert ec.metadata == {"key": "val"}
    assert ec.chunk_id == "c1"


def test_retrieved_chunk_properties() -> None:
    chunk = Chunk(content="found", metadata={"src": "a"})
    rc = RetrievedChunk(chunk=chunk, score=0.95)
    assert rc.content == "found"
    assert rc.score == 0.95
    assert rc.metadata == {"src": "a"}
