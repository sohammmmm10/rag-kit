"""Fixed-size character chunker."""

from __future__ import annotations

from ..errors import ValidationError
from ..models import Chunk, Document
from .base import BaseChunker


class FixedChunker(BaseChunker):
    """Split documents into fixed-size character chunks with optional overlap.

    Parameters
    ----------
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.
    strip_whitespace : bool
        Whether to strip leading/trailing whitespace from each chunk.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strip_whitespace: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValidationError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValidationError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValidationError("chunk_overlap must be less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strip_whitespace = strip_whitespace

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            segment = text[start:end]

            if self.strip_whitespace:
                segment = segment.strip()

            if segment:
                chunks.append(
                    Chunk(
                        content=segment,
                        metadata={**document.metadata, "chunker": "fixed"},
                        doc_id=document.doc_id,
                        index=index,
                    )
                )
                index += 1

            start += step

        return chunks

    def __repr__(self) -> str:
        return (
            f"FixedChunker(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )
