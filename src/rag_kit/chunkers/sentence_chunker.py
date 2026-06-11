"""Sentence-based chunker."""

from __future__ import annotations

import re

from ..errors import ValidationError
from ..models import Chunk, Document
from .base import BaseChunker

# Simple sentence boundary regex — handles ., !, ? followed by space or end.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker(BaseChunker):
    """Split documents by sentences, grouping them into chunks that
    don't exceed ``max_chunk_size`` characters.

    Parameters
    ----------
    max_chunk_size : int
        Maximum number of characters per chunk.
    sentence_overlap : int
        Number of sentences to overlap between consecutive chunks.
    """

    def __init__(
        self,
        *,
        max_chunk_size: int = 512,
        sentence_overlap: int = 1,
    ) -> None:
        if max_chunk_size <= 0:
            raise ValidationError("max_chunk_size must be a positive integer.")
        if sentence_overlap < 0:
            raise ValidationError("sentence_overlap cannot be negative.")

        self.max_chunk_size = max_chunk_size
        self.sentence_overlap = sentence_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
        if not sentences:
            return [
                Chunk(
                    content=text.strip(),
                    metadata={**document.metadata, "chunker": "sentence"},
                    doc_id=document.doc_id,
                    index=0,
                )
            ]

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_length = 0
        index = 0

        for sentence in sentences:
            candidate_length = current_length + len(sentence) + (
                1 if current_sentences else 0
            )

            if candidate_length > self.max_chunk_size and current_sentences:
                # Flush current chunk
                chunks.append(
                    Chunk(
                        content=" ".join(current_sentences),
                        metadata={
                            **document.metadata,
                            "chunker": "sentence",
                            "sentence_count": len(current_sentences),
                        },
                        doc_id=document.doc_id,
                        index=index,
                    )
                )
                index += 1

                # Overlap: keep last N sentences
                if self.sentence_overlap > 0:
                    current_sentences = current_sentences[-self.sentence_overlap :]
                    current_length = sum(len(s) for s in current_sentences) + max(
                        0, len(current_sentences) - 1
                    )
                else:
                    current_sentences = []
                    current_length = 0

            current_sentences.append(sentence)
            current_length = sum(len(s) for s in current_sentences) + max(
                0, len(current_sentences) - 1
            )

        # Don't forget the last chunk
        if current_sentences:
            chunks.append(
                Chunk(
                    content=" ".join(current_sentences),
                    metadata={
                        **document.metadata,
                        "chunker": "sentence",
                        "sentence_count": len(current_sentences),
                    },
                    doc_id=document.doc_id,
                    index=index,
                )
            )

        return chunks

    def __repr__(self) -> str:
        return (
            f"SentenceChunker(max_chunk_size={self.max_chunk_size}, "
            f"sentence_overlap={self.sentence_overlap})"
        )
