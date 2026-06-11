"""Recursive character text splitter."""

from __future__ import annotations

from ..errors import ValidationError
from ..models import Chunk, Document
from .base import BaseChunker

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
    """Recursively split text using a hierarchy of separators.

    Tries to split on the first separator that produces chunks within
    the size limit, falling back to smaller separators as needed.
    This is similar to LangChain's ``RecursiveCharacterTextSplitter``.

    Parameters
    ----------
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.
    separators : list[str] | None
        Ordered list of separators to try (coarsest to finest).
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValidationError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValidationError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValidationError("chunk_overlap must be less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or list(DEFAULT_SEPARATORS)

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        raw_chunks = self._split_text(text, self.separators)

        chunks: list[Chunk] = []
        for idx, raw in enumerate(raw_chunks):
            chunks.append(
                Chunk(
                    content=raw,
                    metadata={**document.metadata, "chunker": "recursive"},
                    doc_id=document.doc_id,
                    index=idx,
                )
            )
        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []

        # Find the appropriate separator
        separator = separators[-1]
        remaining_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining_separators = []
                break
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1 :]
                break

        # Split using the chosen separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge small splits and recursively split large ones
        good_splits: list[str] = []

        for piece in splits:
            piece_stripped = piece.strip()
            if not piece_stripped:
                continue

            if len(piece_stripped) <= self.chunk_size:
                good_splits.append(piece_stripped)
            elif remaining_separators:
                # Flush accumulated good splits first
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                # Recursively split the large piece
                sub_chunks = self._split_text(piece_stripped, remaining_separators)
                final_chunks.extend(sub_chunks)
            else:
                # Can't split further — force add
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                final_chunks.append(piece_stripped)

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge small splits into chunks respecting size and overlap."""
        merged: list[str] = []
        current_parts: list[str] = []
        current_length = 0

        for split in splits:
            join_len = len(separator) if current_parts else 0
            candidate = current_length + len(split) + join_len

            if candidate > self.chunk_size and current_parts:
                merged.append(separator.join(current_parts))

                # Handle overlap
                while current_parts and current_length > self.chunk_overlap:
                    removed = current_parts.pop(0)
                    current_length -= len(removed) + (
                        len(separator) if current_parts else 0
                    )

            current_parts.append(split)
            current_length = sum(len(p) for p in current_parts) + len(separator) * max(
                0, len(current_parts) - 1
            )

        if current_parts:
            merged.append(separator.join(current_parts))

        return merged

    def __repr__(self) -> str:
        return (
            f"RecursiveChunker(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )
