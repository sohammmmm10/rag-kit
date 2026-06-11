"""Default generator that returns context without calling an LLM."""

from __future__ import annotations

from ..models import QueryResult, RetrievedChunk
from .base import BaseGenerator


class DefaultGenerator(BaseGenerator):
    """A simple generator that returns the retrieved context directly.

    This is useful for testing or when you just need retrieval without
    LLM generation. It formats the retrieved chunks into a readable answer.
    """

    def __init__(self, *, include_scores: bool = True) -> None:
        self._include_scores = include_scores

    def generate(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
    ) -> QueryResult:
        if not context_chunks:
            return QueryResult(
                answer="No relevant context found for your query.",
                query=query,
                retrieved_chunks=context_chunks,
                provider="default",
            )

        parts: list[str] = []
        parts.append(f"Query: {query}\n")
        parts.append("Retrieved Context:")
        parts.append("-" * 40)

        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            header = f"[{i}] Source: {source}"
            if self._include_scores:
                header += f" | Score: {chunk.score:.4f}"
            parts.append(header)
            parts.append(chunk.content)
            parts.append("")

        return QueryResult(
            answer="\n".join(parts),
            query=query,
            retrieved_chunks=context_chunks,
            provider="default",
        )

    def __repr__(self) -> str:
        return "DefaultGenerator()"
