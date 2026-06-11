"""ChromaDB vector store adapter."""

from __future__ import annotations

from typing import Any

from ..errors import DependencyError, StoreError
from ..models import Chunk, EmbeddedChunk, RetrievedChunk
from .base import BaseStore


class ChromaStore(BaseStore):
    """Vector store backed by ChromaDB.

    Requires: ``pip install rag-kit[chromadb]``

    Parameters
    ----------
    collection_name : str
        Name of the Chroma collection.
    persist_directory : str | None
        Path for persistent storage. Uses in-memory if None.
    """

    def __init__(
        self,
        *,
        collection_name: str = "rag-kit-default",
        persist_directory: str | None = None,
    ) -> None:
        try:
            import chromadb  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyError(
                "chromadb is required for ChromaStore. "
                "Install with: pip install rag-kit[chromadb]"
            ) from exc

        try:
            if persist_directory:
                self._client = chromadb.PersistentClient(path=persist_directory)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise StoreError(f"Failed to initialize ChromaDB: {exc}") from exc

        self._collection_name = collection_name

    def add(self, chunks: list[EmbeddedChunk]) -> int:
        if not chunks:
            return 0

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for ec in chunks:
            ids.append(ec.chunk_id)
            embeddings.append(ec.vector)
            documents.append(ec.content)
            # ChromaDB requires metadata values to be str, int, float, or bool
            clean_meta: dict[str, Any] = {}
            for key, val in ec.metadata.items():
                if isinstance(val, (str, int, float, bool)):
                    clean_meta[key] = val
                else:
                    clean_meta[key] = str(val)
            metadatas.append(clean_meta)

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise StoreError(f"Failed to add chunks to ChromaDB: {exc}") from exc

        return len(chunks)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, max(1, self.count())),
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception as exc:
            raise StoreError(f"ChromaDB search failed: {exc}") from exc

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        retrieved: list[RetrievedChunk] = []
        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        embeddings = results.get("embeddings", [[]])[0] if results.get("embeddings") else [None] * len(ids)

        for i, chunk_id in enumerate(ids):
            # ChromaDB returns distances (cosine distance), convert to similarity
            distance = distances[i] if i < len(distances) else 1.0
            score = 1.0 - distance  # cosine distance to cosine similarity

            if score < min_score:
                continue

            chunk = Chunk(
                content=documents[i] if i < len(documents) else "",
                metadata=dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {},
                chunk_id=chunk_id,
            )

            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=round(score, 6),
                    vector=embeddings[i] if embeddings[i] is not None else None,
                )
            )

        return retrieved

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise StoreError(f"Failed to clear ChromaDB collection: {exc}") from exc

    def delete(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        try:
            self._collection.delete(ids=chunk_ids)
            return len(chunk_ids)
        except Exception as exc:
            raise StoreError(f"Failed to delete from ChromaDB: {exc}") from exc

    def __repr__(self) -> str:
        return (
            f"ChromaStore(collection={self._collection_name!r}, "
            f"count={self.count()})"
        )
