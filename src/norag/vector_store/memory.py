import math
from threading import RLock
from typing import Any
from uuid import UUID

from norag.embeddings.models import EmbeddedChunk
from norag.vector_store.interfaces import VectorIndex
from norag.vector_store.models import VectorSearchResult
from norag.vector_store.similarity import cosine_similarity


class InMemoryVectorIndex(VectorIndex):
    """Brute-force vector index using cosine similarity."""

    def __init__(self, dimensions: int | None = None) -> None:
        if dimensions is not None and dimensions < 1:
            raise ValueError("Dimensions must be positive")

        self._dimensions = dimensions
        self._chunks: dict[UUID, EmbeddedChunk] = {}
        self._lock = RLock()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def add(self, chunk: EmbeddedChunk) -> None:
        self.add_many([chunk])

    def add_many(self, chunks: list[EmbeddedChunk]) -> None:
        """Add chunks atomically after validating the entire batch."""

        if not chunks:
            return

        with self._lock:
            batch_ids = [chunk.chunk_id for chunk in chunks]

            if len(batch_ids) != len(set(batch_ids)):
                raise ValueError("Duplicate chunk IDs in batch")

            existing_ids = set(self._chunks)

            if any(chunk_id in existing_ids for chunk_id in batch_ids):
                raise ValueError("Chunk already exists in index")

            expected_dimensions = self._dimensions

            for chunk in chunks:
                if expected_dimensions is None:
                    expected_dimensions = chunk.dimensions

                if chunk.dimensions != expected_dimensions:
                    raise ValueError("Vector dimension mismatch")

                self._validate_vector(chunk.vector)

            # Only mutate the index after all validation succeeds.
            if self._dimensions is None:
                self._dimensions = expected_dimensions

            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        document_id: UUID | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Return the most similar chunks."""

        if top_k < 1:
            raise ValueError("top_k must be positive")

        self._validate_vector(query_vector)

        with self._lock:
            dimensions = self._dimensions
            snapshot = list(self._chunks.values())

        if dimensions is not None and len(query_vector) != dimensions:
            raise ValueError("Query vector dimension mismatch")

        filters = metadata_filters or {}

        results: list[VectorSearchResult] = []

        for chunk in snapshot:
            if document_id is not None and chunk.document_id != document_id:
                continue

            if not all(
                chunk.metadata.get(key) == value for key, value in filters.items()
            ):
                continue

            score = cosine_similarity(
                query_vector,
                chunk.vector,
            )

            results.append(
                VectorSearchResult(
                    chunk=chunk,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]

    def remove(self, chunk_id: UUID) -> bool:
        """Remove a chunk by ID."""

        with self._lock:
            if chunk_id not in self._chunks:
                return False

            del self._chunks[chunk_id]
            return True

    def clear(self) -> None:
        """Remove all indexed chunks."""

        with self._lock:
            self._chunks.clear()
            self._dimensions = None

    @staticmethod
    def _validate_vector(vector: list[float]) -> None:
        if not vector:
            raise ValueError("Vector cannot be empty")

        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Vector contains non-finite values")

        if all(value == 0 for value in vector):
            raise ValueError("Zero vectors cannot be indexed")
