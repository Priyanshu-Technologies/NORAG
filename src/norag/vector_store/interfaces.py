from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from norag.embeddings.models import EmbeddedChunk
from norag.vector_store.models import VectorSearchResult


class VectorIndex(ABC):
    """Contract for interchangeable vector index backends."""

    @abstractmethod
    def add(self, chunk: EmbeddedChunk) -> None: ...

    @abstractmethod
    def add_many(self, chunks: list[EmbeddedChunk]) -> None: ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        document_id: UUID | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]: ...

    @abstractmethod
    def remove(self, chunk_id: UUID) -> bool: ...

    @abstractmethod
    def clear(self) -> None: ...

    @property
    @abstractmethod
    def count(self) -> int: ...
