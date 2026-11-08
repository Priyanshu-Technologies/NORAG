from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from norag.chunking.models import Chunk
from norag.sparse_retrieval.models import SparseSearchResult


class SparseIndex(ABC):
    """Contract for interchangeable sparse retrieval backends."""

    @abstractmethod
    def add(self, chunk: Chunk) -> None: ...

    @abstractmethod
    def add_many(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: UUID | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[SparseSearchResult]: ...

    @abstractmethod
    def remove(self, chunk_id: UUID) -> bool: ...

    @abstractmethod
    def clear(self) -> None: ...

    @property
    @abstractmethod
    def count(self) -> int: ...
