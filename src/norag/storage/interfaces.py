from abc import ABC, abstractmethod
from uuid import UUID

from norag.chunking.models import Chunk
from norag.embeddings.models import EmbeddedChunk
from norag.ingestion.models import Document


class DocumentRepository(ABC):
    """Contract for persistent document and retrieval data storage."""

    @abstractmethod
    def initialize(self) -> None:
        """Create or validate the storage schema."""

    @abstractmethod
    def save(
        self,
        document: Document,
        chunks: list[Chunk],
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        """Persist a document and all its indexed chunks atomically."""

    @abstractmethod
    def load(
        self,
    ) -> tuple[list[Document], list[Chunk], list[EmbeddedChunk]]:
        """Load all persisted documents, chunks and embeddings."""

    @abstractmethod
    def delete(self, document_id: UUID) -> bool:
        """Delete a document and its associated chunks."""
