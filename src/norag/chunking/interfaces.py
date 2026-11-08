from abc import ABC, abstractmethod

from norag.chunking.models import Chunk
from norag.ingestion.models import Document


class Chunker(ABC):
    """Contract for all NORAG chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Convert a document into retrievable chunks."""
        ...
