from abc import ABC, abstractmethod
from pathlib import Path

from norag.ingestion.models import Document


class DocumentParser(ABC):
    """Contract for all NORAG document parsers."""

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """Return supported file extensions."""
        ...

    @abstractmethod
    def parse(self, source: Path) -> Document:
        """Parse a document and return normalized output."""
        ...
