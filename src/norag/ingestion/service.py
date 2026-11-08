from pathlib import Path

from norag.ingestion.models import Document
from norag.ingestion.registry import ParserRegistry


class IngestionService:
    """Coordinate document validation and parsing."""

    def __init__(self, registry: ParserRegistry) -> None:
        self.registry = registry

    def ingest(self, source: Path) -> Document:
        source = Path(source)

        if not source.exists():
            raise FileNotFoundError(f"Document not found: {source}")

        if not source.is_file():
            raise ValueError(f"Source is not a file: {source}")

        if source.stat().st_size == 0:
            raise ValueError(f"Document is empty: {source}")

        return self.registry.parse(source)
