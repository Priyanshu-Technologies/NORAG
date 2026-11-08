from norag.ingestion.parsers import (
    HTMLParser,
    MarkdownParser,
    PDFParser,
    TextParser,
)
from norag.ingestion.registry import ParserRegistry
from norag.ingestion.service import IngestionService


def create_ingestion_service() -> IngestionService:
    """Create the default NORAG ingestion pipeline."""

    registry = ParserRegistry()

    registry.register(TextParser())
    registry.register(MarkdownParser())
    registry.register(HTMLParser())
    registry.register(PDFParser())

    return IngestionService(registry)
