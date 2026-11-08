import hashlib
import re
from pathlib import Path

from norag.ingestion.interfaces import DocumentParser
from norag.ingestion.models import (
    Document,
    DocumentElement,
    DocumentMetadata,
    ElementType,
    SourceLocation,
)


class TextParser(DocumentParser):
    """Parser for UTF-8 plain-text documents."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".txt",)

    def parse(self, source: Path) -> Document:
        source = Path(source)

        if not source.is_file():
            raise FileNotFoundError(f"Document not found: {source}")

        raw_content = source.read_bytes()
        content = raw_content.decode("utf-8")

        elements = []

        # Split content into paragraphs using blank lines
        pattern = re.compile(
            r"\S(?:.*?\S)?(?=(?:\r?\n[ \t]*\r?\n)|\Z)",
            re.DOTALL,
        )

        for order, match in enumerate(pattern.finditer(content)):
            paragraph = match.group()

            elements.append(
                DocumentElement(
                    element_type=ElementType.PARAGRAPH,
                    text=paragraph,
                    order=order,
                    source_location=SourceLocation(
                        char_start=match.start(),
                        char_end=match.end(),
                    ),
                )
            )

        return Document(
            source_uri=str(source.resolve()),
            content_hash=hashlib.sha256(raw_content).hexdigest(),
            metadata=DocumentMetadata(
                file_name=source.name,
                mime_type="text/plain",
                language=None,
            ),
            elements=elements,
        )
