import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from norag.ingestion.interfaces import DocumentParser
from norag.ingestion.models import (
    Document,
    DocumentElement,
    DocumentMetadata,
    ElementType,
    SourceLocation,
)


class PDFParser(DocumentParser):
    """Extract text from PDF documents while preserving page locations."""

    PARAGRAPH_PATTERN = re.compile(
        r"\S(?:.*?\S)?(?=(?:\r?\n[ \t]*\r?\n)|\s*\Z)",
        re.DOTALL,
    )

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".pdf",)

    def parse(self, source: Path) -> Document:
        source = Path(source)

        if not source.is_file():
            raise FileNotFoundError(f"Document not found: {source}")

        raw_content = source.read_bytes()
        reader = PdfReader(source, strict=False)

        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:
                raise ValueError(
                    "Cannot parse an encrypted PDF without its password"
                ) from exc

            if not decrypted:
                raise ValueError("Cannot parse an encrypted PDF without its password")

        elements: list[DocumentElement] = []
        order = 0

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            for match in self.PARAGRAPH_PATTERN.finditer(page_text):
                paragraph = match.group().strip()

                if not paragraph:
                    continue

                elements.append(
                    DocumentElement(
                        element_type=ElementType.PARAGRAPH,
                        text=paragraph,
                        order=order,
                        source_location=SourceLocation(
                            page_number=page_number,
                            char_start=match.start(),
                            char_end=match.end(),
                        ),
                    )
                )

                order += 1

        pdf_metadata = reader.metadata

        return Document(
            source_uri=str(source.resolve()),
            content_hash=hashlib.sha256(raw_content).hexdigest(),
            metadata=DocumentMetadata(
                title=(
                    str(pdf_metadata.title)
                    if pdf_metadata and pdf_metadata.title
                    else None
                ),
                author=(
                    str(pdf_metadata.author)
                    if pdf_metadata and pdf_metadata.author
                    else None
                ),
                mime_type="application/pdf",
                file_name=source.name,
            ),
            elements=elements,
        )
