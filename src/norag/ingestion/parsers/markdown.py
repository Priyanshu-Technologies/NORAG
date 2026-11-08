import hashlib
from pathlib import Path
from uuid import UUID

from markdown_it import MarkdownIt
from markdown_it.token import Token

from norag.ingestion.interfaces import DocumentParser
from norag.ingestion.models import (
    Document,
    DocumentElement,
    DocumentMetadata,
    ElementType,
    SourceLocation,
)


class MarkdownParser(DocumentParser):
    """Parse Markdown while preserving document hierarchy."""

    def __init__(self) -> None:
        self._markdown = MarkdownIt("commonmark")

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".md", ".markdown")

    @staticmethod
    def _inline_text(token: Token) -> str:
        """Extract readable text from parsed inline tokens."""
        if not token.children:
            return token.content

        parts = []

        for child in token.children:
            if child.type in {
                "text",
                "code_inline",
                "html_inline",
            }:
                parts.append(child.content)
            elif child.type in {"softbreak", "hardbreak"}:
                parts.append("\n")
            elif child.type == "image":
                parts.append(child.content)

        return "".join(parts)

    @staticmethod
    def _source_offsets(
        token: Token,
        line_offsets: list[int],
        content_length: int,
    ) -> tuple[int, int]:
        """Convert parser line positions into character offsets."""
        if token.map is None:
            return 0, 0

        start_line, end_line = token.map

        start = line_offsets[start_line]
        end = line_offsets[end_line] if end_line < len(line_offsets) else content_length

        return start, end

    def parse(self, source: Path) -> Document:
        source = Path(source)

        if not source.is_file():
            raise FileNotFoundError(f"Document not found: {source}")

        raw_content = source.read_bytes()
        content = raw_content.decode("utf-8")

        tokens = self._markdown.parse(content)

        lines = content.splitlines(keepends=True)
        line_offsets = [0]

        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))

        elements: list[DocumentElement] = []
        heading_stack: list[tuple[int, UUID]] = []
        container_stack: list[str] = []

        order = 0
        index = 0

        while index < len(tokens):
            token = tokens[index]

            if token.type == "heading_open":
                inline = tokens[index + 1]
                level = int(token.tag[1])

                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()

                parent_id = heading_stack[-1][1] if heading_stack else None

                start, end = self._source_offsets(token, line_offsets, len(content))

                element = DocumentElement(
                    element_type=ElementType.HEADING,
                    text=self._inline_text(inline),
                    order=order,
                    parent_id=parent_id,
                    source_location=SourceLocation(
                        char_start=start,
                        char_end=end,
                    ),
                    metadata={"heading_level": level},
                )

                elements.append(element)
                heading_stack.append((level, element.id))
                order += 1
                index += 2
                continue

            if token.type == "paragraph_open":
                inline = tokens[index + 1]

                element_type = (
                    ElementType.LIST_ITEM
                    if "list_item_open" in container_stack
                    else ElementType.PARAGRAPH
                )

                start, end = self._source_offsets(token, line_offsets, len(content))

                elements.append(
                    DocumentElement(
                        element_type=element_type,
                        text=self._inline_text(inline),
                        order=order,
                        parent_id=(heading_stack[-1][1] if heading_stack else None),
                        source_location=SourceLocation(
                            char_start=start,
                            char_end=end,
                        ),
                    )
                )

                order += 1
                index += 2
                continue

            if token.type in {"fence", "code_block"}:
                start, end = self._source_offsets(token, line_offsets, len(content))

                elements.append(
                    DocumentElement(
                        element_type=ElementType.CODE,
                        text=token.content,
                        order=order,
                        parent_id=(heading_stack[-1][1] if heading_stack else None),
                        source_location=SourceLocation(
                            char_start=start,
                            char_end=end,
                        ),
                        metadata={
                            "language": token.info.strip() or None,
                        },
                    )
                )

                order += 1
                index += 1
                continue

            if token.type == "list_item_open":
                container_stack.append(token.type)

            elif token.type == "list_item_close" and container_stack:
                container_stack.pop()

            index += 1

        return Document(
            source_uri=str(source.resolve()),
            content_hash=hashlib.sha256(raw_content).hexdigest(),
            metadata=DocumentMetadata(
                file_name=source.name,
                mime_type="text/markdown",
            ),
            elements=elements,
        )
