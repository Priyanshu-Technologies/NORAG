import hashlib
from pathlib import Path
from typing import ClassVar
from uuid import UUID

from bs4 import BeautifulSoup, Tag

from norag.ingestion.interfaces import DocumentParser
from norag.ingestion.models import (
    Document,
    DocumentElement,
    DocumentMetadata,
    ElementType,
)


class HTMLParser(DocumentParser):
    """Parse HTML documents while preserving their semantic structure."""

    IGNORED_TAGS: ClassVar[set[str]] = {
        "script",
        "style",
        "noscript",
        "nav",
        "footer",
        "form",
        "svg",
        "iframe",
        "button",
    }

    def __init__(self) -> None:
        self._heading_stack: list[tuple[int, UUID]] = []
        self._elements: list[DocumentElement] = []
        self._order = 0

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".html", ".htm")

    def _current_parent(self) -> UUID | None:
        return self._heading_stack[-1][1] if self._heading_stack else None

    def _add_element(
        self,
        element_type: ElementType,
        text: str,
        tag: Tag | None = None,
        *,
        parent_id: UUID | None = None,
        structured_data: dict | None = None,
        metadata: dict | None = None,
    ) -> DocumentElement:
        element = DocumentElement(
            element_type=element_type,
            text=text,
            order=self._order,
            parent_id=parent_id if parent_id is not None else self._current_parent(),
            structured_data=structured_data or {},
            metadata=metadata or {},
        )

        self._elements.append(element)
        self._order += 1

        return element

    @staticmethod
    def _clean_text(tag: Tag) -> str:
        return tag.get_text(" ", strip=True)

    def _extract_list(self, tag: Tag) -> None:
        for item in tag.find_all("li", recursive=False):
            text = self._clean_text(item)

            if text:
                self._add_element(
                    ElementType.LIST_ITEM,
                    text,
                    item,
                    metadata={
                        "list_type": tag.name,
                    },
                )

            # Preserve nested lists as separate list items.
            for nested_list in item.find_all(["ul", "ol"], recursive=False):
                self._extract_list(nested_list)

    def _extract_table(self, tag: Tag) -> None:
        rows = []

        for row in tag.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True)
                for cell in row.find_all(["th", "td"], recursive=False)
            ]

            if cells:
                rows.append(cells)

        if rows:
            self._add_element(
                ElementType.TABLE,
                "\n".join(" | ".join(row) for row in rows),
                tag,
                structured_data={"rows": rows},
            )

    def _extract_code(self, tag: Tag) -> None:
        code_tag = tag.find("code") or tag
        language = None

        classes = code_tag.get("class", [])

        for class_name in classes:
            if class_name.startswith("language-"):
                language = class_name.removeprefix("language-")
                break

        self._add_element(
            ElementType.CODE,
            code_tag.get_text(),
            tag,
            metadata={"language": language},
        )

    def _walk(self, container: Tag) -> None:
        for child in container.children:
            if not isinstance(child, Tag):
                continue

            if child.name in self.IGNORED_TAGS:
                continue

            if child.name in {"article", "main", "section", "div", "body"}:
                self._walk(child)
                continue

            if child.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                level = int(child.name[1])

                while self._heading_stack and self._heading_stack[-1][0] >= level:
                    self._heading_stack.pop()

                element = self._add_element(
                    ElementType.HEADING,
                    self._clean_text(child),
                    child,
                    parent_id=self._current_parent(),
                    metadata={"heading_level": level},
                )

                self._heading_stack.append((level, element.id))
                continue

            if child.name == "p":
                text = self._clean_text(child)

                if text:
                    links = [
                        {
                            "text": link.get_text(" ", strip=True),
                            "href": link.get("href"),
                        }
                        for link in child.find_all("a")
                    ]

                    self._add_element(
                        ElementType.PARAGRAPH,
                        text,
                        child,
                        metadata={"links": links},
                    )

                continue

            if child.name in {"ul", "ol"}:
                self._extract_list(child)
                continue

            if child.name == "table":
                self._extract_table(child)
                continue

            if child.name == "pre":
                self._extract_code(child)
                continue

            if child.name == "blockquote":
                text = self._clean_text(child)

                if text:
                    self._add_element(
                        ElementType.PARAGRAPH,
                        text,
                        child,
                        metadata={"block_type": "blockquote"},
                    )

                continue

            self._walk(child)

    def parse(self, source: Path) -> Document:
        source = Path(source)

        if not source.is_file():
            raise FileNotFoundError(f"Document not found: {source}")

        raw_content = source.read_bytes()
        content = raw_content.decode("utf-8")

        soup = BeautifulSoup(content, "html.parser")

        # Reset parser state for every document.
        self._heading_stack = []
        self._elements = []
        self._order = 0

        html_tag = soup.find("html")
        title_tag = soup.find("title")
        author_tag = soup.find(
            "meta",
            attrs={"name": "author"},
        )

        root = soup.find("main") or soup.find("article") or soup.body or soup

        self._walk(root)

        return Document(
            source_uri=str(source.resolve()),
            content_hash=hashlib.sha256(raw_content).hexdigest(),
            metadata=DocumentMetadata(
                title=title_tag.get_text(strip=True) if title_tag else None,
                author=author_tag.get("content") if author_tag else None,
                language=html_tag.get("lang") if html_tag else None,
                mime_type="text/html",
                file_name=source.name,
            ),
            elements=self._elements,
        )
