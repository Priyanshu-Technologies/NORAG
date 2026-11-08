from pathlib import Path

from norag.ingestion.interfaces import DocumentParser
from norag.ingestion.models import Document


class ParserRegistry:
    """Registry for document parsers."""

    def __init__(self) -> None:
        self._parsers: dict[str, DocumentParser] = {}

    def register(self, parser: DocumentParser) -> None:
        extensions = tuple(
            extension.lower() for extension in parser.supported_extensions
        )

        if not extensions:
            raise ValueError("Parser must support at least one extension")

        if any(not extension.startswith(".") for extension in extensions):
            raise ValueError("Extensions must start with a dot")

        if len(extensions) != len(set(extensions)):
            raise ValueError("Parser contains duplicate extensions")

        conflicts = set(extensions) & self._parsers.keys()

        if conflicts:
            raise ValueError(f"Parser already registered for: {sorted(conflicts)}")

        for extension in extensions:
            self._parsers[extension] = parser

    def get_parser(self, source: Path) -> DocumentParser:
        extension = Path(source).suffix.lower()

        try:
            return self._parsers[extension]
        except KeyError:
            raise ValueError(
                f"No parser registered for extension: {extension or '<none>'}"
            ) from None

    def parse(self, source: Path) -> Document:
        return self.get_parser(source).parse(Path(source))
