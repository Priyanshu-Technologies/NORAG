import math
from collections.abc import Callable

from norag.chunking.interfaces import Chunker
from norag.chunking.models import Chunk, ChunkingConfig
from norag.ingestion.models import Document, DocumentElement, ElementType


class ApproximateTokenCounter:
    """Estimate tokens using approximately four characters per token."""

    def count(self, text: str) -> int:
        return math.ceil(len(text) / 4) if text else 0


class StructuralChunker(Chunker):
    """Create structure-aware chunks with token-aware overlap."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.config = (config or ChunkingConfig()).validate_limits()
        self._count_tokens = token_counter or ApproximateTokenCounter().count

    def _split_text(self, text: str, budget: int) -> list[str]:
        """Split text into pieces that fit the available token budget."""

        words = text.split()
        parts: list[str] = []
        current: list[str] = []

        for word in words:
            candidate = " ".join([*current, word])

            if current and self._count_tokens(candidate) > budget:
                parts.append(" ".join(current))
                current = [word]
            else:
                current.append(word)

        if current:
            parts.append(" ".join(current))

        return parts

    def _get_overlap(
        self,
        segments: list[tuple[str, DocumentElement]],
    ) -> list[tuple[str, DocumentElement]]:
        """Get the smallest word-aligned suffix within the overlap budget."""

        if self.config.overlap_tokens == 0:
            return []

        words = [(word, element) for text, element in segments for word in text.split()]

        suffix: list[tuple[str, DocumentElement]] = []

        for item in reversed(words):
            candidate = " ".join([item[0], *(word for word, _ in suffix)])

            if self._count_tokens(candidate) > self.config.overlap_tokens:
                break

            suffix.insert(0, item)

        grouped: list[tuple[str, DocumentElement]] = []

        for word, element in suffix:
            if grouped and grouped[-1][1].id == element.id:
                previous_text, previous_element = grouped[-1]
                grouped[-1] = (
                    f"{previous_text} {word}",
                    previous_element,
                )
            else:
                grouped.append((word, element))

        return grouped

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        heading_stack: list[tuple[int, str]] = []
        current_segments: list[tuple[str, DocumentElement]] = []

        def section_context() -> str:
            return " > ".join(title for _, title in heading_stack)

        def render(
            segments: list[tuple[str, DocumentElement]],
        ) -> str:
            body = "\n\n".join(text for text, _ in segments)
            context = section_context()
            return f"{context}\n\n{body}" if context else body

        def flush(keep_overlap: bool = False) -> None:
            nonlocal current_segments

            if not current_segments:
                return

            text = render(current_segments)

            # Preserve source elements without duplicate IDs.
            unique_elements = list(
                {element.id: element for _, element in current_segments}.values()
            )

            chunks.append(
                Chunk(
                    document_id=document.id,
                    index=len(chunks),
                    text=text,
                    token_count=self._count_tokens(text),
                    source_element_ids=[element.id for element in unique_elements],
                    source_locations=[
                        element.source_location
                        for element in unique_elements
                        if element.source_location is not None
                    ],
                    metadata={
                        "section_path": section_context(),
                        "element_types": [
                            element.element_type.value for element in unique_elements
                        ],
                        "strategy": "structural",
                        "overlap_tokens": self.config.overlap_tokens
                        if keep_overlap
                        else 0,
                    },
                )
            )

            current_segments = (
                self._get_overlap(current_segments) if keep_overlap else []
            )

        for element in document.elements:
            if element.element_type == ElementType.HEADING:
                # Never carry overlap across section boundaries.
                flush()

                level = int(element.metadata.get("heading_level", 1))

                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()

                heading_stack.append((level, element.text))
                continue

            if not element.text.strip():
                continue

            context = section_context()
            context_cost = self._count_tokens(context) if context else 0

            budget = self.config.max_tokens - context_cost

            if budget < 1:
                raise ValueError("Heading context leaves no room for chunk content")

            pieces = self._split_text(element.text, budget)

            for piece in pieces:
                candidate_segments = [
                    *current_segments,
                    (piece, element),
                ]
                candidate_text = render(candidate_segments)
                candidate_tokens = self._count_tokens(candidate_text)

                if current_segments and (candidate_tokens > self.config.target_tokens):
                    flush(keep_overlap=True)

                    candidate_segments = [
                        *current_segments,
                        (piece, element),
                    ]
                    candidate_text = render(candidate_segments)

                    # If overlap prevents the next piece from fitting,
                    # discard overlap rather than exceed the maximum.
                    if self._count_tokens(candidate_text) > self.config.max_tokens:
                        current_segments = []
                        candidate_segments = [(piece, element)]

                current_segments.append((piece, element))

        flush()

        return chunks
