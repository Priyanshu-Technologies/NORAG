import math

from norag.context.models import (
    AssembledContext,
    ContextConfig,
    ContextItem,
)
from norag.reranking.models import RerankedSearchResult


class ApproximateContextTokenCounter:
    """Estimate tokens using approximately four characters per token."""

    def count(self, text: str) -> int:
        return math.ceil(len(text) / 4) if text else 0


class ContextAssembler:
    """Select and assemble reranked evidence within a token budget."""

    def __init__(
        self,
        config: ContextConfig | None = None,
        token_counter=None,
    ) -> None:
        self.config = config or ContextConfig()
        self.token_counter = token_counter or ApproximateContextTokenCounter()

    def _is_duplicate(
        self,
        candidate: RerankedSearchResult,
        selected: list[ContextItem],
    ) -> bool:
        candidate_chunk = candidate.result.chunk

        for item in selected:
            existing_chunk = item.result.result.chunk

            if candidate_chunk.id == existing_chunk.id:
                return True

            if candidate_chunk.text.strip() == existing_chunk.text.strip():
                return True

        return False

    def assemble(
        self,
        results: list[RerankedSearchResult],
        reserved_tokens: int = 0,
    ) -> AssembledContext:
        if reserved_tokens < 0:
            raise ValueError("reserved_tokens cannot be negative")

        if reserved_tokens >= self.config.max_tokens:
            raise ValueError("Reserved tokens exceed context budget")

        available_tokens = self.config.max_tokens - reserved_tokens

        selected: list[ContextItem] = []
        seen_count = 0
        used_tokens = 0
        sections: list[str] = []

        for result in results:
            seen_count += 1

            if (
                self.config.min_relevance_score is not None
                and result.rerank_score < self.config.min_relevance_score
            ):
                continue

            chunk = result.result.chunk

            if self.config.deduplicate and self._is_duplicate(
                result,
                selected,
            ):
                continue

            section = chunk.metadata.get("section_path", "Unknown section")

            formatted_text = (
                f"[Source {len(selected) + 1}]\n"
                f"Section: {section}\n"
                f"Content:\n{chunk.text}"
            )

            token_count = self.token_counter.count(formatted_text)

            if used_tokens + token_count > available_tokens:
                continue

            selected.append(
                ContextItem(
                    result=result,
                    position=len(selected) + 1,
                    token_count=token_count,
                )
            )

            sections.append(formatted_text)
            used_tokens += token_count

        return AssembledContext(
            items=selected,
            context_text="\n\n".join(sections),
            token_count=used_tokens,
            available_tokens=available_tokens,
            included_count=len(selected),
            excluded_count=seen_count - len(selected),
        )
