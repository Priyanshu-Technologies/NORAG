from math import ceil
from typing import Any

from norag.context.models import AssembledContext
from norag.generation.models import GenerationRequest
from norag.prompting.interfaces import TokenCounter
from norag.prompting.models import (
    PromptConfig,
    PromptConstruction,
    PromptSource,
)


class ApproximatePromptTokenCounter:
    """Estimate tokens using approximately four characters per token."""

    def count(self, text: str) -> int:
        if not text:
            return 0

        return ceil(len(text) / 4)


class PromptBuilder:
    """Construct grounded, citation-aware generation prompts."""

    def __init__(
        self,
        config: PromptConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.config = config or PromptConfig()
        self.token_counter = token_counter or ApproximatePromptTokenCounter()

    def build(
        self,
        query: str,
        context: AssembledContext,
    ) -> PromptConstruction:
        """Build a generation request from a query and assembled context."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        evidence = context.context_text.strip()

        if not evidence:
            evidence = "No retrieved evidence was available."

        user_prompt = self._build_user_prompt(
            query=query.strip(),
            evidence=evidence,
        )

        estimated_tokens = self.token_counter.count(
            f"{self.config.system_prompt}\n{user_prompt}"
        )

        if estimated_tokens > self.config.max_input_tokens:
            raise ValueError(
                "Constructed prompt exceeds the input token budget: "
                f"{estimated_tokens} > {self.config.max_input_tokens}."
            )

        request = GenerationRequest(
            system_prompt=self.config.system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )

        return PromptConstruction(
            request=request,
            sources=self._build_sources(context),
            estimated_input_tokens=estimated_tokens,
        )

    def _build_user_prompt(
        self,
        query: str,
        evidence: str,
    ) -> str:
        return (
            f"Question:\n{query}\n\n"
            f"Retrieved evidence:\n{evidence}\n\n"
            "Answering requirements:\n"
            "- Answer using only the retrieved evidence.\n"
            "- Cite factual claims using the exact source labels "
            "provided, such as [Source 1].\n"
            "- Never invent citations or supporting facts.\n"
            "- If the evidence is insufficient, explicitly explain "
            "what cannot be established.\n"
            "- Distinguish facts from reasonable interpretations.\n"
        )

    def _build_sources(
        self,
        context: AssembledContext,
    ) -> list[PromptSource]:
        sources = []

        for item in context.items:
            hybrid_result = item.result.result
            chunk: Any = hybrid_result.chunk

            document_id = chunk.document_id
            chunk_id = (
                getattr(chunk, "id", None)
                or getattr(chunk, "chunk_id", None)
                or f"{document_id}:{getattr(chunk, 'index', item.position)}"
            )

            metadata = chunk.metadata or {}
            section = metadata.get("section_path")

            if isinstance(section, list):
                section = " > ".join(str(part) for part in section)

            page_numbers = sorted(
                {
                    location.page_number
                    for location in chunk.source_locations
                    if location.page_number is not None
                }
            )

            sources.append(
                PromptSource(
                    source_id=f"Source {item.position}",
                    position=item.position,
                    document_id=document_id,
                    chunk_id=str(chunk_id),
                    section=section,
                    page_numbers=page_numbers,
                    source_element_ids=chunk.source_element_ids,
                )
            )

        return sources
