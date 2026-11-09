from uuid import UUID

from pydantic import BaseModel, Field

from norag.generation.models import GenerationRequest


class PromptConfig(BaseModel):
    """Configuration controlling grounded answer generation."""

    system_prompt: str = (
        "You are NORAG, an evidence-grounded research assistant. "
        "Answer only what the user asked, in at most three sentences, "
        "using only the supplied retrieved evidence, and end every "
        "sentence with its source label, such as [Source 1]. "
        "Never add projects, achievements, examples, or background that "
        "the question did not ask for. Never fabricate facts, numbers, "
        "or citations. If the evidence does not establish something, "
        "say so explicitly in one sentence."
    )

    max_input_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=768, ge=1)


class PromptSource(BaseModel):
    """Maps a prompt citation to its original retrieved chunk."""

    source_id: str = Field(min_length=1)
    position: int = Field(ge=1)
    document_id: UUID
    chunk_id: str = Field(min_length=1)
    section: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    source_element_ids: list[UUID] = Field(default_factory=list)


class PromptConstruction(BaseModel):
    """Complete generation request with citation metadata."""

    request: GenerationRequest
    sources: list[PromptSource]
    estimated_input_tokens: int = Field(ge=0)
