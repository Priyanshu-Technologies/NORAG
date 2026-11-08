from uuid import UUID

from pydantic import BaseModel, Field

from norag.generation.models import GenerationRequest


class PromptConfig(BaseModel):
    """Configuration for prompt construction."""

    system_prompt: str = (
        "You are NORAG, a technical research assistant. "
        "Provide accurate, clear, evidence-grounded answers."
    )
    max_input_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=1)


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
