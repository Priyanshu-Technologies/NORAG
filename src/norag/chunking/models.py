from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from norag.ingestion.models import SourceLocation


class Chunk(BaseModel):
    """A retrievable unit derived from a source document."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    index: int = Field(ge=0)

    text: str = Field(min_length=1)
    token_count: int = Field(ge=0)

    source_element_ids: list[UUID] = Field(default_factory=list)
    source_locations: list[SourceLocation] = Field(default_factory=list)

    metadata: dict = Field(default_factory=dict)


class ChunkingConfig(BaseModel):
    """Configuration controlling chunk generation."""

    target_tokens: int = Field(default=400, gt=0)
    max_tokens: int = Field(default=500, gt=0)
    overlap_tokens: int = Field(default=50, ge=0)

    def validate_limits(self) -> "ChunkingConfig":
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens cannot exceed max_tokens")

        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")

        return self
