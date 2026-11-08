import math
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from norag.ingestion.models import SourceLocation


class EmbeddedChunk(BaseModel):
    """A chunk together with its numerical embedding."""

    chunk_id: UUID
    document_id: UUID
    index: int = Field(ge=0)

    text: str = Field(min_length=1)
    vector: list[float]
    dimensions: int = Field(gt=0)
    model_name: str = Field(min_length=1)

    source_element_ids: list[UUID] = Field(default_factory=list)
    source_locations: list[SourceLocation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("Embedding vector cannot be empty")

        if not all(math.isfinite(item) for item in value):
            raise ValueError("Embedding vector must contain finite values")

        return value

    @model_validator(mode="after")
    def validate_dimensions(self) -> "EmbeddedChunk":
        if len(self.vector) != self.dimensions:
            raise ValueError("Vector length does not match declared dimensions")

        return self

    @property
    def id(self) -> UUID:
        """Expose a consistent chunk ID interface."""
        return self.chunk_id
