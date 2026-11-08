from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ElementType(StrEnum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    CODE = "code"
    EQUATION = "equation"
    UNKNOWN = "unknown"


class SourceLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_number: int | None = Field(default=None, ge=1)

    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    # Coordinates: x0, y0, x1, y1
    bounding_box: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def validator_offsets(self):
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError("char_end cannot be smaller than char_start")

        return self

    @model_validator(mode="after")
    def validate_bounding_box(self):
        if self.bounding_box is not None:
            x0, y0, x1, y1 = self.bounding_box

            if x1 < x0 or y1 < y0:
                raise ValueError("Invalid bounding box coordinates")

        return self


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    author: str | None = None
    language: str | None = None
    mime_type: str | None = None
    file_name: str | None = None

    custom_fields: dict[str, Any] = Field(default_factory=dict)


class DocumentElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)

    element_type: ElementType

    text: str | None = None

    structured_data: dict[str, Any] = Field(default_factory=dict)

    order: int = Field(ge=0)

    parent_id: UUID | None = None

    source_location: SourceLocation = Field(default_factory=SourceLocation)

    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)

    source_uri: str | None = None

    content_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

    elements: list[DocumentElement] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    schema_version: str = "1.0"

    @model_validator(mode="after")
    def validate_elements(self):
        element_ids = [element.id for element in self.elements]
        orders = [element.order for element in self.elements]

        if len(element_ids) != len(set(element_ids)):
            raise ValueError("Document contains duplicate element IDs")

        if len(orders) != len(set(orders)):
            raise ValueError("Document contains duplicate element orders")

        return self
