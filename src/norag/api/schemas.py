from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    file_name: str
    content_hash: str
    chunk_count: int


class DocumentSummary(BaseModel):
    document_id: UUID
    file_name: str | None = None
    title: str | None = None
    content_hash: str
    created_at: datetime


class DeleteDocumentResponse(BaseModel):
    document_id: UUID
    deleted: bool


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=50)


class QueryResponse(BaseModel):
    result: dict[str, Any]
