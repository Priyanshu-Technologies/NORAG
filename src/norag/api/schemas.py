from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from norag.answering.models import AnswerResponse
from norag.evaluation.models import EvaluationReport


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


class DocumentStats(BaseModel):
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    dense_chunks: int = Field(ge=0)
    sparse_chunks: int = Field(ge=0)
    queries: int = Field(ge=0)
    uploads: int = Field(ge=0)
    deletions: int = Field(ge=0)
    errors: int = Field(ge=0)
    environment: str
    ollama_model: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=50)
    system: Literal["dense", "sparse", "hybrid", "hybrid_rerank"] | None = None


class QueryResponse(BaseModel):
    """Typed answer trace returned by the query endpoint."""

    result: AnswerResponse


class EvaluationCaseRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    relevant_chunk_ids: list[UUID] = Field(min_length=1)


class EvaluationRequest(BaseModel):
    system: Literal["dense", "sparse", "hybrid", "hybrid_rerank"] = "hybrid"
    k: int = Field(default=5, ge=1, le=50)
    cases: list[EvaluationCaseRequest] = Field(min_length=1, max_length=200)


class EvaluationResponse(BaseModel):
    report: EvaluationReport
