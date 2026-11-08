from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class EvaluationCase(BaseModel):
    """A labelled query with its known relevant chunks."""

    query: str = Field(min_length=1)
    relevant_chunk_ids: set[UUID] = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query cannot be empty")
        return value


class RankingMetrics(BaseModel):
    """Retrieval quality metrics for one query."""

    recall_at_k: float = Field(ge=0, le=1)
    precision_at_k: float = Field(ge=0, le=1)
    mrr_at_k: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)


class QueryEvaluation(BaseModel):
    """Evaluation results for an individual query."""

    query: str
    relevant_count: int = Field(gt=0)
    retrieved_count: int = Field(ge=0)
    metrics: RankingMetrics


class EvaluationReport(BaseModel):
    """Aggregated evaluation report across multiple queries."""

    k: int = Field(ge=1)
    query_count: int = Field(gt=0)

    average_metrics: RankingMetrics
    per_query: list[QueryEvaluation]
