from pydantic import BaseModel, Field

from norag.hybrid_retrieval.models import HybridSearchResult


class RerankedSearchResult(BaseModel):
    """A hybrid result with its cross-encoder relevance score."""

    result: HybridSearchResult
    rerank_score: float
    original_rank: int = Field(ge=1)
