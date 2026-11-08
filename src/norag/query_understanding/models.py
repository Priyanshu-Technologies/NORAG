from enum import StrEnum

from pydantic import BaseModel, Field

from norag.hybrid_retrieval.models import HybridSearchResult


class QueryIntent(StrEnum):
    SEMANTIC = "semantic"
    EXACT = "exact"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi_hop"


class RetrievalRoute(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class QueryAnalysis(BaseModel):
    """Classification and retrieval strategy for a query."""

    query: str = Field(min_length=1)
    intent: QueryIntent
    route: RetrievalRoute

    dense_weight: float = Field(ge=0)
    sparse_weight: float = Field(ge=0)

    rule: str


class RoutedSearchResponse(BaseModel):
    """Query analysis and its resulting retrieval results."""

    analysis: QueryAnalysis
    results: list[HybridSearchResult]
