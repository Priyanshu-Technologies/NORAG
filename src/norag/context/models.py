from pydantic import BaseModel, Field

from norag.reranking.models import RerankedSearchResult


class ContextConfig(BaseModel):
    max_tokens: int = Field(default=3000, ge=1)
    min_relevance_score: float | None = None
    deduplicate: bool = True


class ContextItem(BaseModel):
    result: RerankedSearchResult
    position: int = Field(ge=1)
    token_count: int = Field(ge=0)


class AssembledContext(BaseModel):
    items: list[ContextItem]
    context_text: str
    token_count: int = Field(ge=0)
    available_tokens: int = Field(ge=0)
    included_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
