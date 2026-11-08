from pydantic import BaseModel, Field

from norag.chunking.models import Chunk
from norag.embeddings.models import EmbeddedChunk


class HybridSearchResult(BaseModel):
    """A chunk retrieved through hybrid search."""

    chunk: Chunk | EmbeddedChunk
    score: float = Field(ge=0)

    dense_rank: int | None = Field(default=None, ge=1)
    sparse_rank: int | None = Field(default=None, ge=1)

    dense_score: float | None = None
    sparse_score: float | None = None
