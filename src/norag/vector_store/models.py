from pydantic import BaseModel

from norag.embeddings.models import EmbeddedChunk


class VectorSearchResult(BaseModel):
    """A retrieved chunk and its similarity score."""

    chunk: EmbeddedChunk
    score: float
