from pydantic import BaseModel

from norag.chunking.models import Chunk


class SparseSearchResult(BaseModel):
    """A chunk retrieved through sparse lexical search."""

    chunk: Chunk
    score: float
