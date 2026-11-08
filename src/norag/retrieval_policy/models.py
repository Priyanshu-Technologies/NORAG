from pydantic import BaseModel, Field

from norag.query_understanding.models import QueryIntent


class RetrievalPlan(BaseModel):
    intent: QueryIntent
    final_top_k: int = Field(ge=1)
    candidate_k: int = Field(ge=1)
    dense_weight: float = Field(ge=0, le=1)
    sparse_weight: float = Field(ge=0, le=1)

    def validate_weights(self) -> "RetrievalPlan":
        total = self.dense_weight + self.sparse_weight

        if abs(total - 1.0) > 1e-6:
            raise ValueError("Retrieval weights must sum to 1")

        return self
