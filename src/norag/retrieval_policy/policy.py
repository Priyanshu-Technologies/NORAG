from math import ceil
from typing import ClassVar

from norag.query_understanding.models import QueryAnalysis, QueryIntent
from norag.retrieval_policy.models import RetrievalPlan


class AdaptiveRetrievalPolicy:
    """Select retrieval settings based on the detected query intent."""

    DEFAULT_SETTINGS: ClassVar[dict[QueryIntent, tuple[int, float, float]]] = {
        QueryIntent.SEMANTIC: (3, 0.80, 0.20),
        QueryIntent.EXACT: (2, 0.20, 0.80),
        QueryIntent.COMPARATIVE: (5, 0.50, 0.50),
        QueryIntent.MULTI_HOP: (6, 0.55, 0.45),
    }

    def __init__(
        self,
        min_candidates: int = 5,
        max_candidates: int = 100,
    ) -> None:
        if min_candidates < 1:
            raise ValueError("min_candidates must be positive")

        if max_candidates < min_candidates:
            raise ValueError(
                "max_candidates must be greater than or equal to min_candidates"
            )

        self.min_candidates = min_candidates
        self.max_candidates = max_candidates

    def create_plan(
        self,
        analysis: QueryAnalysis,
        top_k: int = 5,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> RetrievalPlan:
        if top_k < 1:
            raise ValueError("top_k must be positive")

        if (dense_weight is None) != (sparse_weight is None):
            raise ValueError("dense_weight and sparse_weight must be provided together")

        multiplier, default_dense, default_sparse = self.DEFAULT_SETTINGS[
            analysis.intent
        ]

        effective_dense = default_dense if dense_weight is None else dense_weight
        effective_sparse = default_sparse if sparse_weight is None else sparse_weight

        if top_k > self.max_candidates:
            raise ValueError("top_k cannot exceed max_candidates")

        candidate_k = ceil(top_k * multiplier)
        candidate_k = max(candidate_k, self.min_candidates, top_k)
        candidate_k = min(candidate_k, self.max_candidates)

        plan = RetrievalPlan(
            intent=analysis.intent,
            final_top_k=top_k,
            candidate_k=candidate_k,
            dense_weight=effective_dense,
            sparse_weight=effective_sparse,
        )

        return plan.validate_weights()
