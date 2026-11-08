import math

from norag.hybrid_retrieval.models import HybridSearchResult
from norag.reranking.interfaces import RerankerProvider
from norag.reranking.models import RerankedSearchResult


class RerankingService:
    """Rerank hybrid search candidates using a relevance model."""

    def __init__(
        self,
        provider: RerankerProvider,
    ) -> None:
        self.provider = provider

    def rerank(
        self,
        query: str,
        candidates: list[HybridSearchResult],
        top_k: int | None = None,
    ) -> list[RerankedSearchResult]:
        """Reorder candidates using cross-encoder relevance scores."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")

        if not candidates:
            return []

        documents = [candidate.chunk.text for candidate in candidates]

        scores = self.provider.score(
            query,
            documents,
        )

        if len(scores) != len(candidates):
            raise RuntimeError("Reranker returned an unexpected number of scores")

        ranked_results: list[RerankedSearchResult] = []

        for original_rank, (candidate, score) in enumerate(
            zip(candidates, scores, strict=True),
            start=1,
        ):
            if not math.isfinite(score):
                raise ValueError("Reranker returned a non-finite score")

            ranked_results.append(
                RerankedSearchResult(
                    result=candidate,
                    rerank_score=score,
                    original_rank=original_rank,
                )
            )

        # Python's sort is stable, so tied scores preserve the
        # original retrieval order.
        ranked_results.sort(
            key=lambda item: item.rerank_score,
            reverse=True,
        )

        if top_k is not None:
            return ranked_results[:top_k]

        return ranked_results
