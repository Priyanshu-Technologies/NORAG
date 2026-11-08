from collections.abc import Callable, Sequence

from pydantic import BaseModel

from norag.hybrid_retrieval.models import HybridSearchResult
from norag.query_understanding.models import QueryAnalysis
from norag.reranking.models import RerankedSearchResult
from norag.retrieval_policy.models import RetrievalPlan
from norag.retrieval_policy.policy import AdaptiveRetrievalPolicy

AnalyzeFunction = Callable[[str], QueryAnalysis]

RetrieveFunction = Callable[
    [str, RetrievalPlan],
    Sequence[HybridSearchResult],
]

RerankFunction = Callable[
    [str, Sequence[HybridSearchResult], int],
    Sequence[RerankedSearchResult],
]


class AdaptiveRetrievalResponse(BaseModel):
    analysis: QueryAnalysis
    plan: RetrievalPlan
    results: list[RerankedSearchResult]


class AdaptiveRetrievalService:
    """Coordinate query analysis, adaptive retrieval, and reranking."""

    def __init__(
        self,
        analyze: AnalyzeFunction,
        retrieve: RetrieveFunction,
        rerank: RerankFunction,
        policy: AdaptiveRetrievalPolicy | None = None,
    ) -> None:
        self.analyze = analyze
        self.retrieve = retrieve
        self.rerank = rerank
        self.policy = policy or AdaptiveRetrievalPolicy()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> AdaptiveRetrievalResponse:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        analysis = self.analyze(query)

        plan = self.policy.create_plan(
            analysis=analysis,
            top_k=top_k,
        )

        candidates = self.retrieve(query, plan)

        ranked_results = self.rerank(
            query,
            candidates,
            top_k,
        )

        return AdaptiveRetrievalResponse(
            analysis=analysis,
            plan=plan,
            results=list(ranked_results)[:top_k],
        )
