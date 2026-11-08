import pytest

from norag.query_understanding.models import (
    QueryAnalysis,
    QueryIntent,
    RetrievalRoute,
)
from norag.retrieval_policy import AdaptiveRetrievalService, RetrievalPlan


def make_analysis(query: str) -> QueryAnalysis:
    return QueryAnalysis(
        query=query,
        intent=QueryIntent.SEMANTIC,
        route=RetrievalRoute.HYBRID,
        dense_weight=0.5,
        sparse_weight=0.5,
        rule="test",
    )


def test_service_connects_all_retrieval_stages() -> None:
    calls = []

    def analyze(query: str) -> QueryAnalysis:
        calls.append(("analyze", query))
        return make_analysis(query)

    def retrieve(query: str, plan: RetrievalPlan) -> list:
        calls.append(("retrieve", query, plan.candidate_k))
        return []

    def rerank(query: str, candidates: list, top_k: int) -> list:
        calls.append(("rerank", query, top_k))
        return []

    service = AdaptiveRetrievalService(
        analyze=analyze,
        retrieve=retrieve,
        rerank=rerank,
    )

    response = service.search("How does RAG work?", top_k=3)

    assert response.analysis.intent == QueryIntent.SEMANTIC
    assert response.plan.final_top_k == 3
    assert response.plan.candidate_k == 9
    assert response.results == []

    assert [call[0] for call in calls] == [
        "analyze",
        "retrieve",
        "rerank",
    ]


def test_service_passes_adaptive_plan_to_retriever() -> None:
    received_plans = []

    def retrieve(query: str, plan: RetrievalPlan) -> list:
        received_plans.append(plan)
        return []

    service = AdaptiveRetrievalService(
        analyze=make_analysis,
        retrieve=retrieve,
        rerank=lambda query, candidates, top_k: [],
    )

    service.search("Find an exact identifier", top_k=4)

    assert len(received_plans) == 1
    assert received_plans[0].candidate_k == 12


def test_service_rejects_empty_query() -> None:
    service = AdaptiveRetrievalService(
        analyze=make_analysis,
        retrieve=lambda query, plan: [],
        rerank=lambda query, candidates, top_k: [],
    )

    with pytest.raises(ValueError, match="Query cannot be empty"):
        service.search("   ")


def test_service_rejects_invalid_top_k() -> None:
    service = AdaptiveRetrievalService(
        analyze=make_analysis,
        retrieve=lambda query, plan: [],
        rerank=lambda query, candidates, top_k: [],
    )

    with pytest.raises(ValueError, match="top_k"):
        service.search("Test query", top_k=0)


def test_service_limits_final_results_to_top_k() -> None:
    rerank_calls = []

    def rerank(query: str, candidates: list, top_k: int) -> list:
        rerank_calls.append(top_k)
        return []

    service = AdaptiveRetrievalService(
        analyze=make_analysis,
        retrieve=lambda query, plan: [],
        rerank=rerank,
    )

    response = service.search("Test query", top_k=2)

    assert rerank_calls == [2]
    assert len(response.results) <= 2
