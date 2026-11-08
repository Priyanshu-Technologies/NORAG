from unittest.mock import Mock

from norag.hybrid_retrieval.service import HybridRetriever
from norag.query_understanding.classifier import QueryClassifier
from norag.query_understanding.models import (
    QueryAnalysis,
    QueryIntent,
    RetrievalRoute,
)
from norag.reranking.service import RerankingService
from norag.retrieval_policy.factory import (
    create_adaptive_retrieval_service,
)


def test_factory_connects_actual_components() -> None:
    classifier = Mock(spec=QueryClassifier)
    hybrid = Mock(spec=HybridRetriever)
    reranker = Mock(spec=RerankingService)

    classifier.classify.return_value = QueryAnalysis(
        query="Explain how RAG works",
        intent=QueryIntent.SEMANTIC,
        route=RetrievalRoute.DENSE,
        dense_weight=1.0,
        sparse_weight=0.0,
        rule="semantic_fallback",
    )

    hybrid.search.return_value = []
    reranker.rerank.return_value = []

    service = create_adaptive_retrieval_service(
        classifier=classifier,
        hybrid_retriever=hybrid,
        reranking_service=reranker,
    )

    response = service.search("Explain how RAG works", top_k=3)

    classifier.classify.assert_called_once_with("Explain how RAG works")

    hybrid.search.assert_called_once_with(
        query="Explain how RAG works",
        top_k=9,
        dense_weight=0.80,
        sparse_weight=0.20,
    )

    reranker.rerank.assert_called_once_with(
        "Explain how RAG works",
        [],
        3,
    )

    assert response.plan.candidate_k == 9
    assert response.results == []
