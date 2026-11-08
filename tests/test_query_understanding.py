from uuid import uuid4

import pytest

from norag.query_understanding import (
    QueryClassifier,
    QueryIntent,
    QueryRouter,
    RetrievalRoute,
)


def test_classifier_detects_semantic_query():
    analysis = QueryClassifier().classify("Explain how transformer models work")

    assert analysis.intent == QueryIntent.SEMANTIC
    assert analysis.route == RetrievalRoute.DENSE
    assert analysis.dense_weight == 1.0
    assert analysis.sparse_weight == 0.0


def test_classifier_detects_exact_identifier():
    analysis = QueryClassifier().classify("Find references to YOLO11n-seg")

    assert analysis.intent == QueryIntent.EXACT
    assert analysis.route == RetrievalRoute.SPARSE
    assert analysis.dense_weight == 0.0
    assert analysis.sparse_weight == 1.0


def test_classifier_detects_acronym():
    analysis = QueryClassifier().classify("Find references to RAG")

    assert analysis.intent == QueryIntent.EXACT


def test_classifier_detects_version():
    analysis = QueryClassifier().classify("Find references to version 2.5.1")

    assert analysis.intent == QueryIntent.EXACT


def test_classifier_detects_quoted_phrase():
    analysis = QueryClassifier().classify('Find the phrase "attention is all you need"')

    assert analysis.intent == QueryIntent.EXACT


def test_classifier_detects_comparative_query():
    analysis = QueryClassifier().classify("Compare BERT and RoBERTa")

    assert analysis.intent == QueryIntent.COMPARATIVE
    assert analysis.route == RetrievalRoute.HYBRID


def test_classifier_detects_versus():
    analysis = QueryClassifier().classify("YOLO11 versus Faster R-CNN")

    assert analysis.intent == QueryIntent.COMPARATIVE


def test_classifier_detects_difference_query():
    analysis = QueryClassifier().classify("What is the difference between CNN and RNN?")

    assert analysis.intent == QueryIntent.COMPARATIVE


def test_classifier_detects_multi_hop_query():
    analysis = QueryClassifier().classify(
        "Which model performed best, and what dataset was used?"
    )

    assert analysis.intent == QueryIntent.MULTI_HOP
    assert analysis.route == RetrievalRoute.HYBRID


def test_classifier_detects_based_on_query():
    analysis = QueryClassifier().classify(
        "Based on the results, which model is most accurate?"
    )

    assert analysis.intent == QueryIntent.MULTI_HOP


def test_comparison_precedes_multi_hop_pattern():
    analysis = QueryClassifier().classify(
        "Compare the models and then identify the best dataset"
    )

    assert analysis.intent == QueryIntent.COMPARATIVE


def test_comparison_has_precedence_over_exact_identifiers():
    analysis = QueryClassifier().classify("Compare YOLO11 and BERT")

    assert analysis.intent == QueryIntent.COMPARATIVE


def test_classifier_provides_rule_explanation():
    analysis = QueryClassifier().classify("Find references to YOLO11")

    assert analysis.rule == "exact_identifier_pattern"


def test_classifier_rejects_empty_query():
    with pytest.raises(ValueError, match="empty"):
        QueryClassifier().classify("   ")


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return []


def test_router_analyzes_without_searching():
    retriever = FakeRetriever()
    router = QueryRouter(retriever)

    analysis = router.analyze("Explain neural networks")

    assert analysis.intent == QueryIntent.SEMANTIC
    assert retriever.calls == []


def test_router_uses_dense_strategy_for_semantic_query():
    retriever = FakeRetriever()
    router = QueryRouter(retriever)

    router.search("Explain neural networks")

    call = retriever.calls[0]

    assert call["dense_weight"] == 1.0
    assert call["sparse_weight"] == 0.0


def test_router_uses_sparse_strategy_for_exact_query():
    retriever = FakeRetriever()
    router = QueryRouter(retriever)

    router.search("Find YOLO11n-seg")

    call = retriever.calls[0]

    assert call["dense_weight"] == 0.0
    assert call["sparse_weight"] == 1.0


def test_router_uses_hybrid_strategy_for_comparison():
    retriever = FakeRetriever()
    router = QueryRouter(retriever)

    router.search("Compare BERT and RoBERTa")

    call = retriever.calls[0]

    assert call["dense_weight"] == 1.0
    assert call["sparse_weight"] == 1.0


def test_router_returns_analysis_and_results():
    retriever = FakeRetriever()
    router = QueryRouter(retriever)

    response = router.search("Explain neural networks")

    assert response.analysis.intent == QueryIntent.SEMANTIC
    assert response.results == []


def test_router_forwards_document_filter():
    retriever = FakeRetriever()
    router = QueryRouter(retriever)

    document_id = uuid4()

    router.search(
        "Explain neural networks",
        document_id=document_id,
    )

    assert retriever.calls[0]["document_id"] == document_id


def test_router_forwards_metadata_filters():
    retriever = FakeRetriever()
    router = QueryRouter(retriever)

    filters = {"category": "research"}

    router.search(
        "Explain neural networks",
        metadata_filters=filters,
    )

    assert retriever.calls[0]["metadata_filters"] == filters


def test_router_forwards_top_k():
    retriever = FakeRetriever()
    router = QueryRouter(retriever)

    router.search("Explain neural networks", top_k=10)

    assert retriever.calls[0]["top_k"] == 10


def test_bm25_definition_is_semantic() -> None:
    analysis = QueryClassifier().classify("What is BM25?")

    assert analysis.intent == QueryIntent.SEMANTIC
    assert analysis.rule == "definition_query"
