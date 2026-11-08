from uuid import uuid4

import pytest

from norag.chunking.models import Chunk
from norag.hybrid_retrieval.models import HybridSearchResult
from norag.reranking.service import RerankingService


class FakeReranker:
    model_name = "fake-reranker"

    def __init__(self, scores=None):
        self.scores = scores or []
        self.calls = []

    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        return self.scores


def make_candidate(text, score=0.01):
    chunk = Chunk(
        document_id=uuid4(),
        index=0,
        text=text,
        token_count=len(text.split()),
    )

    return HybridSearchResult(
        chunk=chunk,
        score=score,
    )


def test_reranking_reorders_candidates():
    candidates = [
        make_candidate("Computer vision uses neural networks."),
        make_candidate("RAG combines retrieval and generation."),
        make_candidate("Python is a programming language."),
    ]

    provider = FakeReranker([0.2, 0.9, 0.1])
    service = RerankingService(provider)

    results = service.rerank("Explain RAG", candidates)

    assert results[0].result.chunk.text.startswith("RAG")
    assert results[1].result.chunk.text.startswith("Computer")
    assert results[2].result.chunk.text.startswith("Python")


def test_reranking_preserves_original_rank():
    candidates = [
        make_candidate("First"),
        make_candidate("Second"),
        make_candidate("Third"),
    ]

    service = RerankingService(FakeReranker([0.1, 0.9, 0.5]))

    results = service.rerank("query", candidates)

    assert [item.original_rank for item in results] == [2, 3, 1]


def test_reranking_preserves_hybrid_scores():
    candidates = [
        make_candidate("First", score=0.045),
        make_candidate("Second", score=0.035),
    ]

    service = RerankingService(FakeReranker([0.1, 0.9]))

    results = service.rerank("query", candidates)

    assert results[0].result.score == pytest.approx(0.035)
    assert results[0].rerank_score == pytest.approx(0.9)


def test_reranking_accepts_negative_scores():
    candidates = [
        make_candidate("First"),
        make_candidate("Second"),
    ]

    service = RerankingService(FakeReranker([-4.0, -1.0]))

    results = service.rerank("query", candidates)

    assert results[0].rerank_score == -1.0


def test_reranking_respects_top_k():
    candidates = [
        make_candidate("First"),
        make_candidate("Second"),
        make_candidate("Third"),
    ]

    service = RerankingService(FakeReranker([0.3, 0.9, 0.5]))

    results = service.rerank(
        "query",
        candidates,
        top_k=2,
    )

    assert len(results) == 2


def test_reranking_returns_all_results_without_top_k():
    candidates = [
        make_candidate("First"),
        make_candidate("Second"),
        make_candidate("Third"),
    ]

    service = RerankingService(FakeReranker([0.3, 0.9, 0.5]))

    results = service.rerank("query", candidates)

    assert len(results) == 3


def test_reranking_preserves_order_for_tied_scores():
    candidates = [
        make_candidate("First"),
        make_candidate("Second"),
    ]

    service = RerankingService(FakeReranker([0.5, 0.5]))

    results = service.rerank("query", candidates)

    assert results[0].original_rank == 1
    assert results[1].original_rank == 2


def test_reranking_returns_empty_for_empty_candidates():
    service = RerankingService(FakeReranker())

    assert service.rerank("query", []) == []


def test_reranking_rejects_empty_query():
    service = RerankingService(FakeReranker())

    with pytest.raises(ValueError, match="empty"):
        service.rerank("   ", [])


def test_reranking_rejects_invalid_top_k():
    candidates = [make_candidate("Test")]

    service = RerankingService(FakeReranker([0.5]))

    with pytest.raises(ValueError, match="top_k"):
        service.rerank("query", candidates, top_k=0)


def test_reranking_rejects_unexpected_score_count():
    candidates = [
        make_candidate("First"),
        make_candidate("Second"),
    ]

    service = RerankingService(FakeReranker([0.5]))

    with pytest.raises(RuntimeError, match="number of scores"):
        service.rerank("query", candidates)


def test_reranking_rejects_non_finite_scores():
    candidates = [make_candidate("First")]

    service = RerankingService(FakeReranker([float("nan")]))

    with pytest.raises(ValueError, match="non-finite"):
        service.rerank("query", candidates)


def test_reranking_sends_query_and_documents_to_provider():
    candidates = [
        make_candidate("First document"),
        make_candidate("Second document"),
    ]

    provider = FakeReranker([0.5, 0.8])
    service = RerankingService(provider)

    service.rerank("What is NORAG?", candidates)

    assert provider.calls == [
        (
            "What is NORAG?",
            ["First document", "Second document"],
        )
    ]


def test_reranking_preserves_candidate_identity():
    candidates = [make_candidate("NORAG document")]

    service = RerankingService(FakeReranker([0.8]))

    results = service.rerank("query", candidates)

    assert results[0].result.chunk.id == candidates[0].chunk.id
