from uuid import uuid4

import pytest

from norag.chunking.models import Chunk
from norag.embeddings.models import EmbeddedChunk
from norag.hybrid_retrieval.fusion import ReciprocalRankFusion
from norag.hybrid_retrieval.service import HybridRetriever
from norag.sparse_retrieval.models import SparseSearchResult
from norag.vector_store.models import VectorSearchResult


def make_chunk(text="Test chunk"):
    return Chunk(
        document_id=uuid4(),
        index=0,
        text=text,
        token_count=len(text.split()),
    )


def dense_result(chunk, score=0.9):
    return VectorSearchResult(chunk=make_embedded_chunk(chunk), score=score)


def make_embedded_chunk(chunk):
    return EmbeddedChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        index=chunk.index,
        text=chunk.text,
        vector=[1.0, 0.0, 0.0],
        dimensions=3,
        model_name="test-model",
        metadata=chunk.metadata,
    )


def sparse_result(chunk, score=2.5):
    return SparseSearchResult(chunk=chunk, score=score)


def test_rrf_combines_shared_chunks():
    chunk = make_chunk()

    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse(
        dense_results=[dense_result(chunk)],
        sparse_results=[sparse_result(chunk)],
    )

    expected = (1 / 61) + (1 / 61)

    assert len(results) == 1
    assert results[0].chunk.id == chunk.id
    assert results[0].score == pytest.approx(expected)


def test_rrf_keeps_unique_chunks():
    dense_chunk = make_chunk("Dense result")
    sparse_chunk = make_chunk("Sparse result")

    fusion = ReciprocalRankFusion()

    results = fusion.fuse(
        dense_results=[dense_result(dense_chunk)],
        sparse_results=[sparse_result(sparse_chunk)],
    )

    assert len(results) == 2
    assert {result.chunk.id for result in results} == {
        dense_chunk.id,
        sparse_chunk.id,
    }


def test_rrf_uses_rank_not_raw_score():
    chunk_a = make_chunk("A")
    chunk_b = make_chunk("B")

    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse(
        dense_results=[
            dense_result(chunk_a, score=0.8),
            dense_result(chunk_b, score=0.99),
        ],
        sparse_results=[],
    )

    assert results[0].chunk.id == chunk_a.id


def test_rrf_respects_source_weights():
    chunk = make_chunk()

    fusion = ReciprocalRankFusion(
        k=60,
        dense_weight=2.0,
        sparse_weight=1.0,
    )

    results = fusion.fuse(
        dense_results=[dense_result(chunk)],
        sparse_results=[sparse_result(chunk)],
    )

    expected = (2 / 61) + (1 / 61)

    assert results[0].score == pytest.approx(expected)


def test_rrf_ignores_zero_weight_source():
    chunk = make_chunk()

    fusion = ReciprocalRankFusion(
        dense_weight=0,
        sparse_weight=1,
    )

    results = fusion.fuse(
        dense_results=[dense_result(chunk)],
        sparse_results=[],
    )

    assert results == []


def test_rrf_preserves_source_ranks_and_scores():
    chunk = make_chunk()

    fusion = ReciprocalRankFusion()

    results = fusion.fuse(
        dense_results=[dense_result(chunk, 0.87)],
        sparse_results=[sparse_result(chunk, 3.25)],
    )

    result = results[0]

    assert result.dense_rank == 1
    assert result.sparse_rank == 1
    assert result.dense_score == pytest.approx(0.87)
    assert result.sparse_score == pytest.approx(3.25)


def test_rrf_handles_different_source_ranks():
    chunk = make_chunk()

    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse(
        dense_results=[
            dense_result(make_chunk("First")),
            dense_result(chunk),
        ],
        sparse_results=[
            sparse_result(make_chunk("First sparse")),
            sparse_result(make_chunk("Second sparse")),
            sparse_result(chunk),
        ],
    )

    result = next(item for item in results if item.chunk.id == chunk.id)

    expected = (1 / 62) + (1 / 63)

    assert result.dense_rank == 2
    assert result.sparse_rank == 3
    assert result.score == pytest.approx(expected)


def test_rrf_respects_top_k():
    chunks = [make_chunk(str(i)) for i in range(5)]

    fusion = ReciprocalRankFusion()

    results = fusion.fuse(
        dense_results=[dense_result(chunk) for chunk in chunks],
        sparse_results=[],
        top_k=2,
    )

    assert len(results) == 2


def test_rrf_deduplicates_within_ranking():
    chunk = make_chunk()
    another = make_chunk("Another")

    fusion = ReciprocalRankFusion()

    results = fusion.fuse(
        dense_results=[
            dense_result(chunk),
            dense_result(chunk, 0.8),
            dense_result(another),
        ],
        sparse_results=[],
    )

    assert len(results) == 2

    result = next(item for item in results if item.chunk.id == chunk.id)

    assert result.dense_rank == 1


def test_rrf_returns_empty_for_empty_rankings():
    fusion = ReciprocalRankFusion()

    assert fusion.fuse([], []) == []


def test_rrf_rejects_invalid_k():
    with pytest.raises(ValueError, match="k"):
        ReciprocalRankFusion(k=0)


def test_rrf_rejects_negative_weights():
    with pytest.raises(ValueError, match="negative"):
        ReciprocalRankFusion(dense_weight=-1)


def test_rrf_rejects_non_finite_weights():
    with pytest.raises(ValueError, match="finite"):
        ReciprocalRankFusion(dense_weight=float("nan"))


def test_rrf_rejects_both_weights_zero():
    with pytest.raises(ValueError, match="At least one"):
        ReciprocalRankFusion(
            dense_weight=0,
            sparse_weight=0,
        )


def test_rrf_rejects_invalid_top_k():
    fusion = ReciprocalRankFusion()

    with pytest.raises(ValueError, match="top_k"):
        fusion.fuse([], [], top_k=0)


class FakeEmbeddingService:
    def __init__(self):
        self.calls = []

    def embed_query(self, query):
        self.calls.append(query)
        return [1.0, 0.0, 0.0]


class FakeDenseIndex:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class FakeSparseIndex:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


def make_retriever(
    dense_results=None,
    sparse_results=None,
    dense_weight=1.0,
    sparse_weight=1.0,
    candidate_multiplier=3,
):
    embedding = FakeEmbeddingService()
    dense = FakeDenseIndex(dense_results)
    sparse = FakeSparseIndex(sparse_results)

    retriever = HybridRetriever(
        embedding_service=embedding,
        dense_index=dense,
        sparse_index=sparse,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        candidate_multiplier=candidate_multiplier,
    )

    return retriever, embedding, dense, sparse


def test_hybrid_retriever_combines_both_engines():
    chunk = make_chunk()

    retriever, _, _, _ = make_retriever(
        dense_results=[dense_result(chunk)],
        sparse_results=[sparse_result(chunk)],
    )

    results = retriever.search("test query")

    assert len(results) == 1
    assert results[0].chunk.id == chunk.id


def test_hybrid_retriever_uses_candidate_multiplier():
    retriever, _, dense, sparse = make_retriever(
        candidate_multiplier=4,
    )

    retriever.search("test query", top_k=5)

    assert dense.calls[0]["top_k"] == 20
    assert sparse.calls[0]["top_k"] == 20


def test_hybrid_retriever_accepts_explicit_candidate_k():
    retriever, _, dense, sparse = make_retriever()

    retriever.search(
        "test query",
        top_k=3,
        candidate_k=10,
    )

    assert dense.calls[0]["top_k"] == 10
    assert sparse.calls[0]["top_k"] == 10


def test_hybrid_retriever_rejects_small_candidate_k():
    retriever, _, _, _ = make_retriever()

    with pytest.raises(ValueError, match="candidate_k"):
        retriever.search(
            "test query",
            top_k=5,
            candidate_k=3,
        )


def test_hybrid_retriever_rejects_empty_query():
    retriever, _, _, _ = make_retriever()

    with pytest.raises(ValueError, match="empty"):
        retriever.search("   ")


def test_hybrid_retriever_rejects_invalid_top_k():
    retriever, _, _, _ = make_retriever()

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("test query", top_k=0)


def test_hybrid_retriever_skips_disabled_dense_search():
    retriever, embedding, dense, sparse = make_retriever(
        dense_weight=0,
    )

    retriever.search("test query")

    assert embedding.calls == []
    assert dense.calls == []
    assert len(sparse.calls) == 1


def test_hybrid_retriever_skips_disabled_sparse_search():
    retriever, embedding, dense, sparse = make_retriever(
        sparse_weight=0,
    )

    retriever.search("test query")

    assert len(embedding.calls) == 1
    assert len(dense.calls) == 1
    assert sparse.calls == []


def test_hybrid_retriever_forwards_filters():
    target_document = uuid4()
    filters = {"category": "research"}

    retriever, _, dense, sparse = make_retriever()

    retriever.search(
        "test query",
        document_id=target_document,
        metadata_filters=filters,
    )

    for call in [dense.calls[0], sparse.calls[0]]:
        assert call["document_id"] == target_document
        assert call["metadata_filters"] == filters
