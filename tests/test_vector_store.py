from uuid import uuid4

import pytest

from norag.embeddings.models import EmbeddedChunk
from norag.vector_store.memory import InMemoryVectorIndex
from norag.vector_store.similarity import cosine_similarity


def make_embedded_chunk(
    vector: list[float],
    document_id=None,
    metadata=None,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=uuid4(),
        document_id=document_id or uuid4(),
        index=0,
        text="Test document chunk",
        vector=vector,
        dimensions=len(vector),
        model_name="test-model",
        metadata=metadata or {},
    )


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_similarity_perpendicular_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_rejects_different_dimensions():
    with pytest.raises(ValueError, match="equal dimensions"):
        cosine_similarity([1, 2], [1, 2, 3])


def test_cosine_similarity_rejects_zero_vector():
    with pytest.raises(ValueError, match="Zero vectors"):
        cosine_similarity([0, 0], [1, 1])


def test_index_adds_chunk():
    index = InMemoryVectorIndex(dimensions=3)
    chunk = make_embedded_chunk([1, 0, 0])

    index.add(chunk)

    assert index.count == 1


def test_index_search_returns_most_similar_first():
    index = InMemoryVectorIndex(dimensions=3)

    first = make_embedded_chunk([1, 0, 0])
    second = make_embedded_chunk([0, 1, 0])
    third = make_embedded_chunk([0.9, 0.1, 0])

    index.add_many([first, second, third])

    results = index.search([1, 0, 0], top_k=3)

    assert results[0].chunk.chunk_id == first.chunk_id
    assert results[1].chunk.chunk_id == third.chunk_id
    assert results[2].chunk.chunk_id == second.chunk_id


def test_index_respects_top_k():
    index = InMemoryVectorIndex(dimensions=3)

    chunks = [
        make_embedded_chunk([1, 0, 0]),
        make_embedded_chunk([0, 1, 0]),
        make_embedded_chunk([0, 0, 1]),
    ]

    index.add_many(chunks)

    results = index.search([1, 0, 0], top_k=2)

    assert len(results) == 2


def test_index_filters_by_document_id():
    index = InMemoryVectorIndex(dimensions=3)
    target_document = uuid4()

    first = make_embedded_chunk(
        [1, 0, 0],
        document_id=target_document,
    )
    second = make_embedded_chunk([0, 1, 0])

    index.add_many([first, second])

    results = index.search(
        [1, 0, 0],
        document_id=target_document,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == first.chunk_id


def test_index_filters_by_metadata():
    index = InMemoryVectorIndex(dimensions=3)

    first = make_embedded_chunk(
        [1, 0, 0],
        metadata={"category": "research"},
    )
    second = make_embedded_chunk(
        [0, 1, 0],
        metadata={"category": "documentation"},
    )

    index.add_many([first, second])

    results = index.search(
        [1, 0, 0],
        metadata_filters={"category": "research"},
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == first.chunk_id


def test_index_rejects_duplicate_chunks():
    index = InMemoryVectorIndex(dimensions=3)
    chunk = make_embedded_chunk([1, 0, 0])

    index.add(chunk)

    with pytest.raises(ValueError, match="already exists"):
        index.add(chunk)


def test_index_rejects_dimension_mismatch():
    index = InMemoryVectorIndex(dimensions=3)
    chunk = make_embedded_chunk([1, 0])

    with pytest.raises(ValueError, match="dimension mismatch"):
        index.add(chunk)


def test_index_add_many_is_atomic():
    index = InMemoryVectorIndex(dimensions=3)

    valid = make_embedded_chunk([1, 0, 0])
    invalid = make_embedded_chunk([1, 0])

    with pytest.raises(ValueError):
        index.add_many([valid, invalid])

    assert index.count == 0


def test_index_removes_chunk():
    index = InMemoryVectorIndex(dimensions=3)
    chunk = make_embedded_chunk([1, 0, 0])

    index.add(chunk)

    assert index.remove(chunk.chunk_id) is True
    assert index.count == 0


def test_index_clear():
    index = InMemoryVectorIndex(dimensions=3)

    index.add(make_embedded_chunk([1, 0, 0]))
    index.clear()

    assert index.count == 0


def test_index_rejects_invalid_top_k():
    index = InMemoryVectorIndex(dimensions=3)

    with pytest.raises(ValueError, match="top_k"):
        index.search([1, 0, 0], top_k=0)
