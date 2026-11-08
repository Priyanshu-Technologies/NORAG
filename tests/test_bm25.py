from uuid import uuid4

import pytest

from norag.chunking.models import Chunk
from norag.sparse_retrieval.bm25 import BM25Index
from norag.sparse_retrieval.tokenizer import tokenize


def make_chunk(
    text: str,
    document_id=None,
    metadata=None,
) -> Chunk:
    return Chunk(
        document_id=document_id or uuid4(),
        index=0,
        text=text,
        token_count=len(text.split()),
        metadata=metadata or {},
    )


def test_tokenize_lowercases_text():
    assert tokenize("NORAG BM25 Retrieval") == [
        "norag",
        "bm25",
        "retrieval",
    ]


def test_tokenize_normalizes_unicode():
    assert tokenize("ＣＯＭＰＵＴＥＲ") == ["computer"]


def test_tokenize_ignores_punctuation():
    assert tokenize("RAG, retrieval! augmented-generation.") == [
        "rag",
        "retrieval",
        "augmented",
        "generation",
    ]


def test_index_adds_chunk():
    index = BM25Index()

    index.add(make_chunk("Python programming"))

    assert index.count == 1


def test_index_adds_multiple_chunks():
    index = BM25Index()

    index.add_many(
        [
            make_chunk("Python programming"),
            make_chunk("Machine learning"),
        ]
    )

    assert index.count == 2


def test_bm25_ranks_matching_chunk_first():
    index = BM25Index()

    python_chunk = make_chunk("Python is a programming language")
    vision_chunk = make_chunk("Computer vision uses neural networks")
    retrieval_chunk = make_chunk("Retrieval augmented generation uses Python")

    index.add_many(
        [
            python_chunk,
            vision_chunk,
            retrieval_chunk,
        ]
    )

    results = index.search("Python programming", top_k=3)

    assert results[0].chunk.id == python_chunk.id


def test_bm25_rewards_rare_terms():
    index = BM25Index()

    rare_chunk = make_chunk("Python uses the transformer architecture")
    common_chunk = make_chunk("Python is a programming language")
    another_chunk = make_chunk("Python supports many libraries")

    index.add_many(
        [
            rare_chunk,
            common_chunk,
            another_chunk,
        ]
    )

    results = index.search("transformer", top_k=3)

    assert len(results) == 1
    assert results[0].chunk.id == rare_chunk.id


def test_bm25_is_case_insensitive():
    index = BM25Index()

    chunk = make_chunk("RETRIEVAL AUGMENTED GENERATION")
    index.add(chunk)

    results = index.search("retrieval augmented generation")

    assert len(results) == 1
    assert results[0].chunk.id == chunk.id


def test_bm25_returns_empty_for_unmatched_query():
    index = BM25Index()
    index.add(make_chunk("Python programming"))

    assert index.search("quantum entanglement") == []


def test_bm25_respects_top_k():
    index = BM25Index()

    index.add_many(
        [
            make_chunk("Python programming language"),
            make_chunk("Python programming tutorials"),
            make_chunk("Python programming examples"),
        ]
    )

    assert len(index.search("Python programming", top_k=2)) == 2


def test_bm25_filters_by_document_id():
    index = BM25Index()
    target_document = uuid4()

    first = make_chunk(
        "Python programming",
        document_id=target_document,
    )
    second = make_chunk("Python programming")

    index.add_many([first, second])

    results = index.search(
        "Python",
        document_id=target_document,
    )

    assert len(results) == 1
    assert results[0].chunk.id == first.id


def test_bm25_filters_by_metadata():
    index = BM25Index()

    first = make_chunk(
        "Neural networks and deep learning",
        metadata={"category": "research"},
    )
    second = make_chunk(
        "Neural networks and deep learning",
        metadata={"category": "documentation"},
    )

    index.add_many([first, second])

    results = index.search(
        "neural networks",
        metadata_filters={"category": "research"},
    )

    assert len(results) == 1
    assert results[0].chunk.id == first.id


def test_bm25_rejects_duplicate_chunks():
    index = BM25Index()
    chunk = make_chunk("Python programming")

    index.add(chunk)

    with pytest.raises(ValueError, match="already exists"):
        index.add(chunk)


def test_bm25_rejects_duplicate_ids_in_batch():
    index = BM25Index()
    chunk = make_chunk("Python programming")

    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        index.add_many([chunk, chunk])

    assert index.count == 0


def test_bm25_remove_updates_index():
    index = BM25Index()
    chunk = make_chunk("Python programming")

    index.add(chunk)

    assert index.remove(chunk.id) is True
    assert index.count == 0
    assert index.search("Python") == []


def test_bm25_remove_unknown_chunk():
    index = BM25Index()

    assert index.remove(uuid4()) is False


def test_bm25_clear():
    index = BM25Index()
    index.add(make_chunk("Python programming"))

    index.clear()

    assert index.count == 0
    assert index.search("Python") == []


def test_bm25_rejects_empty_query():
    index = BM25Index()

    with pytest.raises(ValueError, match="no searchable terms"):
        index.search("   !!! ")


def test_bm25_rejects_invalid_top_k():
    index = BM25Index()

    with pytest.raises(ValueError, match="top_k"):
        index.search("Python", top_k=0)


def test_bm25_rejects_invalid_k1():
    with pytest.raises(ValueError, match="k1"):
        BM25Index(k1=0)


def test_bm25_rejects_invalid_b():
    with pytest.raises(ValueError, match="b"):
        BM25Index(b=1.5)


def test_bm25_add_many_is_atomic():
    index = BM25Index()

    valid = make_chunk("Python programming")
    duplicate = make_chunk("Python programming")
    duplicate.id = valid.id

    with pytest.raises(ValueError):
        index.add_many([valid, duplicate])

    assert index.count == 0
