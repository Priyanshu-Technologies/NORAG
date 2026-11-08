from uuid import uuid4

import pytest

from norag.chunking.models import Chunk
from norag.embeddings.models import EmbeddedChunk
from norag.embeddings.service import EmbeddingService


class FakeEmbeddingProvider:
    model_name = "fake-embedding-model"
    embedding_dimensions = 3

    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))

        return [[float(len(text)), 0.5, 0.25] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 0.5, 0.25]


def make_chunk(text, index=0):
    return Chunk(
        document_id=uuid4(),
        index=index,
        text=text,
        token_count=5,
    )


def test_embedding_service_embeds_chunks():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    chunks = [make_chunk("NORAG chunk")]

    results = service.embed_chunks(chunks)

    assert len(results) == 1
    assert results[0].vector == [11.0, 0.5, 0.25]
    assert results[0].model_name == "fake-embedding-model"


def test_embedding_service_preserves_chunk_identity():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    chunk = make_chunk("Identity test")
    result = service.embed_chunks([chunk])[0]

    assert result.chunk_id == chunk.id
    assert result.document_id == chunk.document_id
    assert result.index == chunk.index
    assert result.text == chunk.text


def test_embedding_service_processes_batches():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider, batch_size=2)

    chunks = [make_chunk(f"Chunk {index}", index) for index in range(5)]

    results = service.embed_chunks(chunks)

    assert len(results) == 5
    assert len(provider.calls) == 3
    assert [len(batch) for batch in provider.calls] == [2, 2, 1]


def test_embedding_service_preserves_order():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider, batch_size=2)

    chunks = [make_chunk(f"Chunk {index}", index) for index in range(5)]

    results = service.embed_chunks(chunks)

    assert [result.index for result in results] == [0, 1, 2, 3, 4]


def test_embedding_service_embeds_query():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    vector = service.embed_query("What is NORAG?")

    assert vector == [14.0, 0.5, 0.25]


def test_embedding_service_rejects_empty_query():
    service = EmbeddingService(FakeEmbeddingProvider())

    with pytest.raises(ValueError, match="empty"):
        service.embed_query("   ")


def test_embedding_service_rejects_dimension_mismatch():
    class IncorrectProvider(FakeEmbeddingProvider):
        def embed_documents(self, texts):
            return [[1.0, 2.0] for _ in texts]

    service = EmbeddingService(IncorrectProvider())

    with pytest.raises(ValueError, match="dimension mismatch"):
        service.embed_chunks([make_chunk("Test")])


def test_embedding_service_rejects_unexpected_vector_count():
    class IncorrectProvider(FakeEmbeddingProvider):
        def embed_documents(self, texts):
            return []

    service = EmbeddingService(IncorrectProvider())

    with pytest.raises(RuntimeError, match="number of vectors"):
        service.embed_chunks([make_chunk("Test")])


def test_embedding_service_rejects_non_finite_values():
    class IncorrectProvider(FakeEmbeddingProvider):
        def embed_documents(self, texts):
            return [[float("nan"), 0.5, 0.25] for _ in texts]

    service = EmbeddingService(IncorrectProvider())

    with pytest.raises(ValueError, match="invalid embedding"):
        service.embed_chunks([make_chunk("Test")])


def test_embedding_service_rejects_empty_chunk_text():
    service = EmbeddingService(FakeEmbeddingProvider())

    with pytest.raises(ValueError, match="empty text"):
        service.embed_chunks([make_chunk("   ")])


def test_embedding_service_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingService(FakeEmbeddingProvider(), batch_size=0)


def test_embedded_chunk_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        EmbeddedChunk(
            chunk_id=uuid4(),
            document_id=uuid4(),
            index=0,
            text="NORAG",
            vector=[0.1, 0.2],
            dimensions=3,
            model_name="test-model",
        )
