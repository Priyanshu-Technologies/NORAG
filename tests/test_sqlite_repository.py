from uuid import uuid4

import pytest

from norag.chunking.models import Chunk
from norag.embeddings.models import EmbeddedChunk
from norag.ingestion.models import Document, DocumentMetadata
from norag.storage.sqlite import SQLiteDocumentRepository


@pytest.fixture
def repository(tmp_path):
    instance = SQLiteDocumentRepository(tmp_path / "norag.sqlite3")
    instance.initialize()
    return instance


@pytest.fixture
def records():
    document_id = uuid4()
    chunk_id = uuid4()

    document = Document(
        id=document_id,
        content_hash="a" * 64,
        metadata=DocumentMetadata(
            file_name="sample.txt",
            title="Persistence Test",
        ),
    )

    chunk = Chunk(
        id=chunk_id,
        document_id=document_id,
        index=0,
        text="NORAG persistence test.",
        token_count=5,
        metadata={"section": "testing"},
    )

    embedded = EmbeddedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        index=0,
        text=chunk.text,
        vector=[0.1, 0.2, 0.3],
        dimensions=3,
        model_name="test-model",
        metadata=chunk.metadata,
    )

    return document, [chunk], [embedded]


def test_save_and_restore_records(repository, records):
    document, chunks, embeddings = records

    repository.save(document, chunks, embeddings)

    documents, restored_chunks, restored_embeddings = repository.load()

    assert documents == [document]
    assert restored_chunks == chunks
    assert restored_embeddings == embeddings


def test_data_survives_repository_recreation(
    repository,
    records,
):
    document, chunks, embeddings = records

    repository.save(document, chunks, embeddings)

    reopened = SQLiteDocumentRepository(repository.database_path)
    reopened.initialize()

    documents, restored_chunks, restored_embeddings = reopened.load()

    assert documents[0].id == document.id
    assert restored_chunks[0].id == chunks[0].id
    assert restored_embeddings[0].vector == embeddings[0].vector


def test_delete_removes_document_and_chunks(repository, records):
    document, chunks, embeddings = records

    repository.save(document, chunks, embeddings)

    assert repository.delete(document.id) is True

    documents, restored_chunks, restored_embeddings = repository.load()

    assert documents == []
    assert restored_chunks == []
    assert restored_embeddings == []


def test_delete_returns_false_for_unknown_document(repository):
    assert repository.delete(uuid4()) is False


def test_save_rejects_mismatched_chunk_and_embedding_ids(
    repository,
    records,
):
    document, chunks, embeddings = records

    embeddings[0].chunk_id = uuid4()

    with pytest.raises(ValueError, match="matching IDs"):
        repository.save(document, chunks, embeddings)

    documents, restored_chunks, restored_embeddings = repository.load()

    assert documents == []
    assert restored_chunks == []
    assert restored_embeddings == []


def test_save_rejects_chunks_from_another_document(
    repository,
    records,
):
    document, chunks, embeddings = records

    chunks[0].document_id = uuid4()

    with pytest.raises(ValueError, match="another document"):
        repository.save(document, chunks, embeddings)


def test_initialize_is_idempotent(repository):
    repository.initialize()
    repository.initialize()

    documents, chunks, embeddings = repository.load()

    assert documents == []
    assert chunks == []
    assert embeddings == []
