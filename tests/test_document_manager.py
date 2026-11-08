from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from norag.api.document_manager import DocumentManager


@pytest.fixture
def manager():
    ingestion = MagicMock()
    chunker = MagicMock()
    embedding = MagicMock()
    dense_index = MagicMock()
    sparse_index = MagicMock()

    document = SimpleNamespace(id=uuid4())
    chunks = [
        SimpleNamespace(id=uuid4()),
        SimpleNamespace(id=uuid4()),
    ]

    ingestion.ingest.return_value = document
    chunker.chunk.return_value = chunks
    embedding.embed_chunks.return_value = [
        SimpleNamespace(id=chunk.id) for chunk in chunks
    ]

    instance = DocumentManager(
        ingestion_service=ingestion,
        chunker=chunker,
        embedding_service=embedding,
        dense_index=dense_index,
        sparse_index=sparse_index,
    )

    return SimpleNamespace(
        manager=instance,
        ingestion=ingestion,
        chunker=chunker,
        embedding=embedding,
        dense_index=dense_index,
        sparse_index=sparse_index,
        document=document,
        chunks=chunks,
    )


def test_upload_successfully_indexes_document(manager):
    result = manager.manager.upload(Path("sample.pdf"))

    assert result.document is manager.document
    assert result.chunk_count == 2

    manager.ingestion.ingest.assert_called_once_with(Path("sample.pdf"))
    manager.chunker.chunk.assert_called_once_with(manager.document)
    manager.embedding.embed_chunks.assert_called_once_with(manager.chunks)

    manager.dense_index.add_many.assert_called_once()
    manager.sparse_index.add_many.assert_called_once()


def test_upload_rejects_documents_without_chunks(manager):
    manager.chunker.chunk.return_value = []

    with pytest.raises(
        ValueError,
        match="Document contains no indexable content",
    ):
        manager.manager.upload(Path("empty.pdf"))

    manager.embedding.embed_chunks.assert_not_called()
    manager.dense_index.add_many.assert_not_called()
    manager.sparse_index.add_many.assert_not_called()


def test_upload_rolls_back_dense_index_if_sparse_index_fails(manager):
    manager.sparse_index.add_many.side_effect = RuntimeError("Sparse indexing failed")

    with pytest.raises(RuntimeError, match="Sparse indexing failed"):
        manager.manager.upload(Path("sample.pdf"))

    expected_ids = [chunk.id for chunk in manager.chunks]

    assert manager.dense_index.remove.call_count == len(expected_ids)

    removed_ids = [call.args[0] for call in manager.dense_index.remove.call_args_list]
    assert set(removed_ids) == set(expected_ids)

    assert manager.manager.list_documents() == []


def test_list_documents_returns_uploaded_documents(manager):
    manager.manager.upload(Path("sample.pdf"))

    documents = manager.manager.list_documents()

    assert len(documents) == 1
    assert documents[0] is manager.document


def test_get_document_returns_document_by_id(manager):
    manager.manager.upload(Path("sample.pdf"))

    result = manager.manager.get_document(manager.document.id)

    assert result is manager.document


def test_get_document_returns_none_for_unknown_id(manager):
    result = manager.manager.get_document(uuid4())

    assert result is None


def test_delete_removes_document_and_its_index_entries(manager):
    manager.manager.upload(Path("sample.pdf"))

    result = manager.manager.delete(manager.document.id)

    assert result is True
    assert manager.manager.list_documents() == []

    expected_ids = [chunk.id for chunk in manager.chunks]

    assert manager.dense_index.remove.call_count == len(expected_ids)
    assert manager.sparse_index.remove.call_count == len(expected_ids)


def test_delete_returns_false_for_unknown_document(manager):
    result = manager.manager.delete(uuid4())

    assert result is False

    manager.dense_index.remove.assert_not_called()
    manager.sparse_index.remove.assert_not_called()


def test_upload_rejects_duplicate_document(manager):
    manager.document.content_hash = "a" * 64

    manager.manager.upload(Path("first.pdf"))

    duplicate = SimpleNamespace(
        id=uuid4(),
        content_hash="a" * 64,
    )
    manager.ingestion.ingest.return_value = duplicate

    with pytest.raises(
        ValueError,
        match="identical content",
    ):
        manager.manager.upload(Path("duplicate.pdf"))

    assert len(manager.manager.list_documents()) == 1

    # The duplicate must not be processed or indexed.
    assert manager.chunker.chunk.call_count == 1
    assert manager.embedding.embed_chunks.call_count == 1
    assert manager.dense_index.add_many.call_count == 1
    assert manager.sparse_index.add_many.call_count == 1


def test_upload_accepts_documents_with_different_content(manager):
    manager.document.content_hash = "a" * 64

    manager.manager.upload(Path("first.pdf"))

    second_document = SimpleNamespace(
        id=uuid4(),
        content_hash="b" * 64,
    )
    manager.ingestion.ingest.return_value = second_document

    result = manager.manager.upload(Path("second.pdf"))

    assert result.document is second_document
    assert len(manager.manager.list_documents()) == 2
