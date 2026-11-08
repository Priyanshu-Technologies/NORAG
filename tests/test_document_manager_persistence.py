from unittest.mock import MagicMock

from norag.api.document_manager import DocumentManager
from norag.chunking.models import Chunk
from norag.embeddings.models import EmbeddedChunk
from norag.ingestion.models import Document, DocumentMetadata
from norag.sparse_retrieval.bm25 import BM25Index
from norag.storage.sqlite import SQLiteDocumentRepository
from norag.vector_store.memory import InMemoryVectorIndex


def create_manager(repository, document, chunk, embedded):
    ingestion = MagicMock()
    ingestion.ingest.return_value = document

    chunker = MagicMock()
    chunker.chunk.return_value = [chunk]

    embedding = MagicMock()
    embedding.embed_chunks.return_value = [embedded]

    dense_index = InMemoryVectorIndex()
    sparse_index = BM25Index()

    manager = DocumentManager(
        ingestion_service=ingestion,
        chunker=chunker,
        embedding_service=embedding,
        dense_index=dense_index,
        sparse_index=sparse_index,
        repository=repository,
    )

    return manager, dense_index, sparse_index


def test_document_and_indexes_survive_restart(tmp_path):
    repository = SQLiteDocumentRepository(tmp_path / "norag.sqlite3")

    document = Document(
        content_hash="a" * 64,
        metadata=DocumentMetadata(file_name="persistent.txt"),
    )

    chunk = Chunk(
        document_id=document.id,
        index=0,
        text="NORAG persistence survives server restarts.",
        token_count=7,
    )

    embedded = EmbeddedChunk(
        chunk_id=chunk.id,
        document_id=document.id,
        index=0,
        text=chunk.text,
        vector=[0.1, 0.2, 0.3],
        dimensions=3,
        model_name="test-model",
    )

    manager, _, _ = create_manager(repository, document, chunk, embedded)

    manager.upload(tmp_path / "persistent.txt")

    # Simulate a new application process with fresh indexes.
    restored_manager, dense, sparse = create_manager(
        repository, document, chunk, embedded
    )

    assert len(restored_manager.list_documents()) == 1
    assert restored_manager.get_document(document.id) == document

    assert dense.count == 1
    assert sparse.count == 1

    dense_results = dense.search([0.1, 0.2, 0.3])
    sparse_results = sparse.search("persistence")

    assert dense_results[0].chunk.chunk_id == chunk.id
    assert sparse_results[0].chunk.id == chunk.id


def test_deletion_survives_restart(tmp_path):
    repository = SQLiteDocumentRepository(tmp_path / "norag.sqlite3")

    document = Document(
        content_hash="b" * 64,
        metadata=DocumentMetadata(file_name="delete.txt"),
    )

    chunk = Chunk(
        document_id=document.id,
        index=0,
        text="This document will be deleted.",
        token_count=6,
    )

    embedded = EmbeddedChunk(
        chunk_id=chunk.id,
        document_id=document.id,
        index=0,
        text=chunk.text,
        vector=[0.2, 0.3, 0.4],
        dimensions=3,
        model_name="test-model",
    )

    manager, _, _ = create_manager(repository, document, chunk, embedded)

    manager.upload(tmp_path / "delete.txt")
    assert manager.delete(document.id) is True

    restored_manager, dense, sparse = create_manager(
        repository, document, chunk, embedded
    )

    assert restored_manager.list_documents() == []
    assert dense.count == 0
    assert sparse.count == 0
