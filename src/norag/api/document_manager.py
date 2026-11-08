from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from uuid import UUID

from norag.chunking.interfaces import Chunker
from norag.embeddings.service import EmbeddingService
from norag.ingestion.models import Document
from norag.ingestion.service import IngestionService
from norag.sparse_retrieval.interfaces import SparseIndex
from norag.storage.filesystem import LocalFileStorage
from norag.storage.interfaces import DocumentRepository
from norag.vector_store.interfaces import VectorIndex


@dataclass(frozen=True)
class DocumentIndexResult:
    document: Document
    chunk_count: int


class DuplicateDocumentError(ValueError):
    """Raised when a document with identical content already exists."""


class DocumentManager:
    """Manage ingestion, indexing, persistence and original files."""

    def __init__(
        self,
        ingestion_service: IngestionService,
        chunker: Chunker,
        embedding_service: EmbeddingService,
        dense_index: VectorIndex,
        sparse_index: SparseIndex,
        repository: DocumentRepository | None = None,
        file_storage: LocalFileStorage | None = None,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.repository = repository
        self.file_storage = file_storage

        self._documents: dict[UUID, Document] = {}
        self._chunk_ids: dict[UUID, list[UUID]] = {}
        self._lock = RLock()

        if self.repository is not None:
            self._restore()

        if self.file_storage is not None:
            self.file_storage.initialize()

    def _restore(self) -> None:
        """Restore persisted documents and rebuild retrieval indexes."""

        self.repository.initialize()

        documents, chunks, embedded_chunks = self.repository.load()

        if not documents:
            if chunks or embedded_chunks:
                raise RuntimeError("Storage contains indexing data without documents.")
            return

        document_ids = {document.id for document in documents}

        if len(document_ids) != len(documents):
            raise RuntimeError("Storage contains duplicate document IDs.")

        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        embeddings_by_id = {embedded.chunk_id: embedded for embedded in embedded_chunks}

        if len(chunks_by_id) != len(chunks):
            raise RuntimeError("Storage contains duplicate chunk IDs.")

        if len(embeddings_by_id) != len(embedded_chunks):
            raise RuntimeError("Storage contains duplicate embedding IDs.")

        if set(chunks_by_id) != set(embeddings_by_id):
            raise RuntimeError("Stored chunks and embeddings do not match.")

        chunk_ids: dict[UUID, list[UUID]] = {document.id: [] for document in documents}

        for chunk in chunks:
            if chunk.document_id not in document_ids:
                raise RuntimeError("Storage contains chunks with missing documents.")

            embedded = embeddings_by_id[chunk.id]

            if embedded.document_id != chunk.document_id:
                raise RuntimeError("Chunk and embedding document IDs do not match.")

            chunk_ids[chunk.document_id].append(chunk.id)

        for embedded in embedded_chunks:
            if embedded.document_id not in document_ids:
                raise RuntimeError(
                    "Storage contains embeddings with missing documents."
                )

        if any(not ids for ids in chunk_ids.values()):
            raise RuntimeError("Storage contains documents without indexed chunks.")

        self.dense_index.add_many(embedded_chunks)

        try:
            self.sparse_index.add_many(chunks)
        except Exception:
            self.dense_index.clear()
            raise

        self._documents = {document.id: document for document in documents}
        self._chunk_ids = chunk_ids

    def upload(self, source: Path) -> DocumentIndexResult:
        """Ingest, index, persist and store the original document."""

        with self._lock:
            document = self.ingestion_service.ingest(source)

            content_hash = getattr(document, "content_hash", None)

            if content_hash and any(
                existing.content_hash == content_hash
                for existing in self._documents.values()
            ):
                raise DuplicateDocumentError(
                    "A document with identical content has already been uploaded."
                )

            if self.file_storage is not None:
                document.metadata.file_name = source.name

            chunks = self.chunker.chunk(document)

            if not chunks:
                raise ValueError("Document contains no indexable content.")

            embedded_chunks = self.embedding_service.embed_chunks(chunks)

            chunk_ids = [chunk.id for chunk in chunks]
            embedding_ids = [embedded.id for embedded in embedded_chunks]

            if (
                len(chunk_ids) != len(set(chunk_ids))
                or len(embedding_ids) != len(set(embedding_ids))
                or set(chunk_ids) != set(embedding_ids)
            ):
                raise ValueError("Chunks and embeddings must have matching unique IDs.")

            dense_added = False
            sparse_added = False
            file_stored = False

            try:
                self.dense_index.add_many(embedded_chunks)
                dense_added = True

                self.sparse_index.add_many(chunks)
                sparse_added = True

                if self.file_storage is not None:
                    self.file_storage.store(source, document)
                    file_stored = True

                if self.repository is not None:
                    self.repository.save(
                        document=document,
                        chunks=chunks,
                        embedded_chunks=embedded_chunks,
                    )

                self._documents[document.id] = document
                self._chunk_ids[document.id] = chunk_ids

            except Exception:
                if file_stored:
                    self.file_storage.delete(document)

                if sparse_added:
                    for chunk_id in chunk_ids:
                        self.sparse_index.remove(chunk_id)

                if dense_added:
                    for chunk_id in chunk_ids:
                        self.dense_index.remove(chunk_id)

                raise

            return DocumentIndexResult(
                document=document,
                chunk_count=len(chunks),
            )

    def list_documents(self) -> list[Document]:
        with self._lock:
            return list(self._documents.values())

    def get_document(self, document_id: UUID) -> Document | None:
        with self._lock:
            return self._documents.get(document_id)

    def delete(self, document_id: UUID) -> bool:
        """Delete document records, indexes and original file."""

        with self._lock:
            document = self._documents.get(document_id)

            if document is None:
                return False

            if self.repository is not None:
                deleted = self.repository.delete(document_id)

                if not deleted:
                    raise RuntimeError("Document exists in memory but not in storage.")

            chunk_ids = self._chunk_ids[document_id]

            for chunk_id in chunk_ids:
                self.dense_index.remove(chunk_id)
                self.sparse_index.remove(chunk_id)

            del self._documents[document_id]
            del self._chunk_ids[document_id]

            if self.file_storage is not None:
                self.file_storage.delete(document)

            return True
