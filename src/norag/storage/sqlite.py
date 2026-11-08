import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import UUID

from norag.chunking.models import Chunk
from norag.embeddings.models import EmbeddedChunk
from norag.ingestion.models import Document
from norag.storage.interfaces import DocumentRepository


class SQLiteDocumentRepository(DocumentRepository):
    """SQLite-backed persistent document repository."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
        )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        """Initialize the database schema."""

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        document_json TEXT NOT NULL
                    )
                    """
            )

            connection.execute(
                """
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        chunk_json TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,

                        FOREIGN KEY (document_id)
                            REFERENCES documents(id)
                            ON DELETE CASCADE
                    )
                    """
            )

            connection.execute(
                """
                    CREATE INDEX IF NOT EXISTS
                    idx_chunks_document_id
                    ON chunks(document_id)
                    """
            )

    def save(
        self,
        document: Document,
        chunks: list[Chunk],
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        """Persist a complete document indexing operation atomically."""

        if not chunks:
            raise ValueError("Cannot persist a document without chunks.")

        chunk_ids = [chunk.id for chunk in chunks]
        embedding_ids = [embedded.chunk_id for embedded in embedded_chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Duplicate chunk IDs.")

        if len(embedding_ids) != len(set(embedding_ids)):
            raise ValueError("Duplicate embedding IDs.")

        if set(chunk_ids) != set(embedding_ids):
            raise ValueError("Chunks and embeddings must have matching IDs.")

        for chunk in chunks:
            if chunk.document_id != document.id:
                raise ValueError("Chunk belongs to another document.")

        for embedded in embedded_chunks:
            if embedded.document_id != document.id:
                raise ValueError("Embedding belongs to another document.")

        embeddings_by_id = {embedded.chunk_id: embedded for embedded in embedded_chunks}

        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                    INSERT INTO documents (id, document_json)
                    VALUES (?, ?)
                    """,
                (
                    str(document.id),
                    document.model_dump_json(),
                ),
            )

            connection.executemany(
                """
                    INSERT INTO chunks (
                        id,
                        document_id,
                        chunk_json,
                        embedding_json
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                [
                    (
                        str(chunk.id),
                        str(document.id),
                        chunk.model_dump_json(),
                        embeddings_by_id[chunk.id].model_dump_json(),
                    )
                    for chunk in chunks
                ],
            )

    def load(
        self,
    ) -> tuple[list[Document], list[Chunk], list[EmbeddedChunk]]:
        """Restore all persisted objects."""

        with closing(self._connect()) as connection:
            document_rows = connection.execute(
                """
                SELECT document_json
                FROM documents
                ORDER BY rowid
                """
            ).fetchall()

            chunk_rows = connection.execute(
                """
                SELECT chunk_json, embedding_json
                FROM chunks
                ORDER BY rowid
                """
            ).fetchall()

        documents = [Document.model_validate_json(row[0]) for row in document_rows]

        chunks = [Chunk.model_validate_json(row[0]) for row in chunk_rows]

        embedded_chunks = [
            EmbeddedChunk.model_validate_json(row[1]) for row in chunk_rows
        ]

        return documents, chunks, embedded_chunks

    def delete(self, document_id: UUID) -> bool:
        """Delete a document and cascade-delete its chunks."""

        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE id = ?",
                (str(document_id),),
            )

            return cursor.rowcount > 0
