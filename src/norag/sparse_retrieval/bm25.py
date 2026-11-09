import math
from collections import Counter
from threading import RLock
from typing import Any
from uuid import UUID

from norag.chunking.models import Chunk
from norag.sparse_retrieval.interfaces import SparseIndex
from norag.sparse_retrieval.models import SparseSearchResult
from norag.sparse_retrieval.tokenizer import tokenize


class BM25Index(SparseIndex):
    """In-memory BM25 index implemented from scratch."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b

        self._chunks: dict[UUID, Chunk] = {}
        self._term_frequencies: dict[UUID, Counter[str]] = {}
        self._document_lengths: dict[UUID, int] = {}
        self._document_frequency: Counter[str] = Counter()
        self._lock = RLock()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def add(self, chunk: Chunk) -> None:
        self.add_many([chunk])

    def add_many(self, chunks: list[Chunk]) -> None:
        """Add chunks atomically and update corpus statistics."""

        if not chunks:
            return

        with self._lock:
            batch_ids = [chunk.id for chunk in chunks]

            if len(batch_ids) != len(set(batch_ids)):
                raise ValueError("Duplicate chunk IDs in batch")

            if any(chunk_id in self._chunks for chunk_id in batch_ids):
                raise ValueError("Chunk already exists in index")

            prepared = []

            for chunk in chunks:
                terms = tokenize(chunk.text)
                frequencies = Counter(terms)
                prepared.append((chunk, frequencies, len(terms)))

            # All validation is complete. Now update the index.
            for chunk, frequencies, length in prepared:
                chunk_id = chunk.id

                self._chunks[chunk_id] = chunk
                self._term_frequencies[chunk_id] = frequencies
                self._document_lengths[chunk_id] = length

                for term in frequencies:
                    self._document_frequency[term] += 1

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: UUID | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[SparseSearchResult]:
        """Retrieve chunks ranked by their BM25 scores."""

        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_terms = set(tokenize(query))

        if not query_terms:
            raise ValueError("Query contains no searchable terms")

        with self._lock:
            total_documents = len(self._chunks)

            if total_documents == 0:
                return []

            average_length = sum(self._document_lengths.values()) / total_documents
            average_length = max(average_length, 1.0)

            snapshot = [
                (
                    chunk,
                    self._term_frequencies[chunk_id],
                    self._document_lengths[chunk_id],
                )
                for chunk_id, chunk in self._chunks.items()
            ]
            document_frequency = dict(self._document_frequency)

        filters = metadata_filters or {}
        results: list[SparseSearchResult] = []

        for chunk, frequencies, document_length in snapshot:
            if document_id is not None and chunk.document_id != document_id:
                continue

            if not all(
                chunk.metadata.get(key) == value for key, value in filters.items()
            ):
                continue

            score = self._calculate_score(
                query_terms=query_terms,
                frequencies=frequencies,
                document_length=document_length,
                average_length=average_length,
                total_documents=total_documents,
                document_frequency=document_frequency,
            )

            # Avoid returning unrelated chunks with zero scores.
            if score > 0:
                results.append(
                    SparseSearchResult(
                        chunk=chunk,
                        score=score,
                    )
                )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]

    def _calculate_score(
        self,
        query_terms: set[str],
        frequencies: Counter[str],
        document_length: int,
        average_length: float,
        total_documents: int,
        document_frequency: dict[str, int],
    ) -> float:
        """Calculate the BM25 score for one document."""

        score = 0.0

        for term in query_terms:
            term_frequency = frequencies.get(term, 0)

            if term_frequency == 0:
                continue

            frequency = document_frequency.get(term, 0)

            inverse_document_frequency = math.log1p(
                (total_documents - frequency + 0.5) / (frequency + 0.5)
            )

            length_normalization = (
                1 - self.b + self.b * document_length / average_length
            )

            numerator = term_frequency * (self.k1 + 1)

            denominator = term_frequency + self.k1 * length_normalization

            score += inverse_document_frequency * numerator / denominator

        return score

    def remove(self, chunk_id: UUID) -> bool:
        """Remove a chunk and update corpus statistics."""

        with self._lock:
            if chunk_id not in self._chunks:
                return False

            frequencies = self._term_frequencies.pop(chunk_id)

            for term in frequencies:
                self._document_frequency[term] -= 1

                if self._document_frequency[term] == 0:
                    del self._document_frequency[term]

            del self._document_lengths[chunk_id]
            del self._chunks[chunk_id]

            return True

    def clear(self) -> None:
        """Remove all chunks and corpus statistics."""

        with self._lock:
            self._chunks.clear()
            self._term_frequencies.clear()
            self._document_lengths.clear()
            self._document_frequency.clear()
