from typing import Any
from uuid import UUID

from norag.embeddings.service import EmbeddingService
from norag.hybrid_retrieval.fusion import ReciprocalRankFusion
from norag.hybrid_retrieval.models import HybridSearchResult
from norag.sparse_retrieval.bm25 import BM25Index
from norag.vector_store.memory import InMemoryVectorIndex


class HybridRetriever:
    """Coordinate dense and sparse retrieval."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        dense_index: InMemoryVectorIndex,
        sparse_index: BM25Index,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        candidate_multiplier: int = 3,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")

        self.embedding_service = embedding_service
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.candidate_multiplier = candidate_multiplier

        self.fusion = ReciprocalRankFusion(
            k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        document_id: UUID | None = None,
        metadata_filters: dict[str, Any] | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> list[HybridSearchResult]:
        """Execute retrieval using configurable per-query weights."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        effective_dense_weight = (
            self.fusion.dense_weight if dense_weight is None else dense_weight
        )

        effective_sparse_weight = (
            self.fusion.sparse_weight if sparse_weight is None else sparse_weight
        )

        # Validates weights and ensures at least one source is enabled.
        query_fusion = ReciprocalRankFusion(
            k=self.fusion.k,
            dense_weight=effective_dense_weight,
            sparse_weight=effective_sparse_weight,
        )

        if candidate_k is None:
            candidate_k = max(
                top_k,
                top_k * self.candidate_multiplier,
            )

        if candidate_k < top_k:
            raise ValueError("candidate_k cannot be smaller than top_k")

        dense_results = []
        sparse_results = []

        if query_fusion.dense_weight > 0:
            query_vector = self.embedding_service.embed_query(query)

            dense_results = self.dense_index.search(
                query_vector=query_vector,
                top_k=candidate_k,
                document_id=document_id,
                metadata_filters=metadata_filters,
            )

        if query_fusion.sparse_weight > 0:
            sparse_results = self.sparse_index.search(
                query=query,
                top_k=candidate_k,
                document_id=document_id,
                metadata_filters=metadata_filters,
            )

        return query_fusion.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k,
        )
