import math

from norag.hybrid_retrieval.models import HybridSearchResult
from norag.sparse_retrieval.models import SparseSearchResult
from norag.vector_store.models import VectorSearchResult


class ReciprocalRankFusion:
    """Combine ranked retrieval results using weighted RRF."""

    def __init__(
        self,
        k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")

        weights = [dense_weight, sparse_weight]

        if not all(math.isfinite(weight) for weight in weights):
            raise ValueError("Weights must be finite")

        if any(weight < 0 for weight in weights):
            raise ValueError("Weights cannot be negative")

        if not any(weight > 0 for weight in weights):
            raise ValueError("At least one weight must be positive")

        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def fuse(
        self,
        dense_results: list[VectorSearchResult],
        sparse_results: list[SparseSearchResult],
        top_k: int = 5,
    ) -> list[HybridSearchResult]:
        """Fuse dense and sparse rankings into one result list."""

        if top_k < 1:
            raise ValueError("top_k must be positive")

        fused: dict[str, dict] = {}

        rankings = [
            ("dense", dense_results, self.dense_weight),
            ("sparse", sparse_results, self.sparse_weight),
        ]

        for source, results, weight in rankings:
            if weight == 0:
                continue

            seen_ids: set[str] = set()

            for rank, result in enumerate(results, start=1):
                chunk = result.chunk
                chunk_id = str(chunk.id)

                # One chunk should contribute only once per ranking.
                if chunk_id in seen_ids:
                    continue

                seen_ids.add(chunk_id)

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        "chunk": chunk,
                        "score": 0.0,
                        "dense_rank": None,
                        "sparse_rank": None,
                        "dense_score": None,
                        "sparse_score": None,
                    }

                entry = fused[chunk_id]

                entry["score"] += weight / (self.k + rank)

                entry[f"{source}_rank"] = rank
                entry[f"{source}_score"] = result.score

        results = [HybridSearchResult(**entry) for entry in fused.values()]

        # Stable, deterministic ordering for tied scores.
        results.sort(
            key=lambda result: (
                -result.score,
                min(
                    rank
                    for rank in (
                        result.dense_rank,
                        result.sparse_rank,
                    )
                    if rank is not None
                ),
                str(result.chunk.id),
            )
        )

        return results[:top_k]
