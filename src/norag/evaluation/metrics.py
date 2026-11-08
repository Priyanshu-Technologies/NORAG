import math
from collections.abc import Collection, Sequence
from uuid import UUID

from norag.evaluation.models import RankingMetrics


def _validate_inputs(
    ranked_ids: Sequence[UUID],
    relevant_ids: Collection[UUID],
    k: int,
) -> None:
    if k < 1:
        raise ValueError("k must be positive")

    if not relevant_ids:
        raise ValueError("Relevant IDs cannot be empty")


def _unique_ranked_ids(
    ranked_ids: Sequence[UUID],
) -> list[UUID]:
    """Deduplicate results while preserving their ranking."""

    return list(dict.fromkeys(ranked_ids))


def recall_at_k(
    ranked_ids: Sequence[UUID],
    relevant_ids: Collection[UUID],
    k: int = 5,
) -> float:
    """Calculate the fraction of relevant chunks retrieved in top K."""

    _validate_inputs(ranked_ids, relevant_ids, k)

    retrieved = set(_unique_ranked_ids(ranked_ids)[:k])

    return len(retrieved & set(relevant_ids)) / len(relevant_ids)


def precision_at_k(
    ranked_ids: Sequence[UUID],
    relevant_ids: Collection[UUID],
    k: int = 5,
) -> float:
    """Calculate the fraction of top K results that are relevant."""

    _validate_inputs(ranked_ids, relevant_ids, k)

    retrieved = _unique_ranked_ids(ranked_ids)[:k]

    relevant_count = sum(chunk_id in relevant_ids for chunk_id in retrieved)

    return relevant_count / k


def mrr_at_k(
    ranked_ids: Sequence[UUID],
    relevant_ids: Collection[UUID],
    k: int = 5,
) -> float:
    """Calculate reciprocal rank of the first relevant result."""

    _validate_inputs(ranked_ids, relevant_ids, k)

    retrieved = _unique_ranked_ids(ranked_ids)[:k]

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[UUID],
    relevant_ids: Collection[UUID],
    k: int = 5,
) -> float:
    """Calculate normalized discounted cumulative gain at K."""

    _validate_inputs(ranked_ids, relevant_ids, k)

    retrieved = _unique_ranked_ids(ranked_ids)[:k]

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved, start=1)
        if chunk_id in relevant_ids
    )

    ideal_relevant_count = min(len(relevant_ids), k)

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1)
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def calculate_metrics(
    ranked_ids: Sequence[UUID],
    relevant_ids: Collection[UUID],
    k: int = 5,
) -> RankingMetrics:
    """Calculate all supported ranking metrics."""

    return RankingMetrics(
        recall_at_k=recall_at_k(ranked_ids, relevant_ids, k),
        precision_at_k=precision_at_k(ranked_ids, relevant_ids, k),
        mrr_at_k=mrr_at_k(ranked_ids, relevant_ids, k),
        ndcg_at_k=ndcg_at_k(ranked_ids, relevant_ids, k),
    )
