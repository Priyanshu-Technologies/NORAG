from collections.abc import Callable, Sequence
from statistics import fmean
from uuid import UUID

from norag.evaluation.metrics import calculate_metrics
from norag.evaluation.models import (
    EvaluationCase,
    EvaluationReport,
    QueryEvaluation,
    RankingMetrics,
)

SearchFunction = Callable[[str, int], Sequence[UUID]]


class RetrievalEvaluator:
    """Evaluate retrieval systems against labelled query datasets."""

    def evaluate(
        self,
        cases: list[EvaluationCase],
        search_fn: SearchFunction,
        k: int = 5,
    ) -> EvaluationReport:
        """Evaluate a retrieval function over multiple queries."""

        if k < 1:
            raise ValueError("k must be positive")

        if not cases:
            raise ValueError("Evaluation cases cannot be empty")

        query_results: list[QueryEvaluation] = []

        for case in cases:
            ranked_ids = search_fn(case.query, k)

            metrics = calculate_metrics(
                ranked_ids=ranked_ids,
                relevant_ids=case.relevant_chunk_ids,
                k=k,
            )

            query_results.append(
                QueryEvaluation(
                    query=case.query,
                    relevant_count=len(case.relevant_chunk_ids),
                    retrieved_count=len(set(ranked_ids)),
                    metrics=metrics,
                )
            )

        average_metrics = RankingMetrics(
            recall_at_k=fmean(result.metrics.recall_at_k for result in query_results),
            precision_at_k=fmean(
                result.metrics.precision_at_k for result in query_results
            ),
            mrr_at_k=fmean(result.metrics.mrr_at_k for result in query_results),
            ndcg_at_k=fmean(result.metrics.ndcg_at_k for result in query_results),
        )

        return EvaluationReport(
            k=k,
            query_count=len(query_results),
            average_metrics=average_metrics,
            per_query=query_results,
        )
