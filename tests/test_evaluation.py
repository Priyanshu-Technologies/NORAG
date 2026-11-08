from uuid import uuid4

import pytest
from pydantic import ValidationError

from norag.evaluation import (
    EvaluationCase,
    RetrievalEvaluator,
    calculate_metrics,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k():
    a, b, c, x = [uuid4() for _ in range(4)]

    result = recall_at_k(
        [a, x, b],
        {a, b, c},
        k=3,
    )

    assert result == pytest.approx(2 / 3)


def test_precision_at_k():
    a, b, x = [uuid4() for _ in range(3)]

    result = precision_at_k(
        [a, x, b],
        {a, b},
        k=3,
    )

    assert result == pytest.approx(2 / 3)


def test_mrr_at_k():
    a, x = [uuid4() for _ in range(2)]

    result = mrr_at_k(
        [x, a],
        {a},
        k=2,
    )

    assert result == pytest.approx(0.5)


def test_mrr_returns_zero_without_relevant_results():
    a, b = uuid4(), uuid4()

    assert mrr_at_k([a, b], {uuid4()}, k=2) == 0.0


def test_ndcg_returns_one_for_ideal_ranking():
    a, b = uuid4(), uuid4()

    result = ndcg_at_k(
        [a, b],
        {a, b},
        k=2,
    )

    assert result == pytest.approx(1.0)


def test_ndcg_rewards_relevant_results_at_higher_positions():
    a, b, x = [uuid4() for _ in range(3)]

    ideal = ndcg_at_k([a, b, x], {a, b}, k=3)
    poor = ndcg_at_k([x, a, b], {a, b}, k=3)

    assert ideal == pytest.approx(1.0)
    assert poor < ideal


def test_metrics_deduplicate_ranked_ids():
    a, x = uuid4(), uuid4()

    result = recall_at_k(
        [x, x, a],
        {a},
        k=2,
    )

    assert result == pytest.approx(1.0)


def test_empty_retrieval_returns_zero_metrics():
    relevant = {uuid4()}

    metrics = calculate_metrics(
        [],
        relevant,
        k=5,
    )

    assert metrics.recall_at_k == 0.0
    assert metrics.precision_at_k == 0.0
    assert metrics.mrr_at_k == 0.0
    assert metrics.ndcg_at_k == 0.0


def test_metrics_reject_invalid_k():
    with pytest.raises(ValueError, match="k"):
        recall_at_k([], {uuid4()}, k=0)


def test_metrics_reject_empty_relevant_ids():
    with pytest.raises(ValueError, match="Relevant IDs"):
        recall_at_k([], set(), k=5)


def test_calculate_metrics_returns_all_metrics():
    a, b, x = [uuid4() for _ in range(3)]

    metrics = calculate_metrics(
        [a, x, b],
        {a, b},
        k=3,
    )

    assert metrics.recall_at_k == pytest.approx(1.0)
    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.mrr_at_k == pytest.approx(1.0)
    assert 0.0 <= metrics.ndcg_at_k <= 1.0


def test_evaluator_calculates_macro_averages():
    relevant_a = uuid4()
    relevant_b = uuid4()
    unrelated = uuid4()

    cases = [
        EvaluationCase(
            query="Query one",
            relevant_chunk_ids={relevant_a},
        ),
        EvaluationCase(
            query="Query two",
            relevant_chunk_ids={relevant_b},
        ),
    ]

    def search(query, k):
        if query == "Query one":
            return [relevant_a]

        return [unrelated]

    report = RetrievalEvaluator().evaluate(
        cases,
        search,
        k=1,
    )

    assert report.query_count == 2
    assert report.average_metrics.recall_at_k == pytest.approx(0.5)
    assert report.average_metrics.precision_at_k == pytest.approx(0.5)
    assert report.average_metrics.mrr_at_k == pytest.approx(0.5)
    assert report.average_metrics.ndcg_at_k == pytest.approx(0.5)


def test_evaluator_preserves_per_query_results():
    relevant = uuid4()

    cases = [
        EvaluationCase(
            query="Find relevant document",
            relevant_chunk_ids={relevant},
        )
    ]

    report = RetrievalEvaluator().evaluate(
        cases,
        lambda query, k: [relevant],
        k=1,
    )

    assert len(report.per_query) == 1
    assert report.per_query[0].query == "Find relevant document"
    assert report.per_query[0].metrics.recall_at_k == 1.0


def test_evaluator_passes_k_to_search_function():
    relevant = uuid4()
    received = []

    def search(query, k):
        received.append((query, k))
        return [relevant]

    cases = [
        EvaluationCase(
            query="Test query",
            relevant_chunk_ids={relevant},
        )
    ]

    RetrievalEvaluator().evaluate(cases, search, k=10)

    assert received == [("Test query", 10)]


def test_evaluator_rejects_empty_cases():
    with pytest.raises(ValueError, match="cannot be empty"):
        RetrievalEvaluator().evaluate([], lambda query, k: [], k=5)


def test_evaluator_rejects_invalid_k():
    cases = [
        EvaluationCase(
            query="Test",
            relevant_chunk_ids={uuid4()},
        )
    ]

    with pytest.raises(ValueError, match="k"):
        RetrievalEvaluator().evaluate(
            cases,
            lambda query, k: [],
            k=0,
        )


def test_evaluator_handles_no_retrieved_results():
    cases = [
        EvaluationCase(
            query="Test",
            relevant_chunk_ids={uuid4()},
        )
    ]

    report = RetrievalEvaluator().evaluate(
        cases,
        lambda query, k: [],
        k=5,
    )

    assert report.per_query[0].retrieved_count == 0
    assert report.average_metrics.recall_at_k == 0.0


def test_evaluator_counts_unique_retrieved_chunks():
    chunk_id = uuid4()

    cases = [
        EvaluationCase(
            query="Test",
            relevant_chunk_ids={chunk_id},
        )
    ]

    report = RetrievalEvaluator().evaluate(
        cases,
        lambda query, k: [chunk_id, chunk_id],
        k=5,
    )

    assert report.per_query[0].retrieved_count == 1


def test_evaluation_case_rejects_empty_query():
    with pytest.raises(ValidationError):
        EvaluationCase(
            query="   ",
            relevant_chunk_ids={uuid4()},
        )


def test_evaluation_case_rejects_empty_relevant_ids():
    with pytest.raises(ValidationError):
        EvaluationCase(
            query="Test",
            relevant_chunk_ids=set(),
        )
