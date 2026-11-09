import pytest

from norag.query_understanding.models import (
    QueryAnalysis,
    QueryIntent,
    RetrievalRoute,
)
from norag.retrieval_policy import AdaptiveRetrievalPolicy


def make_analysis(intent: QueryIntent) -> QueryAnalysis:
    return QueryAnalysis(
        query="How does this system work?",
        intent=intent,
        route=RetrievalRoute.HYBRID,
        dense_weight=0.5,
        sparse_weight=0.5,
        rule="test",
    )


@pytest.mark.parametrize(
    ("intent", "expected_multiplier", "expected_dense", "expected_sparse"),
    [
        (QueryIntent.SEMANTIC, 3, 0.80, 0.20),
        (QueryIntent.EXACT, 2, 0.20, 0.80),
        (QueryIntent.COMPARATIVE, 5, 0.50, 0.50),
        (QueryIntent.MULTI_HOP, 6, 0.55, 0.45),
    ],
)
def test_policy_selects_settings_by_intent(
    intent: QueryIntent,
    expected_multiplier: int,
    expected_dense: float,
    expected_sparse: float,
) -> None:
    policy = AdaptiveRetrievalPolicy()
    plan = policy.create_plan(make_analysis(intent), top_k=5)

    assert plan.intent == intent
    assert plan.final_top_k == 5
    assert plan.candidate_k == 5 * expected_multiplier
    assert plan.dense_weight == expected_dense
    assert plan.sparse_weight == expected_sparse


def test_policy_respects_max_candidate_limit() -> None:
    policy = AdaptiveRetrievalPolicy(max_candidates=20)

    plan = policy.create_plan(
        make_analysis(QueryIntent.MULTI_HOP),
        top_k=10,
    )

    assert plan.candidate_k == 20


def test_policy_respects_minimum_candidate_limit() -> None:
    policy = AdaptiveRetrievalPolicy(min_candidates=10)

    plan = policy.create_plan(
        make_analysis(QueryIntent.EXACT),
        top_k=2,
    )

    assert plan.candidate_k == 10


def test_policy_rejects_invalid_top_k() -> None:
    policy = AdaptiveRetrievalPolicy()

    with pytest.raises(ValueError, match="top_k"):
        policy.create_plan(make_analysis(QueryIntent.SEMANTIC), top_k=0)


def test_policy_rejects_invalid_candidate_limits() -> None:
    with pytest.raises(ValueError, match="max_candidates"):
        AdaptiveRetrievalPolicy(min_candidates=20, max_candidates=10)


def test_policy_rejects_zero_minimum_candidates() -> None:
    with pytest.raises(ValueError, match="min_candidates"):
        AdaptiveRetrievalPolicy(min_candidates=0)


def test_policy_rejects_top_k_above_maximum() -> None:
    policy = AdaptiveRetrievalPolicy(max_candidates=10)

    with pytest.raises(ValueError, match="max_candidates"):
        policy.create_plan(
            make_analysis(QueryIntent.SEMANTIC),
            top_k=11,
        )


def test_policy_accepts_weight_overrides() -> None:
    policy = AdaptiveRetrievalPolicy()

    plan = policy.create_plan(
        make_analysis(QueryIntent.SEMANTIC),
        top_k=5,
        dense_weight=0.0,
        sparse_weight=1.0,
    )

    assert plan.dense_weight == 0.0
    assert plan.sparse_weight == 1.0


def test_policy_rejects_partial_weight_override() -> None:
    policy = AdaptiveRetrievalPolicy()

    with pytest.raises(ValueError, match="provided together"):
        policy.create_plan(
            make_analysis(QueryIntent.SEMANTIC),
            top_k=5,
            dense_weight=1.0,
        )
