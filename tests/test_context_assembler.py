from types import SimpleNamespace
from uuid import uuid4

import pytest

from norag.context import ContextAssembler, ContextConfig
from norag.reranking.models import RerankedSearchResult


class FakeTokenCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def make_result(
    text: str,
    score: float = 1.0,
    section: str = "Test Section",
) -> RerankedSearchResult:
    chunk = SimpleNamespace(
        id=uuid4(),
        text=text,
        metadata={"section_path": section},
    )

    hybrid_result = SimpleNamespace(chunk=chunk)

    return RerankedSearchResult.model_construct(
        result=hybrid_result,
        rerank_score=score,
        original_rank=1,
    )


def test_assembler_includes_relevant_results() -> None:
    assembler = ContextAssembler(token_counter=FakeTokenCounter())

    results = [make_result("Hybrid retrieval combines dense and sparse search.")]

    context = assembler.assemble(results)

    assert context.included_count == 1
    assert context.excluded_count == 0
    assert "Hybrid retrieval" in context.context_text


def test_assembler_preserves_section_information() -> None:
    assembler = ContextAssembler(token_counter=FakeTokenCounter())

    context = assembler.assemble(
        [make_result("BM25 uses lexical matching.", section="Sparse Retrieval")]
    )

    assert "Sparse Retrieval" in context.context_text


def test_assembler_respects_token_budget() -> None:
    assembler = ContextAssembler(
        config=ContextConfig(max_tokens=12),
        token_counter=FakeTokenCounter(),
    )

    results = [
        make_result("one two three four five six"),
        make_result("seven eight nine ten eleven twelve"),
    ]

    context = assembler.assemble(results)

    assert context.token_count <= context.available_tokens
    assert context.included_count < 2


def test_assembler_respects_reserved_tokens() -> None:
    assembler = ContextAssembler(
        config=ContextConfig(max_tokens=20),
        token_counter=FakeTokenCounter(),
    )

    context = assembler.assemble(
        [make_result("One two three.")],
        reserved_tokens=5,
    )

    assert context.available_tokens == 15


def test_assembler_filters_low_relevance_results() -> None:
    assembler = ContextAssembler(
        config=ContextConfig(min_relevance_score=0.5),
        token_counter=FakeTokenCounter(),
    )

    context = assembler.assemble(
        [
            make_result("Relevant passage.", score=1.0),
            make_result("Weak passage.", score=0.1),
        ]
    )

    assert context.included_count == 1
    assert context.excluded_count == 1


def test_assembler_deduplicates_identical_text() -> None:
    assembler = ContextAssembler(token_counter=FakeTokenCounter())

    context = assembler.assemble(
        [
            make_result("Identical passage."),
            make_result("Identical passage."),
        ]
    )

    assert context.included_count == 1
    assert context.excluded_count == 1


def test_assembler_assigns_sequential_positions() -> None:
    assembler = ContextAssembler(token_counter=FakeTokenCounter())

    context = assembler.assemble(
        [
            make_result("First passage."),
            make_result("Second passage."),
        ]
    )

    assert [item.position for item in context.items] == [1, 2]


def test_assembler_rejects_negative_reserved_tokens() -> None:
    assembler = ContextAssembler()

    with pytest.raises(ValueError, match="reserved_tokens"):
        assembler.assemble([], reserved_tokens=-1)


def test_assembler_rejects_exhausted_context_budget() -> None:
    assembler = ContextAssembler(
        config=ContextConfig(max_tokens=100),
    )

    with pytest.raises(ValueError, match="context budget"):
        assembler.assemble([], reserved_tokens=100)
