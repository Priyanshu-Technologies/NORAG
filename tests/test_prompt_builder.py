from types import SimpleNamespace
from uuid import uuid4

import pytest

from norag.prompting.builder import PromptBuilder
from norag.prompting.models import PromptConfig


class FixedTokenCounter:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens

    def count(self, text: str) -> int:
        return self.tokens


def create_context():
    document_id = uuid4()
    chunk_id = uuid4()
    element_id = uuid4()

    chunk = SimpleNamespace(
        id=chunk_id,
        document_id=document_id,
        index=0,
        metadata={"section_path": ["Retrieval", "Hybrid Search"]},
        source_locations=[
            SimpleNamespace(page_number=3),
            SimpleNamespace(page_number=3),
            SimpleNamespace(page_number=5),
        ],
        source_element_ids=[element_id],
    )

    hybrid_result = SimpleNamespace(chunk=chunk)
    reranked_result = SimpleNamespace(result=hybrid_result)
    item = SimpleNamespace(position=1, result=reranked_result)

    context = SimpleNamespace(
        context_text=(
            "[Source 1]\n"
            "Section: Retrieval > Hybrid Search\n"
            "Content:\n"
            "Hybrid retrieval combines dense and sparse search."
        ),
        items=[item],
        token_count=20,
        available_tokens=100,
        included_count=1,
        excluded_count=0,
    )

    return context, document_id, chunk_id, element_id


def test_prompt_builder_creates_generation_request():
    context, _, _, _ = create_context()

    result = PromptBuilder().build(
        query="How does hybrid retrieval work?",
        context=context,
    )

    assert result.request.user_prompt
    assert result.request.system_prompt
    assert result.request.temperature == 0.2
    assert result.request.max_output_tokens == 512


def test_prompt_contains_query_and_evidence():
    context, _, _, _ = create_context()

    result = PromptBuilder().build(
        query="How does hybrid retrieval work?",
        context=context,
    )

    assert "How does hybrid retrieval work?" in result.request.user_prompt
    assert "Hybrid retrieval combines dense and sparse search." in (
        result.request.user_prompt
    )


def test_prompt_contains_citation_instructions():
    context, _, _, _ = create_context()

    result = PromptBuilder().build(
        query="Explain hybrid retrieval.",
        context=context,
    )

    assert "[Source 1]" in result.request.user_prompt
    assert "Never invent citations" in result.request.user_prompt


def test_prompt_preserves_source_mapping():
    context, document_id, chunk_id, element_id = create_context()

    result = PromptBuilder().build(
        query="Explain hybrid retrieval.",
        context=context,
    )

    assert len(result.sources) == 1

    source = result.sources[0]

    assert source.source_id == "Source 1"
    assert source.document_id == document_id
    assert source.chunk_id == str(chunk_id)
    assert source.section == "Retrieval > Hybrid Search"
    assert source.page_numbers == [3, 5]
    assert source.source_element_ids == [element_id]


def test_prompt_rejects_empty_query():
    context, _, _, _ = create_context()

    with pytest.raises(ValueError, match="Query cannot be empty"):
        PromptBuilder().build(
            query="   ",
            context=context,
        )


def test_prompt_rejects_exceeding_token_budget():
    context, _, _, _ = create_context()

    builder = PromptBuilder(
        config=PromptConfig(max_input_tokens=100),
        token_counter=FixedTokenCounter(tokens=150),
    )

    with pytest.raises(ValueError, match="exceeds the input token budget"):
        builder.build(
            query="Explain hybrid retrieval.",
            context=context,
        )


def test_prompt_supports_empty_evidence():
    context, _, _, _ = create_context()
    context.context_text = ""
    context.items = []

    result = PromptBuilder().build(
        query="Explain hybrid retrieval.",
        context=context,
    )

    assert "No retrieved evidence was available." in (result.request.user_prompt)
    assert result.sources == []


def test_prompt_uses_custom_configuration():
    context, _, _, _ = create_context()

    config = PromptConfig(
        system_prompt="You are a scientific assistant.",
        temperature=0.5,
        max_output_tokens=256,
    )

    result = PromptBuilder(config=config).build(
        query="Explain hybrid retrieval.",
        context=context,
    )

    assert result.request.system_prompt == ("You are a scientific assistant.")
    assert result.request.temperature == 0.5
    assert result.request.max_output_tokens == 256


def test_prompt_reports_estimated_token_count():
    context, _, _, _ = create_context()

    result = PromptBuilder(
        token_counter=FixedTokenCounter(tokens=120),
    ).build(
        query="Explain hybrid retrieval.",
        context=context,
    )

    assert result.estimated_input_tokens == 120
