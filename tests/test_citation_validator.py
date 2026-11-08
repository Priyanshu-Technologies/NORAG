from uuid import uuid4

import pytest

from norag.prompting.models import PromptSource
from norag.verification.models import CitationStatus
from norag.verification.parser import CitationParser
from norag.verification.validator import CitationValidator


@pytest.fixture
def sources() -> list[PromptSource]:
    return [
        PromptSource(
            source_id="Source 1",
            position=1,
            document_id=uuid4(),
            chunk_id="chunk-1",
            section="Hybrid Retrieval",
            page_numbers=[],
            source_element_ids=[],
        ),
        PromptSource(
            source_id="Source 2",
            position=2,
            document_id=uuid4(),
            chunk_id="chunk-2",
            section="Reranking",
            page_numbers=[],
            source_element_ids=[],
        ),
    ]


@pytest.fixture
def validator() -> CitationValidator:
    return CitationValidator()


def test_extract_citations():
    parser = CitationParser()

    result = parser.extract_citations("Dense retrieval captures meaning [Source 1].")

    assert result == ["1"]


def test_extract_multiple_citations():
    parser = CitationParser()

    result = parser.extract_citations(
        "Hybrid retrieval combines both methods [Source 1] [Source 2]."
    )

    assert result == ["1", "2"]


def test_split_sentences():
    parser = CitationParser()

    result = parser.split_sentences(
        "Dense retrieval captures meaning. Sparse retrieval matches keywords."
    )

    assert len(result) == 2


def test_valid_citation(
    validator: CitationValidator,
    sources: list[PromptSource],
):
    report = validator.validate(
        "Dense retrieval captures semantic similarity [Source 1].",
        sources,
    )

    assert report.citations_are_valid
    assert report.cited_sentences == 1
    assert report.uncited_sentences == 0
    assert report.citation_coverage == 1.0


def test_invalid_source_reference(
    validator: CitationValidator,
    sources: list[PromptSource],
):
    report = validator.validate(
        "This claim references a missing source [Source 99].",
        sources,
    )

    assert not report.citations_are_valid
    assert report.invalid_citation_count == 1
    assert report.sentences[0].status == CitationStatus.INVALID


def test_uncited_sentence(
    validator: CitationValidator,
    sources: list[PromptSource],
):
    report = validator.validate(
        "Dense retrieval captures semantic similarity.",
        sources,
    )

    assert report.uncited_sentences == 1
    assert report.citation_coverage == 0.0
    assert report.sentences[0].status == CitationStatus.UNCITED


def test_mixed_citation_coverage(
    validator: CitationValidator,
    sources: list[PromptSource],
):
    answer = (
        "Dense retrieval captures semantic similarity [Source 1]. "
        "Sparse retrieval uses lexical matching."
    )

    report = validator.validate(answer, sources)

    assert report.total_sentences == 2
    assert report.cited_sentences == 1
    assert report.uncited_sentences == 1
    assert report.citation_coverage == 0.5


def test_malformed_citation(
    validator: CitationValidator,
    sources: list[PromptSource],
):
    report = validator.validate(
        "This sentence has a malformed citation [Source X].",
        sources,
    )

    assert not report.citations_are_valid
    assert report.invalid_citation_count == 1


def test_empty_answer_is_rejected(
    validator: CitationValidator,
    sources: list[PromptSource],
):
    with pytest.raises(ValueError):
        validator.validate("   ", sources)


def test_empty_source_map(
    validator: CitationValidator,
):
    report = validator.validate(
        "This answer cites a source [Source 1].",
        [],
    )

    assert report.invalid_citation_count == 1
    assert not report.citations_are_valid
