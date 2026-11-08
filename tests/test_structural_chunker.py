import pytest

from norag.chunking import ChunkingConfig, StructuralChunker
from norag.ingestion.models import (
    Document,
    DocumentElement,
    DocumentMetadata,
    ElementType,
)


def make_document(elements):
    return Document(
        source_uri="test://document",
        metadata=DocumentMetadata(title="NORAG test"),
        elements=elements,
    )


def make_element(element_type, text, order, metadata=None):
    return DocumentElement(
        element_type=element_type,
        text=text,
        order=order,
        metadata=metadata or {},
    )


def test_structural_chunker_creates_chunks():
    document = make_document(
        [
            make_element(ElementType.PARAGRAPH, "First paragraph.", 0),
            make_element(ElementType.PARAGRAPH, "Second paragraph.", 1),
        ]
    )

    chunks = StructuralChunker().chunk(document)

    assert len(chunks) == 1
    assert "First paragraph." in chunks[0].text
    assert "Second paragraph." in chunks[0].text


def test_structural_chunker_preserves_heading_context():
    document = make_document(
        [
            make_element(
                ElementType.HEADING,
                "Introduction",
                0,
                {"heading_level": 1},
            ),
            make_element(ElementType.PARAGRAPH, "Introduction content.", 1),
        ]
    )

    chunks = StructuralChunker().chunk(document)

    assert len(chunks) == 1
    assert "Introduction" in chunks[0].text
    assert chunks[0].metadata["section_path"] == "Introduction"


def test_structural_chunker_preserves_nested_headings():
    document = make_document(
        [
            make_element(ElementType.HEADING, "Chapter 1", 0, {"heading_level": 1}),
            make_element(ElementType.HEADING, "Section 1.1", 1, {"heading_level": 2}),
            make_element(ElementType.PARAGRAPH, "Section content.", 2),
        ]
    )

    chunks = StructuralChunker().chunk(document)

    assert chunks[0].metadata["section_path"] == ("Chapter 1 > Section 1.1")


def test_structural_chunker_splits_large_elements():
    document = make_document(
        [
            make_element(
                ElementType.PARAGRAPH,
                " ".join(["NORAG"] * 100),
                0,
            )
        ]
    )

    config = ChunkingConfig(
        target_tokens=30,
        max_tokens=40,
        overlap_tokens=5,
    )

    chunks = StructuralChunker(config).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 40 for chunk in chunks)


def test_structural_chunker_preserves_document_id():
    document = make_document([make_element(ElementType.PARAGRAPH, "NORAG content.", 0)])

    chunks = StructuralChunker().chunk(document)

    assert chunks[0].document_id == document.id


def test_structural_chunker_assigns_sequential_indices():
    document = make_document(
        [
            make_element(
                ElementType.PARAGRAPH,
                " ".join(["word"] * 500),
                0,
            )
        ]
    )

    config = ChunkingConfig(
        target_tokens=30,
        max_tokens=40,
        overlap_tokens=5,
    )
    chunks = StructuralChunker(config).chunk(document)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_structural_chunker_handles_empty_document():
    document = make_document([])

    assert StructuralChunker().chunk(document) == []


def test_chunking_config_rejects_invalid_limits():
    config = ChunkingConfig(
        target_tokens=100,
        max_tokens=50,
    )

    with pytest.raises(ValueError):
        config.validate_limits()


def test_structural_chunker_applies_overlap():
    document = make_document(
        [
            make_element(
                ElementType.PARAGRAPH,
                " ".join(["alpha"] * 20 + ["beta"] * 20),
                0,
            )
        ]
    )

    config = ChunkingConfig(
        target_tokens=10,
        max_tokens=15,
        overlap_tokens=3,
    )

    chunks = StructuralChunker(
        config,
        token_counter=lambda text: len(text.split()),
    ).chunk(document)

    assert len(chunks) > 1

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()

    assert first_words[-3:] == second_words[:3]


def test_structural_chunker_does_not_overlap_across_headings():
    document = make_document(
        [
            make_element(
                ElementType.HEADING,
                "First section",
                0,
                {"heading_level": 1},
            ),
            make_element(
                ElementType.PARAGRAPH,
                " ".join(["alpha"] * 20),
                1,
            ),
            make_element(
                ElementType.HEADING,
                "Second section",
                2,
                {"heading_level": 1},
            ),
            make_element(
                ElementType.PARAGRAPH,
                "Second section content.",
                3,
            ),
        ]
    )

    config = ChunkingConfig(
        target_tokens=10,
        max_tokens=15,
        overlap_tokens=3,
    )

    chunks = StructuralChunker(
        config,
        token_counter=lambda text: len(text.split()),
    ).chunk(document)

    second_section = next(
        chunk for chunk in chunks if chunk.metadata["section_path"] == "Second section"
    )

    assert "alpha" not in second_section.text
