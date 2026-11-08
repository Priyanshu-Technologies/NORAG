import pytest
from pydantic import ValidationError

from norag.ingestion.models import (
    Document,
    DocumentElement,
    DocumentMetadata,
    ElementType,
    SourceLocation,
)


def test_document_creation():
    document = Document(
        source_uri="research/paper.pdf",
        metadata=DocumentMetadata(
            title="Attention Is All You Need",
            author="Research Team",
            file_name="paper.pdf",
        ),
    )

    assert document.metadata.title == "Attention Is All You Need"
    assert document.schema_version == "1.0"
    assert document.elements == []


def test_document_preserves_element_structure():
    heading = DocumentElement(
        element_type=ElementType.HEADING,
        text="Introduction",
        order=0,
        source_location=SourceLocation(page_number=1),
    )

    paragraph = DocumentElement(
        element_type=ElementType.PARAGRAPH,
        text="This paper introduces a new architecture.",
        order=1,
        parent_id=heading.id,
        source_location=SourceLocation(page_number=1),
    )

    document = Document(elements=[heading, paragraph])

    assert len(document.elements) == 2
    assert document.elements[0].element_type == ElementType.HEADING
    assert document.elements[1].parent_id == heading.id


def test_invalid_page_number():
    with pytest.raises(ValidationError):
        SourceLocation(page_number=0)


def test_invalid_character_offsets():
    with pytest.raises(ValidationError):
        SourceLocation(
            char_start=100,
            char_end=50,
        )


def test_duplicate_element_orders():
    first = DocumentElement(
        element_type=ElementType.PARAGRAPH,
        text="First paragraph",
        order=0,
    )

    second = DocumentElement(
        element_type=ElementType.PARAGRAPH,
        text="Second paragraph",
        order=0,
    )

    with pytest.raises(ValidationError):
        Document(elements=[first, second])


def test_document_serialization():
    document = Document(metadata=DocumentMetadata(title="Test Paper"))

    serialized = document.model_dump(mode="json")

    assert serialized["metadata"]["title"] == "Test Paper"
    assert serialized["schema_version"] == "1.0"
