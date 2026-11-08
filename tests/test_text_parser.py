import hashlib

import pytest

from norag.ingestion.models import ElementType
from norag.ingestion.parsers import TextParser
from norag.ingestion.registry import ParserRegistry


def test_text_parser_extracts_paragraphs(tmp_path):
    source = tmp_path / "sample.txt"

    source.write_text(
        "First paragraph.\n\nSecond paragraph.",
        encoding="utf-8",
    )

    document = TextParser().parse(source)

    assert len(document.elements) == 2
    assert document.elements[0].text == "First paragraph."
    assert document.elements[1].text == "Second paragraph."


def test_text_parser_preserves_element_order(tmp_path):
    source = tmp_path / "ordered.txt"

    source.write_text(
        "Alpha.\n\nBeta.\n\nGamma.",
        encoding="utf-8",
    )

    document = TextParser().parse(source)

    assert [element.order for element in document.elements] == [0, 1, 2]


def test_text_parser_preserves_character_offsets(tmp_path):
    source = tmp_path / "offsets.txt"
    content = "First paragraph.\n\nSecond paragraph."

    source.write_text(content, encoding="utf-8")

    document = TextParser().parse(source)

    for element in document.elements:
        start = element.source_location.char_start
        end = element.source_location.char_end

        assert content[start:end] == element.text


def test_text_parser_generates_content_hash(tmp_path):
    source = tmp_path / "hash.txt"
    content = "NORAG document intelligence."

    source.write_text(content, encoding="utf-8")

    document = TextParser().parse(source)

    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    assert document.content_hash == expected_hash


def test_text_parser_handles_empty_document(tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")

    document = TextParser().parse(source)

    assert document.elements == []


def test_text_parser_rejects_invalid_utf8(tmp_path):
    source = tmp_path / "invalid.txt"
    source.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(UnicodeDecodeError):
        TextParser().parse(source)


def test_text_parser_rejects_missing_file(tmp_path):
    source = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        TextParser().parse(source)


def test_parser_registry_selects_text_parser(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("Hello NORAG.", encoding="utf-8")

    registry = ParserRegistry()
    registry.register(TextParser())

    document = registry.parse(source)

    assert document.elements[0].element_type == ElementType.PARAGRAPH


def test_parser_registry_rejects_unknown_extension(tmp_path):
    registry = ParserRegistry()
    registry.register(TextParser())

    with pytest.raises(ValueError, match="No parser registered"):
        registry.get_parser(tmp_path / "document.pdf")
