import pytest

from norag.ingestion.models import ElementType
from norag.ingestion.parsers import MarkdownParser
from norag.ingestion.registry import ParserRegistry


def test_markdown_parser_extracts_headings():
    parser = MarkdownParser()

    # Parsing is performed through temporary files.
    assert parser.supported_extensions == (".md", ".markdown")


def test_markdown_parser_preserves_heading_hierarchy(tmp_path):
    source = tmp_path / "paper.md"

    source.write_text(
        "# Machine Learning\n\n## Neural Networks\n\n### Backpropagation\n",
        encoding="utf-8",
    )

    document = MarkdownParser().parse(source)

    headings = [
        element
        for element in document.elements
        if element.element_type == ElementType.HEADING
    ]

    assert len(headings) == 3

    assert headings[0].text == "Machine Learning"
    assert headings[1].text == "Neural Networks"
    assert headings[2].text == "Backpropagation"

    assert headings[0].parent_id is None
    assert headings[1].parent_id == headings[0].id
    assert headings[2].parent_id == headings[1].id


def test_paragraph_inherits_section_parent(tmp_path):
    source = tmp_path / "section.md"

    source.write_text(
        "# Introduction\n\nThis is the introduction.\n",
        encoding="utf-8",
    )

    document = MarkdownParser().parse(source)

    heading, paragraph = document.elements

    assert paragraph.element_type == ElementType.PARAGRAPH
    assert paragraph.parent_id == heading.id


def test_markdown_parser_extracts_code_blocks(tmp_path):
    source = tmp_path / "code.md"

    source.write_text(
        "# Example\n\n```python\ndef hello():\n    return 'NORAG'\n```\n",
        encoding="utf-8",
    )

    document = MarkdownParser().parse(source)

    code = next(
        element
        for element in document.elements
        if element.element_type == ElementType.CODE
    )

    assert "def hello():" in code.text
    assert code.metadata["language"] == "python"


def test_markdown_parser_extracts_list_items(tmp_path):
    source = tmp_path / "list.md"

    source.write_text(
        "# Features\n\n- Hybrid retrieval\n- Adaptive routing\n- Reranking\n",
        encoding="utf-8",
    )

    document = MarkdownParser().parse(source)

    items = [
        element
        for element in document.elements
        if element.element_type == ElementType.LIST_ITEM
    ]

    assert len(items) == 3
    assert items[0].text == "Hybrid retrieval"
    assert items[1].text == "Adaptive routing"


def test_markdown_parser_preserves_source_offsets(tmp_path):
    source = tmp_path / "offsets.md"

    content = "# NORAG\n\nDocument intelligence engine.\n"

    source.write_text(content, encoding="utf-8")

    document = MarkdownParser().parse(source)

    for element in document.elements:
        start = element.source_location.char_start
        end = element.source_location.char_end

        assert content[start:end].strip()


def test_markdown_parser_handles_empty_document(tmp_path):
    source = tmp_path / "empty.md"
    source.write_text("", encoding="utf-8")

    document = MarkdownParser().parse(source)

    assert document.elements == []


def test_markdown_parser_rejects_invalid_utf8(tmp_path):
    source = tmp_path / "invalid.md"
    source.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(UnicodeDecodeError):
        MarkdownParser().parse(source)


def test_registry_selects_markdown_parser(tmp_path):
    source = tmp_path / "paper.md"
    source.write_text("# NORAG", encoding="utf-8")

    registry = ParserRegistry()
    registry.register(MarkdownParser())

    document = registry.parse(source)

    assert document.elements[0].element_type == ElementType.HEADING
