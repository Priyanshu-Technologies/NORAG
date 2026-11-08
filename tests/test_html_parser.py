from norag.ingestion.models import ElementType
from norag.ingestion.parsers import HTMLParser
from norag.ingestion.registry import ParserRegistry


def test_html_parser_supported_extensions():
    assert HTMLParser().supported_extensions == (".html", ".htm")


def test_html_parser_extracts_metadata(tmp_path):
    source = tmp_path / "page.html"
    source.write_text(
        '<html lang="en"><head>'
        "<title>NORAG Documentation</title>"
        '<meta name="author" content="NORAG Team">'
        "</head><body><p>Hello</p></body></html>",
        encoding="utf-8",
    )

    document = HTMLParser().parse(source)

    assert document.metadata.title == "NORAG Documentation"
    assert document.metadata.author == "NORAG Team"
    assert document.metadata.language == "en"


def test_html_parser_preserves_heading_hierarchy(tmp_path):
    source = tmp_path / "headings.html"
    source.write_text(
        "<h1>Machine Learning</h1><h2>Neural Networks</h2><h3>Backpropagation</h3>",
        encoding="utf-8",
    )

    document = HTMLParser().parse(source)

    headings = [
        item for item in document.elements if item.element_type == ElementType.HEADING
    ]

    assert len(headings) == 3
    assert headings[0].parent_id is None
    assert headings[1].parent_id == headings[0].id
    assert headings[2].parent_id == headings[1].id


def test_html_parser_extracts_paragraphs(tmp_path):
    source = tmp_path / "paragraph.html"
    source.write_text(
        "<p>NORAG is an intelligent retrieval system.</p>",
        encoding="utf-8",
    )

    document = HTMLParser().parse(source)

    assert document.elements[0].text == ("NORAG is an intelligent retrieval system.")


def test_html_parser_extracts_lists(tmp_path):
    source = tmp_path / "list.html"
    source.write_text(
        "<ul><li>Hybrid retrieval</li><li>Adaptive routing</li></ul>",
        encoding="utf-8",
    )

    document = HTMLParser().parse(source)

    items = [
        item for item in document.elements if item.element_type == ElementType.LIST_ITEM
    ]

    assert len(items) == 2
    assert items[0].text == "Hybrid retrieval"


def test_html_parser_extracts_tables(tmp_path):
    source = tmp_path / "table.html"
    source.write_text(
        "<table><tr><th>Model</th><th>Score</th></tr>"
        "<tr><td>NORAG</td><td>95</td></tr></table>",
        encoding="utf-8",
    )

    document = HTMLParser().parse(source)

    table = next(
        item for item in document.elements if item.element_type == ElementType.TABLE
    )

    assert table.structured_data["rows"] == [
        ["Model", "Score"],
        ["NORAG", "95"],
    ]


def test_html_parser_extracts_code_blocks(tmp_path):
    source = tmp_path / "code.html"
    source.write_text(
        "<pre><code class=\"language-python\">print('NORAG')</code></pre>",
        encoding="utf-8",
    )

    document = HTMLParser().parse(source)

    code = next(
        item for item in document.elements if item.element_type == ElementType.CODE
    )

    assert code.text == "print('NORAG')"
    assert code.metadata["language"] == "python"


def test_html_parser_ignores_scripts_and_navigation(tmp_path):
    source = tmp_path / "ignored.html"
    source.write_text(
        "<nav>Navigation content</nav>"
        "<script>secret_function()</script>"
        "<main><p>Actual document content</p></main>",
        encoding="utf-8",
    )

    document = HTMLParser().parse(source)

    extracted = " ".join(item.text for item in document.elements)

    assert "Actual document content" in extracted
    assert "Navigation content" not in extracted
    assert "secret_function" not in extracted


def test_registry_selects_html_parser(tmp_path):
    source = tmp_path / "page.html"
    source.write_text("<h1>NORAG</h1>", encoding="utf-8")

    registry = ParserRegistry()
    registry.register(HTMLParser())

    document = registry.parse(source)

    assert document.elements[0].text == "NORAG"
