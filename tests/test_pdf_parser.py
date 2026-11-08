from io import BytesIO

import pytest
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from norag.ingestion.models import ElementType
from norag.ingestion.parsers import PDFParser
from norag.ingestion.registry import ParserRegistry


def create_pdf(path, pages, title="NORAG Research", author="NORAG Team"):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.setTitle(title)
    pdf.setAuthor(author)

    for lines in pages:
        y = 750
        for line in lines:
            pdf.drawString(72, y, line)
            y -= 24
        pdf.showPage()

    pdf.save()
    path.write_bytes(buffer.getvalue())


def test_pdf_parser_supported_extensions():
    assert PDFParser().supported_extensions == (".pdf",)


def test_pdf_parser_extracts_text(tmp_path):
    source = tmp_path / "research.pdf"
    create_pdf(source, [["NORAG is a retrieval system."]])

    document = PDFParser().parse(source)

    assert len(document.elements) == 1
    assert document.elements[0].element_type == ElementType.PARAGRAPH
    assert "NORAG is a retrieval system." in document.elements[0].text


def test_pdf_parser_extracts_metadata(tmp_path):
    source = tmp_path / "research.pdf"
    create_pdf(source, [["Research content."]])

    document = PDFParser().parse(source)

    assert document.metadata.title == "NORAG Research"
    assert document.metadata.author == "NORAG Team"
    assert document.metadata.mime_type == "application/pdf"


def test_pdf_parser_preserves_page_numbers(tmp_path):
    source = tmp_path / "multipage.pdf"
    create_pdf(
        source,
        [
            ["Content on the first page."],
            ["Content on the second page."],
        ],
    )

    document = PDFParser().parse(source)

    assert len(document.elements) == 2
    assert document.elements[0].source_location.page_number == 1
    assert document.elements[1].source_location.page_number == 2


def test_pdf_parser_preserves_character_offsets(tmp_path):
    source = tmp_path / "offsets.pdf"
    create_pdf(source, [["Character offsets are important."]])

    document = PDFParser().parse(source)
    element = document.elements[0]

    assert element.source_location.char_start >= 0
    assert element.source_location.char_end > element.source_location.char_start


def test_pdf_parser_generates_content_hash(tmp_path):
    source = tmp_path / "hash.pdf"
    create_pdf(source, [["Hash verification."]])

    document = PDFParser().parse(source)

    assert len(document.content_hash) == 64


def test_pdf_parser_handles_empty_pages(tmp_path):
    source = tmp_path / "empty.pdf"
    create_pdf(source, [[]])

    document = PDFParser().parse(source)

    assert document.elements == []


def test_pdf_parser_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        PDFParser().parse(tmp_path / "missing.pdf")


def test_pdf_parser_rejects_encrypted_pdf(tmp_path):
    source = tmp_path / "encrypted.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret")

    with source.open("wb") as file:
        writer.write(file)

    with pytest.raises(ValueError, match="encrypted PDF"):
        PDFParser().parse(source)


def test_registry_selects_pdf_parser(tmp_path):
    source = tmp_path / "research.pdf"
    create_pdf(source, [["NORAG PDF test."]])

    registry = ParserRegistry()
    registry.register(PDFParser())

    document = registry.parse(source)

    assert document.elements[0].text == "NORAG PDF test."
