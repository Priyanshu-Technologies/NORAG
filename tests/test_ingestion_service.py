import pytest

from norag.ingestion.factory import create_ingestion_service


def test_ingestion_service_parses_text(tmp_path):
    source = tmp_path / "document.txt"
    source.write_text("NORAG ingestion test.", encoding="utf-8")

    service = create_ingestion_service()
    document = service.ingest(source)

    assert document.elements[0].text == "NORAG ingestion test."


def test_ingestion_service_parses_markdown(tmp_path):
    source = tmp_path / "document.md"
    source.write_text("# NORAG", encoding="utf-8")

    service = create_ingestion_service()
    document = service.ingest(source)

    assert document.elements[0].text == "NORAG"


def test_ingestion_service_parses_html(tmp_path):
    source = tmp_path / "document.html"
    source.write_text("<p>NORAG HTML</p>", encoding="utf-8")

    service = create_ingestion_service()
    document = service.ingest(source)

    assert document.elements[0].text == "NORAG HTML"


def test_ingestion_service_rejects_missing_file(tmp_path):
    service = create_ingestion_service()

    with pytest.raises(FileNotFoundError):
        service.ingest(tmp_path / "missing.txt")


def test_ingestion_service_rejects_empty_file(tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")

    service = create_ingestion_service()

    with pytest.raises(ValueError, match="empty"):
        service.ingest(source)


def test_ingestion_service_rejects_directory(tmp_path):
    service = create_ingestion_service()

    with pytest.raises(ValueError, match="not a file"):
        service.ingest(tmp_path)
