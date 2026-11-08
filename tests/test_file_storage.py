from norag.ingestion.models import Document, DocumentMetadata
from norag.storage.filesystem import LocalFileStorage


def test_store_preserves_original_file(tmp_path):
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"PDF test content")

    storage = LocalFileStorage(tmp_path / "managed")
    document = Document(metadata=DocumentMetadata(file_name="sample.pdf"))

    stored_path = storage.store(source, document)

    assert stored_path.is_file()
    assert stored_path.read_bytes() == b"PDF test content"
    assert stored_path.parent == tmp_path / "managed"


def test_storage_uses_generated_filename(tmp_path):
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"PDF test content")

    storage = LocalFileStorage(tmp_path / "managed")
    document = Document(metadata=DocumentMetadata(file_name="../../sample.pdf"))

    stored_path = storage.store(source, document)

    assert stored_path.parent == tmp_path / "managed"
    assert stored_path.name == f"{document.id.hex}.pdf"


def test_delete_removes_original_file(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("NORAG file storage")

    storage = LocalFileStorage(tmp_path / "managed")
    document = Document(metadata=DocumentMetadata(file_name="sample.txt"))

    stored_path = storage.store(source, document)

    assert storage.delete(document) is True
    assert not stored_path.exists()


def test_delete_returns_false_for_missing_file(tmp_path):
    storage = LocalFileStorage(tmp_path / "managed")
    document = Document(metadata=DocumentMetadata(file_name="missing.pdf"))

    assert storage.delete(document) is False
