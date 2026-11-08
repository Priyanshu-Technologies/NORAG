from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from norag.api.routes import validate_filename
import norag.api.main as api_main


class FakeDocumentManager:
    def __init__(self):
        self.documents = {}
        self.deleted_ids = []

    def upload(self, source):
        document = SimpleNamespace(
            id=uuid4(),
            content_hash="a" * 64,
            metadata=SimpleNamespace(
                file_name=source.name,
                title="Test document",
            ),
            created_at=datetime.now(UTC),
        )

        self.documents[document.id] = document

        return SimpleNamespace(document=document, chunk_count=3)

    def list_documents(self):
        return list(self.documents.values())

    def delete(self, document_id):
        if document_id not in self.documents:
            return False

        del self.documents[document_id]
        self.deleted_ids.append(document_id)
        return True


class FakeAnsweringService:
    def answer(self, query, top_k=5):
        return {
            "query": query,
            "answer": "NORAG test answer.",
            "top_k": top_k,
        }


@pytest.fixture
def client(monkeypatch):
    manager = FakeDocumentManager()

    services = SimpleNamespace(
        ingestion_service=object(),
        chunker=object(),
        embedding_service=object(),
        dense_index=object(),
        sparse_index=object(),
        answering_service=FakeAnsweringService(),
    )

    monkeypatch.setattr(
        api_main,
        "create_application_services",
        lambda settings: services,
    )

    monkeypatch.setattr(
        api_main,
        "DocumentManager",
        lambda **kwargs: manager,
    )

    with TestClient(api_main.create_app()) as test_client:
        yield test_client, manager


def test_root_endpoint(client):
    test_client, _ = client

    response = test_client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "NORAG"
    assert response.json()["status"] == "operational"


def test_health_endpoint(client):
    test_client, _ = client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_upload_document(client):
    test_client, manager = client

    response = test_client.post(
        "/documents",
        files={"file": ("sample.txt", b"NORAG test document.", "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["file_name"] == "sample.txt"
    assert response.json()["chunk_count"] == 3
    assert len(manager.documents) == 1


def test_upload_rejects_unsupported_file(client):
    test_client, manager = client

    response = test_client.post(
        "/documents",
        files={"file": ("sample.exe", b"invalid file", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert manager.documents == {}


def test_upload_rejects_empty_file(client):
    test_client, manager = client

    response = test_client.post(
        "/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert manager.documents == {}


def test_list_documents(client):
    test_client, manager = client

    test_client.post(
        "/documents",
        files={"file": ("sample.txt", b"Test content", "text/plain")},
    )

    response = test_client.get("/documents")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["file_name"] == "sample.txt"


def test_delete_document(client):
    test_client, manager = client

    upload_response = test_client.post(
        "/documents",
        files={"file": ("sample.txt", b"Test content", "text/plain")},
    )

    document_id = upload_response.json()["document_id"]

    response = test_client.delete(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert manager.documents == {}


def test_delete_unknown_document(client):
    test_client, _ = client

    response = test_client.delete(f"/documents/{uuid4()}")

    assert response.status_code == 404


def test_query_endpoint(client):
    test_client, _ = client

    response = test_client.post(
        "/query",
        json={"query": "What is NORAG?", "top_k": 5},
    )

    assert response.status_code == 200
    assert response.json()["result"]["answer"] == "NORAG test answer."


def test_query_rejects_empty_query(client):
    test_client, _ = client

    response = test_client.post(
        "/query",
        json={"query": "", "top_k": 5},
    )

    assert response.status_code == 422


def test_query_rejects_invalid_top_k(client):
    test_client, _ = client

    response = test_client.post(
        "/query",
        json={"query": "What is NORAG?", "top_k": 0},
    )

    assert response.status_code == 422


def test_upload_rejects_invalid_pdf_signature(client):
    test_client, manager = client

    response = test_client.post(
        "/documents",
        files={
            "file": (
                "fake.pdf",
                b"This is not a PDF file.",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415
    assert manager.documents == {}


def test_upload_rejects_invalid_utf8_text(client):
    test_client, manager = client

    response = test_client.post(
        "/documents",
        files={
            "file": (
                "invalid.txt",
                b"\xff\xfe\xfa",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
    assert manager.documents == {}


def test_upload_rejects_binary_content_disguised_as_text(client):
    test_client, manager = client

    response = test_client.post(
        "/documents",
        files={
            "file": (
                "binary.txt",
                b"valid text\x00binary data",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
    assert manager.documents == {}


def test_upload_accepts_pdf_signature(client):
    test_client, manager = client

    response = test_client.post(
        "/documents",
        files={
            "file": (
                "sample.pdf",
                b"%PDF-1.7\nFake PDF content for API testing.",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["file_name"] == "sample.pdf"
    assert len(manager.documents) == 1


def test_validate_filename_rejects_control_characters():
    with pytest.raises(HTTPException) as exc_info:
        validate_filename("sample\x01.txt")

    assert exc_info.value.status_code == 400


def test_validate_filename_rejects_empty_filename():
    with pytest.raises(HTTPException) as exc_info:
        validate_filename("")

    assert exc_info.value.status_code == 400


def test_validate_filename_sanitizes_path():
    assert validate_filename("../../sample.txt") == "sample.txt"


def test_upload_rejects_file_over_size_limit(client, monkeypatch):
    import norag.api.routes as api_routes

    test_client, manager = client

    monkeypatch.setattr(api_routes, "MAX_UPLOAD_BYTES", 10)

    response = test_client.post(
        "/documents",
        files={
            "file": (
                "large.txt",
                b"This file is larger than ten bytes.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 413
    assert manager.documents == {}
