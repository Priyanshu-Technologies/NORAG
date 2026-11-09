from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import norag.api.main as api_main
from norag.answering.models import AnswerResponse
from norag.api.document_manager import DocumentManagerStats
from norag.api.routes import validate_filename
from norag.context.models import AssembledContext
from norag.generation.models import GenerationRequest, GenerationResponse
from norag.prompting.models import PromptConstruction
from norag.query_understanding.models import (
    QueryAnalysis,
    QueryIntent,
    RetrievalRoute,
)
from norag.retrieval_policy.models import RetrievalPlan
from norag.retrieval_policy.service import AdaptiveRetrievalResponse
from norag.verification.models import (
    CitationValidationReport,
    ClaimVerificationReport,
)


def make_answer(query: str, top_k: int = 5) -> AnswerResponse:
    retrieval = AdaptiveRetrievalResponse(
        analysis=QueryAnalysis(
            query=query,
            intent=QueryIntent.SEMANTIC,
            route=RetrievalRoute.DENSE,
            dense_weight=1.0,
            sparse_weight=0.0,
            rule="semantic_fallback",
        ),
        plan=RetrievalPlan(
            intent=QueryIntent.SEMANTIC,
            final_top_k=top_k,
            candidate_k=max(top_k * 3, 5),
            dense_weight=0.8,
            sparse_weight=0.2,
        ),
        results=[],
    )

    context = AssembledContext(
        items=[],
        context_text="",
        token_count=0,
        available_tokens=3000,
        included_count=0,
        excluded_count=0,
    )

    generation = GenerationResponse(
        content="NORAG test answer.",
        provider_name="fake",
        model_name="fake-model",
    )

    prompt = PromptConstruction(
        request=GenerationRequest(user_prompt=query),
        sources=[],
        estimated_input_tokens=1,
    )

    citations = CitationValidationReport(
        total_sentences=1,
        cited_sentences=0,
        uncited_sentences=1,
        invalid_citation_count=0,
        citation_coverage=0.0,
    )

    claims = ClaimVerificationReport(
        total_claims=0,
        supported_claims=0,
        contradicted_claims=0,
        insufficient_claims=0,
    )

    return AnswerResponse(
        query=query,
        answer=generation.content,
        retrieval=retrieval,
        context=context,
        prompt=prompt,
        generation=generation,
        citation_validation=citations,
        claim_verification=claims,
    )


class FakeDocumentManager:
    def __init__(self):
        self.documents = {}
        self.deleted_ids = []
        self.replace_flags = []

    def upload(self, source, replace_existing=False):
        self.replace_flags.append(replace_existing)

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

    def stats(self):
        chunks = len(self.documents) * 3

        return DocumentManagerStats(
            documents=len(self.documents),
            chunks=chunks,
            dense_chunks=chunks,
            sparse_chunks=chunks,
        )

    def delete(self, document_id):
        if document_id not in self.documents:
            return False

        del self.documents[document_id]
        self.deleted_ids.append(document_id)
        return True


class FakeAnsweringService:
    def answer(self, query, top_k=5):
        return make_answer(query, top_k)


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

    monkeypatch.setattr(
        api_main,
        "mount_frontend",
        lambda app, directory: False,
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


def test_requests_carry_a_request_id(client):
    test_client, _ = client

    response = test_client.get("/health")

    assert response.headers["X-Request-ID"]


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


def test_upload_forwards_replace_flag(client):
    test_client, manager = client

    response = test_client.post(
        "/documents?replace=true",
        files={"file": ("sample.txt", b"NORAG test document.", "text/plain")},
    )

    assert response.status_code == 201
    assert manager.replace_flags == [True]


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
    test_client, _ = client

    test_client.post(
        "/documents",
        files={"file": ("sample.txt", b"Test content", "text/plain")},
    )

    response = test_client.get("/documents")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["file_name"] == "sample.txt"


def test_list_documents_supports_pagination(client):
    test_client, _ = client

    for index in range(3):
        test_client.post(
            "/documents",
            files={"file": (f"sample-{index}.txt", b"Test content", "text/plain")},
        )

    response = test_client.get("/documents?limit=1&offset=1")

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_list_documents_rejects_invalid_pagination(client):
    test_client, _ = client

    assert test_client.get("/documents?limit=0").status_code == 422
    assert test_client.get("/documents?offset=-1").status_code == 422


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
    assert response.json()["result"]["retrieval"]["plan"]["final_top_k"] == 5


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


def test_stats_endpoint_reports_counts(client):
    test_client, _ = client

    test_client.post(
        "/documents",
        files={"file": ("sample.txt", b"Test content", "text/plain")},
    )

    response = test_client.get("/stats")

    payload = response.json()

    assert response.status_code == 200
    assert payload["documents"] == 1
    assert payload["chunks"] == 3
    assert payload["dense_chunks"] == 3
    assert payload["uploads"] == 1
    assert payload["environment"] == "development"
    assert payload["ollama_model"] == "qwen3:1.7b"


def test_routes_are_available_under_the_api_prefix(client):
    test_client, _ = client

    prefixed = test_client.get("/api/stats")
    plain = test_client.get("/stats")

    assert prefixed.status_code == 200
    assert prefixed.json() == plain.json()


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
