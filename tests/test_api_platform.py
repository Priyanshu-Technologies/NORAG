import json
import sqlite3
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import norag.api.main as api_main
from norag.api.document_manager import DocumentManagerStats
from norag.config import Settings


class FakeVectorIndex:
    def __init__(self):
        self.count = 0
        self.calls = []

    def search(self, vector, top_k=5, document_id=None, metadata_filters=None):
        self.calls.append(("dense", top_k))
        return []


class FakeSparseIndex:
    def __init__(self):
        self.count = 0
        self.calls = []

    def search(self, query, top_k=5, document_id=None, metadata_filters=None):
        self.calls.append(("sparse", top_k))
        return []


class FakeEmbeddingService:
    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


class FakeHybridRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k=5, candidate_k=None):
        self.calls.append({"query": query, "top_k": top_k, "candidate_k": candidate_k})
        return []


class FakeRerankingService:
    def rerank(self, query, candidates, top_k=None):
        return []


class FakeManager:
    def __init__(self):
        self.documents = {}

    def upload(self, source, replace_existing=False):
        raise AssertionError("not used")

    def list_documents(self):
        return []

    def stats(self):
        return DocumentManagerStats(
            documents=0,
            chunks=0,
            dense_chunks=0,
            sparse_chunks=0,
        )

    def delete(self, document_id):
        return False


class FakeServices:
    def __init__(self):
        self.ingestion_service = object()
        self.chunker = object()
        self.embedding_service = FakeEmbeddingService()
        self.dense_index = FakeVectorIndex()
        self.sparse_index = FakeSparseIndex()
        self.hybrid_retriever = FakeHybridRetriever()
        self.reranking_service = FakeRerankingService()


@pytest.fixture
def build_app(monkeypatch):
    manager = FakeManager()
    services = FakeServices()

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

    def factory(settings: Settings) -> TestClient:
        return TestClient(api_main.create_app(settings))

    return factory, manager, services


def make_settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "NORAG_OLLAMA_READY_CHECK": False,
    }
    values.update(overrides)
    return Settings(**values)


def test_api_key_is_optional_by_default(build_app):
    factory, _, _ = build_app

    with factory(make_settings()) as client:
        assert client.get("/api/stats").status_code == 200


def test_api_key_rejects_missing_credentials(build_app):
    factory, _, _ = build_app

    with factory(make_settings(NORAG_API_KEY="top-secret")) as client:
        assert client.get("/api/stats").status_code == 401


def test_api_key_accepts_header_and_bearer(build_app):
    factory, _, _ = build_app

    with factory(make_settings(NORAG_API_KEY="top-secret")) as client:
        assert (
            client.get("/api/stats", headers={"X-API-Key": "top-secret"}).status_code
            == 200
        )
        assert (
            client.get(
                "/api/stats",
                headers={"Authorization": "Bearer top-secret"},
            ).status_code
            == 200
        )
        assert (
            client.get("/api/stats", headers={"X-API-Key": "wrong"}).status_code == 401
        )


def test_health_endpoint_stays_public_with_api_key(build_app):
    factory, _, _ = build_app

    with factory(make_settings(NORAG_API_KEY="top-secret")) as client:
        assert client.get("/health").status_code == 200


def test_cors_headers_are_emitted_for_allowed_origins(build_app):
    factory, _, _ = build_app

    settings = make_settings(NORAG_CORS_ALLOW_ORIGINS="https://norag.example")

    with factory(settings) as client:
        response = client.options(
            "/api/query",
            headers={
                "Origin": "https://norag.example",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://norag.example"


def test_no_cors_headers_for_unlisted_origins(build_app):
    factory, _, _ = build_app

    settings = make_settings(NORAG_CORS_ALLOW_ORIGINS="https://norag.example")

    with factory(settings) as client:
        response = client.options(
            "/api/query",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert "access-control-allow-origin" not in response.headers


def test_ready_reports_component_status(build_app, tmp_path):
    factory, _, _ = build_app

    storage = tmp_path / "documents"
    storage.mkdir()

    settings = make_settings(
        NORAG_DATABASE_PATH=str(tmp_path / "norag.sqlite3"),
        NORAG_FILE_STORAGE_PATH=str(storage),
    )

    with factory(settings) as client:
        response = client.get("/ready")

    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ready"
    assert payload["checks"]["ollama"] == "skipped"
    assert payload["checks"]["storage"] == "ok"


def test_ready_degrades_when_storage_is_missing(build_app, tmp_path):
    factory, _, _ = build_app

    settings = make_settings(
        NORAG_DATABASE_PATH=str(tmp_path / "norag.sqlite3"),
        NORAG_FILE_STORAGE_PATH=str(tmp_path / "missing"),
    )

    with factory(settings) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"]["storage"] == "missing"


def test_frontend_is_served_with_spa_fallback(build_app, tmp_path):
    factory, _, _ = build_app

    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)

    (dist / "index.html").write_text("<html>NORAG UI</html>", encoding="utf-8")
    (assets / "app.js").write_text("console.log('norag')", encoding="utf-8")

    settings = make_settings(NORAG_FRONTEND_DIST_PATH=str(dist))

    with factory(settings) as client:
        root = client.get("/")
        deep_link = client.get("/documents")
        asset = client.get("/assets/app.js")
        api = client.get("/api/documents")

    assert root.text == "<html>NORAG UI</html>"
    assert deep_link.text == "<html>NORAG UI</html>"
    assert asset.text == "console.log('norag')"
    assert api.json() == []


def test_static_handler_blocks_path_traversal(build_app, tmp_path):
    factory, _, _ = build_app

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>NORAG UI</html>", encoding="utf-8")

    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve", encoding="utf-8")

    settings = make_settings(NORAG_FRONTEND_DIST_PATH=str(dist))

    with factory(settings) as client:
        response = client.get("/%2e%2e/secret.txt")

    assert response.text == "<html>NORAG UI</html>"


def test_evaluate_endpoint_returns_a_report(build_app):
    factory, _, services = build_app

    chunk_id = str(uuid4())

    with factory(make_settings()) as client:
        response = client.post(
            "/api/evaluate",
            json={
                "system": "hybrid",
                "k": 3,
                "cases": [
                    {
                        "query": "How does hybrid retrieval work?",
                        "relevant_chunk_ids": [chunk_id],
                    }
                ],
            },
        )

    payload = response.json()

    assert response.status_code == 200
    assert payload["report"]["query_count"] == 1
    assert payload["report"]["k"] == 3
    assert payload["report"]["average_metrics"]["recall_at_k"] == 0.0
    assert services.hybrid_retriever.calls[0]["top_k"] == 3


def test_evaluate_endpoint_rejects_invalid_payload(build_app):
    factory, _, _ = build_app

    with factory(make_settings()) as client:
        response = client.post("/api/evaluate", json={"system": "hybrid", "cases": []})

    assert response.status_code == 422


def test_stats_endpoint_uses_typed_schema(build_app):
    factory, _, _ = build_app

    with factory(make_settings()) as client:
        payload = client.get("/api/stats").json()

    assert set(payload) >= {
        "documents",
        "chunks",
        "dense_chunks",
        "sparse_chunks",
        "queries",
        "uploads",
        "deletions",
        "errors",
        "environment",
        "ollama_model",
    }


def test_openapi_documents_query_response_as_answer():
    schema = api_main.create_app().openapi()

    query_schema = schema["components"]["schemas"]["QueryResponse"]
    reference = query_schema["properties"]["result"]["$ref"]

    assert reference.endswith("/AnswerResponse")


def test_rate_limit_middleware_rejects_excess_requests(build_app):
    factory, _, _ = build_app

    settings = make_settings(NORAG_RATE_LIMIT_PER_MINUTE=2)

    with factory(settings) as client:
        statuses = [client.get("/api/stats").status_code for _ in range(3)]

    assert statuses[:2] == [200, 200]
    assert statuses[2] == 429


def test_database_schema_version_guard(tmp_path):
    from norag.storage.sqlite import SQLiteDocumentRepository

    database = tmp_path / "norag.sqlite3"

    SQLiteDocumentRepository(database).initialize()

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 99")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="newer"):
        SQLiteDocumentRepository(database).initialize()


def test_settings_read_norad_environment_variables(monkeypatch):
    monkeypatch.setenv("NORAG_ENVIRONMENT", "production")
    monkeypatch.setenv("NORAG_DEBUG", "false")
    monkeypatch.setenv("NORAG_API_PORT", "9000")

    settings = Settings(_env_file=None)

    assert settings.environment == "production"
    assert settings.debug is False
    assert settings.api_port == 9000


def test_settings_parse_cors_origins():
    settings = Settings(
        _env_file=None,
        NORAG_CORS_ALLOW_ORIGINS="https://a.example, https://b.example ,",
    )

    assert settings.allowed_origins == ["https://a.example", "https://b.example"]


def test_settings_are_json_serializable_for_diagnostics():
    settings = Settings(_env_file=None)

    assert json.dumps(settings.model_dump(mode="json"))
    assert settings.default_top_k == 5
    assert settings.ollama_model == "qwen3:1.7b"


def test_document_manager_stats_contract():
    manager = SimpleNamespace(stats=lambda: DocumentManagerStats(0, 0, 0, 0))

    assert manager.stats().documents == 0
