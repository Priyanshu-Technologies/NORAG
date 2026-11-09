import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from norag.api.document_manager import DocumentManager
from norag.api.metrics import ServiceMetrics
from norag.api.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from norag.api.routes import router
from norag.api.static import mount_frontend
from norag.application import create_application_services
from norag.config import Settings, get_settings
from norag.logging import configure_logging
from norag.storage.filesystem import LocalFileStorage
from norag.storage.sqlite import SQLiteDocumentRepository

logger = logging.getLogger(__name__)


def _directory_status(path: Path) -> str:
    path = Path(path)

    if not path.is_dir():
        return "missing"

    return "ok" if os.access(path, os.W_OK) else "read-only"


async def _check_ollama(settings: Settings) -> bool:
    base_url = settings.ollama_base_url.rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{base_url}/api/tags")

        return response.status_code == 200
    except httpx.HTTPError:
        return False


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = create_application_services(app_settings)

        repository = SQLiteDocumentRepository(app_settings.database_path)

        file_storage = LocalFileStorage(app_settings.file_storage_path)

        document_manager = DocumentManager(
            ingestion_service=services.ingestion_service,
            chunker=services.chunker,
            embedding_service=services.embedding_service,
            dense_index=services.dense_index,
            sparse_index=services.sparse_index,
            repository=repository,
            file_storage=file_storage,
        )

        app.state.services = services
        app.state.document_manager = document_manager
        app.state.file_storage = file_storage
        app.state.metrics = ServiceMetrics()

        stats = document_manager.stats()

        logger.info(
            "NORAG ready with %s document(s), %s chunk(s) and model %s.",
            stats.documents,
            stats.chunks,
            app_settings.ollama_model,
        )

        yield

    application = FastAPI(
        title=app_settings.app_name,
        description="Adaptive Retrieval-Augmented Generation Infrastructure",
        version="1.0.0",
        lifespan=lifespan,
    )

    application.state.settings = app_settings
    application.state.metrics = ServiceMetrics()

    if app_settings.rate_limit_per_minute > 0:
        application.add_middleware(
            RateLimitMiddleware,
            limit_per_minute=app_settings.rate_limit_per_minute,
        )

    if app_settings.allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=app_settings.allowed_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    application.add_middleware(RequestLoggingMiddleware)

    application.include_router(router, prefix=app_settings.api_prefix)

    @application.get("/health")
    async def health():
        return {
            "status": "healthy",
            "environment": app_settings.environment,
        }

    @application.get("/ready")
    async def ready():
        checks = {
            "database": _directory_status(app_settings.database_path.parent),
            "storage": _directory_status(app_settings.file_storage_path),
        }

        if app_settings.ollama_ready_check:
            reachable = await _check_ollama(app_settings)
            checks["ollama"] = "ok" if reachable else "unavailable"
        else:
            checks["ollama"] = "skipped"

        is_ready = all(value in {"ok", "skipped"} for value in checks.values())

        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={
                "status": "ready" if is_ready else "degraded",
                "environment": app_settings.environment,
                "checks": checks,
            },
        )

    dist_path = app_settings.frontend_dist_path or Path("frontend/dist")

    if mount_frontend(application, dist_path):
        logger.info("Serving frontend assets from %s", dist_path)

    else:
        # API-only deployments also accept unprefixed routes. The alias is
        # skipped when the UI is served so page URLs stay reachable.
        application.include_router(router, include_in_schema=False)

        @application.get("/")
        async def root():
            return {
                "name": app_settings.app_name,
                "version": "1.0.0",
                "status": "operational",
            }

    return application


app = create_app()
