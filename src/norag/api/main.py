from contextlib import asynccontextmanager

from fastapi import FastAPI

from norag.api.document_manager import DocumentManager
from norag.api.routes import router
from norag.application import create_application_services
from norag.config import Settings, get_settings
from norag.storage.filesystem import LocalFileStorage
from norag.storage.sqlite import SQLiteDocumentRepository


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

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

        yield

    application = FastAPI(
        title=app_settings.app_name,
        description="Adaptive Retrieval-Augmented Generation Infrastructure",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.include_router(router)

    @application.get("/")
    async def root():
        return {
            "name": app_settings.app_name,
            "version": "0.1.0",
            "status": "operational",
        }

    @application.get("/health")
    async def health():
        return {
            "status": "healthy",
            "environment": app_settings.environment,
        }

    return application


app = create_app()
