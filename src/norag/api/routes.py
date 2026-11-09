import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from norag.api.document_manager import (
    DocumentManager,
    DuplicateDocumentError,
)
from norag.api.metrics import ServiceMetrics
from norag.api.schemas import (
    DeleteDocumentResponse,
    DocumentStats,
    DocumentSummary,
    DocumentUploadResponse,
    EvaluationRequest,
    EvaluationResponse,
    QueryRequest,
    QueryResponse,
)
from norag.api.security import require_api_key
from norag.application import ApplicationServices
from norag.evaluation.evaluator import RetrievalEvaluator
from norag.evaluation.models import EvaluationCase
from norag.storage.filesystem import LocalFileStorage

router = APIRouter(dependencies=[Depends(require_api_key)])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_READ_SIZE = 1024 * 1024

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".pdf"}
TEXT_EXTENSIONS = {".txt", ".md", ".html"}

SearchFunction = Callable[[str, int], Sequence[UUID]]


def get_document_manager(request: Request) -> DocumentManager:
    manager = getattr(request.app.state, "document_manager", None)

    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Document service is not initialized.",
        )

    return manager


def get_application_services(request: Request) -> ApplicationServices:
    services = getattr(request.app.state, "services", None)

    if services is None:
        raise HTTPException(
            status_code=503,
            detail="Application services are not initialized.",
        )

    return services


def get_file_storage(request: Request) -> LocalFileStorage:
    storage = getattr(request.app.state, "file_storage", None)

    if storage is None:
        raise HTTPException(
            status_code=503,
            detail="File storage is not initialized.",
        )

    return storage


def get_metrics(request: Request) -> ServiceMetrics:
    metrics = getattr(request.app.state, "metrics", None)

    if metrics is None:
        raise HTTPException(
            status_code=503,
            detail="Service metrics are not initialized.",
        )

    return metrics


def validate_filename(filename: str) -> str:
    original_name = filename.replace("\\", "/").split("/")[-1]

    if (
        not original_name
        or original_name in {".", ".."}
        or any(
            ord(character) < 32 or ord(character) == 127 for character in original_name
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    return original_name


def validate_file_content(path: Path, suffix: str) -> None:
    content = path.read_bytes()

    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=415,
                detail="The uploaded file does not have a valid PDF signature.",
            )
        return

    if suffix in TEXT_EXTENSIONS:
        if b"\x00" in content:
            raise HTTPException(
                status_code=415,
                detail="The uploaded file contains invalid binary content.",
            )

        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=415,
                detail="The uploaded text file must use UTF-8 encoding.",
            ) from exc


def _build_search_function(
    system: str,
    services: ApplicationServices,
) -> SearchFunction:
    """Map an evaluation system name to a concrete retrieval callable."""

    if system == "sparse":

        def search(query: str, k: int) -> Sequence[UUID]:
            results = services.sparse_index.search(query, top_k=k)
            return [result.chunk.id for result in results]

        return search

    if system == "hybrid":

        def search(query: str, k: int) -> Sequence[UUID]:
            results = services.hybrid_retriever.search(query, top_k=k)
            return [result.chunk.id for result in results]

        return search

    if system == "hybrid_rerank":

        def search(query: str, k: int) -> Sequence[UUID]:
            candidate_k = max(k, services.dense_index.count)
            candidates = services.hybrid_retriever.search(
                query=query,
                top_k=candidate_k,
                candidate_k=candidate_k,
            )
            ranked = services.reranking_service.rerank(
                query,
                candidates,
                top_k=k,
            )
            return [item.result.chunk.id for item in ranked]

        return search

    def dense_search(query: str, k: int) -> Sequence[UUID]:
        vector = services.embedding_service.embed_query(query)
        results = services.dense_index.search(vector, top_k=k)
        return [result.chunk.chunk_id for result in results]

    return dense_search


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    replace: Annotated[bool, Query()] = False,
):
    try:
        manager = get_document_manager(request)

        original_name = validate_filename(file.filename or "")
        suffix = Path(original_name).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type. Supported formats: TXT, MD, HTML, PDF.",
            )

        with tempfile.TemporaryDirectory(prefix="norag-upload-") as directory:
            destination = Path(directory) / original_name
            total_bytes = 0

            with destination.open("wb") as output:
                while chunk := await file.read(UPLOAD_READ_SIZE):
                    total_bytes += len(chunk)

                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="File exceeds the 25 MB upload limit.",
                        )

                    output.write(chunk)

            if total_bytes == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty.",
                )

            validate_file_content(destination, suffix)

            result = await run_in_threadpool(
                lambda: manager.upload(destination, replace_existing=replace)
            )
            document = result.document

            get_metrics(request).record_upload()

            return DocumentUploadResponse(
                document_id=document.id,
                file_name=original_name,
                content_hash=document.content_hash,
                chunk_count=result.chunk_count,
            )

    except HTTPException:
        raise
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    request: Request,
    limit: Annotated[int | None, Query(ge=1, le=1000)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    manager = get_document_manager(request)

    documents = manager.list_documents()
    window = documents[offset:] if limit is None else documents[offset : offset + limit]

    return [
        DocumentSummary(
            document_id=document.id,
            file_name=document.metadata.file_name,
            title=document.metadata.title,
            content_hash=document.content_hash,
            created_at=document.created_at,
        )
        for document in window
    ]


@router.get("/documents/{document_id}/file")
async def download_document(request: Request, document_id: UUID):
    manager = get_document_manager(request)
    storage = get_file_storage(request)

    document = manager.get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    path = storage.path_for(document)

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Original document file is unavailable.",
        )

    return FileResponse(
        path=path,
        filename=document.metadata.file_name or path.name,
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
)
async def delete_document(request: Request, document_id: UUID):
    manager = get_document_manager(request)

    deleted = await run_in_threadpool(manager.delete, document_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    get_metrics(request).record_deletion()

    return DeleteDocumentResponse(
        document_id=document_id,
        deleted=True,
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: Request, payload: QueryRequest):
    services = get_application_services(request)
    metrics = get_metrics(request)

    overrides = {"system": payload.system} if payload.system else {}

    try:
        answer = await run_in_threadpool(
            lambda: services.answering_service.answer(
                query=payload.query,
                top_k=payload.top_k,
                **overrides,
            )
        )

    except ValueError as exc:
        metrics.record_error()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, TypeError) as exc:
        metrics.record_error()
        raise HTTPException(
            status_code=502,
            detail=f"Answer generation failed: {exc}",
        ) from exc

    metrics.record_query()

    return QueryResponse(result=answer)


@router.get("/stats", response_model=DocumentStats)
async def get_stats(request: Request):
    manager = get_document_manager(request)

    snapshot = manager.stats()
    counters = get_metrics(request).snapshot()
    settings = request.app.state.settings

    return DocumentStats(
        documents=snapshot.documents,
        chunks=snapshot.chunks,
        dense_chunks=snapshot.dense_chunks,
        sparse_chunks=snapshot.sparse_chunks,
        queries=counters.queries,
        uploads=counters.uploads,
        deletions=counters.deletions,
        errors=counters.errors,
        environment=settings.environment,
        ollama_model=settings.ollama_model,
    )


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_retrieval(request: Request, payload: EvaluationRequest):
    services = get_application_services(request)

    cases = [
        EvaluationCase(
            query=case.query,
            relevant_chunk_ids=set(case.relevant_chunk_ids),
        )
        for case in payload.cases
    ]

    search_function = _build_search_function(payload.system, services)

    def run() -> EvaluationResponse:
        report = RetrievalEvaluator().evaluate(
            cases=cases,
            search_fn=search_function,
            k=payload.k,
        )
        return EvaluationResponse(report=report)

    try:
        return await run_in_threadpool(run)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
