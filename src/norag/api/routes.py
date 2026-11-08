import tempfile
from pathlib import Path
from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from norag.api.document_manager import DocumentManager
from norag.api.schemas import (
    DeleteDocumentResponse,
    DocumentSummary,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
)
from norag.application import ApplicationServices
from norag.storage.filesystem import LocalFileStorage
from norag.api.document_manager import (
    DocumentManager,
    DuplicateDocumentError,
)

router = APIRouter()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_READ_SIZE = 1024 * 1024

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".pdf"}
TEXT_EXTENSIONS = {".txt", ".md", ".html"}


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


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File()],
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

            result = manager.upload(destination)
            document = result.document

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
async def list_documents(request: Request):
    manager = get_document_manager(request)

    return [
        DocumentSummary(
            document_id=document.id,
            file_name=document.metadata.file_name,
            title=document.metadata.title,
            content_hash=document.content_hash,
            created_at=document.created_at,
        )
        for document in manager.list_documents()
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

    deleted = manager.delete(document_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return DeleteDocumentResponse(
        document_id=document_id,
        deleted=True,
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: Request, payload: QueryRequest):
    services = get_application_services(request)

    try:
        answer = services.answering_service.answer(
            query=payload.query,
            top_k=payload.top_k,
        )

        return QueryResponse(result=jsonable_encoder(answer))

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Answer generation failed: {exc}",
        ) from exc
