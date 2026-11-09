from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def mount_frontend(app: FastAPI, directory: Path | str) -> bool:
    """Serve a built single-page frontend with history fallback.

    Returns ``True`` when a valid build directory was mounted. Unknown paths
    return ``index.html`` so client-side routing works, while API routes are
    matched first because they are registered before this function runs.
    """

    root = Path(directory).expanduser()

    if not root.is_dir():
        return False

    root = root.resolve()
    index = root / "index.html"

    if not index.is_file():
        return False

    assets = root / "assets"

    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def frontend(full_path: str) -> FileResponse:
        candidate = (root / full_path).resolve()

        if (candidate == root or root in candidate.parents) and candidate.is_file():
            return FileResponse(candidate)

        return FileResponse(index)

    return True
