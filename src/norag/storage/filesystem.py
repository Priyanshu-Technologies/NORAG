import os
import shutil
from pathlib import Path

from norag.ingestion.models import Document


class LocalFileStorage:
    """Store original uploaded documents on the local filesystem."""

    def __init__(self, storage_path: Path | str) -> None:
        self.storage_path = Path(storage_path)

    def initialize(self) -> None:
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def path_for(self, document: Document) -> Path:
        """Return the managed path for a document."""

        file_name = document.metadata.file_name or ""
        suffix = Path(file_name).suffix.lower()

        return self.storage_path / f"{document.id.hex}{suffix}"

    def store(self, source: Path, document: Document) -> Path:
        """Atomically copy an original file into managed storage."""

        if not source.is_file():
            raise FileNotFoundError(f"Source file does not exist: {source}")

        self.initialize()

        destination = self.path_for(document)
        temporary = destination.with_name(f".{destination.name}.uploading")

        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

        return destination

    def delete(self, document: Document) -> bool:
        """Delete a managed original file."""

        path = self.path_for(document)

        if not path.exists():
            return False

        path.unlink()
        return True
