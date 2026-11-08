from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NORAG"
    environment: str = "development"
    debug: bool = True

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    verification_strategy: Literal["lexical", "ollama"] = Field(
        default="lexical",
        validation_alias="NORAG_VERIFICATION_STRATEGY",
    )

    database_path: Path = Field(
        default=Path("data/norag.sqlite3"),
        validation_alias="NORAG_DATABASE_PATH",
    )

    file_storage_path: Path = Field(
        default=Path("data/documents"),
        validation_alias="NORAG_FILE_STORAGE_PATH",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
