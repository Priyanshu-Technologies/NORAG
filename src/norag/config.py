from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for NORAG, overridable through environment variables."""

    app_name: str = Field(default="NORAG", validation_alias="NORAG_APP_NAME")
    environment: str = Field(
        default="development",
        validation_alias="NORAG_ENVIRONMENT",
    )
    debug: bool = Field(default=True, validation_alias="NORAG_DEBUG")
    log_level: str = Field(default="INFO", validation_alias="NORAG_LOG_LEVEL")

    api_host: str = Field(default="127.0.0.1", validation_alias="NORAG_API_HOST")
    api_port: int = Field(default=8000, validation_alias="NORAG_API_PORT")
    api_prefix: str = Field(default="/api", validation_alias="NORAG_API_PREFIX")

    api_key: str | None = Field(default=None, validation_alias="NORAG_API_KEY")
    cors_allow_origins: str = Field(
        default="",
        validation_alias="NORAG_CORS_ALLOW_ORIGINS",
    )
    rate_limit_per_minute: int = Field(
        default=0,
        ge=0,
        validation_alias="NORAG_RATE_LIMIT_PER_MINUTE",
    )

    frontend_dist_path: Path | None = Field(
        default=None,
        validation_alias="NORAG_FRONTEND_DIST_PATH",
    )

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

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="NORAG_OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(
        default="qwen3:1.7b",
        validation_alias="NORAG_OLLAMA_MODEL",
    )
    ollama_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        validation_alias="NORAG_OLLAMA_TIMEOUT_SECONDS",
    )
    ollama_think: bool = Field(
        default=False,
        validation_alias="NORAG_OLLAMA_THINK",
    )
    ollama_ready_check: bool = Field(
        default=True,
        validation_alias="NORAG_OLLAMA_READY_CHECK",
    )

    chunk_target_tokens: int = Field(
        default=400,
        gt=0,
        validation_alias="NORAG_CHUNK_TARGET_TOKENS",
    )
    chunk_max_tokens: int = Field(
        default=500,
        gt=0,
        validation_alias="NORAG_CHUNK_MAX_TOKENS",
    )
    chunk_overlap_tokens: int = Field(
        default=50,
        ge=0,
        validation_alias="NORAG_CHUNK_OVERLAP_TOKENS",
    )

    context_max_tokens: int = Field(
        default=3000,
        ge=1,
        validation_alias="NORAG_CONTEXT_MAX_TOKENS",
    )
    prompt_max_input_tokens: int = Field(
        default=4096,
        ge=1,
        validation_alias="NORAG_PROMPT_MAX_INPUT_TOKENS",
    )
    prompt_max_output_tokens: int = Field(
        default=768,
        ge=1,
        validation_alias="NORAG_PROMPT_MAX_OUTPUT_TOKENS",
    )
    generation_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        validation_alias="NORAG_GENERATION_TEMPERATURE",
    )

    default_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        validation_alias="NORAG_DEFAULT_TOP_K",
    )
    max_top_k: int = Field(
        default=50,
        ge=1,
        le=200,
        validation_alias="NORAG_MAX_TOP_K",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def allowed_origins(self) -> list[str]:
        """Return the configured CORS origins as a normalized list."""

        return [
            origin.strip()
            for origin in self.cors_allow_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
