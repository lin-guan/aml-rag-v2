from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AML Naive RAG"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"

    api_key: str | None = None
    auth_mode: str = "bearer"

    database_path: Path = Path("data/memory.db")
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_device: str | None = None
    embedding_batch_size: int = Field(default=64, ge=1, le=1024)
    max_concurrent_encodes: int = Field(default=1, ge=1, le=32)

    include_options_in_query: bool = True
    max_top_k: int = Field(default=100, ge=1, le=1000)
    enable_hybrid_retrieval: bool = True
    lexical_candidate_k: int = Field(default=100, ge=1, le=1000)
    dense_weight: float = Field(default=1.0, gt=0)
    lexical_weight: float = Field(default=0.9, gt=0)
    neighborhood_radius: int = Field(default=1, ge=0, le=3)
    context_embedding_radius: int = Field(default=2, ge=0, le=5)
    context_embedding_weight: float = Field(default=0.3, ge=0, le=1)
    neighbor_result_ratio: float = Field(default=0.2, ge=0, le=0.5)
    index_window_enabled: bool = True
    index_window_size: int = Field(default=6, ge=2, le=50)
    index_window_overlap: int = Field(default=2, ge=0, le=49)
    window_retrieval_weight: float = Field(default=0.7, ge=0, le=2)
    code_retrieval_enabled: bool = True
    code_exact_match_weight: float = Field(default=0.08, ge=0, le=1)

    @property
    def index_window_step(self) -> int:
        return self.index_window_size - self.index_window_overlap

    @field_validator("index_window_overlap")
    @classmethod
    def validate_window_overlap(cls, value: int, info) -> int:
        size = info.data.get("index_window_size", 6)
        if value >= size:
            raise ValueError("INDEX_WINDOW_OVERLAP must be smaller than INDEX_WINDOW_SIZE")
        return value

    @field_validator("auth_mode")
    @classmethod
    def validate_auth_mode(cls, value: str) -> str:
        normalized = value.lower().replace("-", "_")
        if normalized not in {"none", "token", "bearer", "x_api_key"}:
            raise ValueError("AUTH_MODE must be none, token, bearer, or x_api_key")
        return normalized

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("database_path", mode="after")
    @classmethod
    def normalize_database_path(cls, value: Path) -> Path:
        return value.expanduser()

    def validate_runtime(self) -> None:
        if self.auth_mode != "none" and not self.api_key:
            raise ValueError("API_KEY is required unless AUTH_MODE=none")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
