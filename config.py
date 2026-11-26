"""Configuration loading and validation."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseModel):
    """Application-wide settings."""
    name: str = "Local RAG"
    log_level: str = "INFO"
    debug_content_logging: bool = False

class PathsConfig(BaseModel):
    """Project data paths."""
    books_dir: str = "data/books"
    index_dir: str = "data/index"
    logs_dir: str = "logs"
    security_rules: str = "security_rules.yaml"

class ModelsConfig(BaseModel):
    """Local model settings."""
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b-instruct"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "auto"
    embedding_batch_size: int = 16

class RetrievalConfig(BaseModel):
    """Chunking and retrieval parameters."""
    collection_name: str = "local_books"
    top_k: int = 8
    min_similarity: float = 0.0
    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_chars: int = 50
    max_context_chars: int = 12000
    max_fragment_chars: int = 1800

class SecurityConfig(BaseModel):
    """Resource and request limits."""
    max_file_size_mb: int = 100
    max_query_chars: int = 4000
    requests_per_minute: int = 10
    block_jailbreak_patterns: bool = True
    block_sensitive_output: bool = True

class Settings(BaseSettings):
    """Validated application configuration."""
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")
    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @classmethod
    def load(cls, config_path: str | Path = "config.yaml") -> "Settings":
        """Load YAML settings and apply selected environment overrides."""
        path = Path(config_path)
        data: dict[str, Any] = {}
        if path.exists():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError("config.yaml должен содержать YAML-объект.")
            data = loaded
        result = cls(**data)
        if os.getenv("RAG_LOG_LEVEL"):
            result.app.log_level = os.environ["RAG_LOG_LEVEL"]
        if os.getenv("RAG_DEBUG_CONTENT_LOGGING"):
            result.app.debug_content_logging = os.environ["RAG_DEBUG_CONTENT_LOGGING"].lower() in {"1", "true", "yes"}
        return result

    def resolve(self, configured_path: str) -> Path:
        """Resolve a configured path relative to current working directory."""
        return Path(configured_path).expanduser().resolve()
