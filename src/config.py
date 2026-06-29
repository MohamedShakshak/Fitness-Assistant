"""Pydantic Settings configuration loaded from .env file.

All runtime values (especially secrets) come from .env.
This file defines the schema, types, and defaults.
Pydantic Settings auto-matches .env variable names to these fields.
"""

from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "fitness_exercises"

    # Embedding
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIM: int = 768

    # Reranker
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # LLM - Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3.5:9b"

    # LLM - OpenRouter
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "liquid/lfm-2.5-1.2b-instruct:free"

    # LLM Provider: "ollama" or "openrouter"
    LLM_PROVIDER: str = "openrouter"

    # Retrieval
    TOP_K: int = 5
    RERANK_TOP_K: int = 5

    # Logging
    LOG_LEVEL: str = "INFO"

    # Chat
    MAX_HISTORY_TURNS: int = 5

    # Application
    API_URL: str = "http://localhost:8000"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LLM_MAX_RETRIES: int = 3
    LLM_TIMEOUT: int = 120
    LLM_MAX_TOKENS: int = 2048
    EMBEDDING_DEVICE: str = "cpu"
    SPARSE_MODEL: str = "Qdrant/bm25"

    # Evaluation
    EVAL_MODEL: str = "gemini/gemini-1.5-flash"
    EVAL_BATCH_SIZE: int = 5
    EVAL_PROVIDER: str = "openrouter"

    # Google Gemini (for evaluation, free tier: 1500 req/day)
    GEMINI_API_KEY: str = ""

    # Paths
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    CHUNKS_DIR: Path = PROJECT_ROOT / "data" / "chunks"
    EVAL_DIR: Path = PROJECT_ROOT / "data" / "eval"
    EXPERIMENTS_DIR: Path = PROJECT_ROOT / "evals" / "experiments"

    @model_validator(mode="after")
    def validate_llm_config(self) -> "Settings":
        if self.LLM_PROVIDER == "openrouter" and not self.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is required when LLM_PROVIDER='openrouter'. "
                "Set it in .env or switch LLM_PROVIDER to 'ollama'."
            )
        if self.EVAL_PROVIDER == "google" and not self.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is required when EVAL_PROVIDER='google'. "
                "Set it in .env or switch EVAL_PROVIDER to 'openrouter'."
            )
        return self

    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "env_file_encoding": "utf-8"}


settings = Settings()

import logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)