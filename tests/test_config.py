"""Tests for configuration defaults and validation."""

import pytest
from src.config import settings


class TestSettings:
    def test_qdrant_defaults(self):
        assert settings.QDRANT_URL == "http://localhost:6333"
        assert settings.QDRANT_COLLECTION == "fitness_exercises"

    def test_embedding_defaults(self):
        assert "bge" in settings.EMBEDDING_MODEL.lower()
        assert settings.EMBEDDING_DIM == 768

    def test_retrieval_defaults(self):
        assert settings.TOP_K == 5
        assert settings.RERANK_TOP_K == 5

    def test_chat_defaults(self):
        assert settings.MAX_HISTORY_TURNS == 5

    def test_llm_defaults(self):
        assert settings.LLM_MAX_RETRIES == 3
        assert settings.LLM_TIMEOUT == 120
        assert settings.LLM_MAX_TOKENS == 2048

    def test_eval_defaults(self):
        assert settings.EVAL_MODEL != ""
        assert settings.EVAL_BATCH_SIZE == 5

    def test_paths_exist(self):
        assert settings.EVAL_DIR.exists()