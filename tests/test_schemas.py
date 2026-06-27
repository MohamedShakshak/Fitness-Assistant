"""Tests for API Pydantic schemas — validation and defaults."""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    ChatMessage,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResult,
    SourceInfo,
    FiltersResponse,
)


class TestChatMessage:
    def test_valid(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestQueryRequest:
    def test_minimal(self):
        req = QueryRequest(query="What exercises target the chest?")
        assert req.query == "What exercises target the chest?"
        assert req.chat_history is None
        assert req.filters is None
        assert req.llm_provider is None
        assert req.top_k is None

    def test_with_all_fields(self):
        history = [ChatMessage(role="user", content="hi")]
        req = QueryRequest(
            query="test",
            chat_history=history,
            filters={"level": "beginner"},
            llm_provider="openrouter",
            top_k=10,
        )
        assert req.chat_history[0].role == "user"
        assert req.filters["level"] == "beginner"
        assert req.llm_provider == "openrouter"
        assert req.top_k == 10


class TestSourceInfo:
    def test_defaults(self):
        src = SourceInfo(name="Squat")
        assert src.name == "Squat"
        assert src.category == ""
        assert src.score == 0
        assert src.rerank_score is None

    def test_full(self):
        src = SourceInfo(
            name="Bench Press",
            category="strength",
            body_part="chest",
            equipment="barbell",
            level="intermediate",
            score=0.95,
            rerank_score=0.88,
            source_db="wrkout",
        )
        assert src.score == 0.95
        assert src.rerank_score == 0.88


class TestQueryResponse:
    def test_construct(self):
        resp = QueryResponse(
            answer="Here's what I found",
            sources=[SourceInfo(name="Squat")],
            context_used=True,
        )
        assert resp.answer == "Here's what I found"
        assert len(resp.sources) == 1
        assert resp.context_used is True


class TestSearchRequest:
    def test_minimal(self):
        req = SearchRequest(query="bicep")
        assert req.query == "bicep"
        assert req.filters is None

    def test_with_filters(self):
        req = SearchRequest(query="test", filters={"equipment": "dumbbell"})
        assert req.filters["equipment"] == "dumbbell"


class TestSearchResult:
    def test_construct(self):
        sr = SearchResult(id=1, text="text", score=0.9, metadata={"name": "Squat"})
        assert sr.id == 1
        assert sr.score == 0.9


class TestFiltersResponse:
    def test_construct(self):
        fr = FiltersResponse(
            body_part=["chest", "legs"],
            equipment=["barbell", "dumbbell"],
            level=["beginner", "intermediate"],
            category=["strength"],
        )
        assert "chest" in fr.body_part
        assert "barbell" in fr.equipment