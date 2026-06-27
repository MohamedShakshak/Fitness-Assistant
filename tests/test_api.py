"""Integration tests for FastAPI endpoints using TestClient with mocked pipeline."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_query():
    with patch("src.api.main.query") as m:
        m.return_value = {
            "answer": "The Barbell Squat is a compound exercise [Source: wrkout].",
            "sources": [{
                "name": "Barbell Squat",
                "category": "strength",
                "body_part": "legs",
                "equipment": "barbell",
                "level": "beginner",
                "score": 0.95,
                "rerank_score": 0.88,
                "source_db": "wrkout",
            }],
            "context_used": True,
        }
        yield m


@pytest.fixture
def mock_search_only():
    with patch("src.api.main.search_only") as m:
        m.return_value = [{
            "id": 1,
            "text": "Exercise: Squat",
            "score": 0.9,
            "metadata": {"name": "Squat"},
        }]
        yield m


@pytest.fixture
def mock_get_filters():
    with patch("src.api.main.get_filters") as m:
        m.return_value = {
            "body_part": ["chest", "legs"],
            "equipment": ["barbell", "dumbbell"],
            "level": ["beginner", "intermediate"],
            "category": ["strength"],
        }
        yield m


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestQueryEndpoint:
    def test_valid_query(self, client, mock_query):
        response = client.post("/query", json={"query": "What is a squat?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["context_used"] is True
        assert len(data["sources"]) == 1

    def test_query_with_filters(self, client, mock_query):
        response = client.post("/query", json={
            "query": "beginner exercises",
            "filters": {"level": "beginner"},
        })
        assert response.status_code == 200

    def test_query_with_history(self, client, mock_query):
        response = client.post("/query", json={
            "query": "tell me more",
            "chat_history": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        })
        assert response.status_code == 200

    def test_query_empty_body(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_server_error(self, client):
        with patch("src.api.main.query", side_effect=RuntimeError("oops")):
            response = client.post("/query", json={"query": "test"})
            assert response.status_code == 500

    def test_query_value_error(self, client):
        with patch("src.api.main.query", side_effect=ValueError("bad request")):
            response = client.post("/query", json={"query": "test"})
            assert response.status_code == 400


class TestSearchEndpoint:
    def test_valid_search(self, client, mock_search_only):
        response = client.post("/search", json={"query": "squat"})
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) >= 1

    def test_search_empty_body(self, client):
        response = client.post("/search", json={})
        assert response.status_code == 422


class TestFiltersEndpoint:
    def test_filters(self, client, mock_get_filters):
        response = client.get("/filters")
        assert response.status_code == 200
        data = response.json()
        assert "body_part" in data
        assert "equipment" in data
        assert "level" in data
        assert "chest" in data["body_part"]