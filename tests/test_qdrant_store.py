"""Tests for Qdrant store metadata serialization/deserialization and filter building."""

import json
import pytest

from src.vectorstore.qdrant_store import (
    _serialize_metadata,
    _deserialize_payload,
    _build_filter,
)


class TestSerializeMetadata:
    def test_list_serialized_to_json(self):
        meta = {"source": ["wrkout", "kaggle"], "name": "Squat"}
        result = _serialize_metadata(meta)
        assert isinstance(result["source"], str)
        assert json.loads(result["source"]) == ["wrkout", "kaggle"]
        assert result["name"] == "Squat"

    def test_empty_list_becomes_empty_string(self):
        meta = {"source": []}
        result = _serialize_metadata(meta)
        assert result["source"] == ""

    def test_none_becomes_empty_string(self):
        meta = {"category": None}
        result = _serialize_metadata(meta)
        assert result["category"] == ""

    def test_scalars_preserved(self):
        meta = {"level": "beginner", "count": 5}
        result = _serialize_metadata(meta)
        assert result["level"] == "beginner"
        assert result["count"] == 5


class TestDeserializePayload:
    def test_json_list_deserialized(self):
        payload = {
            "source": json.dumps(["wrkout", "kaggle"]),
            "name": "Squat",
            "level": "beginner",
        }
        meta = _deserialize_payload(payload)
        assert meta["source"] == ["wrkout", "kaggle"]
        assert meta["name"] == "Squat"
        assert meta["level"] == "beginner"

    def test_text_key_excluded(self):
        payload = {"text": "content", "name": "Test"}
        meta = _deserialize_payload(payload)
        assert "text" not in meta
        assert meta["name"] == "Test"

    def test_empty_string_source(self):
        payload = {"source": ""}
        meta = _deserialize_payload(payload)
        assert meta["source"] == ""

    def test_round_trip(self):
        original = {
            "source": ["wrkout"],
            "primary_muscles": ["chest"],
            "name": "Bench",
            "level": "intermediate",
        }
        serialized = _serialize_metadata(original)
        payload = {"text": "content", **serialized}
        deserialized = _deserialize_payload(payload)
        assert deserialized["source"] == ["wrkout"]
        assert deserialized["primary_muscles"] == ["chest"]
        assert deserialized["name"] == "Bench"
        assert deserialized["level"] == "intermediate"


class TestBuildFilter:
    def test_none_returns_none(self):
        assert _build_filter(None) is None

    def test_empty_dict_returns_none(self):
        assert _build_filter({}) is None

    def test_single_value(self):
        f = _build_filter({"level": "beginner"})
        assert f is not None
        assert len(f.must) == 1

    def test_list_value(self):
        f = _build_filter({"equipment": ["dumbbell", "barbell"]})
        assert f is not None
        assert len(f.must) == 1

    def test_mixed_values(self):
        f = _build_filter({"level": "beginner", "equipment": ["dumbbell"]})
        assert f is not None
        assert len(f.must) == 2

    def test_empty_value_skipped(self):
        f = _build_filter({"level": "", "equipment": "dumbbell"})
        assert f is not None
        assert len(f.must) == 1