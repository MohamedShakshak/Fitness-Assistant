"""Tests for retrieval evaluation metric functions — name normalization and scoring."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_retrieval import (
    _normalize_name,
    _names_match,
    compute_recall_at_k,
    compute_mrr,
    compute_hit_rate,
)


class TestNormalizeName:
    def test_lowercase(self):
        assert _normalize_name("Barbell Curl") == "barbell curl"

    def test_strip_trailing_dash(self):
        assert _normalize_name("Dumbbell bench press-") == "dumbbell bench press"

    def test_strip_trailing_colon(self):
        assert _normalize_name("Plank:") == "plank"

    def test_collapse_whitespace(self):
        assert _normalize_name("Squat   Press") == "squat press"

    def test_empty(self):
        assert _normalize_name("") == ""


class TestNamesMatch:
    def test_exact(self):
        assert _names_match("Barbell Curl", "barbell curl")

    def test_trailing_dash(self):
        assert _names_match("Dumbbell bench press-", "Dumbbell Bench Press")

    def test_no_match(self):
        assert not _names_match("Plank", "Squat")

    def test_empty(self):
        assert not _names_match("", "Squat")


class TestRecallAtK:
    def test_all_matched(self):
        retrieved = {"squat", "deadlift", "press"}
        expected = ["Squat", "Deadlift"]
        assert compute_recall_at_k(retrieved, expected, 5) == 1.0

    def test_partial_match(self):
        retrieved = {"squat", "curl"}
        expected = ["Squat", "Deadlift"]
        assert compute_recall_at_k(retrieved, expected, 5) == 0.5

    def test_no_match(self):
        retrieved = {"curl"}
        expected = ["Squat"]
        assert compute_recall_at_k(retrieved, expected, 5) == 0.0

    def test_empty_expected(self):
        assert compute_recall_at_k({"squat"}, [], 5) == 0.0

    def test_trailing_dash_normalized(self):
        retrieved = {"dumbbell bench press"}
        expected = ["Dumbbell Bench Press-"]
        assert compute_recall_at_k(retrieved, expected, 5) == 1.0


class TestMRR:
    def test_first_position(self):
        assert compute_mrr(["Squat", "Curl"], ["Squat"]) == 1.0

    def test_second_position(self):
        assert compute_mrr(["Curl", "Squat"], ["Squat"]) == 0.5

    def test_no_match(self):
        assert compute_mrr(["Curl", "Press"], ["Squat"]) == 0.0

    def test_trailing_dash_normalized(self):
        assert compute_mrr(["Dumbbell bench press-"], ["Dumbbell Bench Press"]) == 1.0


class TestHitRate:
    def test_hit(self):
        assert compute_hit_rate({"squat", "curl"}, ["Squat"]) == 1.0

    def test_miss(self):
        assert compute_hit_rate({"curl"}, ["Squat"]) == 0.0

    def test_empty_expected(self):
        assert compute_hit_rate({"squat"}, []) == 0.0