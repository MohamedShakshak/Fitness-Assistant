"""Tests for ExerciseChunker — labeled-section format and metadata extraction."""

import json

from src.chunking.chunker import ExerciseChunker


class TestFormatExercise:
    def test_full_record(self):
        chunker = ExerciseChunker()
        record = {
            "name": "Barbell Squat",
            "category": "strength",
            "mechanic": "compound",
            "primary_muscles": ["quadriceps"],
            "body_part": "legs",
            "secondary_muscles": ["glutes", "hamstrings"],
            "equipment": "barbell",
            "level": "beginner",
            "force": "push",
            "description": "A compound leg exercise.",
            "instructions": ["Step 1", "Step 2", "Step 3"],
        }
        text = chunker.format_exercise(record)

        assert "Exercise: Barbell Squat" in text
        assert "Category: Strength (Compound)" in text
        assert "Target: Legs (quadriceps)" in text
        assert "Secondary: glutes, hamstrings" in text
        assert "Equipment: barbell" in text
        assert "Level: Beginner" in text
        assert "Force: Push" in text
        assert "A compound leg exercise." in text
        assert "1. Step 1" in text
        assert "2. Step 2" in text
        assert "3. Step 3" in text

    def test_minimal_record(self):
        chunker = ExerciseChunker()
        record = {"name": "Plank", "description": "Core stability exercise."}
        text = chunker.format_exercise(record)

        assert "Exercise: Plank" in text
        assert "Core stability exercise." in text
        assert "Category" not in text
        assert "Equipment" not in text

    def test_single_instruction(self):
        chunker = ExerciseChunker()
        record = {
            "name": "Push-up",
            "instructions": ["Do a push-up"],
        }
        text = chunker.format_exercise(record)
        assert "How to perform: Do a push-up" in text

    def test_no_content_omitted(self):
        chunker = ExerciseChunker()
        record = {"name": "Empty Exercise"}
        text = chunker.format_exercise(record)
        assert text.strip() == "Exercise: Empty Exercise"

    def test_string_muscles_not_list(self):
        chunker = ExerciseChunker()
        record = {
            "name": "Test",
            "primary_muscles": "chest",
            "secondary_muscles": "triceps",
        }
        text = chunker.format_exercise(record)
        assert "chest" in text
        assert "triceps" in text


class TestChunk:
    def test_skips_empty_exercises(self):
        chunker = ExerciseChunker()
        exercises = [
            {"name": "Real", "description": "Has content"},
            {"name": "Empty", "category": "strength"},
        ]
        docs = chunker.chunk(exercises)
        assert len(docs) == 1
        assert docs[0].metadata["name"] == "Real"

    def test_metadata_fields(self):
        chunker = ExerciseChunker()
        record = {
            "name": "Test Exercise",
            "category": "strength",
            "body_part": "chest",
            "equipment": "dumbbell",
            "level": "intermediate",
            "force": "push",
            "mechanic": "compound",
            "primary_muscles": ["chest"],
            "secondary_muscles": ["triceps"],
            "source": ["wrkout", "kaggle"],
            "description": "Test description",
        }
        docs = chunker.chunk([record])
        meta = docs[0].metadata

        assert meta["name"] == "Test Exercise"
        assert meta["category"] == "strength"
        assert meta["body_part"] == "chest"
        assert meta["equipment"] == "dumbbell"
        assert meta["level"] == "intermediate"
        assert meta["source"] == ["wrkout", "kaggle"]
        assert meta["has_description"] is True
        assert meta["has_instructions"] is False
        assert meta["chunk_type"] == "exercise"

    def test_chunk_id_unique(self):
        chunker = ExerciseChunker()
        exercises = [{"name": "Squat", "description": "x"}] * 3
        docs = chunker.chunk(exercises)
        ids = [d.id for d in docs]
        assert len(set(ids)) == 3

    def test_returns_documents(self):
        chunker = ExerciseChunker()
        exercises = [{"name": "Plank", "description": "Core exercise"}]
        docs = chunker.chunk(exercises)
        assert len(docs) == 1