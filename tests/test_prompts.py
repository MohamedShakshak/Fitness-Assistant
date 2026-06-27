"""Tests for prompt templates and format_context."""

from langchain_core.messages import HumanMessage, AIMessage

from src.generation.prompts import SYSTEM_PROMPT, RAG_PROMPT, NO_CONTEXT_PROMPT, format_context


class TestSystemPrompt:
    def test_medical_disclaimer_instruction(self):
        assert "medical disclaimer" in SYSTEM_PROMPT.lower()

    def test_fitness_only_guardrail(self):
        assert "fitness" in SYSTEM_PROMPT.lower()

    def test_citation_instruction(self):
        assert "cite" in SYSTEM_PROMPT.lower()
        assert "[Source:" in SYSTEM_PROMPT


class TestFormatContext:
    def test_empty_results(self):
        assert format_context([]) == ""

    def test_single_result(self):
        results = [{
            "metadata": {"name": "Squat", "source": "wrkout"},
            "text": "Exercise: Squat\n\nDescription here",
        }]
        text = format_context(results)
        assert "[1]" in text
        assert "Squat" in text
        assert "Source: wrkout" in text

    def test_multiple_results(self):
        results = [
            {"metadata": {"name": "Exercise A", "source": "wrkout"}, "text": "A"},
            {"metadata": {"name": "Exercise B", "source": "kaggle"}, "text": "B"},
        ]
        text = format_context(results)
        assert "[1]" in text
        assert "[2]" in text
        assert "Exercise A" in text
        assert "Exercise B" in text

    def test_source_list_serialized(self):
        results = [{
            "metadata": {"name": "Test", "source": ["wrkout", "kaggle"]},
            "text": "Text",
        }]
        text = format_context(results)
        assert "wrkout, kaggle" in text

    def test_missing_fields(self):
        results = [{}]
        text = format_context(results)
        assert "[1]" in text