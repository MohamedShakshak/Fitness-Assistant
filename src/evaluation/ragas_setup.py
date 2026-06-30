"""RAGAS LLM setup for evaluation — supports OpenRouter."""

from openai import OpenAI
from ragas.llms import llm_factory

from src.config import settings


def get_ragas_llm():
    client = OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
    )
    return llm_factory(
        model=settings.EVAL_MODEL,
        client=client,
        max_tokens=4096,
    )