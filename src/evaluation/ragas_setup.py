"""RAGAS LLM setup for evaluation — supports OpenRouter and Google Gemini."""

import os
from openai import OpenAI
from ragas.llms import llm_factory

from src.config import settings


def get_ragas_llm():
    provider = settings.EVAL_PROVIDER

    if provider == "google":
        os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
        from litellm import OpenAI as LiteLLMClient
        client = LiteLLMClient(
            api_key=settings.GEMINI_API_KEY,
            model=settings.EVAL_MODEL,
        )
        return llm_factory(
            model=settings.EVAL_MODEL,
            client=client,
            max_tokens=4096,
        )

    client = OpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
    )
    return llm_factory(
        model=settings.EVAL_MODEL,
        client=client,
        max_tokens=4096,
    )