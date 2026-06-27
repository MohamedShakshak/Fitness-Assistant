"""LangChain LLM clients for Ollama (local) and OpenRouter (cloud)."""

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from src.config import settings


def get_llm(provider: str = None, temperature: float = 0.7) -> ChatOllama | ChatOpenAI:
    provider = provider or settings.LLM_PROVIDER

    if provider == "ollama":
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=temperature,
        )
    elif provider == "openrouter":
        return ChatOpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_MODEL,
            temperature=temperature,
            default_headers={
                "HTTP-Referer": settings.API_URL,
                "X-Title": "FitAssist",
            },
            max_retries=settings.LLM_MAX_RETRIES,
            timeout=settings.LLM_TIMEOUT,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'ollama' or 'openrouter'.")