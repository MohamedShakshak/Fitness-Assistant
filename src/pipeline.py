"""End-to-end RAG pipeline using LangChain: retrieve -> rerank -> generate."""

from langchain_core.messages import HumanMessage, AIMessage

from src.config import settings
from src.generation.llm_client import get_llm
from src.generation.prompts import RAG_PROMPT, NO_CONTEXT_PROMPT, format_context
from src.retrieval.retriever import search
from src.vectorstore.qdrant_store import get_available_filters


def query(
    question: str,
    chat_history: list[dict] = None,
    filters: dict = None,
    llm_provider: str = None,
    top_k: int = None,
    rerank: bool = True,
) -> dict:
    llm = get_llm(llm_provider)

    results = search(
        query=question,
        top_k=top_k,
        rerank_results=rerank,
        filters=filters,
    )

    context = format_context(results)

    langchain_history = _convert_history(chat_history)

    if not context:
        chain = NO_CONTEXT_PROMPT | llm
        answer = chain.invoke({
            "question": question,
            "chat_history": langchain_history,
        }).content
        return {
            "answer": answer,
            "sources": [],
            "context_used": False,
        }

    chain = RAG_PROMPT | llm
    answer = chain.invoke({
        "question": question,
        "context": context,
        "chat_history": langchain_history,
    }).content

    sources = []
    seen = set()
    for result in results:
        meta = result.get("metadata", {})
        name = meta.get("name", "Unknown")
        if name not in seen:
            source_val = meta.get("source", "")
            if isinstance(source_val, list):
                source_val = ", ".join(source_val)
            sources.append({
                "name": name,
                "category": meta.get("category", ""),
                "body_part": meta.get("body_part", ""),
                "equipment": meta.get("equipment", ""),
                "level": meta.get("level", ""),
                "score": result.get("score", 0),
                "rerank_score": result.get("rerank_score"),
                "source_db": source_val,
            })
            seen.add(name)

    return {
        "answer": answer,
        "sources": sources,
        "context_used": True,
    }


def search_only(query: str, filters: dict = None, top_k: int = None) -> list[dict]:
    return search(query=query, filters=filters, top_k=top_k, rerank_results=True)


def get_filters() -> dict[str, list[str]]:
    return get_available_filters()


def _convert_history(chat_history: list[dict] = None) -> list:
    if not chat_history:
        return []

    max_messages = settings.MAX_HISTORY_TURNS * 2
    recent = chat_history[-max_messages:]

    messages = []
    for msg in recent:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages