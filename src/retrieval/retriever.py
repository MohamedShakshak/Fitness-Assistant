"""LangChain-based retriever combining hybrid search with reranking."""

from src.config import settings
from src.embedding.embedder import get_embedder
from src.embedding.reranker import rerank
from src.vectorstore.qdrant_store import hybrid_search


def search(
    query: str,
    top_k: int = None,
    rerank_results: bool = True,
    filters: dict = None,
) -> list[dict]:
    top_k = top_k or settings.TOP_K

    embedder = get_embedder()
    query_vector = embedder.embed_query(query)

    fetch_k = top_k * 3 if rerank_results else top_k

    results = hybrid_search(
        query=query,
        dense_vector=query_vector,
        top_k=fetch_k,
        filters=filters,
    )

    if rerank_results and results:
        results = rerank(query, results, top_k=top_k)

    return results[:top_k]