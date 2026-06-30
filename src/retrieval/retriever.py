"""LangChain-based retriever combining hybrid search with reranking."""

from src.config import settings
from src.embedding.embedder import get_embedder
from src.embedding.reranker import rerank
from src.vectorstore.qdrant_store import hybrid_search
from src.retrieval.query_filters import extract_filters
from src.retrieval.query_expansion import expand_query


def _merge_results(results_list: list[list[dict]], top_k: int) -> list[dict]:
    seen = {}
    for results in results_list:
        for r in results:
            name = r.get("metadata", {}).get("name", "")
            if not name:
                continue
            score = r.get("score", 0)
            if name not in seen or score > seen[name].get("score", 0):
                seen[name] = r
    merged = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)
    return merged[:top_k]


def search(
    query: str,
    top_k: int = None,
    rerank_results: bool = True,
    filters: dict = None,
    expand: bool = False,
    llm=None,
) -> list[dict]:
    top_k = top_k or settings.TOP_K
    embedder = get_embedder()

    if not filters:
        parsed = extract_filters(query)
        if parsed:
            filters = parsed

    if expand:
        variants = expand_query(query, llm=llm)
    else:
        variants = [query]

    fetch_k = top_k * 3
    if rerank_results:
        fetch_k = top_k * 7
    elif expand:
        fetch_k = top_k * 2

    all_results = []
    for q in variants:
        qv = embedder.embed_query(q)
        results = hybrid_search(query=q, dense_vector=qv, top_k=fetch_k, filters=filters)
        all_results.append(results)

    if expand:
        merged = _merge_results(all_results, top_k * 3)
    else:
        merged = all_results[0]

    if rerank_results and merged and not filters:
        merged = rerank(query, merged, top_k=top_k)

    return merged[:top_k]