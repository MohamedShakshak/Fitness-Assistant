"""Reranker using cross-encoder for retrieval result refinement."""

from sentence_transformers import CrossEncoder
from src.config import settings


_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker


def rerank(query: str, documents: list[dict], top_k: int = None) -> list[dict]:
    if not documents:
        return []

    top_k = top_k or settings.RERANK_TOP_K
    model = get_reranker()

    pairs = [(query, doc["text"]) for doc in documents]
    scores = model.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
    return sorted_docs[:top_k]