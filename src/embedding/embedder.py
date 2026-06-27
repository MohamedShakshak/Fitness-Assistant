"""LangChain-based embedding wrapper using HuggingFace sentence-transformers."""

from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings


_embedder = None


def get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedder