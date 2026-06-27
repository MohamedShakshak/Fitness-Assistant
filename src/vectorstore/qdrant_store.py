"""Qdrant vector store with hybrid search (dense + BM25 sparse)."""

import json
import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore as LangChainQdrant
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Prefetch,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
)
from fastembed import SparseTextEmbedding

from src.config import settings

logger = logging.getLogger(__name__)

LIST_METADATA_KEYS = {"source", "primary_muscles", "secondary_muscles"}

_qdrant_client = None
_sparse_model = None
_vectorstore = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.QDRANT_URL)
    return _qdrant_client


def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL)
    return _sparse_model


def create_collection(dim: Optional[int] = None):
    dim = dim or settings.EMBEDDING_DIM
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION in existing:
        logger.info("Collection '%s' already exists, recreating", settings.QDRANT_COLLECTION)
        client.delete_collection(settings.QDRANT_COLLECTION)

    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config={"": VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={
            "bm25": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            )
        },
    )


def _serialize_metadata(metadata: dict) -> dict:
    serialized = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            serialized[key] = json.dumps(value) if value else ""
        elif value is not None:
            serialized[key] = value
        else:
            serialized[key] = ""
    return serialized


def _deserialize_payload(payload: dict) -> dict:
    metadata = {}
    for key, value in payload.items():
        if key == "text":
            continue
        if key in LIST_METADATA_KEYS and isinstance(value, str) and value.startswith("["):
            try:
                metadata[key] = json.loads(value)
            except json.JSONDecodeError:
                metadata[key] = value
        else:
            metadata[key] = value
    return metadata


def upsert_chunks(documents: list[Document], embeddings: list[list[float]], start_id: int = 0):
    from qdrant_client.models import PointStruct

    client = get_qdrant_client()
    sparse_model = get_sparse_model()
    sparse_embeddings = list(sparse_model.embed([doc.page_content for doc in documents], batch_size=256))

    points = []
    for i, (doc, dense_vec, sparse_vec) in enumerate(zip(documents, embeddings, sparse_embeddings)):
        metadata = _serialize_metadata(doc.metadata)

        points.append(
            PointStruct(
                id=start_id + i,
                vector={
                    "": dense_vec,
                    "bm25": sparse_vec.as_object(),
                },
                payload={
                    "text": doc.page_content,
                    **metadata,
                },
            )
        )

    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    logger.info("Upserted %d points (starting from id %d)", len(points), start_id)


def get_vectorstore(embedder: HuggingFaceEmbeddings = None) -> LangChainQdrant:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    from src.embedding.embedder import get_embedder
    if embedder is None:
        embedder = get_embedder()

    client = get_qdrant_client()
    _vectorstore = LangChainQdrant(
        client=client,
        collection_name=settings.QDRANT_COLLECTION,
        embeddings=embedder,
    )
    return _vectorstore


def _build_filter(filters: Optional[dict]) -> Optional[Filter]:
    if not filters:
        return None

    conditions = []
    for key, value in filters.items():
        if not value:
            continue
        if isinstance(value, list):
            conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=str(value))))

    return Filter(must=conditions) if conditions else None


def hybrid_search(
    query: str,
    dense_vector: list[float],
    top_k: Optional[int] = None,
    filters: Optional[dict] = None,
) -> list[dict]:
    top_k = top_k or settings.TOP_K
    client = get_qdrant_client()
    sparse_model = get_sparse_model()

    qdrant_filter = _build_filter(filters)

    sparse_query = list(sparse_model.embed([query]))[0]

    dense_prefetch = Prefetch(
        query=dense_vector,
        using="",
        limit=top_k * 3,
        filter=qdrant_filter,
    )
    sparse_prefetch = Prefetch(
        query=sparse_query.as_object(),
        using="bm25",
        limit=top_k * 3,
        filter=qdrant_filter,
    )

    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        prefetch=[dense_prefetch, sparse_prefetch],
        query=dense_vector,
        using="",
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "id": point.id,
            "text": point.payload.get("text", ""),
            "score": point.score,
            "metadata": _deserialize_payload(point.payload),
        }
        for point in results.points
    ]


def get_available_filters() -> dict[str, list[str]]:
    client = get_qdrant_client()
    body_parts = set()
    equipment = set()
    levels = set()
    categories = set()

    scroll_result = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        limit=10000,
        with_payload=True,
    )

    for point in scroll_result[0]:
        p = point.payload
        if p.get("body_part"):
            body_parts.add(p["body_part"])
        if p.get("equipment"):
            equipment.add(p["equipment"])
        if p.get("level"):
            levels.add(p["level"])
        if p.get("category"):
            categories.add(p["category"])

    return {
        "body_part": sorted(body_parts),
        "equipment": sorted(equipment),
        "level": sorted(levels),
        "category": sorted(categories),
    }


def get_point_count() -> int:
    client = get_qdrant_client()
    info = client.get_collection(settings.QDRANT_COLLECTION)
    return info.points_count