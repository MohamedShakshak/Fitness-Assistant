"""Debug hybrid search."""
import sys
sys.path.insert(0, ".")

from src.config import settings
from src.embedding.embedder import get_embedder
from src.vectorstore.qdrant_store import get_qdrant_client, get_sparse_model
from qdrant_client.models import Prefetch

client = get_qdrant_client()
embedder = get_embedder()
sparse_model = get_sparse_model()

query = "chest exercises with barbell"
dense_vector = embedder.embed_query(query)
sparse_query = list(sparse_model.embed([query]))[0]

print(f"Query: {query}")
print(f"Dense vector dim: {len(dense_vector)}")
print(f"Sparse query type: {type(sparse_query)}")

try:
    dense_prefetch = Prefetch(
        query=dense_vector,
        using="",
        limit=15,
    )
    sparse_prefetch = Prefetch(
        query=sparse_query.as_object(),
        using="bm25",
        limit=15,
    )

    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        prefetch=[dense_prefetch, sparse_prefetch],
        query=dense_vector,
        using="",
        limit=5,
        with_payload=True,
    )
    print(f"Hybrid results: {len(results.points)}")
    for p in results.points[:3]:
        print(f"  - {p.payload.get('name', '?')} | score: {p.score:.4f}")
except Exception as e:
    print(f"Hybrid search error: {e}")
    import traceback
    traceback.print_exc()

try:
    results2 = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=dense_vector,
        using="",
        limit=5,
        with_payload=True,
    )
    print(f"\nDense-only results: {len(results2.points)}")
    for p in results2.points[:3]:
        print(f"  - {p.payload.get('name', '?')} | score: {p.score:.4f}")
except Exception as e:
    print(f"Dense search error: {e}")