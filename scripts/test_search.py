"""Quick test script for Qdrant search."""
import sys
sys.path.insert(0, ".")

from src.config import settings
from src.embedding.embedder import get_embedder
from src.vectorstore.qdrant_store import get_qdrant_client

client = get_qdrant_client()
embedder = get_embedder()

query = "chest exercises with barbell"
dense_vector = embedder.embed_query(query)

info = client.get_collection(settings.QDRANT_COLLECTION)
print(f"Points in collection: {info.points_count}")
print(f"Query vector dim: {len(dense_vector)}")
print(f"Searching for: {query}")

results = client.query_points(
    collection_name=settings.QDRANT_COLLECTION,
    query=dense_vector,
    using="",
    limit=5,
    with_payload=True,
)

print(f"Results: {len(results.points)}")
for p in results.points[:5]:
    name = p.payload.get("name", "?")
    body = p.payload.get("body_part", "?")
    equip = p.payload.get("equipment", "?")
    print(f"  - {name} | score: {p.score:.4f} | body: {body} | equip: {equip}")