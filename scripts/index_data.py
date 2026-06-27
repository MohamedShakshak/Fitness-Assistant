"""One-command indexing script: load -> chunk -> embed -> upsert to Qdrant."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.chunking.chunker import load_and_chunk, save_chunks
from src.embedding.embedder import get_embedder
from src.vectorstore.qdrant_store import create_collection, upsert_chunks, get_point_count


def main():
    print("=" * 60)
    print("FitAssist - Data Indexing Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading and chunking data...")
    exercise_chunks = load_and_chunk()
    print(f"  Exercise chunks: {len(exercise_chunks)}")

    print("\n[2/4] Saving chunks to disk...")
    ex_path = save_chunks(exercise_chunks, settings.CHUNKS_DIR / "exercises", "exercise_chunks.json")
    print(f"  Exercise chunks saved to {ex_path}")

    print("\n[3/4] Generating embeddings...")
    embedder = get_embedder()
    texts = [doc.page_content for doc in exercise_chunks]
    embeddings = embedder.embed_documents(texts)
    print(f"  Generated {len(embeddings)} embeddings")

    print("\n[4/4] Upserting to Qdrant...")
    print(f"  Connecting to Qdrant at {settings.QDRANT_URL}...")
    print(f"  Creating collection '{settings.QDRANT_COLLECTION}'...")
    create_collection(dim=settings.EMBEDDING_DIM)

    batch_size = 100
    for i in range(0, len(exercise_chunks), batch_size):
        batch_chunks = exercise_chunks[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        upsert_chunks(batch_chunks, batch_embeddings, start_id=i)
        print(f"  Upserted {min(i + batch_size, len(exercise_chunks))}/{len(exercise_chunks)} chunks")

    count = get_point_count()
    print(f"\n{'=' * 60}")
    print(f"Indexing complete! {count} points in collection '{settings.QDRANT_COLLECTION}'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()