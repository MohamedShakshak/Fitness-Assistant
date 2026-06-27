"""Quick end-to-end RAG test."""
import sys
sys.path.insert(0, ".")

from src.pipeline import query

print("Testing full RAG pipeline (OpenRouter/Nemotron)...")
print("-" * 60)

result = query(
    question="What are some good beginner chest exercises I can do with a barbell?",
    llm_provider="openrouter",
    top_k=5,
)

print("Answer:")
answer = result["answer"]
# Remove emojis for terminal compatibility
safe_answer = answer.encode("ascii", errors="replace").decode("ascii")
print(safe_answer[:1500])
print()
print(f"Context used: {result['context_used']}")
print(f"Sources: {len(result['sources'])}")
for src in result["sources"][:5]:
    print(f"  - {src['name']} | {src['body_part']} | {src['equipment']} | {src['level']} | [Source: {src['source_db']}]")