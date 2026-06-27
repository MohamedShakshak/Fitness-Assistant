"""Deterministic retrieval evaluation: Recall@K, MRR, Hit Rate.

No LLM calls — compares retrieved exercise names against expected names
from the Q&A pairs dataset.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.retrieval.retriever import search


def load_eval_data() -> list[dict]:
    eval_path = settings.EVAL_DIR / "qa_pairs.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_name(name: str) -> str:
    import re
    n = name.lower().strip()
    n = re.sub(r"[-:/]+$", "", n).strip()
    n = re.sub(r"\s+", " ", n)
    return n


def extract_exercise_names(results: list[dict]) -> set[str]:
    names = set()
    for r in results:
        meta = r.get("metadata", {})
        name = meta.get("name", "")
        if name:
            names.add(_normalize_name(name))
    return names


def _names_match(retrieved: str, expected: str) -> bool:
    return _normalize_name(retrieved) == _normalize_name(expected)


def compute_recall_at_k(retrieved_names: set[str], expected_names: list[str], k: int) -> float:
    if not expected_names:
        return 0.0
    top_k = set(list(retrieved_names)[:k])
    hits = sum(1 for name in expected_names if any(_names_match(name, tn) for tn in top_k))
    return hits / len(expected_names)


def compute_mrr(retrieved_names: list[str], expected_names: list[str]) -> float:
    for rank, name in enumerate(retrieved_names, 1):
        if any(_names_match(name, exp) for exp in expected_names):
            return 1.0 / rank
    return 0.0


def compute_hit_rate(retrieved_names: set[str], expected_names: list[str]) -> float:
    if not expected_names:
        return 0.0
    return 1.0 if any(any(_names_match(rn, exp) for rn in retrieved_names) for exp in expected_names) else 0.0


def evaluate_retrieval(rerank: bool = True) -> dict:
    qa_pairs = load_eval_data()
    results = []

    for qa in qa_pairs:
        question = qa["question"]
        expected = qa.get("expected_exercise_names", [])

        t0 = time.time()
        search_results = search(query=question, top_k=settings.TOP_K, rerank_results=rerank)
        elapsed = time.time() - t0

        retrieved_names_ordered = [
            r.get("metadata", {}).get("name", "") for r in search_results
        ]
        retrieved_names_set = set(n.lower() for n in retrieved_names_ordered if n)

        recall = compute_recall_at_k(retrieved_names_set, expected, settings.TOP_K)
        mrr = compute_mrr(retrieved_names_ordered, expected)
        hit = compute_hit_rate(retrieved_names_set, expected)

        results.append({
            "id": qa["id"],
            "category": qa["category"],
            "question": question,
            "expected_exercise_names": expected,
            "retrieved_exercise_names": retrieved_names_ordered,
            "recall_at_k": round(recall, 4),
            "mrr": round(mrr, 4),
            "hit_rate": round(hit, 4),
            "latency_seconds": round(elapsed, 3),
        })

    num_questions = len(results)
    avg_recall = sum(r["recall_at_k"] for r in results) / num_questions if num_questions else 0
    avg_mrr = sum(r["mrr"] for r in results) / num_questions if num_questions else 0
    avg_hit = sum(r["hit_rate"] for r in results) / num_questions if num_questions else 0
    avg_latency = sum(r["latency_seconds"] for r in results) / num_questions if num_questions else 0

    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"recall": [], "mrr": [], "hit_rate": []}
        by_category[cat]["recall"].append(r["recall_at_k"])
        by_category[cat]["mrr"].append(r["mrr"])
        by_category[cat]["hit_rate"].append(r["hit_rate"])

    category_summary = {}
    for cat, scores in by_category.items():
        n = len(scores["recall"])
        category_summary[cat] = {
            "recall_at_k": round(sum(scores["recall"]) / n, 4),
            "mrr": round(sum(scores["mrr"]) / n, 4),
            "hit_rate": round(sum(scores["hit_rate"]) / n, 4),
            "count": n,
        }

    summary = {
        "overall": {
            "recall_at_k": round(avg_recall, 4),
            "mrr": round(avg_mrr, 4),
            "hit_rate": round(avg_hit, 4),
            "avg_latency_seconds": round(avg_latency, 3),
            "num_questions": num_questions,
        },
        "by_category": category_summary,
        "per_question": results,
    }

    return summary


def print_summary(summary: dict):
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)

    overall = summary["overall"]
    print(f"\nOverall (n={overall['num_questions']}):")
    print(f"  Recall@{settings.TOP_K}:  {overall['recall_at_k']:.4f}")
    print(f"  MRR:          {overall['mrr']:.4f}")
    print(f"  Hit Rate:     {overall['hit_rate']:.4f}")
    print(f"  Avg Latency:  {overall['avg_latency_seconds']:.3f}s")

    print(f"\nBy Category:")
    print(f"  {'Category':<20} {'Recall@K':>10} {'MRR':>10} {'HitRate':>10} {'N':>5}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*5}")
    for cat, scores in summary["by_category"].items():
        print(f"  {cat:<20} {scores['recall_at_k']:>10.4f} {scores['mrr']:>10.4f} "
              f"{scores['hit_rate']:>10.4f} {scores['count']:>5}")

    print(f"\nPer-Question:")
    print(f"  {'ID':<8} {'Category':<20} {'Recall':>8} {'MRR':>8} {'HitRate':>8}")
    print(f"  {'-'*8} {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for r in summary["per_question"]:
        print(f"  {r['id']:<8} {r['category']:<20} {r['recall_at_k']:>8.4f} "
              f"{r['mrr']:>8.4f} {r['hit_rate']:>8.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    args = parser.parse_args()

    rerank = not args.no_rerank
    if not rerank:
        print("NOTE: Running without reranker (model download issue)")
    summary = evaluate_retrieval(rerank=rerank)
    print_summary(summary)

    results_dir = settings.EXPERIMENTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "retrieval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")