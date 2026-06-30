"""CLI entry point for full RAG evaluation: retrieval + generation + RAGAS + guardrails."""

"""CLI entry point for full RAG evaluation: retrieval + generation + RAGAS + guardrails."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import run_evaluation
from src.evaluation.report import save_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full RAG evaluation")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    args = parser.parse_args()

    rerank = not args.no_rerank
    if not rerank:
        print("NOTE: Running without reranker")

    print("=" * 60)
    print("FULL RAG EVALUATION")
    print("=" * 60)

    results = run_evaluation(rerank=rerank)
    save_results(results)


if __name__ == "__main__":
    main()
