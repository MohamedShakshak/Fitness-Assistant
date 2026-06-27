"""Full RAG evaluation: deterministic retrieval metrics + RAGAS Faithfulness + custom guardrail metrics."""

import asyncio
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.pipeline import query as rag_query
from src.retrieval.retriever import search as retrieval_search
from src.evaluation.metrics import disclaimer_metric, citation_metric, refusal_metric
from scripts.eval_retrieval import (
    extract_exercise_names,
    compute_recall_at_k,
    compute_mrr,
    compute_hit_rate,
)


def load_eval_data() -> list[dict]:
    qa_path = settings.EVAL_DIR / "qa_pairs.json"
    off_topic_path = settings.EVAL_DIR / "off_topic_questions.json"

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    with open(off_topic_path, "r", encoding="utf-8") as f:
        off_topic = json.load(f)

    return qa_pairs + off_topic


async def score_ragas_metric(metric, llm, response: str) -> str:
    try:
        result = await metric.ascore(llm=llm, response=response)
        return str(result)
    except Exception as e:
        return f"error: {e}"


async def score_faithfulness(faithfulness_metric, llm, question, answer, contexts):
    try:
        result = await faithfulness_metric.ascore(
            llm=llm,
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        return float(result) if result is not None else 0.0
    except Exception as e:
        print(f"  Faithfulness error: {e}")
        return 0.0


def run_generation(question: str, rerank: bool = True) -> dict:
    t0 = time.time()
    result = rag_query(question=question, rerank=rerank)
    elapsed = time.time() - t0
    result["latency"] = elapsed
    return result


async def run_evaluation(rerank: bool = True):
    from src.evaluation.ragas_setup import get_ragas_llm
    from ragas.metrics import Faithfulness

    print("Initializing RAGAS judge LLM...")
    judge_llm = get_ragas_llm()
    faithfulness = Faithfulness(llm=judge_llm)

    eval_data = load_eval_data()
    print(f"Loaded {len(eval_data)} evaluation questions")

    all_results = []

    for i, qa in enumerate(eval_data, 1):
        qid = qa["id"]
        category = qa.get("category", "unknown")
        question = qa["question"]
        is_off_topic = qa.get("is_off_topic", False)

        print(f"\n[{i}/{len(eval_data)}] {qid} ({category})")
        print(f"  Q: {question[:80].encode('ascii', 'replace').decode()}...")

        gen_result = run_generation(question, rerank=rerank)
        answer = gen_result["answer"]
        sources = gen_result.get("sources", [])
        context_used = gen_result.get("context_used", False)

        row = {
            "id": qid,
            "category": category,
            "question": question,
            "answer": answer[:500],
            "is_off_topic": is_off_topic,
            "context_used": context_used,
            "num_sources": len(sources),
            "generation_latency_s": round(gen_result["latency"], 3),
        }

        expected_names = qa.get("expected_exercise_names", [])
        if expected_names and not is_off_topic:
            t0 = time.time()
            search_results = retrieval_search(
                query=question, top_k=settings.TOP_K, rerank_results=rerank
            )
            ret_elapsed = time.time() - t0

            retrieved_names_ordered = [
                r.get("metadata", {}).get("name", "") for r in search_results
            ]
            retrieved_names_set = set(n.lower() for n in retrieved_names_ordered if n)

            row["recall_at_k"] = round(compute_recall_at_k(retrieved_names_set, expected_names, settings.TOP_K), 4)
            row["mrr"] = round(compute_mrr(retrieved_names_ordered, expected_names), 4)
            row["hit_rate"] = round(compute_hit_rate(retrieved_names_set, expected_names), 4)

            context_texts = [r.get("text", "") for r in search_results if r.get("text")]
            faith_score = await score_faithfulness(
                faithfulness, judge_llm, question, answer, context_texts or ["No context retrieved."]
            )
            row["faithfulness"] = round(faith_score, 4)

            citation_raw = await score_ragas_metric(citation_metric, judge_llm, answer)
            row["citation_present"] = citation_raw

            disclaimer_raw = await score_ragas_metric(disclaimer_metric, judge_llm, answer)
            row["disclaimer_present"] = disclaimer_raw
        else:
            row["recall_at_k"] = ""
            row["mrr"] = ""
            row["hit_rate"] = ""
            row["faithfulness"] = ""
            row["citation_present"] = ""
            row["disclaimer_present"] = ""

            refusal_raw = await score_ragas_metric(refusal_metric, judge_llm, answer)
            row["off_topic_refusal"] = refusal_raw

        all_results.append(row)

        print(f"  Answer: {answer[:120].encode('ascii', 'replace').decode()}...")
        if expected_names and not is_off_topic:
            print(f"  Recall@{settings.TOP_K}={row['recall_at_k']} MRR={row['mrr']} "
                  f"Hit={row['hit_rate']} Faith={row['faithfulness']}")
            print(f"  Citation={row['citation_present']} Disclaimer={row['disclaimer_present']}")
        else:
            print(f"  Off-topic refusal={row.get('off_topic_refusal', 'N/A')}")

        if i < len(eval_data):
            await asyncio.sleep(2)

    return all_results


def save_results(results: list[dict]):
    results_dir = settings.EXPERIMENTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"eval_{timestamp}.csv"
    json_path = results_dir / f"eval_{timestamp}.json"

    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary_path = results_dir / "summary.md"
    generate_summary_markdown(results, summary_path)

    print(f"\n{'='*60}")
    print(f"RESULTS SAVED")
    print(f"{'='*60}")
    print(f"  CSV:      {csv_path}")
    print(f"  JSON:     {json_path}")
    print(f"  Summary:  {summary_path}")


def generate_summary_markdown(results: list[dict], path: Path):
    fitness_results = [r for r in results if not r["is_off_topic"]]
    off_topic_results = [r for r in results if r["is_off_topic"]]

    lines = ["# Evaluation Summary", ""]

    if fitness_results:
        n = len(fitness_results)
        recall_vals = [r["recall_at_k"] for r in fitness_results if r["recall_at_k"] != ""]
        mrr_vals = [r["mrr"] for r in fitness_results if r["mrr"] != ""]
        hit_vals = [r["hit_rate"] for r in fitness_results if r["hit_rate"] != ""]
        faith_vals = [r["faithfulness"] for r in fitness_results if r["faithfulness"] != ""]
        cite_pass = sum(1 for r in fitness_results if r.get("citation_present") == "pass")
        discl_pass = sum(1 for r in fitness_results if r.get("disclaimer_present") == "pass")

        lines.append("## Fitness Questions (n={})".format(n))
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        if recall_vals:
            lines.append(f"| Recall@{settings.TOP_K} | {sum(recall_vals)/len(recall_vals):.4f} |")
        if mrr_vals:
            lines.append(f"| MRR | {sum(mrr_vals)/len(mrr_vals):.4f} |")
        if hit_vals:
            lines.append(f"| Hit Rate | {sum(hit_vals)/len(hit_vals):.4f} |")
        if faith_vals:
            lines.append(f"| Faithfulness (RAGAS) | {sum(faith_vals)/len(faith_vals):.4f} |")
        if fitness_results:
            lines.append(f"| Citation Rate | {cite_pass/len(fitness_results):.4f} |")
            lines.append(f"| Disclaimer Rate | {discl_pass/len(fitness_results):.4f} |")
        lines.append("")

        lines.append("### By Category")
        lines.append("")
        categories = {}
        for r in fitness_results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        lines.append("| Category | n | Recall@K | MRR | HitRate | Faith | CitRate | DisclRate |")
        lines.append("|----------|---|----------|-----|---------|-------|---------|-----------|")
        for cat, cat_results in sorted(categories.items()):
            cn = len(cat_results)
            cr = [r["recall_at_k"] for r in cat_results if r["recall_at_k"] != ""]
            cm = [r["mrr"] for r in cat_results if r["mrr"] != ""]
            ch = [r["hit_rate"] for r in cat_results if r["hit_rate"] != ""]
            cf = [r["faithfulness"] for r in cat_results if r["faithfulness"] != ""]
            cc = sum(1 for r in cat_results if r.get("citation_present") == "pass")
            cd = sum(1 for r in cat_results if r.get("disclaimer_present") == "pass")

            r_val = f"{sum(cr)/len(cr):.4f}" if cr else "—"
            m_val = f"{sum(cm)/len(cm):.4f}" if cm else "—"
            h_val = f"{sum(ch)/len(ch):.4f}" if ch else "—"
            f_val = f"{sum(cf)/len(cf):.4f}" if cf else "—"
            c_val = f"{cc/cn:.4f}" if cn else "—"
            d_val = f"{cd/cn:.4f}" if cn else "—"

            lines.append(f"| {cat} | {cn} | {r_val} | {m_val} | {h_val} | {f_val} | {c_val} | {d_val} |")
        lines.append("")

    if off_topic_results:
        n = len(off_topic_results)
        refusal_pass = sum(1 for r in off_topic_results if r.get("off_topic_refusal") == "pass")
        lines.append("## Off-Topic Guardrail Questions (n={})".format(n))
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        lines.append(f"| Refusal Rate | {refusal_pass/n:.4f} |")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    args = parser.parse_args()

    rerank = not args.no_rerank
    if not rerank:
        print("NOTE: Running without reranker")

    print("=" * 60)
    print("FULL RAG EVALUATION")
    print("=" * 60)

    results = asyncio.run(run_evaluation(rerank=rerank))
    save_results(results)