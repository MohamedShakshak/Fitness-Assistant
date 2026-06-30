"""Report generation for evaluation results — CSV, JSON, and markdown summary."""

import csv
import json
import time
from pathlib import Path

from src.config import settings


def _build_categories_map(results: list[dict]) -> dict:
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    return categories


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
    _generate_summary_markdown(results, summary_path)

    print(f"\n{'='*60}")
    print("RESULTS SAVED")
    print(f"{'='*60}")
    print(f"  CSV:      {csv_path}")
    print(f"  JSON:     {json_path}")
    print(f"  Summary:  {summary_path}")


def _avg(values: list) -> str:
    if not values:
        return "—"
    return f"{sum(values)/len(values):.4f}"


def _generate_summary_markdown(results: list[dict], path: Path):
    fitness = [r for r in results if not r["is_off_topic"]]
    off_topic = [r for r in results if r["is_off_topic"]]

    lines = ["# Evaluation Summary", ""]

    if fitness:
        n = len(fitness)
        recall_vals = [r["recall_at_k"] for r in fitness if r.get("recall_at_k") != ""]
        mrr_vals = [r["mrr"] for r in fitness if r.get("mrr") != ""]
        hit_vals = [r["hit_rate"] for r in fitness if r.get("hit_rate") != ""]
        faith_vals = [r["faithfulness"] for r in fitness if r.get("faithfulness") != ""]
        cite_pass = sum(1 for r in fitness if r.get("citation_present") == "pass")
        discl_pass = sum(1 for r in fitness if r.get("disclaimer_present") == "pass")

        lines.append(f"## Fitness Questions (n={n})")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        lines.append(f"| Recall@{settings.TOP_K} | {_avg(recall_vals)} |")
        lines.append(f"| MRR | {_avg(mrr_vals)} |")
        lines.append(f"| Hit Rate | {_avg(hit_vals)} |")
        lines.append(f"| Faithfulness | {_avg(faith_vals)} |")
        lines.append(f"| Citation Rate | {cite_pass/n:.4f} |")
        lines.append(f"| Disclaimer Rate | {discl_pass/n:.4f} |")
        lines.append("")

        lines.append("### By Category")
        lines.append("")
        lines.append("| Category | n | Recall@K | MRR | HitRate | Faith | CitRate | DisclRate |")
        lines.append("|----------|---|----------|-----|---------|-------|---------|-----------|")

        categories = _build_categories_map(fitness)
        for cat, cat_results in sorted(categories.items()):
            cn = len(cat_results)
            cr = [r["recall_at_k"] for r in cat_results if r.get("recall_at_k") != ""]
            cm = [r["mrr"] for r in cat_results if r.get("mrr") != ""]
            ch = [r["hit_rate"] for r in cat_results if r.get("hit_rate") != ""]
            cf = [r["faithfulness"] for r in cat_results if r.get("faithfulness") != ""]
            cc = sum(1 for r in cat_results if r.get("citation_present") == "pass")
            cd = sum(1 for r in cat_results if r.get("disclaimer_present") == "pass")

            lines.append(f"| {cat} | {cn} | {_avg(cr)} | {_avg(cm)} | {_avg(ch)} | {_avg(cf)} | {cc/cn:.4f} | {cd/cn:.4f} |")
        lines.append("")

    if off_topic:
        n = len(off_topic)
        refusal_pass = sum(1 for r in off_topic if r.get("off_topic_refusal") == "pass")
        lines.append(f"## Off-Topic Guardrail Questions (n={n})")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        lines.append(f"| Refusal Rate | {refusal_pass/n:.4f} |")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
