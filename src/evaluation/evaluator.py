"""Evaluation orchestrator — runs retrieval + generation + scoring for each question."""

import json
import time

from src.config import settings
from src.pipeline import query as rag_query
from src.retrieval.retriever import search as retrieval_search
from src.evaluation.metrics import disclaimer_metric, citation_metric, refusal_metric
from scripts.eval_retrieval import (
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


def score_ragas_metric(metric, llm, response: str) -> str:
    try:
        result = metric.score(llm=llm, response=response)
        return str(result)
    except Exception as e:
        return f"error: {e}"


def score_faithfulness(faithfulness_metric, llm, question, answer, contexts):
    try:
        from ragas.dataset_schema import SingleTurnSample
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        result = faithfulness_metric.single_turn_score(sample)
        return float(result) if result is not None else 0.0
    except Exception as e:
        print(f"  Faithfulness error: {e}")
        return 0.0


def _run_generation(question: str, rerank: bool = True) -> dict:
    t0 = time.time()
    result = rag_query(question=question, rerank=rerank)
    result["latency"] = time.time() - t0
    return result


def _compute_retrieval_metrics(question: str, expected_names: list[str], rerank: bool) -> dict:
    t0 = time.time()
    search_results = retrieval_search(
        query=question, top_k=settings.TOP_K, rerank_results=rerank
    )
    elapsed = time.time() - t0

    names_ordered = [
        r.get("metadata", {}).get("name", "") for r in search_results
    ]
    names_set = set(n.lower() for n in names_ordered if n)

    return {
        "recall_at_k": round(compute_recall_at_k(names_set, expected_names, settings.TOP_K), 4),
        "mrr": round(compute_mrr(names_ordered, expected_names), 4),
        "hit_rate": round(compute_hit_rate(names_set, expected_names), 4),
        "context_texts": [r.get("text", "") for r in search_results if r.get("text")],
        "latency": round(elapsed, 3),
    }


def _score_fitness_question(question: str, expected_names: list[str], rerank: bool, faithfulness, judge_llm) -> dict:
    gen = _run_generation(question, rerank=rerank)
    ret = _compute_retrieval_metrics(question, expected_names, rerank=rerank)

    faith = score_faithfulness(
        faithfulness, judge_llm, question, gen["answer"],
        ret["context_texts"] or ["No context retrieved."],
    )
    citation = score_ragas_metric(citation_metric, judge_llm, gen["answer"])
    disclaimer = score_ragas_metric(disclaimer_metric, judge_llm, gen["answer"])

    return {
        "answer": gen["answer"],
        "context_used": gen.get("context_used", False),
        "num_sources": len(gen.get("sources", [])),
        "generation_latency_s": round(gen["latency"], 3),
        "recall_at_k": ret["recall_at_k"],
        "mrr": ret["mrr"],
        "hit_rate": ret["hit_rate"],
        "retrieval_latency_s": ret["latency"],
        "faithfulness": round(faith, 4),
        "citation_present": citation,
        "disclaimer_present": disclaimer,
    }


def _score_off_topic_question(question: str, rerank: bool, judge_llm) -> dict:
    gen = _run_generation(question, rerank=rerank)
    refusal = score_ragas_metric(refusal_metric, judge_llm, gen["answer"])

    return {
        "answer": gen["answer"],
        "context_used": gen.get("context_used", False),
        "num_sources": len(gen.get("sources", [])),
        "generation_latency_s": round(gen["latency"], 3),
        "off_topic_refusal": refusal,
    }


def run_evaluation(rerank: bool = True) -> list[dict]:
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

        row = {"id": qid, "category": category, "question": question}
        expected_names = qa.get("expected_exercise_names", [])

        if expected_names and not is_off_topic:
            scores = _score_fitness_question(question, expected_names, rerank, faithfulness, judge_llm)
        else:
            scores = _score_off_topic_question(question, rerank, judge_llm)

        row.update(scores)
        all_results.append(row)

        print(f"  Answer: {row.get('answer', '')[:120].encode('ascii', 'replace').decode()}...")
        if expected_names and not is_off_topic:
            print(f"  Recall@{settings.TOP_K}={row['recall_at_k']} MRR={row['mrr']} Hit={row['hit_rate']} Faith={row['faithfulness']}")
            print(f"  Citation={row['citation_present']} Disclaimer={row['disclaimer_present']}")
        else:
            print(f"  Off-topic refusal={row.get('off_topic_refusal', 'N/A')}")

    return all_results
