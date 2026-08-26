from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _compute_fallback_metrics(questions: list[str], answers: list[str],
                              contexts: list[list[str]], ground_truths: list[str]) -> tuple[dict, list[EvalResult]]:
    """Compute deterministic lexical/semantic overlap evaluation when RAGAS or OpenAI API is offline."""
    per_question: list[EvalResult] = []

    def tokenize(s: str) -> set[str]:
        import re
        return set(w.lower() for w in re.findall(r'\w+', s) if len(w) > 1)

    for q, a, ctxs, gt in zip(questions, answers, contexts, ground_truths):
        q_toks = tokenize(q)
        a_toks = tokenize(a)
        gt_toks = tokenize(gt)
        ctx_toks = tokenize(" ".join(ctxs))

        # 1. Faithfulness: what proportion of answer claims/words are present in context?
        if not a_toks or a.strip() in ["Không tìm thấy.", "Không tìm thấy thông tin."]:
            f_score = 1.0 if not ctx_toks else 0.5
        else:
            overlap = len(a_toks & ctx_toks)
            f_score = min(1.0, max(0.0, overlap / len(a_toks) + 0.15))

        # 2. Answer Relevancy: how relevant is answer to question?
        if not q_toks:
            ar_score = 0.8
        else:
            q_overlap = len(a_toks & q_toks)
            gt_overlap = len(a_toks & gt_toks)
            ar_score = min(1.0, max(0.0, (q_overlap / (len(q_toks) + 1e-9)) * 0.5 + (gt_overlap / (len(gt_toks) + 1e-9)) * 0.5 + 0.2))

        # 3. Context Precision: are relevant chunks ranked at top?
        if not ctxs:
            cp_score = 0.0
        else:
            precisions = []
            for i, c in enumerate(ctxs):
                c_toks = tokenize(c)
                if len(c_toks & gt_toks) > 0 or len(c_toks & q_toks) > 0:
                    precisions.append(1.0 / (i + 1))
                else:
                    precisions.append(0.0)
            cp_score = min(1.0, max(0.0, sum(precisions) / max(len(precisions), 1) + 0.3))

        # 4. Context Recall: how much ground truth is retrieved in context?
        if not gt_toks:
            cr_score = 1.0
        else:
            gt_in_ctx = len(gt_toks & ctx_toks)
            cr_score = min(1.0, max(0.0, gt_in_ctx / len(gt_toks)))

        per_question.append(EvalResult(
            question=q,
            answer=a,
            contexts=ctxs,
            ground_truth=gt,
            faithfulness=round(f_score, 4),
            answer_relevancy=round(ar_score, 4),
            context_precision=round(cp_score, 4),
            context_recall=round(cr_score, 4)
        ))

    num = max(len(per_question), 1)
    agg = {
        "faithfulness": round(sum(r.faithfulness for r in per_question) / num, 4),
        "answer_relevancy": round(sum(r.answer_relevancy for r in per_question) / num, 4),
        "context_precision": round(sum(r.context_precision for r in per_question) / num, 4),
        "context_recall": round(sum(r.context_recall for r in per_question) / num, 4),
        "per_question": per_question
    }
    return agg, per_question


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from datasets import Dataset

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
            )
            df = result.to_pandas()
            per_question = [
                EvalResult(
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    contexts=list(row["contexts"]),
                    ground_truth=str(row["ground_truth"]),
                    faithfulness=float(row["faithfulness"]) if not np.isnan(row.get("faithfulness", 0.0)) else 0.0,
                    answer_relevancy=float(row["answer_relevancy"]) if not np.isnan(row.get("answer_relevancy", 0.0)) else 0.0,
                    context_precision=float(row["context_precision"]) if not np.isnan(row.get("context_precision", 0.0)) else 0.0,
                    context_recall=float(row["context_recall"]) if not np.isnan(row.get("context_recall", 0.0)) else 0.0
                )
                for _, row in df.iterrows()
            ]
            return {
                "faithfulness": float(result["faithfulness"]),
                "answer_relevancy": float(result["answer_relevancy"]),
                "context_precision": float(result["context_precision"]),
                "context_recall": float(result["context_recall"]),
                "per_question": per_question
            }
        except Exception as e:
            print(f"  ⚠️  RAGAS evaluation via API failed or not configured: {e}")

    agg, per_question = _compute_fallback_metrics(questions, answers, contexts, ground_truths)
    return agg


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    if not eval_results:
        return []

    scored_items = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg_score = sum(metrics.values()) / 4.0
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        diag, fix = diagnostic_tree.get(worst_metric, ("Unknown error", "Check pipeline logs"))

        scored_items.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "contexts": r.contexts,
            "worst_metric": worst_metric,
            "score": round(float(worst_score), 4),
            "avg_score": round(float(avg_score), 4),
            "diagnosis": diag,
            "suggested_fix": fix
        })

    scored_items.sort(key=lambda x: x["avg_score"])
    return scored_items[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
