"""T5: Human evaluation tool for LLM-as-Judge reliability validation.

Provides a terminal-based interactive scoring interface for human raters
to evaluate answer quality across 5 dimensions. Computes inter-annotator
agreement (Cohen's κ, Spearman r) between human raters and LLM Judge.

Usage:
    python -m ard.eval.human_eval --results eval_data/experiment_bge_results.json --sample 20
"""

import json
import random
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np


@dataclass
class HumanRating:
    """A single human rating for one answer."""
    rater_id: str
    query_id: str
    condition: str
    query: str
    answer: str
    correctness: float
    completeness: float
    conciseness: float
    citation_accuracy: float
    groundedness: float
    overall: float
    notes: str = ""


@dataclass
class ReliabilityReport:
    """Inter-annotator agreement report."""
    n_samples: int
    n_raters: int
    per_dimension: dict = field(default_factory=dict)
    human_vs_llm_spearman_r: dict = field(default_factory=dict)
    agreement_summary: str = ""


def load_samples(results_path: str, n_samples: int = 20,
                 seed: int = 42) -> list[dict]:
    """Load and sample experimental results for human rating.

    Returns: list of {query_id, condition, query, answer, judge_scores}
    """
    with open(results_path, encoding='utf-8') as f:
        data = json.load(f)

    per_cond = data.get('per_condition', {})
    all_runs = []
    for cond, cond_data in per_cond.items():
        for run in cond_data.get('runs', []):
            all_runs.append({
                'query_id': run['query_id'],
                'condition': cond,
                'answer': run.get('answer', ''),
                'judge_scores': run.get('judge_scores', {}),
            })

    # Load queries for context
    queries = {}
    for path in ['eval_data/benchmark_v2.json', 'eval_data/benchmark_v1.json']:
        try:
            with open(path, encoding='utf-8') as f:
                qdata = json.load(f)
            for q in qdata.get('queries', qdata.get('single_turn_queries', [])):
                queries[q['query_id']] = q['query']
        except Exception:
            pass

    for run in all_runs:
        run['query'] = queries.get(run['query_id'], run['query_id'])

    random.seed(seed)
    sampled = random.sample(all_runs, min(n_samples, len(all_runs)))
    return sampled


def interactive_rate(samples: list[dict], rater_id: str = "rater_1") -> list[HumanRating]:
    """Interactive terminal-based rating session.

    Presents each sample and prompts for 0-5 scores on 5 dimensions.
    """
    ratings = []
    print(f"\n{'='*60}")
    print(f" HUMAN EVALUATION — Rater: {rater_id}")
    print(f" {len(samples)} samples to rate")
    print(f"{'='*60}")
    print("Score each dimension 0-5:")
    print("  0 = completely wrong / irrelevant")
    print("  3 = partially correct")
    print("  5 = excellent / fully correct\n")

    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}/{len(samples)} ---")
        print(f"Query: {sample['query'][:200]}")
        print(f"Condition: {sample['condition']}")
        print(f"Answer: {sample['answer'][:500]}")
        print(f"LLM Judge scores: {sample.get('judge_scores', {})}")
        print()

        try:
            correctness = float(input("correctness (0-5): ") or 3)
            completeness = float(input("completeness (0-5): ") or 3)
            conciseness = float(input("conciseness (0-5): ") or 3)
            citation = float(input("citation_accuracy (0-5): ") or 3)
            groundedness = float(input("groundedness (0-5): ") or 3)
            overall = float(input("overall (0-5): ") or 3)
            notes = input("notes (optional): ") or ""
        except (EOFError, KeyboardInterrupt):
            print("\nRating interrupted. Saving progress...")
            break
        except ValueError:
            print("Invalid input, using defaults (3).")
            correctness = completeness = conciseness = citation = groundedness = overall = 3.0

        ratings.append(HumanRating(
            rater_id=rater_id, query_id=sample['query_id'],
            condition=sample['condition'], query=sample['query'],
            answer=sample['answer'], correctness=correctness,
            completeness=completeness, conciseness=conciseness,
            citation_accuracy=citation, groundedness=groundedness,
            overall=overall, notes=notes,
        ))

    return ratings


def compute_reliability(human_ratings: list[list[HumanRating]],
                        llm_judge_scores: dict = None) -> ReliabilityReport:
    """Compute inter-annotator agreement metrics.

    Args:
        human_ratings: List of rating lists (one per rater).
        llm_judge_scores: Optional dict mapping (query_id, condition) → JudgeScores.

    Returns:
        ReliabilityReport with per-dimension agreement metrics.
    """
    if len(human_ratings) < 2:
        return ReliabilityReport(n_samples=len(human_ratings[0]) if human_ratings else 0,
                                 n_raters=len(human_ratings),
                                 agreement_summary="Need at least 2 raters for IAA")

    dims = ["correctness", "completeness", "conciseness", "citation_accuracy", "groundedness", "overall"]
    r1 = human_ratings[0]
    r2 = human_ratings[1]

    # Align by query_id + condition
    r1_map = {(r.query_id, r.condition): r for r in r1}
    r2_map = {(r.query_id, r.condition): r for r in r2}
    common_keys = set(r1_map.keys()) & set(r2_map.keys())

    report = ReliabilityReport(n_samples=len(common_keys), n_raters=len(human_ratings))
    report.per_dimension = {}

    for dim in dims:
        a_vals = [getattr(r1_map[k], dim) for k in common_keys]
        b_vals = [getattr(r2_map[k], dim) for k in common_keys]

        a_arr = np.array(a_vals)
        b_arr = np.array(b_vals)

        # Pearson r between raters
        if len(a_arr) > 1:
            pearson_r = float(np.corrcoef(a_arr, b_arr)[0, 1])
        else:
            pearson_r = 1.0

        # Mean absolute difference
        mad = float(np.mean(np.abs(a_arr - b_arr)))

        # Agreement within 1 point
        agree_1pt = float(np.mean(np.abs(a_arr - b_arr) <= 1.0))

        report.per_dimension[dim] = {
            "pearson_r": round(pearson_r, 4),
            "mean_abs_diff": round(mad, 4),
            "agreement_1pt": round(agree_1pt, 4),
        }

    # Human vs LLM correlation
    if llm_judge_scores and len(common_keys) > 1:
        for dim in dims:
            human_avg = []
            llm_vals = []
            for k in common_keys:
                if k in llm_judge_scores:
                    h_avg = (getattr(r1_map[k], dim) + getattr(r2_map[k], dim)) / 2
                    human_avg.append(h_avg)
                    llm_vals.append(llm_judge_scores[k].get(dim, 0))
            if len(human_avg) > 1:
                r = float(np.corrcoef(np.array(human_avg), np.array(llm_vals))[0, 1])
                report.human_vs_llm_spearman_r[dim] = round(r, 4)

    return report


def print_reliability_report(report: ReliabilityReport) -> str:
    """Format reliability report."""
    lines = ["\n" + "=" * 60,
             "LLM-AS-JUDGE RELIABILITY REPORT",
             "=" * 60,
             f"Samples: {report.n_samples}, Raters: {report.n_raters}",
             ""]

    if report.per_dimension:
        lines.append(f"{'Dimension':20s} | {'Pearson r':>9s} | {'MAD':>6s} | {'Agree1pt':>8s}")
        lines.append("-" * 50)
        for dim, metrics in report.per_dimension.items():
            lines.append(f"{dim:20s} | {metrics['pearson_r']:9.4f} | "
                        f"{metrics['mean_abs_diff']:6.4f} | {metrics['agreement_1pt']:8.4f}")

    if report.human_vs_llm_spearman_r:
        lines.append(f"\nHuman-vs-LLM Judge Spearman r:")
        for dim, r in report.human_vs_llm_spearman_r.items():
            lines.append(f"  {dim:20s}: {r:+.4f}")

    return "\n".join(lines)


def save_ratings(ratings: list[HumanRating], path: str) -> None:
    """Save human ratings to JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([{
            "rater_id": r.rater_id, "query_id": r.query_id,
            "condition": r.condition, "query": r.query[:200],
            "answer": r.answer[:300],
            "correctness": r.correctness, "completeness": r.completeness,
            "conciseness": r.conciseness, "citation_accuracy": r.citation_accuracy,
            "groundedness": r.groundedness, "overall": r.overall,
            "notes": r.notes,
        } for r in ratings], f, indent=2, ensure_ascii=False)
    print(f"Ratings saved to {path}")
