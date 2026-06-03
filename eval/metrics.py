"""Evaluation metrics for retrieval and verification assessment.

Implements standard IR metrics for the Agent-OS evaluation framework:
precision@k, recall@k, MRR, nDCG@k, hit@k.
"""

import math
from typing import Sequence


def precision_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """Precision@k: fraction of top-k results that are relevant.

    Args:
        retrieved: Ordered list of retrieved item IDs (best first).
        relevant: Set of relevant item IDs (ground truth).
        k: Number of top results to consider.

    Returns:
        Precision score in [0.0, 1.0].
    """
    if k <= 0 or len(retrieved) == 0:
        return 0.0

    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """Recall@k: fraction of all relevant items found in top-k results.

    Args:
        retrieved: Ordered list of retrieved item IDs (best first).
        relevant: Set of relevant item IDs (ground truth).
        k: Number of top results to consider.

    Returns:
        Recall score in [0.0, 1.0].
    """
    if len(relevant) == 0:
        return 0.0

    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def mrr(
    results: list[Sequence[str]],
    relevant_sets: list[set[str]],
) -> float:
    """Mean Reciprocal Rank: average of 1/rank for the first relevant result.

    Args:
        results: List of ranked result lists, one per query.
        relevant_sets: List of relevant-item sets, one per query.

    Returns:
        MRR score in [0.0, 1.0].
    """
    if len(results) == 0:
        return 0.0

    reciprocal_ranks = []
    for ranked, relevant in zip(results, relevant_sets):
        for rank, item in enumerate(ranked, start=1):
            if item in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Args:
        retrieved: Ordered list of retrieved item IDs (best first).
        relevance_scores: Dict mapping item ID to relevance score (higher = more relevant).
        k: Number of top results to consider.

    Returns:
        nDCG@k score in [0.0, 1.0].
    """
    if k <= 0 or len(retrieved) == 0:
        return 0.0

    # DCG: sum of (2^rel_i - 1) / log2(i + 1)
    def dcg(items):
        score = 0.0
        for i, item in enumerate(items[:k], start=1):
            rel = relevance_scores.get(item, 0.0)
            if rel > 0:
                score += (2 ** rel - 1) / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved)

    # IDCG: ideal ordering (sort by relevance descending then compute DCG)
    ideal_order = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
    ideal_dcg = dcg(ideal_order)

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def hit_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
) -> bool:
    """Hit@k: whether at least one relevant item appears in top-k.

    Args:
        retrieved: Ordered list of retrieved item IDs (best first).
        relevant: Set of relevant item IDs (ground truth).
        k: Number of top results to consider.

    Returns:
        True if at least one relevant item found in top-k.
    """
    top_k = retrieved[:k]
    return any(item in relevant for item in top_k)


def compute_all_metrics(
    retrieved_all: list[Sequence[str]],
    relevant_all: list[set[str]],
    relevance_scores_all: list[dict[str, float]] | None = None,
    k: int = 10,
) -> dict[str, float]:
    """Compute all retrieval metrics for a query set.

    Args:
        retrieved_all: Per-query ranked result lists.
        relevant_all: Per-query relevant-item sets.
        relevance_scores_all: Per-query relevance scores (for nDCG). If None, nDCG is skipped.
        k: Cutoff rank.

    Returns:
        Dict with metric_name -> average score.
    """
    metrics = {}

    # Precision@k
    precisions = [precision_at_k(r, rel, k) for r, rel in zip(retrieved_all, relevant_all)]
    metrics[f"precision@{k}"] = sum(precisions) / len(precisions) if precisions else 0.0

    # Recall@k
    recalls = [recall_at_k(r, rel, k) for r, rel in zip(retrieved_all, relevant_all)]
    metrics[f"recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0

    # MRR
    metrics["mrr"] = mrr(retrieved_all, relevant_all)

    # nDCG@k
    if relevance_scores_all:
        ndcgs = []
        for retrieved, scores in zip(retrieved_all, relevance_scores_all):
            ndcgs.append(ndcg_at_k(retrieved, scores, k))
        metrics[f"ndcg@{k}"] = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

    # Hit@k
    hits = [hit_at_k(r, rel, k) for r, rel in zip(retrieved_all, relevant_all)]
    metrics[f"hit@{k}"] = sum(hits) / len(hits) if hits else 0.0

    return metrics
