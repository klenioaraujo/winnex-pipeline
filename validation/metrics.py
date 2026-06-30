"""
Metrics — NDCG@K, Recall@K, Bound Violations
==============================================
Validated: same metric functions used in madhava_qrjl_benchmark.py
"""
import math
import numpy as np


def ndcg_at_k(ranked, true_scores, k=10):
    """
    Normalized Discounted Cumulative Gain @ K.

    Args:
        ranked: list of retrieved indices (ordered by relevance)
        true_scores: dict mapping index -> relevance score
        k: depth

    Returns:
        float: NDCG@K
    """
    dcg = 0.0
    for j, idx in enumerate(ranked[:k]):
        rel = true_scores.get(int(idx), 0.0)
        dcg += (2 ** rel - 1) / math.log2(j + 2)

    # Ideal DCG
    sorted_by_score = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
    idcg = 0.0
    for j, (idx, rel) in enumerate(sorted_by_score[:k]):
        idcg += (2 ** rel - 1) / math.log2(j + 2)

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, true_indices, k=10):
    """
    Recall @ K: fraction of ground truth results found in retrieved set.

    Args:
        ranked: list of retrieved indices
        true_indices: list/array of ground truth indices
        k: depth

    Returns:
        float: Recall@K
    """
    retrieved = set(ranked[:k])
    relevant = set(true_indices[:k])
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


def compute_metrics(retrieved, true_indices, true_scores=None, k=10):
    """
    Compute both Recall@K and NDCG@K.

    Args:
        retrieved: list of retrieved indices
        true_indices: ground truth indices (list or array)
        true_scores: dict for NDCG (optional)
        k: depth

    Returns:
        (recall, ndcg) tuple
    """
    rec = recall_at_k(retrieved, true_indices, k)
    if true_scores is not None:
        ndcg = ndcg_at_k(retrieved, true_scores, k)
    else:
        # Use binary relevance from true_indices
        scores = {int(idx): 1.0 for idx in true_indices[:k]}
        ndcg = ndcg_at_k(retrieved, scores, k)
    return rec, ndcg


def bound_violation_rate(violations, total_pairs):
    """
    Compute bound violation rate from check_bounds results.

    Args:
        violations: dict mapping dim -> violation count
        total_pairs: total query-vector pairs checked

    Returns:
        dict mapping dim -> violation rate
    """
    return {k: v / max(total_pairs, 1) for k, v in violations.items()}
