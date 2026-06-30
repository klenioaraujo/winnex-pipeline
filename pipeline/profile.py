"""
Pipeline Profile — Timing Breakdown for Search
===============================================
"""
import math, time
import numpy as np


def profile_search(index, query, k=10):
    """
    Profile a single search with detailed timing breakdown.

    Args:
        index: built search index (MadhavaCore or MadHybrid)
        query: query vector, np.ndarray (D,)
        k: number of results

    Returns:
        dict with timing profile
    """
    from winnex_pipeline.core.madhava import MadhavaCore

    if isinstance(index, MadhavaCore):
        return _profile_madhava(index, query, k)
    elif hasattr(index, 'search_profile'):
        return index.search_profile(query, k)
    else:
        t0 = time.time()
        result = index.search(query, k=k)
        elapsed = (time.time() - t0) * 1000
        return {
            'method': type(index).__name__,
            'latency_ms': round(elapsed, 3),
            'n_results': len(result),
        }


def _profile_madhava(index, q, k):
    """Profile MadhavaCore with stage-level breakdown."""
    q = q.astype(np.float64).flatten()
    qn = np.linalg.norm(q)
    cfg = index.cfg
    s = cfg['search']

    # Stage 1
    t0 = time.time()
    d1 = index.dims[0]
    q1 = (q.astype(np.float32) @ index.proj_matrices[d1].T.astype(np.float32)).astype(np.float64)
    qr1 = math.sqrt(max(0, qn**2 - np.linalg.norm(q1)**2))
    B1 = index._upper_bound(index.proj_L[d1], index.error[d1], q1, qr1)
    t_s1 = (time.time() - t0) * 1000

    # Adaptive keep
    b_range = float(B1.max() - B1.min())
    raw_keep = s['adaptive_keep_base'] * s['adaptive_bounds_sensitivity'] / max(b_range, 0.01)
    adapt_k = min(s['adaptive_keep_max'], max(s['adaptive_keep_min'], raw_keep))
    k1 = min(max(int(index.n * adapt_k), 100), index.n)
    if index.n <= k1:
        idx1 = np.arange(index.n)
    else:
        idx1 = np.argpartition(-B1, k1 - 1)[:k1]
    n_candidates = len(idx1)

    # Stage 2
    t0 = time.time()
    d2 = index.dims[1]
    q2 = (q.astype(np.float32) @ index.proj_matrices[d2].T.astype(np.float32)).astype(np.float64)
    qr2 = math.sqrt(max(0, qn**2 - np.linalg.norm(q2)**2))
    B2 = index._upper_bound(index.proj_L[d2][idx1], index.error[d2][idx1], q2, qr2)

    if cfg['modulation']['error_backprop']:
        e1 = index.error[d1][idx1]
        e2 = index.error[d2][idx1]
        alpha = 1.0 / (1.0 + np.exp(
            -(e1 - e2) / max(np.mean(e1), 1e-9) * cfg['modulation']['alpha_smoothing']
        ))
        alpha = np.clip(alpha, cfg['modulation']['alpha_min'], cfg['modulation']['alpha_max'])
        scores = B1[idx1] + alpha * (B2 - B1[idx1])
    else:
        scores = B2
    t_s2 = (time.time() - t0) * 1000

    # Stage 3: exact cosine
    t0 = time.time()
    k2 = min(s['stage2_topk'], len(idx1))
    idx2 = idx1[np.argpartition(-scores, k2 - 1)[:k2]]
    cos = index.vectors[idx2].astype(np.float64) @ q
    final = idx2[np.argsort(-cos)[:k]]
    t_s3 = (time.time() - t0) * 1000

    return {
        'method': 'MadhavaCore',
        'stage_dims': index.dims,
        'stage1_ms': round(t_s1, 4),
        'stage2_ms': round(t_s2, 4),
        'exact_ms': round(t_s3, 4),
        'total_ms': round(t_s1 + t_s2 + t_s3, 4),
        'n_candidates_stage1': n_candidates,
        'n_candidates_stage2': len(idx2),
        'n_final': len(final),
        'prune_ratio': round(1.0 - n_candidates / max(index.n, 1), 4),
        'n_total': index.n,
    }


def benchmark_profile(index, queries, k=10):
    """
    Profile multiple queries and return aggregate statistics.

    Args:
        index: built search index
        queries: np.ndarray (NQ, D)
        k: number of results

    Returns:
        dict with aggregate profile
    """
    profiles = []
    for qi in range(len(queries)):
        p = profile_search(index, queries[qi], k)
        profiles.append(p)

    agg = {
        'method': profiles[0]['method'],
        'n_queries': len(profiles),
        'stage1_ms_mean': float(np.mean([p['stage1_ms'] for p in profiles])),
        'stage2_ms_mean': float(np.mean([p['stage2_ms'] for p in profiles])),
        'exact_ms_mean': float(np.mean([p['exact_ms'] for p in profiles])),
        'total_ms_mean': float(np.mean([p['total_ms'] for p in profiles])),
        'total_ms_std': float(np.std([p['total_ms'] for p in profiles])),
        'avg_candidates': float(np.mean([p['n_candidates_stage1'] for p in profiles])),
        'avg_prune_ratio': float(np.mean([p['prune_ratio'] for p in profiles])),
    }
    return agg
