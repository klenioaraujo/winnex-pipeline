"""
Ground Truth — FAISS FlatIP Exact Search
==========================================
Compute ground truth top-K via exhaustive FAISS FlatIP.

Validated: FlatIP = exact cosine similarity (ground truth for all benchmarks)
"""
import time
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


def build_ground_truth(vectors, queries, k=10):
    """
    Compute exact ground truth using FAISS FlatIP.

    Args:
        vectors: corpus embeddings, np.ndarray (N, D)
        queries: query embeddings, np.ndarray (NQ, D)
        k: number of results to return

    Returns:
        gt_indices: np.ndarray (NQ, K) — ground truth indices
        gt_times: list of latencies in ms
    """
    if not _HAS_FAISS:
        raise ImportError("FAISS required for ground truth computation. "
                          "Install with: pip install faiss-cpu")

    D = vectors.shape[1]
    idx = faiss.IndexFlatIP(D)
    idx.add(vectors)

    NQ = len(queries)
    gt_indices = np.zeros((NQ, k), dtype=np.int32)
    gt_times = []

    for qi in range(NQ):
        t0 = time.time()
        _, I = idx.search(queries[qi:qi + 1], k)
        elapsed = (time.time() - t0) * 1000
        gt_indices[qi] = I[0]
        gt_times.append(elapsed)

    return gt_indices, gt_times
