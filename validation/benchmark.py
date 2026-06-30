"""
Benchmark — Full Comparison Suite
===================================
Compare MadhavaCore, MadHybrid, HNSW, IVF against ground truth.

Inherits benchmark structure from madhava_v12.py (Zenodo 21066971).

Usage:
    results = run_benchmark(vectors, queries, k=10)
    print(results['summary'])
"""
import math, time, gc
import numpy as np


def _build_gt(vectors, queries, k):
    """Get ground truth via FAISS FlatIP."""
    from winnex_pipeline.validation.ground_truth import build_ground_truth
    return build_ground_truth(vectors, queries, k)


def run_benchmark(vectors, queries, k=10, methods=None, config=None):
    """
    Run benchmark comparing all available methods.

    Args:
        vectors: np.ndarray (N, D) — corpus
        queries: np.ndarray (NQ, D) — queries
        k: recall/ndcg depth
        methods: list of method names (default: all available)
        config: config dict or path

    Returns:
        dict with all results and summary
    """
    from winnex_pipeline.config import load_config
    from winnex_pipeline.validation.metrics import compute_metrics

    cfg = config if isinstance(config, dict) else load_config(config)
    N = len(vectors)
    NQ = len(queries)
    D = vectors.shape[1]

    # Ground truth
    print("Computing ground truth (FAISS FlatIP)...")
    gt_indices, gt_times = _build_gt(vectors, queries, k)
    flat_latency = float(np.mean(gt_times))

    results = {
        'config': {
            'N': N, 'D': D, 'n_queries': NQ, 'k': k,
        },
        'ground_truth': {
            'method': 'FAISS FlatIP',
            'latency_ms_mean': flat_latency,
        },
        'methods': {},
    }

    if methods is None:
        methods = ['madhava', 'madhybrid', 'hnsw', 'ivf']

    for method in methods:
        method = method.lower()

        if method == 'madhava':
            from winnex_pipeline.core.madhava import MadhavaCore
            idx = MadhavaCore(cfg)
            idx.build(vectors)
            latencies, recalls, ndcgs = [], [], []
            for qi in range(NQ):
                retrieved, prof = idx.search(queries[qi], k=k, return_profile=True)
                r, n = compute_metrics(retrieved, gt_indices[qi], k=k)
                recalls.append(r)
                ndcgs.append(n)
                latencies.append(prof.get('latency_ms', 0))
            results['methods']['MadhavaCore'] = {
                'recall_mean': float(np.mean(recalls)),
                'ndcg_mean': float(np.mean(ndcgs)),
                'latency_ms_mean': float(np.mean(latencies)),
                'build_s': idx.build_time,
            }

        elif method == 'madhybrid':
            if not cfg['hybrid']['enabled']:
                continue
            from winnex_pipeline.core.madhybrid import MadHybrid
            idx = MadHybrid(cfg)
            idx.build(vectors)
            for np_ in cfg['hybrid']['n_probe'][:2]:
                latencies, recalls, ndcgs = [], [], []
                for qi in range(NQ):
                    t0 = time.time()
                    retrieved = idx.search(queries[qi], k=k, n_probe=np_)
                    elapsed = (time.time() - t0) * 1000
                    r, n = compute_metrics(retrieved, gt_indices[qi], k=k)
                    recalls.append(r)
                    ndcgs.append(n)
                    latencies.append(elapsed)
                results['methods'][f'MadHybrid(np={np_})'] = {
                    'recall_mean': float(np.mean(recalls)),
                    'ndcg_mean': float(np.mean(ndcgs)),
                    'latency_ms_mean': float(np.mean(latencies)),
                    'build_s': idx.build_time,
                    'n_probe': np_,
                }

        elif method == 'hnsw':
            try:
                import faiss
                idx = faiss.IndexHNSWFlat(D, 32)
                idx.hnsw.efConstruction = 200
                idx.add(vectors)
                for ef in [64, 128]:
                    idx.hnsw.efSearch = ef
                    latencies, recalls, ndcgs = [], [], []
                    for qi in range(NQ):
                        t0 = time.time()
                        _, I = idx.search(queries[qi:qi + 1], k)
                        elapsed = (time.time() - t0) * 1000
                        r, n = compute_metrics(I[0], gt_indices[qi], k=k)
                        recalls.append(r)
                        ndcgs.append(n)
                        latencies.append(elapsed)
                    results['methods'][f'HNSW(ef={ef})'] = {
                        'recall_mean': float(np.mean(recalls)),
                        'ndcg_mean': float(np.mean(ndcgs)),
                        'latency_ms_mean': float(np.mean(latencies)),
                    }
            except ImportError:
                pass

        elif method == 'ivf':
            try:
                import faiss
                quant = faiss.IndexFlatIP(D)
                nlist = min(int(math.sqrt(N)), 256)
                idx = faiss.IndexIVFFlat(quant, D, nlist, faiss.METRIC_INNER_PRODUCT)
                idx.train(vectors)
                idx.add(vectors)
                for npb in [5, 10]:
                    idx.nprobe = npb
                    latencies, recalls, ndcgs = [], [], []
                    for qi in range(NQ):
                        t0 = time.time()
                        _, I = idx.search(queries[qi:qi + 1], k)
                        elapsed = (time.time() - t0) * 1000
                        r, n = compute_metrics(I[0], gt_indices[qi], k=k)
                        recalls.append(r)
                        ndcgs.append(n)
                        latencies.append(elapsed)
                    results['methods'][f'IVF(nprobe={npb})'] = {
                        'recall_mean': float(np.mean(recalls)),
                        'ndcg_mean': float(np.mean(ndcgs)),
                        'latency_ms_mean': float(np.mean(latencies)),
                    }
            except ImportError:
                pass

        gc.collect()

    # Summary table
    results['summary'] = _format_summary(results)
    return results


def _format_summary(results):
    """Build plain-text summary table."""
    lines = []
    lines.append(f"Benchmark: N={results['config']['N']}, "
                 f"D={results['config']['D']}, k={results['config']['k']}")
    lines.append(f"Ground truth: {results['ground_truth']['method']} "
                 f"({results['ground_truth']['latency_ms_mean']:.2f}ms)")
    lines.append("")
    lines.append(f"{'Method':>30} {'R@{k}':>8} {'NDCG':>8} {'Lat(ms)':>10} {'Build(s)':>10}"
                 .format(k=results['config']['k']))
    lines.append(f"{'─' * 66}")
    for method, data in results['methods'].items():
        lines.append(f"{method:>30} "
                     f"{data.get('recall_mean', 0):>8.4f} "
                     f"{data.get('ndcg_mean', 0):>8.4f} "
                     f"{data.get('latency_ms_mean', 0):>10.3f} "
                     f"{data.get('build_s', 0):>10.3f}")
    return '\n'.join(lines)
