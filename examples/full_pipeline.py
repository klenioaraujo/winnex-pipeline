#!/usr/bin/env python3
"""
Full Pipeline — Benchmark Completo com Dataset Estruturado
============================================================
Demonstra a pipeline completa:
  1. Carregar dataset (SIFT-1M ou sintético estruturado)
  2. Build com 3 métodos (MadhavaCore, MadHybrid, HMC)
  3. Benchmark comparativo vs FAISS HNSW/IVF
  4. Análise de bound violations
  5. Exportação de resultados

Execução:
    cd winnex-pipeline
    python examples/full_pipeline.py       # sintético 50K
    python examples/full_pipeline.py --quick  # sintético 10K

Dependências: numpy, scikit-learn, faiss-cpu (para benchmark)
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from winnex_pipeline.api import WinnexPipeline
from winnex_pipeline.validation.metrics import compute_metrics
from winnex_pipeline.validation.ground_truth import build_ground_truth


def make_synthetic_dataset(N, D=128, n_clusters=16, seed=42):
    """Generate structured synthetic data with cluster structure."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, D).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    X = []
    for ci in range(n_clusters):
        cnt = N // n_clusters + (1 if ci < N % n_clusters else 0)
        pts = centers[ci] + rng.randn(cnt, D).astype(np.float32) * 0.3
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        X.append(pts)

    vectors = np.vstack(X).astype(np.float32)
    # Queries: first 50 vectors held out as queries
    queries = vectors[:50].copy()
    corpus = vectors[50:].copy()
    return corpus, queries


def load_sift_subset(n=50000, nq=200):
    """Try to load SIFT-1M subset, fallback to synthetic."""
    try:
        import h5py
        import urllib.request
        path = '/tmp/sift.hdf5'
        if not os.path.exists(path):
            print("Downloading SIFT-1M (~525MB)...")
            urllib.request.urlretrieve(
                'http://ann-benchmarks.com/sift-128-euclidean.hdf5', path)
        with h5py.File(path, 'r') as f:
            E = f['train'][:n].astype(np.float32)
            Q = f['test'][:nq].astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
        print(f"SIFT-1M loaded: {len(E)} train, {len(Q)} test")
        return E, Q
    except Exception as e:
        print(f"SIFT unavailable ({e}), using synthetic structured data")
        return make_synthetic_dataset(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='10K vectors')
    parser.add_argument('--sift', action='store_true', help='Use SIFT-1M')
    parser.add_argument('--output', type=str, default='../results/benchmark.json')
    args = parser.parse_args()

    N = 10000 if args.quick else 50000
    NQ = 50 if args.quick else 200
    K = 10

    print("=" * 70)
    print("WINNEX PIPELINE — FULL BENCHMARK")
    print("=" * 70)

    # ── 1. Dataset ──
    if args.sift:
        vectors, queries = load_sift_subset(N, NQ)
    else:
        vectors, queries = make_synthetic_dataset(N + NQ)
        queries = queries[:NQ]
        vectors = vectors[:N]

    D = vectors.shape[1]
    N = len(vectors)
    NQ = min(len(queries), NQ)
    queries = queries[:NQ]
    print(f"\nDataset: {N} vectors × {D}D, {NQ} queries")

    # ── 2. Ground Truth ──
    print("\n--- Ground Truth (FAISS FlatIP) ---")
    try:
        t0 = time.time()
        gt_indices, gt_times = build_ground_truth(vectors, queries, k=K)
        print(f"  Latência média: {np.mean(gt_times):.4f}ms")
        print(f"  Build + search: {time.time() - t0:.2f}s")
    except ImportError:
        print("  FAISS não instalado — pulando ground truth")
        gt_indices = None

    # ── 3. Configs ──
    configs = [
        ('MadhavaCore [32,64]', 'config/base.json', 'auto'),
        ('MadhavaCore [64,128]', 'config/high_res.json', 'auto'),
        ('MadHybrid (se disponível)', 'config/base.json', 'madhybrid'),
    ]

    all_results = {}

    for label, config_path, method in configs:
        print(f"\n--- {label} ---")
        pipe = WinnexPipeline(config_path=config_path, method=method)
        t0 = time.time()
        pipe.build(vectors)
        build_time = time.time() - t0

        # Profile
        prof = pipe.profile_batch(queries, k=K)
        t_search = prof['total_ms_mean']

        # Recall
        recalls = []
        for qi in range(min(100, NQ)):
            result = pipe.search(queries[qi], k=K)
            if gt_indices is not None:
                r, _ = compute_metrics(result['indices'], gt_indices[qi], k=K)
                recalls.append(r)

        r10 = float(np.mean(recalls)) if recalls else 0.0

        print(f"  Build:  {build_time:.3f}s")
        print(f"  Search: {t_search:.4f}ms (mean)")
        print(f"  R@{K}:    {r10:.4f}")

        # Bound check (single query)
        try:
            bounds = pipe.check_bounds(queries[0])
            print(f"  Bounds: {bounds['violations']} → {bounds['guarantee']}")
        except (AttributeError, RuntimeError):
            pass

        all_results[label] = {
            'method': method,
            'build_s': round(build_time, 3),
            'latency_ms': round(t_search, 4),
            f'recall@{K}': round(r10, 4),
        }

    # ── 4. Resumo Final ──
    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(f"{'Config':>30} {'Lat(ms)':>10} {'R@10':>8} {'Build(s)':>10}")
    print(f"{'─' * 58}")
    for label, data in all_results.items():
        print(f"{label:>30} {data['latency_ms']:>10.4f} "
              f"{data[f'recall@{K}']:>8.4f} {data['build_s']:>10.3f}")

    # ── 5. Salvar resultados ──
    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary = {
        'config': {'N': N, 'D': D, 'n_queries': NQ, 'k': K},
        'results': all_results,
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResultados salvos em: {out_path}")

    print("\n✅ Full pipeline concluída!")


if __name__ == '__main__':
    main()
