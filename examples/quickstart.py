#!/usr/bin/env python3
"""
Quickstart — Winnex Pipeline Minimal Example
==============================================
Demonstra a pipeline completa com dados sintéticos.

Execução:
    cd winnex-pipeline
    python examples/quickstart.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from winnex_pipeline.api import WinnexPipeline


def main():
    print("=" * 60)
    print("Winnex Pipeline — Quickstart")
    print("=" * 60)

    # ── 1. Gerar dados sintéticos na esfera ──
    N, D = 10000, 128
    rng = np.random.RandomState(42)
    vectors = rng.randn(N, D).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    queries = rng.randn(5, D).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    print(f"\nDados: {N} vetores × {D}D, {len(queries)} queries")

    # ── 2. Construir pipeline ──
    pipe = WinnexPipeline(config_path='config/base.json', method='auto')
    pipe.build(vectors)

    print(f"Índice: {type(pipe.index).__name__}")
    print(f"Build:  {getattr(pipe.index, 'build_time', 0):.3f}s")

    # ── 3. Busca única ──
    print("\n--- Busca única ---")
    result = pipe.search(queries[0], k=10, return_profile=True)
    print(f"Indices: {result['indices']}")
    print(f"Latência: {result['latency_ms']:.4f}ms")
    if 'profile' in result:
        p = result['profile']
        dims = p.get('stage_dims', '?')
        s1 = p.get('stage1_keep', '?')
        s2 = p.get('stage2_candidates', '?')
        print(f"Cascade: {dims}, Stage1={s1} candidatos, Stage2={s2}")

    # ── 4. Busca em lote ──
    print("\n--- Busca em lote ---")
    results, agg = pipe.search_batch(queries, k=10, return_profile=True)
    print(f"Queries: {agg['n_queries']}")
    print(f"Latência média: {agg['latency_ms_mean']:.4f}ms")
    print(f"Latência (min, max): {agg['latency_ms_min']:.4f}, {agg['latency_ms_max']:.4f}ms")

    # ── 5. Perfil detalhado ──
    print("\n--- Perfil detalhado ---")
    profile = pipe.profile(queries[0])
    print(f"Stage1: {profile.get('stage1_ms', 0):.4f}ms — "
          f"{profile.get('n_candidates_stage1', 0)} candidatos")
    print(f"Stage2: {profile.get('stage2_ms', 0):.4f}ms — "
          f"{profile.get('n_candidates_stage2', 0)} candidatos")
    print(f"Exact:  {profile.get('exact_ms', 0):.4f}ms — "
          f"{profile.get('n_final', 0)} resultados")

    # ── 6. Verificação de bounds ──
    print("\n--- Verificação de bounds ---")
    try:
        bounds = pipe.check_bounds(queries[0])
        viol = bounds['violations']
        print(f"Violações: {viol}")
        print(f"Garantia: {bounds['guarantee']}")
    except AttributeError:
        print("Este método não suporta verificação de bounds")

    # ── 7. Info ──
    print()
    pipe.info()

    print("\n✅ Quickstart concluído com sucesso!")


if __name__ == '__main__':
    main()
