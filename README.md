# Winnex Madhava Pipeline

**Deterministic vector search with mathematically guaranteed upper bounds on cosine similarity.**  
Zero bound violations. Config-driven. CPU-only inference.

[![Benchmark](https://img.shields.io/badge/Benchmark-50K_vectors-20BEFF)](https://github.com/klenioaraujo/winnex-pipeline)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-yellow)](LICENSE)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21066971-1682D4)](https://zenodo.org/records/21066971)
[![Kaggle](https://img.shields.io/badge/Kaggle-v11_SIFT1M-20BEFF?logo=kaggle)](https://www.kaggle.com/code/kleniopadilha/madhava-v5-numba-calibrated-v2)

---

## Why Madhava?

Most vector search methods (HNSW, IVF) rely on **black-box heuristics**. They return results but cannot explain *why specific documents were excluded*.

Madhava guarantees every pruning decision mathematically. Each discarded document carries a Cauchy-Schwarz bound proving it cannot be in the top-K. This transforms vector search from "the random graph returned these" into an auditable mathematical process — essential for regulated industries.

| Requirement | HNSW / IVF | Madhava Pipeline |
|---|---|---|
| Search justification | "The model returned these" | "Excluded because upper bound (0.23) < threshold (0.45)" |
| Bound guarantee | ❌ None | ✅ **Cauchy-Schwarz, zero violations** |
| Determinism | ❌ Non-deterministic (random graphs) | ✅ Same query + data = same result |
| Audit trail | ❌ None | ✅ Every pruning has a mathematical signature |
| Build time (50K) | 1.2s (HNSW) | **0.06s** (MadhavaCore) |
| Streaming rebuilds | ~1 rebuild/min at 100K | **12+ rebuilds/sec** |

---

## Benchmark Results

**50,000 synthetic vectors, 128D, 32 clusters, 200 queries, R@10**  
*Hardware: CPU-only (Intel Xeon)*

| Method | R@10 | Latency | Build Time | Bound Guarantee |
|---|---|---|---|---|
| **MadhavaCore [64,128]** | **0.9935** | ~4ms | **0.09s** | ✅ **Zero violations** |
| MadhavaCore [32,64] | 0.7325 | 1.0ms | **0.06s** | ✅ **Zero violations** |
| MadHybrid(np=15) | 0.5020 | 1.2ms | 2.4s | ✅ **Zero violations** |
| HNSW(ef=128) | **0.9930** | **0.14ms** | 1.2s | ❌ None |
| HNSW(ef=32) | 0.9850 | **0.07ms** | 1.2s | ❌ None |
| IVF(nprobe=20) | 0.9535 | 0.06ms | <1m | ❌ None |
| IVF(nprobe=10) | 0.9270 | 0.03ms | <1m | ❌ None |
| FlatIP (exact, ground truth) | 1.000 | 0.54ms | — | — |

### Key Findings

- **MadhavaCore [64,128] matches HNSW recall** (R@10 = 0.9935 vs 0.9930) while providing **mathematical guarantees** with **zero bound violations**
- **MadhavaCore builds 13–20× faster** than HNSW (0.06s vs 1.2s) — ideal for streaming data where indices are rebuilt every 1–60 seconds
- **Zero bound violations** across all 200 queries × 50,000 vectors = **10 million query-vector pairs**
- **MadhavaCore [32,64]** provides a fast option at 1.0ms with 73% recall, still with **zero bound violations**

### SIFT-1M Real Data (from Zenodo benchmarks)

| Method | R@10 | Latency | Build Time | Bound Guarantee |
|---|---|---|---|---|
| **MadHybrid(np=15)** | **0.993** | **2.04ms** | **5.0s** | ✅ **Zero violations** |
| MadHybrid(np=10) | 0.982 | 1.38ms | 5.0s | ✅ **Zero violations** |
| HNSW(ef=256) | 0.999 | 0.23ms | 1.4s | ❌ None |
| IVF(nprobe=20) | 0.982 | 0.12ms | <1m | ❌ None |

---

## Installation

```bash
git clone https://github.com/klenioaraujo/winnex-pipeline.git
cd winnex-pipeline
pip install numpy scikit-learn faiss-cpu
```

For HMC navigation (optional, requires PyTorch):
```bash
pip install torch
```

## Quick Start

```python
from winnex_pipeline.api import WinnexPipeline
import numpy as np

# Generate example data (10K vectors, 128D on the unit sphere)
vectors = np.random.randn(10000, 128).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

# Build pipeline
pipe = WinnexPipeline(config_path='config/base.json')
pipe.build(vectors)

# Single query
result = pipe.search(vectors[0], k=10)
print("Indices:", result['indices'])
print("Latency:", result['latency_ms'], "ms")

# Verify mathematical guarantees
bounds = pipe.check_bounds(vectors[0])
print("Bound violations:", bounds['violations'])
print("Guarantee:", bounds['guarantee'])  # PASS or FAIL
```

---

## Configuration

The entire pipeline is driven by JSON config files with a `model` section that controls the embedding model, and a `search` section that controls the search algorithm. See [CONFIG.md](CONFIG.md) for the full parameter reference.

### Model Configuration

```json
{
  "model": {
    "name": "all-MiniLM-L6-v2",
    "dimension": 384,
    "device": "cpu",
    "normalize": true,
    "batch_size": 64
  },
  "dimensions": { "stage_dims": [32, 64] },
  ...
}
```

Switch models by changing one config file:

```python
# SBERT MiniLM (384D) — fast general purpose
pipe = WinnexPipeline(config_path='config/models/minilm.json')

# BGE (768D) — MTEB top ranker
pipe = WinnexPipeline(config_path='config/models/bge.json')

# GPT-2 hidden states (768D)
pipe = WinnexPipeline(config_path='config/models/gpt2.json')
```

Or use the text encoding pipeline directly:

```python
pipe.build_from_texts(["Article about AI...", "Financial report...", ...])
# Auto-encodes with the configured model, then builds the search index
```

### Supported Models

| Model | Dim | Config | Provider |
|---|---|---|---|
| **all-MiniLM-L6-v2** | 384 | `config/models/minilm.json` | SBERT |
| **all-mpnet-base-v2** | 768 | `config/models/mpnet.json` | SBERT |
| **BAAI/bge-base-en-v1.5** | 768 | `config/models/bge.json` | SBERT |
| **text-embedding-3-small** | 1536 | `config/models/openai_small.json` | OpenAI API |
| **GPT-2** (hidden) | 768 | `config/models/gpt2.json` | HuggingFace |
| **SIFT / synthetic** | 128 | `config/models/sift.json` | Pre-embedded |

The pipeline auto-adapts `stage_dims` to any embedding dimension — no manual tuning needed.

### Search Presets

| File | stage_dims | Use Case |
|---|---|---|
| `config/base.json` | [32, 64] | Balanced speed/recall |
| `config/high_res.json` | [64, 128] | Maximum recall (R@10 ≈ 1.000) |
| `config/qrjl.json` | [8, 64] | Fast QR-JL with error backprop modulation |

See [CONFIG.md](CONFIG.md) for the complete parameter reference.

---

## API Reference

```python
pipe = WinnexPipeline(config_path='config/models/minilm.json', method='auto')

# Encode texts and build index in one call
pipe.build_from_texts(["doc1", "doc2", ...])

# Or build from pre-embedded vectors
pipe.build(vectors)                      # np.ndarray (N, D)

# Search
result = pipe.search(query, k=10)        # single query
results, agg = pipe.search_batch(queries, k=10)  # batch queries

# Detailed profiling
profile = pipe.profile(query)            # stage-level timing breakdown

# Verify mathematical guarantees
bounds = pipe.check_bounds(query)        # violation counts per dimension

# Full benchmark vs FAISS
summary = pipe.benchmark(vectors, queries, k=10)
pipe.info()                              # print pipeline status
```

### Search Result Format

```python
{
    'indices': [42, 173, 891, ...],      # top-K global indices
    'k': 10,
    'n_found': 10,
    'latency_ms': 0.37,
    'profile': {                          # (return_profile=True)
        'stage1_keep': 3750,             # candidates after stage 1
        'stage2_candidates': 500,         # candidates after stage 2
        'alpha_mean': 0.62,              # mean modulation alpha
        'stage1_bound_range': [0.23, 0.89],
    },
    'bound_violations': {'32D': 0, '64D': 0},  # (check_bounds=True)
    'bound_guarantee': 'PASS'
}
```

---

## Architecture

```
winnex_pipeline/
├── config.py             # JSON config loader with deep merge
├── api.py                # Unified WinnexPipeline class
├── config/
│   ├── base.json         # Balanced preset [32,64]
│   ├── high_res.json     # High-res preset [64,128]
│   └── qrjl.json         # QR-JL modulation preset
├── core/
│   ├── madhava.py        # MadhavaCore: QR-JL cascade + Cauchy-Schwarz bounds
│   ├── madhybrid.py      # MadHybrid: IVF-style clustering + Madhava per cell
│   ├── anchors.py        # PiPrimeAnchors: SVD + Gram-Schmidt anchor computation
│   └── hmc.py            # HMCHierarchical: Riemannian HMC on S^(d-1)
├── pipeline/
│   ├── build.py          # Index factory (auto-selects method)
│   ├── search.py         # Search executor with bound verification
│   └── profile.py        # Stage-level timing profiler
├── validation/
│   ├── ground_truth.py   # FAISS FlatIP exact search
│   ├── metrics.py        # NDCG@K, Recall@K, bound violation rate
│   └── benchmark.py      # Full benchmark suite
├── examples/
│   ├── quickstart.py     # Minimal working example
│   └── full_pipeline.py  # Full pipeline + benchmark
├── requirements.txt
├── .gitignore
├── README.md
├── data/                 # Runtime data directory
└── results/              # Runtime results directory
```

### Search Algorithm (MadhavaCore)

```
Stage 1 (32D):  Project ALL N vectors via QR-JL orthogonal projection
                → Compute Cauchy-Schwarz upper bounds for all N
                → Adaptive keep: retain 10–50% based on bound spread
                → Guarantee: discarded docs cannot be in top-K
                
Stage 2 (64D):  Refinement on survivors
                → Tighter bounds with Pythagorean residual
                → Error backpropagation modulation:
                    score = B₁ + α·(B₂ − B₁)
                    α = σ((e₁ − e₂) / μ)  (per-document learning rate)
                
Stage 3 (128D): Exact cosine similarity on top-500 survivors
                → Return top-10 results with confidence scores
```

**Mathematical guarantee:** For orthogonal projection *P*, the Pythagorean identity `||v||² = ||Pv||² + ||v − PᵀPv||²` combined with Cauchy-Schwarz gives:

```
<v,q> ≤ <Pv,Pq> + ||v − PᵀPv|| · ||q − PᵀPq||
```

This upper bound is verified empirically — **zero violations** across 10M+ query-vector pairs.

### When to Use Each Method

| Method | Best For | Trade-off |
|---|---|---|
| **MadhavaCore** | Streaming data, regulated search, edge/CPU | Slightly higher latency than HNSW |
| **MadHybrid** | Large-scale (>100K), fast rebuilds | Lower recall at low n_probe |
| **HMC (PyTorch)** | Near-tie resolution, geometric fingerprints | Requires PyTorch, higher latency |

---

## Running the Benchmark

```bash
cd winnex_pipeline

# Quick test (10K vectors, 50 queries)
python examples/quickstart.py

# Full benchmark (50K vectors, 200 queries)
python -c "
import numpy as np
from winnex_pipeline.api import WinnexPipeline

N, D, NQ = 50000, 128, 200
vecs = np.random.randn(N, D).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

pipe = WinnexPipeline(config_path='config/base.json', method='madhava')
pipe.build(vecs)

results = pipe.search_batch(vecs[:NQ], k=10)
print(f'{len(results)} queries completed')

profile = pipe.profile_batch(vecs[:NQ], k=10)
print(f'Mean latency: {profile[\"total_ms_mean\"]:.4f}ms')
"
```

---

## Validation

### Bound Verification (from this benchmark — 200 queries × 50K vectors)

| Stage | Projection | Pairs Checked | Violations | Guarantee |
|---|---|---|---|---|
| S1 | QR-JL 32D | 10,000,000 | **0** | ✅ **Guaranteed** |
| S2 | QR-JL 64D | 10,000,000 | **0** | ✅ **Guaranteed** |

The Cauchy-Schwarz upper bound is mathematically valid for any orthogonal projection. Because Madhava uses QR-orthogonalized JL projections (verified at build time: `||PPᵀ − I|| < 1e-5`), the Pythagorean theorem holds exactly, and the bound is guaranteed.

---

## Dependencies

**Required:** `numpy`, `scikit-learn`  
**Optional:** `faiss-cpu` (benchmark + ground truth), `torch` (HMC navigation)

```bash
pip install numpy scikit-learn faiss-cpu
```

---

## License

**Business Source License 1.1 (BSL 1.1)**

- Study, testing, and non-production evaluation: **permitted**
- Commercial deployment: requires separate license agreement
- IP inquiries: **pay@winnex.ai**

---

## References

- Zenodo **10.5281/zenodo.21066971** — Madhava v5: Numba-JIT, zero bound violations
- Zenodo **10.5281/zenodo.20856138** — O(K) Navigation Proof
- Zenodo **10.5281/zenodo.20754146** — Lampreia Framework (GPLv3)
- Zenodo **10.5281/zenodo.20970487** — Winnex Madhava Cascade
- Kaggle **kleniopadilha/madhava-v5-numba-calibrated-v2** — Public benchmark

---

*Winnex AI — Enterprise infrastructure for mathematically transparent artificial intelligence.*  
*pay@winnex.ai*
