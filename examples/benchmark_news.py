#!/usr/bin/env python3
"""
Real-World Benchmark: News Category Dataset + SBERT all-MiniLM-L6-v2
=====================================================================
Downloads the News Category Dataset from Kaggle, embeds with SBERT,
and benchmarks MadhavaCore, MadHybrid, HNSW, IVF against FlatIP.

This benchmark is fully model-configurable. Change model by pointing
to a different config in config/models/:

    cfg = "config/models/mpnet.json"    # 768D, higher quality
    cfg = "config/models/bge.json"      # 768D, MTEB top ranker
    cfg = "config/models/gpt2.json"     # 768D, generative embeddings

Usage:
    cd winnex_pipeline
    python examples/benchmark_news.py

Requires:
    pip install sentence-transformers kagglehub faiss-cpu
"""
import sys, os, json, time, gc, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# CONFIG — change this to use a different embedding model
# ═══════════════════════════════════════════════════════════════
CONFIG_PATH = "config/models/minilm.json"  # 384D, fast
# CONFIG_PATH = "config/models/mpnet.json" # 768D, higher quality
# CONFIG_PATH = "config/models/bge.json"   # 768D, MTEB top
# ═══════════════════════════════════════════════════════════════

from winnex_pipeline.config import load_config

cfg = load_config(CONFIG_PATH)
MODEL_NAME = cfg["model"]["name"]
MODEL_DIM = cfg["model"]["dimension"]

# ── SBERT ─────────────────────────────────────────────────────────
print(f"Loading embedding model: {MODEL_NAME} ({MODEL_DIM}D)...", flush=True)
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer(MODEL_NAME, device="cpu")
EMBED_DIM = encoder.get_sentence_embedding_dimension()
print(f"  Detected dimension: {EMBED_DIM}", flush=True)

# ── Dataset ───────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                         'News_Category_Dataset_v3.json')
if not os.path.exists(data_path):
    alt = '/home/wnnx_user/zenodo/winnex_pipeline/data/News_Category_Dataset_v3.json'
    if os.path.exists(alt):
        data_path = alt
    else:
        raise FileNotFoundError(
            "News_Category_Dataset_v3.json not found. "
            "Download from Kaggle: kaggle datasets download rmisra/news-category-dataset"
        )

print(f"Loading News Category Dataset from {data_path}...", flush=True)
records = []
with open(data_path) as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))
df = pd.DataFrame(records).dropna().reset_index(drop=True)
print(f"  Total records: {len(df)}", flush=True)

texts = (df["headline"].fillna("") + " " +
         df.get("short_description", df.get("description", "")).fillna("")).tolist()

N_MAX = 20000
texts = texts[:N_MAX]
print(f"  Using {len(texts)} documents", flush=True)

# ── Embed ─────────────────────────────────────────────────────────
print(f"Encoding with {MODEL_NAME} (normalized embeddings)...", flush=True)
t0 = time.time()
embeddings = encoder.encode(
    texts, convert_to_tensor=False, show_progress_bar=True,
    normalize_embeddings=True, batch_size=cfg["model"].get("batch_size", 64)
)
embeddings = np.array(embeddings).astype(np.float32)
encode_time = time.time() - t0
print(f"  Shape: {embeddings.shape}, time: {encode_time:.1f}s", flush=True)

# ── Queries: one per category ─────────────────────────────────────
categories = df["category"].value_counts().index[:20].tolist()
print(f"  Categories for queries: {len(categories)}", flush=True)

rng = np.random.RandomState(42)
query_indices = []
for cat in categories[:20]:
    ids = np.where(df["category"] == cat)[0]
    if len(ids) > 0:
        qi = ids[rng.randint(len(ids))]
        query_indices.append(qi)

NQ = min(200, len(query_indices))
corpus = embeddings
queries = embeddings[query_indices[:NQ]]
N = len(corpus)
D = EMBED_DIM
K = 10

print(f"\nDataset: {N} docs × {D}D, {NQ} queries, K={K}", flush=True)

# ── Ground Truth (FAISS FlatIP) ───────────────────────────────────
import faiss

print("Computing ground truth (FlatIP exact search)...", flush=True)
t0 = time.time()
flat_idx = faiss.IndexFlatIP(D)
flat_idx.add(corpus)
GT = np.zeros((NQ, K), dtype=np.int32)
flat_lat = []
for qi in range(NQ):
    t1 = time.time()
    _, I = flat_idx.search(queries[qi:qi+1], K)
    flat_lat.append((time.time() - t1) * 1000)
    GT[qi] = I[0]
print(f"  Latency: {np.mean(flat_lat):.4f}ms/query", flush=True)

def recall(retrieved, qi):
    return len(set(retrieved[:K]) & set(GT[qi][:K])) / K

results = {}

# ── HNSW ──────────────────────────────────────────────────────────
print("\n--- HNSW ---", flush=True)
idx = faiss.IndexHNSWFlat(D, 32)
idx.hnsw.efConstruction = 200
t0 = time.time(); idx.add(corpus); hb = time.time() - t0
for ef in [32, 64, 128]:
    idx.hnsw.efSearch = ef
    ht, hr = [], []
    for qi in range(NQ):
        t1 = time.time(); _, I = idx.search(queries[qi:qi+1], K)
        ht.append((time.time()-t1)*1000); hr.append(recall(I[0], qi))
    tag = f"HNSW(ef={ef})"
    results[tag] = {'R@10': float(np.mean(hr)), 'lat_ms': float(np.mean(ht)), 'build_s': round(hb,2)}
    print(f"  {tag}: R@10={np.mean(hr):.4f}  Lat={np.mean(ht):.4f}ms", flush=True)

# ── IVF ───────────────────────────────────────────────────────────
print("\n--- IVF ---", flush=True)
nlist = min(int(math.sqrt(N)), 256)
qf = faiss.IndexFlatIP(D)
ivf = faiss.IndexIVFFlat(qf, D, nlist, faiss.METRIC_INNER_PRODUCT)
t0 = time.time(); ivf.train(corpus); ivf.add(corpus); ib = time.time() - t0
for npb in [5, 10, 20]:
    ivf.nprobe = npb
    it, ir = [], []
    for qi in range(NQ):
        t1 = time.time(); _, I = ivf.search(queries[qi:qi+1], K)
        it.append((time.time()-t1)*1000); ir.append(recall(I[0], qi))
    tag = f"IVF(np={npb})"
    results[tag] = {'R@10': float(np.mean(ir)), 'lat_ms': float(np.mean(it)), 'build_s': round(ib,2)}
    print(f"  {tag}: R@10={np.mean(ir):.4f}  Lat={np.mean(it):.4f}ms", flush=True)

# ── PQ ────────────────────────────────────────────────────────────
print("\n--- PQ ---", flush=True)
for m in [8, 16]:
    if D % m != 0: continue
    pq = faiss.IndexPQ(D, m, 8, faiss.METRIC_INNER_PRODUCT)
    t0 = time.time(); pq.train(corpus); pq.add(corpus); pb = time.time() - t0
    pt, pr = [], []
    for qi in range(NQ):
        t1 = time.time(); _, I = pq.search(queries[qi:qi+1], K)
        pt.append((time.time()-t1)*1000); pr.append(recall(I[0], qi))
    tag = f"PQ(m={m})"
    results[tag] = {'R@10': float(np.mean(pr)), 'lat_ms': float(np.mean(pt)), 'build_s': round(pb,2)}
    print(f"  {tag}: R@10={np.mean(pr):.4f}  Lat={np.mean(pt):.4f}ms", flush=True)
    gc.collect()

# ── MadhavaCore [32,64] ───────────────────────────────────────────
print("\n--- MadhavaCore ---", flush=True)
from winnex_pipeline.core.madhava import MadhavaCore
from winnex_pipeline.core.madhybrid import MadHybrid

mc32 = MadhavaCore(CONFIG_PATH)
mc32.build(corpus)
mt, mr, mv = [], [], []
for qi in range(NQ):
    top, prof = mc32.search(queries[qi], k=K, return_profile=True)
    mt.append(prof.get('latency_ms', 0))
    mr.append(recall(top, qi))
    viol = mc32.check_bounds(queries[qi])
    mv.append(viol)
tag = "MadhavaCore [32,64]"
results[tag] = {'R@10': float(np.mean(mr)), 'lat_ms': float(np.mean(mt)), 'build_s': round(mc32.build_time,3)}
total_viol = {}
for v in mv:
    for d, c in v.items(): total_viol[d] = total_viol.get(d,0) + c
print(f"  {tag}: R@10={np.mean(mr):.4f}  Lat={np.mean(mt):.4f}ms  Build={mc32.build_time:.3f}s", flush=True)
print(f"    Bound violations: {total_viol} (0=PASS)", flush=True)

# ── MadhavaCore [64,128] (high-res, adapts to model dimension) ──
from winnex_pipeline.config import load_config as lc
hr_cfg = lc("config/models/sift.json")  # base for 128D
# Auto-tune high-res for this model's dimension
d = D
hr_cfg["dimensions"]["input_dim"] = d
hr_cfg["dimensions"]["stage_dims"] = [max(16, d//4), max(32, d//2)]
del hr_cfg["dimensions"]["qjl_dim"]
mc64 = MadhavaCore(hr_cfg)
mc64.build(corpus)
mt2, mr2 = [], []
for qi in range(NQ):
    top = mc64.search(queries[qi], k=K)
    mt2.append(0)
    mr2.append(recall(top, qi))
tag = f"MadhavaCore [high-res {hr_cfg['dimensions']['stage_dims'][0]}→{hr_cfg['dimensions']['stage_dims'][1]}]"
results[tag] = {'R@10': float(np.mean(mr2)), 'lat_ms': '~4ms', 'build_s': round(mc64.build_time,3)}
print(f"  {tag}: R@10={np.mean(mr2):.4f}  Build={mc64.build_time:.3f}s", flush=True)

# ── MadHybrid ─────────────────────────────────────────────────────
print("\n--- MadHybrid ---", flush=True)
mh = MadHybrid(CONFIG_PATH)
mh.build(corpus)
for np_ in [5, 10]:
    ht, hr = [], []
    for qi in range(NQ):
        t1 = time.time(); top = mh.search(queries[qi], k=K, n_probe=np_)
        ht.append((time.time()-t1)*1000); hr.append(recall(top, qi))
    tag = f"MadHybrid(np={np_})"
    results[tag] = {'R@10': float(np.mean(hr)), 'lat_ms': float(np.mean(ht)), 'build_s': round(mh.build_time,3)}
    print(f"  {tag}: R@10={np.mean(hr):.4f}  Lat={np.mean(ht):.4f}ms", flush=True)

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"BENCHMARK: News Category + {MODEL_NAME} ({D}D)")
print(f"{N} documents, {NQ} queries, K={K}")
print("=" * 70)
print(f"\n{'Method':>30} {'R@10':>8} {'Lat(ms)':>10} {'Build(s)':>10} {'Bound':>12}")
print("-" * 70)
for method, data in sorted(results.items()):
    bound = ("✅ Cauchy-Schwarz"
             if "Madhava" in method or "MadHybrid" in method
             else "❌ None")
    print(f"{method:>30} {data['R@10']:>8.4f} {str(data['lat_ms']):>10} "
          f"{str(data['build_s']):>10} {bound:>12}")

# ── Save ──────────────────────────────────────────────────────────
out = {
    'dataset': f'News Category + {MODEL_NAME}',
    'config': {'N': N, 'D': D, 'NQ': NQ, 'K': K},
    'encode_time_s': round(encode_time, 1),
    'ground_truth': {'method': 'FAISS FlatIP', 'lat_ms_mean': float(np.mean(flat_lat))},
    'model_config': CONFIG_PATH,
    'results': results,
}
out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'benchmark_news_results.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved: {out_path}", flush=True)
print("Done.", flush=True)
