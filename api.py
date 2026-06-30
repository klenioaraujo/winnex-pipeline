"""
Winnex Pipeline — Unified Inference API
=========================================
Single entry point for the entire Winnex search stack.

Usage:
    from winnex_pipeline.api import WinnexPipeline

    # Load config + build index
    pipe = WinnexPipeline(config_path='config/base.json')
    pipe.build(embeddings)  # np.ndarray (N, D)

    # Single query
    results = pipe.search(query, k=10)
    # -> {"indices": [...], "scores": [...], "latency_ms": ...}

    # Batch
    results = pipe.search_batch(queries, k=10)
    # -> [result_dict, ...]

    # Profile
    profile = pipe.profile(query)
    # -> detailed stage breakdown

    # Benchmark against FAISS
    summary = pipe.benchmark(vectors, queries, k=10)
"""
import time
import numpy as np
from winnex_pipeline.config import load_config
from winnex_pipeline.pipeline.build import build_index
from winnex_pipeline.pipeline.search import search, search_batch
from winnex_pipeline.pipeline.profile import profile_search, benchmark_profile
from winnex_pipeline.core import _HAS_HMC


class WinnexPipeline:
    """
    Unified Winnex search pipeline.

    Supports:
        - MadhavaCore:  QR-JL cascade (configurable dims)
        - MadHybrid:    IVF clustering + Madhava per cell
        - HMC:          Hierarchical HMC (optional, requires PyTorch)

    Args:
        config_path: path to JSON config file
        method: one of 'auto', 'madhava', 'madhybrid', 'hmc'
    """

    def __init__(self, config_path=None, method='auto'):
        self.cfg = load_config(config_path)
        self.method = method
        self.index = None
        self.is_built = False
        self.n_vectors = 0
        self.dim = 0
        self.raw_dim = self.cfg['dimensions'].get('input_dim', 0)
        self._encoder = None
        self._encoder_type = None
        self._encoder_tokenizer = None
        self._encoder_model = None
        self._encoder_device = None

    # ── Build ─────────────────────────────────────────────────────

    def build(self, vectors):
        """
        Build index over corpus vectors.

        Args:
            vectors: np.ndarray of shape (N, D)

        Returns:
            self
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        self.n_vectors = len(vectors)
        self.dim = vectors.shape[1]

        # Apply model dimensions from config
        model_dim = self.cfg.get('model', {}).get('dimension', 0)
        input_dim = self.cfg['dimensions']['input_dim']
        if model_dim > 0 and model_dim != input_dim:
            self.cfg['dimensions']['input_dim'] = model_dim
            input_dim = model_dim

        # Override config dimensions if vectors don't match
        if self.dim != input_dim:
            self.cfg['dimensions']['input_dim'] = self.dim
            input_dim = self.dim
            # Auto-tune stage dims based on actual dimension
            sd = [d for d in self.cfg['dimensions']['stage_dims'] if d < self.dim]
            if len(sd) < 2:
                sd = [max(4, self.dim // 8), max(8, self.dim // 2)]
            # Ensure second stage does not exceed full dim
            if sd[-1] >= self.dim:
                sd[-1] = max(sd[0] * 2, self.dim // 2)
            self.cfg['dimensions']['stage_dims'] = sd

        self.index = build_index(vectors, method=self.method, config=self.cfg)
        self.is_built = True
        return self

    def is_valid(self):
        """Check if index is built and ready."""
        return self.is_built and self.index is not None

    # ── Search ────────────────────────────────────────────────────

    def search(self, query, k=None, return_profile=False, check_bounds=False):
        """
        Search for top-k nearest neighbors.

        Args:
            query: np.ndarray of shape (D,)
            k: number of results (default: config.final_results)
            return_profile: include detailed timing
            check_bounds: verify mathematical guarantees

        Returns:
            dict with keys: indices, latency_ms, (profile), (bounds)
        """
        if not self.is_valid():
            raise RuntimeError("Pipeline not built. Call .build(vectors) first.")

        query = np.asarray(query, dtype=np.float32).flatten()
        if k is None:
            k = self.cfg['search']['final_results']

        indices, report = search(
            self.index, query, k=k,
            return_profile=True,
            check_bounds=check_bounds,
        )

        result = {
            'indices': indices,
            'k': k,
            'n_found': len(indices),
            'latency_ms': report.get('latency_ms', report.get('total_ms', 0)),
        }

        if return_profile:
            result['profile'] = {k: v for k, v in report.items()
                                 if k not in ('indices',)}
        if check_bounds and 'bound_violations' in report:
            result['bound_violations'] = report['bound_violations']
            result['bound_guarantee'] = report['bound_guarantee']

        return result

    def search_batch(self, queries, k=None, return_profile=False, check_bounds=False):
        """
        Search multiple queries.

        Args:
            queries: np.ndarray (NQ, D)
            k: number of results
            return_profile: include aggregate profile
            check_bounds: verify guarantees

        Returns:
            list of result dicts
            OR (list, aggregate) if return_profile
        """
        if not self.is_valid():
            raise RuntimeError("Pipeline not built. Call .build(vectors) first.")

        queries = np.asarray(queries, dtype=np.float32)
        if k is None:
            k = self.cfg['search']['final_results']

        all_results, agg = search_batch(
            self.index, queries, k=k,
            return_profile=True,
            check_bounds=check_bounds,
        )

        results = []
        for idx, indices in enumerate(all_results):
            results.append({
                'query_idx': idx,
                'indices': indices,
                'k': k,
            })

        if return_profile:
            return results, agg
        return results

    # ── Profile ───────────────────────────────────────────────────

    def profile(self, query, k=None):
        """
        Detailed timing breakdown for a single query.

        Returns:
            dict with stage-level timing
        """
        if not self.is_valid():
            raise RuntimeError("Pipeline not built. Call .build(vectors) first.")

        query = np.asarray(query, dtype=np.float32).flatten()
        if k is None:
            k = self.cfg['search']['final_results']

        return profile_search(self.index, query, k=k)

    def profile_batch(self, queries, k=None):
        """
        Aggregate profile over multiple queries.

        Returns:
            dict with aggregate statistics
        """
        if not self.is_valid():
            raise RuntimeError("Pipeline not built. Call .build(vectors) first.")

        queries = np.asarray(queries, dtype=np.float32)
        if k is None:
            k = self.cfg['search']['final_results']

        return benchmark_profile(self.index, queries, k=k)

    # ── Embedding ──────────────────────────────────────────────────

    def _load_encoder(self):
        """
        Load the embedding model specified in config.

        Supports:
          - SentenceTransformer models (e.g. all-MiniLM-L6-v2)
          - HuggingFace AutoModels via model.name

        Returns:
            encoder with .encode(texts, ...) -> np.ndarray
        """
        model_cfg = self.cfg.get('model', {})
        name = model_cfg.get('name', 'all-MiniLM-L6-v2')
        device = model_cfg.get('device', 'cpu')

        if not hasattr(self, '_encoder') or self._encoder is None:
            # Lazy: try SentenceTransformer first
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(name, device=device)
                self._encoder_type = 'sentence_transformers'
            except ImportError:
                # Fallback: try HuggingFace transformers
                from transformers import AutoTokenizer, AutoModel
                import torch
                self._encoder_tokenizer = AutoTokenizer.from_pretrained(name)
                self._encoder_model = AutoModel.from_pretrained(name).to(device).eval()
                self._encoder_type = 'transformers'
                self._encoder_device = device

        return self._encoder

    def encode(self, texts, show_progress=True):
        """
        Encode a list of texts into normalized embeddings.

        Args:
            texts: list of strings
            show_progress: show progress bar

        Returns:
            np.ndarray of shape (len(texts), model_dimension)
        """
        model_cfg = self.cfg.get('model', {})
        normalize = model_cfg.get('normalize', True)
        batch_size = model_cfg.get('batch_size', 64)
        max_length = model_cfg.get('max_length', 256)

        encoder = self._load_encoder()

        if self._encoder_type == 'sentence_transformers':
            embs = encoder.encode(
                texts, convert_to_tensor=False,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                batch_size=batch_size
            )
        else:
            import torch
            import torch.nn.functional as F
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self._encoder_tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=max_length, return_tensors='pt'
                ).to(self._encoder_device)
                with torch.no_grad():
                    outputs = self._encoder_model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1)
                    if normalize:
                        emb = F.normalize(emb, p=2, dim=1)
                    all_embs.append(emb.cpu().numpy())
            embs = np.vstack(all_embs) if all_embs else np.array([])

        return np.asarray(embs, dtype=np.float32)

    def build_from_texts(self, texts, show_progress=True):
        """
        Encode texts and build index in one call.

        Args:
            texts: list of strings
            show_progress: show encoding progress bar

        Returns:
            self
        """
        print(f"Encoding {len(texts)} texts with "
              f"{self.cfg.get('model', {}).get('name', 'default')}...")
        vectors = self.encode(texts, show_progress=show_progress)
        self.build(vectors)
        print(f"Done. {len(vectors)} vectors, {vectors.shape[1]}D")
        return self

    # ── Validation ────────────────────────────────────────────────

    def check_bounds(self, query):
        """
        Verify Cauchy-Schwarz bound guarantees.

        Returns:
            dict mapping "dD" -> violation count
        """
        if not self.is_valid():
            raise RuntimeError("Pipeline not built. Call .build(vectors) first.")

        query = np.asarray(query, dtype=np.float32).flatten()

        if hasattr(self.index, 'check_bounds'):
            violations = self.index.check_bounds(query)
            guarantee = all(v == 0 for v in violations.values())
            return {
                'violations': violations,
                'guarantee': 'PASS' if guarantee else 'FAIL',
            }

        raise AttributeError("Index type does not support bound checking")

    def benchmark(self, vectors, queries, k=10):
        """
        Run full benchmark with FAISS comparison.

        Args:
            vectors: corpus embeddings (N, D)
            queries: query embeddings (NQ, D)
            k: depth

        Returns:
            dict with benchmark results
        """
        from winnex_pipeline.validation.benchmark import run_benchmark

        # Ensure built on the same vectors
        self.build(vectors)
        return run_benchmark(vectors, queries, k=k, config=self.cfg)

    # ── Info ──────────────────────────────────────────────────────

    def info(self):
        """Print pipeline configuration and status."""
        from winnex_pipeline.core import _HAS_HMC
        print(f"{'=' * 60}")
        print(f"Winnex Pipeline v{self.cfg.get('version', '12.0.0')}")
        print(f"{'=' * 60}")
        print(f"  Index built:     {self.is_built}")
        if self.is_built:
            print(f"  Vectors:         {self.n_vectors} × {self.dim}D")
            print(f"  Method:          {type(self.index).__name__}")
            print(f"  Build time:      {getattr(self.index, 'build_time', 0):.3f}s")
        model_cfg = self.cfg.get('model', {})
        print(f"  Model:")
        print(f"    Name:          {model_cfg.get('name', 'default')}")
        print(f"    Dimension:     {model_cfg.get('dimension', '?')}D")
        print(f"    Device:        {model_cfg.get('device', 'cpu')}")
        print(f"  Config:")
        print(f"    Stage dims:    {self.cfg['dimensions']['stage_dims']}")
        qjl_dim = self.cfg['dimensions'].get('qjl_dim')
        qjl_status = f"✅ {self.raw_dim}→{qjl_dim}" if (qjl_dim and self.raw_dim != qjl_dim and
                     hasattr(self.index, 'qjl') and self.index.qjl is not None) else "❌ inactive"
        print(f"    QJL:           {qjl_status}")
        print(f"    Keep ratio:    [{self.cfg['search']['adaptive_keep_min']}, "
              f"{self.cfg['search']['adaptive_keep_max']}]")
        print(f"    Final k:       {self.cfg['search']['final_results']}")
        print(f"    Hybrid:        {self.cfg['hybrid']['enabled']}")
        print(f"    Modulation:    {self.cfg['modulation']['error_backprop']}")
        _has_faiss = False
        try:
            import faiss
            _has_faiss = True
        except ImportError:
            pass
        print(f"  Extras:")
        print(f"    HMC (PyTorch): {'available' if _HAS_HMC else 'not installed'}")
        print(f"    FAISS:         {'available' if _has_faiss else 'not installed'}")
