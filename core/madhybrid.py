"""
MadHybrid: Clustered Index + Madhava per Cell
===============================================
IVF-style partitioning with independent MadhavaCore per Voronoi cell.

Validated (SIFT-1M):
  MadHybrid(np=15):  R@10=0.993, 2.04ms, 0% bound violations
  MadHybrid(np=10):  R@10=0.982, 1.38ms
  Rebuild: ~5s (vs HNSW 1.4s at 100K, faster above 1M)

Reference: madhava_v12.py (Zenodo 10.5281/zenodo.21066971)
License: BSL 1.1 | pay@winnex.ai
"""
import time
import numpy as np
from winnex_pipeline.config import load_config
from winnex_pipeline.core.madhava import MadhavaCore


class MadHybrid:
    """
    IVF-style clustering + independent MadhavaCore per cell.

    Args:
        config: dict or config path

    Usage:
        mh = MadHybrid()
        mh.build(vectors)
        results = mh.search(query, k=10, n_probe=5)
    """

    def __init__(self, config=None):
        self.cfg = config if isinstance(config, dict) else load_config(config)
        self.nc = self.cfg['hybrid']['n_cells']
        self.clust_cfg = self.cfg['hybrid']['clustering']

        # State
        self.vectors = None
        self.n = 0
        self.centroids = None
        self.cells = {}        # cell_id -> MadhavaCore
        self.members = {}      # cell_id -> global indices
        self.labels_ = None    # cluster assignment for each vector
        self.build_time = 0.0

    def _cluster(self, vectors):
        """Run MiniBatchKMeans clustering."""
        from sklearn.cluster import MiniBatchKMeans
        bs = min(self.clust_cfg['batch_size'], len(vectors))
        km = MiniBatchKMeans(
            n_clusters=self.nc,
            random_state=self.clust_cfg['random_state'],
            batch_size=bs,
            n_init=self.clust_cfg['n_init'],
            max_iter=self.clust_cfg['max_iter'],
        )
        labs = km.fit_predict(vectors)
        return km.cluster_centers_.astype(np.float32), labs

    def build(self, vectors):
        """
        Cluster vectors and train one MadhavaCore per cell.

        Args:
            vectors: np.ndarray of shape (N, D)
        """
        t0 = time.time()
        self.vectors = vectors
        self.n = len(vectors)

        self.centroids, self.labels_ = self._cluster(vectors)

        self.cells = {}
        self.members = {}
        for cid in range(self.nc):
            idxs = np.where(self.labels_ == cid)[0]
            if len(idxs) == 0:
                continue
            self.members[cid] = idxs
            mc = MadhavaCore(self.cfg)
            mc.build(vectors[idxs])
            self.cells[cid] = mc

        self.build_time = time.time() - t0
        return self

    def search(self, q, k=None, n_probe=None):
        """
        Search over top-n_probe cells closest to query.

        Args:
            q: query vector, np.ndarray (D,)
            k: number of results
            n_probe: number of cells to probe (default: config[0])

        Returns:
            list of top-k global indices
        """
        if n_probe is None:
            n_probe = self.cfg['hybrid']['n_probe'][0]
        if k is None:
            k = self.cfg['search']['final_results']

        q = q.astype(np.float32).flatten()
        sims = self.centroids @ q
        top_cells = np.argsort(-sims)[:n_probe]

        candidates = []  # (global_idx, score)
        for cid in top_cells:
            mc = self.cells.get(cid)
            if mc is None or mc.n == 0:
                continue
            idxs = mc.search(q, k=min(k * 2, mc.n))
            scores = mc.vectors[idxs].astype(np.float64) @ q.astype(np.float64)
            for local_i, score in zip(idxs, scores):
                candidates.append((int(self.members[cid][local_i]), float(score)))

        # Sort globally
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:k]]

    def check_bounds(self, q):
        """
        Aggregate bound violations across all cells.

        Args:
            q: query vector, np.ndarray (D,)

        Returns:
            dict mapping dim -> total violation count
        """
        q = q.astype(np.float64).flatten()
        agg = {}
        for cid, mc in self.cells.items():
            cell_viol = mc.check_bounds(q)
            for k, v in cell_viol.items():
                agg[k] = agg.get(k, 0) + v
        return agg

    def search_profile(self, q, k=None, n_probe=None):
        """
        Search with profiling breakdown.

        Returns:
            dict with timing profile
        """
        t0 = time.time()
        results = self.search(q, k, n_probe)
        latency = (time.time() - t0) * 1000

        return {
            'method': 'MadHybrid',
            'n_cells': self.nc,
            'n_probe': n_probe or self.cfg['hybrid']['n_probe'][0],
            'latency_ms': round(latency, 3),
            'n_candidates': len(results),
            'n_results': len(results),
            'total_ms': round(latency, 3),
            'stage1_ms': 0,
            'stage2_ms': 0,
            'exact_ms': 0,
            'n_candidates_stage1': 0,
            'n_candidates_stage2': 0,
            'n_final': len(results),
            'prune_ratio': 0,
            'n_total': len(self.vectors) if hasattr(self, 'vectors') else 0,
        }
