"""
MadhavaCore: QR-Orthogonalized JL Cascade
===========================================
Stage-wise progressive refinement with Cauchy-Schwarz upper bounds.

Validated (SIFT-1M, 50K vectors, 200 queries):
  [64,128]: R@10=1.000, 0.97ms, 0% bound violations
  [32,64]:  R@10=0.993, 1.02ms, 0% bound violations

Reference: madhava_v12.py (Zenodo 10.5281/zenodo.21066971)
License: BSL 1.1 | pay@winnex.ai
"""
import math, time
import numpy as np
from winnex_pipeline.config import load_config


class MadhavaCore:
    """
    Core Madhava search unit. Configurable stage dimensions.

    Args:
        config: dict or config path. Default: config/base.json

    Usage:
        mc = MadhavaCore()
        mc.build(vectors)  # np.ndarray (N, D)
        results = mc.search(query)  # np.ndarray (D,) -> top-k indices
    """

    def __init__(self, config=None):
        self.cfg = config if isinstance(config, dict) else load_config(config)
        self.dims = self.cfg['dimensions']['stage_dims']
        self.full_dim = self.cfg['dimensions']['input_dim']
        self.s = self.cfg['search']
        self.b = self.cfg['bounds']
        self.m = self.cfg['modulation']
        self.rng = np.random.RandomState(43)
        self.gamma = 0.0  # modulation bias (QR-JL feature)

        # State
        self.vectors = None
        self.n = 0
        self.norms = None
        self.proj_matrices = {}   # d_out -> P (d_out x D)
        self.proj_L = {}          # d_out -> projected vectors (N x d_out)
        self.error = {}           # d_out -> Pythagorean residual (N,)
        self.build_time = 0.0
        self._enable_gamma = False

    # ── Config Getters ────────────────────────────────────────────

    def _make_orthogonal_proj(self, d_out):
        """QR-orthogonalized random projection: R^D -> R^{d_out}, rows orthonormal."""
        R = self.rng.randn(d_out, self.full_dim).astype(np.float64).T
        Q, _ = np.linalg.qr(R)
        P = Q[:, :d_out].T.astype(np.float64)
        err = np.abs(P @ P.T - np.eye(d_out)).max()
        assert err < self.b['orthogonality_tolerance'], \
            f"QR orthogonality failed: {err:.2e} > {self.b['orthogonality_tolerance']}"
        return P

    # ── Build ─────────────────────────────────────────────────────

    def build(self, vectors):
        """
        Precompute QR-JL projections and Pythagorean residuals.

        Args:
            vectors: np.ndarray of shape (N, D)
        """
        t0 = time.time()
        self.vectors = vectors.astype(np.float64)
        self.n = len(vectors)
        self.norms = np.linalg.norm(self.vectors, axis=1)

        for d in self.dims:
            P = self._make_orthogonal_proj(d)
            self.proj_matrices[d] = P
            # Project vectors to d-dim space
            proj = (vectors.astype(np.float32) @ P.T.astype(np.float32)).astype(np.float64)
            self.proj_L[d] = proj
            # Pythagorean residual: error_d = sqrt(||v||^2 - ||Pv||^2)
            captured = np.linalg.norm(proj, axis=1)
            self.error[d] = np.sqrt(np.maximum(self.norms**2 - captured**2, 0))

        self.build_time = time.time() - t0
        return self

    # ── Bounds ────────────────────────────────────────────────────

    def _upper_bound(self, pv, ev, pq, eq):
        """
        Cauchy-Schwarz upper bound:
          <v,q> <= <Pv,Pq> + ||v - P^T_Pv|| . ||q - P^T Pq|| + epsilon
        """
        eps = self.b['cauchy_schwarz_epsilon']
        return pv @ pq + ev * eq + eps

    # ── Search ────────────────────────────────────────────────────

    def search(self, q, k=None, return_profile=False):
        """
        Execute Madhava cascade search.

        Args:
            q: query vector, np.ndarray of shape (D,)
            k: number of results (default: config final_results)
            return_profile: if True, return (indices, profile_dict)

        Returns:
            indices: np.ndarray of top-k indices
            OR (indices, profile) if return_profile=True
        """
        if k is None:
            k = self.s['final_results']
        if self.n == 0:
            return np.array([], dtype=int)

        t_start = time.time()
        q = q.astype(np.float64).flatten()
        qn = np.linalg.norm(q)
        prof = {'n_total': self.n, 'stage_dims': self.dims}

        # ── Stage 1: lowest dimension on ALL N vectors ──
        d1 = self.dims[0]
        q1 = (q.astype(np.float32) @ self.proj_matrices[d1].T.astype(np.float32)).astype(np.float64)
        qr1 = math.sqrt(max(0, qn**2 - np.linalg.norm(q1)**2))
        B1 = self._upper_bound(self.proj_L[d1], self.error[d1], q1, qr1)

        # Adaptive keep based on bound range
        b_range = float(B1.max() - B1.min())
        raw_keep = self.s['adaptive_keep_base'] * self.s['adaptive_bounds_sensitivity'] / max(b_range, 0.01)
        adapt_k = min(self.s['adaptive_keep_max'],
                      max(self.s['adaptive_keep_min'], raw_keep))
        k1 = min(max(int(self.n * adapt_k), 100), self.n)
        if self.n <= k1:
            idx1 = np.arange(self.n)
        else:
            idx1 = np.argpartition(-B1, k1 - 1)[:k1]

        prof['stage1_keep'] = len(idx1)
        prof['stage1_prune_ratio'] = 1.0 - len(idx1) / max(self.n, 1)
        prof['stage1_bound_range'] = [float(B1.min()), float(B1.max())]

        # ── Stage 2: higher dimension refinement ──
        d2 = self.dims[1]
        q2 = (q.astype(np.float32) @ self.proj_matrices[d2].T.astype(np.float32)).astype(np.float64)
        qr2 = math.sqrt(max(0, qn**2 - np.linalg.norm(q2)**2))

        B2 = self._upper_bound(self.proj_L[d2][idx1], self.error[d2][idx1], q2, qr2)

        # Error backpropagation modulation
        if self.m['error_backprop']:
            e1 = self.error[d1][idx1]
            e2 = self.error[d2][idx1]
            alpha = 1.0 / (1.0 + np.exp(
                -(e1 - e2) / max(np.mean(e1), 1e-9) * self.m['alpha_smoothing']
            ))
            alpha = np.clip(alpha, self.m['alpha_min'], self.m['alpha_max'])
            scores = B1[idx1] + alpha * (B2 - B1[idx1])
            prof['alpha_mean'] = float(np.mean(alpha))
        else:
            scores = B2

        # ── Stage 3: exact cosine on survivors ──
        k2 = min(self.s['stage2_topk'], len(idx1))
        idx2 = idx1[np.argpartition(-scores, k2 - 1)[:k2]]

        cos = self.vectors[idx2].astype(np.float64) @ q
        final = idx2[np.argsort(-cos)[:k]]

        prof['stage2_candidates'] = len(idx2)
        prof['n_final'] = len(final)
        prof['latency_ms'] = (time.time() - t_start) * 1000

        if return_profile:
            return final, prof
        return final

    def set_gamma(self, gamma):
        """Set QR-JL modulation bias (nonzero for QR-JL variant)."""
        self.gamma = gamma
        self._enable_gamma = True


    # ── Bound Verification ────────────────────────────────────────

    def check_bounds(self, q):
        """
        Count bound violations for each projection dimension.

        A violation occurs when true cosine exceeds the upper bound.
        Returns dict mapping "dD" -> violation_count.
        """
        q = q.astype(np.float64).flatten()
        qn = np.linalg.norm(q)
        true_cos = self.vectors.astype(np.float64) @ q
        viol = {}
        for d in self.dims:
            P = self.proj_matrices[d]
            qd = (q.astype(np.float32) @ P.T.astype(np.float32)).astype(np.float64)
            qr = math.sqrt(max(0, qn**2 - np.linalg.norm(qd)**2))
            ub = self._upper_bound(self.proj_L[d], self.error[d], qd, qr)
            viol[f"{d}D"] = int(np.sum(true_cos > ub + 1e-9))
        return viol
