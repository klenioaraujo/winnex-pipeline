"""
PiPrimeAnchors: SVD + Gram-Schmidt Anchor Computation
=======================================================
K anchors on the unit sphere S^(d-1), orthogonalized via Gram-Schmidt.

The anchors define the navigation space for HMC search.
Dimension D estimated via von Neumann entropy of singular values.

Reference: hmc_hierarchical.py, attack.py (Zenodo 20856138)
License: BSL 1.1 | pay@winnex.ai
"""
import math
import numpy as np


def _sieve(n):
    """Sieve of Eratosthenes."""
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]


class PiPrimeAnchors:
    """
    Pi-Prime Anchors: K structural anchors via SVD + Gram-Schmidt.

    Args:
        E: corpus embeddings, np.ndarray (N, D)
        n_primes: number of anchors (default: 8)
        seed: random seed for reproducibility

    Attributes:
        K: number of anchors
        anchors: np.ndarray (K, D) — orthonormal anchor vectors
        best_anchor: np.ndarray (N,) — anchor assignment per vector
        D_int: intrinsic dimension estimate (von Neumann entropy)
        ap: anchor weights
    """

    def __init__(self, E, n_primes=8, seed=42):
        self.n_primes = n_primes
        N, dim = E.shape
        self.dim = dim
        self.seed = seed

        # Sieve primes (used in original formulation)
        primes = _sieve(100)[:n_primes]
        self.pf = [math.pi * p for p in primes]  # kept for API compat

        # ── Singular value analysis (Eq. 13) ──
        if N > 100000:
            from sklearn.utils.extmath import randomized_svd
            _, s, _ = randomized_svd(E.astype(np.float64), n_components=min(dim, 256),
                                      n_oversamples=10, random_state=seed)
        else:
            _, s, _ = np.linalg.svd(E.astype(np.float64), full_matrices=False)

        # von Neumann entropy -> intrinsic dimension
        e2 = np.maximum(s ** 2, 1e-15)
        e2 /= e2.sum() + 1e-15
        S_vn = -float(np.sum(e2 * np.log(e2 + 1e-15)))
        D_int = S_vn / math.log(N) if N > 1 else 1.0
        self.D = float(np.clip(0.5 + 1.5 * D_int, 0.5, 2.0))
        self.D_int = D_int

        # Anchor weights from primes
        self.ap = [float(np.clip(1.0 * (1 + (self.D - 1)) * math.log(p + 1) / math.log(3),
                                 0.1, 3.0))
                   for p in primes]

        # ── SVD seeds ──
        if N > 1000000:
            from sklearn.utils.extmath import randomized_svd
            U, s, Vt = randomized_svd(E - E.mean(0), n_components=min(n_primes + 5, dim),
                                       n_oversamples=5, random_state=seed)
        else:
            U, s, Vt = np.linalg.svd(E - E.mean(0), full_matrices=False)

        rng = np.random.RandomState(seed)
        seeds = [E.mean(0).copy()]
        for i in range(min(n_primes - 1, len(Vt))):
            pc = Vt[i].copy().astype(np.float32)
            noise = rng.randn(dim).astype(np.float32) * 1e-4
            noise -= np.dot(noise, pc) * pc
            pc += noise
            seeds.append(pc)
        while len(seeds) < n_primes:
            seeds.append(rng.randn(dim).astype(np.float32))

        # ── Gram-Schmidt orthogonalization ──
        anch = []
        for v in seeds[:n_primes]:
            v = v.copy()
            for a in anch:
                v -= np.dot(v, a) * a
            nrm = np.linalg.norm(v)
            if nrm > 1e-9:
                anch.append(v / nrm)
            else:
                x = rng.randn(dim).astype(np.float32)
                for a in anch:
                    x -= np.dot(x, a) * a
                nx = np.linalg.norm(x)
                anch.append(x / nx if nx > 1e-9 else np.eye(dim)[len(anch)] / dim ** 0.5)

        self.anchors = np.array(anch[:n_primes], dtype=np.float32)
        self.K = len(self.anchors)

        # ── Assign each vector to nearest anchor ──
        scores = E.astype(np.float32) @ self.anchors.T
        self.best_anchor = np.argmax(scores, axis=1)

        # Anchor weights: log-frequency
        aw = np.zeros(n_primes, dtype=np.float32)
        for i in range(N):
            aw[self.best_anchor[i]] += 1.0
        self.anchor_weights = np.log1p(aw)

    def potential_energy(self, q, qry, temp=0.5):
        """
        U(a) = 0.7·sim(a,q) + 0.3·repulsion(a, other_anchors)

        Args:
            q: anchor vector
            qry: query vector
            temp: temperature scaling

        Returns:
            float: potential energy
        """
        sim = -float(np.dot(q, qry)) / temp
        tw = sum(self.ap)
        frac = 0.0
        for ap_i, pp_i, a in zip(self.ap, self.pf, self.anchors):
            d = float(np.linalg.norm(q - a)) + 1e-9
            frac += (ap_i / tw) * math.log(1 + 1 / d)
        return 0.7 * sim + 0.3 * (-0.1 * frac)

    def gradient(self, q, qry, temp=0.5):
        """
        Gradient of potential energy.

        Args:
            q: anchor vector
            qry: query vector
            temp: temperature scaling

        Returns:
            np.ndarray: gradient vector
        """
        sim = float(np.dot(q, qry))
        g = 0.7 * (-qry + sim * q) / temp
        tw = sum(self.ap)
        for ap_i, pp_i, a in zip(self.ap, self.pf, self.anchors):
            diff = q - a
            r = float(np.linalg.norm(diff)) + 1e-9
            g += 0.03 * (ap_i / tw) * diff / (r * (r + 0.1) * (r + 1.1) + 1e-9)
        return g
