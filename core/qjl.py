"""
QJL: Johnson-Lindenstrauss Compression
========================================
Compresses high-dimensional embeddings (e.g. 384D SBERT) to a lower
dimension (e.g. 128D) with guaranteed distortion bounds.

This is the standard Winnex AI pipeline:
  Text → SBERT 384D → QJL 128D → Madhava Cascade

Reference: Winnex AI Zenodo 20856138 (O(K) Navigation Proof)
License: BSL 1.1 | pay@winnex.ai
"""
import math, time
import numpy as np


class QJLCompressor:
    """
    Johnson-Lindenstrauss compressor with QR-orthogonalized projection.

    Maps R^D_in → R^D_out with rows orthonormal, ensuring the Pythagorean
    theorem holds exactly for downstream Cauchy-Schwarz bound computation.

    Args:
        d_in: input dimension (e.g. 384 for all-MiniLM-L6-v2)
        d_out: output dimension (e.g. 128 for standard QJL)
        seed: random seed for reproducibility

    Usage:
        qjl = QJLCompressor(384, 128)
        vectors_128d = qjl.compress(vectors_384d)
        query_128d = qjl.compress_query(query_384d)
        # -> pass to MadhavaCore
    """

    def __init__(self, d_in=384, d_out=128, seed=42):
        self.d_in = d_in
        self.d_out = d_out
        self.rng = np.random.RandomState(seed)
        self.P = None
        self.build_time = 0.0

        # Build QR-orthogonalized projection matrix
        t0 = time.time()
        R = self.rng.randn(d_out, d_in).astype(np.float64).T
        Q, _ = np.linalg.qr(R)
        P = Q[:, :d_out].T.astype(np.float32)
        ortho_err = np.abs(P @ P.T - np.eye(d_out, dtype=np.float32)).max()
        assert ortho_err < 1e-5, f"QJL orthogonality failed: {ortho_err:.2e}"
        self.P = P
        self.ortho_err = ortho_err
        self.build_time = time.time() - t0

    def compress(self, vectors):
        """
        Compress batch of vectors: R^{N x D_in} → R^{N x D_out}

        Args:
            vectors: np.ndarray (N, D_in)

        Returns:
            np.ndarray (N, D_out)
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        compressed = vectors @ self.P.T
        return compressed.astype(np.float32)

    def compress_query(self, query):
        """
        Compress a single query vector: R^{D_in} → R^{D_out}

        Args:
            query: np.ndarray (D_in,) or (1, D_in)

        Returns:
            np.ndarray (D_out,)
        """
        query = np.asarray(query, dtype=np.float32).flatten()
        compressed = query @ self.P.T  # (D_in,) @ (D_in, D_out) = (D_out,)
        return compressed.astype(np.float32)

    def distortion(self, vectors, sample=1000):
        """
        Measure empirical JL distortion: max |||u-v||² - ||Pu-Pv||²| / ||u-v||².

        Args:
            vectors: np.ndarray (N, D_in) — sample to measure
            sample: number of pairs to check

        Returns:
            dict with max_distortion, mean_distortion, theoretical_bound
        """
        n = min(len(vectors), sample * 2)
        idxs = np.random.RandomState(0).choice(n, sample * 2, replace=False)
        pairs = [(idxs[i], idxs[i + sample]) for i in range(sample)]

        compressed = self.compress(vectors)

        distortions = []
        for i, j in pairs:
            orig_dist = np.linalg.norm(vectors[i] - vectors[j]) ** 2
            comp_dist = np.linalg.norm(compressed[i] - compressed[j]) ** 2
            if orig_dist > 1e-12:
                d = abs(comp_dist - orig_dist) / orig_dist
                distortions.append(d)

        eps = self._jl_bound()
        return {
            'max_distortion': float(np.max(distortions)),
            'mean_distortion': float(np.mean(distortions)),
            'theoretical_bound': eps,
            'n_pairs': len(distortions),
        }

    def _jl_bound(self):
        """Theoretical JL bound: ε = sqrt(4 * log(N) / d_out)."""
        # Standard JL lemma: for N points, d_out >= 4*log(N)/ε²
        # So ε ≈ sqrt(4*log(N)/d_out)
        return math.sqrt(4 * math.log(1e6) / self.d_out)  # N≈1M conservative
