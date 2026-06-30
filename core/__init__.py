"""
Winnex Pipeline Core
====================
Core search algorithms validated by benchmarks:
  - MadhavaCore:    QR-JL orthogonal projection cascade (32D/64D/128D)
  - MadHybrid:      IVF clustering + Madhava per cell
  - PiPrimeAnchors: SVD + Gram-Schmidt anchor computation
  - HMCHierarchical: Riemannian HMC on S^(d-1) with local buckets
"""

from winnex_pipeline.core.madhava import MadhavaCore
from winnex_pipeline.core.madhybrid import MadHybrid

try:
    from winnex_pipeline.core.anchors import PiPrimeAnchors
    from winnex_pipeline.core.hmc import HMCHierarchical, BucketHMC
    _HAS_HMC = True
except ImportError:
    _HAS_HMC = False

__all__ = [
    'MadhavaCore', 'MadHybrid',
    'PiPrimeAnchors', 'HMCHierarchical', 'BucketHMC',
    '_HAS_HMC',
]
