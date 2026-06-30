"""
Pipeline Build — Index Factory
================================
Unified builder that selects the appropriate index type.
"""
import numpy as np
from winnex_pipeline.config import load_config


def build_index(vectors, method='auto', config=None):
    """
    Build a search index over the given vectors.

    Args:
        vectors: np.ndarray of shape (N, D)
        method: one of 'auto', 'madhava', 'madhybrid', 'hmc'
        config: dict or path to JSON config

    Returns:
        index object with .search(q, k) method

    Raises:
        ValueError: if method is unknown
    """
    cfg = config if isinstance(config, dict) else load_config(config)

    if method == 'auto':
        # Auto-select based on config
        if cfg['hybrid']['enabled'] and len(vectors) > 5000:
            method = 'madhybrid'
        else:
            method = 'madhava'

    if method == 'madhava':
        from winnex_pipeline.core.madhava import MadhavaCore
        idx = MadhavaCore(cfg)
        idx.build(vectors)
        return idx

    elif method == 'madhybrid':
        from winnex_pipeline.core.madhybrid import MadHybrid
        idx = MadHybrid(cfg)
        idx.build(vectors)
        return idx

    elif method == 'hmc':
        from winnex_pipeline.core.hmc import HMCHierarchical
        idx = HMCHierarchical(
            dim=cfg['dimensions']['input_dim'],
            n_a=min(cfg.get('hmc', {}).get('n_anchors', 8),
                     max(4, len(vectors) // 200)),
            n_sub=cfg.get('hmc', {}).get('n_sub', 4),
        )
        idx.build(vectors)
        return idx

    else:
        raise ValueError(f"Unknown method: {method}. Options: auto, madhava, madhybrid, hmc")


def list_methods():
    """Return list of available search methods."""
    methods = ['madhava', 'madhybrid']
    try:
        from winnex_pipeline.core.hmc import HMCHierarchical
        methods.append('hmc')
    except ImportError:
        pass
    return methods
