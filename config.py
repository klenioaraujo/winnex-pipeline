"""
Config Loader for Winnex Pipeline.

Supports JSON config files with deep merge of defaults.
Usage:
    from config import load_config
    cfg = load_config('config/base.json')
    cfg = load_config()  # loads default base.json
"""
import json, os

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'base.json')


def _default_config():
    return {
        "version": "12.2.0",
        "model": {
            "name": "all-MiniLM-L6-v2",
            "dimension": 384,
            "device": "cpu",
            "normalize": True,
            "max_length": 256,
            "batch_size": 64
        },
        "dimensions": {
            "input_dim": 384,
            "stage_dims": [32, 64],
            "qjl_dim": 128
        },
        "search": {
            "adaptive_keep_base": 0.25,
            "adaptive_keep_min": 0.05,
            "adaptive_keep_max": 0.50,
            "adaptive_bounds_sensitivity": 0.12,
            "stage2_topk": 500,
            "stage2_topk_max": 2000,
            "final_results": 10,
            "epsilon": 1e-5
        },
        "hybrid": {
            "enabled": False,
            "n_cells": 64,
            "n_probe": [3, 5, 8, 10, 15],
            "clustering": {
                "algorithm": "MiniBatchKMeans",
                "random_state": 42,
                "batch_size": 20000,
                "n_init": 3,
                "max_iter": 50
            }
        },
        "bounds": {
            "cauchy_schwarz_epsilon": 1e-5,
            "orthogonality_tolerance": 1e-5
        },
        "modulation": {
            "error_backprop": True,
            "alpha_smoothing": 0.5,
            "alpha_min": 0.01,
            "alpha_max": 0.99
        }
    }


def _deep_merge(base, override):
    """Recursive dict merge. override values win."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path=None):
    """
    Load a JSON config file merged with defaults.

    Args:
        path: Path to JSON config file. If None, loads config/base.json.
              If relative, resolved from winnex-pipeline/ directory.

    Returns:
        dict with all config parameters
    """
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    elif not os.path.isabs(path):
        # Resolve relative to winnex-pipeline/ directory
        path = os.path.join(os.path.dirname(__file__), path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        cfg = json.load(f)

    defaults = _default_config()
    merged = _deep_merge(defaults, cfg)
    return merged
