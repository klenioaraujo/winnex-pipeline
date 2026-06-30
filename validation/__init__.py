"""
Winnex Pipeline — Validation Suite
====================================
Ground truth computation, metrics, and benchmark comparisons.
"""
from winnex_pipeline.validation.ground_truth import build_ground_truth
from winnex_pipeline.validation.metrics import ndcg_at_k, recall_at_k, compute_metrics
from winnex_pipeline.validation.benchmark import run_benchmark

__all__ = ['build_ground_truth', 'ndcg_at_k', 'recall_at_k',
           'compute_metrics', 'run_benchmark']
