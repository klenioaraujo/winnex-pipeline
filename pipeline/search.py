"""
Pipeline Search — Search Executor with Bound Report
=====================================================
"""
import numpy as np


def search(index, query, k=10, return_profile=False, check_bounds=False):
    """
    Execute search and optionally verify mathematical guarantees.

    Args:
        index: built search index (MadhavaCore, MadHybrid, HMCHierarchical)
        query: query vector, np.ndarray (D,)
        k: number of results
        return_profile: if True, include profiling info
        check_bounds: if True, verify bound violations

    Returns:
        indices: list/array of top-k indices
        OR (indices, report) if return_profile or check_bounds
    """
    from winnex_pipeline.core.madhava import MadhavaCore

    prof = {}
    if isinstance(index, MadhavaCore):
        result, inner_prof = index.search(query, k=k, return_profile=True)
        prof.update(inner_prof)
        prof['method'] = 'MadhavaCore'
    else:
        # MadHybrid or HMCHierarchical
        t0 = __import__('time').time()
        result = index.search(query, k=k)
        prof['latency_ms'] = (__import__('time').time() - t0) * 1000
        prof['method'] = type(index).__name__

    if isinstance(result, np.ndarray):
        result_list = result.tolist() if hasattr(result, 'tolist') else list(result)
    else:
        result_list = list(result)

    report = {
        'indices': result_list,
        'k': k,
        'n_found': len(result_list),
        **prof,
    }

    if check_bounds and hasattr(index, 'check_bounds'):
        viol = index.check_bounds(query)
        report['bound_violations'] = viol
        report['bound_guarantee'] = 'PASS' if all(v == 0 for v in viol.values()) else 'FAIL'

    if return_profile or check_bounds:
        return result_list, report
    return result_list


def search_batch(index, queries, k=10, return_profile=False, check_bounds=False):
    """
    Execute search over a batch of queries.

    Args:
        index: built search index
        queries: np.ndarray of shape (NQ, D)
        k: number of results
        return_profile: if True, include profiling info
        check_bounds: if True, verify bound violations

    Returns:
        list of result lists
        OR (results, aggregate_report) if return_profile
    """
    import time
    times = []
    all_results = []
    all_reports = []

    for qi in range(len(queries)):
        t0 = time.time()
        res, rep = search(index, queries[qi], k=k, return_profile=True,
                          check_bounds=check_bounds)
        elapsed = (time.time() - t0) * 1000
        times.append(elapsed)
        all_results.append(res)
        all_reports.append(rep)

    agg = {
        'method': all_reports[0].get('method', '?'),
        'n_queries': len(queries),
        'latency_ms_mean': float(np.mean(times)),
        'latency_ms_std': float(np.std(times)),
        'latency_ms_median': float(np.median(times)),
        'latency_ms_min': float(np.min(times)),
        'latency_ms_max': float(np.max(times)),
    }

    if check_bounds and 'bound_violations' in all_reports[0]:
        total_viol = {}
        for rep in all_reports:
            for k, v in rep['bound_violations'].items():
                total_viol[k] = total_viol.get(k, 0) + v
        agg['bound_violations_total'] = total_viol
        agg['bound_guarantee'] = 'PASS' if all(v == 0 for v in total_viol.values()) else 'FAIL'

    if return_profile:
        return all_results, agg
    return all_results
