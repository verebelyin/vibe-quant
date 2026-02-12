"""Performance benchmarks for math-heavy bottleneck functions.

Run with: python3.13 tests/benchmarks/bench_math_perf.py
"""

from __future__ import annotations

import math
import random
import time
from functools import lru_cache

import numpy as np


def _time_it(func, *args, iterations=1000, **kwargs):
    """Time a function call, return avg time in microseconds."""
    # Warmup
    for _ in range(min(10, iterations)):
        func(*args, **kwargs)

    start = time.perf_counter()
    for _ in range(iterations):
        func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1e6  # microseconds


# ============================================================
# 1. Pareto Front: NumPy vectorized vs Python loop
# ============================================================

def pareto_front_python(objectives_list):
    """Original Python-loop Pareto front."""
    n = len(objectives_list)
    is_pareto = [True] * n
    for i in range(n):
        if not is_pareto[i]:
            continue
        obj_i = objectives_list[i]
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            obj_j = objectives_list[j]
            j_ge_i = all(obj_j[k] >= obj_i[k] for k in range(3))
            j_gt_i_any = any(obj_j[k] > obj_i[k] for k in range(3))
            if j_ge_i and j_gt_i_any:
                is_pareto[i] = False
                break
    return [i for i in range(n) if is_pareto[i]]


def pareto_front_numpy_loop(objectives_array):
    """NumPy with Python loop (per-point vectorization)."""
    n = len(objectives_array)
    is_pareto = np.ones(n, dtype=np.bool_)
    for i in range(n):
        if not is_pareto[i]:
            continue
        diff = objectives_array[is_pareto] - objectives_array[i]
        all_ge = np.all(diff >= 0, axis=1)
        any_gt = np.any(diff > 0, axis=1)
        dominators = all_ge & any_gt
        if np.any(dominators):
            is_pareto[i] = False
    return np.flatnonzero(is_pareto).tolist()


def pareto_front_numpy_full(objectives_array):
    """Fully vectorized NumPy (no Python loops)."""
    diff = objectives_array[:, np.newaxis, :] - objectives_array[np.newaxis, :, :]
    all_ge = np.all(diff >= 0, axis=2)
    any_gt = np.any(diff > 0, axis=2)
    dominates = all_ge & any_gt
    np.fill_diagonal(dominates, False)
    is_dominated = np.any(dominates, axis=0)
    return np.flatnonzero(~is_dominated).tolist()


def pareto_front_python_inlined(objectives_list):
    """Python with inlined comparisons (no tuple/zip overhead)."""
    n = len(objectives_list)
    is_pareto = [True] * n
    for i in range(n):
        if not is_pareto[i]:
            continue
        si, di, pi = objectives_list[i]
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            sj, dj, pj = objectives_list[j]
            if (sj >= si and dj >= di and pj >= pi
                    and (sj > si or dj > di or pj > pi)):
                is_pareto[i] = False
                break
    return [i for i in range(n) if is_pareto[i]]


# ============================================================
# 2. DSR: scipy.stats.norm.sf vs math.erfc
# ============================================================

def dsr_pvalue_scipy(z):
    from scipy import stats as scipy_stats
    return float(scipy_stats.norm.sf(z))


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def dsr_pvalue_erfc(z):
    return 0.5 * math.erfc(z * _INV_SQRT2)


# ============================================================
# 3. Expected max Sharpe: cached vs uncached
# ============================================================

EULER_MASCHERONI = 0.5772156649015329
_EULER_MASCHERONI_SQ = EULER_MASCHERONI * EULER_MASCHERONI


def expected_max_sharpe_uncached(num_trials):
    if num_trials <= 1:
        return 0.0
    log_n = math.log(num_trials)
    base = math.sqrt(2 * log_n)
    correction = 1 - EULER_MASCHERONI / log_n
    if log_n > 0:
        correction += (EULER_MASCHERONI**2) / (2 * log_n**2)
    return base * correction


@lru_cache(maxsize=1024)
def expected_max_sharpe_cached(num_trials):
    if num_trials <= 1:
        return 0.0
    log_n = math.log(num_trials)
    base = math.sqrt(2.0 * log_n)
    inv_log_n = 1.0 / log_n
    correction = 1.0 - EULER_MASCHERONI * inv_log_n + _EULER_MASCHERONI_SQ * 0.5 * inv_log_n * inv_log_n
    return base * correction


# ============================================================
# 4. Fitness score: function call vs inlined
# ============================================================

SHARPE_MIN = -1.0
SHARPE_MAX = 4.0
PF_MIN = 0.0
PF_MAX = 5.0
SHARPE_WEIGHT = 0.4
DRAWDOWN_WEIGHT = 0.3
PROFIT_FACTOR_WEIGHT = 0.3
_SHARPE_INV_RANGE = 1.0 / (SHARPE_MAX - SHARPE_MIN)
_PF_INV_RANGE = 1.0 / (PF_MAX - PF_MIN)


def fitness_score_original(sharpe, max_dd, pf):
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))
    def _normalize(v, lo, hi):
        if hi <= lo:
            return 0.0
        return (v - lo) / (hi - lo)
    sn = _normalize(_clamp(sharpe, SHARPE_MIN, SHARPE_MAX), SHARPE_MIN, SHARPE_MAX)
    dn = 1.0 - _clamp(max_dd, 0.0, 1.0)
    pn = _normalize(_clamp(pf, PF_MIN, PF_MAX), PF_MIN, PF_MAX)
    return SHARPE_WEIGHT * sn + DRAWDOWN_WEIGHT * dn + PROFIT_FACTOR_WEIGHT * pn


def fitness_score_inlined(sharpe, max_dd, pf):
    s = sharpe
    if s < SHARPE_MIN:
        s = SHARPE_MIN
    elif s > SHARPE_MAX:
        s = SHARPE_MAX
    sn = (s - SHARPE_MIN) * _SHARPE_INV_RANGE

    dd = max_dd
    if dd < 0.0:
        dd = 0.0
    elif dd > 1.0:
        dd = 1.0
    dn = 1.0 - dd

    p = pf
    if p < PF_MIN:
        p = PF_MIN
    elif p > PF_MAX:
        p = PF_MAX
    pn = (p - PF_MIN) * _PF_INV_RANGE

    return SHARPE_WEIGHT * sn + DRAWDOWN_WEIGHT * dn + PROFIT_FACTOR_WEIGHT * pn


# ============================================================
# 5. Aggregation: multi-pass vs single-pass
# ============================================================

def aggregate_multipass(data):
    """Original multi-pass aggregation."""
    sharpes = [d[0] for d in data]
    returns = [d[1] for d in data]
    is_returns = [d[2] for d in data]
    mean_sharpe = sum(sharpes) / len(sharpes)
    mean_return = sum(returns) / len(returns)
    mean_is = sum(is_returns) / len(is_returns)
    variance = sum((s - mean_sharpe) ** 2 for s in sharpes) / (len(sharpes) - 1)
    profitable = sum(1 for r in returns if r > 0)
    return mean_sharpe, mean_return, mean_is, variance, profitable


def aggregate_singlepass(data):
    """Optimized single-pass aggregation."""
    n = len(data)
    inv_n = 1.0 / n
    sum_s = 0.0
    sum_r = 0.0
    sum_is = 0.0
    sum_s_sq = 0.0
    profitable = 0
    for s, r, is_r in data:
        sum_s += s
        sum_s_sq += s * s
        sum_r += r
        sum_is += is_r
        if r > 0:
            profitable += 1
    mean_s = sum_s * inv_n
    mean_r = sum_r * inv_n
    mean_is = sum_is * inv_n
    variance = (sum_s_sq - n * mean_s * mean_s) / (n - 1)
    return mean_s, mean_r, mean_is, variance, profitable


# ============================================================
# Run benchmarks
# ============================================================

def main():
    print("=" * 70)
    print("PERFORMANCE BENCHMARKS: Math Optimization")
    print("=" * 70)
    print()

    # 1. Pareto Front
    for n in [50, 200, 500, 1000, 2000]:
        random.seed(42)
        objectives_list = [(random.random() * 3, random.random(), random.random() * 5) for _ in range(n)]
        objectives_array = np.array(objectives_list)

        iters = max(1, 2000 // n)
        t_py_orig = _time_it(pareto_front_python, objectives_list, iterations=iters)
        t_py_inline = _time_it(pareto_front_python_inlined, objectives_list, iterations=iters)
        t_np_full = _time_it(pareto_front_numpy_full, objectives_array, iterations=iters)
        sp_inline = t_py_orig / t_py_inline if t_py_inline > 0 else float("inf")
        sp_np = t_py_orig / t_np_full if t_np_full > 0 else float("inf")
        print(f"Pareto (n={n:>4}): Orig={t_py_orig:>9.0f}μs  Inlined={t_py_inline:>9.0f}μs({sp_inline:.1f}x)  NP-Full={t_np_full:>9.0f}μs({sp_np:.1f}x)")

    print()

    # 2. DSR p-value
    z_values = [1.5, 2.0, 2.5, -1.0, 0.5]
    t_scipy = _time_it(lambda: [dsr_pvalue_scipy(z) for z in z_values], iterations=1000)
    t_erfc = _time_it(lambda: [dsr_pvalue_erfc(z) for z in z_values], iterations=1000)
    speedup = t_scipy / t_erfc if t_erfc > 0 else float("inf")
    print(f"DSR p-value (5 calls): scipy={t_scipy:>10.1f}μs  erfc={t_erfc:>10.1f}μs  Speedup={speedup:.1f}x")

    # Validate accuracy
    for z in z_values:
        s = dsr_pvalue_scipy(z)
        e = dsr_pvalue_erfc(z)
        diff = abs(s - e)
        assert diff < 1e-12, f"Accuracy mismatch at z={z}: scipy={s}, erfc={e}, diff={diff}"
    print("  -> Accuracy verified: max absolute error < 1e-12")
    print()

    # 3. Expected max Sharpe
    # Reset cache
    expected_max_sharpe_cached.cache_clear()
    trial_values = [10, 50, 100, 200, 500, 1000]

    # Uncached
    t_uncached = _time_it(lambda: [expected_max_sharpe_uncached(t) for t in trial_values], iterations=5000)
    # First call (cold cache)
    expected_max_sharpe_cached.cache_clear()
    _time_it(lambda: [expected_max_sharpe_cached(t) for t in trial_values], iterations=1)
    # Warm cache
    t_cached = _time_it(lambda: [expected_max_sharpe_cached(t) for t in trial_values], iterations=5000)
    speedup_warm = t_uncached / t_cached if t_cached > 0 else float("inf")
    print(f"Expected max Sharpe (6 trial vals): uncached={t_uncached:>8.1f}μs  cached_warm={t_cached:>8.1f}μs  Speedup={speedup_warm:.1f}x")
    print()

    # 4. Fitness score
    test_cases = [(1.5, 0.1, 2.0), (-0.5, 0.3, 0.8), (3.0, 0.05, 4.5), (0.0, 0.5, 1.0)]
    t_orig = _time_it(lambda: [fitness_score_original(s, d, p) for s, d, p in test_cases], iterations=10000)
    t_inlined = _time_it(lambda: [fitness_score_inlined(s, d, p) for s, d, p in test_cases], iterations=10000)
    speedup = t_orig / t_inlined if t_inlined > 0 else float("inf")
    print(f"Fitness score (4 calls): original={t_orig:>8.1f}μs  inlined={t_inlined:>8.1f}μs  Speedup={speedup:.1f}x")

    # Validate accuracy
    for s, d, p in test_cases:
        o = fitness_score_original(s, d, p)
        i = fitness_score_inlined(s, d, p)
        assert abs(o - i) < 1e-10, f"Mismatch: original={o}, inlined={i}"
    print("  -> Accuracy verified: exact match")
    print()

    # 5. Aggregation
    random.seed(42)
    data = [(random.gauss(0.5, 1.0), random.gauss(0.1, 0.5), random.gauss(0.3, 0.4)) for _ in range(20)]
    t_multi = _time_it(aggregate_multipass, data, iterations=10000)
    t_single = _time_it(aggregate_singlepass, data, iterations=10000)
    speedup = t_multi / t_single if t_single > 0 else float("inf")
    print(f"Aggregation (20 windows): multipass={t_multi:>8.1f}μs  singlepass={t_single:>8.1f}μs  Speedup={speedup:.1f}x")

    # Validate accuracy
    r1 = aggregate_multipass(data)
    r2 = aggregate_singlepass(data)
    for a, b in zip(r1[:4], r2[:4], strict=True):
        assert abs(a - b) < 1e-10, f"Mismatch: {a} vs {b}"
    assert r1[4] == r2[4]
    print("  -> Accuracy verified: exact match")
    print()

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
