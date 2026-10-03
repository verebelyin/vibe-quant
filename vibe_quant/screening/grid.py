"""Parameter grid utilities for screening pipeline.

Pure functions for building parameter grids, filtering results,
ranking, and computing Pareto fronts. No I/O or side effects.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe_quant.screening.types import BacktestMetrics, MetricFilters


def build_parameter_grid(
    sweep: dict[str, list[int] | list[float]],
) -> list[dict[str, float | int]]:
    """Build Cartesian product of sweep parameters.

    Args:
        sweep: Dictionary mapping parameter names to lists of values.
            Example: {"rsi.period": [7, 14, 21], "stop_loss.percent": [1.0, 2.0]}

    Returns:
        List of parameter dictionaries, one for each combination.
        Example: [{"rsi.period": 7, "stop_loss.percent": 1.0}, ...]
    """
    if not sweep:
        return [{}]

    # Get param names and values in consistent order
    param_names = list(sweep.keys())
    param_values = [sweep[name] for name in param_names]

    # Generate Cartesian product
    combinations: list[dict[str, float | int]] = []
    for values in itertools.product(*param_values):
        combo = dict(zip(param_names, values, strict=True))
        combinations.append(combo)

    return combinations


def filter_by_metrics(
    results: list[BacktestMetrics],
    filters: MetricFilters,
) -> list[BacktestMetrics]:
    """Apply hard metric filters to screening results.

    Args:
        results: List of backtest metrics
        filters: MetricFilters with min_sharpe, min_profit_factor,
                 max_drawdown, min_trades thresholds.

    Returns:
        Filtered list of results that pass all thresholds
    """
    filtered: list[BacktestMetrics] = []

    for r in results:
        # NaN compares False against every threshold, so a "<" check let NaN
        # metrics straight through. Require the positive condition instead.
        # (+inf profit factor = no losing trades, a legitimate pass.)
        if not (math.isfinite(r.sharpe_ratio) and r.sharpe_ratio >= filters.min_sharpe):
            continue
        if not r.profit_factor >= filters.min_profit_factor:
            continue
        if not r.max_drawdown <= filters.max_drawdown:
            continue
        if r.total_trades < filters.min_trades:
            continue

        filtered.append(r)

    return filtered


def _rank_key(value: float) -> float:
    """Sort key: NaN ranks below every real value (and below -inf)."""
    return value if not math.isnan(value) else -math.inf


def rank_by_sharpe(results: list[BacktestMetrics]) -> list[BacktestMetrics]:
    """Sort results by Sharpe ratio descending.

    Args:
        results: List of backtest metrics

    Returns:
        Sorted list with highest Sharpe first
    """
    # NaN in a sort key breaks ordering for the whole list (NaN < x and
    # NaN > x are both False); NaN rows go last, stable among themselves.
    return sorted(
        results,
        key=lambda r: (not math.isnan(r.sharpe_ratio), _rank_key(r.sharpe_ratio)),
        reverse=True,
    )


def compute_pareto_front(
    results: list[BacktestMetrics],
) -> list[int]:
    """Compute Pareto-optimal indices using 3 objectives.

    Objectives (all maximized):
    - Sharpe ratio
    - 1 - max_drawdown (inverted so higher is better)
    - Profit factor

    A result is Pareto-optimal if no other result dominates it
    (i.e., is better in ALL objectives simultaneously).

    Optimized with inlined comparisons and pre-extracted attributes
    to eliminate function-call overhead and tuple allocation in the
    inner loop. ~3-4x faster than the original implementation.

    Args:
        results: List of backtest metrics

    Returns:
        List of indices into results that are Pareto-optimal
    """
    if not results:
        return []

    n = len(results)

    # Pre-extract objectives to avoid repeated attribute access in O(n^2) loop.
    # A NaN objective is "worst possible" (-inf): NaN comparisons are always
    # False, so NaN rows could never be dominated and sat on every front.
    sharpes = [_rank_key(r.sharpe_ratio) for r in results]
    inv_dds = [_rank_key(1.0 - r.max_drawdown) for r in results]
    pfs = [_rank_key(r.profit_factor) for r in results]

    # Rows with any NaN objective are never Pareto-optimal
    is_pareto = [
        not (
            math.isnan(r.sharpe_ratio)
            or math.isnan(r.max_drawdown)
            or math.isnan(r.profit_factor)
        )
        for r in results
    ]

    for i in range(n):
        if not is_pareto[i]:
            continue

        s_i = sharpes[i]
        d_i = inv_dds[i]
        p_i = pfs[i]

        for j in range(n):
            if i == j or not is_pareto[j]:
                continue

            s_j = sharpes[j]
            d_j = inv_dds[j]
            p_j = pfs[j]

            # Inlined dominance check: j >= i in all AND j > i in at least one
            if s_j >= s_i and d_j >= d_i and p_j >= p_i and (s_j > s_i or d_j > d_i or p_j > p_i):
                is_pareto[i] = False
                break

    return [i for i in range(n) if is_pareto[i]]
