"""Fitness evaluation for genetic strategy discovery.

Multi-objective fitness: Sharpe, MaxDD, ProfitFactor with commission-aware
overtrading penalty, complexity penalty, and Pareto ranking for selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from vibe_quant.discovery.operators import StrategyChromosome

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scoring weights (must sum to 1.0)
SHARPE_WEIGHT: float = 0.35
DRAWDOWN_WEIGHT: float = 0.25
PROFIT_FACTOR_WEIGHT: float = 0.20
RETURN_WEIGHT: float = 0.20

# Normalization bounds
SHARPE_MIN: float = -1.0
SHARPE_MAX: float = 4.0
PF_MIN: float = 0.0
PF_MAX: float = 5.0
RETURN_MIN: float = -1.0  # -100%
RETURN_MAX: float = 2.0   # +200%

# Complexity penalty
COMPLEXITY_PENALTY_PER_GENE: float = 0.02
COMPLEXITY_FREE_GENES: int = 2
COMPLEXITY_PENALTY_CAP: float = 0.1

# Minimum trade threshold
MIN_TRADES: int = 50

# Overtrading penalty: commission-aware
# Assumes ~0.1% round-trip commission (taker fees on crypto perps)
COMMISSION_RATE_PER_TRADE: float = 0.001
# Trades above this threshold incur escalating penalty
OVERTRADE_THRESHOLD: int = 300
OVERTRADE_PENALTY_SCALE: float = 0.05  # penalty per 100 excess trades


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FitnessResult:
    """Result of fitness evaluation for a single chromosome.

    Attributes:
        sharpe_ratio: Sharpe ratio from backtest.
        max_drawdown: Max drawdown as fraction 0-1 (lower is better).
        profit_factor: Gross profit / gross loss.
        total_trades: Number of trades executed.
        total_return: Total return as fraction (e.g. 0.45 = +45%).
        complexity_penalty: Penalty applied for strategy complexity.
        overtrade_penalty: Penalty for excessive trading / commission drag.
        raw_score: Weighted score before penalties.
        adjusted_score: Final score after all penalties.
        passed_filters: Whether candidate passed overfitting filters.
        filter_results: Per-filter pass/fail results.
    """

    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    total_return: float
    complexity_penalty: float
    overtrade_penalty: float
    raw_score: float
    adjusted_score: float
    passed_filters: bool
    filter_results: dict[str, bool]


# Pre-compute inverse ranges for normalization to avoid repeated division
_SHARPE_INV_RANGE: float = 1.0 / (SHARPE_MAX - SHARPE_MIN)
_PF_INV_RANGE: float = 1.0 / (PF_MAX - PF_MIN)
_RETURN_INV_RANGE: float = 1.0 / (RETURN_MAX - RETURN_MIN)

# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------


def compute_fitness_score(
    sharpe_ratio: float,
    max_drawdown: float,
    profit_factor: float,
    total_return: float = 0.0,
) -> float:
    """Compute weighted multi-objective fitness score.

    Components normalized to [0,1] then combined:
    - Sharpe (35%): clamped to [-1, 4], normalized
    - MaxDD (25%): inverted (1 - MaxDD), already 0-1
    - ProfitFactor (20%): clamped to [0, 5], normalized
    - TotalReturn (20%): clamped to [-100%, +200%], normalized

    Uses inlined clamp/normalize with pre-computed inverse ranges
    to eliminate function-call overhead in hot loops.

    Args:
        sharpe_ratio: Strategy Sharpe ratio.
        max_drawdown: Max drawdown as fraction 0-1.
        profit_factor: Gross profit / gross loss.
        total_return: Total return as fraction (e.g. 0.45 = +45%).

    Returns:
        Weighted score in [0, 1].
    """
    # Inline clamp + normalize for sharpe
    s = sharpe_ratio
    if s < SHARPE_MIN:
        s = SHARPE_MIN
    elif s > SHARPE_MAX:
        s = SHARPE_MAX
    sharpe_norm = (s - SHARPE_MIN) * _SHARPE_INV_RANGE

    # Inline clamp for drawdown (already in 0-1 range conceptually)
    dd = max_drawdown
    if dd < 0.0:
        dd = 0.0
    elif dd > 1.0:
        dd = 1.0
    dd_norm = 1.0 - dd

    # Inline clamp + normalize for profit factor
    pf = profit_factor
    if pf < PF_MIN:
        pf = PF_MIN
    elif pf > PF_MAX:
        pf = PF_MAX
    pf_norm = (pf - PF_MIN) * _PF_INV_RANGE

    # Inline clamp + normalize for total return
    r = total_return
    if r < RETURN_MIN:
        r = RETURN_MIN
    elif r > RETURN_MAX:
        r = RETURN_MAX
    return_norm = (r - RETURN_MIN) * _RETURN_INV_RANGE

    return (
        SHARPE_WEIGHT * sharpe_norm
        + DRAWDOWN_WEIGHT * dd_norm
        + PROFIT_FACTOR_WEIGHT * pf_norm
        + RETURN_WEIGHT * return_norm
    )


def compute_overtrade_penalty(total_trades: int) -> float:
    """Compute commission-aware overtrading penalty.

    Strategies exceeding OVERTRADE_THRESHOLD trades get penalized
    proportionally to excess trade count, simulating commission drag.

    Args:
        total_trades: Number of trades executed.

    Returns:
        Penalty value >= 0. Applied as subtraction from raw score.
    """
    if total_trades <= OVERTRADE_THRESHOLD:
        return 0.0
    excess = total_trades - OVERTRADE_THRESHOLD
    return min(0.3, OVERTRADE_PENALTY_SCALE * (excess / 100.0))


def compute_complexity_penalty(num_genes: int) -> float:
    """Compute Occam's razor complexity penalty.

    Penalty = 0.02 * (num_genes - 2) for num_genes > 2.
    Clamped to [0, 0.1].

    Args:
        num_genes: Total number of genes (entry + exit).

    Returns:
        Penalty value in [0, 0.1].
    """
    if num_genes <= COMPLEXITY_FREE_GENES:
        return 0.0
    penalty = COMPLEXITY_PENALTY_PER_GENE * (num_genes - COMPLEXITY_FREE_GENES)
    return min(penalty, COMPLEXITY_PENALTY_CAP)


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------

def pareto_dominates(a: FitnessResult, b: FitnessResult) -> bool:
    """Check if strategy A Pareto-dominates strategy B.

    A dominates B iff A is >= B in all objectives and > in at least one.

    Uses direct float comparisons instead of tuple iteration
    for reduced overhead on this frequently-called function.

    Args:
        a: First fitness result.
        b: Second fitness result.

    Returns:
        True if a dominates b.
    """
    # Inline objective extraction and comparison to avoid tuple allocation
    a_sharpe = a.sharpe_ratio
    b_sharpe = b.sharpe_ratio
    a_dd = 1.0 - a.max_drawdown
    b_dd = 1.0 - b.max_drawdown
    a_pf = a.profit_factor
    b_pf = b.profit_factor

    # All >= check with early exit
    if a_sharpe < b_sharpe or a_dd < b_dd or a_pf < b_pf:
        return False

    # At least one strictly better
    return a_sharpe > b_sharpe or a_dd > b_dd or a_pf > b_pf


def pareto_rank(population_fitness: Sequence[FitnessResult]) -> list[int]:
    """Assign Pareto front ranks to a population.

    Front 0 = non-dominated, front 1 = dominated only by front 0, etc.

    Uses a hybrid approach: for each front extraction, uses Python loops
    for small remaining populations and vectorized dominance for larger ones.
    The pareto_dominates() function has been inlined for critical path speed.

    Args:
        population_fitness: Fitness results for each individual.

    Returns:
        List of rank integers (0-indexed), parallel to input.
    """
    n = len(population_fitness)
    if n == 0:
        return []

    ranks = [-1] * n
    remaining = set(range(n))
    current_rank = 0

    # Pre-extract objectives to avoid repeated attribute access
    sharpes = [f.sharpe_ratio for f in population_fitness]
    inv_dds = [1.0 - f.max_drawdown for f in population_fitness]
    pfs = [f.profit_factor for f in population_fitness]

    while remaining:
        # Find non-dominated set in remaining
        front: list[int] = []
        remaining_list = list(remaining)

        for i in remaining_list:
            dominated = False
            s_i = sharpes[i]
            d_i = inv_dds[i]
            p_i = pfs[i]

            for j in remaining_list:
                if i == j:
                    continue
                s_j = sharpes[j]
                d_j = inv_dds[j]
                p_j = pfs[j]

                # Inline pareto_dominates: j dominates i
                if (s_j >= s_i and d_j >= d_i and p_j >= p_i
                        and (s_j > s_i or d_j > d_i or p_j > p_i)):
                    dominated = True
                    break

            if not dominated:
                front.append(i)

        for i in front:
            ranks[i] = current_rank
            remaining.discard(i)

        current_rank += 1

    return ranks


# ---------------------------------------------------------------------------
# Population evaluation
# ---------------------------------------------------------------------------


def _evaluate_single(
    chrom: StrategyChromosome,
    backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]],
    filter_fn: Callable[[StrategyChromosome, dict[str, float | int]], dict[str, bool]] | None = None,
) -> FitnessResult:
    """Evaluate fitness for a single chromosome (picklable for multiprocessing)."""
    _zero = FitnessResult(
        sharpe_ratio=0.0, max_drawdown=1.0, profit_factor=0.0,
        total_trades=0, total_return=-1.0, complexity_penalty=0.0,
        overtrade_penalty=0.0, raw_score=0.0, adjusted_score=0.0,
        passed_filters=False, filter_results={},
    )

    try:
        bt = backtest_fn(chrom)
    except Exception:
        logger.warning("Backtest failed for chromosome, assigning zero fitness", exc_info=True)
        return _zero

    sharpe = float(bt["sharpe_ratio"])
    max_dd = float(bt["max_drawdown"])
    pf = float(bt["profit_factor"])
    trades = int(bt["total_trades"])
    total_return = float(bt.get("total_return", 0.0))

    num_genes = len(chrom.entry_genes) + len(chrom.exit_genes)
    complexity_pen = compute_complexity_penalty(num_genes)
    overtrade_pen = compute_overtrade_penalty(trades)

    # Filter evaluation
    filter_results = filter_fn(chrom, bt) if filter_fn is not None else {}
    passed_filters = all(filter_results.values()) if filter_results else True

    # Compute raw score including total return
    raw = compute_fitness_score(sharpe, max_dd, pf, total_return)

    # Apply minimum trade filter + all penalties
    if trades < MIN_TRADES:
        adjusted = 0.0
    else:
        adjusted = max(0.0, raw - complexity_pen - overtrade_pen)

    return FitnessResult(
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        profit_factor=pf,
        total_trades=trades,
        total_return=total_return,
        complexity_penalty=complexity_pen,
        overtrade_penalty=overtrade_pen,
        raw_score=raw,
        adjusted_score=adjusted,
        passed_filters=passed_filters,
        filter_results=filter_results,
    )


def evaluate_population(
    chromosomes: list[StrategyChromosome],
    backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]],
    filter_fn: Callable[[StrategyChromosome, dict[str, float | int]], dict[str, bool]] | None = None,
    *,
    max_workers: int | None = None,
    executor: object | None = None,
) -> list[FitnessResult]:
    """Evaluate fitness for an entire population, optionally in parallel.

    When max_workers > 1, uses ProcessPoolExecutor for parallel evaluation.
    Falls back to sequential if parallelization fails.
    Pass `executor` to reuse a long-lived pool across generations.

    Args:
        chromosomes: List of strategy chromosomes.
        backtest_fn: Callable that runs a backtest and returns dict with keys:
            sharpe_ratio, max_drawdown, profit_factor, total_trades, total_return.
        filter_fn: Optional callable that returns per-filter pass/fail dict.
        max_workers: Max parallel workers. None = sequential. 0 = auto (cpu_count).

    Returns:
        FitnessResult for each chromosome, parallel to input.
    """
    if max_workers is not None and max_workers != 1:
        try:
            return _evaluate_parallel(chromosomes, backtest_fn, filter_fn, max_workers, executor)
        except Exception:
            logger.warning("Parallel evaluation failed, falling back to sequential", exc_info=True)

    return [_evaluate_single(chrom, backtest_fn, filter_fn) for chrom in chromosomes]


def _evaluate_parallel(
    chromosomes: list[StrategyChromosome],
    backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]],
    filter_fn: Callable[[StrategyChromosome, dict[str, float | int]], dict[str, bool]] | None,
    max_workers: int | None,
    executor: object | None = None,
) -> list[FitnessResult]:
    """Evaluate population using ProcessPoolExecutor.

    If `executor` is provided, it's reused (caller manages lifecycle).
    Otherwise a temporary pool is created and destroyed.
    """
    import os
    from concurrent.futures import Executor, ProcessPoolExecutor, as_completed

    workers = max_workers if max_workers and max_workers > 0 else os.cpu_count() or 4
    workers = min(workers, len(chromosomes))

    _zero = FitnessResult(
        sharpe_ratio=0.0, max_drawdown=1.0, profit_factor=0.0,
        total_trades=0, total_return=-1.0, complexity_penalty=0.0,
        overtrade_penalty=0.0, raw_score=0.0, adjusted_score=0.0,
        passed_filters=False, filter_results={},
    )

    results: list[FitnessResult | None] = [None] * len(chromosomes)

    def _run_with(pool: Executor) -> None:
        future_to_idx = {
            pool.submit(_evaluate_single, chrom, backtest_fn, filter_fn): i
            for i, chrom in enumerate(chromosomes)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                logger.warning("Parallel eval failed for chromosome %d", idx, exc_info=True)
                results[idx] = _zero

    if executor is not None and isinstance(executor, Executor):
        _run_with(executor)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            _run_with(pool)

    return [r if r is not None else _zero for r in results]
