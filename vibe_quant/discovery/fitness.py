"""Fitness evaluation for genetic strategy discovery.

Multi-objective fitness: Sharpe, MaxDD, ProfitFactor with complexity penalty
and Pareto ranking for selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from vibe_quant.discovery.genome import StrategyChromosome

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scoring weights
SHARPE_WEIGHT: float = 0.4
DRAWDOWN_WEIGHT: float = 0.3
PROFIT_FACTOR_WEIGHT: float = 0.3

# Normalization bounds
SHARPE_MIN: float = -1.0
SHARPE_MAX: float = 4.0
PF_MIN: float = 0.0
PF_MAX: float = 5.0

# Complexity penalty
COMPLEXITY_PENALTY_PER_GENE: float = 0.02
COMPLEXITY_FREE_GENES: int = 2
COMPLEXITY_PENALTY_CAP: float = 0.1

# Minimum trade threshold
MIN_TRADES: int = 50


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
        complexity_penalty: Penalty applied for strategy complexity.
        raw_score: Weighted score before penalty.
        adjusted_score: Final score after penalty.
        passed_filters: Whether candidate passed overfitting filters.
        filter_results: Per-filter pass/fail results.
    """

    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    complexity_penalty: float
    raw_score: float
    adjusted_score: float
    passed_filters: bool
    filter_results: dict[str, bool]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _normalize(value: float, lo: float, hi: float) -> float:
    """Normalize value from [lo, hi] to [0, 1]."""
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)


# Pre-compute inverse ranges for normalization to avoid repeated division
_SHARPE_INV_RANGE: float = 1.0 / (SHARPE_MAX - SHARPE_MIN)
_PF_INV_RANGE: float = 1.0 / (PF_MAX - PF_MIN)

# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------


def compute_fitness_score(
    sharpe_ratio: float,
    max_drawdown: float,
    profit_factor: float,
) -> float:
    """Compute weighted multi-objective fitness score.

    Components normalized to [0,1] then combined:
    - Sharpe: clamped to [-1, 4], normalized
    - MaxDD: inverted (1 - MaxDD), already 0-1
    - ProfitFactor: clamped to [0, 5], normalized

    Uses inlined clamp/normalize with pre-computed inverse ranges
    to eliminate function-call overhead in hot loops.

    Args:
        sharpe_ratio: Strategy Sharpe ratio.
        max_drawdown: Max drawdown as fraction 0-1.
        profit_factor: Gross profit / gross loss.

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

    return (
        SHARPE_WEIGHT * sharpe_norm
        + DRAWDOWN_WEIGHT * dd_norm
        + PROFIT_FACTOR_WEIGHT * pf_norm
    )


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

# Objectives: (sharpe, 1-maxdd, profit_factor) -- all higher is better
ObjectivesTuple = tuple[float, float, float]


def _objectives(result: FitnessResult) -> ObjectivesTuple:
    """Extract objective values (all higher-is-better) from FitnessResult."""
    return (result.sharpe_ratio, 1.0 - result.max_drawdown, result.profit_factor)


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


def evaluate_population(
    chromosomes: list[StrategyChromosome],
    backtest_fn: Callable[[StrategyChromosome], dict[str, Any]],
    filter_fn: Callable[[StrategyChromosome, dict[str, Any]], dict[str, bool]] | None = None,
) -> list[FitnessResult]:
    """Evaluate fitness for an entire population.

    Args:
        chromosomes: List of strategy chromosomes.
        backtest_fn: Callable that runs a backtest and returns dict with keys:
            sharpe_ratio, max_drawdown, profit_factor, total_trades.
        filter_fn: Optional callable that returns per-filter pass/fail dict.
            If None, all filters considered passed.

    Returns:
        FitnessResult for each chromosome, parallel to input.
    """
    results: list[FitnessResult] = []

    for chrom in chromosomes:
        bt = backtest_fn(chrom)
        sharpe = float(bt["sharpe_ratio"])
        max_dd = float(bt["max_drawdown"])
        pf = float(bt["profit_factor"])
        trades = int(bt["total_trades"])

        num_genes = len(chrom.entry_genes) + len(chrom.exit_genes)
        penalty = compute_complexity_penalty(num_genes)

        # Filter evaluation
        filter_results = filter_fn(chrom, bt) if filter_fn is not None else {}
        passed_filters = all(filter_results.values()) if filter_results else True

        # Compute raw score
        raw = compute_fitness_score(sharpe, max_dd, pf)

        # Apply minimum trade filter and complexity penalty
        adjusted = 0.0 if trades < MIN_TRADES else max(0.0, raw - penalty)

        results.append(
            FitnessResult(
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                profit_factor=pf,
                total_trades=trades,
                complexity_penalty=penalty,
                raw_score=raw,
                adjusted_score=adjusted,
                passed_filters=passed_filters,
                filter_results=filter_results,
            )
        )

    return results
