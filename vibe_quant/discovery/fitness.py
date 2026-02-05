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
    return max(lo, min(hi, value))


def _normalize(value: float, lo: float, hi: float) -> float:
    """Normalize value from [lo, hi] to [0, 1]."""
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)


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

    Args:
        sharpe_ratio: Strategy Sharpe ratio.
        max_drawdown: Max drawdown as fraction 0-1.
        profit_factor: Gross profit / gross loss.

    Returns:
        Weighted score in [0, 1].
    """
    sharpe_norm = _normalize(_clamp(sharpe_ratio, SHARPE_MIN, SHARPE_MAX), SHARPE_MIN, SHARPE_MAX)
    dd_norm = 1.0 - _clamp(max_drawdown, 0.0, 1.0)
    pf_norm = _normalize(_clamp(profit_factor, PF_MIN, PF_MAX), PF_MIN, PF_MAX)

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

    Args:
        a: First fitness result.
        b: Second fitness result.

    Returns:
        True if a dominates b.
    """
    obj_a = _objectives(a)
    obj_b = _objectives(b)

    at_least_one_better = False
    for va, vb in zip(obj_a, obj_b, strict=True):
        if va < vb:
            return False
        if va > vb:
            at_least_one_better = True
    return at_least_one_better


def pareto_rank(population_fitness: Sequence[FitnessResult]) -> list[int]:
    """Assign Pareto front ranks to a population.

    Front 0 = non-dominated, front 1 = dominated only by front 0, etc.

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

    while remaining:
        # Find non-dominated set in remaining
        front: list[int] = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and pareto_dominates(population_fitness[j], population_fitness[i]):
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
