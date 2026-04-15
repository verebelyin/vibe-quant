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
RETURN_MAX: float = 2.0  # +200%

# Complexity penalty
COMPLEXITY_PENALTY_PER_GENE: float = 0.02
COMPLEXITY_FREE_GENES: int = 2
COMPLEXITY_PENALTY_CAP: float = 0.1

# Minimum trade thresholds
MIN_TRADES: int = 50  # Default for 4h and longer timeframes
MIN_TRADES_1M: int = 50  # For 1m timeframe (lowered from 100 — 3mo windows produce fewer trades in some regimes)

# Overtrading penalty: commission-aware
# Assumes ~0.1% round-trip commission (taker fees on crypto perps)
COMMISSION_RATE_PER_TRADE: float = 0.001
# Trades above this threshold incur escalating penalty. Historic default
# calibrated for 4h (~300 trades/year is healthy); sub-4h timeframes
# naturally generate far more trades, so we scale per timeframe below.
OVERTRADE_THRESHOLD: int = 300
OVERTRADE_PENALTY_SCALE: float = 0.05  # penalty per 100 excess trades

# Timeframe-scaled thresholds. A healthy 1m strategy emits ~3000 trades
# over a year; 15m ~500. Gating those at 300 zeros every adjusted_score
# and causes the discovery bug seen in prod (bd-pbgl). Values picked to
# match healthy-strategy trade counts seen in the discovery journal.
OVERTRADE_THRESHOLD_BY_TIMEFRAME: dict[str, int] = {
    "1m": 3000,
    "5m": 1000,
    "15m": 500,
    "1h": 300,
    "4h": 300,
    "1d": 150,
}

# SL/TP imbalance penalty: discourages ultra-tight TP scalpers that
# don't generalize beyond the training window (bd-v7zj).
# Penalty kicks in when SL/TP ratio > SL_TP_RATIO_THRESHOLD (e.g. 7% SL / 0.7% TP = 10x).
SL_TP_RATIO_THRESHOLD: float = 5.0
SL_TP_RATIO_PENALTY_SCALE: float = 0.02  # per unit above threshold
SL_TP_RATIO_PENALTY_CAP: float = 0.15


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
    sl_tp_penalty: float
    raw_score: float
    adjusted_score: float
    passed_filters: bool
    filter_results: dict[str, bool]
    skewness: float = 0.0
    kurtosis: float = 3.0
    trade_returns: tuple[float, ...] = ()


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


def compute_overtrade_penalty(total_trades: int, timeframe: str | None = None) -> float:
    """Commission-aware overtrading penalty, timeframe-scaled.

    Threshold scales per ``OVERTRADE_THRESHOLD_BY_TIMEFRAME`` because
    healthy sub-4h strategies naturally generate many more trades (a 1m
    year = ~525k bars vs 4h year = ~2k bars). Falls back to the global
    ``OVERTRADE_THRESHOLD`` when ``timeframe`` is unknown.

    Args:
        total_trades: Number of trades executed.
        timeframe: Strategy timeframe (e.g. "5m", "4h"). When ``None``,
            the global 4h-calibrated threshold is used.

    Returns:
        Penalty value >= 0. Applied as subtraction from raw score.
    """
    threshold = OVERTRADE_THRESHOLD_BY_TIMEFRAME.get(timeframe or "", OVERTRADE_THRESHOLD)
    if total_trades <= threshold:
        return 0.0
    excess = total_trades - threshold
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


def compute_sl_tp_penalty(sl_pct: float, tp_pct: float) -> float:
    """Compute penalty for extreme SL/TP ratios.

    Ultra-tight TP scalpers (e.g. 7% SL / 0.7% TP = 10x ratio) look great
    in-sample but don't generalize. Penalty kicks in at SL/TP > 5x.

    Args:
        sl_pct: Stop loss percentage.
        tp_pct: Take profit percentage.

    Returns:
        Penalty value in [0, SL_TP_RATIO_PENALTY_CAP].
    """
    if tp_pct <= 0 or sl_pct <= 0:
        return 0.0
    ratio = sl_pct / tp_pct
    if ratio <= SL_TP_RATIO_THRESHOLD:
        return 0.0
    excess = ratio - SL_TP_RATIO_THRESHOLD
    return min(SL_TP_RATIO_PENALTY_CAP, SL_TP_RATIO_PENALTY_SCALE * excess)


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
                if (
                    s_j >= s_i
                    and d_j >= d_i
                    and p_j >= p_i
                    and (s_j > s_i or d_j > d_i or p_j > p_i)
                ):
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
    filter_fn: Callable[[StrategyChromosome, dict[str, float | int]], dict[str, bool]]
    | None = None,
    min_trades: int = MIN_TRADES,
    timeframe: str | None = None,
) -> FitnessResult:
    """Evaluate fitness for a single chromosome (picklable for multiprocessing)."""
    _zero = FitnessResult(
        sharpe_ratio=0.0,
        max_drawdown=1.0,
        profit_factor=0.0,
        total_trades=0,
        total_return=-1.0,
        complexity_penalty=0.0,
        overtrade_penalty=0.0,
        sl_tp_penalty=0.0,
        raw_score=0.0,
        adjusted_score=0.0,
        passed_filters=False,
        filter_results={},
    )

    try:
        import time as _time
        _bt_start = _time.monotonic()
        bt = backtest_fn(chrom)
        _bt_elapsed = _time.monotonic() - _bt_start
        indicators = [g.indicator_type for g in chrom.entry_genes + chrom.exit_genes]
        logger.debug(
            "Backtest %s: %.1fs indicators=%s trades=%s",
            chrom.uid,
            _bt_elapsed,
            indicators,
            bt.get("total_trades", "?"),
        )
    except Exception:
        logger.warning("Backtest failed for chromosome %s, assigning zero fitness", chrom.uid, exc_info=True)
        return _zero

    import math as _math

    sharpe = float(bt["sharpe_ratio"])
    max_dd = float(bt["max_drawdown"])
    pf = float(bt["profit_factor"])
    trades = int(bt["total_trades"])
    total_return = float(bt.get("total_return", 0.0))

    # Coerce NaN metrics to safe defaults (NT returns NaN for 0-trade strategies)
    if _math.isnan(sharpe):
        sharpe = 0.0
    if _math.isnan(max_dd):
        max_dd = 0.0
    if _math.isnan(pf):
        pf = 0.0
    if _math.isnan(total_return):
        total_return = 0.0

    # Sanity checks on backtest output — flag impossible metric combinations
    _sanity_check_metrics(chrom.uid, sharpe, max_dd, pf, trades, total_return)

    num_genes = len(chrom.entry_genes) + len(chrom.exit_genes)
    complexity_pen = compute_complexity_penalty(num_genes)
    overtrade_pen = compute_overtrade_penalty(trades, timeframe)
    sl_tp_pen = compute_sl_tp_penalty(chrom.stop_loss_pct, chrom.take_profit_pct)

    # Filter evaluation
    filter_results = filter_fn(chrom, bt) if filter_fn is not None else {}
    passed_filters = all(filter_results.values()) if filter_results else True

    # Compute raw score including total return
    raw = compute_fitness_score(sharpe, max_dd, pf, total_return)

    # Hard gates: insufficient trades or negative return → zero fitness
    if trades < min_trades or total_return <= 0:
        adjusted = 0.0
    else:
        adjusted = max(0.0, raw - complexity_pen - overtrade_pen - sl_tp_pen)

    # Log score decomposition for debugging fitness calculation correctness
    if adjusted > 0:
        logger.debug(
            "Score %s: raw=%.4f - complexity=%.4f - overtrade=%.4f - sl_tp=%.4f = adjusted=%.4f "
            "(sharpe=%.2f dd=%.3f pf=%.2f ret=%.3f trades=%d genes=%d sl=%.2f%% tp=%.2f%%)",
            chrom.uid, raw, complexity_pen, overtrade_pen, sl_tp_pen, adjusted,
            sharpe, max_dd, pf, total_return, trades, num_genes,
            chrom.stop_loss_pct, chrom.take_profit_pct,
        )

    return FitnessResult(
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        profit_factor=pf,
        total_trades=trades,
        total_return=total_return,
        complexity_penalty=complexity_pen,
        overtrade_penalty=overtrade_pen,
        sl_tp_penalty=sl_tp_pen,
        raw_score=raw,
        adjusted_score=adjusted,
        passed_filters=passed_filters,
        filter_results=filter_results,
        skewness=float(bt.get("skewness", 0.0)),
        kurtosis=float(bt.get("kurtosis", 3.0)),
        trade_returns=tuple(bt.get("trade_returns", ())),  # type: ignore[arg-type]
    )


def _sanity_check_metrics(
    uid: str,
    sharpe: float,
    max_dd: float,
    pf: float,
    trades: int,
    total_return: float,
) -> None:
    """Log warnings for impossible or suspicious metric combinations.

    These indicate potential bugs in the backtesting engine or data issues,
    not just bad strategies.
    """
    issues: list[str] = []

    # Drawdown must be in [0, 1]
    if max_dd < 0:
        issues.append(f"negative max_drawdown={max_dd:.4f}")
    if max_dd > 1.0:
        issues.append(f"max_drawdown={max_dd:.4f} > 1.0")

    # Profit factor must be non-negative
    if pf < 0:
        issues.append(f"negative profit_factor={pf:.4f}")

    # Zero trades but non-zero metrics
    if trades == 0:
        if sharpe != 0 and sharpe != float("-inf"):
            issues.append(f"0 trades but sharpe={sharpe:.4f}")
        if total_return != 0:
            issues.append(f"0 trades but return={total_return:.4f}")
        if pf != 0:
            issues.append(f"0 trades but pf={pf:.4f}")

    # Positive return with profit_factor < 1 (or vice versa) — plausible but suspicious
    if trades > 10 and total_return > 0.1 and pf < 0.8:
        issues.append(f"return={total_return:.2f} but pf={pf:.2f} (suspicious mismatch)")
    if trades > 10 and total_return < -0.1 and pf > 1.5:
        issues.append(f"return={total_return:.2f} but pf={pf:.2f} (suspicious mismatch)")

    # Extremely high Sharpe with very few trades — likely noise
    if trades < 30 and sharpe > 5.0:
        issues.append(f"sharpe={sharpe:.2f} with only {trades} trades (likely noise)")

    if issues:
        logger.warning("SANITY %s: %s", uid, "; ".join(issues))


_parallel_broken: bool = False


def evaluate_population(
    chromosomes: list[StrategyChromosome],
    backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]],
    filter_fn: Callable[[StrategyChromosome, dict[str, float | int]], dict[str, bool]]
    | None = None,
    *,
    max_workers: int | None = None,
    executor: object | None = None,
    min_trades: int = MIN_TRADES,
    timeframe: str | None = None,
) -> list[FitnessResult]:
    """Evaluate fitness for an entire population, optionally in parallel.

    When max_workers > 1, uses ProcessPoolExecutor for parallel evaluation.
    Falls back to sequential if parallelization fails (cached — only logs once).
    Pass `executor` to reuse a long-lived pool across generations.

    Args:
        chromosomes: List of strategy chromosomes.
        backtest_fn: Callable that runs a backtest and returns dict with keys:
            sharpe_ratio, max_drawdown, profit_factor, total_trades, total_return.
        filter_fn: Optional callable that returns per-filter pass/fail dict.
        max_workers: Max parallel workers. None = sequential. 0 = auto (cpu_count).
        min_trades: Minimum trades hard gate (default 50 for 4h, use 100 for 1m).

    Returns:
        FitnessResult for each chromosome, parallel to input.
    """
    global _parallel_broken  # noqa: PLW0603
    if max_workers is not None and max_workers != 1 and not _parallel_broken:
        try:
            return _evaluate_parallel(chromosomes, backtest_fn, filter_fn, max_workers, executor, min_trades=min_trades, timeframe=timeframe)
        except Exception:
            _parallel_broken = True
            logger.warning("Parallel evaluation failed, falling back to sequential for remainder of run", exc_info=True)

    return [_evaluate_single(chrom, backtest_fn, filter_fn, min_trades=min_trades, timeframe=timeframe) for chrom in chromosomes]


def _evaluate_parallel(
    chromosomes: list[StrategyChromosome],
    backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]],
    filter_fn: Callable[[StrategyChromosome, dict[str, float | int]], dict[str, bool]] | None,
    max_workers: int | None,
    executor: object | None = None,
    min_trades: int = MIN_TRADES,
    timeframe: str | None = None,
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
        sharpe_ratio=0.0,
        max_drawdown=1.0,
        profit_factor=0.0,
        total_trades=0,
        total_return=-1.0,
        complexity_penalty=0.0,
        overtrade_penalty=0.0,
        sl_tp_penalty=0.0,
        raw_score=0.0,
        adjusted_score=0.0,
        passed_filters=False,
        filter_results={},
    )

    results: list[FitnessResult | None] = [None] * len(chromosomes)

    def _run_with(pool: Executor) -> None:
        future_to_idx = {
            pool.submit(_evaluate_single, chrom, backtest_fn, filter_fn, min_trades, timeframe): i
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
        pool = ProcessPoolExecutor(max_workers=workers)
        try:
            _run_with(pool)
        except RuntimeError:
            # macOS spawn method fails inside NautilusTrader's Rust runtime.
            # Explicitly terminate orphaned workers and re-raise so caller
            # falls back to sequential.
            _force_shutdown_pool(pool)
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    return [r if r is not None else _zero for r in results]


def _force_shutdown_pool(pool: object) -> None:
    """Kill all worker processes in a ProcessPoolExecutor.

    Handles the macOS case where spawn-method workers get stuck at 100% CPU
    after a RuntimeError during submit().
    """
    import signal

    processes = getattr(pool, "_processes", None)
    if not processes:
        return
    for pid, proc in list(processes.items()):
        try:
            if proc.is_alive():
                logger.warning("Killing orphaned pool worker pid=%d", pid)
                proc.kill()
                proc.join(timeout=5)
        except Exception:
            # Last resort: SIGKILL via os
            try:
                import os

                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
