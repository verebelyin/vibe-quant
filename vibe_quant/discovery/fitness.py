"""Fitness evaluation for genetic strategy discovery.

Weighted scalar fitness (35% Sharpe, 25% 1-MaxDD, 20% PF, 20% return) minus
commission-aware overtrading, complexity and SL/TP-imbalance penalties. Discovery
selects on this scalar; ``pareto_rank`` below has no caller in discovery.
"""

from __future__ import annotations

import logging
import random
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np

from vibe_quant.discovery.genome import chromosome_to_dsl, genome_summary
from vibe_quant.discovery.operators import scalar_gene_count
from vibe_quant.discovery.param_genome import ParamGenome
from vibe_quant.errors import DataUnavailableError
from vibe_quant.fast_tier import fallbacks

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.discovery.operators import StrategyChromosome
    from vibe_quant.metrics import DailyReturns

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

# Fitness v2 (bead vibe-quant-clmmc.3): the shrunk t-stat that maps to the middle of a
# band's range (``shrunk / (shrunk + C)`` = 1/2), and the smallest multiplier the three
# penalties can leave (caps 0.1 + 0.3 overtrade + 0.15 -> 0.45); the loser band is capped by it.
FITNESS_V2_SCALE: float = 2.0
_V2_MIN_PENALTY_MULTIPLIER: float = 0.45


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


# The v2 plateau step (user ruling Q20 = B, replaces the design's 10% / 8): the top 5% of every
# generation (``ceil(0.05 * P)`` keys) are re-scored on at most 4 one-step neighbours each.
PLATEAU_TOP_FRACTION: float = 0.05
PLATEAU_MAX_NEIGHBOURS: int = 4


@dataclass(frozen=True, slots=True)
class FitnessV2:
    """Fitness v2 detail of one scored genome (``FitnessResult.fitness_v2``).

    ``own`` is the genome's own score before any plateau step; ``t_stat`` and
    ``window_factor`` are None when the series was degenerate (and, in worst mode,
    are the worst symbol's). ``neighbours`` is None until the plateau step scored
    the genome, ``()`` when it was scored without a usable neighbour.
    """

    own: float
    t_stat: float | None
    window_factor: float | None
    neighbours: tuple[float, ...] | None = None
    neighbour_errors: int = 0


class FitnessV2Error(Exception):
    """A v2 evaluation that cannot be scored honestly (partial windows, no series)."""


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
        error: "ExcType: msg" when the evaluation raised (zero-fitness
            stand-in); None for a normal result. Marker only.
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
    error: str | None = None
    # symbol_agg="worst" only (vibe-quant-ox73t): per-symbol adjusted fitness
    # behind the soft GA score, and per-symbol (total_return, trades) for the
    # hard worst-symbol champion gate. None in portfolio mode.
    symbol_scores: dict[str, float] | None = None
    symbol_stats: dict[str, tuple[float, int]] | None = None
    # Output-only market-valued series (bd vibe-quant-wcn55): the realized
    # (non-MTM) twin was removed with its producers (bd vibe-quant-f9zzq).
    # Never read by fitness/ranking; kept under the ``keep_series`` rule for
    # champion candidates only. Worst mode: the equal-weight pool of the
    # symbols plus each symbol's own series (``symbol_mtm_daily``).
    mtm_daily_returns: DailyReturns | None = None
    symbol_mtm_daily: dict[str, DailyReturns | None] | None = None
    # Fee drag (bd vibe-quant-2gyk1.4): reporting only, never read by the score.
    # ``fee_drag`` is fees / gross PnL from the SUMMED fees and gross of the run
    # (windows and, in worst mode, symbols); None when gross <= 0 (no edge to
    # attribute a share of). ``fees_pct_equity_yr`` annualizes the fees against
    # the starting balance (equity assumed constant); None without a run
    # duration/balance. In worst mode every symbol runs on its OWN account, so
    # the bt dict's ``starting_balance`` is the SUM of those accounts (N x
    # balance) and the summed fees are annualized against the summed equity.
    # Kept for every result -- the whole point is showing that fees, not the
    # signal, killed a genome that never became a champion.
    fee_drag: float | None = None
    fees_pct_equity_yr: float | None = None
    # Liquidation accounting (bead vibe-quant-2gyk1.1.9): reporting only, never
    # read by the score (R1 -- the same formula over the liquidation-adjusted
    # metrics, with no penalty and no gate). The bt dict carries the keys ONLY
    # when the run's mode is on, so these stay None off and callers omit them
    # (R4). ``liquidation_count`` is the summed number of liquidations;
    # ``liquidation_forfeited`` the summed notional/PnL given up to them.
    liquidation_count: int | None = None
    liquidation_forfeited: float | None = None
    # symbol_agg="pooled" only (vibe-quant-clmmc.5): per-symbol (return, trades) of the
    # coins pooled into one hypothesis, for the per-coin trade floor (and C-T2's sign
    # floor). None in portfolio and worst mode.
    pooled_symbol_stats: dict[str, tuple[float, int]] | None = None
    # ``fitness="v2"`` only (bead vibe-quant-clmmc.3); None under v1.
    fitness_v2: FitnessV2 | None = None


def _fee_drag_fields(bt: dict[str, float | int]) -> tuple[float | None, float | None]:
    """``(fee_drag, fees_pct_equity_yr)`` from a bt dict's summed fee fields.

    Reporting only, never scored (bd vibe-quant-2gyk1.4). Each ratio is judged
    on its own inputs: a bad ``gross_pnl`` voids only ``fee_drag``, a bad
    ``days``/``starting_balance`` only ``fees_pct_equity_yr`` (the metrics
    helpers return None for non-finite or non-positive denominators). Non-finite
    ``total_fees`` voids both, so an unparseable backtest can never fabricate a
    ratio. A run without fee fields (mock/synthetic/old) yields None for both.
    """
    import math

    from vibe_quant.metrics import fee_drag, fees_pct_equity_per_year

    fees = float(bt.get("total_fees", 0.0))
    if not math.isfinite(fees):
        return None, None
    gross = float(bt.get("gross_pnl", 0.0))
    balance = float(bt.get("starting_balance", 0.0))
    days = float(bt.get("days", 0))
    return fee_drag(fees, gross), fees_pct_equity_per_year(fees, balance, days)


def _liquidation_fields(
    bt: dict[str, float | int],
) -> tuple[int | None, float | None]:
    """``(liquidation_count, liquidation_forfeited)`` from a bt dict.

    Reporting only, never scored (bead vibe-quant-2gyk1.1.9, R1). The keys exist
    only when the run's mode is on, so a mode-off dict yields ``(None, None)``
    and the callers omit them (R4).
    """
    count = bt.get("liquidation_count")
    forfeited = bt.get("liquidation_forfeited")
    return (
        int(count) if count is not None else None,
        float(forfeited) if forfeited is not None else None,
    )


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


def _v2_window_factor(window_returns: Sequence[float] | None) -> float:
    """``max(1, positives) / N`` over the eval windows' returns; 1.0 for one window or none.

    NaN and 0 are non-positive; the floor of one window keeps a genome that loses every
    window above zero (the train gate, not this factor, decides who is a candidate).
    """
    if window_returns is None or len(window_returns) <= 1:
        return 1.0
    positives = sum(1 for r in window_returns if r > 0)
    return max(1, positives) / len(window_returns)


def fitness_v2_own(
    series: DailyReturns | None,
    trades: int,
    min_trades: int,
    window_returns: Sequence[float] | None,
    penalty: float,
) -> tuple[float, float | None, float | None]:
    """Fitness v2 score of one genome: ``(own, t_stat, window_factor)`` (bead vibe-quant-clmmc.3).

    The t-stat of the chained MTM daily series (the series the DSR gate judges),
    ``mean / std(ddof=1) * sqrt(n)``, not annualized. It is shrunk by trades,
    ``t * trades / (trades + min_trades)`` (no cliff at ``min_trades``), and mapped to
    [0, 1] in two bands split by the compound return of the same series:

    - profitable: ``0.5 + 0.5 * shrunk / (shrunk + C)`` in (0.5, 1);
    - loser (return <= 0, incl. a vol-drag genome with t > 0): ``L * sigmoid(shrunk / C)``
      in (0, L) with ``L = 0.5 * 0.45 / max(1, N - 1)`` for N windows, so a loser scores
      below every profitable genome with the same trades whatever the window factors
      (a loser has at most N-1 positive windows; the worst profitable one scores
      ``(1/N) * 0.45 * 0.5``).

    ``own = window_factor * ((1 - penalty) * mapped)``, penalty being the summed
    complexity, overtrade and SL/TP penalties (multiplier >= 0.45, so own stays in
    [0, 1]). Degenerate inputs (no trades, no series, < 2 days, a non-finite value or
    t-stat, a constant series) score ``(0.0, None, None)``.
    """
    if series is None or trades <= 0 or len(series) < 2:
        return 0.0, None, None
    x = np.asarray(series.values, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return 0.0, None, None
    if x.max() == x.min():  # exactly constant (its float std is rounding noise, not 0)
        return 0.0, None, None
    with np.errstate(all="ignore"):
        t_stat = float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(len(x)))
        train_return = float(np.prod(1.0 + x)) - 1.0
    import math

    if not math.isfinite(t_stat):
        return 0.0, None, None
    shrunk = t_stat * trades / (trades + min_trades)
    factor = _v2_window_factor(window_returns)
    if train_return > 0:
        mapped = 0.5 + 0.5 * shrunk / (shrunk + FITNESS_V2_SCALE)
    else:
        n_windows = 1 if window_returns is None else len(window_returns)
        cap = 0.5 * _V2_MIN_PENALTY_MULTIPLIER / max(1, n_windows - 1)
        # the exponent is clamped: a near-constant losing series has an enormous t-stat
        mapped = cap / (1.0 + math.exp(min(-shrunk / FITNESS_V2_SCALE, 700.0)))
    return factor * ((1.0 - penalty) * mapped), t_stat, factor


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


def _fixed_pct(level: object) -> float | None:
    """The percent of a ``fixed_pct`` stop/target, else None (ATR, risk-reward, ...)."""
    if not isinstance(level, dict) or level.get("type") != "fixed_pct":
        return None
    percent = level.get("percent")
    if isinstance(percent, bool) or not isinstance(percent, (int, float)):
        return None
    return float(percent)


def _dsl_sl_tp_penalty(dsl: dict[str, object]) -> float:
    """SL/TP imbalance penalty on a rendered DSL: worst traded side, generic fallback.

    A side whose stop AND target are both ``fixed_pct`` is scored; any other type
    has no percent ratio, so that side costs nothing. Each side falls back to the
    generic ``stop_loss`` / ``take_profit`` when its directional override is unset.
    """
    entry = dsl.get("entry_conditions")
    sides = [side for side in ("long", "short") if isinstance(entry, dict) and entry.get(side)]
    worst = 0.0
    for side in sides:
        sl = _fixed_pct(dsl.get(f"stop_loss_{side}") or dsl.get("stop_loss"))
        tp = _fixed_pct(dsl.get(f"take_profit_{side}") or dsl.get("take_profit"))
        if sl is None or tp is None:
            continue
        worst = max(worst, compute_sl_tp_penalty(sl, tp))
    return worst


def chromosome_sl_tp_penalty(chrom: StrategyChromosome | ParamGenome) -> float:
    """SL/TP imbalance penalty on the SL/TP values the strategy actually trades.

    A ``ParamGenome`` and a genome with exit rules are scored on their rendered DSL
    (``_dsl_sl_tp_penalty``): the DSL is what trades, so ``safety`` / ``none`` render as
    fixed_pct and ARE scored, ATR / risk:reward sides cost nothing.

    ``chromosome_to_dsl`` emits per-direction ``stop_loss_long`` /
    ``take_profit_short`` ... overrides for direction=BOTH, so the generic
    ``stop_loss_pct``/``take_profit_pct`` are not what a BOTH strategy uses.
    Penalize the worst side (each side falls back to the generic value when
    its override is unset, matching the DSL).
    """
    if isinstance(chrom, ParamGenome):
        return _dsl_sl_tp_penalty(chrom.to_dsl())
    if chrom.exit_rules is not None:
        return _dsl_sl_tp_penalty(chromosome_to_dsl(chrom))
    direction = getattr(chrom.direction, "value", chrom.direction)
    if direction != "both":
        return compute_sl_tp_penalty(chrom.stop_loss_pct, chrom.take_profit_pct)

    def _or(value: float | None, fallback: float) -> float:
        return fallback if value is None else value

    long_pen = compute_sl_tp_penalty(
        _or(chrom.stop_loss_long_pct, chrom.stop_loss_pct),
        _or(chrom.take_profit_long_pct, chrom.take_profit_pct),
    )
    short_pen = compute_sl_tp_penalty(
        _or(chrom.stop_loss_short_pct, chrom.stop_loss_pct),
        _or(chrom.take_profit_short_pct, chrom.take_profit_pct),
    )
    return max(long_pen, short_pen)


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


def _const_backtest(
    metrics: dict[str, float | int],
) -> Callable[[StrategyChromosome], dict[str, float | int]]:
    return lambda _chrom: metrics


def pooled_symbol_trade_floor(min_trades: int, n_symbols: int) -> int:
    """Trades every pooled coin needs: ``max(5, min_trades // (2N))`` (vibe-quant-clmmc.5).

    Without it a coin that never trades (an exactly flat series) could ride along in the pool:
    it only rescales the pooled series, which leaves the Sharpe and bootstrap t-stat unchanged.
    """
    return max(5, int(min_trades) // (2 * max(1, n_symbols)))


def soft_worst_score(symbol_scores: Sequence[float]) -> float:
    """Worst-mode GA rank score: ``0.5*min + 0.5*median`` of per-symbol fitness.

    A hard worst-of score is exactly 0 whenever any one symbol loses, which
    leaves the GA without a gradient; this keeps a pull toward the worst symbol
    while still rewarding strength elsewhere. Champions are NOT picked by this
    alone (see the worst-symbol train gate in the pipeline).
    """
    if not symbol_scores:
        return 0.0
    return 0.5 * min(symbol_scores) + 0.5 * statistics.median(symbol_scores)


def _evaluate_single(
    chrom: StrategyChromosome,
    backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]],
    filter_fn: Callable[[StrategyChromosome, dict[str, float | int]], dict[str, bool]]
    | None = None,
    min_trades: int = MIN_TRADES,
    timeframe: str | None = None,
    *,
    fitness: str = "v1",
) -> FitnessResult:
    """Evaluate fitness for a single chromosome (picklable for multiprocessing).

    ``fitness="v2"`` (bead vibe-quant-clmmc.3) scores the chained MTM series with
    :func:`fitness_v2_own` instead of the weighted v1 formula: an evaluation that cannot
    be scored honestly (a window skipped by an early exit, no series) becomes a
    zero-score ``FitnessV2Error`` row. ``"v1"`` is today's formula, byte for byte.
    """
    if fitness not in ("v1", "v2"):
        raise ValueError(f"unknown fitness {fitness!r} (expected 'v1' or 'v2')")
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
        logger.debug(
            "Backtest %s: %.1fs %s trades=%s",
            chrom.uid,
            _bt_elapsed,
            genome_summary(chrom),
            bt.get("total_trades", "?"),
        )
    except DataUnavailableError:
        raise
    except Exception as e:
        logger.warning("Backtest failed for chromosome %s, assigning zero fitness", chrom.uid, exc_info=True)
        return replace(_zero, error=f"{type(e).__name__}: {e}")

    if fitness == "v2" and _has_early_exit(bt):
        return replace(_zero, error=f"FitnessV2Error: {_V2_EARLY_EXIT}")

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
    if _math.isnan(total_return):
        total_return = 0.0
    if _math.isnan(pf):
        # NaN PF = no losing periods (gross loss 0). With trades and a profit
        # that is the BEST case -> score at the PF cap, not as PF=0.
        pf = PF_MAX if trades > 0 and total_return > 0 else 0.0

    # Sanity checks on backtest output — flag impossible metric combinations.
    # Skipped for synthetic results (early exits / failed backtests): their
    # metrics are deliberately forced and always look inconsistent.
    if not {"early_exit", "early_exit_symbol", "error", "synthetic_symbols"} & bt.keys():
        _sanity_check_metrics(chrom.uid, sharpe, max_dd, pf, trades, total_return)

    num_genes = scalar_gene_count(chrom)
    complexity_pen = compute_complexity_penalty(num_genes)
    overtrade_pen = compute_overtrade_penalty(trades, timeframe)
    sl_tp_pen = chromosome_sl_tp_penalty(chrom)

    # Filter evaluation
    filter_results = filter_fn(chrom, bt) if filter_fn is not None else {}
    passed_filters = all(filter_results.values()) if filter_results else True

    # Compute raw score including total return
    raw = compute_fitness_score(sharpe, max_dd, pf, total_return)

    pooled_stats = cast("dict[str, tuple[float, int]] | None", bt.get("pooled_symbol_stats"))
    pooled_symbol_stats = (
        {sym: (float(r), int(t)) for sym, (r, t) in pooled_stats.items()}
        if pooled_stats is not None
        else None
    )

    symbol_scores: dict[str, float] | None = None
    symbol_stats: dict[str, tuple[float, int]] | None = None
    fitness_v2: FitnessV2 | None = None
    sym_results = cast("dict[str, dict[str, float | int]] | None", bt.get("symbol_results"))
    if fitness == "v2":
        try:
            adjusted, fitness_v2, symbol_scores, symbol_stats = _score_v2(
                bt, trades, min_trades, timeframe, complexity_pen, sl_tp_pen, pooled_symbol_stats,
                total_return,
            )
        except FitnessV2Error as exc:
            return replace(_zero, error=f"FitnessV2Error: {exc}")
    # v1 hard gates: insufficient trades or negative return → zero fitness
    # (pooled: also every coin's trade floor)
    elif not _clears_train_gate(trades, total_return, None, min_trades, pooled_symbol_stats):
        adjusted = 0.0
    else:
        adjusted = max(0.0, raw - complexity_pen - overtrade_pen - sl_tp_pen)

    if fitness == "v1" and sym_results:
        # symbol_agg="worst": rank by the soft per-symbol score. Each symbol is
        # scored exactly like a standalone run (own hard gates and penalties);
        # the reported metrics above stay worst-of.
        symbol_scores = {}
        symbol_stats = {}
        for sym, sbt in sym_results.items():
            sres = _evaluate_single(chrom, _const_backtest(sbt), None, min_trades, timeframe)
            symbol_scores[sym] = sres.adjusted_score
            symbol_stats[sym] = (sres.total_return, sres.total_trades)
        adjusted = soft_worst_score(list(symbol_scores.values()))

    # Keep the DSR series only for a champion candidate: every symbol (worst
    # mode) or the run (portfolio) clears the train hard gates.
    keep_series = _clears_train_gate(
        trades, total_return, symbol_stats, min_trades, pooled_symbol_stats
    )

    # Log score decomposition for debugging fitness calculation correctness
    if fitness_v2 is None and adjusted > 0:
        if symbol_scores is not None:
            logger.debug(
                "Score %s: soft=0.5*min+0.5*median of %s = %.4f",
                chrom.uid, {sym: round(sc, 4) for sym, sc in symbol_scores.items()}, adjusted,
            )
        else:
            logger.debug(
                "Score %s: raw=%.4f - complexity=%.4f - overtrade=%.4f - sl_tp=%.4f = adjusted=%.4f "
                "(sharpe=%.2f dd=%.3f pf=%.2f ret=%.3f trades=%d genes=%d %s)",
                chrom.uid, raw, complexity_pen, overtrade_pen, sl_tp_pen, adjusted,
                sharpe, max_dd, pf, total_return, trades, num_genes,
                genome_summary(chrom),
            )

    fee_drag_val, fees_pct_val = _fee_drag_fields(bt)
    liquidation_count_val, liquidation_forfeited_val = _liquidation_fields(bt)

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
        error=str(bt["error"]) if bt.get("error") else None,
        symbol_scores=symbol_scores,
        symbol_stats=symbol_stats,
        mtm_daily_returns=cast("DailyReturns | None", bt.get("mtm_daily_returns"))
        if keep_series
        else None,
        symbol_mtm_daily=cast("dict[str, DailyReturns | None] | None", bt.get("symbol_mtm_daily"))
        if keep_series
        else None,
        fee_drag=fee_drag_val,
        fees_pct_equity_yr=fees_pct_val,
        liquidation_count=liquidation_count_val,
        liquidation_forfeited=liquidation_forfeited_val,
        pooled_symbol_stats=pooled_symbol_stats,
        fitness_v2=fitness_v2,
    )


_V2_EARLY_EXIT = "early exit under fitness v2"


def _has_early_exit(bt: dict[str, float | int]) -> bool:
    """A bt (or, in worst mode, one of its symbols) that stopped before its last window."""
    if "early_exit" in bt:
        return True
    sym_results = cast("dict[str, dict[str, float | int]]", bt.get("symbol_results") or {})
    return any("early_exit" in sbt for sbt in sym_results.values())


def _v2_own_of(
    series: DailyReturns | None,
    trades: int,
    min_trades: int,
    window_returns: Sequence[float] | None,
    penalty: float,
    total_return: float,
) -> tuple[float, float | None, float | None]:
    """:func:`fitness_v2_own` that fails loud on a missing series (a zero-trade run has none).

    A run that went bankrupt (``total_return <= -1.0``) has trades but no series either (the
    balance hit zero, ``mark_to_market_daily_returns`` returns None): a quiet 0, like v1.
    """
    if series is None and trades > 0:
        if total_return <= -1.0:
            return 0.0, None, None
        raise FitnessV2Error("no MTM series")
    return fitness_v2_own(series, trades, min_trades, window_returns, penalty)


def _score_v2(
    bt: dict[str, float | int],
    trades: int,
    min_trades: int,
    timeframe: str | None,
    complexity_pen: float,
    sl_tp_pen: float,
    pooled_symbol_stats: dict[str, tuple[float, int]] | None,
    total_return: float,
) -> tuple[
    float, FitnessV2, dict[str, float] | None, dict[str, tuple[float, int]] | None
]:
    """``(adjusted, FitnessV2, symbol_scores, symbol_stats)`` of a v2 evaluation.

    Portfolio and pooled: one own score of the run's series (pooled: a coin under the
    trade floor scores 0). Worst mode: each symbol is scored on its own series, trades
    and windows with its own overtrade penalty; the GA score is :func:`soft_worst_score`
    of them. Raises :class:`FitnessV2Error` on a missing series.
    """
    import math

    sym_results = cast("dict[str, dict[str, float | int]] | None", bt.get("symbol_results"))
    if sym_results:
        sym_series = cast("dict[str, DailyReturns | None]", bt.get("symbol_mtm_daily") or {})
        scored: dict[str, tuple[float, float | None, float | None]] = {}
        symbol_stats: dict[str, tuple[float, int]] = {}
        for sym, sbt in sym_results.items():
            s_trades = int(sbt["total_trades"])
            s_return = float(sbt.get("total_return", 0.0))
            penalty = complexity_pen + compute_overtrade_penalty(s_trades, timeframe) + sl_tp_pen
            s_return = 0.0 if math.isnan(s_return) else s_return
            scored[sym] = _v2_own_of(
                sym_series.get(sym),
                s_trades,
                min_trades,
                cast("Sequence[float] | None", sbt.get("window_returns")),
                penalty,
                s_return,
            )
            symbol_stats[sym] = (s_return, s_trades)
        symbol_scores = {sym: own for sym, (own, _t, _f) in scored.items()}
        adjusted = soft_worst_score(list(symbol_scores.values()))
        worst = scored[min(symbol_scores, key=lambda sym: symbol_scores[sym])]
        return adjusted, FitnessV2(adjusted, worst[1], worst[2]), symbol_scores, symbol_stats
    if not _pooled_floor_ok(min_trades, pooled_symbol_stats):
        return 0.0, FitnessV2(0.0, None, None), None, None
    penalty = complexity_pen + compute_overtrade_penalty(trades, timeframe) + sl_tp_pen
    own, t_stat, factor = _v2_own_of(
        cast("DailyReturns | None", bt.get("mtm_daily_returns")),
        trades,
        min_trades,
        cast("Sequence[float] | None", bt.get("window_returns")),
        penalty,
        total_return,
    )
    return own, FitnessV2(own, t_stat, factor), None, None


def _pooled_floor_ok(
    min_trades: int, pooled_symbol_stats: dict[str, tuple[float, int]] | None
) -> bool:
    """Every pooled coin has :func:`pooled_symbol_trade_floor` trades (True outside pooled mode)."""
    if pooled_symbol_stats is None:
        return True
    floor = pooled_symbol_trade_floor(min_trades, len(pooled_symbol_stats))
    return all(t >= floor for _r, t in pooled_symbol_stats.values())


def _clears_train_gate(
    trades: int,
    total_return: float,
    symbol_stats: dict[str, tuple[float, int]] | None,
    min_trades: int,
    pooled_symbol_stats: dict[str, tuple[float, int]] | None = None,
) -> bool:
    """The train hard gate: every symbol (worst mode) or the run (portfolio) clears it.

    Pooled mode: the pooled run clears it AND every coin has
    :func:`pooled_symbol_trade_floor` trades.
    """
    if symbol_stats is not None:
        return all(r > 0 and t >= min_trades for r, t in symbol_stats.values())
    if not _pooled_floor_ok(min_trades, pooled_symbol_stats):
        return False
    return trades >= min_trades and total_return > 0


def passes_train_gate(fr: FitnessResult, min_trades: int) -> bool:
    """Whether a scored genome is a champion candidate (keeps its DSR series).

    ``--gate-pool all`` (vibe-quant-clmmc.2) screens exactly these genomes,
    whatever their adjusted score.
    """
    return _clears_train_gate(
        fr.total_trades, fr.total_return, fr.symbol_stats, min_trades, fr.pooled_symbol_stats
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


@contextmanager
def preserve_global_random() -> Iterator[None]:
    """Restore the global ``random`` state on exit (NT's FillModel reseeds it, vibe-quant-0bwe2/qei3p)."""
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


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
    fitness: str = "v1",
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
        fitness: ``"v1"`` (default) or ``"v2"`` (:func:`fitness_v2_own`).

    Returns:
        FitnessResult for each chromosome, parallel to input.
    """
    global _parallel_broken  # noqa: PLW0603
    if max_workers is not None and max_workers != 1 and not _parallel_broken:
        try:
            return _evaluate_parallel(chromosomes, backtest_fn, filter_fn, max_workers, executor, min_trades=min_trades, timeframe=timeframe, fitness=fitness)
        except DataUnavailableError:
            raise
        except Exception:
            _parallel_broken = True
            logger.warning("Parallel evaluation failed, falling back to sequential for remainder of run", exc_info=True)

    with preserve_global_random():
        return [_evaluate_single(chrom, backtest_fn, filter_fn, min_trades=min_trades, timeframe=timeframe, fitness=fitness) for chrom in chromosomes]


_TASK_ERROR = "__task_error__"
_TASK_DATA_UNAVAILABLE = "__task_data_unavailable__"


def _evaluate_single_task(
    chrom: StrategyChromosome,
    backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]],
    filter_fn: Callable[[StrategyChromosome, dict[str, float | int]], dict[str, bool]] | None,
    min_trades: int,
    timeframe: str | None,
    fitness: str = "v1",
) -> tuple[FitnessResult, dict[str, object]]:
    """Pool task: one chromosome plus the worker's cumulative fallback snapshot.

    ``_evaluate_single`` is looked up as a module global so tests that
    monkeypatch it still apply.
    """
    return (
        _evaluate_single(chrom, backtest_fn, filter_fn, min_trades, timeframe, fitness=fitness),
        fallbacks.snapshot(),
    )


def _eval_symbol_task(
    fn: NTBacktestFn, chrom: StrategyChromosome, symbol: str
) -> tuple[dict[str, float | int], dict[str, object]]:
    """Worker task: one chromosome on one symbol (all windows, ``NTBacktestFn.eval_symbol``). Picklable.

    Exceptions never cross the pool boundary raw: one that cannot be unpickled
    in the parent marks the WHOLE pool broken (every chromosome of the
    generation would error). Failures come back as tagged dicts and the parent
    decides, in symbol order.
    """
    try:
        return fn.eval_symbol(chrom, symbol), fallbacks.snapshot()
    except Exception as e:
        desc = f"{type(e).__name__}: {e}"
        tagged: dict[str, float | int] = {_TASK_ERROR: desc}  # type: ignore[dict-item]
        if isinstance(e, DataUnavailableError):
            tagged[_TASK_DATA_UNAVAILABLE] = str(e)  # type: ignore[assignment]
        return tagged, fallbacks.snapshot()


def _evaluate_parallel(
    chromosomes: list[StrategyChromosome],
    backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]],
    filter_fn: Callable[[StrategyChromosome, dict[str, float | int]], dict[str, bool]] | None,
    max_workers: int | None,
    executor: object | None = None,
    min_trades: int = MIN_TRADES,
    timeframe: str | None = None,
    fitness: str = "v1",
) -> list[FitnessResult]:
    """Evaluate population using ProcessPoolExecutor.

    If `executor` is provided, it's reused (caller manages lifecycle).
    Otherwise a temporary pool is created and destroyed.
    """
    import os
    from concurrent.futures import Executor, ProcessPoolExecutor, as_completed

    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    workers = max_workers if max_workers and max_workers > 0 else os.cpu_count() or 4
    split_fn = backtest_fn if isinstance(backtest_fn, NTBacktestFn) and backtest_fn.splits_by_symbol else None
    workers = min(workers, len(chromosomes) * (len(split_fn.symbols) if split_fn else 1))

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

    def _run_split(pool: Executor, fn: NTBacktestFn) -> None:
        # One task per (chromosome, symbol): a slow chromosome no longer pins
        # a worker for all of its symbols. Failures are collected, then decided
        # per chromosome in symbol order (the first failure is the one the
        # sequential path would have hit): DataUnavailableError propagates,
        # anything else gives the whole chromosome today's error result.
        future_to_key = {
            pool.submit(_eval_symbol_task, fn, chrom, sym): (i, j)
            for i, chrom in enumerate(chromosomes)
            for j, sym in enumerate(fn.symbols)
        }
        per_chrom: list[list[dict[str, float | int] | None]] = [
            [None] * len(fn.symbols) for _ in chromosomes
        ]
        for future in as_completed(future_to_key):
            i, j = future_to_key[future]
            try:
                per_chrom[i][j], fsnap = future.result()
                fallbacks.record_worker(fsnap)
            except Exception as e:  # pool-level failure (e.g. broken worker)
                per_chrom[i][j] = {_TASK_ERROR: f"{type(e).__name__}: {e}"}  # type: ignore[dict-item]
        for i, chrom in enumerate(chromosomes):
            outcomes = per_chrom[i]
            failure = next((o for o in outcomes if o is not None and _TASK_ERROR in o), None)
            if failure is not None:
                if _TASK_DATA_UNAVAILABLE in failure:
                    raise DataUnavailableError(str(failure[_TASK_DATA_UNAVAILABLE]))
                agg = fn.error_result(chrom, str(failure[_TASK_ERROR]))
            else:
                # the same reduction as NTBacktestFn.__call__ (worst or pooled)
                agg = fn.aggregate_symbol_results(
                    chrom, cast("list[dict[str, float | int]]", outcomes)
                )
            results[i] = _evaluate_single(
                chrom, _const_backtest(agg), filter_fn, min_trades, timeframe, fitness=fitness
            )

    def _run_with(pool: Executor) -> None:
        if split_fn is not None:
            _run_split(pool, split_fn)
            return
        future_to_idx = {
            pool.submit(
                _evaluate_single_task, chrom, backtest_fn, filter_fn, min_trades, timeframe, fitness
            ): i
            for i, chrom in enumerate(chromosomes)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx], fsnap = future.result()
                fallbacks.record_worker(fsnap)
            except DataUnavailableError:
                raise
            except Exception as e:
                logger.warning("Parallel eval failed for chromosome %d", idx, exc_info=True)
                results[idx] = replace(_zero, error=f"{type(e).__name__}: {e}")

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
