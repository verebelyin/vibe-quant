"""Genetic discovery pipeline for trading strategy evolution.

Orchestrates population initialization, fitness evaluation, selection,
crossover, mutation, and convergence detection to discover profitable
strategy candidates expressed as DSL YAML dicts.

Champion gates (vibe-quant-e70tl.5) all FAIL CLOSED: a candidate that fails a
gate is dropped and recorded in ``guardrail_rejections`` with the reason; when
every candidate fails, the run persists ZERO champions. Order:

1. guardrails (min trades/return, complexity, DSR, bootstrap CI) on top-K
2. cross-window: shifted sub-windows of the TRAIN range (in-sample window
   never counts as a pass)
3. WFA rolling: rolling sub-windows of the TRAIN range
4. holdout: one final out-of-sample pass/fail gate (never used for ranking)

No cross-window / WFA window may overlap the holdout.
"""

from __future__ import annotations

import json
import logging
import math
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.discovery.fitness import FitnessResult, evaluate_population
from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.discovery.guardrails import GuardrailConfig, GuardrailResult, apply_guardrails
from vibe_quant.discovery.operators import (
    StrategyChromosome,
    _random_chromosome,
    apply_elitism,
    crossover,
    crowding_replace,
    initialize_population,
    is_valid_chromosome,
    mutate,
    tournament_select,
)
from vibe_quant.dsl.identity import dsl_body_key
from vibe_quant.errors import DataUnavailableError
from vibe_quant.utils import compute_day_count

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from vibe_quant.discovery.operators import Direction

logger = logging.getLogger(__name__)

# Max retries when generating valid offspring via crossover+mutation
_MAX_OFFSPRING_RETRIES: int = 10

# Shortest shifted cross-window (days) worth backtesting
_MIN_CROSS_WINDOW_DAYS: int = 7


class DiscoveryEvaluationError(RuntimeError):
    """Every evaluation in a batch raised: systemic failure, not bad strategies."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    """Configuration for the genetic discovery pipeline.

    Attributes:
        population_size: Number of individuals per generation.
        max_generations: Maximum evolutionary generations.
        mutation_rate: Per-gene mutation probability.
        crossover_rate: Probability of crossover per pair (0-1). If < 1, parents
            are copied directly with probability (1 - crossover_rate).
        elite_count: Number of top individuals preserved unchanged.
        tournament_size: Tournament selection pool size.
        convergence_generations: Stop after N generations with no improvement.
        top_k: Number of top strategies to output.
        min_trades: Minimum trades for a strategy to be considered valid.
        symbols: Trading symbols to evaluate on.
        timeframe: Bar timeframe (e.g. "1h").
        start_date: TRAIN range start (ISO format). With a holdout this is the
            training slice only.
        end_date: TRAIN range end (ISO format) == holdout start when split.
        eval_windows: Worst-of-N sub-window fitness (see
            ``NTBacktestFn._aggregate_multi_window``); 1 = single window.
        symbol_agg: ``portfolio`` (one shared-account run) or ``worst`` (per-symbol
            runs, worst symbol scored); informational here -- the backtest fns
            are built with the same mode by the CLI.
        train_test_split: TRAIN fraction of the full range (0 = no holdout).
            The CLI/API default is 0.8 (20% holdout).
        cross_window_min_pass: Shifted cross-windows that must pass. ``None``
            = all of them. The in-sample window never counts.
        wfa_oos_step_days: >0 enables the rolling-window stability check over
            the TRAIN range (``wfa_min_consistency`` of windows profitable).
        holdout_min_sharpe: Holdout Sharpe must exceed this.
        holdout_min_trades: Holdout trade floor. ``None`` = derived: half the
            trade rate the train gate demands, ``max(1, min_trades *
            holdout_days / (2 * train_days))``.
    """

    population_size: int = 20
    max_generations: int = 15
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_count: int = 2
    tournament_size: int = 3
    convergence_generations: int = 3
    top_k: int = 5
    min_trades: int = 50
    max_workers: int | None = 0  # 0 = auto (cpu_count), None = sequential
    symbols: list[str] = field(default_factory=list)
    timeframe: str = "4h"
    start_date: str = ""
    end_date: str = ""
    indicator_pool: list[str] | None = None  # None = use all available
    direction: str | None = None  # "long", "short", "both", or None (random)
    use_crowding: bool = True  # Use deterministic crowding (True) or classic tournament (False)
    immigrant_fraction: float = 0.15  # Fraction of population replaced when entropy is low
    entropy_threshold: float = 0.4  # Entropy below this triggers immigrant injection
    min_diversity_distance: float = 0.15  # Min Gower distance for top-K dedup
    eval_windows: int = 3  # worst-of-N sub-window fitness; 1 = single-window
    symbol_agg: str = "portfolio"  # "worst" = score by worst symbol (NTBacktestFn)
    train_test_split: float = 0.0  # 0 = no holdout; >0 = TRAIN fraction (e.g. 0.8)
    holdout_start_date: str = ""  # Pre-computed holdout start (set by CLI, not re-split)
    holdout_end_date: str = ""  # Pre-computed holdout end
    cross_window_months: list[int] = field(default_factory=list)  # shifted windows, e.g. [1, 2]
    cross_window_min_pass: int | None = None  # shifted windows that must pass (None = all)
    cross_window_min_sharpe: float = 0.5  # min Sharpe on each window to count as pass
    wfa_oos_step_days: int = 0  # >0 enables rolling-window stability check over TRAIN range
    wfa_min_consistency: float = 0.75  # fraction of rolling windows that must be profitable
    holdout_min_sharpe: float = 0.0  # holdout gate: Sharpe must exceed this
    holdout_min_trades: int | None = None  # holdout gate trade floor (None = derived)
    require_bootstrap_ci: bool = True  # Bootstrap Sharpe CI guardrail
    bootstrap_min_sharpe: float = 1.0  # Reject if CI lower bound < this
    bootstrap_ci_level: float = 0.95  # Confidence level for bootstrap CI
    require_dsr: bool = True  # Deflated Sharpe Ratio guardrail

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.population_size < 2:
            errors.append("population_size must be >= 2")
        if self.max_generations < 1:
            errors.append("max_generations must be >= 1")
        if not (0.0 <= self.mutation_rate <= 1.0):
            errors.append("mutation_rate must be in [0, 1]")
        if not (0.0 <= self.crossover_rate <= 1.0):
            errors.append("crossover_rate must be in [0, 1]")
        if self.elite_count < 0:
            errors.append("elite_count must be >= 0")
        if self.elite_count >= self.population_size:
            errors.append("elite_count must be < population_size")
        if self.tournament_size < 1:
            errors.append("tournament_size must be >= 1")
        if self.convergence_generations < 1:
            errors.append("convergence_generations must be >= 1")
        # Cap convergence_gens so early stop can trigger before max_gen
        # _check_convergence requires 2*n gens, so cap at max_gen // 2
        max_conv = max(1, self.max_generations // 2)
        if self.convergence_generations > max_conv:
            object.__setattr__(self, "convergence_generations", max_conv)
        if self.top_k < 1:
            errors.append("top_k must be >= 1")
        if self.train_test_split < 0.0 or self.train_test_split >= 1.0:
            errors.append("train_test_split must be in [0, 1)")
        if self.cross_window_min_pass is not None and self.cross_window_min_pass < 1:
            errors.append("cross_window_min_pass must be >= 1 (or None = all shifted windows)")
        if self.holdout_min_trades is not None and self.holdout_min_trades < 0:
            errors.append("holdout_min_trades must be >= 0")
        if errors:
            raise ValueError("; ".join(errors))

        # Auto-set min_trades for sub-5m timeframes if left at default (50).
        # 1m strategies need 100+ trades for statistical significance (bd-yu02).
        if self.min_trades == 50 and self.timeframe in ("1m", "2m", "3m"):
            from vibe_quant.discovery.fitness import MIN_TRADES_1M

            object.__setattr__(self, "min_trades", MIN_TRADES_1M)

    @property
    def has_holdout(self) -> bool:
        """True when a holdout slice was split off the discovery range."""
        return (
            self.train_test_split > 0
            and bool(self.holdout_start_date)
            and bool(self.holdout_end_date)
        )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Metrics for a single generation.

    Attributes:
        generation: Generation index (0-based).
        best_fitness: Highest adjusted_score in this generation.
        mean_fitness: Mean adjusted_score across the population.
        worst_fitness: Lowest adjusted_score in this generation.
        best_chromosome: Chromosome with the highest fitness.
        population_size: Number of individuals evaluated.
        num_passed_filters: Count of individuals that passed overfitting filters.
    """

    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    best_chromosome: StrategyChromosome
    population_size: int
    num_passed_filters: int


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    """Metrics of one strategy on one (holdout / shifted / rolling) window.

    Attributes:
        sharpe_ratio: Sharpe ratio.
        max_drawdown: Max drawdown.
        profit_factor: Profit factor.
        total_trades: Trade count.
        total_return: Total return.
    """

    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    total_return: float
    # symbol_agg="worst" only: portfolio-comparable trade sum and per-symbol
    # {sharpe, trades, return} (total_trades above is the MIN symbol).
    trades_sum: int | None = None
    symbol_metrics: dict[str, dict[str, float | int]] | None = None


@dataclass(frozen=True, slots=True)
class CrossWindowResult:
    """Cross-window validation result for a single strategy.

    Attributes:
        window_results: Per SHIFTED window metrics (the in-sample training
            window is excluded -- it never counts as a pass).
        windows_passed: Number of shifted windows that passed.
        total_windows: Number of shifted windows evaluated.
        passed: Whether ``windows_passed >= required`` shifted windows.
        window_dates: (start, end) per shifted window, parallel to results.
        offsets_months: Month offset per shifted window, parallel to results.
        required: Shifted windows that had to pass.
    """

    window_results: list[HoldoutResult]
    windows_passed: int
    total_windows: int
    passed: bool
    window_dates: list[tuple[str, str]] = field(default_factory=list)
    offsets_months: list[int] = field(default_factory=list)
    required: int = 0


@dataclass(frozen=True, slots=True)
class WFARollingResult:
    """Rolling-window stability check (over the TRAIN range) for one strategy.

    Attributes:
        oos_windows: Per-window HoldoutResult for each rolling window.
        window_dates: (start, end) for each rolling window.
        windows_profitable: Number of windows with total_return > 0.
        windows_sharpe_positive: Number of windows with sharpe > 0.
        total_windows: Total rolling windows.
        consistency: Fraction of profitable (return-based) windows — the gate.
        sharpe_consistency: Fraction of sharpe-positive windows — exposure only.
        passed: Whether consistency >= wfa_min_consistency (return-based).
    """

    oos_windows: list[HoldoutResult]
    window_dates: list[tuple[str, str]]
    windows_profitable: int
    windows_sharpe_positive: int
    total_windows: int
    consistency: float
    sharpe_consistency: float
    passed: bool


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Final output of the discovery pipeline.

    Attributes:
        generations: Per-generation metrics.
        top_strategies: Champions that passed EVERY enabled gate, sorted by
            training score (may be empty).
        total_candidates_evaluated: Distinct strategies backtested (DSR N).
        converged: Whether the pipeline terminated due to convergence.
        convergence_generation: Generation index where convergence detected (None if not).
        holdout_results: Per-champion holdout metrics (parallel to top_strategies).
            Empty if no holdout.
        train_dates: (start, end) for train period. None if no split.
        holdout_dates: (start, end) for holdout period. None if no split.
        guardrail_rejections: One entry per rejected candidate with the
            failing ``stage`` and ``reasons``.
    """

    generations: list[GenerationResult]
    top_strategies: list[tuple[StrategyChromosome, FitnessResult]]
    total_candidates_evaluated: int
    converged: bool
    convergence_generation: int | None
    holdout_results: list[HoldoutResult] = field(default_factory=list)
    train_dates: tuple[str, str] | None = None
    holdout_dates: tuple[str, str] | None = None
    cross_window_results: list[CrossWindowResult] = field(default_factory=list)
    wfa_results: list[WFARollingResult] = field(default_factory=list)
    guardrail_rejections: list[dict[str, object]] = field(default_factory=list)
    holdout_min_trades: int | None = None


def _select_diverse_top_k(
    scored: Sequence[tuple[StrategyChromosome, FitnessResult | float]],
    top_k: int = 5,
    min_distance: float = 0.15,
) -> list[tuple[StrategyChromosome, FitnessResult | float]]:
    """Select top-K strategies with diversity enforcement.

    Iterates through candidates sorted by fitness (descending). A candidate
    is added only if its distance to ALL already-selected strategies exceeds
    min_distance.

    Args:
        scored: List of (chromosome, fitness) tuples, sorted by fitness desc.
        top_k: Maximum number of strategies to select.
        min_distance: Minimum Gower distance to all selected strategies.

    Returns:
        List of up to top_k diverse (chromosome, fitness) tuples.
    """
    from vibe_quant.discovery.distance import chromosome_distance

    selected: list[tuple[StrategyChromosome, FitnessResult | float]] = []

    for chrom, fitness in scored:
        if len(selected) >= top_k:
            break

        is_diverse = all(
            chromosome_distance(chrom, sel_chrom) >= min_distance
            for sel_chrom, _ in selected
        )

        if is_diverse:
            selected.append((chrom, fitness))

    return selected


def worst_symbol_gate_reasons(fitness: FitnessResult, min_trades: int) -> list[str]:
    """Hard train gate for ``symbol_agg="worst"`` champions (vibe-quant-ox73t).

    The soft GA score lets a genome with a losing symbol rank; a champion must
    still have EVERY symbol profitable (train, worst-of-windows) with at least
    ``min_trades`` trades. Empty list = pass (also for portfolio mode).
    """
    if fitness.symbol_stats is None:
        return []
    reasons: list[str] = []
    for sym, (ret, trades) in fitness.symbol_stats.items():
        if not ret > 0:
            reasons.append(f"worst-symbol train return <= 0 ({sym})")
        if trades < min_trades:
            reasons.append(f"worst-symbol train trades < {min_trades} ({sym})")
    return reasons


def _metrics_to_holdout_result(bt: dict[str, float | int]) -> HoldoutResult:
    """Window metrics dict -> HoldoutResult, NaN coerced to failing values."""
    sharpe = float(bt.get("sharpe_ratio", 0.0))
    max_dd = float(bt.get("max_drawdown", 1.0))
    pf = float(bt.get("profit_factor", 0.0))
    trades = int(bt.get("total_trades", 0))
    ret = float(bt.get("total_return", 0.0))
    return HoldoutResult(
        sharpe_ratio=0.0 if math.isnan(sharpe) else sharpe,
        max_drawdown=1.0 if math.isnan(max_dd) else max_dd,
        profit_factor=0.0 if math.isnan(pf) else pf,
        total_trades=trades,
        total_return=0.0 if math.isnan(ret) else ret,
        trades_sum=int(bt["trades_sum"]) if "trades_sum" in bt else None,
        symbol_metrics=bt.get("symbol_metrics"),  # type: ignore[arg-type]
    )


_FAILED_WINDOW = HoldoutResult(
    sharpe_ratio=0.0, max_drawdown=1.0, profit_factor=0.0, total_trades=0, total_return=0.0,
)


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def assert_windows_outside_holdout(
    windows: Sequence[tuple[str, str]], holdout_start: str, label: str
) -> None:
    """Raise if any [start, end) window reaches past ``holdout_start``.

    Windows are half-open: a window ending exactly at the holdout start (the
    train end) does not overlap it.
    """
    if not holdout_start:
        return
    for ws, we in windows:
        if we > holdout_start or ws >= holdout_start:
            msg = (
                f"{label} window {ws}..{we} overlaps the holdout starting "
                f"{holdout_start} -- the holdout must stay unseen until the final gate"
            )
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DiscoveryPipeline:
    """Genetic algorithm pipeline for strategy discovery.

    Args:
        config: Pipeline configuration.
        backtest_fn: Callable that runs a backtest for a chromosome and returns
            a dict with keys: sharpe_ratio, max_drawdown, profit_factor, total_trades.
        filter_fn: Optional callable for overfitting filter evaluation.
        holdout_backtest_fn: Backtest over the holdout range (final gate).
        backtest_fn_factory: ``(start, end) -> backtest_fn`` for cross-window /
            rolling windows (all inside the TRAIN range).
    """

    def __init__(
        self,
        config: DiscoveryConfig,
        backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]],
        filter_fn: Callable[[StrategyChromosome, dict[str, float | int]], dict[str, bool]]
        | None = None,
        progress_file: str | Path | None = None,
        holdout_backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]] | None = None,
        backtest_fn_factory: Callable[[str, str], Callable[[StrategyChromosome], dict[str, float | int]]] | None = None,
        seed_chromosomes: list[StrategyChromosome] | None = None,
    ) -> None:
        self.config = config
        self._backtest_fn = backtest_fn
        self._filter_fn = filter_fn
        self._progress_file = Path(progress_file) if progress_file else None
        self._holdout_backtest_fn = holdout_backtest_fn
        self._backtest_fn_factory = backtest_fn_factory
        self._seed_chromosomes = seed_chromosomes
        self._guardrail_rejections: list[dict[str, object]] = []
        self._direction_constraint: Direction | None = None
        # Evaluation bookkeeping (reset per run)
        self._fitness_cache: dict[str, FitnessResult] = {}
        self._total_evaluated: int = 0
        self._all_scored: list[tuple[StrategyChromosome, FitnessResult]] = []
        self._executor: ProcessPoolExecutor | None = None
        self._ok_evals: int = 0  # evaluations without `error` this run

    # -- public API ---------------------------------------------------------

    @staticmethod
    def _create_executor(
        max_workers: int | None, population_size: int
    ) -> ProcessPoolExecutor | None:
        """Create a reusable worker pool for the entire discovery run.

        Returns None if parallelism is disabled (max_workers=None or 1)
        or if pool creation fails (macOS sandbox, Rust runtime conflicts).
        Falls back to per-generation pool creation in evaluate_population.
        """
        if max_workers is None or max_workers == 1:
            return None
        import os

        workers = max_workers if max_workers > 0 else (os.cpu_count() or 4)
        workers = min(workers, population_size)
        try:
            return ProcessPoolExecutor(max_workers=workers)
        except (OSError, RuntimeError):
            logger.warning("Failed to create shared worker pool, falling back to per-generation pools")
            return None

    def _apply_indicator_pool_filter(self) -> None:
        """Filter operators.INDICATOR_POOL to only include configured indicators.

        Safe to mutate module globals since discovery runs as a subprocess.
        """
        from vibe_quant.discovery.operators import (
            _INDICATOR_NAMES,
            INDICATOR_POOL,
            _ensure_pool,
        )

        _ensure_pool()
        if self.config.indicator_pool is None:
            # Default (all) excludes needs_context specs (FUNDING, ...): they
            # run only when named explicitly.
            from vibe_quant.dsl.indicators import indicator_registry

            allowed = {
                n
                for n in INDICATOR_POOL
                if (sp := indicator_registry.get(n)) is None or not sp.needs_context
            }
        else:
            allowed = set(self.config.indicator_pool)
        available = set(INDICATOR_POOL.keys())
        unknown = sorted(allowed - available)
        if unknown:
            # Moving-average-type plugins register with threshold_range=None and
            # are excluded from the GA pool (gene structure is indicator-vs-threshold).
            raise ValueError(
                f"indicator_pool contains names not available for GA discovery: "
                f"{unknown}. Available: {sorted(available)}. "
                f"(Indicators with threshold_range=None, e.g. price-vs-MA plugins, "
                f"are not yet supported by the genome — see bd-9c1g.)"
            )
        to_remove = [k for k in INDICATOR_POOL if k not in allowed]
        for k in to_remove:
            del INDICATOR_POOL[k]
        _INDICATOR_NAMES.clear()
        _INDICATOR_NAMES.extend(INDICATOR_POOL.keys())
        logger.info("Indicator pool filtered to: %s", list(INDICATOR_POOL.keys()))

    def _preflight_aux_data(self) -> None:
        """Fail before the GA starts if a context indicator in the pool lacks aux data.

        Covers the full range incl. holdout. Without this, only the context
        chromosomes would fail and the GA would quietly breed them out.
        """
        from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH
        from vibe_quant.discovery.operators import INDICATOR_POOL
        from vibe_quant.dsl import aux_data

        cfg = self.config
        end = cfg.holdout_end_date if cfg.has_holdout else cfg.end_date
        # Workers run NTScreeningRunner with default paths; check the same ones.
        aux_data.configure(aux_data.archive_path(), DEFAULT_CATALOG_PATH)
        aux_data.preflight(list(INDICATOR_POOL), cfg.symbols, cfg.start_date, end or cfg.end_date)

    def _preflight_bars(self) -> None:
        """Fail before the GA starts if the catalog lacks bars for a run window.

        Mirrors the GA workers' data setup (NTBacktestFn -> NTScreeningRunner):
        every window a worker will actually run must have catalog bars for
        every run symbol at the strategy timeframe -- the train fn's eval
        sub-windows (``fn.windows``, worst-of-N), the holdout fn's window, and
        the cross-window / WFA gate windows. ``require_bars_in_window`` only
        checks interval OVERLAP, so a covered full-span window can hide empty
        sub-windows; checking the exact worker windows catches that before the
        GA instead of raising MissingBarDataError inside a worker.

        MissingBarDataError (a DataUnavailableError) propagates so the run
        aborts before any evaluation instead of breeding around missing data.
        An unknown timeframe does too (never a silent skip). A plain ValueError
        from the cross-window / WFA range helpers is a CONFIG problem, not data
        coverage: those windows are skipped here and the gates fail closed on
        the same error after the GA exactly as before (chief, 2026-10-09); a
        DataUnavailableError from them re-raises (it is missing data, never
        config).

        Only the real NT backtest path is checked: pipelines built with an
        injected fake backtest_fn (tests, --mock) never read the default
        catalog, so validating it would fail their run() on machines without
        the data.
        """
        from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH, INTERVAL_TO_AGGREGATION
        from vibe_quant.discovery.backtest_fn import NTBacktestFn
        from vibe_quant.screening import nt_runner

        fn = self._backtest_fn
        if not isinstance(fn, NTBacktestFn):
            return
        cfg = self.config

        def _fn_windows(f: NTBacktestFn) -> list[tuple[str, str]]:
            # Eval sub-windows only with 2+ entries: backtest_fn runs
            # multi-window only then (len>=2); 0/1 entries = full-span run.
            return (
                list(f.windows)
                if f.windows and len(f.windows) >= 2
                else [(f.start_date, f.end_date)]
            )

        windows = _fn_windows(fn)
        holdout_fn = self._holdout_backtest_fn
        if isinstance(holdout_fn, NTBacktestFn):
            windows += _fn_windows(holdout_fn)
        elif cfg.has_holdout:
            windows.append((cfg.holdout_start_date, cfg.holdout_end_date))
        if self._backtest_fn_factory is not None:
            try:
                windows += [(ws, we) for _, ws, we in self.cross_window_ranges()]
            except DataUnavailableError:
                raise  # subclasses ValueError: missing data must never be swallowed
            except ValueError as exc:
                logger.debug("Cross-window ranges not checkable in bar preflight: %s", exc)
            if cfg.wfa_oos_step_days > 0:
                try:
                    windows += self.wfa_window_ranges(cfg.start_date, cfg.end_date)
                except DataUnavailableError:
                    raise  # subclasses ValueError: missing data must never be swallowed
                except ValueError as exc:
                    logger.debug("WFA ranges not checkable in bar preflight: %s", exc)
        windows = list(dict.fromkeys(windows))

        tf = fn.timeframe
        if tf not in INTERVAL_TO_AGGREGATION:
            raise ValueError(f"unknown timeframe {tf!r}")
        # DiscoveryConfig has a single timeframe (no additional-timeframes
        # field) and discovery chromosomes compile to single-timeframe DSLs.
        step, agg = INTERVAL_TO_AGGREGATION[tf]
        catalog = str(Path(DEFAULT_CATALOG_PATH).resolve())
        for symbol in fn.symbols:
            bar_type = f"{symbol}-PERP.BINANCE-{step}-{agg.name}-LAST-EXTERNAL"
            for start, end in windows:
                nt_runner.require_bars_in_window(catalog, bar_type, start, end)

    # -- evaluation ---------------------------------------------------------

    @staticmethod
    def _dsl_key(chrom: StrategyChromosome) -> str:
        """Content key of the strategy a chromosome compiles to (name excluded)."""
        try:
            dsl = chromosome_to_dsl(chrom)
        except Exception:
            return f"uid:{chrom.uid}"  # unconvertible: never shares a cache slot
        return dsl_body_key(dsl)

    def _evaluate_new(self, chroms: list[StrategyChromosome]) -> list[FitnessResult]:
        """Evaluate chromosomes; each distinct strategy is backtested once per run.

        Backtests are deterministic for a given DSL + date windows, so
        identical genomes (clones, no-op mutations, elites) reuse the cached
        fitness. ``_total_evaluated`` counts DISTINCT strategies backtested --
        the multiple-testing N for DSR.
        """
        cfg = self.config
        keys = [self._dsl_key(c) for c in chroms]
        todo: dict[str, int] = {}
        for i, key in enumerate(keys):
            if key not in self._fitness_cache and key not in todo:
                todo[key] = i
        if todo:
            reps = [chroms[i] for i in todo.values()]
            fresh = evaluate_population(
                reps,
                self._backtest_fn,
                self._filter_fn,
                max_workers=cfg.max_workers,
                executor=self._executor,
                min_trades=cfg.min_trades,
                timeframe=cfg.timeframe,
            )
            n_failed = sum(1 for fr in fresh if fr.error)
            ok_before = self._ok_evals
            self._ok_evals += len(fresh) - n_failed
            # Systemic failure (pickling, catalog, NT crash) must not end as
            # "completed, 0 champions". Before any success, any all-failed
            # batch aborts. After successes, only a large all-failed batch does:
            # a small late batch (e.g. crowding replacements) can fail on
            # genome-specific errors without dooming a healthy run.
            min_abort = max(2, cfg.population_size // 4)
            if (
                fresh
                and n_failed == len(fresh)
                and (ok_before == 0 or len(fresh) >= min_abort)
            ):
                msg = (
                    f"all {len(fresh)} evaluations failed (ok_evals={ok_before}); "
                    f"first: {fresh[0].error}"
                )
                raise DiscoveryEvaluationError(msg)
            for (key, idx), fr in zip(todo.items(), fresh, strict=True):
                self._fitness_cache[key] = fr
                self._total_evaluated += 1
                if fr.adjusted_score > 0:
                    self._all_scored.append((chroms[idx].clone(), fr))
        return [self._fitness_cache[key] for key in keys]

    def run(self) -> DiscoveryResult:
        """Execute the full evolutionary discovery loop.

        Returns:
            DiscoveryResult containing generation history and top strategies.

        Raises:
            DiscoveryEvaluationError: a whole evaluation batch raised.
        """
        try:
            return self._run()
        finally:
            # Also on abort: never leak the worker pool.
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None

    def _run(self) -> DiscoveryResult:
        cfg = self.config
        self._apply_indicator_pool_filter()
        self._preflight_aux_data()
        self._preflight_bars()

        # Parse direction constraint
        from vibe_quant.discovery.operators import Direction
        direction_constraint: Direction | None = None
        if cfg.direction:
            direction_constraint = Direction(cfg.direction)
        self._direction_constraint = direction_constraint

        population = initialize_population(
            cfg.population_size,
            direction_constraint=direction_constraint,
            seed_chromosomes=self._seed_chromosomes,
        )
        known: list[FitnessResult | None] = [None] * len(population)
        generation_results: list[GenerationResult] = []
        last_fitness_results: list[FitnessResult] = []

        self._fitness_cache = {}
        self._total_evaluated = 0
        self._ok_evals = 0
        self._all_scored = []
        self._guardrail_rejections = []

        converged = False
        convergence_gen: int | None = None
        pipeline_start = time.monotonic()

        logger.info(
            "=== DISCOVERY START: pop=%d max_gen=%d symbols=%s tf=%s ===",
            cfg.population_size,
            cfg.max_generations,
            cfg.symbols,
            cfg.timeframe,
        )
        logger.info(
            "Config: mutation=%.2f crossover=%.2f elite=%d tournament=%d "
            "convergence_gens=%d direction=%s max_workers=%s indicator_pool=%s",
            cfg.mutation_rate,
            cfg.crossover_rate,
            cfg.elite_count,
            cfg.tournament_size,
            cfg.convergence_generations,
            cfg.direction or "random",
            cfg.max_workers,
            cfg.indicator_pool or "all",
        )

        # Create a long-lived worker pool to avoid per-generation pool startup
        # overhead (fixes idle workers when pool creation is slower than work)
        self._executor = self._create_executor(cfg.max_workers, cfg.population_size)

        phase_start = time.monotonic()
        evaluated_at_phase_start = 0
        for gen in range(cfg.max_generations):
            # Evaluate individuals without known fitness (initial population,
            # tournament offspring, immigrants). Crowding offspring were
            # already evaluated during selection.
            todo = [i for i, fr in enumerate(known) if fr is None]
            if todo:
                fresh = self._evaluate_new([population[i] for i in todo])
                for i, fr in zip(todo, fresh, strict=True):
                    known[i] = fr
            fitness_results = [fr for fr in known if fr is not None]
            if len(fitness_results) != len(population):  # pragma: no cover - invariant
                msg = "fitness/population misalignment after evaluation"
                raise RuntimeError(msg)
            last_fitness_results = fitness_results
            total_evaluated = self._total_evaluated

            gen_elapsed = time.monotonic() - phase_start
            evals_this_gen = total_evaluated - evaluated_at_phase_start
            total_elapsed = time.monotonic() - pipeline_start

            # Build generation metrics
            scores = [fr.adjusted_score for fr in fitness_results]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            gen_result = GenerationResult(
                generation=gen,
                best_fitness=scores[best_idx],
                mean_fitness=sum(scores) / len(scores),
                worst_fitness=min(scores),
                best_chromosome=population[best_idx].clone(),
                population_size=len(population),
                num_passed_filters=sum(1 for fr in fitness_results if fr.passed_filters),
            )
            generation_results.append(gen_result)

            self._log_generation(
                gen=gen,
                population=population,
                fitness_results=fitness_results,
                generation_results=generation_results,
                best_idx=best_idx,
                gen_elapsed=gen_elapsed,
                total_elapsed=total_elapsed,
                evals_this_gen=evals_this_gen,
            )

            # ETA calculation
            avg_gen_time = total_elapsed / (gen + 1)
            remaining_gens = cfg.max_generations - gen - 1
            eta_seconds = avg_gen_time * remaining_gens
            best_fr = fitness_results[best_idx]
            self._write_progress(
                generation=gen + 1,
                max_generations=cfg.max_generations,
                best_fitness=gen_result.best_fitness,
                mean_fitness=gen_result.mean_fitness,
                worst_fitness=gen_result.worst_fitness,
                best_trades=best_fr.total_trades,
                best_return=best_fr.total_return,
                gen_time=gen_elapsed,
                total_elapsed=total_elapsed,
                eta_seconds=eta_seconds,
                total_evaluated=total_evaluated,
            )

            # Convergence check with progress tracking
            stagnant_gens = self._stagnant_generations(generation_results)
            if stagnant_gens > 0:
                logger.info(
                    "  Convergence: %d/%d stagnant gens (best unchanged)",
                    stagnant_gens,
                    cfg.convergence_generations,
                )
            if self._check_convergence(generation_results):
                converged = True
                convergence_gen = gen
                logger.info(
                    "=== CONVERGED at gen %d/%d after %.0fs ===",
                    gen + 1,
                    cfg.max_generations,
                    total_elapsed,
                )
                break

            # No evolution after the last generation (crowding would evaluate
            # offspring nobody scores).
            if gen == cfg.max_generations - 1:
                break

            phase_start = time.monotonic()
            evaluated_at_phase_start = self._total_evaluated

            population, known = self._evolve_generation(population, fitness_results)
            population, known = self._maybe_inject_immigrants(population, known)

        # Shut down worker pool after all generations complete
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        total_evaluated = self._total_evaluated
        all_scored = self._all_scored

        # Select top-K with structural diversity enforcement
        all_scored.sort(key=lambda t: t[1].adjusted_score, reverse=True)
        # Worst mode scores softly, so a high scorer may still fail the hard
        # worst-symbol gate: rank gate-passers first (stable) so rejected
        # candidates don't eat top-K slots that real champions could fill.
        all_scored.sort(key=lambda t: bool(worst_symbol_gate_reasons(t[1], cfg.min_trades)))
        top_strategies_raw = _select_diverse_top_k(
            all_scored,
            top_k=cfg.top_k,
            min_distance=cfg.min_diversity_distance,
        )
        # Cast back to the expected type (FitnessResult, not float)
        top_strategies: list[tuple[StrategyChromosome, FitnessResult]] = [
            (chrom, fr) for chrom, fr in top_strategies_raw  # type: ignore[misc]
        ]
        if last_fitness_results:
            exported = self._export_top_strategies(population, last_fitness_results)
            logger.debug("Exported %d top strategy DSL dicts", len(exported))

        # Log empirical Sharpe distribution across GA evaluations (diagnostic only).
        # NOTE: NOT passed to DSR — cross-strategy Sharpe dispersion is a category
        # error for the paper's V[{SR_n}] which measures within-strategy estimation
        # noise. See vibe-quant-fici.
        if len(all_scored) >= 2:
            sharpes = [fr.sharpe_ratio for _, fr in all_scored]
            mean_sr = sum(sharpes) / len(sharpes)
            var_sr = sum((s - mean_sr) ** 2 for s in sharpes) / (len(sharpes) - 1)
            logger.info(
                "GA Sharpe distribution: n=%d mean=%.3f std=%.3f min=%.3f max=%.3f",
                len(sharpes), mean_sr, var_sr ** 0.5, min(sharpes), max(sharpes),
            )

        # Gate 1: guardrails (DSR + min trades + complexity + bootstrap CI).
        # Fails closed: candidates failing ANY guardrail are dropped.
        top_strategies = self._validate_top_strategies(top_strategies, total_evaluated)

        # Final summary
        total_time = time.monotonic() - pipeline_start
        avg_gen_time = total_time / len(generation_results) if generation_results else 0

        logger.info(
            "=== DISCOVERY COMPLETE: %.0fs total (%.1fs/gen avg) | %d gens | %d evaluated | converged=%s ===",
            total_time,
            avg_gen_time,
            len(generation_results),
            total_evaluated,
            converged,
        )

        # Evolution timeline: how best score progressed across generations
        if generation_results:
            timeline = " → ".join(f"{gr.best_fitness:.3f}" for gr in generation_results)
            logger.info("  Evolution: %s", timeline)

            # Find which generation found the overall best
            best_gen_idx = max(range(len(generation_results)), key=lambda i: generation_results[i].best_fitness)
            logger.info(
                "  Peak gen: %d/%d (score=%.4f, %s of total time)",
                best_gen_idx + 1,
                len(generation_results),
                generation_results[best_gen_idx].best_fitness,
                f"{(best_gen_idx + 1) / len(generation_results) * 100:.0f}%",
            )

        self._log_top_strategies(top_strategies)

        return self._apply_validation_gates(
            top_strategies,
            generations=generation_results,
            total_evaluated=total_evaluated,
            converged=converged,
            convergence_gen=convergence_gen,
        )

    # -- validation gates ----------------------------------------------------

    def _reject(
        self,
        chrom: StrategyChromosome,
        fitness: FitnessResult,
        stage: str,
        reasons: list[str],
    ) -> None:
        """Record a rejected candidate (persisted with the run, shown in the UI)."""
        self._guardrail_rejections.append(
            {
                "uid": chrom.uid,
                "stage": stage,
                "score": round(fitness.adjusted_score, 4),
                "sharpe": round(fitness.sharpe_ratio, 3),
                "trades": fitness.total_trades,
                "reasons": list(reasons),
            }
        )
        logger.info("Gate FAIL [%s]: %s reasons=%s", stage, chrom.uid, reasons)

    def holdout_min_trades(self) -> int:
        """Holdout trade floor (explicit, or half the train-gate trade rate)."""
        cfg = self.config
        if cfg.holdout_min_trades is not None:
            return cfg.holdout_min_trades
        train_days = compute_day_count(cfg.start_date, cfg.end_date)
        holdout_days = compute_day_count(cfg.holdout_start_date, cfg.holdout_end_date)
        if not train_days or not holdout_days:
            return 1
        return max(1, int(cfg.min_trades * holdout_days / (2 * train_days)))

    def _holdout_reasons(self, hr: HoldoutResult, min_trades: int) -> list[str]:
        """Failure reasons of the holdout gate (empty = pass)."""
        cfg = self.config
        reasons: list[str] = []
        if hr.total_trades < min_trades:
            reasons.append(f"Holdout: {hr.total_trades} trades < {min_trades}")
        if not hr.total_return > 0:
            reasons.append(f"Holdout: return {hr.total_return:.2%} <= 0")
        if not hr.sharpe_ratio > cfg.holdout_min_sharpe:
            reasons.append(
                f"Holdout: sharpe {hr.sharpe_ratio:.2f} <= {cfg.holdout_min_sharpe:.2f}"
            )
        return reasons

    def _apply_validation_gates(
        self,
        top_strategies: list[tuple[StrategyChromosome, FitnessResult]],
        *,
        generations: list[GenerationResult],
        total_evaluated: int,
        converged: bool,
        convergence_gen: int | None,
    ) -> DiscoveryResult:
        """Run cross-window -> rolling WFA -> holdout on guardrail survivors.

        Each gate only sees the previous gate's survivors and rejects (never
        keeps) failures. The holdout runs last and exactly once per survivor.
        """
        cfg = self.config
        survivors = list(top_strategies)
        cw_by_uid: dict[str, CrossWindowResult] = {}
        wfa_by_uid: dict[str, WFARollingResult] = {}
        holdout_by_uid: dict[str, HoldoutResult] = {}

        train_dates: tuple[str, str] | None = None
        holdout_dates: tuple[str, str] | None = None
        if cfg.train_test_split > 0:
            train_dates = (cfg.start_date, cfg.end_date)
            if cfg.has_holdout:
                holdout_dates = (cfg.holdout_start_date, cfg.holdout_end_date)

        # Gate 2: cross-window (shifted sub-windows of the train range)
        if cfg.cross_window_months:
            if not survivors:
                logger.warning("Cross-window requested but no strategies survived guardrails")
            elif self._backtest_fn_factory is None:
                logger.warning(
                    "Cross-window requested (months=%s) but cannot run: no backtest_fn_factory "
                    "configured — failing closed",
                    cfg.cross_window_months,
                )
                for chrom, fit in survivors:
                    self._reject(chrom, fit, "cross_window", ["Cross-window: no backtest factory"])
                survivors = []
            else:
                results, error = self._evaluate_cross_windows(survivors)
                kept: list[tuple[StrategyChromosome, FitnessResult]] = []
                if error is not None:
                    for chrom, fit in survivors:
                        self._reject(chrom, fit, "cross_window", [error])
                for (chrom, fit), cwr in zip(survivors, results, strict=False):
                    if cwr.passed:
                        cw_by_uid[chrom.uid] = cwr
                        kept.append((chrom, fit))
                    else:
                        self._reject(
                            chrom, fit, "cross_window",
                            [
                                f"Cross-window: {cwr.windows_passed}/{cwr.total_windows} "
                                f"shifted windows passed (need {cwr.required}; "
                                f"sharpe>={cfg.cross_window_min_sharpe} and return>0)"
                            ],
                        )
                survivors = kept

        # Gate 3: rolling-window stability (WFA) over the train range
        if cfg.wfa_oos_step_days > 0:
            skip_reason: str | None = None
            if not survivors:
                skip_reason = "no top strategies survived guardrails"
            elif self._backtest_fn_factory is None:
                skip_reason = "no backtest_fn_factory configured"

            if skip_reason is not None:
                logger.warning(
                    "WFA requested (wfa_oos_step_days=%d) but not run: %s",
                    cfg.wfa_oos_step_days, skip_reason,
                )
                for chrom, fit in survivors:  # fail closed
                    self._reject(chrom, fit, "wfa", [f"WFA: {skip_reason}"])
                survivors = []
            else:
                wfa_results, error = self._evaluate_wfa_rolling(
                    survivors, cfg.start_date, cfg.end_date,
                )
                kept = []
                if error is not None:
                    for chrom, fit in survivors:
                        self._reject(chrom, fit, "wfa", [error])
                else:
                    for (chrom, fit), wfa in zip(survivors, wfa_results, strict=True):
                        if wfa.passed:
                            wfa_by_uid[chrom.uid] = wfa
                            kept.append((chrom, fit))
                        else:
                            self._reject(
                                chrom, fit, "wfa",
                                [
                                    f"WFA: {wfa.windows_profitable}/{wfa.total_windows} rolling "
                                    f"windows profitable ({wfa.consistency:.0%} < "
                                    f"{cfg.wfa_min_consistency:.0%})"
                                ],
                            )
                survivors = kept

        # Gate 4: holdout -- the single final out-of-sample pass/fail
        holdout_min_trades: int | None = self.holdout_min_trades() if cfg.has_holdout else None
        if cfg.train_test_split > 0:
            if not cfg.has_holdout:
                logger.warning(
                    "train_test_split=%.2f but no holdout dates — failing closed",
                    cfg.train_test_split,
                )
                for chrom, fit in survivors:
                    self._reject(chrom, fit, "holdout", ["Holdout: no holdout dates"])
                survivors = []
            elif self._holdout_backtest_fn is None:
                logger.warning("Holdout configured but no holdout_backtest_fn — failing closed")
                for chrom, fit in survivors:
                    self._reject(chrom, fit, "holdout", ["Holdout: no holdout backtest"])
                survivors = []
            elif survivors and holdout_min_trades is not None:
                holdout_results = self._evaluate_holdout(survivors)
                kept = []
                for (chrom, fit), hr in zip(survivors, holdout_results, strict=True):
                    reasons = self._holdout_reasons(hr, holdout_min_trades)
                    if reasons:
                        self._reject(chrom, fit, "holdout", reasons)
                    else:
                        holdout_by_uid[chrom.uid] = hr
                        kept.append((chrom, fit))
                logger.info(
                    "  Holdout gate: %d/%d passed (min_trades=%d, sharpe>%.2f, return>0)",
                    len(kept), len(survivors), holdout_min_trades, cfg.holdout_min_sharpe,
                )
                survivors = kept

        if not survivors:
            logger.warning(
                "No champion passed every gate — run persists 0 champions (%d rejections)",
                len(self._guardrail_rejections),
            )

        return DiscoveryResult(
            generations=generations,
            top_strategies=survivors,
            total_candidates_evaluated=total_evaluated,
            converged=converged,
            convergence_generation=convergence_gen,
            holdout_results=[holdout_by_uid[c.uid] for c, _ in survivors if c.uid in holdout_by_uid],
            train_dates=train_dates,
            holdout_dates=holdout_dates,
            cross_window_results=[cw_by_uid[c.uid] for c, _ in survivors if c.uid in cw_by_uid],
            wfa_results=[wfa_by_uid[c.uid] for c, _ in survivors if c.uid in wfa_by_uid],
            guardrail_rejections=list(self._guardrail_rejections),
            holdout_min_trades=holdout_min_trades,
        )

    def _evaluate_holdout(
        self,
        top_strategies: list[tuple[StrategyChromosome, FitnessResult]],
    ) -> list[HoldoutResult]:
        """Evaluate strategies on the holdout (out-of-sample) period.

        Returns HoldoutResult for each strategy, parallel to top_strategies.
        A failed backtest yields failing metrics (fails the gate).
        """
        assert self._holdout_backtest_fn is not None
        holdout_fn = self._holdout_backtest_fn
        results: list[HoldoutResult] = []

        logger.info("=== HOLDOUT EVALUATION: %d strategies ===", len(top_strategies))

        for rank, (chrom, train_fit) in enumerate(top_strategies, 1):
            try:
                hr = _metrics_to_holdout_result(holdout_fn(chrom))
            except DataUnavailableError:
                raise
            except Exception:
                logger.warning("Holdout eval failed for %s", chrom.uid, exc_info=True)
                hr = _FAILED_WINDOW
            results.append(hr)

            # Log train vs holdout comparison
            logger.info(
                "  #%d %s: TRAIN sharpe=%.2f dd=%.1f%% ret=%.1f%% trades=%d → "
                "HOLDOUT sharpe=%.2f dd=%.1f%% ret=%.1f%% trades=%d",
                rank, chrom.uid,
                train_fit.sharpe_ratio, train_fit.max_drawdown * 100,
                train_fit.total_return * 100, train_fit.total_trades,
                hr.sharpe_ratio, hr.max_drawdown * 100,
                hr.total_return * 100, hr.total_trades,
            )

        # Summary: how much did strategies degrade on holdout?
        if results:
            train_sharpes = [f.sharpe_ratio for _, f in top_strategies]
            holdout_sharpes = [h.sharpe_ratio for h in results]
            avg_train = sum(train_sharpes) / len(train_sharpes)
            avg_holdout = sum(holdout_sharpes) / len(holdout_sharpes)
            degradation = (avg_train - avg_holdout) / avg_train * 100 if avg_train > 0 else 0
            logger.info(
                "  Holdout summary: avg_train_sharpe=%.2f → avg_holdout_sharpe=%.2f "
                "(%.1f%% degradation)",
                avg_train, avg_holdout, degradation,
            )

        return results

    def cross_window_ranges(self) -> list[tuple[int, str, str]]:
        """Shifted cross-windows ``(offset_months, start, end)`` inside the TRAIN range.

        Window k starts ``offset_k`` months after the train start and ends at
        the train end -- shifting the END forward would run into the holdout
        (or past the data), so windows are clipped to the train range.

        Raises:
            ValueError: A window is shorter than ``_MIN_CROSS_WINDOW_DAYS`` or
                (defensively) overlaps the holdout.
        """
        cfg = self.config
        from dateutil.relativedelta import relativedelta

        base_start = _parse_date(cfg.start_date)
        base_end = _parse_date(cfg.end_date)
        windows: list[tuple[int, str, str]] = []
        for months in cfg.cross_window_months:
            ws_dt = base_start + relativedelta(months=months)
            if base_end - ws_dt < timedelta(days=_MIN_CROSS_WINDOW_DAYS):
                msg = (
                    f"Cross-window: +{months}mo window {ws_dt:%Y-%m-%d}..{cfg.end_date} "
                    f"shorter than {_MIN_CROSS_WINDOW_DAYS}d inside the train range "
                    f"{cfg.start_date}..{cfg.end_date}"
                )
                raise ValueError(msg)
            windows.append((months, ws_dt.strftime("%Y-%m-%d"), cfg.end_date))
        if cfg.has_holdout:
            assert_windows_outside_holdout(
                [(ws, we) for _, ws, we in windows], cfg.holdout_start_date, "Cross-window",
            )
        return windows

    def _evaluate_cross_windows(
        self,
        top_strategies: list[tuple[StrategyChromosome, FitnessResult]],
    ) -> tuple[list[CrossWindowResult], str | None]:
        """Evaluate strategies on shifted sub-windows of the train range.

        Only the SHIFTED windows count -- the full train window is where the
        GA selected the strategy, so it is never evidence of robustness.

        Returns:
            (cross_window_results parallel to top_strategies, error). ``error``
            is set (and results empty) when the windows can't be built.
        """
        assert self._backtest_fn_factory is not None
        cfg = self.config
        min_sharpe = cfg.cross_window_min_sharpe

        try:
            windows = self.cross_window_ranges()
        except ValueError as exc:
            logger.warning("%s — failing closed", exc)
            return [], str(exc)

        required = len(windows)
        if cfg.cross_window_min_pass is not None:
            required = min(cfg.cross_window_min_pass, len(windows))

        logger.info(
            "=== CROSS-WINDOW VALIDATION: %d strategies × %d shifted windows (need %d) ===",
            len(top_strategies), len(windows), required,
        )
        for months, ws, we in windows:
            logger.info("  Window +%dmo: %s → %s", months, ws, we)

        cross_results: list[CrossWindowResult] = []
        for rank, (chrom, _train_fit) in enumerate(top_strategies, 1):
            window_hrs: list[HoldoutResult] = []
            passes = 0
            for months, ws, we in windows:
                try:
                    hr = _metrics_to_holdout_result(self._backtest_fn_factory(ws, we)(chrom))
                except DataUnavailableError:
                    raise
                except Exception:
                    logger.warning(
                        "Cross-window eval failed: %s window +%dmo", chrom.uid, months,
                        exc_info=True,
                    )
                    hr = _FAILED_WINDOW
                window_hrs.append(hr)
                if hr.total_return > 0 and hr.sharpe_ratio >= min_sharpe:
                    passes += 1

            passed = passes >= required
            cross_results.append(
                CrossWindowResult(
                    window_results=window_hrs,
                    windows_passed=passes,
                    total_windows=len(windows),
                    passed=passed,
                    window_dates=[(ws, we) for _, ws, we in windows],
                    offsets_months=[m for m, _, _ in windows],
                    required=required,
                )
            )

            window_strs = [
                f"W+{months}mo: sharpe={hr.sharpe_ratio:.2f} ret={hr.total_return*100:.1f}% "
                f"[{'PASS' if (hr.total_return > 0 and hr.sharpe_ratio >= min_sharpe) else 'FAIL'}]"
                for (months, _, _), hr in zip(windows, window_hrs, strict=True)
            ]
            logger.info(
                "  #%d %s: %s → %d/%d shifted windows %s",
                rank, chrom.uid,
                " | ".join(window_strs),
                passes, len(windows),
                "PROMOTED" if passed else "REJECTED",
            )

        logger.info(
            "  Cross-window summary: %d/%d strategies promoted",
            sum(1 for r in cross_results if r.passed), len(top_strategies),
        )
        return cross_results, None

    def wfa_window_ranges(self, range_start: str, range_end: str) -> list[tuple[str, str]]:
        """Rolling ``wfa_oos_step_days`` windows tiling [range_start, range_end]."""
        step = self.config.wfa_oos_step_days
        start = _parse_date(range_start)
        end = _parse_date(range_end)
        windows: list[tuple[str, str]] = []
        current = start
        while current + timedelta(days=step) <= end:
            nxt = current + timedelta(days=step)
            windows.append((current.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
            current = nxt
        if self.config.has_holdout:
            assert_windows_outside_holdout(windows, self.config.holdout_start_date, "WFA")
        return windows

    def _evaluate_wfa_rolling(
        self,
        top_strategies: list[tuple[StrategyChromosome, FitnessResult]],
        range_start: str,
        range_end: str,
    ) -> tuple[list[WFARollingResult], str | None]:
        """Rolling-window stability check over the TRAIN range.

        Tiles [range_start, range_end] with ``wfa_oos_step_days`` windows and
        requires ``wfa_min_consistency`` of them to be profitable. The windows
        are in-sample for the GA, so this measures stability over time, not
        out-of-sample skill (the holdout is the out-of-sample gate).

        Returns:
            (results parallel to top_strategies, error). ``error`` is set when
            the range is too short for a single window.
        """
        assert self._backtest_fn_factory is not None
        cfg = self.config
        step = cfg.wfa_oos_step_days
        min_consistency = cfg.wfa_min_consistency

        windows = self.wfa_window_ranges(range_start, range_end)
        if not windows:
            error = f"WFA: range {range_start}..{range_end} too short for {step}d windows"
            logger.warning("%s — failing closed", error)
            return [], error

        logger.info(
            "=== WFA ROLLING VALIDATION: %d strategies × %d windows (%dd each, train range) ===",
            len(top_strategies), len(windows), step,
        )
        for i, (ws, we) in enumerate(windows):
            logger.info("  Window %d: %s → %s", i, ws, we)

        wfa_results: list[WFARollingResult] = []
        for rank, (chrom, _train_fit) in enumerate(top_strategies, 1):
            oos_results: list[HoldoutResult] = []
            profitable = 0
            sharpe_positive = 0

            for ws, we in windows:
                try:
                    hr = _metrics_to_holdout_result(self._backtest_fn_factory(ws, we)(chrom))
                except DataUnavailableError:
                    raise
                except Exception:
                    logger.warning("WFA eval failed: %s", chrom.uid, exc_info=True)
                    hr = _FAILED_WINDOW
                oos_results.append(hr)
                if hr.total_return > 0:
                    profitable += 1
                if hr.sharpe_ratio > 0:
                    sharpe_positive += 1

            consistency = profitable / len(windows)
            sharpe_consistency = sharpe_positive / len(windows)
            passed = consistency >= min_consistency

            wfa_results.append(
                WFARollingResult(
                    oos_windows=oos_results,
                    window_dates=list(windows),
                    windows_profitable=profitable,
                    windows_sharpe_positive=sharpe_positive,
                    total_windows=len(windows),
                    consistency=consistency,
                    sharpe_consistency=sharpe_consistency,
                    passed=passed,
                )
            )

            w_strs = [
                f"W{i}: ret={hr.total_return*100:.1f}% {'OK' if hr.total_return > 0 else 'LOSS'}"
                for i, hr in enumerate(oos_results)
            ]
            logger.info(
                "  #%d %s: %s → %d/%d profitable (%.0f%%) | %d/%d sharpe+ (%.0f%%) %s",
                rank, chrom.uid, " | ".join(w_strs),
                profitable, len(windows), consistency * 100,
                sharpe_positive, len(windows), sharpe_consistency * 100,
                "PASS" if passed else "FAIL",
            )

        logger.info(
            "  WFA summary: %d/%d passed (min_consistency=%.0f%%)",
            sum(1 for r in wfa_results if r.passed), len(top_strategies), min_consistency * 100,
        )
        return wfa_results, None

    def _write_progress(self, **kwargs: object) -> None:
        """Write progress JSON file for API polling."""
        if not self._progress_file:
            return
        try:
            self._progress_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._progress_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(kwargs, default=str))
            tmp.rename(self._progress_file)
        except Exception:
            logger.debug("Failed to write progress file", exc_info=True)

    # -- logging -------------------------------------------------------------

    def _log_generation(
        self,
        *,
        gen: int,
        population: list[StrategyChromosome],
        fitness_results: list[FitnessResult],
        generation_results: list[GenerationResult],
        best_idx: int,
        gen_elapsed: float,
        total_elapsed: float,
        evals_this_gen: int,
    ) -> None:
        """Per-generation population analytics (log only)."""
        cfg = self.config
        gen_result = generation_results[-1]
        scores = [fr.adjusted_score for fr in fitness_results]
        avg_gen_time = total_elapsed / (gen + 1)
        eta_seconds = avg_gen_time * (cfg.max_generations - gen - 1)
        best_fr = fitness_results[best_idx]

        score_std = (sum((s - gen_result.mean_fitness) ** 2 for s in scores) / len(scores)) ** 0.5
        zero_count = sum(1 for s in scores if s <= 0)
        median_score = statistics.median(scores) if scores else 0.0

        active = [fr for fr in fitness_results if fr.total_trades > 0]
        all_sharpes = [fr.sharpe_ratio for fr in active]
        all_trades = [fr.total_trades for fr in active]
        all_returns = [fr.total_return for fr in active]
        all_dds = [fr.max_drawdown for fr in active]

        ind_counter: Counter[str] = Counter()
        for c in population:
            for g in c.entry_genes + c.exit_genes:
                ind_counter[g.indicator_type] += 1
        total_genes = sum(ind_counter.values()) or 1
        ind_pcts = {k: f"{v/total_genes*100:.0f}%" for k, v in ind_counter.most_common()}

        dir_counts: dict[str, int] = {}
        for c in population:
            d = c.direction.value if hasattr(c.direction, "value") else str(c.direction)
            dir_counts[d] = dir_counts.get(d, 0) + 1

        if gen > 0:
            improvement = gen_result.best_fitness - generation_results[-2].best_fitness
            improvement_str = f" Δbest={improvement:+.4f}" if improvement != 0 else " (no change)"
        else:
            improvement_str = " (initial)"

        logger.info(
            "=== GEN %d/%d === best=%.4f mean=%.4f median=%.4f std=%.4f | "
            "zero_score=%d/%d | gen_time=%.1fs total=%.0fs ETA=%.0fs%s",
            gen + 1, cfg.max_generations,
            gen_result.best_fitness, gen_result.mean_fitness, median_score, score_std,
            zero_count, len(scores),
            gen_elapsed, total_elapsed, eta_seconds, improvement_str,
        )

        best_chrom = population[best_idx]
        logger.info(
            "  Best: uid=%s dir=%s entry=%s exit=%s sl=%.1f%% tp=%.1f%% | "
            "sharpe=%.2f pf=%.2f dd=%.1f%% return=%.1f%% trades=%d",
            best_chrom.uid,
            best_chrom.direction.value if hasattr(best_chrom.direction, "value") else best_chrom.direction,
            [g.indicator_type for g in best_chrom.entry_genes],
            [g.indicator_type for g in best_chrom.exit_genes],
            best_chrom.stop_loss_pct, best_chrom.take_profit_pct,
            best_fr.sharpe_ratio, best_fr.profit_factor,
            best_fr.max_drawdown * 100, best_fr.total_return * 100, best_fr.total_trades,
        )
        if best_fr.symbol_scores is not None:
            logger.info(
                "  Score breakdown: soft=0.5*min+0.5*median of %s = %.4f",
                {sym: round(sc, 4) for sym, sc in best_fr.symbol_scores.items()},
                best_fr.adjusted_score,
            )
        else:
            logger.info(
                "  Score breakdown: raw=%.4f - complexity=%.4f - overtrade=%.4f = %.4f",
                best_fr.raw_score, best_fr.complexity_penalty, best_fr.overtrade_penalty,
                best_fr.adjusted_score,
            )
        if all_sharpes:
            logger.info(
                "  Population (n=%d active): sharpe=[%.2f, %.2f, %.2f] "
                "trades=[%d, %d, %d] dd=[%.1f%%, %.1f%%, %.1f%%] return=[%.1f%%, %.1f%%, %.1f%%]",
                len(all_sharpes),
                min(all_sharpes), statistics.median(all_sharpes), max(all_sharpes),
                min(all_trades), int(statistics.median(all_trades)), max(all_trades),
                min(all_dds) * 100, statistics.median(all_dds) * 100, max(all_dds) * 100,
                min(all_returns) * 100, statistics.median(all_returns) * 100, max(all_returns) * 100,
            )

        from vibe_quant.discovery.diversity import population_entropy
        logger.info(
            "  Diversity: entropy=%.3f indicators=%s directions=%s",
            population_entropy(population), ind_pcts, dir_counts,
        )
        logger.info(
            "  Timing: %d new backtests (%.1fs/backtest avg, %.1fs gen total, %d distinct so far)",
            evals_this_gen,
            gen_elapsed / evals_this_gen if evals_this_gen else 0.0,
            gen_elapsed,
            self._total_evaluated,
        )

    def _log_top_strategies(
        self, top_strategies: list[tuple[StrategyChromosome, FitnessResult]],
    ) -> None:
        """Top-K strategy details (log only)."""
        if not top_strategies:
            return
        logger.info("  --- Top %d strategies ---", len(top_strategies))
        for rank, (chrom, fit) in enumerate(top_strategies, 1):
            _dir = chrom.direction.value if hasattr(chrom.direction, "value") else chrom.direction
            logger.info(
                "  #%d uid=%s score=%.4f sharpe=%.2f pf=%.2f dd=%.1f%% return=%.1f%% trades=%d | "
                "dir=%s entry=%s exit=%s sl=%.1f%% tp=%.1f%%",
                rank, chrom.uid, fit.adjusted_score, fit.sharpe_ratio, fit.profit_factor,
                fit.max_drawdown * 100, fit.total_return * 100, fit.total_trades,
                _dir,
                [g.indicator_type for g in chrom.entry_genes],
                [g.indicator_type for g in chrom.exit_genes],
                chrom.stop_loss_pct, chrom.take_profit_pct,
            )
            for label, genes in (("entry", chrom.entry_genes), ("exit", chrom.exit_genes)):
                for i, gene in enumerate(genes):
                    logger.info(
                        "    %s[%d]: %s(%s) %s %.4f%s",
                        label, i, gene.indicator_type,
                        ", ".join(f"{k}={v}" for k, v in gene.parameters.items()),
                        gene.condition.value if hasattr(gene.condition, "value") else gene.condition,
                        gene.threshold,
                        f" sub={gene.sub_value}" if gene.sub_value else "",
                    )

    # -- internal -----------------------------------------------------------

    def _elite_slots(self) -> int:
        """Leading population slots holding elites after evolution."""
        if self.config.use_crowding:
            return min(1, self.config.elite_count)
        return self.config.elite_count

    def _maybe_inject_immigrants(
        self,
        population: list[StrategyChromosome],
        known: list[FitnessResult | None],
    ) -> tuple[list[StrategyChromosome], list[FitnessResult | None]]:
        """Replace random non-elite members with immigrants when entropy is low.

        The new population's members are the ones replaced (never scored with
        another generation's fitness); elites are protected. Immigrants get
        evaluated at the top of the next generation.
        """
        cfg = self.config
        from vibe_quant.discovery.diversity import (
            inject_random_immigrants,
            population_entropy,
            should_inject_immigrants,
        )

        entropy = population_entropy(population)
        if not should_inject_immigrants(entropy, threshold=cfg.entropy_threshold):
            return population, known
        new_pop = inject_random_immigrants(
            population,
            None,
            fraction=cfg.immigrant_fraction,
            direction_constraint=self._direction_constraint,
            protected=range(min(self._elite_slots(), len(population))),
        )
        replaced = [i for i, (a, b) in enumerate(zip(population, new_pop, strict=True)) if a is not b]
        new_known = list(known)
        for i in replaced:
            new_known[i] = None
        if replaced:
            logger.info(
                "  Diversity intervention: entropy=%.3f < %.1f, injected %d random immigrants",
                entropy, cfg.entropy_threshold, len(replaced),
            )
        return new_pop, new_known

    def _evolve_generation(
        self,
        population: list[StrategyChromosome],
        fitness_results: list[FitnessResult],
    ) -> tuple[list[StrategyChromosome], list[FitnessResult | None]]:
        """Produce next generation via crowding or classic tournament.

        Args:
            population: Current generation chromosomes.
            fitness_results: Parallel fitness results.

        Returns:
            (new population, parallel known fitness). ``None`` = not yet
            evaluated (evaluated at the top of the next generation).
        """
        if self.config.use_crowding:
            return self._evolve_crowding(population, fitness_results)
        return self._evolve_tournament(population, fitness_results)

    def _make_valid(self, child: StrategyChromosome) -> tuple[StrategyChromosome, int, bool]:
        """Return (valid child, retries used, random fallback used)."""
        valid_child = child
        for attempt in range(_MAX_OFFSPRING_RETRIES):
            if is_valid_chromosome(valid_child):
                return valid_child, attempt, False
            valid_child = mutate(child, self.config.mutation_rate)
        if is_valid_chromosome(valid_child):
            return valid_child, _MAX_OFFSPRING_RETRIES, False
        return _random_chromosome(direction_constraint=self._direction_constraint), _MAX_OFFSPRING_RETRIES, True

    def _breed(
        self, parent_a: StrategyChromosome, parent_b: StrategyChromosome,
    ) -> tuple[list[StrategyChromosome], int, int]:
        """Crossover + mutation + direction constraint + validity repair.

        Returns (two children, retries, random fallbacks). Children always
        carry fresh uids (crossover/mutate assign them).
        """
        cfg = self.config
        if random.random() < cfg.crossover_rate:
            child_a, child_b = crossover(parent_a, parent_b)
        else:
            child_a, child_b = parent_a, parent_b
        child_a = mutate(child_a, cfg.mutation_rate)
        child_b = mutate(child_b, cfg.mutation_rate)
        if self._direction_constraint is not None:
            child_a.direction = self._direction_constraint
            child_b.direction = self._direction_constraint
        children: list[StrategyChromosome] = []
        retries = 0
        fallbacks = 0
        for child in (child_a, child_b):
            valid, used, fell_back = self._make_valid(child)
            retries += used
            fallbacks += int(fell_back)
            children.append(valid)
        return children, retries, fallbacks

    def _evolve_tournament(
        self,
        population: list[StrategyChromosome],
        fitness_results: list[FitnessResult],
    ) -> tuple[list[StrategyChromosome], list[FitnessResult | None]]:
        """Classic evolution: elitism + tournament selection + crossover + mutation."""
        cfg = self.config
        scores = [fr.adjusted_score for fr in fitness_results]
        new_pop = apply_elitism(population, scores, cfg.elite_count)
        # Elites are unchanged clones: their fitness is known
        elite_fit: dict[str, FitnessResult] = {
            population[i].uid: fitness_results[i] for i in range(len(population))
        }
        new_known: list[FitnessResult | None] = [elite_fit.get(c.uid) for c in new_pop]

        retries = 0
        random_fallbacks = 0
        while len(new_pop) < cfg.population_size:
            parent_a = tournament_select(population, scores, cfg.tournament_size)
            parent_b = tournament_select(population, scores, cfg.tournament_size)
            children, r, f = self._breed(parent_a, parent_b)
            retries += r
            random_fallbacks += f
            for child in children:
                if len(new_pop) >= cfg.population_size:
                    break
                new_pop.append(child)
                new_known.append(None)

        if retries > 0 or random_fallbacks > 0:
            logger.info(
                "  Evolution: %d mutation retries, %d random fallbacks",
                retries,
                random_fallbacks,
            )

        return new_pop, new_known

    def _evolve_crowding(
        self,
        population: list[StrategyChromosome],
        fitness_results: list[FitnessResult],
    ) -> tuple[list[StrategyChromosome], list[FitnessResult | None]]:
        """Deterministic crowding evolution.

        1. Keep 1 elite as safety net
        2. Randomly pair remaining individuals
        3. Each pair produces 2 offspring (crossover + mutation)
        4. Offspring are EVALUATED, then each replaces its most-similar parent
           only if at least as fit (real offspring fitness -- feeding the
           parents' scores in as offspring fitness let every child win and
           removed all selection pressure).
        """
        cfg = self.config
        scores = [fr.adjusted_score for fr in fitness_results]

        n_elite = min(1, cfg.elite_count)
        elite_indices: list[int] = []
        if n_elite:
            elite_indices = [max(range(len(scores)), key=lambda i: scores[i])]

        new_pop: list[StrategyChromosome] = [population[i].clone() for i in elite_indices]
        new_known: list[FitnessResult | None] = [fitness_results[i] for i in elite_indices]

        pool_indices = [i for i in range(len(population)) if i not in elite_indices]
        random.shuffle(pool_indices)
        pairs = [
            (pool_indices[k], pool_indices[k + 1])
            for k in range(0, len(pool_indices) - 1, 2)
        ]

        retries = 0
        random_fallbacks = 0
        offspring: list[list[StrategyChromosome]] = []
        for i, j in pairs:
            children, r, f = self._breed(population[i], population[j])
            retries += r
            random_fallbacks += f
            offspring.append(children)

        flat = [c for kids in offspring for c in kids]
        child_fit = self._evaluate_new(flat) if flat else []

        for p, ((i, j), kids) in enumerate(zip(pairs, offspring, strict=True)):
            kids_fit = child_fit[2 * p: 2 * p + 2]
            parents = [population[i], population[j]]
            parents_fit = [fitness_results[i], fitness_results[j]]
            winners = crowding_replace(
                parents=parents,
                parent_fitness=[scores[i], scores[j]],
                offspring=kids,
                offspring_fitness=[kids_fit[0].adjusted_score, kids_fit[1].adjusted_score],
            )
            for winner in winners:
                fit: FitnessResult | None = None
                for cand, cand_fit in zip(kids + parents, kids_fit + parents_fit, strict=True):
                    if winner is cand:
                        fit = cand_fit
                        break
                new_pop.append(winner)
                new_known.append(fit)

        # Handle odd pool (last unpaired individual)
        if len(pool_indices) % 2 == 1:
            last = pool_indices[-1]
            new_pop.append(population[last].clone())
            new_known.append(fitness_results[last])

        # Trim to population size
        if len(new_pop) > cfg.population_size:
            new_pop = new_pop[: cfg.population_size]
            new_known = new_known[: cfg.population_size]
        # Pad if somehow short
        while len(new_pop) < cfg.population_size:
            new_pop.append(_random_chromosome(direction_constraint=self._direction_constraint))
            new_known.append(None)

        if retries > 0 or random_fallbacks > 0:
            logger.info(
                "  Evolution: %d mutation retries, %d random fallbacks",
                retries,
                random_fallbacks,
            )

        return new_pop, new_known

    def _stagnant_generations(self, generation_results: list[GenerationResult]) -> int:
        """Count consecutive generations without improvement from the end."""
        if len(generation_results) < 2:
            return 0
        current_best = generation_results[-1].best_fitness
        count = 0
        for gr in reversed(generation_results[:-1]):
            if gr.best_fitness >= current_best:
                count += 1
            else:
                break
        return count

    def _check_convergence(self, generation_results: list[GenerationResult]) -> bool:
        """Check if best fitness has stagnated for convergence_generations.

        Requires at least 2*n generations to avoid false convergence from
        unusually good random initialization.

        Args:
            generation_results: All generation results so far.

        Returns:
            True if no improvement for convergence_generations consecutive gens.
        """
        n = self.config.convergence_generations
        if len(generation_results) < 2 * n:
            return False

        recent = generation_results[-n:]
        best_before = max(gr.best_fitness for gr in generation_results[:-n])
        best_recent = max(gr.best_fitness for gr in recent)
        return best_recent <= best_before

    def _validate_top_strategies(
        self,
        top_strategies: list[tuple[StrategyChromosome, FitnessResult]],
        total_evaluated: int,
    ) -> list[tuple[StrategyChromosome, FitnessResult]]:
        """Apply guardrails (DSR, complexity, min trades, bootstrap CI) to top strategies.

        FAILS CLOSED: returns only the candidates that pass every enabled
        guardrail -- an empty list when none do (vibe-quant-e70tl.5). Every
        rejected candidate is recorded in ``_guardrail_rejections`` with its
        reasons so the UI can explain an empty run.

        DSR uses theoretical variance (1/(T-1)) rather than empirical
        cross-strategy variance, which inflates SR₀ beyond what any
        strategy can achieve (see vibe-quant-fici).

        Args:
            top_strategies: Candidates to check.
            total_evaluated: DSR trial count N (distinct strategies tried).
        """
        guardrail_cfg = GuardrailConfig(
            min_trades=self.config.min_trades,
            max_complexity=8,
            require_dsr=self.config.require_dsr,
            require_wfa=False,  # WFA requires separate out-of-sample data
            require_purged_kfold=False,
            require_bootstrap_ci=self.config.require_bootstrap_ci,
            bootstrap_min_sharpe=self.config.bootstrap_min_sharpe,
            bootstrap_ci_level=self.config.bootstrap_ci_level,
        )

        # DSR observation count: NT Sharpe is annualized from daily returns,
        # and apply_discovery_dsr de-annualizes to daily units — so T must be
        # the day count of the window, not the bar count (vibe-quant-zmzzh).
        day_count = compute_day_count(self.config.start_date, self.config.end_date)

        import numpy as np

        validated: list[tuple[StrategyChromosome, FitnessResult]] = []
        for chrom, fitness in top_strategies:
            gate_reasons = worst_symbol_gate_reasons(fitness, self.config.min_trades)
            if gate_reasons:
                self._reject(chrom, fitness, "worst_symbol_train", gate_reasons)
                continue
            num_genes = len(chrom.entry_genes) + len(chrom.exit_genes)
            num_obs = day_count if day_count else max(100, fitness.total_trades * 5)

            trade_ret = np.array(fitness.trade_returns) if fitness.trade_returns else None
            result: GuardrailResult = apply_guardrails(
                fitness=fitness,
                num_genes=num_genes,
                config=guardrail_cfg,
                num_trials=max(1, total_evaluated),
                num_observations=num_obs,
                skewness=fitness.skewness,
                kurtosis=fitness.kurtosis,
                trade_returns=trade_ret,
                # trials_sharpe_variance intentionally omitted — use theoretical
                # 1/(T-1). Cross-strategy Sharpe dispersion from GA is NOT what
                # the paper's V[{SR_n}] measures (see vibe-quant-fici).
            )
            if result.passed:
                validated.append((chrom, fitness))
                logger.info(
                    "Guardrail PASS: %s score=%.4f sharpe=%.2f trades=%d",
                    chrom.uid,
                    fitness.adjusted_score,
                    fitness.sharpe_ratio,
                    fitness.total_trades,
                )
            else:
                self._reject(chrom, fitness, "guardrails", list(result.reasons))

        if not validated and top_strategies:
            logger.warning(
                "All %d top strategies failed guardrails — no champion is statistically "
                "significant (failing closed)",
                len(top_strategies),
            )
        else:
            logger.info("%d/%d top strategies passed guardrails", len(validated), len(top_strategies))
        return validated

    def _export_top_strategies(
        self,
        population: list[StrategyChromosome],
        fitness_results: list[FitnessResult],
    ) -> list[dict[str, object]]:
        """Export top K strategies as DSL-compatible YAML dicts.

        Uses genome.chromosome_to_dsl for conversion (single source of truth).

        Args:
            population: Current population.
            fitness_results: Parallel fitness results.

        Returns:
            List of DSL YAML dicts for the top-K strategies.
        """
        scores = [fr.adjusted_score for fr in fitness_results]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked[: self.config.top_k]

        dsl_dicts: list[dict[str, object]] = []
        for idx in top_indices:
            chrom = population[idx]
            dsl = chromosome_to_dsl(chrom)
            # Override timeframe from pipeline config
            dsl["timeframe"] = self.config.timeframe
            dsl_dicts.append(dsl)
        return dsl_dicts
