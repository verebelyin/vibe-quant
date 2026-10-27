"""Screening pipeline for parallel parameter sweep backtesting.

This module implements the screening pipeline for fast parameter sweeps
using NautilusTrader in simplified execution mode. The pipeline:

1. Takes a StrategyDSL with sweep parameters
2. Generates parameter combinations (Cartesian product)
3. Runs backtests in parallel via multiprocessing
4. Computes performance metrics
5. Applies hard filters and ranks results
6. Stores results in SQLite sweep_results table
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from vibe_quant.balance_settings import (
    DEFAULT_STARTING_BALANCE_USDT,
    validate_starting_balance,
)
from vibe_quant.errors import DataUnavailableError
from vibe_quant.launch_settings import PARSE_ORDER, LaunchSettings, iter_fields
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.overfitting.dsr import DeflatedSharpeRatio, mtm_dsr_calculate
from vibe_quant.screening.grid import (
    build_parameter_grid,
    compute_pareto_front,
    filter_by_metrics,
    rank_by_sharpe,
)
from vibe_quant.screening.types import (
    BacktestMetrics,
    BacktestRunner,
    MetricFilters,
    ScreeningResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from vibe_quant.data.fill_ticks import FillTickSet
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.dsl.schema import StrategyDSL

logger = logging.getLogger(__name__)


def _run_mock_backtest(params: dict[str, float | int]) -> BacktestMetrics:
    """Mock backtest runner for testing.

    This generates deterministic fake metrics based on parameter values.
    Used for testing the pipeline structure without NautilusTrader.

    Args:
        params: Parameter combination

    Returns:
        Mock backtest metrics
    """
    import hashlib

    # Create deterministic seed from params
    param_str = json.dumps(params, sort_keys=True)
    seed = int(hashlib.md5(param_str.encode()).hexdigest()[:8], 16)

    # Generate pseudo-random but deterministic metrics (decimal fractions)
    sharpe = (seed % 300) / 100.0 - 0.5  # Range: -0.5 to 2.5
    sortino = sharpe * 1.2
    max_dd = (seed % 30) / 100.0 + 0.02  # Range: 0.02 to 0.32 (2-32%)
    total_return = (seed % 40) / 100.0 - 0.10  # Range: -0.10 to 0.30 (-10% to 30%)
    pf = (seed % 200) / 100.0 + 0.5  # Range: 0.5 to 2.5
    win_rate = (seed % 40) / 100.0 + 0.30  # Range: 0.30 to 0.70 (30-70%)
    n_trades = (seed % 200) + 20  # Range: 20 to 220

    return BacktestMetrics(
        parameters=params,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        total_return=total_return,
        profit_factor=pf,
        win_rate=win_rate,
        total_trades=n_trades,
        total_fees=n_trades * 0.001,
        total_funding=n_trades * 0.0005,
        execution_time_seconds=0.01,
    )


#: Sharpe assigned to combos whose backtest crashed / stalled (not a measurement)
FAILED_BACKTEST_SHARPE = -999.0


def _trial_sharpe_variance(results: list[BacktestMetrics]) -> float | None:
    """Sample variance of trial mark-to-market Sharpes (daily units) for DSR.

    Same statistic as the tested one (``mtm_sharpe_daily``). Only real
    measurements enter the variance: trials without MTM statistics (crashed /
    stalled combos, unavailable series) and non-finite values are excluded.
    They still count toward the trial count N (they were tried).
    """
    sharpes = [
        r.mtm_sharpe_daily
        for r in results
        if r.mtm_sharpe_daily is not None and math.isfinite(r.mtm_sharpe_daily)
    ]
    if len(sharpes) < 2:
        return None
    mean_sr = sum(sharpes) / len(sharpes)
    return sum((s - mean_sr) ** 2 for s in sharpes) / (len(sharpes) - 1)


def _kill_pool_workers(executor: ProcessPoolExecutor) -> None:
    """Force-kill worker processes of a stalled ProcessPoolExecutor.

    Hung NT backtests never return, so ``shutdown(wait=True)`` would block
    forever. Mirrors ``vibe_quant.discovery.fitness._force_shutdown_pool``.
    """
    processes = getattr(executor, "_processes", None)
    if not processes:
        return
    for pid, proc in list(processes.items()):
        try:
            if proc.is_alive():
                logger.warning("Killing stalled screening worker pid=%d", pid)
                proc.kill()
                proc.join(timeout=5)
        except Exception:
            logger.warning("Could not kill worker pid=%d", pid, exc_info=True)


class ScreeningPipeline:
    """Pipeline for parallel parameter sweep screening.

    The pipeline runs NautilusTrader backtests in simplified mode
    for fast parameter exploration. It:

    1. Compiles the DSL strategy
    2. Builds parameter grid from sweep section
    3. Runs backtests in parallel
    4. Filters and ranks results
    5. Stores results in database

    Example:
        pipeline = ScreeningPipeline(
            dsl=strategy_dsl,
            backtest_runner=run_backtest,  # Or None for mock
        )
        result = pipeline.run(filters=MetricFilters(min_sharpe=0.5))
    """

    #: Abandon the sweep's pending combos if NO backtest completes within
    #: this many seconds (a hung NT worker would otherwise stall forever)
    DEFAULT_STALL_TIMEOUT_S: float = 600.0

    #: Fixed top-level overrides recorded on every row (na00j). Empty by default, so a
    #: pipeline built without ``__init__`` (test doubles) records today's parameters.
    _strategy_overrides: Mapping[str, Any] = MappingProxyType({})

    def __init__(
        self,
        dsl: StrategyDSL,
        backtest_runner: BacktestRunner | None = None,
        max_workers: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        stall_timeout_s: float | None = None,
        liquidation: LiquidationSettings | None = None,
        starting_balance: int = DEFAULT_STARTING_BALANCE_USDT,
        strategy_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize screening pipeline.

        Args:
            dsl: Parsed StrategyDSL with sweep parameters
            backtest_runner: Function to run single backtest. Uses mock if None.
            max_workers: Max parallel workers. Defaults to cpu_count - 1.
            start_date: Backtest start date (YYYY-MM-DD) for DSR day count.
            end_date: Backtest end date (YYYY-MM-DD) for DSR day count.
            stall_timeout_s: Seconds with zero completions before pending
                combos are abandoned with sentinel metrics.
            liquidation: Validated run-level liquidation settings (bead
                vibe-quant-2gyk1.1.8). Off by default; when on, ``save_results``
                records the per-row liquidation summary (D1).
            starting_balance: The run's account size, parsed once from
                ``parameters.initial_balance`` by :func:`create_screening_pipeline`
                (bead vibe-quant-zpbmo). ``save_results`` records it on EVERY sweep
                row of the run, the ``-999`` sentinel rows included (R12), so a
                later overfitting rerun reads the same balance. The default 1000
                keeps every pre-existing row and number byte-identical (R9).
            strategy_overrides: The run's fixed, typed strategy-config values
                (:func:`~vibe_quant.screening.nt_runner.resolve_screening_params`),
                RECORDED on every ``sweep_results`` row under their config-field names
                (bead vibe-quant-na00j). A row otherwise shows only its swept grid
                point, not every value the run used. They ride beside the sweep, so
                strings and booleans stay typed. Empty = today's parameters, byte for
                byte.
        """
        self.dsl = dsl
        self._runner = backtest_runner or _run_mock_backtest
        self._max_workers = max_workers or max(1, cpu_count() - 1)
        self._start_date = start_date
        self._end_date = end_date
        self._stall_timeout_s = stall_timeout_s or self.DEFAULT_STALL_TIMEOUT_S
        self._liquidation = liquidation or LiquidationSettings()
        self._starting_balance = validate_starting_balance(starting_balance)
        self._strategy_overrides = dict(strategy_overrides or {})

        # Build parameter grid
        self._param_grid = build_parameter_grid(dsl.sweep)

    @property
    def starting_balance(self) -> int:
        """The account size every row of this run records (USDT)."""
        return self._starting_balance

    @property
    def strategy_name(self) -> str:
        """Get strategy name from DSL."""
        return self.dsl.name

    @property
    def num_combinations(self) -> int:
        """Get total number of parameter combinations."""
        return len(self._param_grid)

    def run(
        self,
        filters: MetricFilters | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        apply_dsr: bool = True,
        dsr_significance: float = 0.05,
    ) -> ScreeningResult:
        """Run the screening pipeline.

        Args:
            filters: Hard metric filters. Uses defaults if None.
            progress_callback: Optional callback(completed, total) for progress.
            apply_dsr: Apply Deflated Sharpe Ratio filter (default True).
            dsr_significance: DSR p-value threshold (default 0.05).

        Returns:
            ScreeningResult with filtered and ranked results
        """
        start_time = time.time()
        filters = filters or MetricFilters()

        logger.info(
            "Starting screening for %s with %d combinations, %d workers",
            self.strategy_name,
            self.num_combinations,
            self._max_workers,
        )

        # Run backtests: sequential if 1 worker, parallel otherwise
        if self._max_workers == 1:
            all_results = self._run_sequential(progress_callback)
        else:
            all_results = self._run_parallel(progress_callback)

        # Apply hard metric filters
        filtered = filter_by_metrics(all_results, filters)
        logger.info(
            "Filtered %d/%d results pass hard filters",
            len(filtered),
            len(all_results),
        )

        # Apply DSR overfitting filter -- also when exactly ONE combo survives:
        # a lone survivor of a 1000-combo grid is the most selection-biased
        # result there is.
        if apply_dsr and filtered:
            dsr = DeflatedSharpeRatio(significance_level=dsr_significance)
            # Every combo tried counts as a trial, including failed/NaN ones.
            num_trials = len(all_results)
            # DSR tests the MARK-TO-MARKET daily statistics (bd vibe-quant-68630),
            # not NT's realized Sharpe (a multi-day trade books its whole PnL on
            # its exit day there). The Sharpe is already per day and T is the
            # length of that series. A result without MTM statistics fails.
            trials_sharpe_variance = _trial_sharpe_variance(all_results)
            dsr_passed = []
            dsr_missing = 0
            for r in filtered:
                result, reason = mtm_dsr_calculate(
                    dsr,
                    r.mtm_sharpe_daily,
                    r.mtm_skewness,
                    r.mtm_kurtosis,
                    r.mtm_days,
                    num_trials,
                    trials_sharpe_variance=trials_sharpe_variance,
                )
                if reason is not None:
                    dsr_missing += 1
                if result is not None and result.is_significant:
                    dsr_passed.append(r)
            if dsr_missing:
                logger.warning(
                    "DSR filter: %d/%d result(s) have no mark-to-market statistics "
                    "and fail closed",
                    dsr_missing,
                    len(filtered),
                )
            logger.info(
                "DSR filter: %d/%d pass significance test (p<%.2f, %d trials)",
                len(dsr_passed),
                len(filtered),
                dsr_significance,
                num_trials,
            )
            filtered = dsr_passed

        # Rank ALL results by Sharpe (save everything so user can inspect)
        ranked_all = rank_by_sharpe(all_results)

        # Compute Pareto front on filtered results only
        ranked_filtered = rank_by_sharpe(filtered)
        pareto_indices_filtered = compute_pareto_front(ranked_filtered)

        # Map pareto indices back to the all-results list using object
        # identity (id()) instead of json.dumps of parameters, which
        # collides when two results share identical parameters.
        pareto_objects = {id(ranked_filtered[i]) for i in pareto_indices_filtered}
        pareto_indices = [i for i, r in enumerate(ranked_all) if id(r) in pareto_objects]
        logger.info("Found %d Pareto-optimal results", len(pareto_indices))

        execution_time = time.time() - start_time

        return ScreeningResult(
            strategy_name=self.strategy_name,
            total_combinations=self.num_combinations,
            passed_filters=len(filtered),
            execution_time_seconds=execution_time,
            results=ranked_all,
            pareto_optimal_indices=pareto_indices,
        )

    def _run_sequential(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[BacktestMetrics]:
        """Run backtests sequentially (single worker)."""
        results: list[BacktestMetrics] = []
        total = self.num_combinations
        for i, params in enumerate(self._param_grid):
            try:
                result = self._runner(params)
                results.append(result)
            except DataUnavailableError:
                raise
            except Exception as e:
                logger.warning("Backtest failed for params %s: %s", params, e)
                results.append(BacktestMetrics(parameters=params, sharpe_ratio=FAILED_BACKTEST_SHARPE))
            if progress_callback:
                progress_callback(i + 1, total)
        return results

    def _run_parallel(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[BacktestMetrics]:
        """Run backtests in parallel using ProcessPoolExecutor.

        A wait-loop with a stall timeout replaces the old
        ``as_completed`` + ``future.result(timeout=300)`` pattern: that
        timeout was dead code (``as_completed`` only yields already-done
        futures), so a single hung NT worker stalled the sweep forever.
        Now, if NO backtest completes within ``stall_timeout_s``, all
        pending combos get sentinel metrics and worker processes are
        killed so the pipeline still returns.

        Args:
            progress_callback: Optional progress callback

        Returns:
            List of all backtest results
        """
        from concurrent.futures import FIRST_COMPLETED, wait

        results: list[BacktestMetrics] = []
        total = self.num_combinations
        completed = 0
        stalled = False

        executor = ProcessPoolExecutor(max_workers=self._max_workers)
        try:
            future_to_params = {
                executor.submit(self._runner, params): params for params in self._param_grid
            }
            pending = set(future_to_params)

            while pending:
                done, pending = wait(
                    pending, timeout=self._stall_timeout_s, return_when=FIRST_COMPLETED
                )
                if not done:
                    stalled = True
                    logger.warning(
                        "No backtest completed within %.0fs — abandoning %d "
                        "pending combinations with sentinel metrics",
                        self._stall_timeout_s,
                        len(pending),
                    )
                    for future in pending:
                        future.cancel()
                        results.append(
                            BacktestMetrics(
                                parameters=future_to_params[future],
                                sharpe_ratio=FAILED_BACKTEST_SHARPE,
                            )
                        )
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)
                    break

                for future in done:
                    params = future_to_params[future]
                    try:
                        results.append(future.result())
                    except DataUnavailableError:
                        raise
                    except Exception as e:
                        # Log error but continue with other backtests.
                        # Sentinel -999 avoids -inf propagation through
                        # Pareto / ranking arithmetic.
                        logger.warning("Backtest failed for params %s: %s", params, e)
                        results.append(
                            BacktestMetrics(parameters=params, sharpe_ratio=FAILED_BACKTEST_SHARPE)
                        )
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
        finally:
            if stalled:
                _kill_pool_workers(executor)
            # wait=False when stalled: hung workers would block shutdown
            executor.shutdown(wait=not stalled, cancel_futures=True)

        return results

    def save_results(
        self,
        result: ScreeningResult,
        state_manager: StateManager,
        run_id: int,
    ) -> None:
        """Save screening results to database.

        Every row carries this run's ``starting_balance`` (R12), the ``-999``
        sentinel rows included, so all rows in the batch share one key set.

        Args:
            result: ScreeningResult to save
            state_manager: StateManager instance
            run_id: Backtest run ID to associate results with
        """
        # Convert results to dicts for batch insert
        result_dicts: list[dict[str, Any]] = []
        # Use set for O(1) membership test instead of O(k) list scan
        pareto_set = frozenset(result.pareto_optimal_indices)

        for i, metrics in enumerate(result.results):
            is_pareto = i in pareto_set
            # na00j: a row shows every value the run used -- the swept grid point AND
            # the run's fixed top-level overrides. Empty overrides keep the stored
            # JSON byte-identical to before (same dict order, same text).
            parameters: dict[str, Any] = metrics.parameters
            if self._strategy_overrides:
                parameters = {**parameters, **self._strategy_overrides}
            row: dict[str, Any] = {
                "parameters": parameters,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
                "profit_factor": metrics.profit_factor,
                "win_rate": metrics.win_rate,
                "total_trades": metrics.total_trades,
                "total_fees": metrics.total_fees,
                "total_funding": metrics.total_funding,
                "execution_time_seconds": metrics.execution_time_seconds,
                "skewness": metrics.skewness,
                "kurtosis": metrics.kurtosis,
                "mtm_sharpe_daily": metrics.mtm_sharpe_daily,
                "mtm_skewness": metrics.mtm_skewness,
                "mtm_kurtosis": metrics.mtm_kurtosis,
                "mtm_days": metrics.mtm_days,
                # R12: EVERY row of the run records the balance it was measured at,
                # the -999 sentinel rows included, so the batch has one key set and a
                # later overfitting rerun reads the same account size (R11).
                "starting_balance": float(self._starting_balance),
                "is_pareto_optimal": is_pareto,
            }
            # D1: the liquidation summary is written ONLY when the mode is on, so
            # a mode-off run (and every older row) keeps both columns NULL.
            # A sentinel row (crashed / stalled combo, sharpe == -999) is not a
            # measurement: it stores NULLs, so the UI shows "-" and never a fake
            # "0 hits". Both keys are still set on EVERY mode-on row, because
            # save_sweep_results_batch derives its column list from the first row
            # and binds each row's values positionally (mixed key sets inside one
            # batch raise a sqlite3.ProgrammingError).
            if self._liquidation.on:
                if metrics.sharpe_ratio == FAILED_BACKTEST_SHARPE:
                    row["liquidation_count"] = None
                    row["liquidation_forfeited"] = None
                    row["exit_timing"] = None
                else:
                    row["liquidation_count"] = int(metrics.liquidation_count)
                    row["liquidation_forfeited"] = float(metrics.liquidation_forfeited)
                    # D1b: the refined-exit counts as JSON text (a sentinel never
                    # ran the engine, so it stores NULL above, not a fake zero).
                    row["exit_timing"] = (
                        json.dumps(metrics.exit_timing, sort_keys=True)
                        if metrics.exit_timing is not None
                        else None
                    )
            result_dicts.append(row)

        # Batch save
        state_manager.save_sweep_results_batch(run_id, result_dicts)

        logger.info(
            "Saved %d screening results for run %d",
            len(result_dicts),
            run_id,
        )


def create_screening_pipeline(
    dsl: StrategyDSL,
    use_mock: bool = False,
    max_workers: int | None = None,
    symbols: list[str] | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    catalog_path: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    strategy_overrides: Mapping[str, Any] | None = None,
) -> ScreeningPipeline:
    """Factory function to create a screening pipeline.

    When use_mock=False, creates a real NautilusTrader BacktestNode-based
    runner for actual screening. Requires symbols and date range.

    Args:
        dsl: Parsed StrategyDSL
        use_mock: If True, use mock backtest runner (for testing)
        max_workers: Max parallel workers
        symbols: List of symbols (required when use_mock=False)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        catalog_path: Path to ParquetDataCatalog (uses default if None)
        parameters: The run's launch parameters, parsed in this entry point's order
            (:data:`~vibe_quant.launch_settings.PARSE_ORDER`) into one
            :class:`~vibe_quant.launch_settings.LaunchSettings`. The venue-level fee
            schedule (``fee_tier`` / ``bnb_discount``) is derived from them, so an
            API/CLI launch that stored ``bnb_discount`` reaches the runner; it is the
            LAST field and only the real branch parses it (the mock branch never does).
            A non-``"market"`` ``entry_order`` is refused here (post-only is
            validation-only, bead vibe-quant-7eek0), before any engine exists. The
            run-level liquidation settings are derived from the same parameters with
            :meth:`LiquidationSettings.from_params` (the single source; there is no
            screening CLI flag) and reach the runner and the pipeline. The account size
            is ``parameters.initial_balance``, parsed ONCE here with
            :func:`~vibe_quant.balance_settings.parse_initial_balance` (bead
            vibe-quant-zpbmo) before the mock/real split, so an invalid stored value
            fails the run even when no engine would be built. The run-level sizing
            settings are ``parameters.max_position_pct`` / ``risk_per_trade``, parsed
            here with :meth:`SizingSettings.from_params` and checked against the run's
            leverage (S13) in the same place and for the same reason (bead
            vibe-quant-2gyk1.2); the runner merges only the non-default keys into its
            strategy config.
        strategy_overrides: The run's fixed, typed strategy-config values (its resolved
            top-level strategy parameters, see
            :func:`~vibe_quant.screening.nt_runner.resolve_screening_params`), handed to
            the real runner beside the sweep and recorded on EVERY ``sweep_results`` row
            (bead vibe-quant-na00j). The mock runner applies none of them to a config,
            but the pipeline still records them, so the stored row is always truthful.

    Returns:
        Configured ScreeningPipeline

    Raises:
        ValueError: ``parameters["entry_order"]`` is not ``"market"``.
        BalanceConfigError: the stored ``initial_balance`` is not a whole number
            in [1, 1000000000] (never silently defaulted).
        LiquidationConfigError: the stored ``liquidation_mode`` / ``leverage``
            is invalid (E10: the run fails before any engine exists).
        SizingConfigError: the stored ``max_position_pct`` / ``risk_per_trade`` is
            invalid, or the cap exceeds the run's leverage (S13/S15).
    """
    run_params: Mapping[str, Any] = parameters or {}
    # Parse ONCE, before the mock/real split, in this entry point's order (R27): an
    # invalid stored balance, mode, leverage or sizing fails the run even when no
    # engine would be built, and the leverage cap is checked where it always was.
    pre = dict(iter_fields(run_params, PARSE_ORDER["screening_pipeline_pre_tick"]))
    entry_order = cast("str", pre["entry_order"])

    from vibe_quant.screening.nt_runner import refuse_non_market_entry

    refuse_non_market_entry(entry_order)
    launch = LaunchSettings.from_fields(pre)

    if use_mock:
        runner: BacktestRunner = _run_mock_backtest
    else:
        from vibe_quant.screening.nt_runner import NTScreeningRunner

        effective_symbols = symbols or ["BTCUSDT"]
        # Convert DSL to dict for pickling across process boundaries
        dsl_dict = _dsl_to_dict(dsl)
        # Resolve the fill ticks ONCE here, before the worker pool; missing
        # 1m data aborts now, workers never build ticks (vibe-quant-yul7u.12).
        # The fee schedule is parsed AFTER them (pipeline.py:610/613), so missing
        # ticks win over a bad fee tier.
        fill_ticks = resolve_parent_fill_ticks(
            dsl_dict, effective_symbols, start_date, end_date, catalog_path
        )
        post = dict(
            iter_fields(
                run_params, PARSE_ORDER["screening_pipeline_post_tick"], parsed=pre
            )
        )
        launch = LaunchSettings.from_fields({**pre, **post})
        runner = NTScreeningRunner(
            dsl_dict=dsl_dict,
            symbols=effective_symbols,
            start_date=start_date,
            end_date=end_date,
            catalog_path=catalog_path,
            fill_ticks=fill_ticks,
            launch=launch,
            strategy_overrides=strategy_overrides,
        )
    return ScreeningPipeline(
        dsl=dsl,
        backtest_runner=runner,
        max_workers=max_workers,
        start_date=start_date,
        end_date=end_date,
        liquidation=launch.liquidation,
        starting_balance=launch.starting_balance,
        # Recorded on every row beside the swept grid point (na00j); the real runner
        # already merged them into each compiled config above.
        strategy_overrides=strategy_overrides,
    )


def _dsl_to_dict(dsl: StrategyDSL) -> dict[str, Any]:
    """Convert a StrategyDSL to a serializable dict for pickling.

    Handles Pydantic BaseModel (model_dump), dataclasses (asdict),
    and plain dicts.

    Args:
        dsl: Parsed StrategyDSL object.

    Returns:
        Dictionary representation suitable for validate_strategy_dict().
    """
    # Pydantic BaseModel — StrategyDSL is a Pydantic v2 model
    if hasattr(dsl, "model_dump"):
        return dsl.model_dump()

    import dataclasses

    def _to_dict(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, list):
            return [_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        return obj

    return dict(_to_dict(dsl))  # type: ignore[arg-type]


def resolve_fill_ticks_for(
    timeframe: str,
    loaded_timeframes: Iterable[str],
    symbols: Sequence[str],
    start: str,
    end: str,
    catalog_path: str | Path | None = None,
) -> FillTickSet | None:
    """Parent-side fill-tick resolve for a run (vibe-quant-yul7u.12).

    The tick timeframe comes from ``screening_fill_mode`` (the finest LOADED
    timeframe), the same call the runner makes; ``NTScreeningRunner`` raises
    on a mismatching set. 1m strategies and the "bar" release mode use no
    ticks -> None.

    Raises:
        MissingBarDataError: no/partial 1m data for ``[start, end]``.
    """
    from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH
    from vibe_quant.data.fill_ticks import resolve_fill_ticks
    from vibe_quant.screening.nt_runner import screening_fill_mode

    tick_tf = screening_fill_mode(timeframe, loaded_timeframes).tick_timeframe
    if tick_tf is None:
        return None
    cat = DEFAULT_CATALOG_PATH if catalog_path is None else catalog_path
    ticks = resolve_fill_ticks(symbols, tick_tf, start, end, cat)
    logger.info(
        "Fill ticks resolved in parent (pid %d): tf=%s symbols=%s window=%s..%s",
        os.getpid(), tick_tf, sorted(symbols), start, end,
    )
    return ticks


def resolve_parent_fill_ticks(
    dsl_dict: dict[str, Any],
    symbols: Sequence[str],
    start: str,
    end: str,
    catalog_path: str | Path | None = None,
) -> FillTickSet | None:
    """:func:`resolve_fill_ticks_for` for a DSL dict.

    Loaded timeframes come from the SAME helper ``NTScreeningRunner._compile``
    uses (``loaded_timeframes``), so parent and runner always agree on the
    finest loaded timeframe the tick set is resolved for (vibe-quant-yul7u.12).
    """
    from vibe_quant.engine_run import loaded_timeframes

    return resolve_fill_ticks_for(
        str(dsl_dict["timeframe"]),
        loaded_timeframes(dsl_dict),
        symbols,
        start,
        end,
        catalog_path,
    )


class ParentFillTicks:
    """Lazy parent-side resolve for runners that backtest many windows.

    The set's directories do not depend on the window; only the coverage check
    does. So one set is reused for every window inside the span resolved so
    far, and the span is widened (re-resolved over the union) when a window
    falls outside it.
    """

    def __init__(
        self,
        dsl_dict: dict[str, Any],
        symbols: Sequence[str],
        catalog_path: str | Path | None = None,
    ) -> None:
        self._dsl_dict = dsl_dict
        self._symbols = list(symbols)
        self._catalog_path = catalog_path
        self._span: tuple[str, str] | None = None
        self._ticks: FillTickSet | None = None

    def get(self, start: str, end: str) -> FillTickSet | None:
        """The tick set covering ``[start, end]`` (None when the mode has no ticks)."""
        if self._span is not None and self._span[0] <= start and end <= self._span[1]:
            return self._ticks
        lo, hi = (start, end) if self._span is None else (
            min(start, self._span[0]), max(end, self._span[1])
        )
        self._ticks = resolve_parent_fill_ticks(
            self._dsl_dict, self._symbols, lo, hi, self._catalog_path
        )
        self._span = (lo, hi)
        return self._ticks
