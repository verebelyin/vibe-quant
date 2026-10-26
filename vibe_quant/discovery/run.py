"""Discovery run helpers: the real/mock backtest functions and the run record.

``_make_nt_backtest_fn``, ``_log_data_catalog_info``, ``_check_data_available``,
``_make_cell_fns``, ``walk_forward_efficiency`` and ``_resolve_seed`` moved
verbatim from ``vibe_quant.discovery.__main__`` (bead vibe-quant-rabmo.7, step
T14). This module also owns the ``BacktestAdapter`` / ``RecordStore`` seams
(``NTBacktests``, ``MockBacktests``, ``StateRecordStore``) the CLI is wired to
by T15. ``DiscoveryPipeline``, ``NTBacktestFn`` and ``_mock_backtest`` are THIS
module's globals, so a patch of the single-run path targets
``vibe_quant.discovery.run.<name>``.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from vibe_quant.bankroll_grid import BankrollCell, BankrollGridError
from vibe_quant.db.result_notes import dump_notes
from vibe_quant.discovery import pooling, record
from vibe_quant.discovery.backtest_fn import full_range_headline
from vibe_quant.discovery.mock_backtest import mock_backtest as _mock_backtest
from vibe_quant.discovery.pipeline import DiscoveryPipeline as DiscoveryPipeline  # re-export
from vibe_quant.discovery.pooling import _CellFns
from vibe_quant.discovery.run_spec import DiscoveryRunSpec
from vibe_quant.discovery.seed import load_seed_chromosomes
from vibe_quant.utils import compute_day_count, log_dir

if TYPE_CHECKING:
    import argparse

    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.jobs.manager import BacktestJobManager
    from vibe_quant.launch_settings import LaunchSettings
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.sizing_settings import SizingSettings

logger = logging.getLogger(__name__)


# NTBacktestFn lives in vibe_quant.discovery.backtest_fn so it pickles by
# module reference and worker processes can unpickle it. It is imported, never
# redefined, so every cell function builds the same class.
from vibe_quant.discovery.backtest_fn import (  # noqa: E402
    NTBacktestFn,
    SymbolAgg,
)


def _make_nt_backtest_fn(
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    windows: list[tuple[str, str]] | None = None,
    min_trades: int = 0,
    symbol_agg: SymbolAgg = "portfolio",
    *,
    fee_schedule: FeeSchedule,
    entry_order: str,
    liquidation: LiquidationSettings,
    sizing: SizingSettings | None = None,
) -> NTBacktestFn:
    """Create a picklable backtest function using real NautilusTrader screening runner.

    ``fee_schedule``, ``entry_order`` and ``liquidation`` are required (no
    defaults) so a new call site cannot silently run at VIP0/10x while the
    rest of the run is on BNB or at the run's liquidation settings
    (bd vibe-quant-2gyk1.4, vibe-quant-2gyk1.1.9).

    ``sizing`` (bead vibe-quant-2gyk1.2) is the run's parsed
    :class:`~vibe_quant.sizing_settings.SizingSettings`. Every call site in
    ``main`` passes it; ``None`` means the defaults, which add no key and keep
    every pre-existing number byte-identical.
    """
    return NTBacktestFn(
        symbols, timeframe, start_date, end_date, windows=windows, min_trades=min_trades,
        symbol_agg=symbol_agg, fee_schedule=fee_schedule, entry_order=entry_order,
        liquidation=liquidation, sizing=sizing,
    )


def _log_data_catalog_info(symbols: list[str], timeframe: str) -> None:
    """Log data catalog details: available symbols, bar counts, date ranges."""
    try:
        from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

        bar_dir = DEFAULT_CATALOG_PATH / "data" / "bar"
        if not bar_dir.exists():
            logger.info("Data catalog: no bar directory at %s", bar_dir)
            return

        for sym in symbols:
            instrument_id = f"{sym}-PERP.BINANCE"
            found_dirs = [
                d.name for d in bar_dir.iterdir()
                if d.is_dir() and instrument_id in d.name
            ]
            if found_dirs:
                # Count parquet files to estimate data volume
                for dname in found_dirs:
                    dpath = bar_dir / dname
                    parquet_files = list(dpath.glob("*.parquet"))
                    total_size = sum(f.stat().st_size for f in parquet_files)
                    logger.info(
                        "Data catalog: %s → %s (%d files, %.1f MB)",
                        sym,
                        dname,
                        len(parquet_files),
                        total_size / (1024 * 1024),
                    )
            else:
                logger.warning("Data catalog: %s NOT FOUND in catalog", sym)
    except Exception:
        logger.debug("Could not read data catalog info", exc_info=True)


def _check_data_available(symbols: list[str]) -> bool:
    """Check if ParquetDataCatalog has data for the given symbols."""
    try:
        from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

        if not DEFAULT_CATALOG_PATH.exists():
            return False
        # Check if catalog directory has any bar data
        bar_dir = DEFAULT_CATALOG_PATH / "data" / "bar"
        if not bar_dir.exists():
            return False
        # Check for at least one symbol's data
        for sym in symbols:
            instrument_id = f"{sym}-PERP.BINANCE"
            # Look for any bar type directory containing this instrument
            found = False
            for d in bar_dir.iterdir():
                if d.is_dir() and instrument_id in d.name:
                    found = True
                    break
            if not found:
                return False
        return True
    except Exception:
        return False


def _make_cell_fns(
    *,
    use_mock: bool,
    symbols: list[str],
    timeframe: str,
    symbol_agg: SymbolAgg,
    fee_schedule: FeeSchedule,
    entry_order: str,
    liquidation: LiquidationSettings,
    sizing: SizingSettings,
    train_start: str,
    train_end: str,
    eval_windows: list[tuple[str, str]] | None,
    min_trades: int,
    holdout: tuple[str, str] | None,
    needs_factory: bool,
    evaluator: str = "nt",
) -> _CellFns:
    """The train / holdout / cross-window + WFA functions of ONE bankroll setup.

    ``main`` calls it once for a run-level setup and :func:`_run_bankroll_grid` once per cell, so
    every stage of a run is built from the same ``liquidation`` + ``sizing`` (bead
    vibe-quant-t8nrv). ``holdout`` is ``(start, end)`` or ``None`` (no split).
    """
    if use_mock:
        return _CellFns(
            backtest_fn=_mock_backtest,
            holdout_backtest_fn=_mock_backtest if holdout else None,
            backtest_fn_factory=(lambda s, e: _mock_backtest) if needs_factory else None,
        )
    backtest_fn = _make_nt_backtest_fn(
        symbols=symbols,
        timeframe=timeframe,
        start_date=train_start,
        end_date=train_end,
        windows=eval_windows,
        min_trades=min_trades,
        symbol_agg=symbol_agg,
        fee_schedule=fee_schedule,
        entry_order=entry_order,
        liquidation=liquidation,
        sizing=sizing,
    )
    if evaluator == "fast":
        # The fast tier scores the GA (NT for genomes it cannot replay); the pipeline
        # re-runs the top-K on NT before any gate (vibe-quant-z8vqv.5). Holdout,
        # cross-window and WFA below stay NT.
        from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

        backtest_fn = FastBacktestFn(
            symbols, timeframe, train_start, train_end, windows=eval_windows,
            min_trades=min_trades, symbol_agg=symbol_agg, fee_schedule=fee_schedule,
            entry_order=entry_order, liquidation=liquidation, sizing=sizing, nt_fallback=True,
        )
    holdout_backtest_fn = (
        _make_nt_backtest_fn(
            symbols=symbols,
            timeframe=timeframe,
            start_date=holdout[0],
            end_date=holdout[1],
            symbol_agg=symbol_agg,
            fee_schedule=fee_schedule,
            entry_order=entry_order,
            liquidation=liquidation,
            sizing=sizing,
        )
        if holdout
        else None
    )
    backtest_fn_factory = None
    if needs_factory:
        _syms = symbols
        _tf = timeframe
        _agg: SymbolAgg = symbol_agg
        _fee = fee_schedule
        _entry = entry_order
        _liq = liquidation
        _sz = sizing

        def backtest_fn_factory(s: str, e: str) -> NTBacktestFn:
            return NTBacktestFn(
                _syms, _tf, s, e, symbol_agg=_agg, fee_schedule=_fee,
                entry_order=_entry, liquidation=_liq, sizing=_sz,
            )

    return _CellFns(backtest_fn, holdout_backtest_fn, backtest_fn_factory)


def walk_forward_efficiency(
    *,
    oos_return: float,
    oos_days: int | None,
    is_return: float,
    is_days: int | None,
) -> float | None:
    """Length-normalized walk-forward efficiency (Pardo).

    ``(oos_return / oos_days) / (is_return / is_days)``. A stationary edge
    scores ~1.0 regardless of how long each period is. ``None`` when the
    in-sample return isn't positive (no edge to retain) or a length is
    unknown -- a negative IS with a positive OOS is NOT "infinitely efficient".
    """
    if not oos_days or not is_days or is_days <= 0 or oos_days <= 0:
        return None
    is_per_day = is_return / is_days
    if not is_per_day > 0:
        return None
    return (oos_return / oos_days) / is_per_day


def _resolve_seed(arg: int | None) -> int:
    """Explicit --seed, else a fresh OS-entropy seed (recorded so runs can be replayed)."""
    return arg if arg is not None else random.SystemRandom().randrange(2**32)


class BacktestAdapter(Protocol):
    """The backtest engine of a discovery run: real NautilusTrader or the mock."""

    @property
    def mock(self) -> bool:
        """Guards the full-range + train-range re-runs exactly as `use_mock` did."""
        ...

    def data_available(self, symbols: list[str]) -> bool: ...

    def cell_fns(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        symbol_agg: SymbolAgg,
        launch: LaunchSettings,
        train_start: str,
        train_end: str,
        eval_windows: list[tuple[str, str]] | None,
        min_trades: int,
        holdout: tuple[str, str] | None,
        needs_factory: bool,
        evaluator: str,
    ) -> _CellFns: ...

    def nt_fn(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        symbol_agg: SymbolAgg,
        launch: LaunchSettings,
    ) -> NTBacktestFn: ...


@dataclass(frozen=True)
class NTBacktests:
    """The real NautilusTrader adapter."""

    mock: bool = False

    def data_available(self, symbols: list[str]) -> bool:
        return _check_data_available(symbols)

    def cell_fns(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        symbol_agg: SymbolAgg,
        launch: LaunchSettings,
        train_start: str,
        train_end: str,
        eval_windows: list[tuple[str, str]] | None,
        min_trades: int,
        holdout: tuple[str, str] | None,
        needs_factory: bool,
        evaluator: str,
    ) -> _CellFns:
        return _make_cell_fns(
            use_mock=False,
            symbols=symbols,
            timeframe=timeframe,
            symbol_agg=symbol_agg,
            fee_schedule=launch.fee_schedule,
            entry_order=launch.entry_order,
            liquidation=launch.liquidation,
            sizing=launch.sizing,
            train_start=train_start,
            train_end=train_end,
            eval_windows=eval_windows,
            min_trades=min_trades,
            holdout=holdout,
            needs_factory=needs_factory,
            evaluator=evaluator,
        )

    def nt_fn(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        symbol_agg: SymbolAgg,
        launch: LaunchSettings,
    ) -> NTBacktestFn:
        return _make_nt_backtest_fn(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            symbol_agg=symbol_agg,
            fee_schedule=launch.fee_schedule,
            entry_order=launch.entry_order,
            liquidation=launch.liquidation,
            sizing=launch.sizing,
        )


@dataclass(frozen=True)
class MockBacktests:
    """The synthetic adapter: no catalog, no engine."""

    mock: bool = True

    def data_available(self, symbols: list[str]) -> bool:
        return True

    def cell_fns(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        symbol_agg: SymbolAgg,
        launch: LaunchSettings,
        train_start: str,
        train_end: str,
        eval_windows: list[tuple[str, str]] | None,
        min_trades: int,
        holdout: tuple[str, str] | None,
        needs_factory: bool,
        evaluator: str,
    ) -> _CellFns:
        return _make_cell_fns(
            use_mock=True,
            symbols=symbols,
            timeframe=timeframe,
            symbol_agg=symbol_agg,
            fee_schedule=launch.fee_schedule,
            entry_order=launch.entry_order,
            liquidation=launch.liquidation,
            sizing=launch.sizing,
            train_start=train_start,
            train_end=train_end,
            eval_windows=eval_windows,
            min_trades=min_trades,
            holdout=holdout,
            needs_factory=needs_factory,
            evaluator=evaluator,
        )

    def nt_fn(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        symbol_agg: SymbolAgg,
        launch: LaunchSettings,
    ) -> NTBacktestFn:
        raise RuntimeError("mock adapter runs no NT backtest")


class RecordStore(Protocol):
    """The persistence of one discovery run: its row, its result and its job."""

    @property
    def state(self) -> StateManager:
        """The prior-trial scan + seed chromosomes read through it."""
        ...

    def get_run(self, run_id: int) -> dict[str, Any] | None: ...

    def save_result(self, run_id: int, metrics: dict[str, object]) -> None: ...

    def mark_completed(self, run_id: int) -> None: ...

    def mark_failed(self, run_id: int, error: str) -> None: ...

    def mark_job_failed(self, run_id: int, error: str) -> None: ...


@dataclass(frozen=True)
class StateRecordStore:
    """The CLI store: StateManager for the row/result, job manager for the job."""

    state: StateManager
    job_manager: BacktestJobManager

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        return self.state.get_backtest_run(run_id)

    def save_result(self, run_id: int, metrics: dict[str, object]) -> None:
        self.state.save_backtest_result(run_id, metrics)

    def mark_completed(self, run_id: int) -> None:
        self.state.update_backtest_run_status(run_id, "completed")
        self.job_manager.mark_completed(run_id)

    def mark_failed(self, run_id: int, error: str) -> None:
        self.state.update_backtest_run_status(run_id, "failed", error_message=error)
        self.job_manager.mark_completed(run_id, error=error)

    def mark_job_failed(self, run_id: int, error: str) -> None:
        self.job_manager.mark_completed(run_id, error=error)


@dataclass(frozen=True)
class RunOutcome:
    """What a discovery run ended with: the process exit code, the line it printed, its notes."""

    exit_code: int
    message: str
    notes: dict[str, object] | None


def run_discovery(
    args: argparse.Namespace,
    backtests: BacktestAdapter,
    store: RecordStore,
    *,
    run_id: int,
    started_at: float,
) -> RunOutcome:
    """Run one discovery run and record its rows; never raises (a failure is an outcome)."""
    try:
        run = store.get_run(run_id)
        if run is None:
            error = f"Run {run_id} not found"
            store.mark_job_failed(run_id, error)
            print(error)
            return RunOutcome(1, error, None)

        spec = DiscoveryRunSpec.from_args(args)

        symbols = spec.symbol_list()
        if not symbols:
            symbols = run.get("symbols", [])

        # Venue-level fee schedule (bd vibe-quant-2gyk1.4) shared by every
        # screening backtest of this run (train, holdout, cross-window, WFA).
        # The entry-order mode comes from the run parameters: post-only is
        # validation-only and NTBacktestFn refuses it before any engine runs.
        fee_schedule = spec.fee_schedule()
        raw_run_params = run.get("parameters") or {}
        if isinstance(raw_run_params, str):
            import json as _json

            try:
                raw_run_params = _json.loads(raw_run_params)
            except (ValueError, TypeError):
                raw_run_params = {}
        if not isinstance(raw_run_params, dict):
            raw_run_params = {}
        entry_order = str(raw_run_params.get("entry_order", "market"))
        logger.info("Discovery fee schedule: %s", fee_schedule.to_notes())

        # Run-level liquidation settings (bead vibe-quant-6wq6w): one leverage
        # honoured in every mode. --leverage is the setting and
        # --liquidation-leverage a deprecated alias; the shared rule lets the
        # first given win and fails the run when the two disagree. Validated up
        # front so a bad leverage fails the run with the message rather than a
        # silent 10x.
        grid_run = spec.is_grid()  # BankrollGridError -> exit 2, before anything runs
        liquidation = spec.liquidation_settings()
        logger.info("Discovery liquidation settings: %s", liquidation.to_notes())

        # Run-level sizing settings (bead vibe-quant-2gyk1.2): parsed and checked
        # against the resolved leverage BEFORE the window plan, so a corrupt or
        # unmarginable cap fails the run with its message rather than a silent
        # default. An omitted flag means the default; the keys are stored only
        # when non-default (SIP -> notes.sizing).
        sizing = spec.sizing_settings()
        # A grid has no run-level leverage: every cell checks the cap at its own leverage.
        cells = spec.bankroll_cells(sizing) if grid_run else None
        if cells is None:
            sizing.check_leverage(liquidation.leverage)
        else:
            logger.info("Discovery bankroll grid: %d cells", len(cells))
        logger.info("Discovery sizing settings: %s", sizing)

        # The one LaunchSettings every non-grid engine is built from. Built right
        # after the run-level check above so that check's message stays the first
        # raise. A grid never builds it (R15b): its __post_init__ would check the
        # cap against the default 10x leverage the grid path skips; each cell
        # builds its own launch instead, already checked by build_cells.
        from vibe_quant.launch_settings import LaunchSettings

        run_launch: LaunchSettings | None = (
            None if cells is not None else spec.launch_settings(entry_order)
        )

        def launch_for(cell: BankrollCell | None) -> LaunchSettings:
            """The cell's LaunchSettings, or the run-level one outside a grid."""
            if cell is None:
                assert run_launch is not None  # no cells == non-grid == built above
                return run_launch
            return LaunchSettings(
                fee_schedule=fee_schedule,
                liquidation=cell.liquidation,
                sizing=cell.sizing(sizing),
                entry_order=entry_order,
            )

        # Train/test split + eval/cross/WFA windows: one shared check (the API
        # runs the same spec plan before creating a run row).
        plan = spec.plan()
        train_start, train_end = plan.train_start, plan.train_end
        holdout_start, holdout_end = plan.holdout_start, plan.holdout_end
        split_ratio = spec.train_test_split
        if split_ratio > 0:
            logger.info(
                "Train/test split: ratio=%.2f train=%s→%s holdout=%s→%s",
                split_ratio, train_start, train_end, holdout_start, holdout_end,
            )

        # Resolve timeframe-aware bootstrap_min_sharpe default.
        bootstrap_min_sharpe = spec.resolved_bootstrap_min_sharpe()
        if spec.bootstrap_min_sharpe is None:
            logger.info(
                "Bootstrap min t-stat default: %.1f (timeframe=%s)",
                bootstrap_min_sharpe, spec.timeframe,
            )

        cross_window_months = spec.cross_window_month_list()

        # Multi-window evaluation: sub-windows come from the plan
        eval_windows_count = spec.eval_window_count()
        eval_windows = plan.eval_windows
        if eval_windows is not None:
            logger.info(
                "Multi-window fitness: %d windows — %s",
                eval_windows_count,
                " | ".join(f"{s}→{e}" for s, e in eval_windows),
            )

        # Cumulative DSR gate N (bd vibe-quant-megq1): the gate adds the run_n_eff
        # of earlier non-mock discovery runs recorded under the effective rule,
        # with the same timeframe and an overlapping train window; earlier
        # (legacy) runs are tallied but not charged; mock runs keep run-only N.
        prior_trials_fn, prior_run_ids, prior_legacy = record.build_prior_trials_fn(
            store.state, run_id, spec.timeframe, mock=bool(args.mock)
        )

        config = spec.to_config(plan, symbols=symbols, prior_trials_fn=prior_trials_fn)

        # Log environment details for debugging and journal entries
        logger.info(
            "Environment: run_id=%d pid=%d compiler_version=%s",
            run_id,
            os.getpid(),
            record.compiler_version(),
        )
        logger.info(
            "Data range: %s to %s | symbols=%s | timeframe=%s",
            spec.start_date, spec.end_date, symbols, spec.timeframe,
        )
        _log_data_catalog_info(symbols, spec.timeframe)

        # Choose backtest function: real NT, or synthetic ONLY when explicitly
        # requested. Missing catalog data used to silently fall back to mock
        # metrics that then got persisted like real champions.
        use_mock = backtests.mock
        if use_mock:
            logger.warning("Using MOCK backtest — forced via --mock. Results are synthetic.")
        else:
            if not backtests.data_available(symbols):
                msg = (
                    f"No catalog bar data for symbols={symbols} — download data first "
                    "(Data Management) or pass --mock for a synthetic run"
                )
                raise RuntimeError(msg)
            logger.info("Using real NautilusTrader backtest for symbols=%s", symbols)
            if spec.evaluator == "fast" and (
                spec.timeframe != "1m" or (len(symbols) > 1 and spec.symbol_agg == "portfolio")
            ):
                logger.warning(
                    "--evaluator fast replays 1m single-symbol accounts only: with "
                    "timeframe=%s symbols=%d symbol_agg=%s every genome runs on NT "
                    "(use --symbol-agg worst for multi-symbol)",
                    spec.timeframe, len(symbols), spec.symbol_agg,
                )
        if split_ratio <= 0:
            logger.warning(
                "No holdout (--train-test-split 0): champions get NO out-of-sample gate"
            )

        # Train, holdout (when the train/test split is on) and the factory for
        # cross-window and/or WFA validation: one set per bankroll setup.
        holdout_range = (
            (holdout_start, holdout_end)
            if split_ratio > 0 and holdout_start and holdout_end
            else None
        )
        needs_factory = bool(cross_window_months) or spec.wfa_oos_step_days > 0

        def fns_for(launch: LaunchSettings) -> pooling._CellFns:
            return backtests.cell_fns(
                symbols=symbols,
                timeframe=spec.timeframe,
                symbol_agg=cast("SymbolAgg", spec.symbol_agg),
                launch=launch,
                train_start=train_start,
                train_end=train_end,
                eval_windows=eval_windows,
                min_trades=config.min_trades,
                holdout=holdout_range,
                needs_factory=needs_factory,
                evaluator=spec.evaluator,
            )

        # Load seed chromosomes from prior run if requested
        seed_chromosomes = None
        if spec.seed_from_run is not None:
            seed_chromosomes = load_seed_chromosomes(store.state, spec.seed_from_run)
            if seed_chromosomes:
                logger.info(
                    "Loaded %d seed chromosomes from run %d",
                    len(seed_chromosomes), spec.seed_from_run,
                )
            else:
                # An explicitly requested warm-start that can't be honored is
                # an operator error — fail loudly instead of silently running
                # a random population under a warm-start label.
                raise ValueError(
                    f"warm-start run {spec.seed_from_run} has no persisted "
                    "champions to seed from (guardrails may have rejected "
                    "all candidates); launch without seed_run_id instead"
                )

        num_seeds = max(1, spec.num_seeds)
        top_cells: list[BankrollCell] | None = None  # grid: the cell of each top strategy
        progress_file = str(log_dir() / f"discovery_{run_id}_progress.json")
        # A leftover file (another DB's run with the same id) would be served by
        # the progress API as ours until our first generation overwrites it.
        Path(progress_file).unlink(missing_ok=True)

        if cells is not None:
            # Bankroll grid: one pipeline per (cell, seed), pooled like a multi-seed run;
            # each cell gets its own functions so every gate runs at the champion's bankroll.
            seed = _resolve_seed(spec.seed)
            logger.info("Discovery seed: %d", seed)
            result, top_cells = pooling._run_bankroll_grid(
                cells,
                num_seeds,
                config,
                lambda cell: fns_for(launch_for(cell)),
                progress_file,
                seed_chromosomes=seed_chromosomes,
                base_seed=seed,
            )
        elif num_seeds == 1:
            # Single-seed run (default)
            fns = fns_for(launch_for(None))
            seed = _resolve_seed(spec.seed)
            logger.info("Discovery seed: %d", seed)
            random.seed(seed)
            pipeline = DiscoveryPipeline(
                config=config,
                backtest_fn=fns.backtest_fn,  # type: ignore[arg-type]
                progress_file=progress_file,
                holdout_backtest_fn=fns.holdout_backtest_fn,  # type: ignore[arg-type]
                backtest_fn_factory=fns.backtest_fn_factory,  # type: ignore[arg-type]
                seed_chromosomes=seed_chromosomes,
            )
            result = pipeline.run()
        else:
            # Multi-seed ensemble: run N times with different seeds
            fns = fns_for(launch_for(None))
            seed = _resolve_seed(spec.seed)
            logger.info("Discovery seed: %d", seed)
            result = pooling._run_multi_seed(
                num_seeds=num_seeds,
                config=config,
                backtest_fn=fns.backtest_fn,
                progress_file=progress_file,
                holdout_backtest_fn=fns.holdout_backtest_fn,
                backtest_fn_factory=fns.backtest_fn_factory,
                seed_chromosomes=seed_chromosomes,
                base_seed=seed,
            )

        run_notes = spec.to_notes(
            result,
            symbols=symbols,
            use_mock=use_mock,
            seed=seed,
            prior_run_ids=prior_run_ids,
            prior_legacy=prior_legacy,
            cells=cells,
        )

        if not result.top_strategies:
            # No champion passed every gate (guardrails / cross-window / WFA /
            # holdout -- all fail closed). Persist a structured summary with
            # the per-candidate rejection reasons and complete cleanly: with
            # honest gates this is a routine outcome, not a crash.
            execution_time = time.perf_counter() - started_at
            best_gen_fitness = (
                max((gr.best_fitness for gr in result.generations), default=0.0)
                if result.generations
                else 0.0
            )
            summary_notes = record.no_champion_notes(
                run_notes,
                best_raw_score=best_gen_fitness,
                guardrail_rejections=result.guardrail_rejections,
            )
            record.attach_ind_memo(summary_notes, use_mock=use_mock)
            store.save_result(
                run_id,
                {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": 0,
                    "skewness": 0.0,
                    "kurtosis": 3.0,  # normal-distribution default (min valid is 1)
                    "execution_time_seconds": execution_time,
                    "notes": dump_notes(summary_notes),
                },
            )
            store.mark_completed(run_id)
            message = (
                "Discovery complete: no viable strategies "
                f"(evaluated={result.total_candidates_evaluated}, "
                f"best_raw_score={best_gen_fitness:.4f})"
            )
            print(message)
            return RunOutcome(0, message, summary_notes)

        best_chrom, best_fitness = result.top_strategies[0]
        execution_time = time.perf_counter() - started_at

        # Save top strategies as DSL dicts in notes
        # bd vibe-quant-rewru: the GA's stored sharpe/trades are the multi-window
        # fitness aggregate (worst-of-windows / sum-of-windows) and, with a train/test
        # split, cover only the training slice. Promotion replays ONE continuous
        # screening backtest over the full discovery range, so those numbers are a
        # different statistic and replay_drift flags the gap by construction. Re-run
        # each champion ONCE over spec.start_date..spec.end_date (the exact range the
        # promote endpoint replays) and persist THAT as the headline so the drift
        # check compares like-for-like. Skip the extra backtest only when fitness
        # already IS that continuous full-range metric (single window, no split) or
        # in mock mode. Bounded cost: top-K champions, once each, at save time.
        worst_multi = spec.symbol_agg == "worst" and len(symbols) > 1
        # worst mode always needs the run: it carries trades_sum/symbol_metrics
        needs_full_range = not use_mock and (
            eval_windows_count >= 2 or split_ratio > 0 or worst_multi
        )
        # One full-range function per distinct bankroll setup: a grid champion is measured at ITS
        # cell, never at the outer run settings (a replay child inherits that cell, so the
        # headline must be the number the replay reproduces). Without a grid it is the single
        # run-level function, built lazily so a zero-champion run builds none.
        full_range_fns: dict[BankrollCell | None, NTBacktestFn] = {}

        def full_range_fn_for(cell: BankrollCell | None) -> NTBacktestFn:
            if cell not in full_range_fns:
                full_range_fns[cell] = backtests.nt_fn(
                    symbols=symbols,
                    timeframe=spec.timeframe,
                    start_date=spec.start_date,
                    end_date=spec.end_date,
                    symbol_agg=cast("SymbolAgg", spec.symbol_agg),
                    launch=launch_for(cell),
                )
            return full_range_fns[cell]

        if needs_full_range:
            logger.info(
                "Computing full-range headline for top-%d champions over %s..%s "
                "(multi-window/split fitness != continuous replay; bd-rewru)",
                len(result.top_strategies[:5]), spec.start_date, spec.end_date,
            )

        top_dsls = []
        for idx, (chrom, fitness) in enumerate(result.top_strategies[:5]):
            entry = record.champion_entry(
                chrom,
                fitness,
                timeframe=spec.timeframe,
                dsr=result.dsr_records.get(chrom.uid),
                buy_hold=result.buy_hold_records.get(chrom.uid),
            )
            cell = top_cells[idx] if top_cells is not None else None
            if cell is not None:
                entry["bankroll"] = cell.to_note()
            # Full-range headline (single continuous backtest) for like-for-like
            # promotion/replay_drift; sharpe/trades above stay as the multi-window
            # robustness aggregate. The function raises
            # DataUnavailableError on missing data (the run fails); other
            # backtest errors are swallowed into failure metrics. bd
            # vibe-quant-rewru.
            entry.update(
                full_range_headline(
                    full_range_fn_for(cell)(chrom) if needs_full_range else None,
                    fallback_sharpe=fitness.sharpe_ratio,
                    fallback_trades=fitness.total_trades,
                    fallback_max_dd=fitness.max_drawdown,
                    fallback_pf=fitness.profit_factor,
                    fallback_return=fitness.total_return,
                )
            )
            # Attach holdout metrics if available
            if idx < len(result.holdout_results):
                hr = result.holdout_results[idx]
                entry["holdout"] = record.holdout_entry(hr)
            # Attach cross-window results if available
            if idx < len(result.cross_window_results):
                entry["cross_window"] = record.cross_window_entry(result.cross_window_results[idx])
            # Attach WFA rolling results if available
            if idx < len(result.wfa_results):
                entry["wfa"] = record.wfa_entry(result.wfa_results[idx])
            top_dsls.append(entry)

        # Walk-forward efficiency (Pardo): out-of-sample return per day over
        # in-sample return per day, for the best champion. OOS = the holdout,
        # IS = one continuous backtest over the train range (the worst-of-N
        # fitness return is not a train-range return). Undefined without a
        # holdout or with a non-positive IS return.
        wfa_efficiency: float | None = None
        if result.holdout_results and result.train_dates and result.holdout_dates:
            if eval_windows_count >= 2 and not use_mock:
                best_cell = top_cells[0] if top_cells is not None else None
                train_metrics = backtests.nt_fn(
                    symbols=symbols,
                    timeframe=spec.timeframe,
                    start_date=result.train_dates[0],
                    end_date=result.train_dates[1],
                    symbol_agg=cast("SymbolAgg", spec.symbol_agg),
                    launch=launch_for(best_cell),
                )(best_chrom)
                train_return = float(train_metrics.get("total_return", 0.0))
            else:
                train_return = best_fitness.total_return
            wfa_efficiency = walk_forward_efficiency(
                oos_return=result.holdout_results[0].total_return,
                oos_days=compute_day_count(*result.holdout_dates),
                is_return=train_return,
                is_days=compute_day_count(*result.train_dates),
            )

        run_notes["top_strategies"] = top_dsls
        run_notes["guardrail_rejections"] = result.guardrail_rejections or None
        record.attach_ind_memo(run_notes, use_mock=use_mock)
        store.save_result(
            run_id,
            {
                "total_return": best_fitness.total_return,
                "sharpe_ratio": best_fitness.sharpe_ratio,
                "max_drawdown": best_fitness.max_drawdown,
                "profit_factor": best_fitness.profit_factor,
                "total_trades": best_fitness.total_trades,
                "skewness": best_fitness.skewness,
                "kurtosis": best_fitness.kurtosis,
                "execution_time_seconds": execution_time,
                "walk_forward_efficiency": wfa_efficiency,
                "notes": dump_notes(run_notes),
            },
        )
        store.mark_completed(run_id)

        message = (
            "Discovery complete: "
            f"top_score={best_fitness.adjusted_score:.4f}, "
            f"candidates={result.total_candidates_evaluated}, "
            f"mock={use_mock}"
        )
        print(message)
        return RunOutcome(0, message, run_notes)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            store.mark_failed(run_id, error)
        except Exception as write_exc:
            logger.error(
                "could not record run %s as failed (%s)",
                run_id,
                type(write_exc).__name__,
            )
        message = f"Discovery failed: {exc}"
        print(message)
        import traceback

        traceback.print_exc()
        # A bad bankroll grid is an operator error (argparse's own exit code), not a crash.
        return RunOutcome(2 if isinstance(exc, BankrollGridError) else 1, message, None)
