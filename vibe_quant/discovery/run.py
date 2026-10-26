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
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from vibe_quant.discovery.mock_backtest import mock_backtest as _mock_backtest
from vibe_quant.discovery.pipeline import DiscoveryPipeline as DiscoveryPipeline  # re-export
from vibe_quant.discovery.pooling import _CellFns

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.jobs.manager import BacktestJobManager
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

    mock: bool  # guards the full-range + train-range re-runs exactly as `use_mock` did

    def data_available(self, symbols: list[str]) -> bool: ...

    def cell_fns(
        self,
        *,
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
        fee_schedule: FeeSchedule,
        entry_order: str,
        liquidation: LiquidationSettings,
        sizing: SizingSettings | None,
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
        evaluator: str,
    ) -> _CellFns:
        return _make_cell_fns(
            use_mock=False,
            symbols=symbols,
            timeframe=timeframe,
            symbol_agg=symbol_agg,
            fee_schedule=fee_schedule,
            entry_order=entry_order,
            liquidation=liquidation,
            sizing=sizing,
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
        fee_schedule: FeeSchedule,
        entry_order: str,
        liquidation: LiquidationSettings,
        sizing: SizingSettings | None,
    ) -> NTBacktestFn:
        return _make_nt_backtest_fn(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            symbol_agg=symbol_agg,
            fee_schedule=fee_schedule,
            entry_order=entry_order,
            liquidation=liquidation,
            sizing=sizing,
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
        evaluator: str,
    ) -> _CellFns:
        return _make_cell_fns(
            use_mock=True,
            symbols=symbols,
            timeframe=timeframe,
            symbol_agg=symbol_agg,
            fee_schedule=fee_schedule,
            entry_order=entry_order,
            liquidation=liquidation,
            sizing=sizing,
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
        fee_schedule: FeeSchedule,
        entry_order: str,
        liquidation: LiquidationSettings,
        sizing: SizingSettings | None,
    ) -> NTBacktestFn:
        raise RuntimeError("mock adapter runs no NT backtest")


class RecordStore(Protocol):
    """The persistence of one discovery run: its row, its result and its job."""

    state: StateManager  # prior-trial scan + seed chromosomes read through it

    def get_run(self, run_id: int) -> dict[str, object] | None: ...

    def save_result(self, run_id: int, metrics: dict[str, object]) -> None: ...

    def mark_completed(self, run_id: int) -> None: ...

    def mark_failed(self, run_id: int, error: str) -> None: ...

    def mark_job_failed(self, run_id: int, error: str) -> None: ...


@dataclass(frozen=True)
class StateRecordStore:
    """The CLI store: StateManager for the row/result, job manager for the job."""

    state: StateManager
    job_manager: BacktestJobManager

    def get_run(self, run_id: int) -> dict[str, object] | None:
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
