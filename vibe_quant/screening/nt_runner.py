"""NautilusTrader backtest runner for screening mode.

Runs single-parameter-combination backtests using BacktestNode with
screening venue config (no latency, simple fill model). Designed to be
picklable for :class:`concurrent.futures.ProcessPoolExecutor`.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from vibe_quant.balance_settings import (
    DEFAULT_STARTING_BALANCE_USDT,
    validate_starting_balance,
)
from vibe_quant.errors import DataUnavailableError, LiquidationIntegrityError
from vibe_quant.fees import FeeSchedule
from vibe_quant.liquidation_settings import (
    DEFAULT_LEVERAGE,
    LiquidationMode,
    LiquidationSettings,
)
from vibe_quant.sizing_settings import SIZING_FIELDS, SizingSettings
from vibe_quant.validation.funding import FUNDING_PERIOD_HOURS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import numpy as np
    from nautilus_trader.config import BacktestRunConfig

    from vibe_quant.data.fill_ticks import FillTickSet
    from vibe_quant.dsl.pta_replay import PtaPrecomputed
    from vibe_quant.liquidation.overlay import OverlayResult
    from vibe_quant.metrics import DailyReturns, LedgerEvent
    from vibe_quant.screening.types import BacktestMetrics
    from vibe_quant.validation.extraction import OpenPositionMark
    from vibe_quant.validation.funding import FundingAccrual, FundingCalculator
    from vibe_quant.validation.venue import VenueConfig

logger = logging.getLogger(__name__)

# One millisecond in ns. A catalog execution bar's ``ts_init`` is its close less 1 ms, so a
# boundary resting fill (``(ts_event + 1 ms) % execution_bar_ns == 0``, R18) sits exactly on
# it and its 1m window is ``[ts + 1 ms - E, ts]`` (R19).
_ONE_MS = 1_000_000


# Catalog bars of the precompute seam as ARRAYS (never the Bar list: hundreds of MB on a long
# 1m window), keyed (catalog path, bar type, start, end); process-level LRU of 4 entries.
_PTA_ARRAYS_MAX = 4
_PTA_ARRAYS: OrderedDict[tuple[str, str, str, str], dict[str, np.ndarray]] = OrderedDict()


def _pta_bar_arrays(
    catalog_path: str, bar_type: str, start: str, end: str
) -> dict[str, np.ndarray]:
    """``ts_init`` (int64) and OHLCV (float64) arrays of the catalog query a BacktestDataConfig runs."""
    key = (catalog_path, bar_type, start, end)
    cached = _PTA_ARRAYS.get(key)
    if cached is not None:
        _PTA_ARRAYS.move_to_end(key)
        return cached
    import numpy as np
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    bars = ParquetDataCatalog(catalog_path).bars(bar_types=[bar_type], start=start, end=end)
    arrays: dict[str, np.ndarray] = {
        "ts_init": np.fromiter((b.ts_init for b in bars), dtype=np.int64, count=len(bars)),
    }
    for field in ("open", "high", "low", "close", "volume"):
        arrays[field] = np.fromiter(
            (float(getattr(b, field)) for b in bars), dtype=np.float64, count=len(bars)
        )
    _PTA_ARRAYS[key] = arrays
    while len(_PTA_ARRAYS) > _PTA_ARRAYS_MAX:
        _PTA_ARRAYS.popitem(last=False)
    return arrays


def _no_exit_timing() -> dict[str, Any]:
    """The R25b ``exit_timing`` notes of a mode-on run with nothing to refine.

    Returned for a 1m execution timeframe (R17) and for a coarser one with no boundary
    STOP_MARKET/LIMIT fill (R17b), so ``last_liquidation_notes['exit_timing']`` is present on
    EVERY mode-on screening run.
    """
    return {"refined": 0, "kept_bar_close": 0, "by_symbol": {}}


# One 8h funding period in ns: the smallest execution bar whose close can host a settlement
# that validation's 1m fill never charged (vibe-quant-6osqd). A bar of exactly this length is
# NOT refined (R9b: the guard is a strict ``>``).
_FUNDING_REFINE_MIN_BAR_NS = FUNDING_PERIOD_HOURS * 3_600 * 1_000_000_000


def _funding_close_ns(
    pos: Any,
    scan_fill_ts: Mapping[str, Mapping[int, int]] | None,
    execution_bar_ns: int,
) -> int:
    """Funding exit of a CLOSED position: its closing fill's refined ts, else ``ts_closed``.

    R9/R9b: only an execution bar strictly longer than 8h is refined; anything else (and an
    empty scan) returns the NT ``ts_closed`` without importing ``vibe_quant.liquidation``.
    R6: the CLOSING fill is ``events[len - 1]``, so a partial exit refined earlier does not
    move the funding exit.
    """
    nt = int(pos.ts_closed)
    if execution_bar_ns <= _FUNDING_REFINE_MIN_BAR_NS or not scan_fill_ts:
        return nt
    from vibe_quant.screening.liquidation_metrics import position_key

    refined = (scan_fill_ts.get(position_key(pos)) or {}).get(len(pos.events) - 1)
    return nt if refined is None else refined


def refuse_non_market_entry(entry_order: str) -> None:
    """Fail closed on a post-only entry before any engine is built.

    Maker entries are validated only in the validation runner (resting limits
    priced at release). Screening fills resting limits against the whole
    strategy bar, so parity with validation is unproven (bead vibe-quant-7eek0)
    and a post-only screening/discovery run must never silently fall back to
    market entries.
    """
    if entry_order != "market":
        msg = (
            f"entry_order={entry_order!r} is validation-only "
            "(screening maker-fill parity unproven, bead vibe-quant-7eek0)"
        )
        raise ValueError(msg)


# Process-level compile cache: DSL-content key -> (module_path, strategy_cls,
# config_cls, timeframes, strategy timeframe). Multi-window evals
# (eval_windows > 1) construct a fresh runner per window for the SAME
# chromosome; without this cache each window recompiled the identical DSL
# (~1s per compile).
_COMPILE_CACHE: dict[str, tuple[str, str, str, frozenset[str], str]] = {}

# Generated-config fields the runner sets itself (screening fill mode); a
# sweep/override may never set them.
_RUNNER_OWNED_FIELDS = frozenset(
    {"command_release", "command_release_bar_type", "margin_leverage"}
)


class UnknownStrategyParamError(ValueError):
    """A sweep/override key matches no field of the compiled strategy config.

    Raised instead of silently dropping the key: a silently ignored override
    makes every combination of a sweep identical (vibe-quant-e70tl.8).
    """


# Known spelling mismatches between sweep keys and compiled config fields
# (suffix -> replacement), applied only when the key itself is not a field.
_PARAM_KEY_ALIASES: tuple[tuple[str, str], ...] = (
    # take_profit[_long|_short].risk_reward_ratio -> ..._risk_reward
    ("_risk_reward_ratio", "_risk_reward"),
    # STOCH: NT/GA spell k/d as period_k/period_d; the DSL fields are period/d_period
    ("_period_k", "_period"),
    ("_k_period", "_period"),
    ("_period_d", "_d_period"),
)


class MissingBarDataError(DataUnavailableError):
    """A requested bar type has no catalog data in the run window.

    Re-raised by ``NTScreeningRunner.__call__`` (never masked as a -inf result).
    """


class FillTickCatalogModifiedError(RuntimeError):
    """A fill-tick dir changed during a run (NT writes epoch parquet on dispose).

    Tick dirs are immutable and shared by every run; a modified one would feed
    later runs different data. Re-raised by ``NTScreeningRunner.__call__``.
    """


class ScreeningFillMode(NamedTuple):
    """How screening releases a strategy's orders (see :func:`screening_fill_mode`)."""

    command_release: str
    execution_delay_probability: float
    tick_timeframe: str | None


def screening_fill_mode(timeframe: str, loaded_timeframes: Iterable[str]) -> ScreeningFillMode:
    """Screening fill mode, keyed on the finest LOADED timeframe.

    - 1m strategy: no outbox; orders defer one bar (delay 1.0) and fill at the
      next 1m close, like validation.
    - coarser strategy that loads 1m bars (indicator / additional timeframe):
      outbox released on its own 1m bars.
    - coarser: outbox released on fill ticks (1m closes at each boundary of the
      finest loaded timeframe), so fills land where validation's 1m detail
      fills them. Outbox modes need delay 0 (and no LatencyModel).
    """
    from vibe_quant.validation.extraction import finest_timeframe

    if timeframe == "1m":
        return ScreeningFillMode("", 1.0, None)
    finest = finest_timeframe({timeframe, *loaded_timeframes}) or timeframe
    if finest == "1m":
        return ScreeningFillMode("bar", 0.0, None)
    return ScreeningFillMode("trade_tick", 0.0, finest)


def loaded_timeframes(dsl_dict: dict[str, Any]) -> set[str]:
    """Every bar timeframe a DSL loads: primary + ``additional_timeframes`` + indicator ones.

    The fill mode's tick timeframe is the FINEST loaded timeframe
    (:func:`screening_fill_mode`), so ``NTScreeningRunner._compile`` and the
    parent-side resolvers (``screening.pipeline.resolve_parent_fill_ticks``)
    must derive this set the same way. If they diverge, the parent resolves
    ticks for a coarser timeframe and the runner rejects the set
    (``MissingBarDataError``) at run time (vibe-quant-yul7u.12).
    """
    loaded = {str(dsl_dict["timeframe"])}
    loaded.update(dsl_dict.get("additional_timeframes") or [])
    for ind in (dsl_dict.get("indicators") or {}).values():
        tf = ind.get("timeframe") if isinstance(ind, dict) else getattr(ind, "timeframe", None)
        if tf:
            loaded.add(str(tf))
    return loaded


_HAS_DATA_CACHE: set[tuple[str, str, int | None, int | None]] = set()


def _bar_type_has_data(catalog_path: str, bar_type: str, start_ns: int | None, end_ns: int | None) -> bool:
    key = (catalog_path, bar_type, start_ns, end_ns)
    if key in _HAS_DATA_CACHE:
        return True
    from nautilus_trader.model.data import Bar
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    for lo, hi in ParquetDataCatalog(catalog_path).get_intervals(Bar, bar_type):
        if (end_ns is None or lo <= end_ns) and (start_ns is None or hi >= start_ns):
            _HAS_DATA_CACHE.add(key)
            return True
    return False


def require_bars_in_window(catalog_path: str, bar_type: str, start: str, end: str) -> None:
    """Raise MissingBarDataError if the catalog has no bars for ``bar_type`` in [start, end].

    Cheap: reads only the parquet file-name intervals (no data load), cached
    in-process per (catalog, bar type, window) -- only positive results are
    cached, so ingesting data mid-process is picked up. File intervals only:
    internal gaps are NOT detected.
    """
    from vibe_quant.validation.extraction import date_to_ns

    if _bar_type_has_data(catalog_path, bar_type, date_to_ns(start), date_to_ns(end)):
        return
    raise MissingBarDataError(
        f"No catalog bars for {bar_type} in window {start}..{end} (catalog: {catalog_path}). "
        "Ingest/rebuild that timeframe (e.g. `python -m vibe_quant.data rebuild --from-archive`)."
    )


def resolve_strategy_params(
    params: dict[str, Any], config_fields: tuple[str, ...] | list[str]
) -> dict[str, Any]:
    """Map sweep/override keys onto compiled strategy-config field names.

    Dot notation (``ema_fast.period``) becomes underscore notation
    (``ema_fast_period``); known aliases (``take_profit.risk_reward_ratio``,
    STOCH ``period_k``/``period_d``) are mapped to their real fields.

    The runner-owned outbox fields (``command_release``,
    ``command_release_bar_type``) are never overridable.

    A run-level sizing setting (``max_position_pct`` / ``risk_per_trade``) is never a sweep
    key either: it is part of the run's ``parameters`` and the runner merges it into the
    config AFTER this call, so a sweep row carrying one is rejected rather than silently
    resolving to the generated config's own field (bead vibe-quant-2gyk1.2).

    Raises:
        UnknownStrategyParamError: if any key matches no config field, if a key is a
            run-level sizing setting, or two keys resolve to the same field.
    """
    for key in params:
        # Normalized like the config-field lookup below: ``max.position.pct`` is the same
        # target as ``max_position_pct`` and must not reach the field match.
        norm_key = key.replace(".", "_")
        if norm_key in SIZING_FIELDS:
            msg = (
                f"{norm_key} is a run-level sizing setting (parameters.{norm_key}), "
                "not a sweep parameter"
            )
            raise UnknownStrategyParamError(msg)
    fields = set(config_fields) - _RUNNER_OWNED_FIELDS
    if not fields:
        msg = "Compiled strategy config exposes no fields; cannot apply overrides"
        raise UnknownStrategyParamError(msg)
    resolved: dict[str, Any] = {}
    source_key: dict[str, str] = {}
    unknown: list[str] = []
    for key, value in params.items():
        cfg_key = key.replace(".", "_")
        if cfg_key not in fields:
            for suffix, repl in _PARAM_KEY_ALIASES:
                candidate = cfg_key[: -len(suffix)] + repl
                if cfg_key.endswith(suffix) and candidate in fields:
                    cfg_key = candidate
                    break
        if cfg_key not in fields:
            unknown.append(key)
            continue
        if cfg_key in resolved:
            msg = f"Override keys {source_key[cfg_key]!r} and {key!r} both map to {cfg_key!r}"
            raise UnknownStrategyParamError(msg)
        resolved[cfg_key] = value
        source_key[cfg_key] = key
    if unknown:
        from nautilus_trader.trading.config import StrategyConfig

        base_fields = set(getattr(StrategyConfig, "__struct_fields__", ()))
        valid = sorted(fields - base_fields - {"instrument_id"})
        msg = (
            f"Unknown strategy parameter(s) {sorted(unknown)}: no matching field in the "
            f"compiled strategy config. Valid keys (dot or underscore notation): {valid}"
        )
        raise UnknownStrategyParamError(msg)
    return resolved


class NTScreeningRunner:
    """Real NautilusTrader backtest runner for screening mode.

    Runs a single-parameter-combination backtest using BacktestNode with
    screening venue config (no latency, simple fill model). Designed to be
    picklable for ProcessPoolExecutor.

    This is the real screening runner that replaces the mock. It compiles
    the strategy DSL, creates a BacktestNode, runs it, and extracts metrics.
    """

    def __init__(
        self,
        dsl_dict: dict[str, Any],
        symbols: list[str],
        start_date: str,
        end_date: str,
        catalog_path: str | None = None,
        funding_archive_path: str | None = None,
        fill_ticks: FillTickSet | None = None,
        collect_daily_returns: bool = False,
        fee_schedule: FeeSchedule | None = None,
        entry_order: str = "market",
        liquidation_mode: LiquidationMode = "off",
        leverage: float = DEFAULT_LEVERAGE,
        starting_balance: int = DEFAULT_STARTING_BALANCE_USDT,
        sizing: SizingSettings | None = None,
        pta_precompute: bool = True,
    ) -> None:
        """Initialize NTScreeningRunner.

        Args:
            dsl_dict: Strategy DSL as dict (picklable, unlike StrategyDSL).
            symbols: List of symbols to screen (run in sorted order: the
                seeded screening FillModel RNG is shared across instruments,
                so the order used to change results).
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            catalog_path: Path to ParquetDataCatalog. Uses default if None.
            funding_archive_path: Raw-data archive holding funding rates.
                Uses the default archive if None. Funding series are cached
                per worker process (see validation.funding).
            fill_ticks: Fill-tick set resolved by the parent (trade_tick
                mode). None resolves it in-process when the mode needs it.
            collect_daily_returns: Opt-in (discovery DSR, bd vibe-quant-yul7u.24;
                market-valued series, bd vibe-quant-wcn55): keep the run's daily
                mark-to-market returns in ``last_mtm_daily_returns`` (never put
                into BacktestMetrics). The realized-balance twin was removed
                (bd vibe-quant-f9zzq).
            fee_schedule: Binance fee schedule (VIP tier + optional BNB
                discount) applied to the screening venue. None = the default
                (VIP0, no BNB), which keeps NT's MakerTakerFeeModel path and
                every pre-existing number bit-identical (bd vibe-quant-2gyk1.4).
            entry_order: Entry order type. Only ``"market"`` is supported:
                post-only maker entries are validation-only (screening
                maker-fill parity unproven, bead vibe-quant-7eek0) and raise
                before any engine is built.
            liquidation_mode: ``"off"`` (default: no liquidation code runs and every
                number is unchanged), ``"isolated"`` or ``"cross"``: scan the run's
                positions on real MARK bars and recompute every metric from the
                truncated ledgers (bead vibe-quant-2gyk1.1.6). Any failure on that path
                fails the run closed as ``LiquidationIntegrityError``, wrapping the
                original error (missing MARK bars, a fill outside the run, ...).
            leverage: The venue leverage, in EVERY mode (vibe-quant-6wq6w): the screening
                venue runs at this leverage and, when the mode is on, the liquidation prices
                assume it (recorded in ``last_liquidation_notes``). The default 10 keeps every
                pre-existing number byte-identical.
            starting_balance: The run's account size (USDT), already parsed from
                ``parameters.initial_balance`` (bead vibe-quant-zpbmo). It sets the venue's
                ``starting_balance_usdt`` and, when the mode is on, the liquidation wallet
                (R16). The default 1000 keeps every pre-existing number byte-identical (R9).
            sizing: The run's sizing settings (``parameters.max_position_pct`` /
                ``risk_per_trade``, bead vibe-quant-2gyk1.2), already parsed by the parent.
                They are merged into every strategy config AFTER the sweep-key resolution,
                so they can never meet the sweep rejection, and their cap is re-checked
                against ``leverage`` here as a defence. None = the defaults, which add no
                key and keep every pre-existing number byte-identical.
            pta_precompute: Replay each strategy's compute_fn indicators once over the
                run's catalog bars and install the columns on the generated strategies
                instead of recomputing them every bar (bead vibe-quant-surto). Every number
                is bit-identical; a bar the columns do not cover recomputes live. False,
                or env ``VIBE_QUANT_PTA_PRECOMPUTE=0``, keeps the per-bar path. What a run
                installed and fed is in ``last_pta_precompute``.

        Raises:
            ValueError: ``entry_order`` is not ``"market"``, or the liquidation settings
                are invalid (:class:`~vibe_quant.liquidation_settings.LiquidationConfigError`:
                an unknown mode, or a leverage that is not a whole number in [1, 125]).
            BalanceConfigError: ``starting_balance`` is not a whole number in
                [1, 1000000000] (checked HERE, before any engine exists).
            SizingConfigError: the cap of ``sizing`` exceeds this runner's ``leverage``
                (a cap above the venue leverage would be unmarginable, S13).
        """
        refuse_non_market_entry(entry_order)
        # The venue's account size and the liquidation wallet: validated in __init__ so a
        # bad value fails before any bar is read (never silently defaulted).
        self._starting_balance = validate_starting_balance(starting_balance)
        settings = LiquidationSettings(mode=liquidation_mode, leverage=leverage)
        self._liquidation_mode: LiquidationMode = settings.mode
        self._leverage = settings.leverage
        # The run's sizing settings, re-checked HERE against the venue leverage as a
        # defence: a caller that read them from a stored run may not have checked them.
        self._sizing = SizingSettings() if sizing is None else sizing
        self._sizing.check_leverage(self._leverage)
        # Liquidation summary of the last run (``OverlayResult.result.to_notes``) for the
        # discovery/result notes plumbing; None when the mode is off or the run failed.
        self.last_liquidation_notes: dict[str, Any] | None = None
        # R25b: the refined-exit counts of the last scan, attached to that summary as
        # ``exit_timing`` on every mode-on run (all zeros when nothing was refined).
        self._last_exit_timing_notes: dict[str, Any] = _no_exit_timing()
        self._entry_order = entry_order
        self._fee_schedule = FeeSchedule() if fee_schedule is None else fee_schedule
        self._collect_daily_returns = collect_daily_returns
        self.last_mtm_daily_returns: DailyReturns | None = None
        # Fee-drag inputs of the last run, for the discovery bt dict
        # (bd vibe-quant-2gyk1.4); not part of BacktestMetrics.
        self.last_starting_balance: float = 0.0
        self.last_days: int = 0
        self._pta_precompute = pta_precompute
        # One entry per (strategy, tf) the last run installed: instrument, tf, n, fed,
        # failed_at (filled after the run); [] when off. Not part of BacktestMetrics.
        self.last_pta_precompute: list[dict[str, object]] = []
        self._dsl_dict = dsl_dict
        self._symbols = sorted(symbols)
        self._fill_ticks = fill_ticks
        self._start_date = start_date
        self._end_date = end_date
        self._catalog_path = catalog_path
        self._funding_archive_path = funding_archive_path

        # Cached per-process compilation results (populated on first __call__)
        self._compiled = False
        self._module_path: str = ""
        self._strategy_cls_name: str = ""
        self._config_cls_name: str = ""

        # Fail fast on sweep keys that match no config field (they used to be
        # silently dropped, making every grid point identical).
        sweep = dsl_dict.get("sweep") or {}
        if sweep:
            self.validate_param_keys(list(sweep))

    def validate_param_keys(self, keys: list[str]) -> None:
        """Raise :class:`UnknownStrategyParamError` if any key is not overridable."""
        self._ensure_compiled()
        resolve_strategy_params(dict.fromkeys(keys), self._config_fields())

    def _config_fields(self) -> tuple[str, ...]:
        import sys

        config_cls = getattr(sys.modules[self._module_path], self._config_cls_name, None)
        return tuple(getattr(config_cls, "__struct_fields__", ()))

    def __call__(self, params: dict[str, float | int]) -> BacktestMetrics:
        """Run a single screening backtest with the given parameters.

        Args:
            params: Parameter combination to test.

        Returns:
            BacktestMetrics from the backtest run.
        """
        from vibe_quant.screening.types import BacktestMetrics

        start_time = time.time()
        # Per-run output: a failed run or an unavailable series must not leave
        # the previous run's series on a reused runner (bd vibe-quant-wcn55).
        self.last_mtm_daily_returns = None
        self.last_starting_balance = 0.0
        self.last_days = 0
        self.last_liquidation_notes = None
        self.last_pta_precompute = []
        try:
            return self._run_backtest(params, start_time)
        except (UnknownStrategyParamError, DataUnavailableError, FillTickCatalogModifiedError):
            # Configuration error (or a liquidation run that failed closed), not a
            # backtest failure: never mask it as a -inf result (vibe-quant-e70tl.8).
            raise
        except Exception as e:
            logger.warning(
                "NT screening backtest failed: params=%s strategy=%s error=%s",
                params,
                self._dsl_dict.get("name", "unknown"),
                e,
            )
            return BacktestMetrics(
                parameters=params,
                sharpe_ratio=float("-inf"),
                execution_time_seconds=time.time() - start_time,
            )

    def _ensure_compiled(self) -> None:
        """Parse and compile DSL once per worker process.

        Results are cached in instance attributes so subsequent calls
        to _run_backtest skip recompilation.
        """
        import sys

        # needs_context indicators (FUNDING, ...) read the run's archive in-process;
        # configure on EVERY call (another runner in this process may have changed it)
        # and fail before the engine if the archive does not cover the window.
        from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH as _default_catalog
        from vibe_quant.dsl import aux_data

        aux_data.configure(
            self._funding_archive_path,
            Path(self._catalog_path) if self._catalog_path else _default_catalog,
        )
        aux_data.preflight(
            [str(c.get("type", "")) for c in (self._dsl_dict.get("indicators") or {}).values()],
            self._symbols,
            self._start_date,
            self._end_date,
        )

        # A runner pickled into a fresh worker process keeps ``_compiled`` but
        # not the dynamically registered module -- recompile in that case.
        if self._compiled and self._module_path in sys.modules:
            return

        import json

        from vibe_quant.data.catalog import (
            DEFAULT_CATALOG_PATH,
        )

        # Catalog path (instruments already written during data ingest/rebuild;
        # writing here from parallel workers causes parquet corruption)
        self._resolved_catalog_path = (
            Path(self._catalog_path) if self._catalog_path else DEFAULT_CATALOG_PATH
        )
        cache_key = json.dumps(self._dsl_dict, sort_keys=True, default=str)
        cached = _COMPILE_CACHE.get(cache_key)
        if cached is not None and cached[0] in sys.modules:
            self._module_path, self._strategy_cls_name, self._config_cls_name, tfs, tf = cached
            self._all_timeframes: set[str] = set(tfs)
            self._timeframe = tf
            self._compiled = True
            return

        from vibe_quant.dsl.compiler import StrategyCompiler
        from vibe_quant.dsl.parser import validate_strategy_dict

        dsl = validate_strategy_dict(self._dsl_dict)
        compiler = StrategyCompiler()
        # Content-addressed module name (vibe-quant-e70tl.1): two DSLs sharing
        # a name (GA elite + mutant) must never resolve to each other's code.
        module = compiler.compile_to_module(dsl)  # registers in sys.modules

        class_name = "".join(word.capitalize() for word in dsl.name.split("_"))
        self._module_path = module.__name__
        self._strategy_cls_name = f"{class_name}Strategy"
        self._config_cls_name = f"{class_name}Config"

        # Cache parsed DSL fields needed for data config. The loaded set comes
        # from the SHARED helper the parents use too (yul7u.12): both must
        # agree on the finest loaded tf the fill mode keys on.
        self._timeframe = dsl.timeframe
        self._all_timeframes = loaded_timeframes(self._dsl_dict)

        _COMPILE_CACHE[cache_key] = (
            self._module_path,
            self._strategy_cls_name,
            self._config_cls_name,
            frozenset(self._all_timeframes),
            self._timeframe,
        )
        self._compiled = True

    def _resolve_fill_ticks(self, tick_timeframe: str, catalog_path: Path) -> FillTickSet:
        """The run's fill-tick set: the parent's, or resolved in-process.

        Raises:
            MissingBarDataError: no/partial 1m data, or the given set does not
                match the run (timeframe, symbols).
        """
        if self._fill_ticks is None:
            from vibe_quant.data.fill_ticks import resolve_fill_ticks

            ticks = resolve_fill_ticks(
                self._symbols, tick_timeframe, self._start_date, self._end_date, catalog_path
            )
        else:
            from vibe_quant.validation.extraction import date_to_ns

            ticks = self._fill_ticks
            missing = [s for s in self._symbols if s not in ticks.paths]
            if ticks.timeframe != tick_timeframe or missing:
                raise MissingBarDataError(
                    f"Fill-tick set ({ticks.timeframe}, symbols {sorted(ticks.paths)}) does not "
                    f"match the run ({tick_timeframe}, missing {missing})"
                )
            # A set checked for another window would silently score 0 trades
            run_start, run_end = date_to_ns(self._start_date), date_to_ns(self._end_date)
            if (
                ticks.start_ns is None
                or ticks.end_ns is None
                or run_start is None
                or run_end is None
                or run_start < ticks.start_ns
                or run_end > ticks.end_ns
            ):
                raise MissingBarDataError(
                    f"Fill-tick set window [{ticks.start_ns}, {ticks.end_ns}] does not cover "
                    f"the run window {self._start_date}..{self._end_date}"
                )
        for symbol in self._symbols:
            gaps = ticks.missing_boundaries.get(symbol, 0)
            if gaps:
                logger.warning(
                    "Fill ticks %s %s: %d boundar%s without 1m data in %s..%s -- orders "
                    "queued there wait for the next tick",
                    symbol,
                    tick_timeframe,
                    gaps,
                    "y" if gaps == 1 else "ies",
                    self._start_date,
                    self._end_date,
                )
        return ticks

    def _build_run_config(
        self, params: dict[str, float | int]
    ) -> tuple[BacktestRunConfig | None, list[str]]:
        """Run config for ``params`` plus the fill-tick dirs it reads.

        Returns (None, []) when no bar timeframe is loadable. Missing bars or
        fill ticks raise :class:`MissingBarDataError` before any NT config is
        built.
        """
        import os

        from nautilus_trader.config import (
            BacktestDataConfig,
            BacktestEngineConfig,
            BacktestRunConfig,
            ImportableStrategyConfig,
            LoggingConfig,
        )
        from nautilus_trader.model.data import Bar, TradeTick

        from vibe_quant.data.catalog import INTERVAL_TO_AGGREGATION
        from vibe_quant.validation.venue import create_backtest_venue_config

        # Compile DSL once per worker process
        self._ensure_compiled()

        module_path = self._module_path
        strategy_cls_name = self._strategy_cls_name
        config_cls_name = self._config_cls_name
        catalog_path = self._resolved_catalog_path
        catalog_str = str(catalog_path.resolve())

        # Map sweep keys (dot notation, known aliases) onto the generated
        # StrategyConfig's fields; an unknown key raises instead of being
        # silently dropped (NT 1.226+ would also reject it at decode time).
        strategy_params = resolve_strategy_params(params, self._config_fields())
        # The run's sizing settings are a RUN-level override (parameters.max_position_pct /
        # risk_per_trade, bead vibe-quant-2gyk1.2), merged in only now so a stored sweep
        # combination can never smuggle them past the sweep-key rejection above. Only the
        # keys this run moved off the default are added, so an all-default run builds the
        # exact config dict it built before.
        strategy_params.update(self._sizing.strategy_params(self._leverage))

        # Data configs
        data_configs: list[BacktestDataConfig] = []
        for symbol in self._symbols:
            instrument_id = f"{symbol}-PERP.BINANCE"
            for tf in sorted(self._all_timeframes):
                if tf not in INTERVAL_TO_AGGREGATION:
                    continue
                step, agg = INTERVAL_TO_AGGREGATION[tf]
                bar_type_str = f"{instrument_id}-{step}-{agg.name}-LAST-EXTERNAL"
                require_bars_in_window(catalog_str, bar_type_str, self._start_date, self._end_date)
                # NT 1.226+: pass data_cls as the CLASS, not the import
                # string — BacktestDataConfig.query compares `data_cls is Bar`
                # so a string silently disables bar-type narrowing and loads
                # every bar timeframe in the catalog (~300x the needed data).
                data_configs.append(
                    BacktestDataConfig(
                        catalog_path=catalog_str,
                        data_cls=Bar,
                        bar_types=[bar_type_str],
                        start_time=self._start_date,
                        end_time=self._end_date,
                    )
                )

        if not data_configs:
            return None, []

        # Fill mode (vibe-quant-yul7u.10). NT (bar_execution, no latency) fills
        # a market order from on_bar(t) at close[t] -- the signal bar's own
        # close, a same-bar look-ahead. 1m strategies defer one bar (fill at
        # the next 1m close); coarser ones queue every venue command in the
        # strategy's per-instrument outbox and release it on the next own 1m
        # bar / fill tick, so fills land at validation's price and time.
        mode = screening_fill_mode(self._timeframe, self._all_timeframes)
        tick_dirs: list[str] = []
        if mode.tick_timeframe is not None:
            ticks = self._resolve_fill_ticks(mode.tick_timeframe, catalog_path)
            for symbol in self._symbols:
                tick_dirs.append(ticks.paths[symbol])
                # TradeTick CLASS, never the import string (see Bar above)
                data_configs.append(
                    BacktestDataConfig(
                        catalog_path=ticks.paths[symbol],
                        data_cls=TradeTick,
                        instrument_id=f"{symbol}-PERP.BINANCE",
                        start_time=self._start_date,
                        end_time=self._end_date,
                    )
                )

        # Strategy configs (with parameter overrides)
        strategy_configs: list[ImportableStrategyConfig] = []
        for symbol in self._symbols:
            instrument_id = f"{symbol}-PERP.BINANCE"
            config_dict: dict[str, Any] = {"instrument_id": instrument_id, **strategy_params}
            config_dict["execution_delay_probability"] = mode.execution_delay_probability
            if mode.command_release:
                config_dict["command_release"] = mode.command_release
            if mode.command_release == "bar":
                config_dict["command_release_bar_type"] = (
                    f"{instrument_id}-1-MINUTE-LAST-EXTERNAL"
                )
            strategy_configs.append(
                ImportableStrategyConfig(
                    strategy_path=f"{module_path}:{strategy_cls_name}",
                    config_path=f"{module_path}:{config_cls_name}",
                    config=config_dict,
                )
            )

        # Screening venue config: no latency (the outbox needs none), simple fills
        bt_venue_config = create_backtest_venue_config(self._venue_config())

        # Suppress NT's verbose INFO logging in discovery/screening mode
        # (every order/fill/position logs at INFO, generating 100s of MB)
        # Screening runs in parallel workers, so NT engine output defaults to
        # WARNING to keep sweep logs readable. Override per run via
        # VIBE_QUANT_NT_LOG_LEVEL_SCREENING (TRACE/DEBUG/INFO/WARNING/ERROR).
        engine_config = BacktestEngineConfig(
            strategies=strategy_configs,
            run_analysis=True,
            logging=LoggingConfig(
                log_level=os.environ.get("VIBE_QUANT_NT_LOG_LEVEL_SCREENING", "WARNING")
            ),
        )

        # raise_exception: node.build() otherwise swallows engine-build errors
        # (get_engine() -> None, "engine not found").
        run_config = BacktestRunConfig(
            engine=engine_config,
            venues=[bt_venue_config],
            data=data_configs,
            start=self._start_date,
            end=self._end_date,
            dispose_on_completion=False,
            raise_exception=True,
        )
        return run_config, tick_dirs

    def _venue_config(self) -> VenueConfig:
        from dataclasses import replace
        from decimal import Decimal

        from vibe_quant.validation.venue import create_venue_config_for_screening

        config = replace(
            create_venue_config_for_screening(starting_balance_usdt=self._starting_balance),
            fee_schedule=self._fee_schedule,
        )
        # The venue runs at the run's leverage in EVERY mode (vibe-quant-6wq6w): today it is
        # only moved when liquidation is on. The default 10 is the venue's own default, so a
        # default run stays byte-identical.
        return replace(config, default_leverage=Decimal(str(self._leverage)))

    def _install_pta_precompute(
        self, node: Any
    ) -> list[tuple[str, str, PtaPrecomputed]]:
        """Install precomputed compute_fn columns on the built strategies.

        One ``{tf: PtaPrecomputed}`` per strategy with a ``_PTA_META`` (S1 emits none for
        needs_context strategies), from the same catalog query the engine's data config
        runs. An error on a (strategy, tf) leaves that tf on the live per-bar path (the
        reference), so the broad except is safe here. Returns (instrument, tf, installed).
        """
        if not self._pta_precompute or os.environ.get("VIBE_QUANT_PTA_PRECOMPUTE") == "0":
            return []
        import pandas as pd

        from vibe_quant.data.catalog import INTERVAL_TO_AGGREGATION
        from vibe_quant.dsl.pta_replay import PtaPrecomputed, strategy_pta_columns

        catalog_str = str(self._resolved_catalog_path.resolve())
        installed: list[tuple[str, str, PtaPrecomputed]] = []
        for engine in node.get_engines():
            for strategy in engine.trader.strategies():
                meta = getattr(type(strategy), "_PTA_META", None)
                if not meta:
                    continue
                instrument_id = str(strategy.config.instrument_id)
                pre: dict[str, PtaPrecomputed] = {}
                for tf in sorted({tf for _, tf in meta.values()}):
                    try:
                        step, agg = INTERVAL_TO_AGGREGATION[tf]
                        bar_type = f"{instrument_id}-{step}-{agg.name}-LAST-EXTERNAL"
                        arrays = _pta_bar_arrays(
                            catalog_str, bar_type, self._start_date, self._end_date
                        )
                        frame = pd.DataFrame(
                            {k: arrays[k] for k in ("open", "high", "low", "close", "volume")}
                        )
                        columns = strategy_pta_columns(
                            strategy._pta_params, meta, strategy._pta_buffer_cap, tf, frame
                        )
                        precomputed = PtaPrecomputed(
                            arrays["ts_init"].tolist(),
                            arrays["close"].tolist(),
                            [(key, column.tolist()) for key, column in columns],
                        )
                    except Exception as exc:
                        logger.warning(
                            "PTA precompute off for %s %s: %s: %s",
                            instrument_id,
                            tf,
                            type(exc).__name__,
                            exc,
                        )
                    else:
                        pre[tf] = precomputed
                        installed.append((instrument_id, tf, precomputed))
                if pre:
                    strategy._pta_pre = pre
        return installed

    def _report_pta_precompute(
        self, installed: list[tuple[str, str, PtaPrecomputed]]
    ) -> None:
        """Record what the run fed per installed (strategy, tf); warn where speed was lost.

        A fallback is still exact (the feed guard recomputes live); the warning is the
        only signal that the precompute stopped covering the run.
        """
        self.last_pta_precompute = []
        for instrument_id, tf, pre in installed:
            self.last_pta_precompute.append(
                {
                    "instrument": instrument_id,
                    "tf": tf,
                    "n": pre.n,
                    "fed": pre.fed,
                    "failed_at": pre.failed_at,
                }
            )
            if pre.failed_at is not None or pre.fed != pre.n:
                logger.warning(
                    "PTA precompute fell back for %s %s at bar %d of %d",
                    instrument_id,
                    tf,
                    pre.fed if pre.failed_at is None else pre.failed_at,
                    pre.n,
                )

    def _run_backtest(self, params: dict[str, float | int], start_time: float) -> BacktestMetrics:
        """Execute the NautilusTrader backtest."""
        import contextlib

        from nautilus_trader.backtest.node import BacktestNode
        from nautilus_trader.core.nautilus_pyo3 import (
            ProfitFactor,
            SharpeRatio,
            SortinoRatio,
            WinRate,
        )

        from vibe_quant.data.catalog import cleanup_epoch_parquet
        from vibe_quant.data.fill_ticks import fill_tick_dir_signature
        from vibe_quant.nt_compat import retain_log_guard
        from vibe_quant.screening.types import BacktestMetrics

        bt_run_config, tick_dirs = self._build_run_config(params)
        if bt_run_config is None:
            return BacktestMetrics(
                parameters=params,
                sharpe_ratio=float("-inf"),
                execution_time_seconds=time.time() - start_time,
            )
        catalog_path = self._resolved_catalog_path
        starting_balance = float(self._venue_config().starting_balance_usdt)
        tick_signatures = {d: fill_tick_dir_signature(d) for d in tick_dirs}

        # Run the backtest
        node = BacktestNode(configs=[bt_run_config])
        try:
            node.build()

            # Register statistics
            # MaxDrawdown excluded: lacks calculate_from_realized_pnls in NT 1.222
            stats = [SharpeRatio(), SortinoRatio(), WinRate(), ProfitFactor()]
            for engine in node.get_engines():
                analyzer = engine.kernel.portfolio.analyzer
                for stat in stats:
                    analyzer.register_statistic(stat)

            pta_installed = self._install_pta_precompute(node)
            node.run()
            self._report_pta_precompute(pta_installed)

            engine = node.get_engine(bt_run_config.id)
            if engine is None:
                metrics = BacktestMetrics(
                    parameters=params,
                    sharpe_ratio=float("-inf"),
                    execution_time_seconds=time.time() - start_time,
                )
            else:
                bt_result = engine.get_result()
                if self._sizing.max_margin_pct is not None:
                    # Screening records no per-row column for it: the count is a log line.
                    refused = sum(
                        int(getattr(strategy, "_margin_refusals", 0))
                        for strategy in engine.trader.strategies()
                    )
                    logger.info("Screening: margin limit refused %d entries", refused)
                metrics = self._extract_metrics(
                    params,
                    bt_result,
                    engine,
                    start_time,
                    starting_balance=starting_balance,
                )
                # One line per grid point so sweep logs are analyzable
                logger.info(
                    "Screening backtest: params=%s sharpe=%.2f trades=%d return=%.2f%% "
                    "maxDD=%.2f%% (%d orders, %d events, %.1fs)",
                    params or "{}",
                    metrics.sharpe_ratio,
                    metrics.total_trades,
                    metrics.total_return * 100,
                    metrics.max_drawdown * 100,
                    bt_result.total_orders,
                    bt_result.total_events,
                    metrics.execution_time_seconds,
                )
        finally:
            # Reset engines before dispose to avoid
            # InvalidStateTrigger('RUNNING -> DISPOSE')
            for eng in node.get_engines():
                retain_log_guard(eng)
                with contextlib.suppress(Exception):
                    eng.reset()
            node.dispose()  # type: ignore[no-untyped-call]

            # NT writes corrupt epoch-timestamp parquet on dispose(), into
            # every catalog the run read -- the fill-tick catalogs included
            cleanup_epoch_parquet(catalog_path)
            for tick_dir in tick_dirs:
                cleanup_epoch_parquet(Path(tick_dir))

        changed = [d for d, sig in tick_signatures.items() if fill_tick_dir_signature(d) != sig]
        if changed:
            logger.error("Fill-tick dir(s) modified by the engine run: %s", changed)
            raise FillTickCatalogModifiedError(
                f"Fill-tick dir(s) modified by the engine run: {changed}. Tick sets are "
                "immutable; delete the dir(s) so they are rebuilt."
            )
        return metrics

    def _extract_metrics(
        self,
        params: dict[str, float | int],
        bt_result: Any,
        engine: Any,
        start_time: float,
        starting_balance: float = 1000.0,
    ) -> BacktestMetrics:
        """Extract BacktestMetrics from NT BacktestResult.

        Args:
            starting_balance: Venue starting balance (quote currency). Must
                match the venue config: it scales the funding charge in
                total_return, the daily-balance Sharpe and the drawdown.
        """
        from vibe_quant.metrics import (
            closed_trade_drawdown,
            gross_pnl,
            mark_to_market_daily_returns,
            mark_to_market_max_drawdown,
            profit_factor,
            trade_win_rate,
        )
        from vibe_quant.screening.types import BacktestMetrics
        from vibe_quant.utils import compute_day_count
        from vibe_quant.validation.extraction import (
            accrue_position_funding,
            all_positions,
            daily_sharpe_sortino,
            date_to_ns,
            finest_timeframe,
            mark_open_positions,
            mark_to_market_paths,
        )
        from vibe_quant.validation.funding import FundingCalculator

        metrics = BacktestMetrics(
            parameters=params,
            execution_time_seconds=time.time() - start_time,
        )

        if bt_result is None:
            logger.warning("bt_result is None for params %s, returning default metrics", params)
            return metrics

        metrics.total_trades = bt_result.total_positions

        # Extract from PnL stats first (more comprehensive, includes total return)
        _known_pnl_keys = {
            "pnl (total)",
            "pnl% (total)",
            "sharpe",
            "sortino",
            "max drawdown",
            "win rate",
            "profit factor",
        }
        # Track which fields were set by PnL stats so returns-stats fallback
        # doesn't overwrite legitimate zero values (bug: == 0.0 sentinel)
        _populated: set[str] = set()
        stats_pnls = bt_result.stats_pnls or {}
        if not stats_pnls:
            logger.debug("No stats_pnls in bt_result for params %s", params)
        for _currency, pnl_stats in stats_pnls.items():
            for key, value in pnl_stats.items():
                if value is None:
                    continue
                key_lower = key.lower()
                try:
                    fval = float(value)
                except (ValueError, TypeError):
                    logger.warning("Could not convert PnL stat %s=%r to float", key, value)
                    continue
                if key_lower == "pnl% (total)":
                    # NT reports as percentage; store as fraction
                    metrics.total_return = fval / 100.0
                    _populated.add("total_return")
                elif "sharpe" in key_lower:
                    metrics.sharpe_ratio = fval
                    _populated.add("sharpe_ratio")
                elif "sortino" in key_lower:
                    metrics.sortino_ratio = fval
                    _populated.add("sortino_ratio")
                elif key_lower == "max drawdown":
                    metrics.max_drawdown = abs(fval)
                    _populated.add("max_drawdown")
                elif key_lower == "win rate":
                    metrics.win_rate = fval
                    _populated.add("win_rate")
                elif not any(k in key_lower for k in _known_pnl_keys):
                    logger.debug("Unmatched PnL stats key: %s = %s", key, value)

        # Fill from returns stats only if not already set by PnL stats
        _known_returns_keys = {"sharpe", "sortino", "max drawdown", "win rate", "profit factor"}
        stats_returns = bt_result.stats_returns or {}
        for key, value in stats_returns.items():
            if value is None:
                continue
            key_lower = key.lower()
            try:
                fval = float(value)
            except (ValueError, TypeError):
                logger.warning("Could not convert returns stat %s=%r to float", key, value)
                continue
            if "sharpe" in key_lower and "sharpe_ratio" not in _populated:
                metrics.sharpe_ratio = fval
            elif "sortino" in key_lower and "sortino_ratio" not in _populated:
                metrics.sortino_ratio = fval
            elif "max drawdown" in key_lower and "max_drawdown" not in _populated:
                metrics.max_drawdown = abs(fval)
            elif key_lower == "win rate" and "win_rate" not in _populated:
                metrics.win_rate = fval
            elif not any(k in key_lower for k in _known_returns_keys):
                logger.debug("Unmatched returns stats key: %s = %s", key, value)

        # Warn if extraction produced no meaningful metrics
        if (
            metrics.sharpe_ratio == 0.0
            and metrics.total_return == 0.0
            and metrics.total_trades == 0
        ):
            logger.warning(
                "Metric extraction yielded no results for params %s "
                "(stats_pnls keys: %s, stats_returns keys: %s)",
                params,
                list(stats_pnls.keys()) if stats_pnls else "empty",
                list(stats_returns.keys()) if stats_returns else "empty",
            )

        # Fees, funding and net trade PnLs from closed positions, plus
        # positions still open at the end of the run marked to market at the
        # last execution-bar close with an estimated taker exit fee -- the
        # outbox never releases on_stop closes (same as validation).
        # NT netting mode removes closed positions from the main index;
        # combine positions() + position_snapshots() to capture all.
        start_ns = date_to_ns(self._start_date)
        end_ns = date_to_ns(self._end_date)
        execution_tf = finest_timeframe(getattr(self, "_all_timeframes", ()))
        catalog_path = getattr(self, "_resolved_catalog_path", None)
        trade_pnls: list[float] = []
        closed_net: list[tuple[int, float]] = []
        cash_events: list[tuple[int, float]] = []
        funding_cash: dict[str, list[tuple[int, float]]] = {}
        total_funding = 0.0
        funding_fallbacks = 0
        try:
            everything = all_positions(engine)
            closed = [p for p in everything if p.is_closed]
            still_open = [
                p for p in everything if not p.is_closed and getattr(p, "is_open", False)
            ]
        except Exception:
            logger.warning("Could not read positions from engine cache", exc_info=True)
            closed, still_open = [], []
        funding_calc = FundingCalculator(self._funding_archive_path)
        liquidation_on = self._liquidation_mode != "off"
        overlay: OverlayResult | None = None
        scan_funding: dict[str, FundingAccrual] = {}
        # 6osqd D1: the funding exit of a position whose closing fill is a refined boundary
        # resting exit is that refined 1m instant, for an execution bar strictly longer than
        # 8h (mode off AND on). scan_fill_ts is mode on: the scan's own dict, mode off: the
        # one _funding_scan_ts built; None whenever no bar > 8h runs.
        scan_fill_ts: dict[str, dict[int, int]] | None = None
        from vibe_quant.validation.extraction import _TIMEFRAME_SECONDS

        execution_bar_ns = (
            _TIMEFRAME_SECONDS[execution_tf] * 1_000_000_000
            if execution_tf in _TIMEFRAME_SECONDS
            else 0
        )
        # ONE guard on the mode-on liquidation path (bead vibe-quant-2gyk1.1.8): any failure
        # while the mode is on -- the MARK scan, its survivor marks, or the truncated-ledger
        # recompute -- fails the run closed as LiquidationIntegrityError (a
        # DataUnavailableError __call__ re-raises) instead of scoring -inf. Mode off
        # re-raises unchanged, so every non-liquidation path is byte-identical.
        try:
            if liquidation_on:
                # ONE scan runs BEFORE any run-end marking (open positions funded to the
                # horizon mode off would mark them at): a liquidated or dropped position is
                # valued by the overlay, never by LAST bars (a missing LAST bar cannot make
                # it vanish). Only survivors are marked; one without a LAST bar in the final
                # bar fails closed.
                overlay, open_marks, scan_funding, scan_fill_ts = self._scan_with_survivor_marks(
                    engine,
                    closed,
                    still_open,
                    funding_calc,
                    starting_balance=starting_balance,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    execution_tf=execution_tf,
                    catalog_path=catalog_path,
                )
            else:
                # R10/R10c: mode off refines the funding exit BEFORE the run-end marking, so a
                # 1d run whose closing fill is a boundary resting exit fails closed on missing
                # 1m LAST instead of scoring -inf (a bare ValueError would be masked).
                if execution_tf is not None:
                    scan_fill_ts = self._funding_scan_ts(
                        engine, [*closed, *still_open], execution_tf, catalog_path
                    )
                open_marks = mark_open_positions(
                    engine,
                    still_open,
                    execution_timeframe=execution_tf,
                    catalog_path=catalog_path,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    taker_fee=self._fee_schedule.taker_rate,
                )
            trade_inputs: list[tuple[Any, OpenPositionMark | None]] = [
                (p, None) for p in closed
            ]
            trade_inputs.extend((mark.position, mark) for mark in open_marks)
            total_fees = 0.0
            unrealized_at_end = 0.0
            for pos, mark in trade_inputs:
                instrument_cash = funding_cash.setdefault(str(pos.instrument_id), [])
                total_fees += sum(abs(float(c)) for c in pos.commissions())
                # NT realized_pnl is net of commissions
                realized = float(pos.realized_pnl)
                if mark is None:
                    exit_ns = int(pos.ts_closed)
                    # R8: a CLOSED position only; an open mark funds to its mark (R15/R16).
                    funding_exit_ns = _funding_close_ns(pos, scan_fill_ts, execution_bar_ns)
                else:
                    exit_ns = mark.ts
                    funding_exit_ns = mark.ts
                    total_fees += mark.exit_fee
                    realized += mark.unrealized - mark.exit_fee
                    unrealized_at_end += mark.unrealized - mark.exit_fee
                    instrument_cash.append((mark.ts, -mark.exit_fee))
                # Funding is modeled post-hoc (NT's engine applies none) with
                # the same calculator validation uses (bd vibe-quant-e70tl.20).
                # R15: only the funding exit moves; realized PnL, the ledger and the cash
                # flows keep the NT fill timestamp (exit_ns).
                accrual = accrue_position_funding(funding_calc, pos, exit_ns=funding_exit_ns)
                total_funding += accrual.total
                funding_fallbacks += accrual.fallback_settlements
                trade_pnls.append(realized - accrual.total)
                closed_net.append((exit_ns, realized - accrual.total))
                cash_events.append((exit_ns, realized))
                cash_events.extend((ts, -amount) for ts, amount in accrual.payments)
                instrument_cash.extend((ts, -amount) for ts, amount in accrual.payments)
            # Opt-in liquidation: unless the scan hit nothing, every trade-level number
            # below is rebuilt from the truncated ledgers (identity overlay: the NT path
            # above stands).
            liquidated = None
            liquidation_ledgers: dict[str, list[LedgerEvent]] | None = None
            liquidation_floor_ns: int | None = None
            if overlay is not None:
                from vibe_quant.screening.liquidation_metrics import (
                    liquidation_trades,
                    position_key,
                )

                if not overlay.is_identity:
                    # every raw position the venue touched, whether or not NT or a mark can
                    # price it; survivors come with their run-end marks
                    touched = [
                        (p, None)
                        for p in still_open
                        if position_key(p) in overlay.hits_by_key
                        or position_key(p) in overlay.dropped_keys
                    ]
                    liquidated = liquidation_trades(
                        overlay, [*trade_inputs, *touched], scan_funding, funding_calc
                    )
                    liquidation_ledgers = overlay.ledgers
                    liquidation_floor_ns = overlay.result.bust_ts
                    # total_fees includes the clearance fees (a real cost);
                    # liquidation_forfeited is the same clearance total, reported
                    # separately: never add the two
                    total_fees = liquidated.total_fees
                    total_funding = liquidated.total_funding
                    funding_fallbacks = liquidated.funding_fallbacks
                    trade_pnls = liquidated.trade_pnls
                    closed_net = liquidated.closed_net
                    cash_events = liquidated.cash_events
                    metrics.total_trades = len(trade_pnls)
                    metrics.liquidation_count = len(overlay.hits_by_key)
                    metrics.liquidation_forfeited = float(
                        sum(h.clearance_fee for h in overlay.hits_by_key.values())
                    )
                self.last_liquidation_notes = overlay.result.to_notes(metrics.total_trades)
                # R25b: the refined-exit counts of this scan, on EVERY mode-on run
                self.last_liquidation_notes["exit_timing"] = self._last_exit_timing_notes
                # D1b/R25b: metrics travel back from workers; runner attributes do not.
                metrics.exit_timing = dict(self._last_exit_timing_notes)
        except Exception as exc:
            if liquidation_on:
                raise LiquidationIntegrityError.wrap(exc, self._liquidation_mode) from exc
            raise
        metrics.total_fees = total_fees
        metrics.total_funding = total_funding
        metrics.funding_fallback_settlements = funding_fallbacks
        if funding_fallbacks:
            logger.warning(
                "Screening funding for params %s: %d settlement(s) charged at the "
                "fallback rate (no archived rate)",
                params or "{}",
                funding_fallbacks,
            )
        # NT's total return covers realized PnL only: add the open marks
        extra_costs = total_funding - unrealized_at_end
        if liquidated is not None:
            # final equity of the truncated ledgers, not NT's untruncated PnL%
            if starting_balance > 0:
                metrics.total_return = sum(trade_pnls) / starting_balance
        elif extra_costs != 0.0 and starting_balance > 0:
            metrics.total_return -= extra_costs / starting_balance

        # Cost-free PnL = net + reported fees + reported slippage + funding
        # (bd vibe-quant-2gyk1.4), plus the inputs discovery needs to report
        # fee drag and the annualized fee rate. Screening reports no separate
        # slippage: tick slippage is modelled in the fill prices and stays
        # embedded in net, so gross excludes it and screening fee_drag is
        # slightly conservative (overstated). Score and every pre-existing
        # metric are untouched.
        metrics.gross_pnl = gross_pnl(
            metrics.total_return * starting_balance, total_fees, 0.0, total_funding
        )
        self.last_starting_balance = starting_balance
        days = compute_day_count(self._start_date, self._end_date)
        self.last_days = int(days) if days else 0

        # Trade-based profit factor and win rate on net PnL (shared definitions
        # with validation). NT's realized-PnL PF is unimplemented, and its
        # returns-based PF is a daily-return statistic (bd vibe-quant-e70tl.7).
        # NT's WinRate is realized-before-funding and counts an open position as
        # a loser at -entry fee, so it is overwritten from the same per-trade net
        # list, on the mode-off, identity-overlay and liquidated paths alike
        # (bd vibe-quant-neuvu.4).
        metrics.profit_factor = profit_factor(trade_pnls)
        metrics.win_rate = trade_win_rate(trade_pnls)

        # Sharpe/Sortino from the daily realized balance INCLUDING funding,
        # over the whole backtest window (NT's series stops at the last fill
        # and knows nothing of funding). Same function as validation.
        if start_ns is not None and end_ns is not None and end_ns > start_ns:
            metrics.sharpe_ratio, metrics.sortino_ratio = daily_sharpe_sortino(
                starting_balance, cash_events, start_ns, end_ns
            )
        # Compute return distribution moments (skewness/kurtosis) and per-trade returns
        if liquidated is not None:
            metrics.skewness, metrics.kurtosis, metrics.trade_returns = (
                self._liquidation_return_moments(liquidated.returns, liquidated.n_closed)
            )
        else:
            metrics.skewness, metrics.kurtosis, metrics.trade_returns = (
                self._compute_return_moments(engine)
            )

        # Mark-to-market paths, built once: they feed the max drawdown (incl.
        # open-trade intrabar losses and funding, bd vibe-quant-e70tl.11) and,
        # when collecting series, the output-only daily MTM return series
        # (bd vibe-quant-wcn55). NT's own DD stats are daily realized-balance
        # figures and are not used.
        mtm_paths = mark_to_market_paths(
            engine,
            execution_timeframe=execution_tf,
            catalog_path=catalog_path,
            start_ns=start_ns,
            end_ns=end_ns,
            extra_cash=funding_cash if liquidated is None else None,
            ledgers=liquidation_ledgers,
        )
        if mtm_paths is None:
            logger.warning(
                "Mark-to-market drawdown unavailable for params %s (timeframe=%s) — "
                "using closed-trade drawdown, which ignores open-trade losses",
                params or "{}",
                execution_tf,
            )
            mtm_dd = closed_trade_drawdown(starting_balance, closed_net)
        else:
            mtm_dd = mark_to_market_max_drawdown(
                starting_balance,
                mtm_paths,
                floor_from_ns=liquidation_floor_ns,
            )
            if start_ns is not None and end_ns is not None and end_ns > start_ns:
                mtm_series = mark_to_market_daily_returns(
                    starting_balance, mtm_paths, start_ns, end_ns
                )
                if self._collect_daily_returns:
                    self.last_mtm_daily_returns = mtm_series
                # DSR inputs for EVERY run (bd vibe-quant-68630): the overfitting
                # and screening DSR gates test the market-valued series, never
                # the realized Sharpe above. NT's Sharpe drops day 0 (return vs
                # the starting balance); so does this, like discovery's series.
                if mtm_series is not None:
                    from vibe_quant.overfitting.dsr import daily_sharpe_inputs

                    mtm_values = mtm_series.without_first_day().values
                    (
                        metrics.mtm_sharpe_daily,
                        metrics.mtm_skewness,
                        metrics.mtm_kurtosis,
                    ) = daily_sharpe_inputs(mtm_values)
                    metrics.mtm_days = len(mtm_values)
        metrics.max_drawdown = mtm_dd

        return metrics

    @staticmethod
    def _compute_return_moments(engine: Any) -> tuple[float, float, tuple[float, ...]]:
        """Compute skewness, kurtosis, and per-trade returns from closed positions.

        Uses adjusted Fisher-Pearson G1 (skewness) and G2 (excess kurtosis)
        formulas — the same as scipy.stats.skew(bias=False) and
        scipy.stats.kurtosis(bias=False). Returns full kurtosis (G2 + 3).

        Falls back to (0.0, 3.0, ()) — normal distribution defaults — if
        fewer than 4 trades or computation fails.
        """
        try:
            cache = engine.kernel.cache  # type: ignore[union-attr]
            all_positions = list(cache.positions()) + list(cache.position_snapshots())
            closed = [p for p in all_positions if p.is_closed]
            if len(closed) < 4:
                return 0.0, 3.0, ()

            # Per-trade return as fraction of notional entry value
            returns: list[float] = []
            for pos in closed:
                # Use peak_qty (not quantity which is 0 for closed positions)
                entry_val = abs(float(pos.peak_qty) * float(pos.avg_px_open))
                if entry_val > 0:
                    returns.append(float(pos.realized_pnl) / entry_val)

            return NTScreeningRunner._moments_from_returns(returns)
        except Exception:
            logger.warning("Could not compute return moments", exc_info=True)
            return 0.0, 3.0, ()

    @staticmethod
    def _moments_from_returns(returns: list[float]) -> tuple[float, float, tuple[float, ...]]:
        """Skewness, kurtosis and per-trade returns of ``returns`` (see _compute_return_moments)."""
        import math

        n = len(returns)
        if n < 4:
            return 0.0, 3.0, tuple(returns)

        mean = sum(returns) / n
        diffs = [r - mean for r in returns]
        # Biased central moments (population moments)
        m2 = sum(d * d for d in diffs) / n
        if m2 == 0:
            return 0.0, 3.0, tuple(returns)

        m3 = sum(d**3 for d in diffs) / n
        m4 = sum(d**4 for d in diffs) / n

        # Adjusted Fisher-Pearson G1 skewness (bias-corrected)
        # G1 = sqrt(n(n-1)) / (n-2) * (m3 / m2^1.5)
        skewness = (math.sqrt(n * (n - 1)) / (n - 2)) * (m3 / m2**1.5)

        # Adjusted Fisher G2 excess kurtosis (bias-corrected)
        # G2 = (n+1)(n-1) / ((n-2)(n-3)) * m4/m2² - 3(n-1)² / ((n-2)(n-3))
        excess = ((n + 1) * (n - 1) * m4) / (
            (n - 2) * (n - 3) * m2**2
        ) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        # Full kurtosis = excess + 3 (DSR expects gamma_4, not excess)
        kurtosis = excess + 3.0

        # Clamp kurtosis to valid range (>=1 per DSR validator)
        kurtosis = max(1.0, kurtosis)

        return round(skewness, 4), round(kurtosis, 4), tuple(returns)

    @staticmethod
    def _liquidation_return_moments(
        returns: list[float], n_closed: int
    ) -> tuple[float, float, tuple[float, ...]]:
        """The return moments of the liquidation-adjusted trades (same rules as the NT path)."""
        try:
            if n_closed < 4:
                return 0.0, 3.0, ()
            return NTScreeningRunner._moments_from_returns(returns)
        except Exception:
            logger.warning("Could not compute return moments", exc_info=True)
            return 0.0, 3.0, ()

    def _boundary_exit_scan_ts(
        self,
        engine: Any,
        positions: list[Any],
        execution_bar_ns: int,
        catalog_path: Path | str | None,
        purpose: str = "liquidation scan",
    ) -> tuple[dict[str, dict[int, int]] | None, dict[str, Any]]:
        """Refined scan timestamps of the run's boundary resting exits, plus their R25b notes.

        R17: only a coarser-than-1m execution with at least one boundary STOP_MARKET/LIMIT fill
        refines anything; anything else returns ``None`` (byte-identical scan) and all-zero
        counts. R17b: the 1m LAST series is loaded through
        :func:`~vibe_quant.validation.extraction.load_bar_windows` per SYMBOL, only for symbols
        that have such a fill -- never for a symbol without one, never from the engine cache and
        never the MARK series (R19). A symbol whose 1m LAST bars are absent leaves no series
        behind, so :func:`~vibe_quant.liquidation.exit_timing.refine_resting_exits` raises
        ``MissingExitBars`` (R26); the caller fails the run closed.
        """
        from vibe_quant.data.catalog import INTERVAL_TO_AGGREGATION
        from vibe_quant.liquidation.exit_timing import refine_resting_exits
        from vibe_quant.screening.liquidation_metrics import resting_exit_inputs
        from vibe_quant.validation.extraction import load_bar_windows

        if execution_bar_ns <= 60_000_000_000:  # 1m (or finer) execution: nothing to refine
            return None, _no_exit_timing()
        exits = resting_exit_inputs(engine, positions, execution_bar_ns)
        if not exits:
            return None, _no_exit_timing()
        step, agg = INTERVAL_TO_AGGREGATION["1m"]
        windows: dict[str, list[tuple[int, int]]] = {}
        for exit_ in exits:
            # R19: the exit's own bar window, [ts + 1 ms - E, ts], as an inclusive ns pair
            windows.setdefault(exit_.symbol, []).append(
                (exit_.ts + _ONE_MS - execution_bar_ns, exit_.ts)
            )
        bars_1m: dict[str, Any] = {}
        if catalog_path is not None:  # R10c: no catalog -> no 1m LAST -> MissingExitBars
            for symbol, spans in windows.items():
                series = load_bar_windows(
                    catalog_path,
                    f"{symbol}-PERP.BINANCE-{step}-{agg.name}-LAST-EXTERNAL",
                    spans,
                )
                if series is not None:
                    bars_1m[symbol] = series
        timing = refine_resting_exits(exits, bars_1m, execution_bar_ns)
        if timing.kept_bar_close:
            suffix = (
                f" (liquidation_mode={self._liquidation_mode})"
                if purpose == "liquidation scan"
                else ""
            )
            logger.warning(
                "Exit refinement for the %s: %d of %d boundary resting fill(s) kept "
                "their NT strategy-bar timestamp (no 1m bar reached the order's level); the "
                "%s may run past the real exit%s",
                purpose,
                timing.kept_bar_close,
                timing.refined + timing.kept_bar_close,
                purpose,
                suffix,
            )
        scan_fill_ts: dict[str, dict[int, int]] = {}
        for (key, index), ts in timing.scan_ts.items():
            scan_fill_ts.setdefault(key, {})[index] = ts
        return scan_fill_ts, timing.to_notes()

    def _funding_scan_ts(
        self,
        engine: Any,
        positions: list[Any],
        execution_tf: str | None,
        catalog_path: Path | str | None,
    ) -> dict[str, dict[int, int]] | None:
        """The MODE-OFF refined fill timestamps the funding exit is read from (R9/R10-R10c).

        R9/R14: an execution timeframe at or below 8h is excluded before any input is read, so
        no boundary scan runs and no ``vibe_quant.liquidation`` module is imported (R9d). A
        ``MissingExitBars`` from the scan is a :class:`DataUnavailableError` and re-raised
        unchanged (R10/R10c); any other failure becomes one naming the timeframe (R10b). The
        R25b notes of the scan are dropped: mode off records no exit timing (R16).
        """
        from vibe_quant.validation.extraction import _TIMEFRAME_SECONDS

        if execution_tf not in _TIMEFRAME_SECONDS:
            return None
        bar_ns = _TIMEFRAME_SECONDS[execution_tf] * 1_000_000_000
        if bar_ns <= _FUNDING_REFINE_MIN_BAR_NS:
            return None
        try:
            scan_fill_ts, _notes = self._boundary_exit_scan_ts(
                engine, positions, bar_ns, catalog_path, purpose="funding"
            )
        except DataUnavailableError:
            raise
        except Exception as exc:
            raise DataUnavailableError(
                f"funding exit refinement ({execution_tf}): {type(exc).__name__}: {exc}"
            ) from exc
        return scan_fill_ts

    def _scan_with_survivor_marks(
        self,
        engine: Any,
        closed: list[Any],
        still_open: list[Any],
        funding_calc: FundingCalculator,
        *,
        starting_balance: float,
        start_ns: int | None,
        end_ns: int | None,
        execution_tf: str | None,
        catalog_path: Path | str | None,
    ) -> tuple[
        OverlayResult,
        list[OpenPositionMark],
        dict[str, FundingAccrual],
        dict[str, dict[int, int]] | None,
    ]:
        """ONE liquidation scan, then the run-end marks of the positions that SURVIVE it.

        Before the scan every open position gets its valuation horizon: the ts of its
        instrument's last LAST execution bar in the window, where the mode-off path marks it.
        Only a ts inside the run's final execution bar is a horizon; the scan funds the position
        up to it, i.e. exactly the funding the mode-off path charges it. A position without one
        (a LAST data gap) is funded through the run's last instant (settlements < end_ns: the
        window is half-open, one AT the run end is never charged): if the scan liquidates or
        drops it, the overlay truncates that funding at the hit and it needs no valuation; if it
        survives, the run fails closed. Closed positions are funded to their close. Price exposure runs to
        the run end, so a hit after the horizon counts; the overlay cuts the funding at the hit,
        so an open position is funded through min(hit, horizon). The per-position funding is
        returned: the trade-level accounting reuses it (truncated at a hit), never re-accrues.

        The scan's hit set is final (there is no rescan). Liquidated and dropped positions are
        valued by the overlay and never marked; only survivors are. Their end-of-run exit fee
        is a valuation cost, not a liquidation trigger: the scan never sees it, it is booked
        into the ledgers afterwards, so final equity, total_fees and the trade net all carry it
        (equity below zero is reported as is).

        Raises:
            DataUnavailableError: a survivor has no LAST bar in the run's final execution bar
                (none at all, or its coverage ends earlier). The mode-off path keeps the
                marking helper's behaviour (values it at its last bar, or warns and excludes it).
        """
        from dataclasses import replace

        from vibe_quant.liquidation.apply import apply_liquidation
        from vibe_quant.metrics import LedgerEvent
        from vibe_quant.screening.liquidation_metrics import position_key
        from vibe_quant.validation.extraction import (
            _TIMEFRAME_SECONDS,
            accrue_position_funding,
            mark_open_positions,
        )

        # Any failure below is wrapped as LiquidationIntegrityError by the ONE guard in
        # ``_extract_metrics`` (the caller), which fails the run closed.
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            msg = (
                f"liquidation_mode={self._liquidation_mode!r} needs a valid run window, got "
                f"{self._start_date!r}..{self._end_date!r}"
            )
            raise ValueError(msg)
        if execution_tf not in _TIMEFRAME_SECONDS or catalog_path is None:
            msg = (
                f"liquidation_mode={self._liquidation_mode!r} needs an execution timeframe "
                f"and a catalog (timeframe={execution_tf!r}, catalog={catalog_path!r})"
            )
            raise ValueError(msg)
        bar_ns = _TIMEFRAME_SECONDS[execution_tf] * 1_000_000_000
        # vjno2 (R17-R30): a coarser-than-1m run fills a resting exit at the strategy bar's
        # close, a whole bar after the 1m bar validation filled it on. Refine those boundary
        # fills to their real 1m instant so the scan's exposure ends when the position did.
        # 6osqd D1: the SAME refined timestamps move the funding exit of a CLOSED position
        # whose closing fill is such an exit (_funding_close_ns); the ledgers, realized PnL and
        # MTM keep the NT timestamps (R15/R24), and the notes go onto the run's liquidation
        # summary whatever the outcome (R25b). Run before the accruals: they read it.
        positions = [*closed, *still_open]
        scan_fill_ts, self._last_exit_timing_notes = self._boundary_exit_scan_ts(
            engine, positions, bar_ns, catalog_path
        )
        last_bar: dict[str, int | None] = {}  # instrument -> ts of its last LAST bar
        horizons: dict[str, int | None] = {}
        accruals = {
            position_key(p): accrue_position_funding(
                funding_calc, p, exit_ns=_funding_close_ns(p, scan_fill_ts, bar_ns)
            )
            for p in closed
        }
        for pos in still_open:
            iid = str(pos.instrument_id)
            if iid not in last_bar:
                last_bar[iid] = _last_execution_bar_ts(
                    engine, iid, execution_tf, catalog_path, start_ns, end_ns
                )
            ts = last_bar[iid]
            horizon = ts if ts is not None and ts > end_ns - bar_ns else None
            horizons[position_key(pos)] = horizon
            accruals[position_key(pos)] = accrue_position_funding(
                funding_calc, pos, exit_ns=end_ns - 1 if horizon is None else horizon
            )
        overlay = apply_liquidation(
            positions,
            mode=self._liquidation_mode,  # type: ignore[arg-type]  # "off" excluded by caller
            leverage=self._leverage,
            starting_balance=starting_balance,
            catalog_path=catalog_path,
            start_ns=start_ns,
            end_ns=end_ns,
            execution_bar_ns=bar_ns,
            funding_by_position={  # ledger sign: paid < 0
                key: [(ts, -amount) for ts, amount in accrual.payments]
                for key, accrual in accruals.items()
            },
            cash_by_position={},
            scan_fill_ts=scan_fill_ts,
        )
        survivors = [
            p
            for p in still_open
            if position_key(p) not in overlay.hits_by_key
            and position_key(p) not in overlay.dropped_keys
        ]
        for pos in survivors:
            if horizons[position_key(pos)] is None:
                ts = last_bar[str(pos.instrument_id)]
                coverage = (
                    f"no {execution_tf} LAST bars to mark it"
                    if ts is None
                    else f"its {execution_tf} LAST bars end at {ts}, before the run's final "
                    f"execution bar"
                )
                msg = (
                    f"liquidation_mode={self._liquidation_mode!r}: {pos.instrument_id} opened at "
                    f"{int(pos.ts_opened)} is still open at the end of the run and cannot be "
                    f"valued: {coverage} (run end {end_ns})"
                )
                raise DataUnavailableError(msg)
        marks = mark_open_positions(
            engine,
            survivors,
            execution_timeframe=execution_tf,
            catalog_path=catalog_path,
            start_ns=start_ns,
            end_ns=end_ns,
            taker_fee=self._fee_schedule.taker_rate,
        )
        if [(position_key(m.position), m.ts) for m in marks] != [
            (position_key(p), horizons[position_key(p)]) for p in survivors
        ]:
            # the scan funded each survivor to its horizon: the marks must sit exactly there
            msg = "liquidation survivor marks diverge from the horizons the scan funded them to"
            raise RuntimeError(msg)
        if overlay.ledgers is not None and marks:
            ledgers = {iid: list(events) for iid, events in overlay.ledgers.items()}
            for mark in marks:
                events = ledgers.setdefault(str(mark.position.instrument_id), [])
                at = len(events)
                while at and events[at - 1].ts > mark.ts:
                    at -= 1
                events.insert(at, LedgerEvent(ts=mark.ts, cash=-mark.exit_fee))
            overlay = replace(overlay, ledgers=ledgers)
        return overlay, marks, accruals, scan_fill_ts


def _last_execution_bar_ts(
    engine: Any,
    instrument_id: str,
    execution_tf: str,
    catalog_path: Path | str,
    start_ns: int,
    end_ns: int,
) -> int | None:
    """ts of the instrument's last LAST execution bar in ``[start_ns, end_ns]``, or None.

    The same series lookup ``mark_open_positions`` marks an open position on (catalog, else
    the engine cache), without marking or logging anything.
    """
    from vibe_quant.data.catalog import INTERVAL_TO_AGGREGATION
    from vibe_quant.validation.extraction import cache_bars, load_catalog_bars

    if execution_tf not in INTERVAL_TO_AGGREGATION:
        return None
    step, agg = INTERVAL_TO_AGGREGATION[execution_tf]
    bar_type = f"{instrument_id}-{step}-{agg.name}-LAST-EXTERNAL"
    series = load_catalog_bars(catalog_path, bar_type)
    if series is None:
        series = cache_bars(engine, bar_type)
    if series is None:
        return None
    series = series.window(start_ns, end_ns)
    return int(series.ts[-1]) if len(series) else None
