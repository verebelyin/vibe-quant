"""NautilusTrader backtest runner for screening mode.

Runs single-parameter-combination backtests using BacktestNode with
screening venue config (no latency, simple fill model). Designed to be
picklable for :class:`concurrent.futures.ProcessPoolExecutor`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe_quant.errors import DataUnavailableError

if TYPE_CHECKING:
    from vibe_quant.screening.types import BacktestMetrics

logger = logging.getLogger(__name__)

# Process-level compile cache: DSL-content key -> (module_path, strategy_cls,
# config_cls, timeframes). Multi-window evals (eval_windows > 1) construct a
# fresh runner per window for the SAME chromosome; without this cache each
# window recompiled the identical DSL (~1s per compile).
_COMPILE_CACHE: dict[str, tuple[str, str, str, frozenset[str]]] = {}


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

    Raises:
        UnknownStrategyParamError: if any key matches no config field, or two
            keys resolve to the same field.
    """
    fields = set(config_fields)
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
    ) -> None:
        """Initialize NTScreeningRunner.

        Args:
            dsl_dict: Strategy DSL as dict (picklable, unlike StrategyDSL).
            symbols: List of symbols to screen.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            catalog_path: Path to ParquetDataCatalog. Uses default if None.
            funding_archive_path: Raw-data archive holding funding rates.
                Uses the default archive if None. Funding series are cached
                per worker process (see validation.funding).
        """
        self._dsl_dict = dsl_dict
        self._symbols = symbols
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
        try:
            return self._run_backtest(params, start_time)
        except (UnknownStrategyParamError, DataUnavailableError):
            # Configuration error, not a backtest failure: never mask it as a
            # -inf result (vibe-quant-e70tl.8).
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
            self._module_path, self._strategy_cls_name, self._config_cls_name, tfs = cached
            self._all_timeframes: set[str] = set(tfs)
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

        # Cache parsed DSL fields needed for data config
        self._all_timeframes = {dsl.timeframe}
        self._all_timeframes.update(dsl.additional_timeframes)
        for ind_config in dsl.indicators.values():
            if ind_config.timeframe:
                self._all_timeframes.add(ind_config.timeframe)

        _COMPILE_CACHE[cache_key] = (
            self._module_path,
            self._strategy_cls_name,
            self._config_cls_name,
            frozenset(self._all_timeframes),
        )
        self._compiled = True

    def _run_backtest(self, params: dict[str, float | int], start_time: float) -> BacktestMetrics:
        """Execute the NautilusTrader backtest."""
        from nautilus_trader.backtest.node import BacktestNode
        from nautilus_trader.config import (
            BacktestDataConfig,
            BacktestEngineConfig,
            BacktestRunConfig,
            ImportableStrategyConfig,
        )
        from nautilus_trader.core.nautilus_pyo3 import (
            ProfitFactor,
            SharpeRatio,
            SortinoRatio,
            WinRate,
        )
        from nautilus_trader.model.data import Bar

        from vibe_quant.data.catalog import (
            INTERVAL_TO_AGGREGATION,
        )
        from vibe_quant.screening.types import BacktestMetrics
        from vibe_quant.validation.venue import (
            create_backtest_venue_config,
            create_venue_config_for_screening,
        )

        # Compile DSL once per worker process
        self._ensure_compiled()

        module_path = self._module_path
        strategy_cls_name = self._strategy_cls_name
        config_cls_name = self._config_cls_name
        catalog_path = self._resolved_catalog_path

        # Map sweep keys (dot notation, known aliases) onto the generated
        # StrategyConfig's fields; an unknown key raises instead of being
        # silently dropped (NT 1.226+ would also reject it at decode time).
        strategy_params = resolve_strategy_params(params, self._config_fields())

        # Strategy configs (with parameter overrides)
        strategy_configs: list[ImportableStrategyConfig] = []
        for symbol in self._symbols:
            instrument_id = f"{symbol}-PERP.BINANCE"
            config_dict: dict[str, Any] = {"instrument_id": instrument_id, **strategy_params}
            # Always defer entries/exits one bar. NT (bar_execution, no latency)
            # fills a market order from on_bar(t) at close[t] -- the signal bar's
            # own close -- which is same-bar look-ahead. Deferring makes the fill
            # land at close[t+1], matching the validation tier. This governs the
            # screening, discovery, WFA and purged-k-fold paths (all share this runner).
            config_dict["execution_delay_probability"] = 1.0
            strategy_configs.append(
                ImportableStrategyConfig(
                    strategy_path=f"{module_path}:{strategy_cls_name}",
                    config_path=f"{module_path}:{config_cls_name}",
                    config=config_dict,
                )
            )

        # Data configs
        data_configs: list[BacktestDataConfig] = []
        for symbol in self._symbols:
            instrument_id = f"{symbol}-PERP.BINANCE"
            for tf in sorted(self._all_timeframes):
                if tf not in INTERVAL_TO_AGGREGATION:
                    continue
                step, agg = INTERVAL_TO_AGGREGATION[tf]
                bar_type_str = f"{instrument_id}-{step}-{agg.name}-LAST-EXTERNAL"
                require_bars_in_window(
                    str(catalog_path.resolve()), bar_type_str, self._start_date, self._end_date
                )
                # NT 1.226+: pass data_cls as the CLASS, not the import
                # string — BacktestDataConfig.query compares `data_cls is Bar`
                # so a string silently disables bar-type narrowing and loads
                # every bar timeframe in the catalog (~300x the needed data).
                data_configs.append(
                    BacktestDataConfig(
                        catalog_path=str(catalog_path.resolve()),
                        data_cls=Bar,
                        bar_types=[bar_type_str],
                        start_time=self._start_date,
                        end_time=self._end_date,
                    )
                )

        if not data_configs:
            return BacktestMetrics(
                parameters=params,
                sharpe_ratio=float("-inf"),
                execution_time_seconds=time.time() - start_time,
            )

        # Screening venue config: no latency, simple fills
        venue_config = create_venue_config_for_screening()
        bt_venue_config = create_backtest_venue_config(venue_config)

        # Suppress NT's verbose INFO logging in discovery/screening mode
        # (every order/fill/position logs at INFO, generating 100s of MB)
        # Screening runs in parallel workers, so NT engine output defaults to
        # WARNING to keep sweep logs readable. Override per run via
        # VIBE_QUANT_NT_LOG_LEVEL_SCREENING (TRACE/DEBUG/INFO/WARNING/ERROR).
        import os

        from nautilus_trader.config import LoggingConfig

        engine_config = BacktestEngineConfig(
            strategies=strategy_configs,
            run_analysis=True,
            logging=LoggingConfig(
                log_level=os.environ.get("VIBE_QUANT_NT_LOG_LEVEL_SCREENING", "WARNING")
            ),
        )

        bt_run_config = BacktestRunConfig(
            engine=engine_config,
            venues=[bt_venue_config],
            data=data_configs,
            start=self._start_date,
            end=self._end_date,
            dispose_on_completion=False,
        )

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

            node.run()

            engine = node.get_engine(bt_run_config.id)
            if engine is None:
                return BacktestMetrics(
                    parameters=params,
                    sharpe_ratio=float("-inf"),
                    execution_time_seconds=time.time() - start_time,
                )
            bt_result = engine.get_result()

            metrics = self._extract_metrics(
                params,
                bt_result,
                engine,
                start_time,
                starting_balance=float(venue_config.starting_balance_usdt),
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
            return metrics
        finally:
            # Reset engines before dispose to avoid
            # InvalidStateTrigger('RUNNING -> DISPOSE')
            import contextlib

            from vibe_quant.nt_compat import retain_log_guard

            for eng in node.get_engines():
                retain_log_guard(eng)
                with contextlib.suppress(Exception):
                    eng.reset()
            node.dispose()  # type: ignore[no-untyped-call]

            # NT writes corrupt epoch-timestamp instrument parquet on dispose()
            from vibe_quant.data.catalog import cleanup_epoch_parquet

            cleanup_epoch_parquet(catalog_path)

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
        from vibe_quant.metrics import closed_trade_drawdown, profit_factor
        from vibe_quant.screening.types import BacktestMetrics
        from vibe_quant.validation.extraction import (
            accrue_position_funding,
            all_positions,
            daily_sharpe_sortino,
            date_to_ns,
            finest_timeframe,
            mark_to_market_drawdown,
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

        # Fees, funding and net trade PnLs from closed positions.
        # NT netting mode removes closed positions from the main index;
        # combine positions() + position_snapshots() to capture all.
        trade_pnls: list[float] = []
        closed_net: list[tuple[int, float]] = []
        cash_events: list[tuple[int, float]] = []
        funding_cash: dict[str, list[tuple[int, float]]] = {}
        total_funding = 0.0
        funding_fallbacks = 0
        try:
            closed = [p for p in all_positions(engine) if p.is_closed]
        except Exception:
            logger.warning("Could not read positions from engine cache", exc_info=True)
            closed = []
        funding_calc = FundingCalculator(self._funding_archive_path)
        total_fees = 0.0
        for pos in closed:
            total_fees += sum(abs(float(c)) for c in pos.commissions())
            # Funding is modeled post-hoc (NT's engine applies none) with
            # the same calculator validation uses (bd vibe-quant-e70tl.20).
            accrual = accrue_position_funding(funding_calc, pos)
            total_funding += accrual.total
            funding_fallbacks += accrual.fallback_settlements
            # NT realized_pnl is net of commissions
            realized = float(pos.realized_pnl)
            trade_pnls.append(realized - accrual.total)
            closed_net.append((int(pos.ts_closed), realized - accrual.total))
            cash_events.append((int(pos.ts_closed), realized))
            cash_events.extend((ts, -amount) for ts, amount in accrual.payments)
            funding_cash.setdefault(str(pos.instrument_id), []).extend(
                (ts, -amount) for ts, amount in accrual.payments
            )
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
        if total_funding != 0.0 and starting_balance > 0:
            metrics.total_return -= total_funding / starting_balance

        # Trade-based profit factor on net PnL (shared definition with
        # validation). NT's realized-PnL PF is unimplemented, and its
        # returns-based PF is a daily-return statistic (bd vibe-quant-e70tl.7).
        metrics.profit_factor = profit_factor(trade_pnls)

        # Sharpe/Sortino from the daily realized balance INCLUDING funding,
        # over the whole backtest window (NT's series stops at the last fill
        # and knows nothing of funding). Same function as validation.
        start_ns = date_to_ns(self._start_date)
        end_ns = date_to_ns(self._end_date)
        if start_ns is not None and end_ns is not None and end_ns > start_ns:
            metrics.sharpe_ratio, metrics.sortino_ratio = daily_sharpe_sortino(
                starting_balance, cash_events, start_ns, end_ns
            )

        # Compute return distribution moments (skewness/kurtosis) and per-trade returns
        metrics.skewness, metrics.kurtosis, metrics.trade_returns = self._compute_return_moments(engine)

        # Mark-to-market max drawdown incl. open-trade intrabar losses and
        # funding (bd vibe-quant-e70tl.11). NT's own DD stats are daily
        # realized-balance figures and are not used.
        execution_tf = finest_timeframe(getattr(self, "_all_timeframes", ()))
        mtm_dd = mark_to_market_drawdown(
            engine,
            starting_balance,
            execution_timeframe=execution_tf,
            catalog_path=getattr(self, "_resolved_catalog_path", None),
            start_ns=start_ns,
            end_ns=end_ns,
            extra_cash=funding_cash,
        )
        if mtm_dd is None:
            logger.warning(
                "Mark-to-market drawdown unavailable for params %s (timeframe=%s) — "
                "using closed-trade drawdown, which ignores open-trade losses",
                params or "{}",
                execution_tf,
            )
            mtm_dd = closed_trade_drawdown(starting_balance, closed_net)
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
        import math

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
        except Exception:
            logger.warning("Could not compute return moments", exc_info=True)
            return 0.0, 3.0, ()
