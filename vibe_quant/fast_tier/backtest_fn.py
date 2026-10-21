"""``FastBacktestFn``: ``NTBacktestFn`` with the NT engine replaced by the fast-tier replay.

Only ``_run_single`` is overridden, so window aggregation, worst-of-symbols and the
error-row handling of ``__call__`` stay one implementation. The metrics come from the
screening runner's own ``_extract_metrics`` over a facade engine whose cache holds the
replayed positions (and nothing else the metrics read), so every number after the fills
is the screening code.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.fast_tier.errors import FastTierUnsupported

if TYPE_CHECKING:
    from vibe_quant.discovery.operators import StrategyChromosome

_INSTRUMENT_SUFFIX = "-PERP.BINANCE"


class _FacadeCache:
    """``engine.kernel.cache`` as the metrics code reads it (NT netting order)."""

    def __init__(self, positions: list[Any], instrument: Any) -> None:
        # NT keeps the latest position object per id in positions() and every earlier
        # (closed, then reopened) one in position_snapshots(), oldest first.
        self._current = positions[-1:]
        self._snapshots = positions[:-1]
        self._instrument = instrument

    def positions(self) -> list[Any]:
        return list(self._current)

    def position_snapshots(self) -> list[Any]:
        return list(self._snapshots)

    def instrument(self, _instrument_id: object) -> Any:
        return self._instrument

    def bars(self, *_a: object) -> list[Any]:
        return []

    def orders(self) -> list[Any]:
        return []


def _facade_engine(positions: list[Any], instrument: Any) -> Any:
    cache = _FacadeCache(positions, instrument)
    return SimpleNamespace(kernel=SimpleNamespace(cache=cache), cache=cache)


def _facade_result(positions: list[Any], starting_balance: float) -> Any:
    """``BacktestResult`` fields ``_extract_metrics`` reads; PnL% as NT's analyzer computes it."""
    from nautilus_trader.model.objects import Money

    ccy = positions[0].quote_currency if positions else None
    pct = 0.0
    if ccy is not None:
        balance = Money(starting_balance, ccy)
        for pos in positions:
            balance = balance + pos.realized_pnl
        start = Money(starting_balance, ccy)
        # PortfolioAnalyzer.total_pnl_percentage: (balance + 0.0 - start) / start * 100
        current = balance + 0.0
        pct = float(((current - start) / start) * 100)
    stats = {ccy.code: {"PnL% (total)": pct}} if ccy is not None else {}
    return SimpleNamespace(total_positions=len(positions), stats_pnls=stats, stats_returns={})


class FastBacktestFn(NTBacktestFn):
    """Exploration-only twin of :class:`NTBacktestFn` (1m, single symbol, market entries)."""

    def _run_single(
        self,
        chromosome: StrategyChromosome,
        start_date: str,
        end_date: str,
        symbols: list[str] | None = None,
    ) -> dict[str, float | int]:
        from vibe_quant.discovery.genome import chromosome_to_dsl
        from vibe_quant.fast_tier.fills import Replay
        from vibe_quant.fast_tier.signals import (
            build_strategy,
            compute_signals,
            load_bars,
            load_instrument,
        )
        from vibe_quant.fees import FeeSchedule
        from vibe_quant.screening.nt_runner import resolve_strategy_params
        from vibe_quant.validation.fill_model import ScreeningFillModelConfig
        from vibe_quant.validation.venue import (
            SCREENING_FILL_SEED,
            create_venue_config_for_screening,
        )

        run_symbols = self.symbols if symbols is None else symbols
        if self.timeframe != "1m":
            raise FastTierUnsupported(f"timeframe {self.timeframe}: the fast tier replays 1m only")
        if len(run_symbols) != 1:
            raise FastTierUnsupported("one symbol per account only (use symbol_agg=worst)")
        if self.fee_schedule != FeeSchedule():
            raise FastTierUnsupported("non-default fee schedule")
        dsl_dict = chromosome_to_dsl(chromosome)
        dsl_dict["timeframe"] = self.timeframe
        if dsl_dict.get("time_filters"):
            raise FastTierUnsupported("time filters")

        t0 = time.time()
        runner = self._make_runner(dsl_dict, start_date, end_date, run_symbols)
        runner._ensure_compiled()
        if runner._all_timeframes != {"1m"}:
            raise FastTierUnsupported(f"loads {sorted(runner._all_timeframes)}: only 1m is replayed")
        catalog = str(runner._resolved_catalog_path.resolve())
        instrument_id = f"{run_symbols[0]}{_INSTRUMENT_SUFFIX}"
        instrument = load_instrument(catalog, instrument_id)
        bars = load_bars(catalog, f"{instrument_id}-1-MINUTE-LAST-EXTERNAL", start_date, end_date)
        if not bars:
            from vibe_quant.errors import DataUnavailableError

            raise DataUnavailableError(f"no 1m bars for {instrument_id} {start_date}..{end_date}")

        params = resolve_strategy_params({}, runner._config_fields())
        params.update(runner._sizing.strategy_params(runner._leverage))
        config = {"instrument_id": instrument_id, **params, "execution_delay_probability": 1.0}
        strategy, indicators, cache = build_strategy(
            runner._module_path,
            runner._strategy_cls_name,
            runner._config_cls_name,
            config,
            instrument,
        )
        venue = create_venue_config_for_screening(starting_balance_usdt=runner._starting_balance)
        starting_balance = float(venue.starting_balance_usdt)
        fill_cfg = venue.fill_config
        if not isinstance(fill_cfg, ScreeningFillModelConfig):
            raise FastTierUnsupported(f"screening fill config {fill_cfg!r}")
        raw_indicators = dsl_dict.get("indicators") or {}
        indicator_types = (
            {str(k): str(v["type"]) for k, v in raw_indicators.items()}
            if isinstance(raw_indicators, dict)
            else {}
        )
        positions = Replay(
            strategy=strategy,
            cache=cache,
            instrument=instrument,
            bars=bars,
            signals=compute_signals(
                strategy,
                indicators,
                bars,
                indicator_types,
            ),
            starting_balance=starting_balance,
            fill_seed=SCREENING_FILL_SEED,
            prob_slippage=fill_cfg.prob_slippage,
            prob_fill_on_limit=fill_cfg.prob_fill_on_limit,
            end_ts=int(bars[-1].ts_init),  # on_stop closes at the last bar's clock
        ).run()
        self.last_positions = positions

        result = runner._extract_metrics(
            {},
            _facade_result(positions, starting_balance),
            _facade_engine(positions, instrument),
            t0,
            starting_balance=starting_balance,
        )
        out: dict[str, float | int] = {
            "sharpe_ratio": result.sharpe_ratio if result.sharpe_ratio != float("-inf") else -1.0,
            "max_drawdown": result.max_drawdown,
            "profit_factor": result.profit_factor,
            "total_trades": result.total_trades,
            "total_return": getattr(result, "total_return", 0.0),
            "skewness": getattr(result, "skewness", 0.0),
            "kurtosis": getattr(result, "kurtosis", 3.0),
            "trade_returns": getattr(result, "trade_returns", ()),  # type: ignore[arg-type,dict-item]
        }
        if self.collect_daily_returns:
            out["mtm_daily_returns"] = runner.last_mtm_daily_returns  # type: ignore[assignment]
        out["total_fees"] = float(getattr(result, "total_fees", 0.0))
        out["gross_pnl"] = float(getattr(result, "gross_pnl", 0.0))
        out["starting_balance"] = float(runner.last_starting_balance)
        out["days"] = int(runner.last_days)
        if self.liquidation.on:
            out["liquidation_count"] = int(getattr(result, "liquidation_count", 0))
            out["liquidation_forfeited"] = float(getattr(result, "liquidation_forfeited", 0.0))
        return out
