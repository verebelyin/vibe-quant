"""Picklable backtest callable for ProcessPoolExecutor (bd-cm14).

Lives in its own module — NOT in ``vibe_quant.discovery.__main__`` —
because ``python -m vibe_quant.discovery`` loads ``__main__.py`` as
``__main__``, and worker processes can't unpickle classes whose
``__module__`` is ``__main__`` (their own ``__main__`` is the
multiprocessing worker entrypoint, not the discovery CLI).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from vibe_quant.errors import DataUnavailableError
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.sizing_settings import SizingSettings

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vibe_quant.data.fill_ticks import FillTickSet
    from vibe_quant.discovery.operators import StrategyChromosome
    from vibe_quant.discovery.timing_null import TradeSpan, WindowTape
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.metrics import DailyReturns
    from vibe_quant.screening.nt_runner import NTScreeningRunner

logger = logging.getLogger(__name__)

SymbolAgg = Literal["portfolio", "worst", "pooled"]

_DAY_NS = 86_400 * 10**9


class PooledSeriesError(ValueError):
    """The per-symbol MTM series of a pooled candidate cannot be pooled (misaligned/missing).

    Not a :class:`DataUnavailableError`: the candidate gets an error result (score 0), the run
    goes on.
    """

# Default settings for a callable built without one: mode off (B008 wants a
# module-level singleton, not a call in the signature default).
_LIQUIDATION_OFF = LiquidationSettings()


def _finite_or_none(value: float | int) -> float | None:
    """NaN/inf -> None so persisted JSON stays valid."""
    v = float(value)
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def _sum_fee_fields(results: list[dict[str, float | int]]) -> dict[str, float]:
    """Summed fee-drag inputs over windows/symbols (bd vibe-quant-2gyk1.4).

    ``total_fees`` and ``gross_pnl`` are SUMMED so fitness derives the drag
    from the totals, never as an average of per-window/per-symbol ratios.
    """
    return {
        "total_fees": sum(float(r.get("total_fees", 0.0)) for r in results),
        "gross_pnl": sum(float(r.get("gross_pnl", 0.0)) for r in results),
    }


def _sum_liquidation_fields(
    results: list[dict[str, float | int]],
) -> dict[str, float | int]:
    """Summed report-only liquidation counts/flows over windows or symbols.

    Empty when the mode is off: :meth:`NTBacktestFn._run_single` writes the keys
    only when it is on, so a mode-off aggregate stays byte-identical to today
    (R4). The sums cover exactly ``results`` -- for an early exit that is the
    windows that actually ran (R2) -- and per-symbol counts stay in
    ``symbol_results`` (R3).
    """
    if not any("liquidation_count" in r for r in results):
        return {}
    return {
        "liquidation_count": sum(int(r.get("liquidation_count", 0)) for r in results),
        "liquidation_forfeited": sum(
            float(r.get("liquidation_forfeited", 0.0)) for r in results
        ),
    }


@dataclass(frozen=True)
class CapturedRun:
    """A capture re-run of one genome's train windows: the timing-null gate's input.

    ``bt`` is the re-run's result dict (the gate rebuilds the FitnessResult from it and
    requires it to equal the scored one); ``tapes`` / ``spans`` hold each symbol's eval
    windows in date order, symbols in the fn's order (the aggregation order).
    """

    bt: dict[str, Any]
    tapes: dict[str, list[WindowTape]]
    spans: dict[str, list[list[TradeSpan]]]
    b0: float
    fee: float
    boundary_clipped: int


class NTBacktestFn:
    """Picklable backtest callable for ProcessPoolExecutor.

    Top-level class with a stable importable path so multiprocessing
    workers can unpickle it. Supports multi-window evaluation: when
    ``windows`` has 2+ entries, runs the backtest on each window and
    returns WORST-of-N metrics (min Sharpe, min return, max drawdown, min
    profit factor; trades summed), with a per-window trade gate -- see
    :meth:`_aggregate_multi_window`. This forces the GA to find
    regime-robust strategies instead of one great window averaging out
    two losing ones (vibe-quant-e70tl.6).

    ``symbol_agg="worst"`` (opt-in, vibe-quant-kzowb) evaluates each symbol
    as its own single-symbol backtest (all windows per symbol) and scores the
    genome by its worst symbol; the default ``"portfolio"`` runs one
    shared-account backtest over all symbols. One symbol cannot carry the rest
    in worst mode, and ``total_trades`` is the MIN across symbols so the
    min-trades gate applies per symbol. The result also carries
    ``symbol_results`` (each symbol's own window-aggregated metrics) so the
    GA can rank by a soft per-symbol score (vibe-quant-ox73t, see
    ``fitness.soft_worst_score``); there is no across-symbol early exit
    because a losing symbol no longer pins the score to 0.

    ``collect_daily_returns`` (opt-in; the discovery pipeline turns it on for
    its train fn when DSR is required, bd vibe-quant-yul7u.24) adds
    ``mtm_daily_returns``: the run's dense MARK-TO-MARKET daily return series
    (windows chained, see :meth:`_chain_window_series`; worst mode: the
    equal-weight pool of the symbols, plus ``symbol_mtm_daily`` per-symbol
    series). It is the DSR gate's input only -- fitness and ranking never read
    it. The realized (non-MTM) series was removed (bd vibe-quant-f9zzq).

    ``fee_schedule`` (bd vibe-quant-2gyk1.4) is venue-level and is passed to
    every :class:`NTScreeningRunner` this callable builds, including the
    holdout / cross-window / WFA factories. It must survive pickling into the
    worker processes, so it is a plain picklable attribute. ``entry_order``
    defaults to ``"market"``; post-only is validation-only and refused here.

    ``liquidation`` (bead vibe-quant-2gyk1.1.9) is the run's validated
    :class:`~vibe_quant.liquidation_settings.LiquidationSettings`. Its mode and
    leverage are passed to every :class:`NTScreeningRunner` this callable builds,
    and a mode-on run reports the liquidation counts/flows in the bt dict. The
    default is off, so every pre-existing number is byte-identical. It is a plain
    picklable attribute so it survives into worker processes.
    """

    def __init__(
        self,
        symbols: list[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        windows: list[tuple[str, str]] | None = None,
        min_trades: int = 0,
        symbol_agg: SymbolAgg = "portfolio",
        fill_ticks: FillTickSet | None = None,
        collect_daily_returns: bool = False,
        fee_schedule: FeeSchedule | None = None,
        entry_order: str = "market",
        liquidation: LiquidationSettings = _LIQUIDATION_OFF,
        sizing: SizingSettings | None = None,
    ) -> None:
        from vibe_quant.fees import FeeSchedule
        from vibe_quant.screening.nt_runner import refuse_non_market_entry

        refuse_non_market_entry(entry_order)
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.windows = windows
        # Global min-trades gate of the run; drives the per-window gate.
        self.min_trades = min_trades
        self.symbol_agg = symbol_agg
        # Resolved once by the pipeline's parent-side preflight (a frozen
        # dataclass of paths: cheap to pickle); workers never build ticks.
        self.fill_ticks = fill_ticks
        # Pooled mode scores the pooled MTM series itself, so every fn it builds (train,
        # holdout, cross-window/WFA factory, full range) collects it.
        self.collect_daily_returns = collect_daily_returns or symbol_agg == "pooled"
        self.entry_order = entry_order
        self.fee_schedule = FeeSchedule() if fee_schedule is None else fee_schedule
        # Run-level liquidation settings (bead vibe-quant-2gyk1.1.9): the same
        # resolved setting reaches every runner this callable builds, so the
        # train GA and every holdout/cross-window/WFA factory agree. Default is
        # off, which keeps every pre-existing number byte-identical.
        self.liquidation = liquidation
        # Run-level sizing settings (bead vibe-quant-2gyk1.2): the same
        # SizingSettings reaches every runner this callable builds, so the train
        # GA and every holdout/cross-window/WFA factory agree. None = the
        # defaults, which add no key and keep every pre-existing number
        # byte-identical.
        self.sizing = SizingSettings() if sizing is None else sizing
        # Precomputed compute_fn columns in the NT runner (bead vibe-quant-surto). The
        # fast evaluator's NT confirm (FastBacktestFn.nt_twin) turns it off to stay an
        # independent per-bar check of the replay. Plain bool: it pickles into workers.
        self.pta_precompute: bool = True

    def _make_runner(
        self,
        dsl_dict: dict[str, Any],
        start_date: str,
        end_date: str,
        symbols: list[str] | None = None,
    ) -> NTScreeningRunner:
        """Build the screening runner for one window (one launch settings object).

        Single construction site: the venue-level fee schedule, the entry order, the
        liquidation settings and the sizing must reach every runner this callable
        builds, in the parent or in a worker process. The account size stays the
        launch default (R33: discovery runs at 1000 regardless of its parameters).
        """
        from vibe_quant.launch_settings import LaunchSettings
        from vibe_quant.screening.nt_runner import NTScreeningRunner

        # capture_spans only rides along on a timing-null capture re-run (capture_spans())
        capture: dict[str, Any] = (
            {"capture_spans": True} if getattr(self, "_span_sink", None) is not None else {}
        )
        return NTScreeningRunner(
            dsl_dict=dsl_dict,
            symbols=self.symbols if symbols is None else symbols,
            start_date=start_date,
            end_date=end_date,
            fill_ticks=self.fill_ticks,
            collect_daily_returns=self.collect_daily_returns,
            launch=LaunchSettings(
                fee_schedule=self.fee_schedule,
                entry_order=self.entry_order,
                liquidation=self.liquidation,
                sizing=self.sizing,
            ),
            pta_precompute=getattr(self, "pta_precompute", True),
            **capture,
        )

    def capture_spans(self, chromosome: StrategyChromosome) -> CapturedRun:
        """Re-run ``chromosome``'s train windows with the runner's span capture on.

        The same evaluation as :meth:`__call__` (same windows, symbols and early exits) on a
        shallow copy, inside ``preserve_global_random`` (NT's fill model reseeds the global
        RNG): the GA's random stream never moves, so the capture is independent of how many
        genomes the gate sees and in which order.

        Raises:
            TimingNullError: a window or symbol produced no capture (failed run, no bars,
                early exit), so no schedule can be re-placed.
        """
        from vibe_quant.discovery.fitness import preserve_global_random
        from vibe_quant.discovery.timing_null import TimingNullError

        probe = copy.copy(self)
        sink: list[tuple[dict[str, Any] | None, float]] = []
        probe._span_sink = sink  # type: ignore[attr-defined]
        with preserve_global_random():
            bt = probe(chromosome)
        tapes: dict[str, list[WindowTape]] = {sym: [] for sym in self.symbols}
        spans: dict[str, list[list[TradeSpan]]] = {sym: [] for sym in self.symbols}
        clipped = 0
        for captured, _b0 in sink:
            for sym, (tape, window_spans, n_clipped) in (captured or {}).items():
                tapes[sym].append(tape)
                spans[sym].append(window_spans)
                clipped += n_clipped
        n_windows = len(self.windows) if self.windows and len(self.windows) >= 2 else 1
        for sym in self.symbols:
            if len(tapes[sym]) != n_windows:
                raise TimingNullError(
                    f"Timing null: trade capture for {sym} covers {len(tapes[sym])} of "
                    f"{n_windows} windows"
                )
        return CapturedRun(
            bt=dict(bt),
            tapes=tapes,
            spans=spans,
            b0=sink[0][1] if sink else 0.0,
            fee=float(self.fee_schedule.taker_rate),
            boundary_clipped=clipped,
        )

    def evaluate_dsl(self, dsl_dict: Mapping[str, Any]) -> dict[str, float | int]:
        """:meth:`__call__` on a DSL instead of a chromosome (a pooled screening child, Q12).

        The same windows, symbols, runners and aggregation as the GA path, so a replay of a
        champion's DSL reproduces its full-range headline exactly. Runs on a shallow copy.
        """
        from types import SimpleNamespace

        probe = copy.copy(self)
        probe._fixed_dsl = dict(dsl_dict)  # type: ignore[attr-defined]
        name = str(dsl_dict.get("name", "dsl"))
        return probe(cast("StrategyChromosome", SimpleNamespace(uid=name)))

    def _run_single(
        self,
        chromosome: StrategyChromosome,
        start_date: str,
        end_date: str,
        symbols: list[str] | None = None,
    ) -> dict[str, float | int]:
        from vibe_quant.discovery.genome import chromosome_to_dsl

        fixed = getattr(self, "_fixed_dsl", None)  # evaluate_dsl() only
        dsl_dict = dict(fixed) if fixed is not None else chromosome_to_dsl(chromosome)
        dsl_dict["timeframe"] = self.timeframe

        runner = self._make_runner(dsl_dict, start_date, end_date, symbols)
        result = runner({})
        sink = getattr(self, "_span_sink", None)
        if sink is not None:  # timing-null capture re-run only
            sink.append((runner.last_trade_spans, float(runner.last_starting_balance)))

        out: dict[str, float | int] = {
            "sharpe_ratio": result.sharpe_ratio
            if result.sharpe_ratio != float("-inf")
            else -1.0,
            "max_drawdown": result.max_drawdown,
            "profit_factor": result.profit_factor,
            "total_trades": result.total_trades,
            "total_return": getattr(result, "total_return", 0.0),
            "skewness": getattr(result, "skewness", 0.0),
            "kurtosis": getattr(result, "kurtosis", 3.0),
            "trade_returns": getattr(result, "trade_returns", ()),  # type: ignore[arg-type,dict-item]
        }
        if self.collect_daily_returns:
            # Full dense series INCLUDING day 0; trimmed by _chain_window_series.
            out["mtm_daily_returns"] = runner.last_mtm_daily_returns  # type: ignore[assignment]
        if self.symbol_agg == "pooled":
            # The pooled profit factor's input (one PF over every coin's trades).
            out["trade_pnls"] = tuple(runner.last_trade_pnls)  # type: ignore[assignment]
        # Fee-drag inputs (bd vibe-quant-2gyk1.4): raw cost-free PnL and fees,
        # plus the annualization inputs. The aggregates sum fees/gross and
        # fitness derives the ratios -- never an averaged ratio.
        out["total_fees"] = float(getattr(result, "total_fees", 0.0))
        out["gross_pnl"] = float(getattr(result, "gross_pnl", 0.0))
        out["starting_balance"] = float(runner.last_starting_balance)
        out["days"] = int(runner.last_days)
        if self.liquidation.on:
            # Report-only liquidation accounting (bead vibe-quant-2gyk1.1.9): the
            # keys exist ONLY when the mode is on (R4); the aggregates sum them.
            out["liquidation_count"] = int(getattr(result, "liquidation_count", 0))
            out["liquidation_forfeited"] = float(
                getattr(result, "liquidation_forfeited", 0.0)
            )
        return out

    @staticmethod
    def _chain_window_series(
        results: list[dict[str, float | int]], key: str = "mtm_daily_returns"
    ) -> DailyReturns | None:
        """One daily series over consecutive windows, keyed like a continuous run.

        ``key`` selects the series dict key (the market-valued
        ``"mtm_daily_returns"``); every window shares the day grid and the
        chaining. A
        continuous run's series starts the day AFTER its first day (NT's
        Sharpe convention), so the first window drops its day-0 return and
        every later window keeps it (its return vs that window's fresh
        balance): the chain covers exactly the days of one continuous
        backtest -- ``compute_day_count(train) - 1`` values, no boundary day
        lost or duplicated. ``None`` if any window has no series.
        """
        from vibe_quant.metrics import concat_daily_returns

        parts = cast("list[DailyReturns | None]", [r.get(key) for r in results])
        if not parts or parts[0] is None:
            return None
        parts[0] = parts[0].without_first_day()
        return concat_daily_returns(parts)

    @staticmethod
    def per_window_min_trades(min_trades: int, n_windows: int) -> int:
        """Trades each sub-window must have: ``max(1, min_trades // (2*N))``.

        Half the per-window share of the global gate -- a window that barely
        trades has no Sharpe worth taking the minimum of.
        """
        return max(1, int(min_trades) // (2 * max(1, n_windows)))

    @staticmethod
    def _aggregate_multi_window(
        results: list[dict[str, float | int]],
        min_trades: int = 0,
    ) -> dict[str, float | int]:
        """Aggregate per-window metrics as WORST-of-N.

        - every window must have ``per_window_min_trades(min_trades, N)``
          trades, else failure metrics (the strategy doesn't cover that regime)
        - sharpe_ratio: min across windows
        - total_return: min across windows
        - max_drawdown: max across windows
        - profit_factor: min across windows (NaN = no losing period in that
          window; ignored unless every window is NaN)
        - total_trades: sum (feeds the run's global min-trades gate)
        - skewness: mean, kurtosis: max (conservative for DSR);
          trade_returns: concatenated (bootstrap CI over all trades)

        Failure metrics keep the summed trade count (honest logging) but have
        sharpe -1 / return 0, which the fitness hard gate scores as 0.
        """
        import math

        n = len(results)
        per_window_trades = [int(r["total_trades"]) for r in results]
        total_trades_sum = sum(per_window_trades)
        window_min = NTBacktestFn.per_window_min_trades(min_trades, n)

        if any(t < window_min for t in per_window_trades):
            return {
                "sharpe_ratio": -1.0,
                "max_drawdown": 1.0,
                "profit_factor": 0.0,
                "total_trades": total_trades_sum,
                "total_return": 0.0,
                "window_trades": tuple(per_window_trades),  # type: ignore[dict-item]
                **_sum_fee_fields(results),
                **_sum_liquidation_fields(results),
                "starting_balance": float(results[0].get("starting_balance", 0.0)),
                "days": sum(int(r.get("days", 0)) for r in results),
            }

        def _finite_or(value: float, fallback: float) -> float:
            return fallback if math.isnan(value) else value

        sharpes = [_finite_or(float(r["sharpe_ratio"]), 0.0) for r in results]
        returns = [_finite_or(float(r.get("total_return", 0.0)), 0.0) for r in results]
        dds = [_finite_or(float(r["max_drawdown"]), 1.0) for r in results]
        pfs = [float(r["profit_factor"]) for r in results]
        finite_pfs = [p for p in pfs if not math.isnan(p)]
        pf_worst = min(finite_pfs) if finite_pfs else float("nan")

        return {
            "sharpe_ratio": min(sharpes),
            "max_drawdown": max(dds),
            "profit_factor": pf_worst,
            "total_trades": total_trades_sum,
            "total_return": min(returns),
            "skewness": sum(float(r.get("skewness", 0.0)) for r in results) / n,  # type: ignore[arg-type]
            "kurtosis": max(float(r.get("kurtosis", 3.0)) for r in results),  # type: ignore[arg-type]
            "trade_returns": sum(  # type: ignore[dict-item]
                (r.get("trade_returns", ()) for r in results), ()  # type: ignore[arg-type]
            ),
            "window_trades": tuple(per_window_trades),  # type: ignore[dict-item]
            **_sum_fee_fields(results),
            **_sum_liquidation_fields(results),
            "starting_balance": float(results[0].get("starting_balance", 0.0)),
            "days": sum(int(r.get("days", 0)) for r in results),
        }

    @staticmethod
    def _early_exit_result(
        results: list[dict[str, float | int]], window_idx: int, low_trades: bool
    ) -> dict[str, float | int]:
        """Failure metrics (fitness 0) after stopping at window ``window_idx``.

        ``total_trades``/``window_trades`` cover only the windows actually run;
        ``early_exit`` marks the stopping window. For a return <= 0 stop the
        window's real return/Sharpe are kept (progress display; the fitness
        gate zeroes the score); for a trade-gate stop return stays 0 as in
        :meth:`_aggregate_multi_window`.
        """
        per_window_trades = [int(r["total_trades"]) for r in results]
        last = results[window_idx]
        keep = not low_trades
        return {
            "sharpe_ratio": float(last["sharpe_ratio"]) if keep else -1.0,
            "max_drawdown": 1.0,
            "profit_factor": 0.0,
            "total_trades": sum(per_window_trades),
            "total_return": float(last.get("total_return", 0.0)) if keep else 0.0,
            "window_trades": tuple(per_window_trades),  # type: ignore[dict-item]
            "early_exit": window_idx,
            # Fees the windows actually run did pay; a zeroed score must not
            # hide that the signal was killed by costs (bd vibe-quant-2gyk1.4).
            **_sum_fee_fields(results),
            # Report-only: sum over the windows that ran before the exit (R2).
            **_sum_liquidation_fields(results),
            "starting_balance": float(results[0].get("starting_balance", 0.0)),
            "days": sum(int(r.get("days", 0)) for r in results),
        }

    def _eval_symbols(
        self, chromosome: StrategyChromosome, symbols: list[str] | None
    ) -> dict[str, float | int]:
        """Windows-aware evaluation of ``symbols`` (None = all, one portfolio run)."""
        # Portfolio mode keeps the original 3-arg _run_single call shape.
        kw: dict[str, list[str]] = {} if symbols is None else {"symbols": symbols}
        if self.windows and len(self.windows) >= 2:
            n_windows = len(self.windows)
            window_min = self.per_window_min_trades(self.min_trades, n_windows)
            results: list[dict[str, float | int]] = []
            for idx, (ws, we) in enumerate(self.windows):
                res = self._run_single(chromosome, ws, we, **kw)
                results.append(res)
                # Worst-of-N: a window with non-positive (or NaN) return or
                # too few trades pins the aggregate return <= 0 / fails the
                # trade gate, so fitness is exactly 0 whatever the rest do.
                ret = float(res.get("total_return", 0.0))
                low_trades = int(res["total_trades"]) < window_min
                if (low_trades or not (ret > 0)) and idx < n_windows - 1:
                    logger.debug(
                        "Early exit %s at window %d/%d: %s",
                        chromosome.uid,
                        idx,
                        n_windows,
                        "trades below per-window gate" if low_trades else "return <= 0",
                    )
                    return self._early_exit_result(results, idx, low_trades)
            agg = self._aggregate_multi_window(results, self.min_trades)
            if self.collect_daily_returns:
                agg["mtm_daily_returns"] = self._chain_window_series(  # type: ignore[assignment]
                    results, key="mtm_daily_returns"
                )
            return agg
        res = self._run_single(chromosome, self.start_date, self.end_date, **kw)
        if self.collect_daily_returns:
            res["mtm_daily_returns"] = self._chain_window_series(  # type: ignore[assignment]
                [res], key="mtm_daily_returns"
            )
        return res

    @staticmethod
    def _aggregate_symbols(
        results: list[dict[str, float | int]],
        symbols: list[str],
    ) -> dict[str, float | int]:
        """Worst-of-symbols: min sharpe/return/PF (NaN PF ignored), max DD.

        ``total_trades`` is the MIN across symbols (the min-trades gate then
        applies per symbol); trade_returns are concatenated for the bootstrap CI.

        Fee-drag flows (``total_fees``/``gross_pnl``) are SUMMED over symbols, and
        so is ``starting_balance``: each symbol runs as its own backtest on its
        OWN account, so the combined equity behind those fees is N x balance. The
        annualized rate fitness derives (``fees / starting_balance * 365 / days``)
        therefore divides the summed fees by the summed equity -- dividing by a
        single balance would overstate it by the symbol count. ``fee_drag``
        (fees/gross) is unaffected; ``days`` stays the shared window span.
        """
        import math

        n = len(results)
        pfs = [float(r["profit_factor"]) for r in results]
        finite_pfs = [p for p in pfs if not math.isnan(p)]
        sharpes = [float(r["sharpe_ratio"]) for r in results]
        rets = [float(r.get("total_return", 0.0)) for r in results]
        dds = [float(r["max_drawdown"]) for r in results]
        trades = tuple(int(r["total_trades"]) for r in results)
        mtm_collected = any("mtm_daily_returns" in r for r in results)
        mtm_series = [cast("DailyReturns | None", r.get("mtm_daily_returns")) for r in results]
        # The per-symbol series feed only the pools + per-symbol stats below;
        # they must not ride along in symbol_results (memory, bd yul7u.24).
        results = [
            {k: v for k, v in r.items() if k != "mtm_daily_returns"}
            for r in results
        ]
        out: dict[str, float | int] = {
            "sharpe_ratio": min(0.0 if math.isnan(x) else x for x in sharpes),
            "max_drawdown": max(1.0 if math.isnan(x) else x for x in dds),
            "profit_factor": min(finite_pfs) if finite_pfs else float("nan"),
            "total_trades": min(trades),
            "total_return": min(0.0 if math.isnan(x) else x for x in rets),
            "skewness": sum(float(r.get("skewness", 0.0)) for r in results) / n,  # type: ignore[arg-type]
            "kurtosis": max(float(r.get("kurtosis", 3.0)) for r in results),  # type: ignore[arg-type]
            "trade_returns": sum(  # type: ignore[dict-item]
                (r.get("trade_returns", ()) for r in results), ()  # type: ignore[arg-type]
            ),
            "symbol_trades": trades,  # type: ignore[dict-item]
            # Per-symbol breakdown + portfolio-comparable trade count: validation
            # and replay are portfolio runs, so they compare against trades_sum.
            "trades_sum": sum(trades),
            # Each symbol's own aggregate: input of the soft GA score.
            "symbol_results": dict(zip(symbols, results, strict=True)),  # type: ignore[dict-item]
            "symbol_metrics": {  # type: ignore[dict-item]
                sym: {
                    "sharpe": _finite_or_none(r["sharpe_ratio"]),
                    "trades": int(r["total_trades"]),
                    "return": _finite_or_none(r.get("total_return", 0.0)),
                }
                for sym, r in zip(symbols, results, strict=True)
            },
            # Fee-drag flows are SUMMED over symbols (bd vibe-quant-2gyk1.4);
            # days is the shared window span, never summed across symbols.
            **_sum_fee_fields(results),
            # Report-only: summed over symbols, with each symbol's own count kept
            # in ``symbol_results`` (R3).
            **_sum_liquidation_fields(results),
            # Each symbol is its own account, so the summed fees are annualized
            # against the SUMMED equity (bd vibe-quant-2gyk1.4).
            "starting_balance": sum(float(r.get("starting_balance", 0.0)) for r in results),
            "days": max(int(r.get("days", 0)) for r in results),
        }
        synthetic = [
            sym
            for sym, r in zip(symbols, results, strict=True)
            if "early_exit" in r or "error" in r
        ]
        if synthetic:
            # Worst-of metrics inherit forced values from these symbols -> mark
            # the aggregate synthetic so the fitness sanity check skips it.
            out["synthetic_symbols"] = synthetic  # type: ignore[assignment]
        if mtm_collected:
            from vibe_quant.metrics import pool_daily_returns

            # Pooled market-valued series + each symbol's own (worst-mode gate
            # input, bd vibe-quant-wcn55); neither leaks into symbol_results.
            out["mtm_daily_returns"] = pool_daily_returns(mtm_series)  # type: ignore[assignment]
            out["symbol_mtm_daily"] = {  # type: ignore[assignment]
                sym: (sr if sr is not None and len(sr) > 0 else None)
                for sym, sr in zip(symbols, mtm_series, strict=True)
            }
        return out

    def _eval_symbol_windows(
        self, chromosome: StrategyChromosome, symbol: str
    ) -> list[dict[str, float | int]]:
        """Every eval window of one symbol on its own account, no early exit (pooled mode).

        A coin that loses one window still runs the rest: the pool, not the coin, is scored.
        """
        spans = (
            self.windows
            if self.windows and len(self.windows) >= 2
            else [(self.start_date, self.end_date)]
        )
        return [self._run_single(chromosome, ws, we, symbols=[symbol]) for ws, we in spans]

    def eval_symbol(self, chromosome: StrategyChromosome, symbol: str) -> dict[str, float | int]:
        """One symbol of a split evaluation (inline or a worker task).

        Worst mode: the symbol's window-aggregated metrics. Pooled mode: its raw window
        results under ``window_results`` for :meth:`aggregate_symbol_results`.
        """
        if self.symbol_agg == "pooled":
            return {"window_results": self._eval_symbol_windows(chromosome, symbol)}  # type: ignore[dict-item]
        return self._eval_symbols(chromosome, [symbol])

    def aggregate_symbol_results(
        self, chromosome: StrategyChromosome, outcomes: list[dict[str, float | int]]
    ) -> dict[str, float | int]:
        """The ONE reduction of a split evaluation (``__call__`` and the process-pool parent).

        ``outcomes`` are :meth:`eval_symbol` results in ``self.symbols`` order. An aggregate-time
        failure (e.g. misaligned pooled series) becomes this candidate's error result, never an
        exception in the split parent; missing data still fails the run.
        """
        try:
            if self.symbol_agg == "pooled":
                return self._aggregate_pooled(
                    [cast("list[dict[str, float | int]]", o["window_results"]) for o in outcomes],
                    self.symbols,
                )
            return self._aggregate_symbols(outcomes, self.symbols)
        except DataUnavailableError:
            raise
        except Exception as exc:
            return self.error_result(chromosome, exc)

    @staticmethod
    def _pooled_series(
        series: list[DailyReturns | None], symbols: list[str], label: str
    ) -> DailyReturns:
        """Equal-weight pool of aligned series; :class:`PooledSeriesError` names the culprit."""
        from vibe_quant.metrics import pool_daily_returns

        for sym, sr in zip(symbols, series, strict=True):
            if sr is None or len(sr) == 0:
                raise PooledSeriesError(f"pooled series missing ({sym} {label})")
        # The reference grid is the majority's (ties: the earliest first day), so the message
        # names the coin that is off even when it sorts first.
        firsts = [cast("DailyReturns", sr).first_day for sr in series]
        ref_day = min(set(firsts), key=lambda d: (-firsts.count(d), d))
        ref = cast("DailyReturns", series[firsts.index(ref_day)])
        for sym, sr in zip(symbols, series, strict=True):
            sr = cast("DailyReturns", sr)
            if sr.first_day != ref.first_day:
                raise PooledSeriesError(
                    f"pooled series misaligned ({sym} first day {sr.first_day} != {ref.first_day})"
                )
            if len(sr) != len(ref):
                raise PooledSeriesError(
                    f"pooled series misaligned ({sym} days {len(sr)} != {len(ref)})"
                )
        pooled = pool_daily_returns(series)
        if pooled is None:
            raise PooledSeriesError(f"pooled series non-finite ({label})")
        return pooled

    @staticmethod
    def _pooled_window(
        parts: list[dict[str, float | int]], pooled: DailyReturns
    ) -> dict[str, float | int]:
        """One eval window's metrics of the pooled hypothesis (``parts`` in sorted symbol order).

        Sharpe: NT's 252-day Sharpe of the pooled MTM series without day 0 (the screening
        ``mtm_sharpe`` convention); return: its compound INCLUDING day 0; max drawdown: of its
        daily-close equity; PF: one profit factor over every coin's trade PnLs; trades summed.
        Each coin runs on its own account, so balances and fee flows are summed.
        """
        from vibe_quant.metrics import profit_factor
        from vibe_quant.overfitting.dsr import daily_sharpe_inputs
        from vibe_quant.validation.extraction import sharpe_sortino_from_returns

        values = list(pooled.values)
        equity, peak, dd = 1.0, 1.0, 0.0
        for v in values:
            equity *= 1.0 + v
            peak = max(peak, equity)
            dd = max(dd, (peak - equity) / peak)
        body = pooled.without_first_day()
        sharpe = sharpe_sortino_from_returns(
            {(body.first_day + i) * _DAY_NS: v for i, v in enumerate(body.values)}
        )[0]
        _s, skewness, kurtosis = daily_sharpe_inputs(body.values)
        return {
            "sharpe_ratio": sharpe,
            "max_drawdown": min(max(dd, 0.0), 1.0),
            "profit_factor": profit_factor(
                p
                for r in parts
                for p in cast("tuple[float, ...]", r.get("trade_pnls", ()))
            ),
            "total_trades": sum(int(r["total_trades"]) for r in parts),
            "total_return": equity - 1.0,
            "skewness": skewness,
            "kurtosis": kurtosis,
            **_sum_fee_fields(parts),
            **_sum_liquidation_fields(parts),
            "starting_balance": sum(float(r.get("starting_balance", 0.0)) for r in parts),
            "days": max(int(r.get("days", 0)) for r in parts),
        }

    def _aggregate_pooled(
        self,
        per_symbol: list[list[dict[str, float | int]]],
        symbols: list[str],
    ) -> dict[str, float | int]:
        """Pooled symbols (``symbol_agg="pooled"``): N coins scored as ONE hypothesis.

        ``per_symbol[i]`` holds symbol ``symbols[i]``'s raw window results (own account, every
        window). Symbols are sorted ONCE here, so the caller's order never reaches a number.
        Window w is scored on the equal-weight pool of the coins' daily MTM series
        (:meth:`_pooled_window`) and the windows aggregate worst-of-N as everywhere else
        (:meth:`_aggregate_multi_window`).

        Emits the pooled chained series (``mtm_daily_returns``, also the bootstrap input under
        ``trade_returns``: daily values, never per-trade returns), each coin's chained series
        (``symbol_mtm_daily``), ``pooled_symbol_stats`` ``{sym: (return, trades)}`` for the trade
        floor (return = compound of the coin's full window series), ``symbol_metrics`` and
        ``trades_sum``; no ``symbol_results`` (no per-symbol soft score).

        Raises:
            PooledSeriesError: a series is missing or the coins' day grids differ.
        """
        from vibe_quant.validation.extraction import sharpe_sortino_from_returns

        order = sorted(range(len(symbols)), key=lambda i: symbols[i])
        syms = [symbols[i] for i in order]
        wins = [per_symbol[i] for i in order]
        n_windows = len(wins[0])
        window_results = []
        for w in range(n_windows):
            parts = [win[w] for win in wins]
            pooled = self._pooled_series(
                [cast("DailyReturns | None", p.get("mtm_daily_returns")) for p in parts],
                syms,
                f"window {w}",
            )
            window_results.append(self._pooled_window(parts, pooled))
        agg = (
            self._aggregate_multi_window(window_results, self.min_trades)
            if n_windows >= 2
            else window_results[0]
        )
        chained = [self._chain_window_series(win) for win in wins]
        series = self._pooled_series(chained, syms, "chained")
        stats: dict[str, tuple[float, int]] = {}
        metrics: dict[str, dict[str, float | int | None]] = {}
        for sym, win, chain in zip(syms, wins, chained, strict=True):
            equity = 1.0
            for res in win:
                for v in cast("DailyReturns", res["mtm_daily_returns"]).values:
                    equity *= 1.0 + v
            trades = sum(int(r["total_trades"]) for r in win)
            stats[sym] = (equity - 1.0, trades)
            body = cast("DailyReturns", chain)
            sharpe = sharpe_sortino_from_returns(
                {(body.first_day + i) * _DAY_NS: v for i, v in enumerate(body.values)}
            )[0]
            metrics[sym] = {
                "sharpe": _finite_or_none(sharpe),
                "trades": trades,
                "return": _finite_or_none(equity - 1.0),
            }
        agg["mtm_daily_returns"] = series  # type: ignore[assignment]
        agg["trade_returns"] = tuple(series.values)  # type: ignore[assignment]
        agg["symbol_mtm_daily"] = dict(zip(syms, chained, strict=True))  # type: ignore[assignment]
        agg["pooled_symbol_stats"] = stats  # type: ignore[assignment]
        agg["symbol_metrics"] = metrics  # type: ignore[assignment]
        agg["trades_sum"] = sum(t for _r, t in stats.values())
        return agg

    @property
    def splits_by_symbol(self) -> bool:
        """True when each symbol of a chromosome can run as its own parallel task."""
        return self.symbol_agg in ("worst", "pooled") and len(self.symbols) >= 2

    @staticmethod
    def error_result(
        chromosome: StrategyChromosome, exc: Exception | str
    ) -> dict[str, float | int]:
        """Failure metrics of a crashed evaluation (what ``__call__`` returns)."""
        logger.warning("NT backtest failed for chromosome %s: %s", chromosome.uid, exc)
        desc = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
        return {
            "sharpe_ratio": -1.0,
            "max_drawdown": 1.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            # Marker only (numbers unchanged): lets the pipeline tell a
            # crashed evaluation from a genuinely bad strategy.
            "error": desc,  # type: ignore[dict-item]
        }

    def __call__(self, chromosome: StrategyChromosome) -> dict[str, float | int]:
        try:
            if self.splits_by_symbol:
                outcomes = [self.eval_symbol(chromosome, sym) for sym in self.symbols]
                return self.aggregate_symbol_results(chromosome, outcomes)
            return self._eval_symbols(chromosome, None)
        except DataUnavailableError:
            raise  # missing data fails the run loudly, never an 'error' result
        except Exception as exc:
            return self.error_result(chromosome, exc)


def pooled_k_eff(
    symbol_series: Mapping[str, DailyReturns | None],
) -> tuple[float | None, str | None, float | None]:
    """``(k_eff, reason, mean_corr)``: effective number of independent pooled coins.

    ``K_eff = N / (1 + (N - 1) * rho_bar)`` with ``rho_bar`` the mean off-diagonal Pearson
    correlation over pairs of the NON-constant series (sorted symbols, the same aligned days);
    ``N`` counts those series. Reporting only (notes), never a gate. ``k_eff`` is None, with a
    reason, when the series are misaligned, fewer than 2 are non-constant, or
    ``rho_bar <= -1/(N-1)`` (the formula has no positive value).
    """
    import numpy as np

    parts = [symbol_series[sym] for sym in sorted(symbol_series)]
    if not parts or any(p is None or len(p) == 0 for p in parts):
        return None, "symbol series missing", None
    first = cast("DailyReturns", parts[0])
    if any(
        cast("DailyReturns", p).first_day != first.first_day or len(p) != len(first)  # type: ignore[arg-type]
        for p in parts
    ):
        return None, "symbol series misaligned", None
    rows = [np.frombuffer(cast("DailyReturns", p).values, dtype=np.float64) for p in parts]
    centred = [r - r.mean() for r in rows if np.ptp(r) > 0.0]
    n = len(centred)
    if n < 2:
        return None, "fewer than 2 non-constant symbol series", None
    sq = [float(c @ c) for c in centred]
    corrs = [
        float(np.clip(centred[i] @ centred[j] / np.sqrt(sq[i] * sq[j]), -1.0, 1.0))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    rho = sum(corrs) / len(corrs)
    if rho <= -1.0 / (n - 1):
        return None, f"mean correlation {rho:.4f}: K_eff undefined", rho
    return n / (1.0 + (n - 1) * rho), None, rho


def full_range_headline(
    full_range_metrics: dict[str, float | int] | None,
    *,
    fallback_sharpe: float,
    fallback_trades: int,
    fallback_max_dd: float,
    fallback_pf: float,
    fallback_return: float,
) -> dict[str, float | int]:
    """Champion headline metrics measured on ONE continuous backtest over the full range.

    Discovery's multi-window fitness reports min(per-window Sharpe) and
    sum(per-window trades) over ``--eval-windows`` sub-windows; a train/test split
    further restricts fitness to the training slice. Promotion, by contrast,
    replays a single continuous backtest over the whole discovery range, so the
    GA's stored ``sharpe``/``trades`` are a different statistic of the same DSL and
    ``replay_drift`` flags the gap by construction (see bd vibe-quant-1gvyc).

    Persisting a true full-range headline alongside the aggregate lets promotion /
    ``check_replay_drift`` compare like-for-like; the multi-window aggregate stays
    as a labelled robustness signal (bd vibe-quant-rewru).

    Args:
        full_range_metrics: Result of one full-range backtest (the dict
            :class:`NTBacktestFn` returns), or ``None`` when no separate run was
            needed -- i.e. single-window discovery with no train/test split (and
            mock mode), where the GA fitness already IS the continuous full-range
            metric. In that case the ``fallback_*`` values are echoed verbatim.
        fallback_sharpe: GA-fitness Sharpe to fall back on.
        fallback_trades: GA-fitness trade count to fall back on.
        fallback_max_dd: GA-fitness max drawdown to fall back on.
        fallback_pf: GA-fitness profit factor to fall back on.
        fallback_return: GA-fitness total return to fall back on.

    Returns:
        ``{full_range_sharpe, full_range_trades, full_range_max_dd,
        full_range_pf, full_range_return_pct}``.
    """
    if full_range_metrics is not None:
        head: dict[str, float | int] = {
            "full_range_sharpe": float(full_range_metrics["sharpe_ratio"]),
            "full_range_trades": int(full_range_metrics["total_trades"]),
            "full_range_max_dd": float(full_range_metrics["max_drawdown"]),
            "full_range_pf": float(full_range_metrics["profit_factor"]),
            "full_range_return_pct": float(full_range_metrics.get("total_return", 0.0)),
        }
        # symbol_agg="worst": trades above is the MIN symbol; keep the sum
        # (portfolio-comparable) and the per-symbol breakdown.
        if "trades_sum" in full_range_metrics:
            head["full_range_trades_sum"] = int(full_range_metrics["trades_sum"])
            head["full_range_symbol_metrics"] = full_range_metrics["symbol_metrics"]  # type: ignore[assignment]
        # Report-only liquidation count of the continuous full-range run (bead
        # vibe-quant-2gyk1.1.9): present only when the mode is on (R4).
        if "liquidation_count" in full_range_metrics:
            head["full_range_liquidation_count"] = int(full_range_metrics["liquidation_count"])
        return head
    return {
        "full_range_sharpe": float(fallback_sharpe),
        "full_range_trades": int(fallback_trades),
        "full_range_max_dd": float(fallback_max_dd),
        "full_range_pf": float(fallback_pf),
        "full_range_return_pct": float(fallback_return),
    }
