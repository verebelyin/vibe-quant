"""Picklable backtest callable for ProcessPoolExecutor (bd-cm14).

Lives in its own module — NOT in ``vibe_quant.discovery.__main__`` —
because ``python -m vibe_quant.discovery`` loads ``__main__.py`` as
``__main__``, and worker processes can't unpickle classes whose
``__module__`` is ``__main__`` (their own ``__main__`` is the
multiprocessing worker entrypoint, not the discovery CLI).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe_quant.discovery.operators import StrategyChromosome

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        symbols: list[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        windows: list[tuple[str, str]] | None = None,
        min_trades: int = 0,
    ) -> None:
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.windows = windows
        # Global min-trades gate of the run; drives the per-window gate.
        self.min_trades = min_trades

    def _run_single(
        self,
        chromosome: StrategyChromosome,
        start_date: str,
        end_date: str,
    ) -> dict[str, float | int]:
        from vibe_quant.discovery.genome import chromosome_to_dsl
        from vibe_quant.screening.nt_runner import NTScreeningRunner

        dsl_dict = chromosome_to_dsl(chromosome)
        dsl_dict["timeframe"] = self.timeframe

        runner = NTScreeningRunner(
            dsl_dict=dsl_dict,
            symbols=self.symbols,
            start_date=start_date,
            end_date=end_date,
        )
        result = runner({})

        return {
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
        }

    def __call__(self, chromosome: StrategyChromosome) -> dict[str, float | int]:
        try:
            if self.windows and len(self.windows) >= 2:
                results = [
                    self._run_single(chromosome, ws, we)
                    for ws, we in self.windows
                ]
                return self._aggregate_multi_window(results, self.min_trades)
            return self._run_single(chromosome, self.start_date, self.end_date)
        except Exception as exc:
            logger.warning("NT backtest failed for chromosome %s: %s", chromosome.uid, exc)
            return {
                "sharpe_ratio": -1.0,
                "max_drawdown": 1.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                # Marker only (numbers unchanged): lets the pipeline tell a
                # crashed evaluation from a genuinely bad strategy.
                "error": f"{type(exc).__name__}: {exc}",  # type: ignore[dict-item]
            }


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
        return {
            "full_range_sharpe": float(full_range_metrics["sharpe_ratio"]),
            "full_range_trades": int(full_range_metrics["total_trades"]),
            "full_range_max_dd": float(full_range_metrics["max_drawdown"]),
            "full_range_pf": float(full_range_metrics["profit_factor"]),
            "full_range_return_pct": float(full_range_metrics.get("total_return", 0.0)),
        }
    return {
        "full_range_sharpe": float(fallback_sharpe),
        "full_range_trades": int(fallback_trades),
        "full_range_max_dd": float(fallback_max_dd),
        "full_range_pf": float(fallback_pf),
        "full_range_return_pct": float(fallback_return),
    }
