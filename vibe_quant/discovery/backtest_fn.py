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
    returns worst-case metrics across all windows, forcing the GA to
    find regime-robust strategies.
    """

    def __init__(
        self,
        symbols: list[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        windows: list[tuple[str, str]] | None = None,
    ) -> None:
        self.symbols = symbols
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.windows = windows

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
    def _aggregate_multi_window(
        results: list[dict[str, float | int]],
    ) -> dict[str, float | int]:
        """Aggregate metrics across multiple window results.

        Strategy: require ALL windows to produce trades. If any window
        has 0 trades, return failure metrics. Otherwise:
        - total_trades: sum (statistical significance across all data)
        - sharpe_ratio: mean (consistent performance)
        - max_drawdown: max (worst case)
        - profit_factor: trade-weighted mean
        - total_return: mean per-window return
        """
        n = len(results)
        per_window_trades = [int(r["total_trades"]) for r in results]

        # If any window has 0 trades, strategy doesn't cover that regime
        if any(t == 0 for t in per_window_trades):
            return {
                "sharpe_ratio": -1.0,
                "max_drawdown": 1.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "total_return": 0.0,
            }

        total_trades_sum = sum(per_window_trades)

        # Trade-weighted profit factor
        pf_weighted = sum(
            float(r["profit_factor"]) * t
            for r, t in zip(results, per_window_trades)
        ) / total_trades_sum

        return {
            "sharpe_ratio": sum(float(r["sharpe_ratio"]) for r in results) / n,
            "max_drawdown": max(float(r["max_drawdown"]) for r in results),
            "profit_factor": pf_weighted,
            "total_trades": total_trades_sum,
            "total_return": sum(float(r["total_return"]) for r in results) / n,
            "skewness": sum(float(r.get("skewness", 0.0)) for r in results) / n,  # type: ignore[arg-type]
            "kurtosis": max(float(r.get("kurtosis", 3.0)) for r in results),  # type: ignore[arg-type]
            "trade_returns": sum(
                (r.get("trade_returns", ()) for r in results), ()  # type: ignore[arg-type]
            ),
        }

    def __call__(self, chromosome: StrategyChromosome) -> dict[str, float | int]:
        try:
            if self.windows and len(self.windows) >= 2:
                results = [
                    self._run_single(chromosome, ws, we)
                    for ws, we in self.windows
                ]
                return self._aggregate_multi_window(results)
            return self._run_single(chromosome, self.start_date, self.end_date)
        except Exception as exc:
            logger.warning("NT backtest failed for chromosome %s: %s", chromosome.uid, exc)
            return {
                "sharpe_ratio": -1.0,
                "max_drawdown": 1.0,
                "profit_factor": 0.0,
                "total_trades": 0,
            }
