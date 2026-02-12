"""Mock backtest runner for testing overfitting pipelines.

Provides synthetic (deterministic) results for WFA and Purged K-Fold CV
without requiring real NautilusTrader backtests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_quant.overfitting.purged_kfold import FoldResult

if TYPE_CHECKING:
    from datetime import date


class MockBacktestRunner:
    """Mock backtest runner for testing pipeline without real backtests."""

    def __init__(self, oos_sharpe: float = 1.0, oos_return: float = 10.0) -> None:
        """Initialize mock runner with default OOS metrics."""
        self._oos_sharpe = oos_sharpe
        self._oos_return = oos_return

    def optimize(
        self,
        strategy_id: str,
        start_date: date,
        end_date: date,
        param_grid: dict[str, list[object]],
    ) -> tuple[dict[str, object], float, float]:
        """Return mock optimization result."""
        params: dict[str, object] = {k: v[0] for k, v in param_grid.items()}
        return params, self._oos_sharpe * 1.2, self._oos_return * 1.1

    def backtest(
        self,
        strategy_id: str,
        start_date: date,
        end_date: date,
        params: dict[str, object],
    ) -> tuple[float, float]:
        """Return mock backtest result."""
        return self._oos_sharpe, self._oos_return

    def run(self, train_indices: list[int], test_indices: list[int]) -> FoldResult:
        """Return mock fold result for purged k-fold."""
        return FoldResult(
            fold_index=0,
            train_size=len(train_indices),
            test_size=len(test_indices),
            train_sharpe=self._oos_sharpe * 1.2,
            test_sharpe=self._oos_sharpe,
            train_return=self._oos_return * 1.1,
            test_return=self._oos_return,
        )
