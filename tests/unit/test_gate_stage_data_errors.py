"""Gate stages must fail the run loudly on DataUnavailableError.

Missing bars / missing aux data for a gate-only window is a run-level data
problem, not a candidate quality signal: it must propagate out of run()
instead of being swallowed as a rejection reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline
from vibe_quant.screening.nt_runner import MissingBarDataError

if TYPE_CHECKING:
    from vibe_quant.discovery.operators import StrategyChromosome


def _ga_backtest(chrom: StrategyChromosome) -> dict[str, Any]:
    """GA-phase mock that clears guardrails (mirrors test_discovery_pipeline)."""
    import random as _rng

    n_genes = len(chrom.entry_genes) + len(chrom.exit_genes)
    total_return = 0.5 + n_genes * 0.05
    total_trades = 150
    r = _rng.Random(n_genes)
    mean_ret = total_return / total_trades
    trade_returns = tuple(r.gauss(mean_ret, mean_ret * 0.3) for _ in range(total_trades))
    return {
        "sharpe_ratio": 2.0 + n_genes * 0.1,
        "max_drawdown": 0.08,
        "profit_factor": 2.0,
        "total_trades": total_trades,
        "total_return": total_return,
        "trade_returns": trade_returns,
    }


def _missing_bars(_chrom: StrategyChromosome) -> dict[str, Any]:
    raise MissingBarDataError(
        "No catalog bars for BTCUSDT-1h-LAST-EXTERNAL in window 2024-03-16..2024-06-01"
    )


def _crash(_chrom: StrategyChromosome) -> dict[str, Any]:
    raise RuntimeError("boom")


def _factory(fn: Any) -> Any:
    """backtest_fn_factory that ignores the window and returns fn."""

    def factory(_start: str, _end: str) -> Any:
        return fn

    return factory


def _config(**overrides: Any) -> DiscoveryConfig:
    defaults: dict[str, Any] = {
        "population_size": 6,
        "max_generations": 2,
        "elite_count": 1,
        "top_k": 2,
        "min_trades": 50,
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-03-16",
        "train_test_split": 0.5,
        "holdout_start_date": "2024-03-16",
        "holdout_end_date": "2024-06-01",
        "require_dsr": False,
        "max_workers": 1,
    }
    defaults.update(overrides)
    return DiscoveryConfig(**defaults)


class TestHoldoutGateDataErrors:
    def test_holdout_missing_bar_data_fails_the_run(self) -> None:
        pipe = DiscoveryPipeline(_config(), _ga_backtest, holdout_backtest_fn=_missing_bars)
        with pytest.raises(MissingBarDataError):
            pipe.run()

    def test_holdout_generic_error_still_rejects_with_reason(self) -> None:
        pipe = DiscoveryPipeline(_config(), _ga_backtest, holdout_backtest_fn=_crash)
        result = pipe.run()
        assert result.top_strategies == []
        holdout_rejects = [r for r in result.guardrail_rejections if r["stage"] == "holdout"]
        assert holdout_rejects
        assert all(r["reasons"] for r in holdout_rejects)


class TestCrossWindowGateDataErrors:
    def test_cross_window_missing_bar_data_fails_the_run(self) -> None:
        pipe = DiscoveryPipeline(
            _config(cross_window_months=[1]),
            _ga_backtest,
            holdout_backtest_fn=_ga_backtest,
            backtest_fn_factory=_factory(_missing_bars),
        )
        with pytest.raises(MissingBarDataError):
            pipe.run()


class TestWFAGateDataErrors:
    def test_wfa_missing_bar_data_fails_the_run(self) -> None:
        pipe = DiscoveryPipeline(
            _config(wfa_oos_step_days=30, wfa_min_consistency=0.5),
            _ga_backtest,
            holdout_backtest_fn=_ga_backtest,
            backtest_fn_factory=_factory(_missing_bars),
        )
        with pytest.raises(MissingBarDataError):
            pipe.run()
