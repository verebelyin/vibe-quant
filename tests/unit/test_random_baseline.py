"""Tests for random short entry baseline."""

from __future__ import annotations

import numpy as np
import pytest

from vibe_quant.validation.random_baseline import (
    BaselineConfig,
    OHLCBar,
    _compute_metrics,
    _simulate_single_run,
    run_random_short_baseline,
)


def _make_bars(prices: list[float], spread: float = 0.5) -> list[OHLCBar]:
    """Create synthetic bars from close prices with fixed spread."""
    bars = []
    for i, p in enumerate(prices):
        bars.append(
            OHLCBar(
                ts=i * 60000,
                open=p,
                high=p + spread,
                low=p - spread,
                close=p,
            )
        )
    return bars


class TestSimulateSingleRun:
    def test_tp_hit(self):
        """SHORT trade hits TP when price drops enough."""
        # Entry at 100, TP at 5% → exit at 95
        prices = [100.0, 99.0, 97.0, 94.0, 93.0]
        bars = _make_bars(prices, spread=0.5)
        entries = np.array([0])
        trades = _simulate_single_run(bars, entries, sl_pct=10.0, tp_pct=5.0, taker_fee=0.0)
        assert len(trades) == 1
        assert trades[0].hit_tp is True
        assert trades[0].hit_sl is False
        assert trades[0].exit_price == pytest.approx(95.0, abs=0.01)

    def test_sl_hit(self):
        """SHORT trade hits SL when price rises enough."""
        # Entry at 100, SL at 2% → exit at 102
        prices = [100.0, 101.0, 102.5, 103.0]
        bars = _make_bars(prices, spread=0.1)
        entries = np.array([0])
        trades = _simulate_single_run(bars, entries, sl_pct=2.0, tp_pct=20.0, taker_fee=0.0)
        assert len(trades) == 1
        assert trades[0].hit_sl is True
        assert trades[0].hit_tp is False

    def test_non_overlapping(self):
        """Trades should not overlap — next entry must be after previous exit."""
        # Two entries at idx 0 and 1, but trade from idx 0 takes multiple bars
        prices = [100.0, 99.0, 98.0, 97.0, 96.0, 94.0]
        bars = _make_bars(prices, spread=0.1)
        entries = np.array([0, 1, 2])  # Try 3 consecutive entries
        trades = _simulate_single_run(bars, entries, sl_pct=10.0, tp_pct=5.0, taker_fee=0.0)
        # Only 1 trade should happen — TP hit at ~95 blocks entries at 1 and 2
        assert len(trades) == 1

    def test_fees_reduce_pnl(self):
        """Fees should reduce trade PnL."""
        prices = [100.0, 94.0]  # TP hit instantly
        bars = _make_bars(prices, spread=0.1)
        entries = np.array([0])

        # Without fees
        trades_nofee = _simulate_single_run(bars, entries, sl_pct=10.0, tp_pct=5.0, taker_fee=0.0)
        # With fees
        trades_fee = _simulate_single_run(bars, entries, sl_pct=10.0, tp_pct=5.0, taker_fee=0.001)

        assert trades_fee[0].pnl_pct < trades_nofee[0].pnl_pct


class TestComputeMetrics:
    def test_empty_trades(self):
        metrics = _compute_metrics([], taker_fee=0.0005)
        assert metrics.total_trades == 0
        assert metrics.sharpe == 0.0

    def test_mostly_winners(self):
        """Mostly winning trades should produce positive metrics."""
        from vibe_quant.validation.random_baseline import TradeResult

        trades = [
            TradeResult(entry_idx=i, exit_idx=i + 1, entry_price=100, exit_price=95, pnl_pct=4.0 + i * 0.1, hit_tp=True, hit_sl=False)
            for i in range(18)
        ] + [
            TradeResult(entry_idx=18, exit_idx=19, entry_price=100, exit_price=101, pnl_pct=-1.1, hit_tp=False, hit_sl=True),
            TradeResult(entry_idx=20, exit_idx=21, entry_price=100, exit_price=101, pnl_pct=-1.1, hit_tp=False, hit_sl=True),
        ]
        metrics = _compute_metrics(trades, taker_fee=0.0005)
        assert metrics.win_rate == 0.9
        assert metrics.total_return > 0
        assert metrics.sharpe > 0
        assert metrics.profit_factor > 1.0


class TestRunRandomShortBaseline:
    def test_basic_run(self):
        """Smoke test — should produce valid result structure."""
        # Create 500 bars of trending-down data
        rng = np.random.default_rng(123)
        prices = 100.0 + np.cumsum(rng.normal(-0.01, 0.1, 500))
        prices = np.maximum(prices, 50.0)  # Floor at 50
        bars = _make_bars(prices.tolist(), spread=0.2)

        config = BaselineConfig(sl_pct=2.0, tp_pct=3.0, target_trades=20)
        result = run_random_short_baseline(bars, config, n_simulations=50, seed=99)

        assert result.n_simulations == 50
        assert result.n_bars == 500
        assert len(result.metrics) == 50
        assert result.sharpe_mean != 0.0  # Should have some signal
        assert 0.0 <= result.pct_sharpe_above_1 <= 1.0
        assert 0.0 <= result.pct_sharpe_above_2 <= 1.0

    def test_summary_does_not_crash(self):
        """summary() should return a non-empty string."""
        rng = np.random.default_rng(456)
        prices = 100.0 + np.cumsum(rng.normal(-0.01, 0.1, 200))
        prices = np.maximum(prices, 50.0)
        bars = _make_bars(prices.tolist(), spread=0.2)

        config = BaselineConfig(sl_pct=1.0, tp_pct=5.0, target_trades=10)
        result = run_random_short_baseline(bars, config, n_simulations=10, seed=42)
        summary = result.summary()
        assert len(summary) > 100
        assert "VERDICT" in summary
