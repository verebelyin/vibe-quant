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


def _daily_bars(n: int, price: float = 100.0) -> list[OHLCBar]:
    """One bar per UTC day (ts in ms) so daily-return Sharpe has a series."""
    return [
        OHLCBar(ts=1_735_689_600_000 + i * 86_400_000, open=price, high=price, low=price,
                close=price)
        for i in range(n)
    ]


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
        metrics = _compute_metrics([], taker_fee=0.0005, bars=_daily_bars(5))
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
        metrics = _compute_metrics(trades, taker_fee=0.0005, bars=_daily_bars(30))
        assert metrics.win_rate == 0.9
        assert metrics.total_return > 0
        assert metrics.sharpe > 0
        assert metrics.profit_factor > 1.0

    def test_all_winners_pf_capped(self):
        """All-winning trades produce the finite PF cap, not 0 and not inf (e70tl.7)."""
        from math import isfinite

        from vibe_quant.metrics import PROFIT_FACTOR_CAP
        from vibe_quant.validation.random_baseline import TradeResult

        trades = [
            TradeResult(entry_idx=i, exit_idx=i + 1, entry_price=100, exit_price=95, pnl_pct=5.0, hit_tp=True, hit_sl=False)
            for i in range(10)
        ]
        metrics = _compute_metrics(trades, taker_fee=0.0, bars=_daily_bars(15))
        assert isfinite(metrics.profit_factor)
        assert metrics.profit_factor == PROFIT_FACTOR_CAP


class TestRunRandomShortBaseline:
    def test_basic_run(self):
        """Smoke test — should produce valid result structure."""
        # Create 500 bars of trending-down data
        rng = np.random.default_rng(123)
        prices = 100.0 + np.cumsum(rng.normal(-0.01, 0.1, 500))
        prices = np.maximum(prices, 50.0)  # Floor at 50
        # Hourly bars (~21 days): the daily-return Sharpe needs several days
        bars = [
            OHLCBar(ts=i * 3_600_000, open=p, high=p + 0.2, low=p - 0.2, close=p)
            for i, p in enumerate(prices.tolist())
        ]

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


class TestAnnualizedSharpeAndPValue:
    """Baseline Sharpe = champion's annualized daily Sharpe, not a t-stat (e70tl.23)."""

    def test_sharpe_matches_daily_balance_statistic(self):
        from vibe_quant.validation.extraction import daily_sharpe_sortino
        from vibe_quant.validation.random_baseline import TradeResult

        bars = _daily_bars(10)
        pnls = [2.0, -1.0, 3.0, -0.5]
        trades = [
            TradeResult(entry_idx=2 * i, exit_idx=2 * i + 1, entry_price=100, exit_price=100,
                        pnl_pct=p, hit_tp=p > 0, hit_sl=p < 0)
            for i, p in enumerate(pnls)
        ]
        metrics = _compute_metrics(trades, taker_fee=0.0, bars=bars)
        equity, events = 1.0, []
        for i, p in enumerate(pnls):
            events.append((bars[2 * i + 1].ts * 1_000_000, equity * p / 100.0))
            equity *= 1 + p / 100.0
        expected = daily_sharpe_sortino(
            1.0, events, bars[0].ts * 1_000_000, (bars[-1].ts + 86_400_000) * 1_000_000
        )
        assert metrics.sharpe == pytest.approx(expected[0], rel=1e-12)
        assert metrics.sortino == pytest.approx(expected[1], rel=1e-12)
        # old t-stat: mean/std*sqrt(n) over trades
        t_stat = np.mean(pnls) / np.std(pnls, ddof=1) * np.sqrt(len(pnls))
        assert metrics.sharpe != pytest.approx(t_stat)

    def test_p_value(self):
        rng = np.random.default_rng(1)
        prices = 100.0 + np.cumsum(rng.normal(-0.01, 0.1, 3000))
        bars = [OHLCBar(ts=i * 3_600_000, open=p, high=p + 0.2, low=p - 0.2, close=p)
                for i, p in enumerate(np.maximum(prices, 50.0))]
        result = run_random_short_baseline(
            bars, BaselineConfig(sl_pct=1.0, tp_pct=2.0, target_trades=30), n_simulations=40,
            seed=3,
        )
        sharpes = sorted(m.sharpe for m in result.metrics)
        assert result.p_value(float("inf")) == pytest.approx(1 / 41)
        assert result.p_value(float("-inf")) == 1.0
        assert result.p_value(sharpes[-5]) == pytest.approx((1 + 5) / 41)
        assert "p-value" in result.summary(champion_sharpe=sharpes[-1])
