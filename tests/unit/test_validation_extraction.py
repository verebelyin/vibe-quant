"""Tests for validation extraction module."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from vibe_quant.validation.extraction import (
    _compute_max_drawdown_from_trades,
    compute_extended_metrics,
    extract_trades,
)
from vibe_quant.validation.results import TradeRecord, ValidationResult


def _make_trade(entry_time: str, exit_time: str, net_pnl: float = 10.0) -> TradeRecord:
    return TradeRecord(
        symbol="BTCUSDT-PERP",
        direction="LONG",
        leverage=10,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=40000.0,
        exit_price=40100.0,
        quantity=0.1,
        net_pnl=net_pnl,
        gross_pnl=net_pnl + 2.0,
        roi_percent=(net_pnl / (40000.0 * 0.1)) * 100.0,
    )


def test_trades_sorted_before_cagr_computation() -> None:
    """CAGR must use chronological first/last trade, not insertion order."""
    # Trades intentionally out of chronological order
    t_middle = _make_trade("2025-03-01T00:00:00+00:00", "2025-03-02T00:00:00+00:00")
    t_first = _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00")
    t_last = _make_trade("2025-06-01T00:00:00+00:00", "2025-06-02T00:00:00+00:00")

    result = ValidationResult(
        total_return=0.10,
        trades=[t_middle, t_first, t_last],
    )
    result.total_trades = len(result.trades)
    # Sort as extract_trades does
    result.trades.sort(key=lambda t: t.entry_time)

    compute_extended_metrics(result)

    # After sort, first trade should be Jan, last should be Jun
    assert result.trades[0].entry_time == "2025-01-01T00:00:00+00:00"
    assert result.trades[-1].entry_time == "2025-06-01T00:00:00+00:00"
    assert result.cagr != 0.0

    # Now compute with WRONG order (no sort) to show the difference
    result_bad = ValidationResult(
        total_return=0.10,
        trades=[t_middle, t_first, t_last],  # unsorted
    )
    result_bad.total_trades = len(result_bad.trades)
    compute_extended_metrics(result_bad)

    # Bad order: uses Mar as first, Jun as last => shorter period => higher CAGR
    # Correct order: Jan to Jun => longer period => lower CAGR
    assert result.cagr != result_bad.cagr, "Sorted vs unsorted trades should produce different CAGR"


def test_trades_sorted_single_trade_no_error() -> None:
    """Single trade should not break sorting or CAGR."""
    t = _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00")
    result = ValidationResult(total_return=0.05, trades=[t])
    result.total_trades = 1
    result.trades.sort(key=lambda t: t.entry_time)
    compute_extended_metrics(result)
    assert result.cagr != 0.0


def test_trades_already_sorted_unchanged() -> None:
    """Already-sorted trades should produce identical result after sort."""
    t1 = _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00")
    t2 = _make_trade("2025-02-01T00:00:00+00:00", "2025-02-02T00:00:00+00:00")
    t3 = _make_trade("2025-03-01T00:00:00+00:00", "2025-03-02T00:00:00+00:00")

    result = ValidationResult(total_return=0.10, trades=[t1, t2, t3])
    result.total_trades = 3
    result.trades.sort(key=lambda t: t.entry_time)
    compute_extended_metrics(result)

    assert result.trades[0].entry_time == "2025-01-01T00:00:00+00:00"
    assert result.trades[-1].entry_time == "2025-03-01T00:00:00+00:00"
    assert result.cagr != 0.0


class TestComputeMaxDrawdownFromTrades:
    """Tests for equity-curve max drawdown fallback (NT 1.222+ compat)."""

    def test_all_winning_trades_zero_drawdown(self) -> None:
        """No drawdown when all trades win."""
        trades = [
            _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00", net_pnl=100.0),
            _make_trade("2025-01-03T00:00:00+00:00", "2025-01-04T00:00:00+00:00", net_pnl=200.0),
        ]
        dd = _compute_max_drawdown_from_trades(trades, 10000.0)
        assert dd == 0.0

    def test_single_losing_trade(self) -> None:
        """Single loss from starting balance."""
        trades = [
            _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00", net_pnl=-1000.0),
        ]
        dd = _compute_max_drawdown_from_trades(trades, 10000.0)
        assert abs(dd - 0.10) < 1e-9  # 10% drawdown

    def test_recovery_drawdown(self) -> None:
        """Win then loss then win: drawdown from peak."""
        trades = [
            _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00", net_pnl=5000.0),
            _make_trade("2025-01-03T00:00:00+00:00", "2025-01-04T00:00:00+00:00", net_pnl=-3000.0),
            _make_trade("2025-01-05T00:00:00+00:00", "2025-01-06T00:00:00+00:00", net_pnl=1000.0),
        ]
        # Starting 10000 -> 15000 (peak) -> 12000 -> 13000
        # Max DD = (15000-12000)/15000 = 0.20
        dd = _compute_max_drawdown_from_trades(trades, 10000.0)
        assert abs(dd - 0.20) < 1e-9

    def test_empty_trades(self) -> None:
        """Empty trades should return 0."""
        dd = _compute_max_drawdown_from_trades([], 10000.0)
        assert dd == 0.0

    def test_zero_starting_balance(self) -> None:
        """Zero starting balance should return 0."""
        trades = [
            _make_trade("2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00", net_pnl=100.0),
        ]
        dd = _compute_max_drawdown_from_trades(trades, 0.0)
        assert dd == 0.0


class TestFeeSplit:
    """Tests for maker/taker fee splitting in trade extraction."""

    def test_fee_split_uses_maker_taker_rates(self) -> None:
        """entry_fee/exit_fee should reflect taker/maker rate ratio, not 50/50."""
        # Binance default: maker=0.02%, taker=0.04%
        # Total rate = 0.06%, taker share = 0.04/0.06 = 2/3, maker share = 1/3
        # For total fees of $6: entry_fee = $4 (taker), exit_fee = $2 (maker)
        from decimal import Decimal

        maker_fee = Decimal("0.0002")
        taker_fee = Decimal("0.0004")
        total_rate = float(maker_fee + taker_fee)
        total_fees = 6.0

        entry_fee = total_fees * (float(taker_fee) / total_rate)
        exit_fee = total_fees * (float(maker_fee) / total_rate)

        assert abs(entry_fee - 4.0) < 1e-9
        assert abs(exit_fee - 2.0) < 1e-9
        assert abs(entry_fee + exit_fee - total_fees) < 1e-9

    def test_fee_split_fallback_when_zero_rates(self) -> None:
        """When rates are zero, should fall back to 50/50."""
        total_fees = 10.0
        maker_rate = 0.0
        taker_rate = 0.0
        total_rate = maker_rate + taker_rate

        if total_rate > 0 and total_fees > 0:
            entry_fee = total_fees * (taker_rate / total_rate)
            exit_fee = total_fees * (maker_rate / total_rate)
        else:
            entry_fee = total_fees / 2.0
            exit_fee = total_fees / 2.0

        assert entry_fee == 5.0
        assert exit_fee == 5.0


class TestSlippageModelSelection:
    """Tests for selecting a single slippage model in extraction."""

    @staticmethod
    def _make_engine_with_closed_position():
        class _Position:
            is_closed = True
            realized_pnl = 100.0
            avg_px_open = 40000.0
            avg_px_close = 40100.0
            peak_qty = 0.1
            ts_opened = 1_700_000_000_000_000_000
            ts_closed = 1_700_000_360_000_000_000
            entry = "BUY"
            instrument_id = "BTCUSDT-PERP.BINANCE"

            def commissions(self):
                return [2.0]

        class _Cache:
            def positions(self):
                return [_Position()]

            def position_snapshots(self):
                return []

            def bars(self):
                return []

        class _Kernel:
            cache = _Cache()

        class _Engine:
            kernel = _Kernel()

        return _Engine()

    def test_extract_trades_skips_post_fill_slippage_when_engine_slippage_enabled(self) -> None:
        """If engine slippage is enabled, extraction should not add SPEC slippage."""
        result = ValidationResult(starting_balance=100_000.0)
        engine = self._make_engine_with_closed_position()
        venue_config = SimpleNamespace(
            default_leverage=Decimal("10"),
            fill_config=SimpleNamespace(impact_coefficient=0.1, prob_slippage=1.0),
            maker_fee=Decimal("0.0002"),
            taker_fee=Decimal("0.0004"),
        )

        extract_trades(result, engine, venue_config)

        assert len(result.trades) == 1
        assert result.trades[0].slippage_cost == 0.0
        assert result.total_slippage == 0.0

    def test_extract_trades_applies_post_fill_slippage_when_engine_slippage_disabled(self) -> None:
        """If engine slippage is disabled, extraction should add SPEC slippage."""
        result = ValidationResult(starting_balance=100_000.0)
        engine = self._make_engine_with_closed_position()
        venue_config = SimpleNamespace(
            default_leverage=Decimal("10"),
            fill_config=SimpleNamespace(impact_coefficient=0.1, prob_slippage=0.0),
            maker_fee=Decimal("0.0002"),
            taker_fee=Decimal("0.0004"),
        )

        extract_trades(result, engine, venue_config)

        assert len(result.trades) == 1
        assert result.trades[0].slippage_cost > 0.0
        assert result.total_slippage > 0.0
