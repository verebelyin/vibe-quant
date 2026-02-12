"""Tests for validation extraction module."""

from __future__ import annotations

from vibe_quant.validation.extraction import compute_extended_metrics
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
    assert result.cagr != result_bad.cagr, (
        "Sorted vs unsorted trades should produce different CAGR"
    )


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
