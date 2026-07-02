"""Tests for validation cost accounting.

Covers beads:
- vibe-quant-2b9cp: modeled slippage charged into net PnL + headline return
- vibe-quant-ctg86: post-hoc funding accrual from archived rates
- vibe-quant-lp1vb: CAGR over run window, not trade span
"""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from vibe_quant.validation.extraction import compute_extended_metrics, extract_trades
from vibe_quant.validation.funding import (
    DEFAULT_FUNDING_RATE_PER_PERIOD,
    FundingCalculator,
)
from vibe_quant.validation.results import TradeRecord, ValidationResult

_HOUR_NS = 3_600 * 1_000_000_000


# =============================================================================
# FundingCalculator
# =============================================================================


class TestFundingCalculator:
    @pytest.fixture
    def archive_path(self, tmp_path: Path) -> Path:
        """Archive with 0.03%/8h rates (3x default, so archive path is provable)."""
        path = tmp_path / "archive.db"
        conn = sqlite3.connect(str(path))
        conn.execute("""
            CREATE TABLE raw_funding_rates (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                funding_time INTEGER NOT NULL,
                funding_rate REAL NOT NULL,
                mark_price REAL,
                source TEXT,
                UNIQUE(symbol, funding_time)
            )
        """)
        # Settlements every 8h starting 2025-01-01T00:00Z (ms timestamps)
        base_ms = 1_735_689_600_000  # 2025-01-01T00:00:00Z
        for i in range(9):  # 3 days of 8h settlements
            conn.execute(
                "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate)"
                " VALUES (?, ?, ?)",
                ("BTCUSDT", base_ms + i * 8 * 3_600_000, 0.0003),
            )
        conn.commit()
        conn.close()
        return path

    def test_symbol_from_instrument_id(self) -> None:
        assert (
            FundingCalculator.symbol_from_instrument_id("BTCUSDT-PERP.BINANCE") == "BTCUSDT"
        )

    def test_long_pays_three_settlements(self, archive_path: Path) -> None:
        calc = FundingCalculator(archive_path)
        base_ns = 1_735_689_600_000 * 1_000_000  # 2025-01-01T00:00Z
        entry_ns = base_ns + 1 * _HOUR_NS  # 01:00
        exit_ns = base_ns + 25 * _HOUR_NS  # next day 01:00 -> 08/16/00 = 3 settlements
        funding = calc.compute_funding(
            "BTCUSDT-PERP.BINANCE", "LONG", 10_000.0, entry_ns, exit_ns
        )
        assert funding == pytest.approx(3 * 0.0003 * 10_000.0)

    def test_short_receives(self, archive_path: Path) -> None:
        calc = FundingCalculator(archive_path)
        base_ns = 1_735_689_600_000 * 1_000_000
        funding = calc.compute_funding(
            "BTCUSDT-PERP.BINANCE",
            "SHORT",
            10_000.0,
            base_ns + 1 * _HOUR_NS,
            base_ns + 25 * _HOUR_NS,
        )
        assert funding == pytest.approx(-3 * 0.0003 * 10_000.0)

    def test_intra_period_hold_pays_nothing(self, archive_path: Path) -> None:
        calc = FundingCalculator(archive_path)
        base_ns = 1_735_689_600_000 * 1_000_000
        funding = calc.compute_funding(
            "BTCUSDT-PERP.BINANCE",
            "LONG",
            10_000.0,
            base_ns + 1 * _HOUR_NS,
            base_ns + 2 * _HOUR_NS,
        )
        assert funding == 0.0

    def test_fallback_default_rate_when_no_archive(self, tmp_path: Path) -> None:
        calc = FundingCalculator(tmp_path / "missing.db")
        base_ns = 1_735_689_600_000 * 1_000_000
        funding = calc.compute_funding(
            "ETHUSDT-PERP.BINANCE",
            "LONG",
            10_000.0,
            base_ns + 1 * _HOUR_NS,
            base_ns + 25 * _HOUR_NS,
        )
        assert funding == pytest.approx(3 * DEFAULT_FUNDING_RATE_PER_PERIOD * 10_000.0)

    def test_open_trade_no_funding(self, archive_path: Path) -> None:
        calc = FundingCalculator(archive_path)
        assert calc.compute_funding("BTCUSDT-PERP.BINANCE", "LONG", 1.0, 1, None) == 0.0


# =============================================================================
# Slippage + funding charged into net PnL and headline return
# =============================================================================


def _make_engine_with_closed_position() -> SimpleNamespace:
    base_ns = 1_735_689_600_000 * 1_000_000  # 2025-01-01T00:00Z

    class _Position:
        is_closed = True
        realized_pnl = 100.0
        avg_px_open = 40000.0
        avg_px_close = 40100.0
        peak_qty = 0.1
        ts_opened = base_ns + 1 * _HOUR_NS
        ts_closed = base_ns + 25 * _HOUR_NS
        entry = "BUY"
        instrument_id = "BTCUSDT-PERP.BINANCE"

        def commissions(self) -> list[float]:
            return [2.0]

    cache = SimpleNamespace(
        positions=lambda: [_Position()],
        position_snapshots=list,
        bars=list,
    )
    return SimpleNamespace(kernel=SimpleNamespace(cache=cache))


def _venue_config(prob_slippage: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        default_leverage=Decimal("10"),
        fill_config=SimpleNamespace(impact_coefficient=0.1, prob_slippage=prob_slippage),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0005"),
    )


class TestCostsInNetPnl:
    def test_slippage_reduces_net_pnl_and_headline_return(self) -> None:
        result = ValidationResult(starting_balance=100_000.0, total_return=0.10)
        engine = _make_engine_with_closed_position()

        extract_trades(result, engine, _venue_config(prob_slippage=0.0))

        trade = result.trades[0]
        assert trade.slippage_cost > 0.0
        assert trade.net_pnl == pytest.approx(100.0 - trade.slippage_cost)
        # Headline return charged by total_slippage / starting_balance
        assert result.total_return == pytest.approx(
            0.10 - result.total_slippage / 100_000.0
        )

    def test_funding_charged_when_calculator_provided(self, tmp_path: Path) -> None:
        result = ValidationResult(starting_balance=100_000.0, total_return=0.10)
        engine = _make_engine_with_closed_position()
        calc = FundingCalculator(tmp_path / "missing.db")  # default-rate fallback

        extract_trades(
            result, engine, _venue_config(prob_slippage=1.0), funding_calculator=calc
        )

        trade = result.trades[0]
        # 24h hold -> 3 settlements at default 0.01% on 4000 notional
        expected_funding = 3 * DEFAULT_FUNDING_RATE_PER_PERIOD * 4000.0
        assert trade.funding_fees == pytest.approx(expected_funding)
        assert result.total_funding == pytest.approx(expected_funding)
        assert trade.net_pnl == pytest.approx(100.0 - expected_funding)
        assert result.total_return == pytest.approx(0.10 - expected_funding / 100_000.0)

    def test_win_loss_counts_use_net_pnl(self, tmp_path: Path) -> None:
        """A trade pushed negative by costs must count as a loss."""

        class _BigCostCalc(FundingCalculator):
            def compute_funding(self, *args: object, **kwargs: object) -> float:
                return 500.0  # exceeds the 100.0 realized PnL

        result = ValidationResult(starting_balance=100_000.0)
        engine = _make_engine_with_closed_position()
        extract_trades(
            result,
            engine,
            _venue_config(prob_slippage=1.0),
            funding_calculator=_BigCostCalc(tmp_path / "missing.db"),
        )
        assert result.winning_trades == 0
        assert result.losing_trades == 1


# =============================================================================
# CAGR over run window
# =============================================================================


def _make_trade(entry: str, exit_: str, net_pnl: float = 10.0) -> TradeRecord:
    return TradeRecord(
        symbol="BTCUSDT-PERP",
        direction="LONG",
        leverage=10,
        entry_time=entry,
        exit_time=exit_,
        entry_price=40000.0,
        exit_price=40100.0,
        quantity=0.1,
        net_pnl=net_pnl,
        gross_pnl=net_pnl + 2.0,
        roi_percent=(net_pnl / 4000.0) * 100.0,
    )


class TestCagrRunWindow:
    def test_run_window_beats_trade_span(self) -> None:
        """One 2-day trade inside a 1-year run must annualize over the year."""
        trade = _make_trade("2025-06-01T00:00:00+00:00", "2025-06-03T00:00:00+00:00")

        by_window = ValidationResult(total_return=0.10, trades=[trade])
        by_window.total_trades = 1
        compute_extended_metrics(
            by_window, run_start_date="2025-01-01", run_end_date="2026-01-01"
        )
        # 10% over 365 days -> CAGR ~= 10%
        assert by_window.cagr == pytest.approx(0.10, rel=0.01)

        by_span = ValidationResult(total_return=0.10, trades=[trade])
        by_span.total_trades = 1
        compute_extended_metrics(by_span)
        # 10% over 2 days -> absurdly inflated CAGR (the old behavior)
        assert by_span.cagr > 10.0

    def test_invalid_window_falls_back_to_trade_span(self) -> None:
        trade = _make_trade("2025-06-01T00:00:00+00:00", "2025-06-03T00:00:00+00:00")
        result = ValidationResult(total_return=0.10, trades=[trade])
        result.total_trades = 1
        compute_extended_metrics(result, run_start_date="garbage", run_end_date="2026-01-01")
        assert result.cagr > 10.0  # trade-span fallback
