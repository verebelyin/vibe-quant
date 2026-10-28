"""Funding in screening/discovery + interior archive holes (bd vibe-quant-e70tl.20).

Screening reported ``total_funding = 0.0`` (GA ranked with no carry cost), and
FundingCalculator only checked the archive's END POINTS: a trade spanning an
interior hole (BTC 2026-02-23 -> 03-10) was silently charged 0 for the hole.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import TYPE_CHECKING, Any

import pytest
from nautilus_trader.model.enums import OrderSide

from tests.unit.fake_engine import FakePosition, fake_bt_result, fake_engine, fake_venue
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.metrics import daily_balance_returns, profit_factor
from vibe_quant.screening.nt_runner import NTScreeningRunner
from vibe_quant.validation.funding import (
    DEFAULT_FUNDING_RATE_PER_PERIOD,
    FundingArchiveUnavailableError,
    FundingCalculator,
    clear_rate_cache,
)

if TYPE_CHECKING:
    from pathlib import Path

_HOUR_NS = 3_600 * 1_000_000_000
_DAY_NS = 24 * _HOUR_NS
_BASE_MS = 1_735_689_600_000  # 2025-01-01T00:00:00Z
_BASE_NS = _BASE_MS * 1_000_000
_RATE = 0.0003  # 3x default so archived vs fallback is provable


def _archive(path: Path, settlements: list[int], symbol: str = "BTCUSDT") -> Path:
    """Archive with ``_RATE`` at the given 8h settlement indices from _BASE."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE raw_funding_rates (
            id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
            funding_rate REAL NOT NULL, mark_price REAL, source TEXT,
            UNIQUE(symbol, funding_time))"""
    )
    for i in settlements:
        # Binance funding_time carries a few ms of jitter after the boundary
        conn.execute(
            "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate) VALUES (?,?,?)",
            (symbol, _BASE_MS + i * 8 * 3_600_000 + (i % 3) * 4, _RATE),
        )
    conn.commit()
    conn.close()
    return path


def empty_archive(path: Path) -> Path:
    """A real archive with no funding rows: 'no funding data' without a missing file.

    A MISSING archive file fails closed (FundingArchiveUnavailableError); a symbol the
    archive does not hold is charged the fallback rate, which is what these tests want.
    """
    archive = RawDataArchive(path)
    try:
        archive.conn.commit()
    finally:
        archive.close()
    return path


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_rate_cache()


class TestInteriorHoles:
    def test_interior_hole_charges_fallback_and_flags(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Settlements 3..5 (day 1) missing inside the archive range."""
        calc = FundingCalculator(_archive(tmp_path / "a.db", [0, 1, 2, 6, 7, 8, 9]))
        with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.funding"):
            acc = calc.accrue("BTCUSDT-PERP.BINANCE", "LONG", 10_000.0,
                              _BASE_NS + _HOUR_NS, _BASE_NS + 3 * _DAY_NS - _HOUR_NS)
        # crossed boundaries: 1,2 (archived) 3,4,5 (hole) 6,7,8 (archived)
        archived = 5 * _RATE * 10_000.0
        fallback = 3 * DEFAULT_FUNDING_RATE_PER_PERIOD * 10_000.0
        assert acc.total == pytest.approx(archived + fallback)
        assert acc.fallback_settlements == 3
        assert "hole" in caplog.text.lower() or "no archived funding" in caplog.text.lower()
        # Old endpoint-only check would have charged 0 for the hole
        assert acc.total > archived

    def test_missing_archive_file_is_not_created(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope" / "raw.db"
        calc = FundingCalculator(missing)
        with pytest.raises(FundingArchiveUnavailableError, match="raw.db"):
            calc.accrue("BTCUSDT-PERP.BINANCE", "LONG", 10_000.0, _BASE_NS + 1,
                        _BASE_NS + _DAY_NS)
        assert not missing.exists()

    def test_rates_cached_per_process(self, tmp_path: Path) -> None:
        path = _archive(tmp_path / "a.db", list(range(6)))
        FundingCalculator(path).accrue("BTCUSDT-PERP.BINANCE", "LONG", 1.0, _BASE_NS + 1,
                                       _BASE_NS + _DAY_NS)
        path.unlink()  # a second calculator must not need the file again
        acc = FundingCalculator(path).accrue("BTCUSDT-PERP.BINANCE", "LONG", 1.0,
                                             _BASE_NS + 1, _BASE_NS + _DAY_NS)
        assert acc.fallback_settlements == 0


# ---------------------------------------------------------------------------
# Screening applies funding
# ---------------------------------------------------------------------------


def _bt(n: int) -> Any:
    return fake_bt_result(
        n,
        stats_pnls={"USDT": {"PnL% (total)": 2.0}},  # +2% realized
        stats_returns={"Sharpe Ratio (252 days)": 1.0},
    )


def _runner(archive: Path) -> NTScreeningRunner:
    return NTScreeningRunner(
        dsl_dict={}, symbols=["BTCUSDT"], start_date="2025-01-01", end_date="2025-01-11",
        funding_archive_path=str(archive),
    )


def _positions() -> list[FakePosition]:
    """0.1 BTC at 40000: a long that pays 3 settlements and a short that receives 2."""
    return [
        FakePosition(
            id="L",
            instrument_id="BTCUSDT-PERP.BINANCE",
            ts_opened=_BASE_NS + _HOUR_NS,
            ts_closed=_BASE_NS + 25 * _HOUR_NS,
            events=[],
            realized_pnl=20.0,
            entry=OrderSide.BUY,
            avg_px_open=40_000.0,
            avg_px_close=40_400.0,
            peak_qty=0.1,
            commission_list=[4.0],
        ),
        FakePosition(
            id="S",
            instrument_id="BTCUSDT-PERP.BINANCE",
            ts_opened=_BASE_NS + 2 * _DAY_NS + _HOUR_NS,
            ts_closed=_BASE_NS + 2 * _DAY_NS + 17 * _HOUR_NS,
            events=[],
            realized_pnl=-5.0,
            entry=OrderSide.SELL,
            avg_px_open=40_000.0,
            avg_px_close=40_400.0,
            peak_qty=0.1,
            commission_list=[4.0],
        ),
    ]


def test_screening_charges_funding_like_validation_calculator(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "a.db", list(range(30)))
    positions = _positions()
    metrics = _runner(archive)._extract_metrics(
        {}, _bt(2), fake_engine(positions), time.time(), starting_balance=1000.0
    )
    calc = FundingCalculator(archive)
    expected = [
        calc.compute_funding("BTCUSDT-PERP.BINANCE", "LONG", 4000.0, p.ts_opened, p.ts_closed)
        if p.entry == OrderSide.BUY else
        calc.compute_funding("BTCUSDT-PERP.BINANCE", "SHORT", 4000.0, p.ts_opened, p.ts_closed)
        for p in positions
    ]
    assert expected == [pytest.approx(3 * _RATE * 4000.0), pytest.approx(-2 * _RATE * 4000.0)]
    assert metrics.total_funding == pytest.approx(sum(expected))
    assert metrics.total_return == pytest.approx(0.02 - sum(expected) / 1000.0)
    net = [p.realized_pnl - f for p, f in zip(positions, expected, strict=True)]
    assert metrics.profit_factor == pytest.approx(profit_factor(net))
    assert metrics.funding_fallback_settlements == 0


def test_screening_missing_funding_recorded_not_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    archive = _archive(tmp_path / "a.db", [0, 1])  # ends before both trades finish
    with caplog.at_level(logging.WARNING):
        metrics = _runner(archive)._extract_metrics(
            {}, _bt(2), fake_engine(_positions()), time.time(), starting_balance=1000.0
        )
    # long crosses settlements 1 (archived), 2, 3; short crosses 7, 8 -> 4 fallbacks
    assert metrics.funding_fallback_settlements == 4
    assert metrics.total_funding != 0.0
    assert "funding" in caplog.text.lower()


def test_daily_balance_returns_include_funding_and_span_window() -> None:
    """Daily realized-balance returns: events booked on their UTC day, window-padded."""
    start = _BASE_NS
    end = _BASE_NS + 4 * _DAY_NS
    rets = daily_balance_returns(
        1000.0,
        [(_BASE_NS + _DAY_NS + 5, 10.0), (_BASE_NS + _DAY_NS + 6, -2.0)],
        start,
        end,
    )
    days = sorted(rets)
    assert days == [_BASE_NS + _DAY_NS, _BASE_NS + 2 * _DAY_NS, _BASE_NS + 3 * _DAY_NS]
    assert rets[days[0]] == pytest.approx(8.0 / 1000.0)
    assert rets[days[1]] == 0.0  # padded to the window end, not cut at last event
    assert rets[days[2]] == 0.0


def test_screening_and_validation_agree_on_funding_return_pf_sharpe(tmp_path: Path) -> None:
    """Same trades + archive -> identical funding, return, PF and Sharpe in both tiers."""
    from vibe_quant.validation.extraction import extract_results

    archive = _archive(tmp_path / "a.db", [0, 1, 2, 4, 5, 6, 7, 8, 9])  # hole at 3
    positions = _positions()
    screening = _runner(archive)._extract_metrics(
        {}, _bt(2), fake_engine(positions), time.time(), starting_balance=1000.0
    )
    venue = fake_venue(slippage=1.0, balance=1000)
    validation = extract_results(
        1, "s", _bt(2), fake_engine(positions), venue,
        funding_calculator=FundingCalculator(archive),
        run_start_date="2025-01-01", run_end_date="2025-01-11",
    )
    assert validation.total_funding == pytest.approx(screening.total_funding)
    assert validation.funding_fallback_settlements == screening.funding_fallback_settlements
    assert screening.funding_fallback_settlements > 0
    assert validation.total_return == pytest.approx(screening.total_return)
    assert validation.profit_factor == pytest.approx(screening.profit_factor)
    assert validation.sharpe_ratio == pytest.approx(screening.sharpe_ratio)
    assert screening.sharpe_ratio != 1.0  # recomputed, not NT's funding-blind value
