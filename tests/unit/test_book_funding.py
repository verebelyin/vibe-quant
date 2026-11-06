"""Funding for a resized portfolio leg.

A cross-sectional book resizes NETTING legs every week, so per-position
funding on ``peak_qty x avg_px_open`` is wrong: each settlement must be paid on
the signed quantity actually held just before it, at that settlement's mark.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from vibe_quant.validation.funding import (
    DEFAULT_FUNDING_RATE_PER_PERIOD,
    FundingCalculator,
    clear_rate_cache,
)

if TYPE_CHECKING:
    from pathlib import Path

_HOUR_NS = 3_600 * 1_000_000_000
_BASE_MS = 1_735_689_600_000  # 2025-01-01T00:00:00Z
_BASE_NS = _BASE_MS * 1_000_000
_INST = "BTCUSDT-PERP.BINANCE"


def _settle_ns(i: int) -> int:
    return _BASE_NS + i * 8 * _HOUR_NS


def _archive(path: Path, rates: dict[int, float], symbol: str = "BTCUSDT") -> FundingCalculator:
    """Archive with ``rates[i]`` at 8h settlement ``i`` from _BASE (ms jitter like Binance)."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE raw_funding_rates (
            id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
            funding_rate REAL NOT NULL, mark_price REAL, source TEXT,
            UNIQUE(symbol, funding_time))"""
    )
    for i, rate in rates.items():
        conn.execute(
            "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate) VALUES (?,?,?)",
            (symbol, _BASE_MS + i * 8 * 3_600_000 + 3, rate),
        )
    conn.commit()
    conn.close()
    return FundingCalculator(path)


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_rate_cache()


def _mark(prices: dict[int, float]):  # noqa: ANN202 - test helper
    """Mark price by settlement index (exact lookup; KeyError = test bug)."""
    by_ns = {_settle_ns(i): p for i, p in prices.items()}

    def mark(ts_ns: int) -> float:
        return by_ns[(ts_ns - _BASE_NS) // (8 * _HOUR_NS) * 8 * _HOUR_NS + _BASE_NS]

    return mark


def test_resized_leg_pays_on_quantity_held_before_each_settlement(tmp_path: Path) -> None:
    calc = _archive(tmp_path / "a.db", {1: 0.001, 2: 0.002})
    fills = [(_BASE_NS + _HOUR_NS, 1.0), (_settle_ns(1) + _HOUR_NS, 2.0)]  # 1 -> 3 after settlement 1
    acc = calc.accrue_book(_INST, fills, _mark({1: 100.0, 2: 110.0}), end_ns=_settle_ns(2) + _HOUR_NS)
    # settlement 1: 1 x 100 x 0.001 = 0.1 ; settlement 2: 3 x 110 x 0.002 = 0.66
    assert acc.payments == [(_settle_ns(1) + 3_000_000, pytest.approx(0.1)),
                            (_settle_ns(2) + 3_000_000, pytest.approx(0.66))]
    assert acc.total == pytest.approx(0.76)
    assert acc.fallback_settlements == 0


def test_short_leg_receives_positive_rate(tmp_path: Path) -> None:
    calc = _archive(tmp_path / "a.db", {1: 0.001})
    acc = calc.accrue_book(_INST, [(_BASE_NS, -2.0)], _mark({1: 100.0}), end_ns=_settle_ns(1) + _HOUR_NS)
    assert acc.total == pytest.approx(-0.2)  # negative = received


def test_fill_at_or_after_settlement_is_not_charged_for_it(tmp_path: Path) -> None:
    calc = _archive(tmp_path / "a.db", {1: 0.001, 2: 0.001})
    # opens 1 ms AFTER settlement 1's archived time: only settlement 2 is owed
    fills = [(_settle_ns(1) + 3_000_000 + 1_000_000, 5.0)]
    acc = calc.accrue_book(_INST, fills, _mark({2: 100.0}), end_ns=_settle_ns(2) + _HOUR_NS)
    assert [ts for ts, _ in acc.payments] == [_settle_ns(2) + 3_000_000]
    assert acc.total == pytest.approx(5 * 100.0 * 0.001)


def test_flat_leg_pays_nothing_and_hole_fallback_counts_only_while_held(tmp_path: Path) -> None:
    # archive has settlement 1 only; 2 and 3 are a hole. Leg closes before 3.
    calc = _archive(tmp_path / "a.db", {1: 0.001})
    fills = [(_BASE_NS, 1.0), (_settle_ns(2) + _HOUR_NS, -1.0)]
    acc = calc.accrue_book(_INST, fills, _mark({1: 100.0, 2: 100.0}), end_ns=_settle_ns(3) + _HOUR_NS)
    assert acc.fallback_settlements == 1  # settlement 2 held and uncovered; 3 is flat
    assert acc.total == pytest.approx(0.1 + DEFAULT_FUNDING_RATE_PER_PERIOD * 100.0)


def test_constant_leg_matches_per_trade_accrual_at_entry_mark(tmp_path: Path) -> None:
    """With mark == entry price and no resize, book funding equals the existing per-trade path."""
    calc = _archive(tmp_path / "a.db", {i: 0.0003 for i in range(1, 6)})
    entry, end = _BASE_NS + _HOUR_NS, _settle_ns(5) + _HOUR_NS
    book = calc.accrue_book(_INST, [(entry, 0.5)], lambda _ts: 20_000.0, end_ns=end)
    trade = calc.accrue(_INST, "LONG", 0.5 * 20_000.0, entry, end)
    assert book.total == pytest.approx(trade.total, rel=0, abs=1e-12)
    assert len(book.payments) == len(trade.payments) == 5


def test_fill_exactly_at_settlement_time_is_not_charged_for_it(tmp_path: Path) -> None:
    calc = _archive(tmp_path / "a.db", {1: 0.001, 2: 0.001})
    # 1 held before settlement 1; resized +4 at the same ns as its archived time
    fills = [(_BASE_NS, 1.0), (_settle_ns(1) + 3_000_000, 4.0)]
    acc = calc.accrue_book(_INST, fills, _mark({1: 100.0, 2: 100.0}), end_ns=_settle_ns(2) + _HOUR_NS)
    assert [amount for _, amount in acc.payments] == [pytest.approx(0.1), pytest.approx(0.5)]
