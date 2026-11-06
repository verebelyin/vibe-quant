"""Reference simulator: the shared rules with no NautilusTrader (task T5).

Contract rows R10, R11, R14c, R14d, R15, R15b, R15c and R8b as they flow through
``simulate_reference`` (``data/swarm/20261015-funding-carry/design.md`` §5,
§8-T5). The fixtures are synthetic two-symbol runs (BTCUSDT + ETHUSDT) built by
hand: the ticks are sized so every number below is derivable, and the archive
carries a single nonzero rate so the funding payment is a round number.

Hand numbers (D1 = 2024-06-03T00:05Z, D2 = D1 + 7 d, E0 = 10,000):

* week 1 (D1): equity 10,000; signals tie at 0.0 -> BTC long (+0.5), ETH short
  (-0.5); ``floor(0.5 x 10,000 / (51,000 x 0.001)) = 98`` steps BTC and
  ``floor(5,000 / (3,000 x 0.001)) = 1666`` steps ETH (R11). Both orders fill at
  the first tick after D1, ``D1 + 59.999 s``, at that tick's price (BTC 51,010)
  for the NT taker fee ``qty x price x 0.0005``: 2.49949 / 2.499 (R15).
* the BTC leg pays one settlement of ``0.098 x 51,000 x 0.0001 = 0.4998`` while
  the ETH short receives ``-1.666 x 3,000 x 0.0002 = -0.9996`` (R16), so the
  week-2 equity is ``10,000 - (0.098x51,010 + 2.49949) + (1.666x3,000 - 2.499)
  - (0.4998 - 0.9996) + (0.098x51,000 - 1.666x3,000) = 9994.52131`` (R10).
* week 2 (D2): the signals order unchanged, but the smaller equity sizes 97 /
  1665 steps -- one step inside the 0.10 band, so both resizes are skipped and
  no week-2 order is sent (R14d).

A second fixture walks three weeks: week 2 flips the book (close before open, at
one tick, R14c) and week 3's funding hole leaves one rankable symbol, so the book
goes flat (R8b). Two further fixtures pin R15b (stale release) and R15c (no tick
after the decision -> unreleased).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vibe_quant.data.catalog import create_instrument
from vibe_quant.data.mark_bars import MarkSeries
from vibe_quant.portfolio.ledger import CarryFill, FundingPayment
from vibe_quant.portfolio.reference import simulate_reference
from vibe_quant.portfolio.signal import Settlements
from vibe_quant.portfolio.sizing import InstrumentSpec, OrderIntent
from vibe_quant.portfolio.spec import (
    PRICE_WINDOW_NS,
    CarryExecutionError,
    CarrySpec,
    decision_times,
)
from vibe_quant.portfolio.ticks import CarryInputs, PriceTick
from vibe_quant.validation.funding import FundingCalculator, clear_rate_cache

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

MIN = 60_000_000_000
HOUR = 60 * MIN
DAY = 86_400 * 10**9
WEEK = 7 * DAY
INIT = 59_999_000_000  # 1m catalog bar ts_init = open + 59.999 s
_JITTER_MS = 3  # archived funding_time carries a few ms of jitter after the boundary
_JITTER_NS = _JITTER_MS * 1_000_000
_SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _ns(date: str) -> int:
    """``YYYY-MM-DD`` 00:00 UTC -> epoch ns."""
    return int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp()) * 10**9


def _boundaries(start_ns: int, end_ns: int) -> list[int]:
    """Every 8h UTC funding boundary in ``[start_ns, end_ns)``."""
    first = start_ns - (start_ns % (8 * HOUR))
    return list(range(first, end_ns, 8 * HOUR))


def _marks(price: float, start_ns: int, end_ns: int) -> MarkSeries:
    """A flat 1m MARK series (open = close = ``price``) covering the run."""
    ts = np.arange(start_ns, end_ns, MIN, dtype=np.int64)
    values = np.full(ts.shape, price, dtype=np.float64)
    return MarkSeries(ts=ts, low=values, high=values, close=values, open=values)


def _tick_pair(decision_ns: int, sizing: float, fill: float) -> tuple[PriceTick, PriceTick]:
    """A symbol's sizing tick (``D - 1 ms``) and fill tick (``D + 59.999 s``)."""
    return PriceTick(decision_ns - 1_000_000, sizing), PriceTick(decision_ns + INIT, fill)


def _priced_ticks(
    decisions: Sequence[int],
    sizing: Mapping[str, float],
    fill: Mapping[str, float],
) -> dict[str, tuple[PriceTick, ...]]:
    return {
        symbol: tuple(
            tick
            for decision_ns in decisions
            for tick in _tick_pair(decision_ns, sizing[symbol], fill[symbol])
        )
        for symbol in sizing
    }


def _archive(
    path: Path,
    boundaries: Sequence[int],
    rates: Mapping[str, Mapping[int, float]],
    hole: tuple[str, int] | None,
) -> None:
    """Write ``rates[symbol][boundary]`` (0.0 default) as archived funding rows."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE raw_funding_rates (
            id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
            funding_rate REAL NOT NULL, mark_price REAL, source TEXT,
            UNIQUE(symbol, funding_time))"""
    )
    for symbol, by_boundary in rates.items():
        for boundary in boundaries:
            if hole is not None and symbol == hole[0] and boundary == hole[1]:
                continue
            conn.execute(
                "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate) VALUES (?,?,?)",
                (symbol, boundary // 1_000_000 + _JITTER_MS, by_boundary.get(boundary, 0.0)),
            )
    conn.commit()
    conn.close()


def _add_archive_row(path: Path, symbol: str, boundary_ns: int, rate: float) -> None:
    """Append one archived funding row outside ``_inputs``' own boundary window."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate) VALUES (?,?,?)",
        (symbol, boundary_ns // 1_000_000 + _JITTER_MS, rate),
    )
    conn.commit()
    conn.close()
    clear_rate_cache()


def _inputs(
    tmp_path: Path,
    start_ns: int,
    end_ns: int,
    ticks: Mapping[str, Sequence[PriceTick]],
    rates: Mapping[str, Mapping[int, float]],
    mark_price: Mapping[str, float],
    hole: tuple[str, int] | None = None,
) -> CarryInputs:
    """A two-symbol synthetic ``CarryInputs`` (signal series mirrored in the archive)."""
    clear_rate_cache()
    spec = CarrySpec(symbols=_SYMBOLS, start_ns=start_ns, end_ns=end_ns)
    instruments = {symbol: create_instrument(symbol) for symbol in spec.symbols}
    specs = {symbol: InstrumentSpec.from_instrument(instruments[symbol]) for symbol in spec.symbols}
    boundaries = _boundaries(start_ns - 12 * DAY, end_ns)
    path = tmp_path / "raw_data.db"
    _archive(path, boundaries, rates, hole)
    settlements: dict[str, Settlements] = {}
    for symbol, by_boundary in rates.items():
        kept = [
            boundary
            for boundary in boundaries
            if not (hole is not None and symbol == hole[0] and boundary == hole[1])
        ]
        settlements[symbol] = Settlements(
            times_ns=tuple(boundary + _JITTER_NS for boundary in kept),
            rates=tuple(by_boundary.get(boundary, 0.0) for boundary in kept),
        )
    marks = {
        symbol: _marks(mark_price[symbol], start_ns - 8 * HOUR, end_ns + HOUR)
        for symbol in spec.symbols
    }
    return CarryInputs(
        spec=spec,
        decisions=tuple(decision_times(start_ns, end_ns)),
        instruments=instruments,
        specs=specs,
        ticks={symbol: tuple(ticks[symbol]) for symbol in spec.symbols},
        settlements=settlements,
        marks=marks,
        calc=FundingCalculator(path),
        boundary_prices={},
    )


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_rate_cache()


def test_two_week_fixture_hand_numbers(tmp_path: Path) -> None:
    """Two weeks of the shared rules, hand-computed."""
    start = _ns("2024-06-03")
    end = _ns("2024-06-17")
    d1 = start + 5 * MIN
    d2 = start + WEEK + 5 * MIN
    settlement = start + 8 * HOUR  # Monday 08:00, inside (D1, D2) and the D2 signal window
    inputs = _inputs(
        tmp_path,
        start,
        end,
        _priced_ticks(
            [d1, d2],
            {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0},
            {"BTCUSDT": 51010.0, "ETHUSDT": 3000.0},
        ),
        {"BTCUSDT": {settlement: 0.0001}, "ETHUSDT": {settlement: 0.0002}},
        {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0},
    )

    run = simulate_reference(inputs)

    assert run.engine == "reference"
    assert run.spec is inputs.spec
    assert [decision.ts for decision in run.decisions] == [d1, d2]

    first, second = run.decisions
    assert first.equity == 10000.0
    assert first.signals == {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
    assert first.weights == {"BTCUSDT": 0.5, "ETHUSDT": -0.5}
    assert first.target_steps == {"BTCUSDT": 98, "ETHUSDT": -1666}  # Floor
    assert first.orders == (
        OrderIntent("BTCUSDT", 98, reduce_only=False),
        OrderIntent("ETHUSDT", -1666, reduce_only=False),
    )

    # Filled at the release tick (D1 + 59.999 s) at that tick's price, NT fee.
    assert run.fills == (
        CarryFill("BTCUSDT", d1 + INIT, 98, 0.098, 51010.0, 2.49949, False),
        CarryFill("ETHUSDT", d1 + INIT, -1666, -1.666, 3000.0, 2.499, False),
    )

    # The week-1 fill cash and the funding payment land in the week-2 equity.
    assert second.equity == 9994.521309999998
    assert second.signals == {"BTCUSDT": 0.0001, "ETHUSDT": 0.0002}
    assert second.weights == {"BTCUSDT": 0.5, "ETHUSDT": -0.5}
    assert second.target_steps == {"BTCUSDT": 97, "ETHUSDT": -1665}
    assert second.band_skips == 2  # 98->97 and -1666->-1665 are inside the 0.10 band
    assert second.orders == ()

    assert [payment for payment in run.payments if payment.amount != 0.0] == [
        FundingPayment("BTCUSDT", settlement + _JITTER_NS, 0.4998, False),
        FundingPayment("ETHUSDT", settlement + _JITTER_NS, -0.9996, False),
    ]
    assert not any(payment.fallback for payment in run.payments)
    assert run.final_steps == {"BTCUSDT": 98, "ETHUSDT": -1666}


def test_flip_and_flat_week(tmp_path: Path) -> None:
    """Week 2 flips the book at one tick; week 3's hole flattens it."""
    start = _ns("2024-06-03")
    end = start + 3 * WEEK
    d1 = start + 5 * MIN
    d2 = start + WEEK + 5 * MIN
    d3 = start + 2 * WEEK + 5 * MIN
    hole = start + WEEK + 8 * HOUR  # a hole in ETH's week-3 lookback window
    inputs = _inputs(
        tmp_path,
        start,
        end,
        _priced_ticks(
            [d1, d2, d3],
            {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0},
            {"BTCUSDT": 51010.0, "ETHUSDT": 3000.0},
        ),
        {"BTCUSDT": {start + 8 * HOUR: 0.0001}, "ETHUSDT": {}},
        {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0},
        hole=("ETHUSDT", hole),
    )

    run = simulate_reference(inputs)

    first, second, third = run.decisions
    assert first.weights == {"BTCUSDT": 0.5, "ETHUSDT": -0.5}
    assert first.orders == (
        OrderIntent("BTCUSDT", 98, reduce_only=False),
        OrderIntent("ETHUSDT", -1666, reduce_only=False),
    )

    # A flip is a reduce-only close then an open, same symbol, same release tick.
    assert second.signals == {"BTCUSDT": 0.0001, "ETHUSDT": 0.0}
    assert second.weights == {"BTCUSDT": -0.5, "ETHUSDT": 0.5}
    assert second.target_steps == {"BTCUSDT": -97, "ETHUSDT": 1665}
    assert second.orders == (
        OrderIntent("BTCUSDT", -98, reduce_only=True),
        OrderIntent("BTCUSDT", -97, reduce_only=False),
        OrderIntent("ETHUSDT", 1666, reduce_only=True),
        OrderIntent("ETHUSDT", 1665, reduce_only=False),
    )
    assert run.fills[2:6] == (
        CarryFill("BTCUSDT", d2 + INIT, -98, -0.098, 51010.0, 2.49949, True),
        CarryFill("BTCUSDT", d2 + INIT, -97, -0.097, 51010.0, 2.473985, False),
        CarryFill("ETHUSDT", d2 + INIT, 1666, 1.666, 3000.0, 2.499, True),
        CarryFill("ETHUSDT", d2 + INIT, 1665, 1.665, 3000.0, 2.4975, False),
    )

    # One rankable symbol -> all weights 0 -> the held legs are closed.
    assert third.signals == {"BTCUSDT": 0.0, "ETHUSDT": None}
    assert third.excluded == {"ETHUSDT": "funding_hole"}
    assert third.weights == {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
    assert third.target_steps == {"BTCUSDT": 0, "ETHUSDT": 0}
    assert third.orders == (
        OrderIntent("BTCUSDT", 97, reduce_only=True),
        OrderIntent("ETHUSDT", -1665, reduce_only=True),
    )
    assert run.final_steps == {"BTCUSDT": 0, "ETHUSDT": 0}
    assert [payment.amount for payment in run.payments if payment.fallback] == [0.4995]


def test_no_price_is_excluded_with_none_signal(tmp_path: Path) -> None:
    """A rankable symbol without a sizing tick gets ``no_price`` and signal None."""
    start = _ns("2024-06-03")
    end = start + WEEK
    d1 = start + 5 * MIN
    inputs = _inputs(
        tmp_path,
        start,
        end,
        {
            # BTC has no tick in [D-60 min, D), only a fill tick: unpriceable, not held.
            "BTCUSDT": (PriceTick(d1 + INIT, 51000.0),),
            "ETHUSDT": (PriceTick(d1 - 1_000_000, 3000.0), PriceTick(d1 + INIT, 3000.0)),
        },
        {"BTCUSDT": {}, "ETHUSDT": {}},
        {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0},
    )

    decision = simulate_reference(inputs).decisions[0]

    assert list(decision.signals) == ["BTCUSDT", "ETHUSDT"]  # keys in sorted-symbol order
    assert decision.signals == {"BTCUSDT": None, "ETHUSDT": 0.0}
    assert decision.excluded == {"BTCUSDT": "no_price"}
    assert decision.weights == {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
    assert decision.orders == ()


def test_stale_release_fails(tmp_path: Path) -> None:
    """The first own tick past the 60-minute window fails closed."""
    start = _ns("2024-06-03")
    end = start + WEEK
    d1 = start + 5 * MIN
    inputs = _inputs(
        tmp_path,
        start,
        end,
        {
            "BTCUSDT": (
                PriceTick(d1 - 1_000_000, 51000.0),
                PriceTick(d1 + PRICE_WINDOW_NS + 1, 51000.0),
            ),
            "ETHUSDT": (PriceTick(d1 - 1_000_000, 3000.0), PriceTick(d1 + INIT, 3000.0)),
        },
        {"BTCUSDT": {}, "ETHUSDT": {}},
        {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0},
    )

    with pytest.raises(CarryExecutionError, match="stale release BTCUSDT"):
        simulate_reference(inputs)


def test_unreleased_order_fails(tmp_path: Path) -> None:
    """An order with no own tick after the decision can never be released."""
    start = _ns("2024-06-03")
    end = start + WEEK
    d1 = start + 5 * MIN
    inputs = _inputs(
        tmp_path,
        start,
        end,
        {
            "BTCUSDT": (PriceTick(d1 - 1_000_000, 51000.0),),
            "ETHUSDT": (PriceTick(d1 - 1_000_000, 3000.0), PriceTick(d1 + INIT, 3000.0)),
        },
        {"BTCUSDT": {}, "ETHUSDT": {}},
        {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0},
    )

    with pytest.raises(CarryExecutionError, match="unreleased orders"):
        simulate_reference(inputs)


def test_funding_after_last_decision(tmp_path: Path) -> None:
    """M1: the final ``advance(end_ns)`` pays a settlement after the last decision.

    There is one decision (D1), which opens a BTC long at ``D1 + 59.999 s``; a
    settlement at Monday 08:00 falls in ``(D1, end_ns)`` and is accrued only by
    the final ``funding.advance(fills, spec.end_ns)``. A settlement at exactly
    ``end_ns`` is ignored (it gets a nonzero rate, so it would show if included).
    """
    start = _ns("2024-06-03")
    end = start + WEEK
    d1 = start + 5 * MIN
    settlement = start + 8 * HOUR  # Monday 08:00: after the last (only) decision
    inputs = _inputs(
        tmp_path,
        start,
        end,
        _priced_ticks(
            [d1],
            {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0},
            {"BTCUSDT": 51010.0, "ETHUSDT": 3000.0},
        ),
        {"BTCUSDT": {settlement: 0.0001}, "ETHUSDT": {}},
        {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0},
    )
    _add_archive_row(tmp_path / "raw_data.db", "BTCUSDT", end, 0.0005)

    run = simulate_reference(inputs)

    assert run.decisions[-1].ts == d1
    assert run.final_steps == {"BTCUSDT": 98, "ETHUSDT": -1666}  # still held at the end
    assert [payment for payment in run.payments if payment.amount != 0.0] == [
        FundingPayment("BTCUSDT", settlement + _JITTER_NS, 0.4998, False)
    ]
    assert not any(payment.ts >= end for payment in run.payments)


def test_fill_order_is_ts_then_symbol(tmp_path: Path) -> None:
    """M2: fills are sorted by ``(ts, symbol)``, not by release/plan order.

    ``plan_orders`` emits BTC before ETH (ascending symbol), so the first-planned
    symbol must release LATER for the sort to matter. BTC's D1 bar is missing, so
    BTC releases at ``D1 + 119.999 s`` while ETH releases at ``D1 + 59.999 s``:
    release order is (BTC, ETH) but the run's fills must be (ETH, BTC).
    """
    start = _ns("2024-06-03")
    end = start + WEEK
    d1 = start + 5 * MIN
    inputs = _inputs(
        tmp_path,
        start,
        end,
        {
            "BTCUSDT": (
                PriceTick(d1 - 1_000_000, 51000.0),
                PriceTick(d1 + INIT + MIN, 51010.0),  # D1's 1m bar missing -> a minute later
            ),
            "ETHUSDT": (PriceTick(d1 - 1_000_000, 3000.0), PriceTick(d1 + INIT, 3000.0)),
        },
        {"BTCUSDT": {}, "ETHUSDT": {}},
        {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0},
    )

    run = simulate_reference(inputs)

    assert [order.symbol for order in run.decisions[0].orders] == ["BTCUSDT", "ETHUSDT"]
    assert run.fills == (
        CarryFill("ETHUSDT", d1 + INIT, -1666, -1.666, 3000.0, 2.499, False),
        CarryFill("BTCUSDT", d1 + INIT + MIN, 98, 0.098, 51010.0, 2.49949, False),
    )
