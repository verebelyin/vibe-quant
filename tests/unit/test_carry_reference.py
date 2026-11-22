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

from tests.unit.test_carry_bars import hourly, ms, trend_spec, use_snapshot, write_archive
from vibe_quant.data.catalog import create_instrument
from vibe_quant.data.mark_bars import MarkSeries
from vibe_quant.metrics import BarSeries
from vibe_quant.portfolio.ledger import (
    CarryFill,
    FundingAccumulator,
    FundingPayment,
    hour_mark_rule,
)
from vibe_quant.portfolio.reference import simulate_reference
from vibe_quant.portfolio.signal import Settlements, decision_signals
from vibe_quant.portfolio.sizing import InstrumentSpec, OrderIntent, decide
from vibe_quant.portfolio.spec import (
    PRICE_WINDOW_NS,
    CarryExecutionError,
    CarrySpec,
    decision_times,
)
from vibe_quant.portfolio.ticks import (
    CarryInputs,
    PriceTick,
    load_carry_inputs,
    select_ticks,
    sizing_price,
    trend_closes,
)
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


# ---------------------------------------------------------------------------
# arch-c9-2 1h tier at rule/loader level (R14, R37-R39; chief R-c9t3-1: the
# executor-level versions are T4's). The decision path below is the one
# ``simulate_reference`` runs: sizing_price -> decision_signals -> decide.
# ---------------------------------------------------------------------------

_MS = 1_000_000


def _trend_bars(base: float, bar_ns: int, decision_ns: int, variant: str) -> BarSeries:
    """170 daily closes (bars ending B - 1 ms) plus bars every ``bar_ns`` around D.

    ``variant``: ``full``; ``deleted`` (every bar opening >= D removed); ``x10`` (their
    closes x 10). Nothing at or after D may move the decision at D.
    """
    first_day = decision_ns - decision_ns % DAY
    by_open: dict[int, float] = {}
    for back in range(170):
        b = first_day - back * DAY
        by_open[b - bar_ns] = round(base * (1.0 + 0.002 * (170 - back) + 0.004 * (back % 4)), 1)
    for index, open_ns in enumerate(range(decision_ns - 2 * HOUR, decision_ns + 2 * HOUR, bar_ns)):
        by_open[open_ns] = round(base * (1.4 + 0.0001 * index), 1)
    if variant == "deleted":
        by_open = {o: c for o, c in by_open.items() if o < decision_ns}
    elif variant == "x10":
        by_open = {o: (c * 10 if o >= decision_ns else c) for o, c in by_open.items()}
    opens = sorted(by_open)
    closes = np.array([by_open[o] for o in opens], dtype=np.float64)
    ts = np.array([o + bar_ns - _MS for o in opens], dtype=np.int64)
    return BarSeries(ts=ts, open=closes, high=closes, low=closes, close=closes)


@pytest.mark.parametrize(("tier", "bar_ns"), [("1m", MIN), ("1h", HOUR)])
def test_decision_ignores_post_decision_bars(tier: str, bar_ns: int) -> None:
    """R14 (+ the R16 sizing mutant): the Decision at D is equal -- weights, targets,
    orders, prices -- with every bar opening >= D present, deleted or x10."""
    decision_ns = _ns("2024-03-11") + HOUR
    spec = CarrySpec.from_dates(
        _SYMBOLS,
        "2024-03-11",
        "2024-03-13",
        signal="trend",
        schedule="daily",
        weighting="time_series",
        direction="long_flat",
        vol_target=0.4,
        fill_granularity=tier,
    )
    assert decision_times(spec.start_ns, spec.end_ns, "daily")[0] == decision_ns
    instruments = {s: create_instrument(s) for s in _SYMBOLS}
    specs = {s: InstrumentSpec.from_instrument(instruments[s]) for s in _SYMBOLS}
    bases = {"BTCUSDT": 50_000.0, "ETHUSDT": 3_000.0}
    boundaries = range(decision_ns - 200 * DAY, decision_ns + DAY, DAY)
    decisions = []
    for variant in ("full", "deleted", "x10"):
        bars = {s: _trend_bars(bases[s], bar_ns, decision_ns, variant) for s in _SYMBOLS}
        ticks = {
            s: select_ticks(bars[s], [decision_ns], instruments[s], spec.end_ns, bar_ns=bar_ns)
            for s in _SYMBOLS
        }
        closes = {
            s: trend_closes(bars[s], [b - b % DAY for b in boundaries], instruments[s])
            for s in _SYMBOLS
        }
        sizing = {s: sizing_price(ticks[s], decision_ns) for s in _SYMBOLS}
        eligible = decision_signals(spec, {}, sizing, decision_ns, trend_closes=closes, listings={})
        decisions.append(
            decide(
                decision_ns,
                10_000.0,
                eligible.signals,
                eligible.excluded,
                eligible.prices,
                {},
                specs,
                spec,
                trend_closes=closes,
                regime=eligible.regime,
            )
        )
    full, deleted, scaled = decisions
    assert full.orders and all(full.target_steps[s] > 0 for s in _SYMBOLS)
    assert full == deleted == scaled


def _hour_run_archive(path: Path, last_day: str) -> Path:
    """BTCUSDT + ETHUSDT hourly bars and 8h funding from 2023-09-01 to before ``last_day``."""
    first = ms("2023-09-01T00:00:00")
    last = ms(f"{last_day}T00:00:00")
    return write_archive(
        path,
        {s: hourly(first, last, 100.0 if s == "BTCUSDT" else 50.0) for s in _SYMBOLS},
        dict.fromkeys(_SYMBOLS, first),
        funding_to_ms=last,
    )


def _accrue(inputs: CarryInputs, fills: list[CarryFill]) -> list[FundingPayment]:
    """The trend path's funding: causal grid, 1h as-of marks, one window per decision."""
    assert inputs.hour_bars is not None
    acc = FundingAccumulator(
        inputs.calc,
        inputs.marks,
        inputs.specs,
        inputs.spec.start_ns,
        mark_rule=hour_mark_rule(inputs.hour_bars),
        causal=True,
        end_ns=inputs.spec.end_ns,
    )
    for decision_ns in inputs.decisions:
        acc.advance(fills, decision_ns)
    acc.advance(fills, inputs.spec.end_ns)
    return acc.payments


def _btc_long(inputs: CarryInputs) -> list[CarryFill]:
    tick = inputs.ticks["BTCUSDT"][1]  # the first decision's fill tick
    return [CarryFill("BTCUSDT", tick.ts, 1000, 1.0, tick.price, 0.0, False)]


def test_end_exclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R37: end 2024-03-20 -> last D 03-19T01:00, its fill 01:59:59.999 < end; a held leg
    pays 03-19T16:00 but not the 03-20T00:00 settlement."""
    archive = _hour_run_archive(tmp_path / "a.db", "2024-03-25")
    use_snapshot(tmp_path, monkeypatch, archive)
    inputs = load_carry_inputs(trend_spec(), tmp_path / "no-catalog", archive)
    last = inputs.decisions[-1]
    assert last == _ns("2024-03-19") + HOUR
    fill = inputs.ticks["BTCUSDT"][-1]
    assert fill.ts == _ns("2024-03-19") + 2 * HOUR - _MS
    assert fill.ts < inputs.spec.end_ns
    fills = [CarryFill("BTCUSDT", fill.ts, 1000, 1.0, fill.price, 0.0, False)]
    paid = [p.ts for p in _accrue(inputs, fills)]
    assert paid == [_ns("2024-03-19") + 8 * HOUR + 2 * _MS, _ns("2024-03-19") + 16 * HOUR + 2 * _MS]


def test_end_truncation_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R38 (loader level): runs ending 2024-03-10 and 2024-03-20 on one archive agree on
    every decision, tick, close and payment before the earlier end."""
    archive = _hour_run_archive(tmp_path / "a.db", "2024-03-25")
    use_snapshot(tmp_path, monkeypatch, archive)
    short = load_carry_inputs(trend_spec(end="2024-03-10"), tmp_path / "no-catalog", archive)
    long = load_carry_inputs(trend_spec(end="2024-03-20"), tmp_path / "no-catalog", archive)
    cutoff = _ns("2024-03-10")
    assert short.decisions == long.decisions[: len(short.decisions)]
    assert all(d <= cutoff for d in short.decisions)
    assert short.trend_closes is not None and long.trend_closes is not None
    for symbol in _SYMBOLS:
        assert short.ticks[symbol] == tuple(t for t in long.ticks[symbol] if t.ts < cutoff)
        assert short.trend_closes[symbol] == {
            b: c for b, c in long.trend_closes[symbol].items() if b <= cutoff
        }
    short_paid = _accrue(short, _btc_long(short))
    assert short_paid
    assert short_paid == [p for p in _accrue(long, _btc_long(long)) if p.ts < cutoff]


def test_source_truncation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R39 (loader level): an archive without rows >= 2024-03-25 and the full one (to
    2024-04-10) give equal inputs and payments for a run ending 2024-03-20."""
    full = _hour_run_archive(tmp_path / "full.db", "2024-04-10")
    cut = _hour_run_archive(tmp_path / "cut.db", "2024-03-25")
    use_snapshot(tmp_path, monkeypatch, full)
    spec = trend_spec()
    a = load_carry_inputs(spec, tmp_path / "no-catalog", full)
    b = load_carry_inputs(spec, tmp_path / "no-catalog", cut)
    for field in (
        "decisions",
        "ticks",
        "trend_closes",
        "listings",
        "boundary_prices",
        "settlements",
        "specs",
    ):
        assert getattr(a, field) == getattr(b, field), field
    assert a.hour_bars is not None and b.hour_bars is not None
    for symbol in _SYMBOLS:
        assert np.array_equal(a.hour_bars[symbol].ts, b.hour_bars[symbol].ts)
        assert np.array_equal(a.hour_bars[symbol].close, b.hour_bars[symbol].close)
    assert _accrue(a, _btc_long(a)) == _accrue(b, _btc_long(b))
