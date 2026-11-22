"""Funding-carry ledger rules.

Contract rows R10, R16-R19, R18b, R18c, the R20b tail and the R22 event-instant
half of the funding-carry semantics contract
(``data/swarm/20261015-funding-carry/design.md`` §3, §4, §7). The archive
fixture is copied from ``tests/unit/test_book_funding.py`` so the ledger is
exercised against the same archived-rate shape as the other funding tests.

The windowed accumulator is the point of R18b: an exact seeded quantity per
window (``float(Decimal(steps) * step)``) means a resize-then-close leg leaves
no float residue, so nothing is charged after the leg is flat.
"""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vibe_quant.data.catalog import create_instrument
from vibe_quant.data.mark_bars import MarkSeries
from vibe_quant.portfolio.ledger import (
    CarryFill,
    CarryRun,
    FundingAccumulator,
    FundingPayment,
    equity_at,
    mark_at,
    steps_held,
)
from vibe_quant.portfolio.sizing import InstrumentSpec
from vibe_quant.portfolio.spec import CarryDataError, CarryExecutionError, CarrySpec
from vibe_quant.validation import funding as funding_module
from vibe_quant.validation.funding import FundingCalculator, clear_rate_cache

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_HOUR_NS = 3_600 * 10**9
_MIN_NS = 60 * 10**9
_DAY_NS = 86_400 * 10**9
_BASE_MS = 1_735_689_600_000  # 2025-01-01T00:00:00Z
_BASE_NS = _BASE_MS * 1_000_000
_SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _settle(i: int) -> int:
    """The ``i``-th 8h UTC boundary after the archive base."""
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


def _add_rates(path: Path, symbol: str, rates: dict[int, float]) -> None:
    """Add another symbol's 8h rates to an existing archive (same ms jitter)."""
    conn = sqlite3.connect(str(path))
    for i, rate in rates.items():
        conn.execute(
            "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate) VALUES (?,?,?)",
            (symbol, _BASE_MS + i * 8 * 3_600_000 + 3, rate),
        )
    conn.commit()
    conn.close()


def _marks(price: float) -> MarkSeries:
    """A flat 1m MARK series (open = close = ``price``) covering settlements 0..6."""
    ts = np.arange(_BASE_NS - 8 * _HOUR_NS, _settle(6), _MIN_NS, dtype=np.int64)
    values = np.full(ts.shape, price, dtype=np.float64)
    return MarkSeries(ts=ts, low=values, high=values, close=values, open=values)


def _specs() -> dict[str, InstrumentSpec]:
    return {s: InstrumentSpec.from_instrument(create_instrument(s)) for s in _SYMBOLS}


def _fill(
    symbol: str,
    ts: int,
    steps: int,
    price: float = 0.0,
    fee: float = 0.0,
    reduce_only: bool = False,
) -> CarryFill:
    return CarryFill(
        symbol=symbol,
        ts=ts,
        steps=steps,
        qty=float(Decimal(steps) * _specs()[symbol].step),
        price=price,
        fee=fee,
        reduce_only=reduce_only,
    )


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_rate_cache()


def test_steps_held_sums_steps_strictly_before_instant() -> None:
    """``steps_held`` is the signed step count of fills with ``ts < before_ns``."""
    fills = [
        _fill("BTCUSDT", 1_000, 100),
        _fill("BTCUSDT", 2_000, -300),
        _fill("ETHUSDT", 1_500, 7),
    ]
    assert steps_held(fills, "BTCUSDT", 1_000) == 0
    assert steps_held(fills, "BTCUSDT", 1_001) == 100
    assert steps_held(fills, "BTCUSDT", 2_000) == 100
    assert steps_held(fills, "BTCUSDT", 2_001) == -200
    assert steps_held(fills, "ETHUSDT", 1_500) == 0
    assert steps_held(fills, "ETHUSDT", 1_501) == 7


def test_mark_is_open_of_settlement_minute_and_missing_fails() -> None:
    """The mark is the OPEN of the 1m MARK bar opening at floor_minute(S)."""
    settle = _settle(1) + 3_000_000
    minute = _settle(1)
    series = MarkSeries(
        ts=np.array([minute - _MIN_NS, minute, minute + _MIN_NS], dtype=np.int64),
        low=np.array([5.0, 51000.0, 52000.0]),
        high=np.array([5.0, 51000.0, 52000.0]),
        close=np.array([5.0, 51010.0, 52000.0]),
        open=np.array([5.0, 51000.0, 52000.0]),
    )
    assert mark_at(series, "BTCUSDT", settle) == 51000.0
    assert mark_at(series, "BTCUSDT", minute) == 51000.0
    assert mark_at(series, "BTCUSDT", minute + _MIN_NS - 1) == 51000.0
    assert mark_at(series, "BTCUSDT", minute + _MIN_NS) == 52000.0

    hole = MarkSeries(
        ts=np.array([minute - _MIN_NS, minute + _MIN_NS], dtype=np.int64),
        low=np.array([5.0, 5.0]),
        high=np.array([5.0, 5.0]),
        close=np.array([5.0, 5.0]),
        open=np.array([5.0, 5.0]),
    )
    with pytest.raises(CarryDataError, match="BTCUSDT"):
        mark_at(hole, "BTCUSDT", settle)


def test_funding_sign_long_short_negative_rate(tmp_path: Path) -> None:
    """Paid = qty x MARK x rate; positive rate pays a long, a short receives."""
    calc = _archive(tmp_path / "a.db", {0: 0.0001, 1: 0.0001})
    _add_rates(tmp_path / "a.db", "ETHUSDT", {0: 0.0001, 1: 0.0001})
    marks = {"BTCUSDT": _marks(51000.0), "ETHUSDT": _marks(3000.0)}
    fills = [
        _fill("BTCUSDT", _settle(0) + _HOUR_NS, 98),
        _fill("ETHUSDT", _settle(0) + _HOUR_NS, -1682),
    ]
    acc = FundingAccumulator(calc, marks, _specs(), _settle(0))
    payments = acc.advance(fills, _settle(1) + 10**9)
    assert [(p.symbol, p.ts, p.fallback) for p in payments] == [
        ("BTCUSDT", _settle(1) + 3_000_000, False),
        ("ETHUSDT", _settle(1) + 3_000_000, False),
    ]
    assert payments[0].amount == pytest.approx(0.4998, rel=1e-12)
    assert payments[1].amount == pytest.approx(-0.5046, rel=1e-12)

    neg = _archive(tmp_path / "b.db", {0: -0.0001, 1: -0.0001})
    negative = FundingAccumulator(neg, {"BTCUSDT": _marks(51000.0)}, _specs(), _settle(0))
    negative_payments = negative.advance(
        [_fill("BTCUSDT", _settle(0) + _HOUR_NS, 98)], _settle(1) + 10**9
    )
    assert [(p.ts, p.fallback) for p in negative_payments] == [(_settle(1) + 3_000_000, False)]
    assert negative_payments[0].amount == pytest.approx(-0.4998, rel=1e-12)


def test_hole_fallback_flagged_only_while_held(tmp_path: Path) -> None:
    """An uncovered 8h boundary while held is a flagged fallback; flat pays nothing."""
    calc = _archive(tmp_path / "a.db", {0: 0.0001})
    marks = {"BTCUSDT": _marks(51000.0)}
    specs = _specs()
    held = FundingAccumulator(calc, marks, specs, _settle(0))
    held_payments = held.advance([_fill("BTCUSDT", _settle(0) + _HOUR_NS, 100)], _settle(1) + 10**9)
    assert [(p.ts, p.fallback) for p in held_payments] == [(_settle(1), True)]
    assert held_payments[0].amount == pytest.approx(0.51, rel=1e-12)  # 0.1 x 51000 x 0.0001
    assert sum(p.fallback for p in held.payments) == 1

    flat = FundingAccumulator(calc, marks, specs, _settle(0))
    assert flat.advance([], _settle(1) + 10**9) == []
    assert flat.payments == []


def test_resize_then_close_leaves_no_residue(tmp_path: Path) -> None:
    """+100, +200, -300 steps over three windows -> nothing after the close."""
    calc = _archive(tmp_path / "a.db", {0: 0.0001, 1: 0.0001, 2: 0.0001, 3: 0.0001})
    marks = {"BTCUSDT": _marks(51000.0)}
    specs = _specs()
    acc = FundingAccumulator(calc, marks, specs, _settle(0))
    fills = [_fill("BTCUSDT", _settle(0) + _HOUR_NS, 100)]
    first = acc.advance(fills, _settle(1) + 2 * _HOUR_NS)
    assert [p.ts for p in first] == [_settle(1) + 3_000_000]
    assert first[0].amount == 0.1 * 51000.0 * 0.0001
    fills.append(_fill("BTCUSDT", _settle(1) + 3 * _HOUR_NS, 200))
    second = acc.advance(fills, _settle(2) + 2 * _HOUR_NS)
    assert [p.ts for p in second] == [_settle(2) + 3_000_000]
    assert second[0].amount == (0.1 + 0.2) * 51000.0 * 0.0001  # seed 0.1 + in-window fill 0.2
    fills.append(_fill("BTCUSDT", _settle(2) + 3 * _HOUR_NS, -300))
    assert acc.advance(fills, _settle(3) + 2 * _HOUR_NS) == []  # nothing charged after the close
    assert all(not p.fallback for p in acc.payments)
    assert steps_held(fills, "BTCUSDT", _settle(3)) == 0
    # seed 0 exactly: a later window charges nothing (no running-float residue)
    assert acc.advance(fills, _settle(4) + 2 * _HOUR_NS) == []


def test_settlement_on_window_edge_counted_once(tmp_path: Path) -> None:
    """A settlement exactly at a window boundary lands in the later window, once."""
    calc = _archive(tmp_path / "a.db", {0: 0.0001, 1: 0.0001, 2: 0.0001})
    marks = {"BTCUSDT": _marks(51000.0)}
    specs = _specs()
    acc = FundingAccumulator(calc, marks, specs, _settle(0))
    fills = [_fill("BTCUSDT", _settle(0) + _HOUR_NS, 100)]
    edge_ts = _settle(1) + 3_000_000  # the archived settlement instant == window boundary t
    assert acc.advance(fills, edge_ts) == []  # S == t is not in [last, t)
    second = acc.advance(fills, _settle(2))  # [t, next) holds S, on the qty held before it
    assert [(p.ts, p.fallback) for p in second] == [(edge_ts, False)]
    assert second[0].amount == 0.1 * 51000.0 * 0.0001
    assert [p.ts for p in acc.payments] == [edge_ts]


def test_fill_at_window_start_and_settlement_ts(tmp_path: Path) -> None:
    """A close stamped at the window start == settlement ts is in the window.

    The settlement is charged on the qty before the close; the fill is not dropped,
    so the leg is flat for the next boundary.
    """
    calc = _archive(tmp_path / "a.db", {0: 0.0001, 1: 0.0001, 2: 0.0001})
    marks = {"BTCUSDT": _marks(51000.0)}
    edge_ts = _settle(1) + 3_000_000
    fills = [_fill("BTCUSDT", _settle(0) + _HOUR_NS, 100)]
    acc = FundingAccumulator(calc, marks, _specs(), _settle(0))
    assert acc.advance(fills, edge_ts) == []
    fills.append(_fill("BTCUSDT", edge_ts, -100))
    window = acc.advance(fills, _settle(2) + 10**9)
    assert [(p.ts, p.fallback) for p in window] == [(edge_ts, False)]
    assert window[0].amount == 0.1 * 51000.0 * 0.0001  # qty before the close; none at settle(2)


def test_settlement_at_fill_ts_through_accumulator(tmp_path: Path) -> None:
    """A fill stamped at the settlement instant pays on the qty before it."""
    calc = _archive(tmp_path / "a.db", {1: 0.0001, 2: 0.0001})
    fills = [_fill("BTCUSDT", _settle(1) + 3_000_000, 100)]
    acc = FundingAccumulator(
        calc, {"BTCUSDT": _marks(51000.0)}, _specs(), _settle(1)
    )
    payments = acc.advance(fills, _settle(2) + 10**9)
    assert [p.ts for p in payments] == [_settle(2) + 3_000_000]
    assert payments[0].amount == pytest.approx(0.51, rel=1e-12)


def test_settlements_at_or_after_end_ignored(tmp_path: Path) -> None:
    """The final window excludes settlements at or after ``end_ns``."""
    calc = _archive(tmp_path / "a.db", {1: 0.0001, 2: 0.0001})
    fills = [_fill("BTCUSDT", _settle(1) - _HOUR_NS, 100)]
    acc = FundingAccumulator(
        calc, {"BTCUSDT": _marks(51000.0)}, _specs(), _settle(1)
    )
    payments = acc.advance(fills, _settle(2))
    assert [p.ts for p in payments] == [_settle(1) + 3_000_000]
    assert all(p.ts < _settle(2) for p in acc.payments)


def test_equity_at_cash_form() -> None:
    """E0 + fsum(-q*px - fee) - fsum(funding) + fsum(q_s * p_s) = 10096.0."""
    at_ns = 10_000_000_000
    fills = [_fill("BTCUSDT", at_ns - _HOUR_NS, 100, price=50000.0, fee=2.5)]
    payments = [FundingPayment("BTCUSDT", at_ns - 2 * _HOUR_NS, 1.5, False)]
    assert equity_at(10_000.0, fills, payments, {"BTCUSDT": 51000.0}, at_ns) == 10096.0


def test_events_at_instant_are_excluded() -> None:
    """Only events with ``ts < at_ns`` move the equity."""
    fills = [_fill("BTCUSDT", 1_000, 100, price=100.0, fee=1.0)]
    payments = [FundingPayment("BTCUSDT", 2_000, 5.0, False)]
    prices = {"BTCUSDT": 120.0}
    assert equity_at(10_000.0, fills, payments, prices, 1_000) == 10_000.0
    assert equity_at(10_000.0, fills, payments, prices, 1_001) == 10_001.0
    assert equity_at(10_000.0, fills, payments, prices, 2_000) == 10_001.0
    assert equity_at(10_000.0, fills, payments, prices, 2_001) == 9_996.0


def test_equity_at_held_symbol_without_price_fails_closed() -> None:
    fills = [_fill("BTCUSDT", 1_000, 100, price=100.0)]
    with pytest.raises(CarryDataError, match="no price for held BTCUSDT"):
        equity_at(10_000.0, fills, [], {}, 2_000)
    closed = [_fill("BTCUSDT", 1_000, 100, price=100.0), _fill("BTCUSDT", 1_500, -100, price=100.0)]
    assert equity_at(10_000.0, closed, [], {}, 2_000) == 10_000.0


def test_equity_at_resized_then_closed_leg_is_flat_exactly() -> None:
    """Held = integer step sum, so +100,+200,-300 (float fsum 2.78e-17) is flat, no price."""
    at_ns = 10_000
    resize = [
        _fill("BTCUSDT", 1_000, 100, price=50000.0),
        _fill("BTCUSDT", 2_000, 200, price=50000.0),
        _fill("BTCUSDT", 3_000, -300, price=50000.0),
    ]
    assert equity_at(10_000.0, resize, [], {}, at_ns) == 10_000.0
    assert equity_at(10_000.0, resize, [], {"BTCUSDT": 51000.0}, at_ns) == 10_000.0
    band = [
        _fill("BTCUSDT", 1_000, 98, price=50000.0),
        _fill("BTCUSDT", 2_000, 12, price=50000.0),
        _fill("BTCUSDT", 3_000, -110, price=50000.0),
    ]
    assert equity_at(10_000.0, band, [], {}, at_ns) == 10_000.0


def test_equity_at_held_qty_is_exact_step_product() -> None:
    """A still-held leg is marked at float(Decimal(step_sum) * step), not a float fsum."""
    fills = [
        _fill("BTCUSDT", 1_000, 100, price=0.0),
        _fill("BTCUSDT", 2_000, 200, price=0.0),
        _fill("BTCUSDT", 3_000, -299, price=0.0),
    ]
    assert equity_at(10_000.0, fills, [], {"BTCUSDT": 1000.0}, 10_000) == 10_001.0
    with pytest.raises(CarryDataError, match="no price for held BTCUSDT"):
        equity_at(10_000.0, fills, [], {}, 10_000)


def test_equity_at_inconsistent_step_fails_closed() -> None:
    """Fills of one symbol must share one step, else the ledger is corrupt."""
    good = _fill("BTCUSDT", 1_000, 100)
    bad = CarryFill("BTCUSDT", 2_000, 100, 1.0, 0.0, 0.0, False)  # step 0.01, not 0.001
    with pytest.raises(CarryExecutionError, match="BTCUSDT"):
        equity_at(10_000.0, [good, bad], [], {"BTCUSDT": 1.0}, 10_000)


def test_advance_rejects_non_increasing_window() -> None:
    acc = FundingAccumulator(FundingCalculator(), {}, _specs(), 1_000)
    with pytest.raises(ValueError, match="must be after"):
        acc.advance([], 1_000)
    with pytest.raises(ValueError):
        acc.advance([], 999)


def test_settlements_returns_tuple_copies(tmp_path: Path) -> None:
    """T2a: ``settlements()`` returns tuple copies, never the cache's lists."""
    path = tmp_path / "a.db"
    calc = _archive(path, {0: 0.0001, 1: 0.0002})
    first = calc.settlements("BTCUSDT")
    second = calc.settlements("BTCUSDT")
    assert first == second
    assert first == ((_settle(0) + 3_000_000, _settle(1) + 3_000_000), (0.0001, 0.0002))
    assert isinstance(first[0], tuple)
    assert isinstance(first[1], tuple)
    cached_times, cached_rates = funding_module._RATE_CACHE[(str(path), "BTCUSDT")]
    assert first[0] is not cached_times
    assert first[1] is not cached_rates
    assert first[0] == tuple(cached_times)
    with pytest.raises(TypeError):
        first[0][0] = 7  # type: ignore[index]


def test_carry_run_carries_both_executors_output(tmp_path: Path) -> None:
    """``CarryRun`` is the shared output of both executors (design §8 T2a)."""
    spec = CarrySpec(symbols=_SYMBOLS, start_ns=_BASE_NS, end_ns=_BASE_NS + 40 * _DAY_NS)
    calc = _archive(tmp_path / "a.db", {0: 0.0001, 1: 0.0001})
    fills = [_fill("BTCUSDT", _settle(0) + _HOUR_NS, 100, price=51000.0, fee=2.55)]
    acc = FundingAccumulator(calc, {"BTCUSDT": _marks(51000.0)}, _specs(), _settle(0))
    acc.advance(fills, _settle(1) + 10**9)
    run = CarryRun(
        spec=spec,
        decisions=(),
        fills=tuple(fills),
        payments=tuple(acc.payments),
        final_steps={"BTCUSDT": 100},
        engine="reference",
    )
    assert run.engine == "reference"
    assert run.payments == tuple(acc.payments)
    assert run.final_steps == {"BTCUSDT": 100}
    assert steps_held(run.fills, "BTCUSDT", _settle(2)) == 100
    assert list(run.payments) == sorted(run.payments, key=lambda p: (p.ts, p.symbol))


def _copied_equity_at(
    initial_balance: float,
    fills: list[CarryFill],
    payments: list[FundingPayment],
    prices: dict[str, float],
    at_ns: int,
) -> float:
    """The pre-optimisation body: full scans of fills and payments, a repr per fill."""
    import math

    cash = math.fsum(-fill.qty * fill.price - fill.fee for fill in fills if fill.ts < at_ns)
    funding = math.fsum(payment.amount for payment in payments if payment.ts < at_ns)
    step_sums: dict[str, int] = {}
    steps: dict[str, Decimal] = {}
    for fill in fills:
        if fill.ts >= at_ns:
            continue
        step_sums[fill.symbol] = step_sums.get(fill.symbol, 0) + fill.steps
        if fill.steps:
            step = Decimal(repr(fill.qty)) / Decimal(fill.steps)
            if steps.setdefault(fill.symbol, step) != step:
                raise CarryExecutionError("inconsistent")
    marked: list[float] = []
    for symbol, step_sum in step_sums.items():
        if step_sum == 0:
            continue
        if symbol not in prices:
            raise CarryDataError("no price")
        marked.append(float(Decimal(step_sum) * steps[symbol]) * prices[symbol])
    return initial_balance + cash - funding + math.fsum(marked)


def test_equity_at_equals_the_full_scan_body_bit_for_bit() -> None:
    """The prefix-of-payments / memoised-increment rewrite changes no float, at any instant.

    Random books (both symbols, resizes, flips to flat, zero-step fills, duplicate
    timestamps), non-trivial float amounts, and ``at_ns`` landing exactly on, one
    nanosecond either side of, and between event timestamps.
    """
    import random

    rng = random.Random(11)
    checked = 0
    for _ in range(250):
        stamps = sorted(rng.randrange(0, 400) * 1_000 for _ in range(rng.randrange(0, 40)))
        fills = [
            _fill(
                rng.choice(["BTCUSDT", "ETHUSDT"]),
                ts,
                rng.choice([-300, -17, 0, 5, 98, 1682]),
                price=rng.uniform(1.0, 70_000.0),
                fee=rng.uniform(0.0, 3.0),
            )
            for ts in stamps
        ]
        pay_stamps = sorted(rng.randrange(0, 400) * 1_000 for _ in range(rng.randrange(0, 60)))
        payments = [
            FundingPayment(rng.choice(["BTCUSDT", "ETHUSDT"]), ts, rng.uniform(-9.0, 9.0), False)
            for ts in pay_stamps
        ]
        prices = {"BTCUSDT": rng.uniform(1.0, 70_000.0), "ETHUSDT": rng.uniform(1.0, 5_000.0)}
        probes = {0, 1, 400_000, 10**9}
        for event in [*stamps, *pay_stamps]:
            probes.update({event - 1, event, event + 1})
        for at_ns in sorted(probes):
            expected = _copied_equity_at(10_000.0, fills, payments, prices, at_ns)
            assert equity_at(10_000.0, fills, payments, prices, at_ns) == expected
            assert equity_at(10_000.0, tuple(fills), tuple(payments), prices, at_ns) == expected
            checked += 1
    assert checked > 5_000


def test_accumulated_payments_ascend_by_timestamp_across_windows(tmp_path: Path) -> None:
    """``equity_at`` takes payments before ``at_ns`` as a prefix: windows must stack in order."""
    rates = dict.fromkeys(range(8), 0.0001)
    calc = _archive(tmp_path / "a.db", rates)
    _add_rates(tmp_path / "a.db", "ETHUSDT", rates)
    marks = {"BTCUSDT": _marks(51000.0), "ETHUSDT": _marks(3000.0)}
    acc = FundingAccumulator(calc, marks, _specs(), _settle(0))
    fills = [_fill("BTCUSDT", _settle(0) + _HOUR_NS, 98), _fill("ETHUSDT", _settle(0), -500)]
    for window in range(1, 6):
        acc.advance(fills, _settle(window) + 10**9)
        fills.append(_fill("BTCUSDT", _settle(window) + 2 * _HOUR_NS, 10 * window))
    stamps = [payment.ts for payment in acc.payments]
    assert len(stamps) >= 10
    assert stamps == sorted(stamps)


# --- R65 (arch-c9-2, T1): BookIndex, the indexed equity/held ledger ---------------------------

_INDEX_STEPS = {
    "BTCUSDT": Decimal("0.001"),
    "ETHUSDT": Decimal("0.01"),
    "SOLUSDT": Decimal("1"),
    "DOGEUSDT": Decimal("0.1"),
}


def _outcome(compute: Callable[[], float | int]) -> tuple[object, ...]:
    """A comparable result: ``float.hex`` of a float, the int, or the exception type and text."""
    try:
        value = compute()
    except (OverflowError, CarryDataError, CarryExecutionError) as error:
        return ("raise", type(error), str(error))
    return ("ok", value.hex() if isinstance(value, float) else value)


def _seeded_book(
    seed: int, n_fills: int
) -> tuple[list[list[CarryFill]], list[list[FundingPayment]]]:
    """Rounds of fills and payments the way the reference executor appends them.

    Per round ``r`` (decision ``D_r = r * 1000``): payments in ``[D_{r-1}, D_r)`` ascending,
    then each symbol (random order) releases 0..3 fills at its own tick ``D_r + offset``, so
    fills are ascending per symbol but NOT globally. Two fills of one symbol in one tick are
    an equal-ts flip; ``|qty * price|`` spans 1e-8..1e7; every 5th fill is followed by its
    exact cash cancellation and every 7th payment by its negation.
    """
    import random

    rng = random.Random(seed)
    rounds_fills: list[list[CarryFill]] = []
    rounds_pays: list[list[FundingPayment]] = []
    count = 0
    pays = 0
    r = 0
    while count < n_fills:
        r += 1
        payments: list[FundingPayment] = []
        for ts in sorted(
            rng.randrange((r - 1) * 1000, r * 1000) for _ in range(rng.randrange(0, 8))
        ):
            amount = rng.choice([-1.0, 1.0]) * 10 ** rng.uniform(-9.0, 6.0)
            payments.append(FundingPayment(rng.choice(list(_INDEX_STEPS)), ts, amount, False))
            pays += 1
            if pays % 7 == 0:
                payments.append(FundingPayment(payments[-1].symbol, ts, -amount, False))
        fills: list[CarryFill] = []
        symbols = list(_INDEX_STEPS)
        rng.shuffle(symbols)
        for symbol in symbols:
            tick = r * 1000 + rng.randrange(1, 500)
            for _ in range(rng.choice([0, 1, 1, 2, 3])):
                steps = rng.choice([-1682, -300, -17, -1, 0, 1, 5, 98, 4000])
                qty = float(Decimal(steps) * _INDEX_STEPS[symbol])
                notional = 10 ** rng.uniform(-8.0, 7.0)
                price = notional / abs(qty) if qty else rng.uniform(1.0, 9.0)
                fee = rng.uniform(0.0, 3.0) * 10 ** rng.uniform(-6.0, 1.0)
                fills.append(CarryFill(symbol, tick, steps, qty, price, fee, False))
                count += 1
                if count % 5 == 0:
                    fills.append(CarryFill(symbol, tick, -steps, -qty, price, -fee, False))
                    count += 1
        rounds_fills.append(fills)
        rounds_pays.append(payments)
    return rounds_fills, rounds_pays


def test_book_index_oracle() -> None:
    """R65: BookIndex equals ``equity_at``/``steps_held`` bit for bit on 10,000 seeded fills.

    Executor mode: the index reads the executor's growing lists; queries at the next
    decision (every fill before it), exactly at a fill ts (strict cutoff), one ns either
    side, and backwards in time (the oracle path) -- every one ``float.hex`` equal. Metrics
    mode: the same book sorted by ``(ts, symbol)`` (prefix by bisect). A held symbol without
    a price raises the oracle's ``CarryDataError`` text.
    """
    import random

    from vibe_quant.portfolio import ledger

    rounds_fills, rounds_pays = _seeded_book(seed=65, n_fills=10_000)
    rng = random.Random(66)
    prices = {symbol: 10 ** rng.uniform(-3.0, 4.0) for symbol in _INDEX_STEPS}
    fills: list[CarryFill] = []
    payments: list[FundingPayment] = []
    book = ledger.BookIndex(fills, payments)
    no_sol = {s: p for s, p in prices.items() if s != "SOLUSDT"}
    checkpoints = {*range(5, len(rounds_fills), len(rounds_fills) // 12), len(rounds_fills)}
    checked = 0
    for r, (new_fills, new_pays) in enumerate(zip(rounds_fills, rounds_pays, strict=True), 1):
        payments.extend(new_pays)
        if r in checkpoints:  # the oracle is O(n) per call: ~13 checkpoints keep this < 2 s
            decision = r * 1000
            pick = rng.choice(fills).ts
            last = max(fill.ts for fill in fills)  # at == the newest fill: fast path must not apply
            probes = [decision, decision + 1, last, pick, pick - 1, pick + 1]
            probes.append(rng.randrange(0, decision))
            book_prices = no_sol if r % 2 else prices
            for at_ns in probes:
                assert _outcome(
                    lambda: book.equity(10_000.0, book_prices, at_ns)  # noqa: B023
                ) == _outcome(
                    lambda: equity_at(10_000.0, fills, payments, book_prices, at_ns)  # noqa: B023
                )
                checked += 1
                for symbol in _INDEX_STEPS:
                    assert book.held(symbol, at_ns) == steps_held(fills, symbol, at_ns)
        fills.extend(new_fills)
    assert len(fills) >= 10_000
    assert checked >= 60

    ordered = tuple(sorted(fills, key=lambda fill: (fill.ts, fill.symbol)))
    sorted_book = ledger.BookIndex(ordered, tuple(payments))
    stamps = [fill.ts for fill in ordered]
    probes = [
        0,
        stamps[-1] + 1,
        *(rng.choice(stamps) + delta for _ in range(6) for delta in (-1, 0, 1)),
    ]
    rng.shuffle(probes)  # out-of-order queries
    no_eth = {s: p for s, p in prices.items() if s != "ETHUSDT"}
    no_price: set[tuple[object, ...]] = set()
    for index, at_ns in enumerate(probes):
        book_prices = no_eth if index % 2 else prices
        expected = _outcome(lambda: equity_at(10_000.0, ordered, payments, book_prices, at_ns))  # noqa: B023
        assert _outcome(lambda: sorted_book.equity(10_000.0, book_prices, at_ns)) == expected  # noqa: B023
        if expected[0] == "raise":
            no_price.add(expected[1:])
        for symbol in _INDEX_STEPS:
            assert sorted_book.held(symbol, at_ns) == steps_held(ordered, symbol, at_ns)
    assert no_price  # the missing-price branch was reached
    assert all(kind is CarryDataError and "no price for held" in text for kind, text in no_price)


@pytest.mark.parametrize(
    ("fees", "expected"),
    [
        ((-1e308, -1e308, 1e308), ("raise", OverflowError, "intermediate overflow in fsum")),
        ((-1e308, 1e308, -1e308), ("ok", (1e308 + 10_000.0).hex())),
    ],
)
def test_book_index_keeps_fsum_term_order_overflow(
    fees: tuple[float, ...], expected: tuple[object, ...]
) -> None:
    """R65: cash terms [1e308, 1e308, -1e308] overflow in fsum exactly as the oracle does.

    The index passes the oracle's term list in the oracle's order to ``math.fsum``, so the
    order-sensitive result survives: the reordered list sums to 1e308 in both.
    """
    from vibe_quant.portfolio import ledger

    fills = [CarryFill("BTCUSDT", 10 + i, 0, 0.0, 0.0, fee, False) for i, fee in enumerate(fees)]
    for book_fills in (fills, tuple(fills)):
        book = ledger.BookIndex(book_fills, [])
        oracle = _outcome(lambda: equity_at(10_000.0, book_fills, [], {}, 100))  # noqa: B023
        assert oracle == expected
        assert _outcome(lambda: book.equity(10_000.0, {}, 100)) == oracle  # noqa: B023


def test_book_index_inconsistent_increment_raises_at_append() -> None:
    """R65: a second size increment for one symbol raises when indexed (declared earlier).

    The oracle only sees it once the bad fill is before ``at_ns``; the index raises the same
    ``CarryExecutionError`` text on its first read after the append, at any instant.
    """
    from vibe_quant.portfolio import ledger

    good = _fill("BTCUSDT", 1_000, 100)
    bad = CarryFill("BTCUSDT", 2_000, 100, 1.0, 0.0, 0.0, False)  # step 0.01, not 0.001
    fills = [good]
    book = ledger.BookIndex(fills, [])
    assert book.held("BTCUSDT", 1_500) == 100
    fills.append(bad)
    assert equity_at(10_000.0, fills, [], {"BTCUSDT": 1.0}, 1_500) == 10_000.1
    with pytest.raises(CarryExecutionError) as oracle:
        equity_at(10_000.0, fills, [], {"BTCUSDT": 1.0}, 3_000)
    with pytest.raises(CarryExecutionError) as indexed:
        book.equity(10_000.0, {"BTCUSDT": 1.0}, 1_500)
    assert str(indexed.value) == str(oracle.value)
    assert "inconsistent size increment for BTCUSDT" in str(oracle.value)


def test_book_index_out_of_order_append_falls_back_to_the_oracle(tmp_path: Path) -> None:
    """C9-13: a same-symbol append with a SMALLER ts than the last one keeps oracle results.

    Insertion ``[ts 2, ts 1]`` breaks the per-symbol monotonic invariant the bisect needs;
    equity, held and the funding window then fall back to the full scans and stay equal.
    """
    from vibe_quant.portfolio import ledger

    fills = [
        _fill("BTCUSDT", 2_000, 100, price=100.0, fee=1.0),
        _fill("BTCUSDT", 1_000, -1, price=100.0),
    ]
    book = ledger.BookIndex(fills, [])
    prices = {"BTCUSDT": 100.0}
    for at_ns in (999, 1_000, 1_001, 2_000, 2_001):
        assert book.held("BTCUSDT", at_ns) == steps_held(fills, "BTCUSDT", at_ns)
        assert _outcome(lambda: book.equity(10_000.0, prices, at_ns)) == _outcome(  # noqa: B023
            lambda: equity_at(10_000.0, fills, [], prices, at_ns)  # noqa: B023
        )
    assert book.held("BTCUSDT", 2_000) == -1
    assert book.equity(10_000.0, prices, 2_000) == 10_000.0
    assert book.equity(10_000.0, prices, 2_001) == 9_999.0

    # the windowed funding accrual: with the index == the full scan, out-of-order included
    rates = dict.fromkeys(range(8), 0.0001)
    calc = _archive(tmp_path / "a.db", rates)
    _add_rates(tmp_path / "a.db", "ETHUSDT", rates)
    marks = {"BTCUSDT": _marks(51000.0), "ETHUSDT": _marks(3000.0)}
    plain = FundingAccumulator(calc, marks, _specs(), _settle(0))
    indexed = FundingAccumulator(calc, marks, _specs(), _settle(0))
    run_fills = [_fill("BTCUSDT", _settle(0) + _HOUR_NS, 98), _fill("ETHUSDT", _settle(0), -500)]
    book = ledger.BookIndex(run_fills, indexed.payments)
    for window in range(1, 6):
        before = _settle(window) + 10**9
        assert indexed.advance(run_fills, before, book=book) == plain.advance(run_fills, before)
        run_fills.append(_fill("BTCUSDT", _settle(window) + 2 * _HOUR_NS, 10 * window))
        if window == 3:  # an ETH fill stamped before the previous ETH one: out of order
            run_fills.append(_fill("ETHUSDT", _settle(window) - 4 * _HOUR_NS, 200))
            run_fills.append(_fill("ETHUSDT", _settle(window) - 5 * _HOUR_NS, 100))
    assert indexed.payments == plain.payments
    assert len(plain.payments) >= 10
    assert any(payment.symbol == "ETHUSDT" for payment in plain.payments)
