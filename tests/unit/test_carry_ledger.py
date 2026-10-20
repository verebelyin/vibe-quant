"""Funding-carry ledger rules (epic vibe-quant-x0axf, task T2a).

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
    """R17: the mark is the OPEN of the 1m MARK bar opening at floor_minute(S)."""
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
    """R16: paid = qty x MARK x rate; positive rate pays a long, a short receives."""
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
    """R18: an uncovered 8h boundary while held is a flagged fallback; flat pays nothing."""
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
    """R18b: +100, +200, -300 steps over three windows -> nothing after the close."""
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
    """R18c: a settlement exactly at a window boundary lands in the later window, once."""
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
    """R18c/R19: a close stamped at the window start == settlement ts is in the window.

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
    """R19: a fill stamped at the settlement instant pays on the qty before it."""
    calc = _archive(tmp_path / "a.db", {1: 0.0001, 2: 0.0001})
    fills = [_fill("BTCUSDT", _settle(1) + 3_000_000, 100)]
    acc = FundingAccumulator(
        calc, {"BTCUSDT": _marks(51000.0)}, _specs(), _settle(1)
    )
    payments = acc.advance(fills, _settle(2) + 10**9)
    assert [p.ts for p in payments] == [_settle(2) + 3_000_000]
    assert payments[0].amount == pytest.approx(0.51, rel=1e-12)


def test_settlements_at_or_after_end_ignored(tmp_path: Path) -> None:
    """R20b tail: the final window excludes settlements at or after ``end_ns``."""
    calc = _archive(tmp_path / "a.db", {1: 0.0001, 2: 0.0001})
    fills = [_fill("BTCUSDT", _settle(1) - _HOUR_NS, 100)]
    acc = FundingAccumulator(
        calc, {"BTCUSDT": _marks(51000.0)}, _specs(), _settle(1)
    )
    payments = acc.advance(fills, _settle(2))
    assert [p.ts for p in payments] == [_settle(1) + 3_000_000]
    assert all(p.ts < _settle(2) for p in acc.payments)


def test_equity_at_cash_form() -> None:
    """R10: E0 + fsum(-q*px - fee) - fsum(funding) + fsum(q_s * p_s) = 10096.0."""
    at_ns = 10_000_000_000
    fills = [_fill("BTCUSDT", at_ns - _HOUR_NS, 100, price=50000.0, fee=2.5)]
    payments = [FundingPayment("BTCUSDT", at_ns - 2 * _HOUR_NS, 1.5, False)]
    assert equity_at(10_000.0, fills, payments, {"BTCUSDT": 51000.0}, at_ns) == 10096.0


def test_events_at_instant_are_excluded() -> None:
    """R22 half: only events with ``ts < at_ns`` move the equity."""
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
    """R47: held = integer step sum, so +100,+200,-300 (float fsum 2.78e-17) is flat, no price."""
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
    """R47: a still-held leg is marked at float(Decimal(step_sum) * step), not a float fsum."""
    fills = [
        _fill("BTCUSDT", 1_000, 100, price=0.0),
        _fill("BTCUSDT", 2_000, 200, price=0.0),
        _fill("BTCUSDT", 3_000, -299, price=0.0),
    ]
    assert equity_at(10_000.0, fills, [], {"BTCUSDT": 1000.0}, 10_000) == 10_001.0
    with pytest.raises(CarryDataError, match="no price for held BTCUSDT"):
        equity_at(10_000.0, fills, [], {}, 10_000)


def test_equity_at_inconsistent_step_fails_closed() -> None:
    """R47: fills of one symbol must share one step, else the ledger is corrupt."""
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
