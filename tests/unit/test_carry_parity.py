"""Engine-vs-reference exactness: ``parity.compare_runs`` (task T6b).

Contract rows R30 (the NT engine equals the no-NT reference field by field on the
same ``CarryInputs``, exact equality, no tolerance) and R30b (a one-step / one-ulp
drift in the reference is reported, never tolerated) of the funding-carry
semantics contract (``data/swarm/20261015-funding-carry/design.md`` §5, §8-T6b).

The fixture is a synthetic four-symbol run (BTCUSDT + ETHUSDT + SOLUSDT +
BNBUSDT) over six weekly decisions, built so the path really covers the four
cases the row asks for:

* a **flip** at D2 -- BTC long -> short and SOL short -> long, each a reduce-only
  close followed by an open on the same release tick (R14c);
* a **band skip** at D5 and D6 -- BTC's sizing price moves 51,000 -> 51,500 at
  D5, so every resize lands inside the 0.10 band and nothing is sent (R14d);
* a **funding hole** in D3's 7-day window for ETH/SOL/BNB -- only BTC stays
  rankable, ``n = 1``, so the book goes flat; the rankable count walks 4 -> 1 -> 4
  (R8b, R49);
* a **funding fallback** charged on the held SOL leg at the uncovered boundary
  (R18).

One nonzero rate per decision, placed at the 8h boundary that decision's window
opens on (``D_k - 7 d + 8 h``); every other archived boundary carries rate 0.0,
so the signal of each decision is that one rate and the funding payment is
``qty x mark x rate`` at that boundary. MARK is flat at the per-symbol mark.

Hand numbers (E0 = 10,000, divisor 3, gross 1.0, band 0.10, MARK flat):

* D1, equity 10,000: BTC 0.0001 (lowest) -> long +98 steps at 51,000 x 0.001 and
  SOL 0.0004 (highest) -> short -3,333 steps at 150 x 0.01 (R11); both fill at
  ``D1 + 59.999 s`` for 2.49949 / 2.5014165 (R15).
* D2, equity 9995.3530435 (BTC pays ``0.098 x 51,000 x 0.0005 = 2.499``, SOL
  receives ``33.33 x 150 x 0.0001 = 0.49995``): the ranking reversed sizes
  -97 / +3,331 steps, so both legs flip (R14c).
* D3, equity 9980.159636499999 (the held SOL leg pays the fallback 0.49965 at
  the uncovered boundary): BTC alone rankable -> flat, both legs closed (R8b, R18).
* D4, equity 9977.546736: +97 / -3,325 steps open again.
* D5 / D6, equity 10024.9326385 / 10026.432938500002: at BTC's new 51,500 sizing
  tick BTC's target stays 97 steps while SOL drifts to -3,341 / -3,342 -- 16 and
  17 steps from the held book, inside the 0.10 band, so one skip each and no
  fills.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest

from vibe_quant.data.catalog import create_instrument
from vibe_quant.data.mark_bars import MarkSeries
from vibe_quant.portfolio.ledger import CarryFill, FundingPayment
from vibe_quant.portfolio.metrics import CarryReport, carry_metrics
from vibe_quant.portfolio.parity import compare_runs
from vibe_quant.portfolio.reference import simulate_reference
from vibe_quant.portfolio.runner import run_carry_engine
from vibe_quant.portfolio.signal import Settlements
from vibe_quant.portfolio.sizing import Decision, InstrumentSpec, OrderIntent
from vibe_quant.portfolio.spec import CarrySpec, decision_times
from vibe_quant.portfolio.ticks import CarryInputs, PriceTick
from vibe_quant.validation.funding import FundingCalculator, clear_rate_cache

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

    from vibe_quant.portfolio.ledger import CarryRun

_MIN = 60 * 10**9
_HOUR = 3_600 * 10**9
_DAY = 86_400 * 10**9
_WEEK = 7 * _DAY
_EIGHT_HOURS = 8 * _HOUR
_BAR_TS_INIT = 59_999_000_000  # catalog 1m bar: ts_init = open + 59.999 s
_JITTER_MS = 3  # archived funding_time carries a few ms after the boundary
_JITTER_NS = _JITTER_MS * 1_000_000
_MINUTE_MS = 60_000

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
_MARK = {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0, "SOLUSDT": 150.0, "BNBUSDT": 500.0}
_SIZING = {
    "BTCUSDT": (51000.0,) * 4 + (51500.0,) * 2,
    "ETHUSDT": (3000.0,) * 6,
    "SOLUSDT": (150.0,) * 6,
    "BNBUSDT": (500.0,) * 6,
}
_FILL = {
    "BTCUSDT": (51010.0,) * 4 + (51510.0,) * 2,
    "ETHUSDT": (3010.0,) * 6,
    "SOLUSDT": (150.1,) * 6,
    "BNBUSDT": (500.5,) * 6,
}
# The signal of each decision, carried by the single nonzero rate in its window.
_SIGNALS = (
    {"BTCUSDT": 0.0001, "ETHUSDT": 0.0002, "BNBUSDT": 0.0003, "SOLUSDT": 0.0004},
    {"SOLUSDT": 0.0001, "BNBUSDT": 0.0002, "ETHUSDT": 0.0003, "BTCUSDT": 0.0005},
    dict.fromkeys(_SYMBOLS, 0.0001),
    {"BTCUSDT": 0.0001, "ETHUSDT": 0.0002, "BNBUSDT": 0.0003, "SOLUSDT": 0.0004},
    {"BTCUSDT": 0.0001, "ETHUSDT": 0.0002, "BNBUSDT": 0.0003, "SOLUSDT": 0.0004},
    {"BTCUSDT": 0.0001, "ETHUSDT": 0.0002, "BNBUSDT": 0.0003, "SOLUSDT": 0.0004},
)
# D3's hole: an uncovered 8h boundary inside its lookback window (R3, R49).
_HOLE = None  # set below, once START is known
_HOLE_SYMBOLS = frozenset({"ETHUSDT", "SOLUSDT", "BNBUSDT"})


def _ns(date: str) -> int:
    """``YYYY-MM-DD`` 00:00 UTC -> epoch ns."""
    return int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp()) * 10**9


START = _ns("2024-06-03")  # a Monday 00:00 UTC
END = START + 6 * _WEEK  # six weekly decisions, the last one a week before END
_DECISIONS = tuple(decision_times(START, END))
_HOLE = START + _WEEK + 16 * _HOUR  # Monday week 2, 16:00 -- in D3's window
_SIGNAL_AT = {
    START + (index - 1) * _WEEK + 8 * _HOUR: signals
    for index, signals in enumerate(_SIGNALS)  # D_{index+1}'s window opens on Mon_{index-1}
}


def _rate(symbol: str, boundary: int) -> float:
    """The archived rate of ``symbol`` at an 8h boundary (0.0 unless it is a signal row)."""
    return _SIGNAL_AT.get(boundary, {}).get(symbol, 0.0)


def _boundaries() -> list[int]:
    """Every 8h UTC boundary the six signal windows (and their funding) need."""
    return list(range(START - 8 * _DAY, END, _EIGHT_HOURS))


def _kept(symbol: str) -> list[int]:
    """The boundaries with an archived row: D3's hole is missing for three symbols."""
    return [
        boundary
        for boundary in _boundaries()
        if not (symbol in _HOLE_SYMBOLS and boundary == _HOLE)
    ]


def _settlement(symbol: str) -> Settlements:
    """The symbol's archived settlement series, mirrored by ``_archive``."""
    boundaries = _kept(symbol)
    return Settlements(
        times_ns=tuple(boundary + _JITTER_NS for boundary in boundaries),
        rates=tuple(_rate(symbol, boundary) for boundary in boundaries),
    )


def _archive(path: Path) -> FundingCalculator:
    """A tmp funding archive: one 8h row per boundary, ms-jittered like Binance."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE raw_funding_rates (
            id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
            funding_rate REAL NOT NULL, mark_price REAL, source TEXT,
            UNIQUE(symbol, funding_time))"""
    )
    for symbol in _SYMBOLS:
        for boundary in _kept(symbol):
            conn.execute(
                "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate) VALUES (?,?,?)",
                (symbol, boundary // 1_000_000 + _JITTER_MS, _rate(symbol, boundary)),
            )
    conn.commit()
    conn.close()
    return FundingCalculator(path)


def _marks(price: float) -> MarkSeries:
    """A flat 1m MARK series over the fixture window (one bar opening per minute)."""
    ts = np.arange(START - _EIGHT_HOURS, END, _MIN, dtype=np.int64)
    values = np.full(ts.shape, price, dtype=np.float64)
    return MarkSeries(ts=ts, low=values, high=values, close=values, open=values)


def _ticks() -> dict[str, tuple[PriceTick, ...]]:
    """Per decision: the sizing tick (``D-1 ms``) then the fill tick (``D+59.999 s``)."""
    return {
        symbol: tuple(
            tick
            for index, decision_ns in enumerate(_DECISIONS)
            for tick in (
                PriceTick(decision_ns - 1_000_000, _SIZING[symbol][index]),
                PriceTick(decision_ns + _BAR_TS_INIT, _FILL[symbol][index]),
            )
        )
        for symbol in _SYMBOLS
    }


def _inputs(tmp_path: Path, *, name: str = "a") -> CarryInputs:
    """The synthetic four-symbol carry run over six weekly decisions."""
    clear_rate_cache()
    spec = CarrySpec(symbols=_SYMBOLS, start_ns=START, end_ns=END)
    instruments = {symbol: create_instrument(symbol) for symbol in spec.symbols}
    return CarryInputs(
        spec=spec,
        decisions=_DECISIONS,
        instruments=instruments,
        specs={
            symbol: InstrumentSpec.from_instrument(instrument)
            for symbol, instrument in instruments.items()
        },
        ticks=_ticks(),
        settlements={symbol: _settlement(symbol) for symbol in spec.symbols},
        marks={symbol: _marks(_MARK[symbol]) for symbol in spec.symbols},
        calc=_archive(tmp_path / f"raw_{name}.db"),
        # flat closes for every daily boundary so both executors' metrics agree
        boundary_prices={
            symbol: {boundary: _MARK[symbol] for boundary in range(START, END + _DAY, _DAY)}
            for symbol in spec.symbols
        },
    )


class _Parity(NamedTuple):
    """One fixture run through both executors and both metric passes."""

    inputs: CarryInputs
    engine: CarryRun
    reference: CarryRun
    engine_report: CarryReport
    reference_report: CarryReport


@pytest.fixture(scope="module")
def parity(tmp_path_factory: pytest.TempPathFactory) -> _Parity:
    """Run the fixture once per module: NT engine, reference, and both reports."""
    inputs = _inputs(tmp_path_factory.mktemp("carry-parity"))
    engine = run_carry_engine(inputs)
    reference = simulate_reference(inputs)
    return _Parity(
        inputs,
        engine,
        reference,
        carry_metrics(engine, inputs),
        carry_metrics(reference, inputs),
    )


def _different(value: object) -> object:
    """A value guaranteed to differ from ``value``, for any field type in the fixture.

    R54's drift tests need one mutation per dataclass field; this walks whatever
    the field holds (scalar, dict, tuple, nested dataclass like ``OrderIntent``)
    and returns a value ``==`` comparison must reject.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return 0.0 if math.isnan(value) else math.nextafter(value, math.inf)
    if isinstance(value, str):
        return value + "!"
    if is_dataclass(value) and not isinstance(value, type):
        first = fields(value)[0]
        return replace(value, **{first.name: _different(getattr(value, first.name))})
    if isinstance(value, Mapping):
        if not value:
            return {"__drift__": 1}
        mapping = dict(value)
        key = next(iter(mapping))
        mapping[key] = _different(mapping[key])
        return mapping
    if isinstance(value, (tuple, list)):
        items = list(value) or [1]
        items[0] = _different(items[0])
        return tuple(items) if isinstance(value, tuple) else items
    raise AssertionError(f"no drift mutation for {value!r}")


def _with_field(instance: Any, field_name: str) -> Any:
    """``dataclasses.replace`` with a dynamic field name (one R54 mutant)."""
    return replace(instance, **{field_name: _different(getattr(instance, field_name))})


def _with_first_item_moved(items: Sequence[Any], field_name: str) -> tuple[Any, ...]:
    """Copy of ``items`` whose first element has ``field_name`` changed."""
    return (_with_field(items[0], field_name), *items[1:])


# Every dataclass field of every compared object, plus the run's own fields.
_DRIFT_FIELDS: list[tuple[str, str]] = (
    [("fill", field.name) for field in fields(CarryFill)]
    + [("decision", field.name) for field in fields(Decision)]
    + [("payment", field.name) for field in fields(FundingPayment)]
    + [("report", field.name) for field in fields(CarryReport)]
    + [("run", "final_steps"), ("run", "spec")]
)


def _drifted(parity: _Parity, kind: str, field_name: str) -> tuple[CarryRun, CarryReport]:
    """The reference run/report with exactly one field changed via dataclasses.replace."""
    reference, report = parity.reference, parity.reference_report
    if kind == "fill":
        return replace(reference, fills=_with_first_item_moved(reference.fills, field_name)), report
    if kind == "decision":
        return (
            replace(reference, decisions=_with_first_item_moved(reference.decisions, field_name)),
            report,
        )
    if kind == "payment":
        return (
            replace(reference, payments=_with_first_item_moved(reference.payments, field_name)),
            report,
        )
    if kind == "run":
        return _with_field(reference, field_name), report
    return reference, _with_field(report, field_name)


def test_engine_matches_reference_on_fixture(parity: _Parity) -> None:
    """Exact equality -- flips, a band skip, a funding hole and a fallback."""
    engine, reference = parity.engine, parity.reference

    assert compare_runs(engine, reference, parity.engine_report, parity.reference_report) == []
    assert parity.engine_report.summary == parity.reference_report.summary
    assert (engine.engine, reference.engine) == ("nt", "reference")

    # the four cases the row asks for really happen in this fixture
    rankable = [
        sum(value is not None for value in decision.signals.values())
        for decision in engine.decisions
    ]
    assert rankable == [4, 4, 1, 4, 4, 4]  # n 4 -> 1 -> 4
    # Both executors build Decision.excluded identically (reason included)
    assert engine.decisions[2].excluded == dict.fromkeys(_HOLE_SYMBOLS, "funding_hole")
    assert engine.decisions[2].excluded == reference.decisions[2].excluded
    assert engine.decisions[2].weights == dict.fromkeys(_SYMBOLS, 0.0)
    assert engine.decisions[2].orders == (
        OrderIntent("BTCUSDT", 97, reduce_only=True),
        OrderIntent("SOLUSDT", -3331, reduce_only=True),
    )
    assert engine.decisions[0].weights == {
        "BNBUSDT": 0.0,
        "BTCUSDT": 0.5,
        "ETHUSDT": 0.0,
        "SOLUSDT": -0.5,
    }
    assert engine.decisions[1].weights == {
        "BNBUSDT": 0.0,
        "BTCUSDT": -0.5,
        "ETHUSDT": 0.0,
        "SOLUSDT": 0.5,
    }
    assert engine.decisions[1].target_steps == {
        "BNBUSDT": 0,
        "BTCUSDT": -97,
        "ETHUSDT": 0,
        "SOLUSDT": 3331,
    }
    assert engine.decisions[3].target_steps == {
        "BNBUSDT": 0,
        "BTCUSDT": 97,
        "ETHUSDT": 0,
        "SOLUSDT": -3325,
    }
    assert [decision.band_skips for decision in engine.decisions] == [0, 0, 0, 0, 1, 1]

    # The first week's hand numbers, both legs
    assert engine.fills[:2] == (
        CarryFill("BTCUSDT", _DECISIONS[0] + _BAR_TS_INIT, 98, 0.098, 51010.0, 2.49949, False),
        CarryFill("SOLUSDT", _DECISIONS[0] + _BAR_TS_INIT, -3333, -33.33, 150.1, 2.5014165, False),
    )
    assert len(engine.fills) == 10  # 2 opens + 4 flip fills + 2 closes + 2 re-opens
    assert engine.final_steps == {"BNBUSDT": 0, "BTCUSDT": 97, "ETHUSDT": 0, "SOLUSDT": -3325}

    # Exactly one fallback, for the held SOL leg at the uncovered boundary
    fallback = [payment for payment in engine.payments if payment.fallback]
    assert fallback == [FundingPayment("SOLUSDT", _HOLE, pytest.approx(0.49965), True)]
    assert fallback == [payment for payment in reference.payments if payment.fallback]


def test_compare_runs_detects_drift(parity: _Parity) -> None:
    """A one-step reference fill and a one-ulp reference fee are reported."""
    assert (
        compare_runs(parity.engine, parity.reference, parity.engine_report, parity.reference_report)
        == []
    )

    first = parity.reference.fills[0]
    rest = parity.reference.fills[1:]
    one_step = replace(parity.reference, fills=(replace(first, steps=first.steps + 1), *rest))
    ulp = replace(
        parity.reference, fills=(replace(first, fee=math.nextafter(first.fee, math.inf)), *rest)
    )

    step_messages = compare_runs(
        parity.engine, one_step, parity.engine_report, parity.reference_report
    )
    fee_messages = compare_runs(parity.engine, ulp, parity.engine_report, parity.reference_report)

    assert len(step_messages) == 1 and "fills[0].steps" in step_messages[0]
    assert len(fee_messages) == 1 and "fills[0].fee" in fee_messages[0]


@pytest.mark.parametrize(("kind", "field_name"), _DRIFT_FIELDS)
def test_compare_runs_reports_every_dataclass_field(
    parity: _Parity, kind: str, field_name: str
) -> None:
    """One drift per dataclass field is reported, naming that exact field."""
    assert (
        compare_runs(parity.engine, parity.reference, parity.engine_report, parity.reference_report)
        == []
    )

    reference, report = _drifted(parity, kind, field_name)
    mismatches = compare_runs(parity.engine, reference, parity.engine_report, report)

    assert len(mismatches) == 1, mismatches
    assert field_name in mismatches[0], mismatches


def test_compare_runs_reports_excluded_drift(parity: _Parity) -> None:
    """The reference's excluded reason is compared, not skipped.

    The reviewer's mutant -- reference ``excluded['ETHUSDT']`` ``funding_hole`` ->
    ``no_price`` -- must be reported, so engine == reference is proven for R49.
    """
    reference = parity.reference
    excluded = dict(reference.decisions[2].excluded)
    assert excluded == dict.fromkeys(_HOLE_SYMBOLS, "funding_hole")
    excluded["ETHUSDT"] = "no_price"
    decisions = list(reference.decisions)
    decisions[2] = replace(decisions[2], excluded=excluded)
    drifted = replace(reference, decisions=tuple(decisions))

    mismatches = compare_runs(
        parity.engine, drifted, parity.engine_report, parity.reference_report
    )

    assert any("decisions[2].excluded" in message for message in mismatches), mismatches


def test_compare_runs_ignores_the_engine_field(parity: _Parity) -> None:
    """``CarryRun.engine`` ('nt' vs 'reference') is the ONLY excluded field."""
    assert (parity.engine.engine, parity.reference.engine) == ("nt", "reference")
    # a third label, different from BOTH, is still ignored -- proves it is never read
    swapped = replace(parity.reference, engine="ignored-label")
    assert (
        compare_runs(parity.engine, swapped, parity.engine_report, parity.reference_report) == []
    )


def _priced(run: CarryRun, index: int, price: float) -> CarryRun:
    """``run`` with fill ``index``'s price set to ``price``: one field changed, length kept."""
    fills = list(run.fills)
    fills[index] = replace(fills[index], price=price)
    return replace(run, fills=tuple(fills))


def test_compare_runs_nan_is_equal_on_both_sides(parity: _Parity) -> None:
    """NaN == NaN is not a drift; NaN vs a number still is, on equal-length runs.

    Both sides are copies of the SAME run with the same fill count, so a count
    mismatch can never stand in for a value mismatch: the only difference is the
    one mutated field. The ``isnan(one) or isnan(other)`` mutant in
    ``parity._equal`` (which reads NaN vs a number as EQUAL) leaves case (b) with
    zero mismatches and fails here; the pre-fix test compared a 1-fill run against
    the 10-fill reference, so its count mismatch passed even under that mutant.
    """
    nan = float("nan")
    base, report = parity.reference, parity.reference_report

    # (a) the same field NaN on both sides is not a drift
    assert compare_runs(_priced(base, 0, nan), _priced(base, 0, nan), report, report) == []

    # (b) NaN on one side, a finite number on the other: exactly one mismatch, named
    nan_vs_number = compare_runs(_priced(base, 0, nan), base, report, report)
    assert len(nan_vs_number) == 1, nan_vs_number
    assert "fills[0].price" in nan_vs_number[0]

    # (c) two different finite prices: exactly one mismatch, named
    finite = compare_runs(_priced(base, 0, 1.0), _priced(base, 0, 2.0), report, report)
    assert len(finite) == 1, finite
    assert "fills[0].price" in finite[0]
