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

from tests.unit.test_carry_bars import (
    DEFAULT_FILTERS,
    HOUR_MS,
    hourly,
    kline,
    ms,
    ns,
    use_snapshot,
    write_archive,
)
from vibe_quant.data.catalog import create_instrument
from vibe_quant.data.mark_bars import MarkSeries
from vibe_quant.metrics import BarSeries
from vibe_quant.portfolio import reference as reference_module
from vibe_quant.portfolio import sizing as sizing_module
from vibe_quant.portfolio import strategy as strategy_module
from vibe_quant.portfolio.ledger import CarryFill, FundingPayment
from vibe_quant.portfolio.metrics import CarryReport, carry_metrics, provenance
from vibe_quant.portfolio.parity import compare_runs
from vibe_quant.portfolio.reference import simulate_reference
from vibe_quant.portfolio.runner import run_carry_engine
from vibe_quant.portfolio.signal import Settlements
from vibe_quant.portfolio.sizing import Decision, InstrumentSpec, OrderIntent, spread_cost
from vibe_quant.portfolio.spec import (
    CarryDataError,
    CarryExecutionError,
    CarrySpec,
    decision_times,
)
from vibe_quant.portfolio.ticks import (
    CarryInputs,
    PriceTick,
    boundary_prices,
    load_carry_inputs,
    select_ticks,
    trend_closes,
)
from vibe_quant.validation.funding import FundingCalculator, clear_rate_cache

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
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


def _inputs(
    tmp_path: Path,
    *,
    name: str = "a",
    spec: CarrySpec | None = None,
    vol_closes: dict[str, dict[int, float]] | None = None,
    quote_volume: dict[str, dict[int, float]] | None = None,
) -> CarryInputs:
    """The synthetic four-symbol carry run over six weekly decisions."""
    clear_rate_cache()
    spec = CarrySpec(symbols=_SYMBOLS, start_ns=START, end_ns=END) if spec is None else spec
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
        vol_closes=vol_closes,
        quote_volume=quote_volume,
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
    if value is None:  # Decision.vols at equal weighting
        return {"__drift__": 1.0}
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


# ---------------------------------------------------------------------------
# Variant parity (x0axf.14 T4 R25): inverse_vol vs equal, both divisor 2
# ---------------------------------------------------------------------------

_VOL_LOOKBACK = 5
# Alternating daily closes: the swing (and so sigma) differs per symbol.
_SWING = {"BTCUSDT": 0.01, "ETHUSDT": 0.03, "SOLUSDT": 0.02, "BNBUSDT": 0.04}


def _vol_closes() -> dict[str, dict[int, float]]:
    first = START - _VOL_LOOKBACK * _DAY
    return {
        symbol: {
            day: _MARK[symbol] * (1.0 + _SWING[symbol] * (-1) ** index)
            for index, day in enumerate(range(first, END + _DAY, _DAY))
        }
        for symbol in _SYMBOLS
    }


def _divisor_two(weighting: str) -> CarrySpec:
    return CarrySpec(
        symbols=_SYMBOLS,
        start_ns=START,
        end_ns=END,
        legs_divisor=2,
        weighting=weighting,
        vol_lookback_days=_VOL_LOOKBACK if weighting == "inverse_vol" else 30,
    )


def _sides(decision: Decision) -> tuple[frozenset[str], frozenset[str]]:
    longs = frozenset(s for s, w in decision.weights.items() if w > 0.0)
    shorts = frozenset(s for s, w in decision.weights.items() if w < 0.0)
    return longs, shorts


def _reweighted(variant: CarryRun, control: CarryRun) -> list[int]:
    """Decisions (by ts) with 2 legs a side, the control's selected sets, a changed
    absolute weight within a side, and a vol for every selected leg."""
    by_ts = {decision.ts: decision for decision in control.decisions}
    hits: list[int] = []
    for decision in variant.decisions:
        other = by_ts.get(decision.ts)
        longs, shorts = _sides(decision)
        if other is None or len(longs) != 2 or len(shorts) != 2 or _sides(other) != (longs, shorts):
            continue
        changed = any(
            len({abs(decision.weights[s]) for s in side}) > 1
            and any(abs(decision.weights[s]) != abs(other.weights[s]) for s in side)
            for side in (longs, shorts)
        )
        if changed and decision.vols is not None and (longs | shorts) <= set(decision.vols):
            hits.append(decision.ts)
    return hits


def test_variant_parity_inverse_vol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R25: engine == reference under inverse_vol; it really re-weights vs a matched control."""
    variant_inputs = _inputs(
        tmp_path, name="iv", spec=_divisor_two("inverse_vol"), vol_closes=_vol_closes()
    )
    control_inputs = _inputs(tmp_path, name="eq", spec=_divisor_two("equal"))
    engine = run_carry_engine(variant_inputs)
    reference = simulate_reference(variant_inputs)
    control = simulate_reference(control_inputs)

    assert (
        compare_runs(
            engine,
            reference,
            carry_metrics(engine, variant_inputs),
            carry_metrics(reference, variant_inputs),
        )
        == []
    )
    hits = _reweighted(reference, control)
    assert hits == [_DECISIONS[i] for i in (0, 1, 3, 4, 5)]  # D3 is flat (funding hole)
    first = reference.decisions[0]
    # D1: longs BTC (swing .01) / ETH (.03), shorts BNB (.04) / SOL (.02), gross 1.0;
    # sigma ratio ~1:3 and ~1:2 -> ~0.375/0.125 and ~0.333/0.167 (statistics.stdev
    # cross-check agrees to 1e-16)
    assert first.weights == {
        "BNBUSDT": -0.16653322658126513,
        "BTCUSDT": 0.3750750225067522,
        "ETHUSDT": 0.12492497749324782,
        "SOLUSDT": -0.3334667734187349,
    }
    assert control.decisions[0].weights == {
        "BNBUSDT": -0.25,
        "BTCUSDT": 0.25,
        "ETHUSDT": 0.25,
        "SOLUSDT": -0.25,
    }
    assert set(first.vols or {}) == set(_SYMBOLS)
    assert set(reference.decisions[2].vols or {}) == {"BTCUSDT"}  # D3: the only rankable leg
    assert control.decisions[0].vols is None

    # a no-op inverse_vol (identity re-weighting) must fail the predicate
    monkeypatch.setattr(
        "vibe_quant.portfolio.sizing.inverse_vol_weights",
        lambda weights, vols, gross: dict(weights),
    )
    clear_rate_cache()
    assert _reweighted(simulate_reference(variant_inputs), control) == []


# ---------------------------------------------------------------------------
# Variant parity (x0axf.14 T5 R37): liquidity filter vs a matched threshold-off control
# ---------------------------------------------------------------------------

# SOL trades thin in the week before D4: mean 1e6 < 1e8 at D4 only.
_THIN = range(START + 2 * _WEEK, START + 3 * _WEEK, _DAY)


def _quote_volume() -> dict[str, dict[int, float]]:
    return {
        symbol: {
            day: 1e6 if symbol == "SOLUSDT" and day in _THIN else 1e9
            for day in range(START - _WEEK, END, _DAY)
        }
        for symbol in _SYMBOLS
    }


def test_variant_parity_liquidity(tmp_path: Path) -> None:
    """R37: engine == reference with the filter on; SOL, selected by the control at D4, is
    ``illiquid`` there in both executors (signal None, target 0)."""
    spec = CarrySpec(symbols=_SYMBOLS, start_ns=START, end_ns=END, min_quote_volume=1e8)
    variant_inputs = _inputs(tmp_path, name="liq", spec=spec, quote_volume=_quote_volume())
    control_inputs = _inputs(tmp_path, name="ctl")
    engine = run_carry_engine(variant_inputs)
    reference = simulate_reference(variant_inputs)
    control = simulate_reference(control_inputs)

    assert (
        compare_runs(
            engine,
            reference,
            carry_metrics(engine, variant_inputs),
            carry_metrics(reference, variant_inputs),
        )
        == []
    )
    d4 = _DECISIONS[3]
    assert control.decisions[3].ts == d4
    assert control.decisions[3].weights["SOLUSDT"] == -0.5  # the control shorts SOL at D4
    for run in (engine, reference):
        decision = run.decisions[3]
        assert decision.ts == d4
        assert decision.excluded == {"SOLUSDT": "illiquid"}
        assert decision.signals["SOLUSDT"] is None
        assert decision.target_steps["SOLUSDT"] == 0
        assert decision.prices["SOLUSDT"] == 150.0
        # 3 rankable -> k 1: long BTC (lowest), short BNB (highest left)
        assert {s: w for s, w in decision.weights.items() if w} == {
            "BTCUSDT": 0.5,
            "BNBUSDT": -0.5,
        }
    # only D4 differs: every other decision matches the control's exclusions
    assert [d.excluded for i, d in enumerate(reference.decisions) if i != 3] == [
        d.excluded for i, d in enumerate(control.decisions) if i != 3
    ]


# ---------------------------------------------------------------------------
# arch-c9-2 trend executors (T4): R15, R17, R32, R38, R39, R51, R52; dreview C9-1,
# C9-4 (mixed cohort with a not-yet-listed coin), C9-18 (executor prefix checks)
# ---------------------------------------------------------------------------

_T_PAIR = ("BTCUSDT", "ETHUSDT")
_T_FIRST = ms("2023-09-01T00:00:00")
_T_LAST = ms("2024-03-25T00:00:00")
_T_SPREAD = 0.0001
_T_FILTERS = {**DEFAULT_FILTERS, "SOLUSDT": {"tickSize": "0.01"}}


def _rising(first_ms: int, last_ms: int, base: float) -> list[tuple[float | int, ...]]:
    """Hourly closes rising 0.01 per hour: every trend term +1, a steady long."""
    return [
        kline(open_ms, round(base + 0.01 * index, 2))
        for index, open_ms in enumerate(range(first_ms, last_ms, HOUR_MS))
    ]


def _trend(
    start: str = "2024-03-01",
    end: str = "2024-03-20",
    *,
    symbols: tuple[str, ...] = _T_PAIR,
    tier: str = "1h",
) -> CarrySpec:
    return CarrySpec.from_dates(
        symbols,
        start,
        end,
        signal="trend",
        schedule="daily",
        weighting="time_series",
        direction="long_short",
        vol_target=0.4,
        fill_granularity=tier,
        spread=_T_SPREAD,
    )


def _hour_archive(
    path: Path,
    eth: list[tuple[float | int, ...]] | None = None,
    *,
    last_ms: int = _T_LAST,
    btc: list[tuple[float | int, ...]] | None = None,
    sol: bool = False,
) -> Path:
    """BTC (cycling, so long_short flips) and ETH (rising) 1h bars with 8h funding.

    ``sol`` adds SOLUSDT bars from 2024-04-01: listed after every window here.
    """
    klines = {
        "BTCUSDT": hourly(_T_FIRST, last_ms, 100.0) if btc is None else btc,
        "ETHUSDT": _rising(_T_FIRST, last_ms, 50.0) if eth is None else eth,
    }
    if sol:
        first = ms("2024-04-01T00:00:00")
        klines["SOLUSDT"] = hourly(first, first + 5 * 24 * HOUR_MS, 20.0)
    return write_archive(path, klines, dict.fromkeys(_T_PAIR, _T_FIRST), funding_to_ms=last_ms)


def _both(inputs: CarryInputs, **nt: Any) -> tuple[CarryRun, CarryRun]:
    clear_rate_cache()
    engine = run_carry_engine(inputs, **nt)
    clear_rate_cache()
    return engine, simulate_reference(inputs)


def _same(one: CarryRun, other: CarryRun, inputs: CarryInputs) -> list[str]:
    return compare_runs(one, other, carry_metrics(one, inputs), carry_metrics(other, inputs))


def test_trend_parity_hour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R51/R52 at 1h: NT == reference over every field (spread, Decimal exposure, the
    funding counters), with reversed data and reversed --symbols; a coin listed after
    the end stays ``not_listed`` with no ticks and the mixed cohort still runs (C9-4)."""
    archive = _hour_archive(tmp_path / "a.db", sol=True)
    use_snapshot(tmp_path, monkeypatch, archive, filters=_T_FILTERS)
    spec = _trend(symbols=("SOLUSDT", "ETHUSDT", "BTCUSDT"))
    inputs = load_carry_inputs(spec, tmp_path / "no-catalog", archive)
    engine, reference = _both(inputs)
    reversed_engine, _ = _both(inputs, reverse_data=True)
    flipped = load_carry_inputs(
        _trend(symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT")), tmp_path / "no-catalog", archive
    )
    flipped_reference = simulate_reference(flipped)

    assert _same(engine, reference, inputs) == []
    assert _same(reversed_engine, reference, inputs) == []
    assert _same(flipped_reference, reference, inputs) == []
    assert inputs.ticks["SOLUSDT"] == ()
    assert all(d.excluded.get("SOLUSDT") == "not_listed" for d in reference.decisions)
    assert len(reference.decisions) == 19
    evidence = provenance(reference, inputs, spread_source="test", universe=None)
    assert evidence["price_source"] == {
        "SOLUSDT": None,  # not listed before end: no data, no source
        "ETHUSDT": "archive_1h",
        "BTCUSDT": "archive_1h",
    }
    assert evidence["mark_source"] == {
        "SOLUSDT": None,
        "ETHUSDT": "archive_1h_last_close",
        "BTCUSDT": "archive_1h_last_close",
    }
    # the fixture really trades: opens, reductions/closes, funding, a spread on every fill
    assert any(o.reduce_only for d in reference.decisions for o in d.orders)
    assert reference.payments
    assert reference.fills
    for fill in reference.fills:
        assert fill.spread == spread_cost(fill.qty, fill.price, _T_SPREAD) > 0.0
    assert all(d.exposure_after_plan is not None for d in reference.decisions)


def _minute_bars(base: float, drift: float, spec: CarrySpec, decisions: Sequence[int]) -> BarSeries:
    """1m bars: the 23:59 close of every day from start - 170 d (wiggling, so the trend
    terms change sign) plus the sizing (00:59) and fill (01:00) bars of every decision."""
    first_day = spec.start_ns - 170 * _DAY
    by_open: dict[int, float] = {}
    for k, boundary in enumerate(range(first_day, spec.end_ns + _DAY, _DAY)):
        wiggle = (0.0, 0.02, -0.01, 0.015, -0.02)[k % 5]
        by_open[boundary - _MIN] = round(base * (1.0 + drift * (k % 57) + wiggle), 1)
    for index, decision_ns in enumerate(decisions):
        by_open[decision_ns - _MIN] = round(base * (1.01 + 0.002 * index), 1)
        by_open[decision_ns] = round(base * (1.012 + 0.002 * index), 1)
    opens = sorted(by_open)
    closes = np.array([by_open[o] for o in opens], dtype=np.float64)
    ts = np.array([o + _MIN - 1_000_000 for o in opens], dtype=np.int64)
    return BarSeries(ts=ts, open=closes, high=closes, low=closes, close=closes)


def _minute_trend_inputs(tmp_path: Path) -> CarryInputs:
    """A hand-built 1m trend run (catalog instruments, 1m MARK, causal funding)."""
    clear_rate_cache()
    spec = _trend("2024-03-01", "2024-03-09", tier="1m")
    decisions = tuple(decision_times(spec.start_ns, spec.end_ns, "daily"))
    archive = write_archive(
        tmp_path / "minute.db",
        {},
        dict.fromkeys(_T_PAIR, ms("2024-02-20T00:00:00")),
        funding_to_ms=ms("2024-03-10T00:00:00"),
    )
    calc = FundingCalculator(archive)
    instruments = {symbol: create_instrument(symbol) for symbol in _T_PAIR}
    shape = {"BTCUSDT": (50_000.0, 0.004), "ETHUSDT": (3_000.0, -0.003)}
    bars = {s: _minute_bars(*shape[s], spec, decisions) for s in _T_PAIR}
    trend_days = range(spec.start_ns - 160 * _DAY, spec.end_ns + _DAY, _DAY)
    days = range(spec.start_ns, spec.end_ns + _DAY, _DAY)
    minutes = np.arange(spec.start_ns - _EIGHT_HOURS, spec.end_ns, _MIN, dtype=np.int64)
    return CarryInputs(
        spec=spec,
        decisions=decisions,
        instruments=instruments,
        specs={s: InstrumentSpec.from_instrument(instruments[s]) for s in _T_PAIR},
        ticks={s: select_ticks(bars[s], decisions, instruments[s], spec.end_ns) for s in _T_PAIR},
        settlements={
            s: Settlements(times_ns=calc.settlements(s)[0], rates=calc.settlements(s)[1])
            for s in _T_PAIR
        },
        marks={
            s: MarkSeries(
                ts=minutes,
                low=np.full(minutes.shape, shape[s][0]),
                high=np.full(minutes.shape, shape[s][0]),
                close=np.full(minutes.shape, shape[s][0]),
                open=np.full(minutes.shape, shape[s][0]),
            )
            for s in _T_PAIR
        },
        calc=calc,
        boundary_prices={s: boundary_prices(bars[s], days, instruments[s]) for s in _T_PAIR},
        trend_closes={s: trend_closes(bars[s], trend_days, instruments[s]) for s in _T_PAIR},
        listings={s: int(bars[s].ts[0]) - (_MIN - 1_000_000) for s in _T_PAIR},
    )


def test_trend_parity_minute(tmp_path: Path) -> None:
    """R51 at 1m: NT == reference on a long_short book that flips, with the 1m MARK,
    catalog fees and the causal funding grid; the costs say where the fee came from."""
    inputs = _minute_trend_inputs(tmp_path)
    engine, reference = _both(inputs)
    reversed_engine, _ = _both(inputs, reverse_data=True)

    assert _same(engine, reference, inputs) == []
    assert _same(reversed_engine, reference, inputs) == []
    signs = {
        (d.target_steps["BTCUSDT"] > 0) - (d.target_steps["BTCUSDT"] < 0)
        for d in reference.decisions
    }
    assert {-1, 1} <= signs  # BTC is long at one decision and short at another
    assert reference.payments and reference.fills
    for fill in reference.fills:
        assert fill.spread == spread_cost(fill.qty, fill.price, _T_SPREAD) > 0.0
    costs = provenance(reference, inputs, spread_source="test", universe=None)["costs"]
    assert costs["fee_source"] == "catalog instrument taker_fee"
    assert costs["taker_fee"] == {"BTCUSDT": "0.0005", "ETHUSDT": "0.0005"}


def _recording(target: Callable[..., Decision], seen: list[Decision]) -> Callable[..., Decision]:
    """``decide`` that also records every Decision BEFORE any of its orders is released."""

    def wrapper(*args: Any, **kwargs: Any) -> Decision:
        decision = target(*args, **kwargs)
        seen.append(decision)
        return decision

    return wrapper


def test_trend_decision_ignores_one_legs_future_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C9-1 (+ the C9-18 sizing-time and decision-bar mutants): a two-leg 1h backtest; the
    Decision at D, captured before release, is equal in both executors when only ETH's bars
    opening >= D are repriced x10 or deleted. Deleted, ETH's ordered leg has no fill bar:
    both executors stop with ONE message (R15)."""
    decision_ns = ns("2024-03-01T01:00:00")
    decision_ms = decision_ns // 1_000_000
    full = _rising(_T_FIRST, _T_LAST, 50.0)
    variants = {
        "full": full,
        "x10": [row if row[0] < decision_ms else _times10(row) for row in full],
        "deleted": [row for row in full if row[0] < decision_ms],
    }
    btc = _rising(_T_FIRST, _T_LAST, 100.0)
    use_snapshot(tmp_path, monkeypatch, _hour_archive(tmp_path / "full.db", full, btc=btc))
    captured: dict[tuple[str, str], Decision | None] = {}
    errors: dict[tuple[str, str], str] = {}
    for name, eth in variants.items():
        archive = _hour_archive(tmp_path / f"{name}.db", eth, btc=btc)
        inputs = load_carry_inputs(_trend(end="2024-03-04"), tmp_path / "no-catalog", archive)
        for executor, module, run in (
            ("reference", reference_module, simulate_reference),
            ("nt", strategy_module, run_carry_engine),
        ):
            seen: list[Decision] = []
            monkeypatch.setattr(module, "decide", _recording(sizing_module.decide, seen))
            clear_rate_cache()
            try:
                run(inputs)
            except CarryExecutionError as error:
                errors[(executor, name)] = str(error)
            captured[(executor, name)] = seen[0] if seen else None  # None: stopped before D

    first = captured[("reference", "full")]
    assert first is not None and first.ts == decision_ns
    assert [o.symbol for o in first.orders] == ["BTCUSDT", "ETHUSDT"]
    assert all(decision == first for decision in captured.values()), captured
    unreleased = (
        f"unreleased orders for ETHUSDT at {decision_ns}: "
        "no own tick after the decision before the run ends"
    )
    assert errors == {("reference", "deleted"): unreleased, ("nt", "deleted"): unreleased}


def _times10(row: tuple[float | int, ...]) -> tuple[float | int, ...]:
    open_ms, o, h, low, c, volume, close_ms = row
    return (open_ms, round(o * 10, 2), round(h * 10, 2), round(low * 10, 2), round(c * 10, 2), volume, close_ms)


def test_trend_executor_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R38 (C9-18, executor level): runs ending 2024-03-10 and 2024-03-20 agree on every
    decision up to the cutoff and every fill and payment before it, in both executors."""
    archive = _hour_archive(tmp_path / "a.db")
    use_snapshot(tmp_path, monkeypatch, archive)
    cutoff = ns("2024-03-10T00:00:00")
    short = load_carry_inputs(_trend(end="2024-03-10"), tmp_path / "no-catalog", archive)
    long = load_carry_inputs(_trend(end="2024-03-20"), tmp_path / "no-catalog", archive)
    for one, other in zip(_both(short), _both(long), strict=True):
        assert one.decisions == other.decisions[: len(one.decisions)]
        assert all(d.ts <= cutoff for d in one.decisions)
        assert len(one.decisions) == 9
        assert one.fills == tuple(f for f in other.fills if f.ts < cutoff)
        assert one.payments == tuple(p for p in other.payments if p.ts < cutoff)
        assert one.fills and one.payments


def test_trend_executor_source_truncation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R39 (executor level): an archive cut at 2024-03-25 and one running to 2024-04-10
    give the same CarryRun -- every field, the funding counters included -- in both
    executors for a run ending 2024-03-20."""
    full = _hour_archive(tmp_path / "full.db", last_ms=ms("2024-04-10T00:00:00"))
    cut = _hour_archive(tmp_path / "cut.db")
    use_snapshot(tmp_path, monkeypatch, full)
    a = load_carry_inputs(_trend(), tmp_path / "no-catalog", full)
    b = load_carry_inputs(_trend(), tmp_path / "no-catalog", cut)
    for one, other in zip(_both(a), _both(b), strict=True):
        assert _same(one, other, a) == []


def test_trend_held_leg_without_sizing_bar_fails_in_both_executors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R17 (NT half): BTC is held long when its sizing bar (opening D - 1h) is missing;
    both executors stop with ``CarryDataError("no price for held BTCUSDT at D")``."""
    hole = ms("2024-03-03T00:00:00")
    btc = [row for row in _rising(_T_FIRST, _T_LAST, 100.0) if row[0] != hole]
    archive = _hour_archive(tmp_path / "a.db", btc=btc)
    use_snapshot(tmp_path, monkeypatch, archive)
    inputs = load_carry_inputs(_trend(end="2024-03-05"), tmp_path / "no-catalog", archive)
    decision_ns = ns("2024-03-03T01:00:00")
    for run in (simulate_reference, run_carry_engine):
        clear_rate_cache()
        with pytest.raises(CarryDataError) as error:
            run(inputs)
        assert str(error.value) == f"no price for held BTCUSDT at {decision_ns}"
