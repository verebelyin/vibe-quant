"""Funding-carry sizing, order planning and run-spec validation (task T1).

Contract rows R8d, R9, R11-R14f, R20, R20b, R43-R46 and the NT-commission
formula of the funding-carry semantics contract (``design.md`` §2, §5, §7).
Every number here is a hand-computed contract value; the Decimal paths (floor,
band, min-notional, min-quantity) are pinned at their equality edges so a
``<``/``<=`` flip fails.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from vibe_quant.data.catalog import create_instrument
from vibe_quant.errors import DataUnavailableError
from vibe_quant.portfolio.signal import Settlements, decision_signals
from vibe_quant.portfolio.sizing import (
    InstrumentSpec,
    OrderIntent,
    _scale,
    decide,
    nt_commission,
    plan_orders,
    target_steps,
    time_series_weights,
)
from vibe_quant.portfolio.spec import (
    CarryDataError,
    CarryEquityTooSmall,
    CarryExecutionError,
    CarrySpec,
    decision_times,
)
from vibe_quant.portfolio.ticks import PriceTick

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
_START = 1_654_041_600_000_000_000  # 2022-06-01 00:00 UTC
_END = 1_790_553_600_000_000_000  # 2026-09-28 00:00 UTC
_MONDAY = 1_654_473_900_000_000_000  # Monday 2022-06-06 00:05 UTC


def _specs() -> dict[str, InstrumentSpec]:
    return {s: InstrumentSpec.from_instrument(create_instrument(s)) for s in _SYMBOLS}


def _kwargs(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "symbols": ("BTCUSDT", "ETHUSDT"),
        "start_ns": _START,
        "end_ns": _END,
        "initial_balance": 10_000.0,
        "legs_divisor": 3,
        "gross": 1.0,
        "band": 0.10,
    }
    base.update(over)
    return base


def test_error_classes() -> None:
    """§1.4: the three carry error classes and their bases."""
    assert issubclass(CarryDataError, DataUnavailableError)
    assert issubclass(CarryEquityTooSmall, ValueError)
    assert issubclass(CarryExecutionError, RuntimeError)


def test_instrument_spec_from_catalog() -> None:
    """Sizing constants come from the shipped catalog instrument."""
    btc = InstrumentSpec.from_instrument(create_instrument("BTCUSDT"))
    assert btc.symbol == "BTCUSDT"
    assert btc.step == Decimal("0.001")
    assert btc.min_quantity == Decimal("0.001")
    assert btc.min_notional == 50.0
    assert btc.taker_fee == Decimal("0.0005")


def test_from_dates_utc_pins() -> None:
    """UTC conversion and the 225-decision schedule, pinned in ns."""
    spec = CarrySpec.from_dates(_SYMBOLS, "2022-06-01", "2026-09-28")
    assert spec.start_ns == 1654041600000000000
    assert spec.end_ns == 1790553600000000000
    times = decision_times(spec.start_ns, spec.end_ns)
    assert times[0] == 1654473900000000000
    assert times[-1] == 1789949100000000000
    assert len(times) == 225
    assert spec.symbols == tuple(sorted(_SYMBOLS))


def test_decision_times_never_precede_start() -> None:
    """A start off the Monday 00:05 grid still yields every D >= start_ns."""
    times = decision_times(_MONDAY + 60 * 10**9, _END)  # 00:06 past a Monday 00:05
    assert times[0] == _MONDAY + 7 * 86_400 * 10**9
    assert all(at >= _MONDAY + 60 * 10**9 for at in times)


@pytest.mark.parametrize(
    "value",
    ["2022-06-01T05:00:00", "20220601", "2022-06-01+05:00", "2022-6-1"],
)
def test_from_dates_rejects_non_iso_date_strings(value: str) -> None:
    """The date parser takes exactly ``YYYY-MM-DD``, nothing else."""
    with pytest.raises(ValueError):
        CarrySpec.from_dates(("BTCUSDT", "ETHUSDT"), value, "2026-09-28")
    with pytest.raises(ValueError):
        CarrySpec.from_dates(("BTCUSDT", "ETHUSDT"), "2022-06-01", value)


def test_spec_hash_is_sha256_of_sorted_json() -> None:
    first = CarrySpec(**_kwargs())
    reordered = CarrySpec(**_kwargs(symbols=("ETHUSDT", "BTCUSDT")))
    assert first.symbols == ("BTCUSDT", "ETHUSDT")
    assert first.to_json()["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert first.spec_hash() == reordered.spec_hash()
    assert len(first.spec_hash()) == 64
    assert CarrySpec(**_kwargs(band=0.2)).spec_hash() != first.spec_hash()


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"legs_divisor": 1}, id="divisor-1"),
        pytest.param({"legs_divisor": True}, id="divisor-bool"),
        pytest.param({"legs_divisor": 2.0}, id="divisor-float"),
        pytest.param({"gross": 0}, id="gross-zero"),
        pytest.param({"gross": -1}, id="gross-negative"),
        pytest.param({"gross": 2.0000001}, id="gross-above-two"),
        pytest.param({"band": 1.0}, id="band-one"),
        pytest.param({"band": -0.01}, id="band-negative"),
        pytest.param({"initial_balance": 0}, id="equity-zero"),
        pytest.param({"initial_balance": float("nan")}, id="equity-nan"),
        pytest.param({"initial_balance": float("inf")}, id="equity-inf"),
        pytest.param({"initial_balance": 1e400}, id="equity-overflow"),
        pytest.param({"initial_balance": 10**400}, id="equity-int-overflow"),
        pytest.param({"gross": 10**400}, id="gross-int-overflow"),
        pytest.param({"band": 10**400}, id="band-int-overflow"),
        pytest.param({"symbols": ("BTCUSDT",)}, id="one-symbol"),
        pytest.param({"symbols": ("BTCUSDT", "BTCUSDT")}, id="duplicate-symbol"),
        pytest.param({"symbols": ("BTCUSDT", "BTCUSDT", "ETHUSDT")}, id="duplicate-of-three"),
        pytest.param({"symbols": ("BTCUSDT", "")}, id="empty-symbol"),
        pytest.param({"start_ns": _START + 1}, id="start-not-midnight"),
        pytest.param({"end_ns": _END + 1}, id="end-not-midnight"),
        pytest.param({"end_ns": _START}, id="start-equals-end"),
        pytest.param({"start_ns": _END}, id="start-after-end"),
    ],
)
def test_spec_rejects_bad_params(over: dict[str, Any]) -> None:
    """Every out-of-range boundary value is refused at construction."""
    with pytest.raises(ValueError):
        CarrySpec(**_kwargs(**over))


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"gross": True}, id="gross-bool"),
        pytest.param({"band": True}, id="band-bool"),
        pytest.param({"initial_balance": True}, id="equity-bool"),
        pytest.param({"gross": "1.0"}, id="gross-str"),
        pytest.param({"band": "0.1"}, id="band-str"),
        pytest.param({"initial_balance": "10000"}, id="equity-str"),
        pytest.param({"start_ns": float(_START)}, id="start-float"),
        pytest.param({"end_ns": float(_END)}, id="end-float"),
        pytest.param({"start_ns": True}, id="start-bool"),
        pytest.param({"end_ns": True}, id="end-bool"),
    ],
)
def test_spec_rejects_non_numeric_param_types(over: dict[str, Any]) -> None:
    """Bool/str numbers and non-int ns timestamps are refused, not coerced."""
    with pytest.raises(ValueError):
        CarrySpec(**_kwargs(**over))


def test_spec_stores_converted_floats() -> None:
    """Gross/band/initial_balance are stored as the converted float."""
    spec = CarrySpec(**_kwargs(gross=1, band=0, initial_balance=10_000))
    assert spec.gross == 1.0
    assert isinstance(spec.gross, float)
    assert spec.band == 0.0
    assert isinstance(spec.band, float)
    assert spec.initial_balance == 10_000.0
    assert isinstance(spec.initial_balance, float)
    assert spec.spec_hash() == CarrySpec(**_kwargs(gross=1.0, band=0.0)).spec_hash()


def test_spec_accepts_boundary_values() -> None:
    """The inclusive boundary values on the valid side construct fine."""
    for over in (
        {"legs_divisor": 2},
        {"gross": 2.0, "allow_gross_above_one": True},
        {"band": 0.0},
        {"band": 0.999},
        {"initial_balance": 1e-9},
    ):
        CarrySpec(**_kwargs(**over))


def test_spec_rejects_empty_decision_schedule() -> None:
    """A window with no Monday 00:05 decision is refused."""
    with pytest.raises(ValueError, match="no decision in window"):
        CarrySpec(**_kwargs(end_ns=1_654_214_400_000_000_000))  # 2022-06-03


def test_held_symbol_without_price_fails_closed() -> None:
    """Rankable + no sizing tick -> ``no_price``; held -> CarryDataError."""
    spec = CarrySpec(**_kwargs(symbols=_SYMBOLS))
    specs = _specs()
    signals = {"BTCUSDT": 0.003, "ETHUSDT": 0.002, "SOLUSDT": 0.001, "BNBUSDT": -0.001}
    prices = {"ETHUSDT": 3000.0, "SOLUSDT": 100.0, "BNBUSDT": 400.0}  # BTCUSDT absent
    decision = decide(_START, 10_000.0, signals, {}, prices, {}, specs, spec)
    assert decision.excluded["BTCUSDT"] == "no_price"
    assert decision.weights["BTCUSDT"] == 0.0
    assert decision.target_steps["BTCUSDT"] == 0
    with pytest.raises(CarryDataError, match="no price for held BTCUSDT"):
        decide(_START, 10_000.0, signals, {}, prices, {"BTCUSDT": 10}, specs, spec)


def test_target_steps_floor_in_decimal() -> None:
    """``floor(|w| * E / (price * step))`` in Decimal, sign applied after."""
    targets = target_steps(
        {"BTCUSDT": 0.5, "ETHUSDT": -0.5},
        10096.0,
        {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0},
        _specs(),
    )
    assert targets == {"BTCUSDT": 98, "ETHUSDT": -1682}


def test_min_equity_boundary() -> None:
    """The min-equity floor is inclusive; below it the leg fails closed."""
    specs = _specs()
    assert target_steps({"BTCUSDT": 0.5}, 2000.0, {"BTCUSDT": 100000.0}, specs) == {"BTCUSDT": 10}
    with pytest.raises(CarryEquityTooSmall) as btc_small:
        target_steps({"BTCUSDT": 0.5}, 1999.99, {"BTCUSDT": 100000.0}, specs)
    assert btc_small.value.required == 1000.0  # 10 steps x 100000 x 0.001
    with pytest.raises(CarryEquityTooSmall) as eth_small:
        target_steps({"ETHUSDT": 0.5}, 70.0, {"ETHUSDT": 2000.0}, specs)
    assert eth_small.value.symbol == "ETHUSDT"
    assert eth_small.value.required == 40.0  # 2 x min_notional 20 beats 10 x 2.0


def test_rounding_drift() -> None:
    """Drift ``(50 + 2) / 10096``, computed once per decision."""
    spec = CarrySpec(**_kwargs(symbols=_SYMBOLS))
    signals = {"BTCUSDT": -0.001, "ETHUSDT": 0.003, "SOLUSDT": 0.001, "BNBUSDT": 0.002}
    prices = {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0, "SOLUSDT": 100.0, "BNBUSDT": 400.0}
    decision = decide(_START, 10096.0, signals, {}, prices, {}, _specs(), spec)
    assert decision.weights == {
        "BTCUSDT": 0.5,
        "ETHUSDT": -0.5,
        "SOLUSDT": 0.0,
        "BNBUSDT": 0.0,
    }
    assert decision.target_steps["BTCUSDT"] == 98
    assert decision.target_steps["ETHUSDT"] == -1682
    assert decision.rounding_drift == 0.005150554675118859


def test_plan_orders_cases() -> None:
    """Open, close, flip, and the band on same-side resizes."""
    specs = _specs()
    prices = {"BTCUSDT": 51000.0, "SOLUSDT": 100.0}
    orders, band_skips, min_notional_skips = plan_orders({}, {"BTCUSDT": 98}, prices, specs, 0.10)
    assert orders == (OrderIntent("BTCUSDT", 98, False),)
    assert (band_skips, min_notional_skips) == (0, 0)
    orders, _, _ = plan_orders({"BTCUSDT": 98}, {"BTCUSDT": 0}, prices, specs, 0.10)
    assert orders == (OrderIntent("BTCUSDT", -98, True),)
    orders, _, _ = plan_orders({"SOLUSDT": 10}, {"SOLUSDT": -5}, prices, specs, 0.10)
    assert orders == (OrderIntent("SOLUSDT", -10, True), OrderIntent("SOLUSDT", -5, False))
    orders, band_skips, _ = plan_orders({"BTCUSDT": 98}, {"BTCUSDT": 105}, prices, specs, 0.10)
    assert orders == ()
    assert band_skips == 1
    orders, band_skips, _ = plan_orders({"BTCUSDT": 98}, {"BTCUSDT": 110}, prices, specs, 0.10)
    assert orders == (OrderIntent("BTCUSDT", 12, False),)
    assert band_skips == 0
    orders, _, _ = plan_orders({"BTCUSDT": 90}, {"BTCUSDT": 100}, prices, specs, 0.10)
    assert orders == (OrderIntent("BTCUSDT", 10, False),)
    orders, _, _ = plan_orders({"BTCUSDT": 110}, {"BTCUSDT": 98}, prices, specs, 0.10)
    assert orders == (OrderIntent("BTCUSDT", -12, True),)


def test_band_compare_is_exact() -> None:
    """0.07 * 100 is 7.000000000000001 in float, exactly 7.00 in Decimal."""
    specs = {"BTCUSDT": InstrumentSpec.from_instrument(create_instrument("BTCUSDT"))}
    orders, band_skips, _ = plan_orders(
        {"BTCUSDT": 93}, {"BTCUSDT": 100}, {"BTCUSDT": 51000.0}, specs, 0.07
    )
    assert orders == (OrderIntent("BTCUSDT", 7, False),)
    assert band_skips == 0


def test_min_quantity_edge_is_tradable() -> None:
    """A 1-step BTC order equals ``min_quantity`` and must still be sent."""
    specs = _specs()
    prices = {"BTCUSDT": 51000.0}
    orders, band_skips, min_notional_skips = plan_orders({}, {"BTCUSDT": 1}, prices, specs, 0.1)
    assert orders == (OrderIntent("BTCUSDT", 1, False),)
    assert (band_skips, min_notional_skips) == (0, 0)
    orders, band_skips, min_notional_skips = plan_orders(
        {"BTCUSDT": 9}, {"BTCUSDT": 10}, prices, specs, 0.1
    )
    assert orders == (OrderIntent("BTCUSDT", 1, False),)
    assert (band_skips, min_notional_skips) == (0, 0)


def test_open_leg_below_min_notional_is_dropped() -> None:
    """M5: a fresh 5-step ETH leg is $10 < the $20 MIN_NOTIONAL and is not sent."""
    specs = {"ETHUSDT": InstrumentSpec.from_instrument(create_instrument("ETHUSDT"))}
    orders, band_skips, min_notional_skips = plan_orders(
        {}, {"ETHUSDT": 5}, {"ETHUSDT": 2000.0}, specs, 0.1
    )
    assert orders == ()
    assert (band_skips, min_notional_skips) == (0, 1)


def test_min_notional_skips_only_non_reduce() -> None:
    """An increase below MIN_NOTIONAL is dropped, a reduce-only is sent."""
    specs = {"ETHUSDT": InstrumentSpec.from_instrument(create_instrument("ETHUSDT"))}
    prices = {"ETHUSDT": 2000.0}
    orders, band_skips, min_notional_skips = plan_orders(
        {"ETHUSDT": 45}, {"ETHUSDT": 50}, prices, specs, 0.10
    )
    assert orders == ()
    assert (band_skips, min_notional_skips) == (0, 1)
    orders, band_skips, min_notional_skips = plan_orders(
        {"ETHUSDT": 50}, {"ETHUSDT": 45}, prices, specs, 0.10
    )
    assert orders == (OrderIntent("ETHUSDT", -5, True),)
    assert (band_skips, min_notional_skips) == (0, 0)


def test_flip_open_leg_below_min_notional_is_dropped() -> None:
    """A flip's close is sent, its $1 open leg is dropped and counted."""
    specs = {"SOLUSDT": InstrumentSpec.from_instrument(create_instrument("SOLUSDT"))}
    orders, band_skips, min_notional_skips = plan_orders(
        {"SOLUSDT": 10}, {"SOLUSDT": -1}, {"SOLUSDT": 100.0}, specs, 0.1
    )
    assert orders == (OrderIntent("SOLUSDT", -10, True),)
    assert (band_skips, min_notional_skips) == (0, 1)


def test_band_takes_precedence_over_min_notional() -> None:
    """A band skip never reaches the min-notional check; closes are exempt."""
    specs = {"ETHUSDT": InstrumentSpec.from_instrument(create_instrument("ETHUSDT"))}
    prices = {"ETHUSDT": 2000.0}
    orders, band_skips, min_notional_skips = plan_orders(
        {"ETHUSDT": 45}, {"ETHUSDT": 50}, prices, specs, 0.2
    )
    assert (orders, band_skips, min_notional_skips) == ((), 1, 0)
    orders, band_skips, min_notional_skips = plan_orders(
        {"ETHUSDT": 2}, {"ETHUSDT": 0}, prices, specs, 0.1
    )
    assert orders == (OrderIntent("ETHUSDT", -2, True),)
    assert (band_skips, min_notional_skips) == (0, 0)


def test_decide_passes_signal_hole_through() -> None:
    """A held leg whose signal is a hole is closed, not re-opened."""
    spec = CarrySpec(**_kwargs(symbols=_SYMBOLS))
    signals = {"BTCUSDT": None, "ETHUSDT": 0.002, "SOLUSDT": 0.001, "BNBUSDT": -0.001}
    prices = {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0, "SOLUSDT": 100.0, "BNBUSDT": 400.0}
    decision = decide(
        _START,
        10_000.0,
        signals,
        {"BTCUSDT": "funding_hole"},
        prices,
        {"BTCUSDT": 5},
        _specs(),
        spec,
    )
    assert decision.excluded["BTCUSDT"] == "funding_hole"
    assert decision.weights["BTCUSDT"] == 0.0
    assert decision.target_steps["BTCUSDT"] == 0
    assert OrderIntent("BTCUSDT", -5, True) in decision.orders


@pytest.mark.parametrize(
    ("equity", "live"),
    [
        pytest.param(0.0, False, id="flat-zero"),
        pytest.param(0.0, True, id="live-zero"),
        pytest.param(-1.0, True, id="live-negative"),
        pytest.param(-100.0, False, id="flat-negative"),
        pytest.param(float("nan"), True, id="live-nan"),
        pytest.param(float("nan"), False, id="flat-nan"),
    ],
)
def test_decide_non_positive_equity_fails_closed(equity: float, live: bool) -> None:
    """``not (equity > 0.0)`` fails closed FIRST, for every signal state.

    The book-level payload names ``<book>``/D/equity/0.0: with live signals the
    guard must fire before ``target_steps`` (which would name the first leg).
    """
    spec = CarrySpec(**_kwargs(symbols=_SYMBOLS))
    specs = _specs()
    signals: dict[str, float | None] = (
        {"BTCUSDT": 0.002, "ETHUSDT": -0.002, "SOLUSDT": 0.001, "BNBUSDT": -0.001}
        if live
        else dict.fromkeys(_SYMBOLS)
    )
    prices = {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0, "SOLUSDT": 100.0, "BNBUSDT": 400.0}
    with pytest.raises(CarryEquityTooSmall) as error:
        decide(_START, equity, signals, {}, prices, {}, specs, spec)
    assert error.value.symbol == "<book>"
    assert error.value.decision_ns == _START
    assert error.value.required == 0.0
    if math.isnan(equity):
        assert math.isnan(error.value.leg_notional)
    else:
        assert error.value.leg_notional == equity
    assert (
        decide(_START, 1e-9, dict.fromkeys(_SYMBOLS), {}, {}, {}, specs, spec).rounding_drift == 0.0
    )


@pytest.mark.parametrize("equity", [0.0, -1.0, float("nan")], ids=["zero", "negative", "nan"])
def test_equity_guard_precedes_held_unpriced(equity: float) -> None:
    """With BOTH a non-positive equity and a held-but-unpriced leg, ``CarryEquityTooSmall`` wins.

    ``decide`` checks ``not (equity > 0.0)`` before it walks the signals, so the
    book-level payload is raised even though ``BTCUSDT`` is held (``current`` 10)
    and has no sizing price. Swapping the two checks would raise ``CarryDataError``.
    """
    spec = CarrySpec(**_kwargs(symbols=_SYMBOLS))
    specs = _specs()
    signals = {"BTCUSDT": 0.003, "ETHUSDT": 0.002, "SOLUSDT": 0.001, "BNBUSDT": -0.001}
    prices = {"ETHUSDT": 3000.0, "SOLUSDT": 100.0, "BNBUSDT": 400.0}  # BTCUSDT absent

    with pytest.raises(CarryEquityTooSmall) as error:
        decide(_START, equity, signals, {}, prices, {"BTCUSDT": 10}, specs, spec)

    assert error.value.symbol == "<book>"
    assert error.value.decision_ns == _START
    assert error.value.required == 0.0


def test_nt_commission_matches_formula() -> None:
    """§5: the Decimal copy of NT's taker branch; Decimal(1) is exact."""
    instrument = create_instrument("BTCUSDT")
    quantity = instrument.make_qty(Decimal("0.098"))
    price = instrument.make_price(51010.0)
    assert nt_commission(instrument, quantity, price) == 2.49949
    assert nt_commission(instrument, quantity, price, Decimal(1)) == nt_commission(
        instrument, quantity, price
    )


def test_default_spec_hash_unchanged() -> None:
    """R1: the pinned 4-coin default spec keeps its 7 json keys and its hash."""
    spec = CarrySpec.from_dates(_SYMBOLS, "2022-06-01", "2026-09-28")
    assert list(spec.to_json()) == [
        "symbols",
        "start_ns",
        "end_ns",
        "initial_balance",
        "legs_divisor",
        "gross",
        "band",
    ]
    assert spec.spec_hash() == "aeaa2d4b4a623be018aec2f7c385d6695936990705fa736ee6f7625bd6c896e3"


def test_decide_positional_unchanged() -> None:
    """R9: ``decide`` with today's 8 positional args gives the hand-computed plan.

    k = max(1, 3 // 3) = 1 at 0.5 a side: BTC (highest) short 50 000 / 30 = 1666.67
    -> -1666 steps; ETH (lowest) long 50 000 / 1.8 = 27 777.8 -> 27 777; SOL is a
    funding hole held short 100 -> reduce-only close. Drift (20 + 1.4) / 100 000.
    """
    spec = CarrySpec.from_dates(_SYMBOLS, "2022-06-01", "2026-09-28")
    decision = decide(
        _MONDAY,
        100_000.0,
        {"BTCUSDT": 0.002, "ETHUSDT": -0.001, "SOLUSDT": None, "BNBUSDT": 0.0005},
        {"SOLUSDT": "funding_hole"},
        {"BTCUSDT": 30_000.0, "ETHUSDT": 1_800.0, "BNBUSDT": 300.0, "SOLUSDT": 40.0},
        {"BTCUSDT": 0, "ETHUSDT": 0, "SOLUSDT": -100, "BNBUSDT": 0},
        _specs(),
        spec,
    )
    assert decision.ts == _MONDAY
    assert decision.equity == 100_000.0
    assert decision.excluded == {"SOLUSDT": "funding_hole"}
    assert decision.weights == {"BTCUSDT": -0.5, "ETHUSDT": 0.5, "SOLUSDT": 0.0, "BNBUSDT": 0.0}
    assert decision.prices == {
        "BNBUSDT": 300.0,
        "BTCUSDT": 30_000.0,
        "ETHUSDT": 1_800.0,
        "SOLUSDT": 40.0,
    }
    assert decision.target_steps == {"BTCUSDT": -1666, "ETHUSDT": 27777, "SOLUSDT": 0, "BNBUSDT": 0}
    assert decision.orders == (
        OrderIntent("BTCUSDT", -1666, reduce_only=False),
        OrderIntent("ETHUSDT", 27777, reduce_only=False),
        OrderIntent("SOLUSDT", 100, reduce_only=True),
    )
    assert (decision.band_skips, decision.min_notional_skips) == (0, 0)
    assert decision.rounding_drift == 0.00021400000000001455
    assert decision.vols is None


# ---------------------------------------------------------------------------
# inverse-vol weighting (x0axf.14 T4: R2, R3, R21, R21a, R24)
# ---------------------------------------------------------------------------


def test_inverse_vol_serialization() -> None:
    """R2: inverse_vol adds two json keys (lookback omitted == explicit 30); default 7 keys."""
    omitted = CarrySpec.from_dates(_SYMBOLS, "2022-06-01", "2026-09-28", weighting="inverse_vol")
    explicit = CarrySpec.from_dates(
        _SYMBOLS, "2022-06-01", "2026-09-28", weighting="inverse_vol", vol_lookback_days=30
    )
    equal = CarrySpec.from_dates(_SYMBOLS, "2022-06-01", "2026-09-28", weighting="equal")
    for spec in (omitted, explicit):
        payload = spec.to_json()
        assert list(payload)[7:] == ["weighting", "vol_lookback_days"]
        assert (payload["weighting"], payload["vol_lookback_days"]) == ("inverse_vol", 30)
    assert omitted.spec_hash() == explicit.spec_hash()
    assert omitted.spec_hash() != equal.spec_hash()
    assert equal.spec_hash() == "aeaa2d4b4a623be018aec2f7c385d6695936990705fa736ee6f7625bd6c896e3"
    other = CarrySpec.from_dates(
        _SYMBOLS, "2022-06-01", "2026-09-28", weighting="inverse_vol", vol_lookback_days=60
    )
    assert other.to_json()["vol_lookback_days"] == 60
    assert other.spec_hash() != omitted.spec_hash()


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"vol_lookback_days": 60}, "vol_lookback_days needs weighting='inverse_vol'"),
        ({"vol_lookback_days": 29}, "vol_lookback_days needs weighting='inverse_vol'"),
        *[
            (
                {"weighting": "inverse_vol", "vol_lookback_days": bad},
                f"vol_lookback_days must be an int in [2, 365], got {bad!r}",
            )
            for bad in (True, 2.0, 1, 366, None, "30")
        ],
        *[
            (
                {"weighting": bad},
                f"weighting must be one of ('equal', 'inverse_vol', 'time_series'), got {bad!r}",
            )
            for bad in ("vol", None, "Equal", "")
        ],
    ],
)
def test_spec_variant_ranges(over: dict[str, Any], message: str) -> None:
    """R3: weighting and lookback ranges, each with its own message."""
    with pytest.raises(ValueError) as caught:
        CarrySpec(**_kwargs(**over))
    assert str(caught.value) == message


def test_spec_variant_range_edges() -> None:
    """R3 edges: 2 and 365 are accepted; explicit equal + 30 is the default spec."""
    for days in (2, 365):
        spec = CarrySpec(**_kwargs(weighting="inverse_vol", vol_lookback_days=days))
        assert spec.vol_lookback_days == days
    assert CarrySpec(**_kwargs(weighting="equal", vol_lookback_days=30)) == CarrySpec(**_kwargs())


def _vol_spec() -> CarrySpec:
    return CarrySpec(**_kwargs(symbols=_SYMBOLS, legs_divisor=2, weighting="inverse_vol"))


_VOL_SIGNALS = {"BTCUSDT": 0.002, "ETHUSDT": -0.001, "SOLUSDT": 0.001, "BNBUSDT": 0.0}
_VOL_PRICES = {"BTCUSDT": 30_000.0, "ETHUSDT": 1_800.0, "BNBUSDT": 300.0, "SOLUSDT": 40.0}
_FLAT = dict.fromkeys(_SYMBOLS, 0)


def test_decide_inverse_vol_weights_and_targets() -> None:
    """Longs ETH .02 / BNB .04, shorts BTC .02 / SOL .04 at E 100 000: 1/3 vs 1/6 a side."""
    vols = {"ETHUSDT": 0.02, "BNBUSDT": 0.04, "BTCUSDT": 0.02, "SOLUSDT": 0.04}
    decision = decide(
        _MONDAY, 100_000.0, _VOL_SIGNALS, {}, _VOL_PRICES, _FLAT, _specs(), _vol_spec(), vols=vols
    )
    assert decision.weights == {
        "BTCUSDT": -0.3333333333333333,
        "ETHUSDT": 0.3333333333333333,
        "SOLUSDT": -0.16666666666666666,
        "BNBUSDT": 0.16666666666666666,
    }
    # 33 333.33 / 30, / 1.8; 16 666.67 / 0.4, / 3 -- floored
    assert decision.target_steps == {
        "BTCUSDT": -1111,
        "ETHUSDT": 18518,
        "SOLUSDT": -41666,
        "BNBUSDT": 5555,
    }
    assert decision.vols == vols
    # the same inputs at equal weighting split 0.25 a leg
    equal = CarrySpec(**_kwargs(symbols=_SYMBOLS, legs_divisor=2))
    flat = decide(_MONDAY, 100_000.0, _VOL_SIGNALS, {}, _VOL_PRICES, _FLAT, _specs(), equal)
    assert set(flat.weights.values()) == {0.25, -0.25}


def test_inverse_vol_extreme_ratio() -> None:
    """R21: sigma 1e-310 vs .04 -> BNB 1.25e-309, below its required notional: fail closed."""
    vols = {"ETHUSDT": 1e-310, "BNBUSDT": 0.04, "BTCUSDT": 0.02, "SOLUSDT": 0.04}
    with pytest.raises(CarryEquityTooSmall) as caught:
        decide(
            _MONDAY,
            100_000.0,
            _VOL_SIGNALS,
            {},
            _VOL_PRICES,
            _FLAT,
            _specs(),
            _vol_spec(),
            vols=vols,
        )
    error = caught.value
    assert (error.symbol, error.decision_ns, error.required) == ("BNBUSDT", _MONDAY, 30.0)
    assert error.leg_notional == 1.25e-309 * 100_000.0


def test_inverse_vol_zero_weight_fails() -> None:
    """R21a: sigma 1e-310 vs 1e20 -> BNB's ratio underflows to 0.0: CarryEquityTooSmall, not unselected."""
    vols = {"ETHUSDT": 1e-310, "BNBUSDT": 1e20, "BTCUSDT": 0.02, "SOLUSDT": 0.04}
    with pytest.raises(CarryEquityTooSmall) as caught:
        decide(
            _MONDAY,
            100_000.0,
            _VOL_SIGNALS,
            {},
            _VOL_PRICES,
            _FLAT,
            _specs(),
            _vol_spec(),
            vols=vols,
        )
    error = caught.value
    assert (error.symbol, error.decision_ns, error.leg_notional, error.required) == (
        "BNBUSDT",
        _MONDAY,
        0.0,
        30.0,
    )


def test_no_vol_held_symbol_closes() -> None:
    """R24: a held leg that turns ``no_vol`` gets target 0 and one reduce-only close."""
    signals = {**_VOL_SIGNALS, "SOLUSDT": None}
    vols = {"ETHUSDT": 0.02, "BNBUSDT": 0.04, "BTCUSDT": 0.02}
    current = {**_FLAT, "SOLUSDT": -100}
    decision = decide(
        _MONDAY,
        100_000.0,
        signals,
        {"SOLUSDT": "no_vol"},
        _VOL_PRICES,
        current,
        _specs(),
        _vol_spec(),
        vols=vols,
    )
    assert decision.excluded == {"SOLUSDT": "no_vol"}
    assert decision.target_steps["SOLUSDT"] == 0
    assert [o for o in decision.orders if o.symbol == "SOLUSDT"] == [
        OrderIntent("SOLUSDT", 100, reduce_only=True)
    ]
    # 3 rankable -> k = max(1, 3 // 2) = 1: ETH long, BTC short, each a 1-leg side at 0.5
    assert decision.weights == {"BTCUSDT": -0.5, "ETHUSDT": 0.5, "SOLUSDT": 0.0, "BNBUSDT": 0.0}


# ---------------------------------------------------------------------------
# liquidity filter (x0axf.14 T5: R4, R36)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (0, "min_quote_volume must be > 0, got 0"),
        (-1, "min_quote_volume must be > 0, got -1"),
        (-1e-300, "min_quote_volume must be > 0, got -1e-300"),
        (math.nan, "min_quote_volume must be a finite number, got nan"),
        (math.inf, "min_quote_volume must be a finite number, got inf"),
        (True, "min_quote_volume must be a finite number, got True"),
        ("1e8", "min_quote_volume must be a finite number, got '1e8'"),
        (10**400, f"min_quote_volume must be a finite number, got {10**400!r}"),
    ],
)
def test_min_quote_volume_range(value: object, message: str) -> None:
    """R4: refused via ``_as_float`` (non-finite, bool, str, huge int) or ``> 0``."""
    with pytest.raises(ValueError) as caught:
        CarrySpec(**_kwargs(min_quote_volume=value))
    assert str(caught.value) == message


def test_min_quote_volume_serialization() -> None:
    """R4: 1e8 (or int 10**8, or the 5e-324 edge) is stored as a float and added to json."""
    spec = CarrySpec.from_dates(_SYMBOLS, "2022-06-01", "2026-09-28", min_quote_volume=10**8)
    assert spec.min_quote_volume == 1e8
    assert isinstance(spec.min_quote_volume, float)
    assert list(spec.to_json())[7:] == ["min_quote_volume"]
    assert spec.to_json()["min_quote_volume"] == 1e8
    assert (
        spec.spec_hash() != CarrySpec.from_dates(_SYMBOLS, "2022-06-01", "2026-09-28").spec_hash()
    )
    assert CarrySpec(**_kwargs(min_quote_volume=5e-324)).min_quote_volume == 5e-324


_SIX = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "SUIUSDT", "ZECUSDT")
_SIX_RATES = dict(zip(sorted(_SIX), (1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4), strict=True))
_HOUR_NS = 3_600 * 10**9
_DAY_NS = 24 * _HOUR_NS


def _six_settlements() -> dict[str, Settlements]:
    """Full 8h coverage of ``_MONDAY``'s window; one flat rate per symbol."""
    upper = _MONDAY - 60 * 10**9
    first = ((upper - 7 * _DAY_NS) // (8 * _HOUR_NS) + 1) * 8 * _HOUR_NS
    times = tuple(range(first, upper + 1, 8 * _HOUR_NS))
    return {symbol: Settlements(times, (_SIX_RATES[symbol],) * len(times)) for symbol in _SIX}


def _six_decision(illiquid: set[str], current: dict[str, int]) -> Any:
    spec = CarrySpec(**_kwargs(symbols=_SIX, min_quote_volume=1e8))
    b0 = _MONDAY - _MONDAY % _DAY_NS
    week = {b0 - k * _DAY_NS: 1e9 for k in range(1, 8)}
    volume = {s: {d: 1e7 for d in week} if s in illiquid else week for s in _SIX}
    prices = {
        "BTCUSDT": 30_000.0,
        "ETHUSDT": 1_800.0,
        "SOLUSDT": 40.0,
        "BNBUSDT": 300.0,
        "SUIUSDT": 3.0,
        "ZECUSDT": 50.0,
    }
    sizing = {s: PriceTick(_MONDAY - 10**6, prices[s]) for s in _SIX}
    eligible = decision_signals(spec, _six_settlements(), sizing, _MONDAY, quote_volume=volume)
    specs = {s: InstrumentSpec.from_instrument(create_instrument(s)) for s in _SIX}
    held = {s: current.get(s, 0) for s in _SIX}
    return eligible, decide(
        _MONDAY, 100_000.0, eligible.signals, eligible.excluded, eligible.prices, held, specs, spec
    )


def test_liquidity_shrinks_universe() -> None:
    """R36: 2 illiquid -> 4 ranked (k 1); 5 -> 1 left, flat book; 6 -> 0 left, flat.

    Held priced legs of an illiquid (or newly unranked) symbol close reduce-only.
    """
    eligible, decision = _six_decision({"SUIUSDT", "ZECUSDT"}, {"SUIUSDT": 100})
    assert eligible.excluded == {"SUIUSDT": "illiquid", "ZECUSDT": "illiquid"}
    assert sum(v is not None for v in eligible.signals.values()) == 4
    # k = max(1, 4 // 3) = 1: long BNB (lowest rate), short SOL (highest of the 4)
    assert {s: w for s, w in decision.weights.items() if w} == {"BNBUSDT": 0.5, "SOLUSDT": -0.5}
    assert decision.target_steps["SUIUSDT"] == 0
    assert OrderIntent("SUIUSDT", -100, reduce_only=True) in decision.orders

    held = {"BNBUSDT": 10, "SOLUSDT": -10}
    for illiquid in (set(_SIX) - {"BTCUSDT"}, set(_SIX)):
        eligible, decision = _six_decision(illiquid, held)
        assert eligible.excluded == dict.fromkeys(sorted(illiquid), "illiquid")
        assert all(w == 0.0 for w in decision.weights.values())
        assert decision.orders == (
            OrderIntent("BNBUSDT", -10, reduce_only=True),
            OrderIntent("SOLUSDT", 10, reduce_only=True),
        )
        assert set(decision.prices) == set(_SIX)  # illiquid legs stay priced


# ---------------------------------------------------------------------------
# time-series weighting (clmmc.9 T2: R17, R22-R31, C9-9, C9-10)
# ---------------------------------------------------------------------------

_TS_D = 1_710_118_800_000_000_000  # 2024-03-11T01:00:00Z
_TS_B0 = _TS_D - _TS_D % (86_400 * 10**9)
_TS_DAY = 86_400 * 10**9

# W1 closes, oldest -> B0 (vol lookback 3 needs B0 - 3 d .. B0)
_W1_CLOSES = {
    "A": [100.0, 110.0, 99.0, 108.9],
    "B": [50.0, 51.0, 51.0, 50.49],
    "C": [10.0, 10.0, 10.0, 10.0],
}
_W1_SIGNALS: dict[str, float | None] = {"A": 1.0, "B": 0.6, "C": 0.0}
_W1_A = 0.17811538818154032
_W1_B = 0.1068692329089242


def _keyed(path: list[float]) -> dict[int, float]:
    return {
        _TS_B0 - back * _TS_DAY: close
        for back, close in zip(range(len(path) - 1, -1, -1), path, strict=True)
    }


def _ts_closes(paths: dict[str, list[float]]) -> dict[str, dict[int, float]]:
    return {symbol: _keyed(path) for symbol, path in paths.items()}


def _ts_spec(symbols: tuple[str, ...] = ("A", "B", "C"), **over: Any) -> CarrySpec:
    kw: dict[str, Any] = {
        "signal": "trend",
        "schedule": "daily",
        "weighting": "time_series",
        "direction": "long_flat",
        "vol_target": 0.40,
        "vol_lookback_days": 3,
    }
    kw.update(over)
    return CarrySpec.from_dates(symbols, "2024-03-01", "2024-04-01", **kw)


def _ts_instr(
    symbol: str, step: str, min_notional: float = 5.0, min_quantity: str | None = None
) -> InstrumentSpec:
    return InstrumentSpec(
        symbol, Decimal(step), Decimal(min_quantity or step), min_notional, Decimal("0.0005")
    )


def _w1_specs(**min_notional: float) -> dict[str, InstrumentSpec]:
    return {
        "A": _ts_instr("A", "0.001", min_notional.get("A", 5.0)),
        "B": _ts_instr("B", "0.01", min_notional.get("B", 5.0)),
        "C": _ts_instr("C", "0.01", min_notional.get("C", 5.0)),
    }


_W1_PRICES = {"A": 108.9, "B": 50.49, "C": 10.0}


def _realised_vol(weights: dict[str, float], paths: dict[str, list[float]]) -> float:
    """In-window annual vol of the weighted book, recomputed independently with numpy."""
    rets = np.array(
        [[path[i] / path[i - 1] - 1.0 for i in range(1, len(path))] for path in paths.values()]
    )
    book = np.array(list(weights.values())) @ rets
    return float(np.std(book, ddof=1) * math.sqrt(365.0))


def test_time_series_weights_w1() -> None:
    """R22: sigma, lambda and the weights; C (signal 0) gets 0; targets after the floor."""
    book = time_series_weights(_W1_SIGNALS, _ts_closes(_W1_CLOSES), _TS_D, _ts_spec())
    assert book.state is None
    assert book.sigma == 0.03918238455300111
    assert book.lam == 0.534346164544621
    assert book.weights == {"A": _W1_A, "B": _W1_B, "C": 0.0}
    decision = decide(
        _TS_D,
        10_000.0,
        _W1_SIGNALS,
        {},
        _W1_PRICES,
        {},
        _w1_specs(),
        _ts_spec(),
        trend_closes=_ts_closes(_W1_CLOSES),
    )
    assert decision.weights == {"A": _W1_A, "B": _W1_B, "C": 0.0}
    assert decision.target_steps == {"A": 16355, "B": 2116, "C": 0}
    assert decision.orders == (OrderIntent("A", 16355, False), OrderIntent("B", 2116, False))
    assert (decision.book_state, decision.min_leg_skips) == (None, 0)
    assert decision.vols is None
    # C9-10 uncapped: the scaled book's in-window vol is the target
    assert _realised_vol(book.weights, _W1_CLOSES) == pytest.approx(0.40, rel=1e-12)


def test_time_series_weights_capped() -> None:
    """R23 + C9-10 capped: lambda * total > gross -> scale gross / total; vol below target."""
    book = time_series_weights(_W1_SIGNALS, _ts_closes(_W1_CLOSES), _TS_D, _ts_spec(vol_target=2.0))
    assert book.lam == 2.6717308227231054
    assert book.weights == {"A": 0.625, "B": 0.37499999999999994, "C": 0.0}
    assert math.fsum(book.weights.values()) == 1.0
    assert book.sigma == 0.03918238455300111
    realised = _realised_vol(book.weights, _W1_CLOSES)
    assert realised == pytest.approx(1.875 * 0.03918238455300111 * math.sqrt(365.0), rel=1e-12)
    assert realised < 2.0


def test_time_series_weights_long_short() -> None:
    """R24: long_short keeps a negative signal; C's flat closes add no book vol."""
    signals: dict[str, float | None] = {"A": 1.0, "B": 0.6, "C": -0.2}
    book = time_series_weights(
        signals, _ts_closes(_W1_CLOSES), _TS_D, _ts_spec(direction="long_short")
    )
    assert book.weights == {"A": _W1_A, "B": _W1_B, "C": -0.035623077636308065}
    flat = time_series_weights(signals, _ts_closes(_W1_CLOSES), _TS_D, _ts_spec())
    assert flat.weights == {"A": _W1_A, "B": _W1_B, "C": 0.0}


@pytest.mark.parametrize(
    ("paths", "signals", "direction"),
    [
        # only C has u != 0 and its closes are flat: sigma 0
        (_W1_CLOSES, {"A": 0.0, "B": 0.0, "C": 1.0}, "long_flat"),
        # opposite legs whose return is +inf: products +inf and -inf (fsum ValueError)
        (
            {"A": [1.0, 1.0, 1e-300, 1e300], "B": [1.0, 1.0, 1e-300, 1e300]},
            {"A": 1.0, "B": -1.0},
            "long_short",
        ),
        # finite returns ~1e308 twice: a naive fsum of the book returns overflows
        (
            {"A": [1.0, 1e308, 1.0, 1e308], "B": [1.0, 1.0, 1.0, 1.0]},
            {"A": 1.0, "B": None},
            "long_flat",
        ),
    ],
)
def test_book_without_vol_is_flat(
    paths: dict[str, list[float]], signals: dict[str, float | None], direction: str
) -> None:
    """R25 + C9-10: a book sigma that is 0 or not finite -> flat weights, one book_no_vol."""
    spec = _ts_spec(tuple(paths), direction=direction)
    book = time_series_weights(signals, _ts_closes(paths), _TS_D, spec)
    assert book.state == "book_no_vol"
    assert set(book.weights.values()) == {0.0}
    assert book.lam is None


def test_flat_and_single_coin_books() -> None:
    """R26: nothing eligible (or nothing long) -> flat, no sigma; one coin -> w = lambda."""
    closes = _ts_closes(_W1_CLOSES)
    for signals in ({"A": None, "B": None, "C": None}, {"A": -1.0, "B": 0.0, "C": None}):
        book = time_series_weights(signals, closes, _TS_D, _ts_spec())
        assert (book.state, book.sigma) == ("flat", None)
        assert set(book.weights.values()) == {0.0}
    one = time_series_weights({"A": 1.0, "B": None, "C": None}, closes, _TS_D, _ts_spec())
    assert one.lam == 0.18131936556464978
    assert one.weights == {"A": 0.18131936556464978, "B": 0.0, "C": 0.0}


@pytest.mark.parametrize(
    ("regime", "state"),
    [(-0.2, "filter_off"), (None, "filter_off"), (0.0, "filter_off"), (0.2, None)],
)
def test_regime_filter(regime: float | None, state: str | None) -> None:
    """R27: regime None or <= 0 -> every weight 0 and filter_off; > 0 -> the W1 book."""
    spec = _ts_spec(trend_filter="A")
    book = time_series_weights(_W1_SIGNALS, _ts_closes(_W1_CLOSES), _TS_D, spec, regime=regime)
    assert book.state == state
    expected = {"A": _W1_A, "B": _W1_B, "C": 0.0} if state is None else dict.fromkeys("ABC", 0.0)
    assert book.weights == expected


def test_scale_cap_edge() -> None:
    """R28: lambda * total == gross keeps lambda; just above rescales to gross / total."""
    assert _scale(2.0, 0.5, 1.0) == 2.0
    assert _scale(2.0, 0.5000000000000001, 1.0) == 1.0 / 0.5000000000000001


def test_min_leg_rules() -> None:
    """R30: a leg below its tradable size -> target 0 (+1 skip); E*gross/n below -> raise."""
    closes = _ts_closes(_W1_CLOSES)
    edge = (_W1_B * 10_000.0) / 2  # 2 x min_notional == |w_B| * E exactly
    kept = decide(
        _TS_D,
        10_000.0,
        _W1_SIGNALS,
        {},
        _W1_PRICES,
        {},
        _w1_specs(B=edge),
        _ts_spec(),
        trend_closes=closes,
    )
    assert (kept.target_steps["B"], kept.min_leg_skips) == (2116, 0)
    over = math.nextafter(edge, math.inf)
    skipped = decide(
        _TS_D,
        10_000.0,
        _W1_SIGNALS,
        {},
        _W1_PRICES,
        {},
        _w1_specs(B=over),
        _ts_spec(),
        trend_closes=closes,
    )
    assert (skipped.target_steps, skipped.min_leg_skips) == ({"A": 16355, "B": 0, "C": 0}, 1)
    assert skipped.orders == (OrderIntent("A", 16355, False),)
    # E * gross / n = 3000 / 3 = 1000 == required (2 x 500): no raise
    decide(
        _TS_D,
        3_000.0,
        _W1_SIGNALS,
        {},
        _W1_PRICES,
        {},
        _w1_specs(C=500.0),
        _ts_spec(),
        trend_closes=closes,
    )
    with pytest.raises(CarryEquityTooSmall) as small:
        decide(
            _TS_D,
            3_000.0,
            _W1_SIGNALS,
            {},
            _W1_PRICES,
            {},
            _w1_specs(C=500.25),
            _ts_spec(),
            trend_closes=closes,
        )
    assert (small.value.symbol, small.value.decision_ns) == ("C", _TS_D)
    assert (small.value.leg_notional, small.value.required) == (1000.0, 1000.5)


def test_time_series_no_price_rule() -> None:
    """R17 (rule): unheld without a sizing price -> no_price, target 0; held -> CarryDataError."""
    prices = {"B": 50.49, "C": 10.0}
    closes = _ts_closes(_W1_CLOSES)
    decision = decide(
        _TS_D, 10_000.0, _W1_SIGNALS, {}, prices, {}, _w1_specs(), _ts_spec(), trend_closes=closes
    )
    assert decision.excluded == {"A": "no_price"}
    assert decision.target_steps["A"] == 0
    assert decision.weights["A"] == 0.0
    with pytest.raises(CarryDataError, match="no price for held A"):
        decide(
            _TS_D,
            10_000.0,
            _W1_SIGNALS,
            {},
            prices,
            {"A": 5},
            _w1_specs(),
            _ts_spec(),
            trend_closes=closes,
        )


# --- R31 / C9-9: the gross cap bypasses the band, one Decimal budget ---

_UNIT = {"A": _ts_instr("A", "1", 1.0), "B": _ts_instr("B", "1", 1.0)}
_LOW_VOL = _ts_closes({"A": [100.0, 101.0, 100.0, 101.0], "B": [100.0, 101.0, 100.0, 101.0]})


def _capped(
    held: dict[str, int], signals: dict[str, float | None], gross: float = 1.0, **kw: Any
) -> Any:
    spec = _ts_spec(("A", "B"), vol_target=2.0, gross=gross, band=0.10)
    return decide(
        _TS_D,
        100.0,
        signals,
        {},
        {"A": 1.0, "B": 1.0},
        held,
        kw.get("specs", _UNIT),
        spec,
        trend_closes=_LOW_VOL,
    )


@pytest.mark.parametrize(
    ("held", "signals", "orders", "band_skips", "exposure"),
    [
        ({"A": 108}, {"A": 1.0, "B": None}, (OrderIntent("A", -8, True),), 0, "1"),
        ({"A": 101}, {"A": 1.0, "B": None}, (OrderIntent("A", -1, True),), 0, "1"),
        ({"A": 110}, {"A": 1.0, "B": None}, (OrderIntent("A", -10, True),), 0, "1"),
        ({"A": 52, "B": 48}, {"A": 1.0, "B": 1.0}, (), 2, "1"),
        ({"A": 53, "B": 48}, {"A": 1.0, "B": 1.0}, (OrderIntent("A", -3, True),), 1, "0.98"),
        # B held == target: no zero-step reduction may be emitted (rev-c9t2-1 S1)
        ({"A": 53, "B": 50}, {"A": 1.0, "B": 1.0}, (OrderIntent("A", -3, True),), 0, "1"),
    ],
)
def test_cap_bypasses_band(
    held: dict[str, int],
    signals: dict[str, float | None],
    orders: tuple[OrderIntent, ...],
    band_skips: int,
    exposure: str,
) -> None:
    """R31: post-plan exposure above gross*E emits every band-skipped reduction."""
    decision = _capped(held, signals)
    assert decision.orders == orders
    assert decision.band_skips == band_skips
    assert decision.exposure_after_plan == Decimal(exposure)


def test_cap_decimal_budget_point_three() -> None:
    """C9-9: gross 0.3 at E 100 -> budget Decimal(0.3)*100 < 30, so 29 steps, never 30."""
    assert Decimal(0.3) * Decimal(100) < 30
    opened = _capped({}, {"A": 1.0, "B": None}, gross=0.3)  # w == 0.3 exactly
    assert opened.weights["A"] == 0.3
    assert opened.orders == (OrderIntent("A", 29, False),)
    assert opened.exposure_after_plan == Decimal("0.29")
    just_over = _capped({"A": 30}, {"A": 1.0, "B": None}, gross=0.3)  # 30 > 29.999...: bypass
    assert just_over.orders == (OrderIntent("A", -1, True),)
    assert (just_over.band_skips, just_over.exposure_after_plan) == (0, Decimal("0.29"))
    ulp_over = _capped({}, {"A": 0.28, "B": None}, gross=0.3)  # w = 0.30000000000000004
    assert ulp_over.weights["A"] == 0.30000000000000004
    assert ulp_over.target_steps["A"] == 29
    assert ulp_over.exposure_after_plan == Decimal("0.29")


def test_exposure_applies_order_deltas() -> None:
    """C9-9: a flip whose open is min-quantity skipped leaves only its close: exposure 0."""
    specs = {"A": _ts_instr("A", "1", 1.0, min_quantity="20"), "B": _UNIT["B"]}
    spec = _ts_spec(("A", "B"), vol_target=2.0, gross=0.15, direction="long_short")
    decision = decide(
        _TS_D,
        100.0,
        {"A": -1.0, "B": None},
        {},
        {"A": 1.0, "B": 1.0},
        {"A": 40},
        specs,
        spec,
        trend_closes=_LOW_VOL,
    )
    assert decision.target_steps["A"] == -14
    assert decision.orders == (OrderIntent("A", -40, True),)
    assert decision.min_notional_skips == 1
    assert decision.exposure_after_plan == Decimal(0)


def test_funding_decision_keeps_new_fields_default() -> None:
    """R3: a funding decision carries the trend fields at their defaults."""
    spec = CarrySpec.from_dates(_SYMBOLS, "2022-06-01", "2026-09-28")
    decision = decide(
        _MONDAY,
        100_000.0,
        {"BTCUSDT": 0.002, "ETHUSDT": -0.001, "SOLUSDT": 0.0, "BNBUSDT": 0.0005},
        {},
        {"BTCUSDT": 30_000.0, "ETHUSDT": 1_800.0, "BNBUSDT": 300.0, "SOLUSDT": 40.0},
        {"BTCUSDT": 0, "ETHUSDT": 0, "SOLUSDT": 0, "BNBUSDT": 0},
        _specs(),
        spec,
    )
    assert (decision.book_state, decision.min_leg_skips, decision.exposure_after_plan) == (
        None,
        0,
        None,
    )


# --- clmmc.25: the book sigma reads memoised daily returns, same bits as the scan --------------


def _scan_book_sigma(
    units: dict[str, float], closes: dict[str, dict[int, float]], first_ns: int, lookback: int
) -> float | None:
    """The pre-memo ``_book_sigma``, copied: the oracle of the fast path."""
    import sys

    live = sorted(symbol for symbol, unit in units.items() if unit != 0.0)
    product_limit = sys.float_info.max / (len(live) + 1)
    limit = sys.float_info.max / (lookback + 1)

    def bounded(values: list[float], bound: float) -> bool:
        return all(math.isfinite(value) and abs(value) <= bound for value in values)

    returns: list[float] = []
    for back in range(1, lookback + 1):
        products: list[float] = []
        for symbol in live:
            path = closes.get(symbol, {})
            now = path.get(first_ns - (back - 1) * _TS_DAY)
            before = path.get(first_ns - back * _TS_DAY)
            if now is None or before is None or not (before > 0.0 and now > 0.0):
                return None
            products.append(units[symbol] * (now / before - 1.0))
        if not bounded(products, product_limit):
            return None
        returns.append(math.fsum(products))
    if not bounded(returns, limit):
        return None
    mean = math.fsum(returns) / lookback
    squares = [(value - mean) * (value - mean) for value in returns]
    if not bounded(squares, limit):
        return None
    return math.sqrt(math.fsum(squares) / (lookback - 1))


def _walk(seed: int, length: int = 6) -> list[float]:
    import random

    rng = random.Random(seed)
    path = [100.0 + seed]
    for _ in range(length - 1):
        path.append(path[-1] * (1.0 + rng.uniform(-0.05, 0.05)))
    return path


@pytest.mark.parametrize(
    ("units", "paths"),
    [
        ({"A": 0.4, "B": -0.35, "C": 0.25}, {s: _walk(i) for i, s in enumerate("ABC")}),
        ({"A": 1.0, "B": -1.0}, {s: _walk(i + 3) for i, s in enumerate("AB")}),  # |u| == 1: fast
        ({"A": 1.0000000000000002}, {"A": _walk(7)}),  # |u| just past 1: the scan
        ({"A": 0.5, "B": 0.0}, {"A": _walk(8), "B": _walk(9)}),  # a zero unit is not live
        ({"A": 0.5, "B": 0.5}, {"A": _walk(8), "B": [100.0, 100.0, 0.0, 100.0, 100.0, 100.0]}),
        ({"A": 0.5, "B": 0.5}, {"A": _walk(8), "B": _walk(9)[:4]}),  # a missing close
        ({"A": 0.5}, {"A": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}),  # flat: sigma 0
        # a daily return of exactly 1e100 (the fast bound) and just above it (the scan)
        ({"A": 0.5, "B": 0.5}, {"A": _walk(8), "B": [1.0, 1.0, 1.0, 1.0, 1.0, 1e100 + 1e84]}),
        ({"A": 0.5, "B": 0.5}, {"A": _walk(8), "B": [1.0, 1.0, 1.0, 1.0, 1.0, 1e100 * 1.01]}),
        ({"A": 0.5, "B": 0.5}, {"A": _walk(8), "B": [1.0, 1.0, 1.0, 1.0, 1.0, math.inf]}),
        ({}, {}),
    ],
)
def test_book_sigma_fast_path_equals_the_scan_bit_for_bit(
    units: dict[str, float], paths: dict[str, list[float]]
) -> None:
    from vibe_quant.portfolio.sizing import _book_sigma

    closes = _ts_closes(paths)
    for _ in range(2):  # the second read comes from the memo
        got = _book_sigma(units, closes, _TS_B0, 5)
        want = _scan_book_sigma(units, closes, _TS_B0, 5)
        assert (got is None) == (want is None)
        assert got is None or got.hex() == want.hex()  # type: ignore[union-attr]


def test_book_sigma_memo_is_per_close_map() -> None:
    """The same coin and day over two unrelated load maps gets two sigmas (literal values)."""
    from vibe_quant.portfolio.sizing import _book_sigma

    steady = _ts_closes({"A": [100.0, 101.0, 100.0, 101.0, 100.0, 101.0]})
    wild = _ts_closes({"A": [100.0, 150.0, 90.0, 140.0, 80.0, 130.0]})
    assert _book_sigma({"A": 1.0}, steady, _TS_B0, 5) == 0.010900221193914702
    assert _book_sigma({"A": 1.0}, wild, _TS_B0, 5) == 0.5356689869149175
    assert _book_sigma({"A": 1.0}, steady, _TS_B0, 5) == 0.010900221193914702
