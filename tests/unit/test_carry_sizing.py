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

import pytest

from vibe_quant.data.catalog import create_instrument
from vibe_quant.errors import DataUnavailableError
from vibe_quant.portfolio.sizing import (
    InstrumentSpec,
    OrderIntent,
    decide,
    nt_commission,
    plan_orders,
    target_steps,
)
from vibe_quant.portfolio.spec import (
    CarryDataError,
    CarryEquityTooSmall,
    CarryExecutionError,
    CarrySpec,
    decision_times,
)

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
    """R20: UTC conversion and the 225-decision schedule, pinned in ns."""
    spec = CarrySpec.from_dates(_SYMBOLS, "2022-06-01", "2026-09-28")
    assert spec.start_ns == 1654041600000000000
    assert spec.end_ns == 1790553600000000000
    times = decision_times(spec.start_ns, spec.end_ns)
    assert times[0] == 1654473900000000000
    assert times[-1] == 1789949100000000000
    assert len(times) == 225
    assert spec.symbols == tuple(sorted(_SYMBOLS))


def test_decision_times_never_precede_start() -> None:
    """R20: a start off the Monday 00:05 grid still yields every D >= start_ns."""
    times = decision_times(_MONDAY + 60 * 10**9, _END)  # 00:06 past a Monday 00:05
    assert times[0] == _MONDAY + 7 * 86_400 * 10**9
    assert all(at >= _MONDAY + 60 * 10**9 for at in times)


@pytest.mark.parametrize(
    "value",
    ["2022-06-01T05:00:00", "20220601", "2022-06-01+05:00", "2022-6-1"],
)
def test_from_dates_rejects_non_iso_date_strings(value: str) -> None:
    """R46: the date parser takes exactly ``YYYY-MM-DD``, nothing else."""
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
        pytest.param(
            {"symbols": ("BTCUSDT", "BTCUSDT", "ETHUSDT")}, id="duplicate-of-three"
        ),
        pytest.param({"symbols": ("BTCUSDT", "")}, id="empty-symbol"),
        pytest.param({"start_ns": _START + 1}, id="start-not-midnight"),
        pytest.param({"end_ns": _END + 1}, id="end-not-midnight"),
        pytest.param({"end_ns": _START}, id="start-equals-end"),
        pytest.param({"start_ns": _END}, id="start-after-end"),
    ],
)
def test_spec_rejects_bad_params(over: dict[str, Any]) -> None:
    """R8d: every out-of-range boundary value is refused at construction."""
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
    """R45: bool/str numbers and non-int ns timestamps are refused, not coerced."""
    with pytest.raises(ValueError):
        CarrySpec(**_kwargs(**over))


def test_spec_stores_converted_floats() -> None:
    """R45: gross/band/initial_balance are stored as the converted float."""
    spec = CarrySpec(**_kwargs(gross=1, band=0, initial_balance=10_000))
    assert spec.gross == 1.0
    assert isinstance(spec.gross, float)
    assert spec.band == 0.0
    assert isinstance(spec.band, float)
    assert spec.initial_balance == 10_000.0
    assert isinstance(spec.initial_balance, float)
    assert spec.spec_hash() == CarrySpec(**_kwargs(gross=1.0, band=0.0)).spec_hash()


def test_spec_accepts_boundary_values() -> None:
    """R8d: the inclusive boundary values on the valid side construct fine."""
    for over in (
        {"legs_divisor": 2},
        {"gross": 2.0},
        {"band": 0.0},
        {"band": 0.999},
        {"initial_balance": 1e-9},
    ):
        CarrySpec(**_kwargs(**over))


def test_spec_rejects_empty_decision_schedule() -> None:
    """R20b: a window with no Monday 00:05 decision is refused."""
    with pytest.raises(ValueError, match="no decision in window"):
        CarrySpec(**_kwargs(end_ns=1_654_214_400_000_000_000))  # 2022-06-03


def test_held_symbol_without_price_fails_closed() -> None:
    """R9: rankable + no sizing tick -> ``no_price``; held -> CarryDataError."""
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
    """R11: ``floor(|w| * E / (price * step))`` in Decimal, sign applied after."""
    targets = target_steps(
        {"BTCUSDT": 0.5, "ETHUSDT": -0.5},
        10096.0,
        {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0},
        _specs(),
    )
    assert targets == {"BTCUSDT": 98, "ETHUSDT": -1682}


def test_min_equity_boundary() -> None:
    """R12: the min-equity floor is inclusive; below it the leg fails closed."""
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
    """R13: R11's drift ``(50 + 2) / 10096``, computed once per decision."""
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
    """R14a-R14d: open, close, flip, and the band on same-side resizes."""
    specs = _specs()
    prices = {"BTCUSDT": 51000.0, "SOLUSDT": 100.0}
    orders, band_skips, min_notional_skips = plan_orders(
        {}, {"BTCUSDT": 98}, prices, specs, 0.10
    )
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
    """R14d': 0.07 * 100 is 7.000000000000001 in float, exactly 7.00 in Decimal."""
    specs = {"BTCUSDT": InstrumentSpec.from_instrument(create_instrument("BTCUSDT"))}
    orders, band_skips, _ = plan_orders(
        {"BTCUSDT": 93}, {"BTCUSDT": 100}, {"BTCUSDT": 51000.0}, specs, 0.07
    )
    assert orders == (OrderIntent("BTCUSDT", 7, False),)
    assert band_skips == 0


def test_min_quantity_edge_is_tradable() -> None:
    """R14e/M3: a 1-step BTC order equals ``min_quantity`` and must still be sent."""
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
    """R14e: an increase below MIN_NOTIONAL is dropped, a reduce-only is sent."""
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
    """R43/M4: a flip's close is sent, its $1 open leg is dropped and counted."""
    specs = {"SOLUSDT": InstrumentSpec.from_instrument(create_instrument("SOLUSDT"))}
    orders, band_skips, min_notional_skips = plan_orders(
        {"SOLUSDT": 10}, {"SOLUSDT": -1}, {"SOLUSDT": 100.0}, specs, 0.1
    )
    assert orders == (OrderIntent("SOLUSDT", -10, True),)
    assert (band_skips, min_notional_skips) == (0, 1)


def test_band_takes_precedence_over_min_notional() -> None:
    """R14f: a band skip never reaches the min-notional check; closes are exempt."""
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
    decision = decide(_START, 10_000.0, signals, {"BTCUSDT": "funding_hole"}, prices, {"BTCUSDT": 5}, _specs(), spec)
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
    """R44c: ``not (equity > 0.0)`` fails closed FIRST, for every signal state.

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
    assert decide(_START, 1e-9, dict.fromkeys(_SYMBOLS), {}, {}, {}, specs, spec).rounding_drift == 0.0


@pytest.mark.parametrize(
    "equity", [0.0, -1.0, float("nan")], ids=["zero", "negative", "nan"]
)
def test_equity_guard_precedes_held_unpriced(equity: float) -> None:
    """R1: with BOTH a non-positive equity and a held-but-unpriced leg, ``CarryEquityTooSmall`` wins.

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
