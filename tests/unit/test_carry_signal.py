"""Funding-carry signal rules.

Contract rows R1-R8c of the funding-carry semantics contract
(``data/swarm/20261015-funding-carry/design.md`` §2 and §7). The window is
``(L - 7 d, L]`` with ``L = D - 60 s``; the exact float literals below are the
contract's hand-computed sums and are the point of the tests.
"""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import create_instrument
from vibe_quant.metrics import BarSeries
from vibe_quant.portfolio.signal import (
    Settlements,
    _trailing_funding,
    carry_hours,
    decision_signals,
    inverse_vol_weights,
    rank_weights,
    trailing_funding,
    trailing_quote_volume,
    trailing_vol,
    trend_signal,
)
from vibe_quant.portfolio.sizing import InstrumentSpec, OrderIntent, decide
from vibe_quant.portfolio.spec import CarrySpec
from vibe_quant.portfolio.ticks import PriceTick, boundary_prices, sizing_price

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_NANOS_PER_S = 10**9
_DAY_NS = 86_400 * _NANOS_PER_S
_EIGHT_HOURS_NS = 8 * 3_600 * _NANOS_PER_S
_FOUR_HOURS_NS = 4 * 3_600 * _NANOS_PER_S

_D = 1_654_473_900 * _NANOS_PER_S  # Monday 2022-06-06 00:05:00 UTC
_L = _D - 60 * _NANOS_PER_S
_LO = _L - 7 * _DAY_NS

_ON_GRID_SUM = 0.0021000000000000003


def _grid() -> list[int]:
    """The 21 8h UTC boundaries inside the signal window ``(L-7d, L]``."""
    first = (_LO // _EIGHT_HOURS_NS + 1) * _EIGHT_HOURS_NS
    return [first + i * _EIGHT_HOURS_NS for i in range(21)]


def _four_hour_times() -> list[int]:
    """The 42 4h settlement times inside the signal window."""
    first = (_LO // _FOUR_HOURS_NS + 1) * _FOUR_HOURS_NS
    return [first + i * _FOUR_HOURS_NS for i in range(42)]


def _settlements(times: list[int], rates: list[float]) -> Settlements:
    return Settlements(times_ns=tuple(times), rates=tuple(rates))


def test_settlement_at_safety_edge_is_included() -> None:
    """A row exactly at ``L`` is in the window, at ``L+1ns`` it is out."""
    grid = _grid()
    at_l = _settlements(grid + [_L], [0.0001] * 21 + [0.001])
    assert trailing_funding(at_l, _D) == 0.0031000000000000003
    after_l = _settlements(grid + [_L + 1], [0.0001] * 21 + [0.001])
    assert trailing_funding(after_l, _D) == _ON_GRID_SUM


def test_settlement_at_lookback_edge_is_excluded() -> None:
    """A row exactly at ``L-7d`` is out (open interval), at ``+1ns`` in."""
    grid = _grid()
    at_lo = _settlements([_LO, *grid], [0.0001] * 22)
    assert trailing_funding(at_lo, _D) == _ON_GRID_SUM
    after_lo = _settlements([_LO + 1, *grid], [0.0001] * 22)
    assert trailing_funding(after_lo, _D) == 0.0022


def test_uncovered_8h_boundary_excludes_symbol() -> None:
    """Every 8h boundary needs a row within 60 s, else ``funding_hole``."""
    grid = _grid()
    boundary = grid[5]
    hole = _settlements([*grid[:5], boundary + 61 * _NANOS_PER_S, *grid[6:]], [0.0001] * 21)
    assert trailing_funding(hole, _D) is None
    covered = _settlements([*grid[:5], boundary + 3_000_000, *grid[6:]], [0.0001] * 21)
    assert trailing_funding(covered, _D) == _ON_GRID_SUM


def test_8h_boundary_row_at_upper_safety_edge() -> None:
    """A row exactly at ``b + 60 s`` covers the boundary, ``b + 61 s`` is a hole."""
    grid = _grid()
    boundary = grid[5]
    at_edge = _settlements([*grid[:5], boundary + 60 * _NANOS_PER_S, *grid[6:]], [0.0001] * 21)
    assert trailing_funding(at_edge, _D) == _ON_GRID_SUM
    beyond = _settlements([*grid[:5], boundary + 61 * _NANOS_PER_S, *grid[6:]], [0.0001] * 21)
    assert trailing_funding(beyond, _D) is None


def test_8h_boundary_row_at_lower_safety_edge() -> None:
    """A row exactly at ``b - 60 s`` covers, ``b - 60 s - 1 ns`` is a hole."""
    grid = _grid()
    boundary = grid[5]
    at_edge = _settlements([*grid[:5], boundary - 60 * _NANOS_PER_S, *grid[6:]], [0.0001] * 21)
    assert trailing_funding(at_edge, _D) == _ON_GRID_SUM
    beyond = _settlements([*grid[:5], boundary - 60 * _NANOS_PER_S - 1, *grid[6:]], [0.0001] * 21)
    assert trailing_funding(beyond, _D) is None


def test_4h_and_8h_sums_tie_exactly() -> None:
    """``/(8/N)``-scaled intervals sum to the same total, tie -> key order."""
    grid = _grid()
    eight_hour = _settlements(grid, [0.0001] * 21)
    four_hour = _settlements(_four_hour_times(), [0.00005] * 42)
    signal_8h = trailing_funding(eight_hour, _D)
    signal_4h = trailing_funding(four_hour, _D)
    assert signal_8h == _ON_GRID_SUM
    assert signal_4h == _ON_GRID_SUM
    assert signal_8h == signal_4h
    weights = rank_weights({"BTCUSDT": signal_8h, "ETHUSDT": signal_4h}, 2, 1.0)
    assert weights == {"BTCUSDT": 0.5, "ETHUSDT": -0.5}


def test_missing_4h_row_is_not_detected() -> None:
    """A missing NON-8h row is summed as archived (accepted limitation)."""
    times = _four_hour_times()
    del times[0]  # the 04:00 row: not an 8h boundary
    four_hour = _settlements(times, [0.00005] * 41)
    assert trailing_funding(four_hour, _D) == 0.00205


def test_rank_long_lowest_short_highest() -> None:
    """Total order by ``(signal, symbol)``; k = n // divisor legs a side."""
    weights = rank_weights(
        {
            "BTCUSDT": 0.0021,
            "ETHUSDT": 0.0030,
            "SOLUSDT": -0.0010,
            "BNBUSDT": 0.0005,
        },
        3,
        1.0,
    )
    assert weights == {
        "BTCUSDT": 0.0,
        "ETHUSDT": -0.5,
        "SOLUSDT": 0.5,
        "BNBUSDT": 0.0,
    }


def test_tie_broken_by_symbol() -> None:
    """A tie on the signal is broken by the symbol, tying legs stay ties."""
    weights = rank_weights({"BTCUSDT": 0.0021, "ETHUSDT": 0.0021, "SOLUSDT": -0.0010}, 3, 1.0)
    assert weights["SOLUSDT"] == 0.5
    assert weights["ETHUSDT"] == -0.5
    assert weights["BTCUSDT"] == 0.0


def test_leg_count_edges() -> None:
    """K = max(1, n // divisor) for n >= 2, flat for n < 2."""
    for n in (4, 3, 2):
        signals = {f"S{i}USDT": float(i) for i in range(n)}
        weights = rank_weights(signals, 3, 1.0)
        assert sum(1 for w in weights.values() if w > 0) == 1
        assert sum(1 for w in weights.values() if w < 0) == 1
    middle = rank_weights({"AUSDT": 1.0, "BUSDT": 2.0, "CUSDT": 3.0}, 3, 1.0)
    assert middle == {"AUSDT": 0.5, "BUSDT": 0.0, "CUSDT": -0.5}
    # N = 1 or 0 -> flat
    assert rank_weights({"AUSDT": 1.0}, 3, 1.0) == {"AUSDT": 0.0}
    assert rank_weights({}, 3, 1.0) == {}
    # N = 6 div 3 and n = 5 div 2 both give k = 2 -> +-0.25
    six = rank_weights({f"S{i}USDT": float(i) for i in range(6)}, 3, 1.0)
    assert sorted(w for w in six.values() if w) == [-0.25, -0.25, 0.25, 0.25]
    five = rank_weights({f"S{i}USDT": float(i) for i in range(5)}, 2, 1.0)
    assert sorted(w for w in five.values() if w) == [-0.25, -0.25, 0.25, 0.25]


def test_unrankable_symbols_are_not_ranked() -> None:
    """A ``None`` signal (funding hole / no price) is flat and out of the order."""
    weights = rank_weights({"AUSDT": 0.002, "BUSDT": None, "CUSDT": -0.002}, 3, 1.0)
    assert weights == {"AUSDT": -0.5, "BUSDT": 0.0, "CUSDT": 0.5}


def test_4h_symbol_missing_row_is_unrankable() -> None:
    """A missing 4h row in the window is ``None``; with hours=None it is summed."""
    times = _four_hour_times()
    del times[0]  # the 04:00 row: not an 8h boundary, but 4h on this symbol
    rates = [0.00005] * len(times)
    four_hour = Settlements(times_ns=tuple(times), rates=tuple(rates), hours=(4,) * len(times))

    assert trailing_funding(four_hour, _D) is None

    # the same rows under the default 8h rule: summed as archived, the blind spot
    assert trailing_funding(Settlements(tuple(times), tuple(rates)), _D) == 0.00205


def test_settlements_hours_default_and_mismatch() -> None:
    """``hours`` defaults to None (8h rule exactly as today); mismatch raises."""
    grid = _grid()
    default = _settlements(grid, [0.0001] * 21)

    assert default.hours is None
    assert trailing_funding(default, _D) == _ON_GRID_SUM
    assert trailing_funding(Settlements(tuple(grid), (0.0001,) * 21, (8,) * 21), _D) == _ON_GRID_SUM
    with pytest.raises(ValueError, match=r"^Settlements\.hours has 3 entries for 4 settlements$"):
        Settlements(tuple(grid[:4]), (0.0001,) * 4, (2, 2, 2))


def test_carry_hours_collapses_all_8h() -> None:
    """All-8h (or empty) hours collapse to None; a real mix survives."""
    assert carry_hours(()) is None
    assert carry_hours((8, 8, 8)) is None
    assert carry_hours((8, 2, 8)) == (8, 2, 8)
    assert carry_hours((2, 2)) == (2, 2)


def test_trailing_funding_memo_returns_the_uncached_value_every_time() -> None:
    """The per-Settlements memo hands back exactly what the rule computes, hole or not."""
    grid = _grid()
    full = _settlements(grid, [0.0001] * 21)
    holed = _settlements(grid[:10] + grid[11:], [0.0001] * 20)
    for settlements, expected in ((full, _ON_GRID_SUM), (holed, None)):
        assert _trailing_funding(settlements, _D) == expected
        for _ in range(3):
            assert trailing_funding(settlements, _D) == expected
    # another decision on the same series is its own entry, not the first one's value
    assert trailing_funding(full, _D + _EIGHT_HOURS_NS) == _trailing_funding(
        full, _D + _EIGHT_HOURS_NS
    )
    assert set(full._trailing) == {_D, _D + _EIGHT_HOURS_NS}


def test_trailing_funding_memo_is_per_series_and_not_part_of_equality() -> None:
    """Two series never share a memo; the memo does not change ==, hash or repr."""
    grid = _grid()
    one = _settlements(grid, [0.0001] * 21)
    other = _settlements(grid, [0.0002] * 21)
    assert trailing_funding(one, _D) == _ON_GRID_SUM
    assert trailing_funding(other, _D) == 0.0042000000000000006
    assert one._trailing is not other._trailing
    fresh = _settlements(grid, [0.0001] * 21)
    assert one == fresh
    assert hash(one) == hash(fresh)
    assert repr(one) == repr(fresh)


# --- decision_signals: the one eligibility rule both executors call ------------------

_ELIG_SYMBOLS = ("BNBUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT")


def _elig_spec() -> CarrySpec:
    return CarrySpec.from_dates(_ELIG_SYMBOLS, "2022-06-01", "2026-09-28")


def _loops_today(
    spec: CarrySpec,
    settlements: Mapping[str, Settlements],
    ticks: Mapping[str, Sequence[PriceTick]],
    decision_ns: int,
) -> tuple[dict[str, float | None], dict[str, str], dict[str, float]]:
    """main@42f825f5's reference loops, verbatim: the oracle the helper must equal."""
    prices: dict[str, float] = {}
    for symbol in spec.symbols:
        tick = sizing_price(ticks[symbol], decision_ns)
        if tick is not None:
            prices[symbol] = tick.price
    signals: dict[str, float | None] = {}
    excluded: dict[str, str] = {}
    for symbol in spec.symbols:
        value = trailing_funding(settlements[symbol], decision_ns)
        if value is None:
            signals[symbol] = None
            excluded[symbol] = "funding_hole"
        elif symbol not in prices:
            signals[symbol] = None
            excluded[symbol] = "no_price"
        else:
            signals[symbol] = value
    return signals, excluded, prices


def _elig_inputs(
    states: Mapping[str, tuple[bool, bool]],
) -> tuple[dict[str, Settlements], dict[str, tuple[PriceTick, ...]]]:
    """Per symbol ``(funding hole?, sizing tick?)``; distinct rates so signals differ."""
    grid = _grid()
    settlements: dict[str, Settlements] = {}
    ticks: dict[str, tuple[PriceTick, ...]] = {}
    for index, symbol in enumerate(_ELIG_SYMBOLS):
        hole, priced = states[symbol]
        rate = 0.0001 * (index + 1)
        times = grid[:7] + grid[8:] if hole else grid
        settlements[symbol] = _settlements(times, [rate] * len(times))
        price = 100.0 * (index + 1)
        # a tick outside [D-60 min, D) never sizes; the in-window one does
        stale = PriceTick(_D - 61 * 60 * _NANOS_PER_S, price + 1.0)
        ticks[symbol] = (stale, PriceTick(_D - 1_000_000, price)) if priced else (stale,)
    return settlements, ticks


def _sizing(
    ticks: Mapping[str, Sequence[PriceTick]], decision_ns: int
) -> dict[str, PriceTick | None]:
    return {symbol: sizing_price(ticks[symbol], decision_ns) for symbol in _ELIG_SYMBOLS}


@pytest.mark.parametrize(
    "states",
    [
        # every (hole, tick) combination on each symbol at least once
        dict.fromkeys(_ELIG_SYMBOLS, (False, True)),
        {
            "BNBUSDT": (False, True),
            "BTCUSDT": (True, True),
            "ETHUSDT": (False, False),
            "SOLUSDT": (True, False),
        },
        {
            "BNBUSDT": (True, False),
            "BTCUSDT": (False, False),
            "ETHUSDT": (True, True),
            "SOLUSDT": (False, True),
        },
        dict.fromkeys(_ELIG_SYMBOLS, (True, False)),
    ],
)
def test_decision_signals_matches_loops(states: dict[str, tuple[bool, bool]]) -> None:
    """R7: same signals/excluded/prices as today's loops, for every data-gap combination."""
    spec = _elig_spec()
    settlements, ticks = _elig_inputs(states)
    expected = _loops_today(spec, settlements, ticks, _D)

    got = decision_signals(spec, settlements, _sizing(ticks, _D), _D)

    assert (got.signals, got.excluded, got.prices) == expected
    for symbol, (hole, priced) in states.items():
        if hole:
            assert got.excluded[symbol] == "funding_hole"  # the hole wins over no tick
        elif not priced:
            assert got.excluded[symbol] == "no_price"
        else:
            assert symbol not in got.excluded
            assert got.signals[symbol] is not None
        # a symbol with a sizing tick keeps its price, excluded or not
        assert (symbol in got.prices) is priced
    assert list(got.signals) == list(_ELIG_SYMBOLS)


def test_decision_signals_literal_values() -> None:
    """R7: the helper's own numbers, literal (an oracle copy cannot hide a shared slip)."""
    states = {
        "BNBUSDT": (False, True),
        "BTCUSDT": (True, True),
        "ETHUSDT": (False, False),
        "SOLUSDT": (False, True),
    }
    settlements, ticks = _elig_inputs(states)

    got = decision_signals(_elig_spec(), settlements, _sizing(ticks, _D), _D)

    assert got.signals == {
        "BNBUSDT": _ON_GRID_SUM,
        "BTCUSDT": None,
        "ETHUSDT": None,
        "SOLUSDT": 0.008400000000000001,
    }
    assert got.excluded == {"BTCUSDT": "funding_hole", "ETHUSDT": "no_price"}
    assert got.prices == {"BNBUSDT": 100.0, "BTCUSDT": 200.0, "SOLUSDT": 400.0}


def test_decision_signals_call_contract() -> None:
    """R7a: 4 positional params; the outputs and the ``decide`` plan equal today's loop.

    A held leg excluded for a funding hole keeps its sizing price, so ``decide``
    closes it reduce-only at that price.
    """
    every = list(inspect.signature(decision_signals).parameters.values())
    params = [p for p in every if p.kind is not inspect.Parameter.KEYWORD_ONLY]
    assert [p.name for p in params] == ["spec", "settlements", "sizing", "decision_ns"]
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params)
    assert all(p.default is inspect.Parameter.empty for p in params)
    # variant maps (T4 on) are keyword-only and default to None
    assert all(p.default is None for p in every if p not in params)

    spec = _elig_spec()
    states = {
        "BNBUSDT": (False, True),
        "BTCUSDT": (True, True),  # held, funding hole, priced
        "ETHUSDT": (False, True),
        "SOLUSDT": (False, True),
    }
    settlements, ticks = _elig_inputs(states)
    sizing = {s: sizing_price(ticks[s], _D) for s in spec.symbols}
    specs = {s: InstrumentSpec.from_instrument(create_instrument(s)) for s in spec.symbols}
    current = {"BNBUSDT": 0, "BTCUSDT": 5, "ETHUSDT": 0, "SOLUSDT": 0}

    got = decision_signals(spec, settlements, sizing, _D)
    signals, excluded, prices = _loops_today(spec, settlements, ticks, _D)

    assert (got.signals, got.excluded, got.prices) == (signals, excluded, prices)
    assert got.prices["BTCUSDT"] == 200.0
    plan = decide(_D, 100_000.0, got.signals, got.excluded, got.prices, current, specs, spec)
    today = decide(_D, 100_000.0, signals, excluded, prices, current, specs, spec)
    assert plan == today
    assert OrderIntent("BTCUSDT", -5, reduce_only=True) in plan.orders
    assert plan.prices["BTCUSDT"] == 200.0


# ---------------------------------------------------------------------------
# inverse-vol weighting (x0axf.14 T4: R16-R20, R22)
# ---------------------------------------------------------------------------

_B0 = _D - _D % _DAY_NS  # the decision day's 00:00 boundary


def _closes(*values: float, last: int = _B0) -> dict[int, float]:
    """Daily closes keyed by boundary ns, oldest first, the last one at ``last``."""
    first = last - (len(values) - 1) * _DAY_NS
    return {first + index * _DAY_NS: value for index, value in enumerate(values)}


def test_trailing_vol_numbers() -> None:
    """R16: closes 100, 110, 99, 99 (L=3) -> sigma 0.10000000000000003 (chronological fsum)."""
    assert trailing_vol(_closes(100.0, 110.0, 99.0, 99.0), _D, 3) == 0.10000000000000003
    # an unrelated input: two alternating returns, literal value
    assert (
        trailing_vol(_closes(50.0, 55.0, 50.0), _D, 2) == 0.13499311277197734
    )  # statistics.stdev agrees
    # extra older closes beyond the lookback are never read
    assert trailing_vol({**_closes(100.0, 110.0, 99.0, 99.0), _B0 - 9 * _DAY_NS: 1.0}, _D, 3) == (
        0.10000000000000003
    )


def test_trailing_vol_never_reads_future() -> None:
    """R17: a close after ``B0`` (next day, or later the same day as D) changes nothing."""
    closes = _closes(100.0, 110.0, 99.0, 99.0)
    for later in (_B0 + _DAY_NS, _B0 + 1, _B0 + 7 * _DAY_NS):
        assert trailing_vol({**closes, later: 1e6}, _D, 3) == 0.10000000000000003
    # D exactly on 00:00 reads the B0 close itself (the last step back is j = 0)
    assert trailing_vol(closes, _B0, 3) == 0.10000000000000003


@pytest.mark.parametrize("bad", [None, 0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize("position", [0, 2, 3])
def test_trailing_vol_rejects(bad: float | None, position: int) -> None:
    """R18: a missing / 0 / negative / non-finite close -> None (never forward-filled)."""
    closes = _closes(100.0, 110.0, 99.0, 99.0)
    key = _B0 - (3 - position) * _DAY_NS
    if bad is None:
        del closes[key]
    else:
        closes[key] = bad
    assert trailing_vol(closes, _D, 3) is None


def test_trailing_vol_flat_is_none() -> None:
    """R18: flat closes give sigma 0 -> None (no_vol), and so does a near-zero edge."""
    assert trailing_vol(_closes(5.0, 5.0, 5.0, 5.0), _D, 3) is None
    # a one-ulp drift is a positive sigma and is accepted (statistics.stdev: ...092e-16)
    assert trailing_vol(_closes(5.0, 5.0, 5.0, math.nextafter(5.0, math.inf)), _D, 3) == (
        1.2819751242557095e-16
    )


def _bars(ts: list[int], closes: list[float]) -> BarSeries:
    values = np.array(closes, dtype=np.float64)
    return BarSeries(
        ts=np.array(ts, dtype=np.int64), open=values, high=values, low=values, close=values
    )


def test_trailing_vol_boundary_and_overflow() -> None:
    """R18a: an in-window close stands in for a missing last minute; overflow -> None.

    Closes come from ``boundary_prices`` (``ts_init`` in ``(B-60 min, B]``). B1's
    last minute is missing but a bar 30 min earlier is used; B2 has its only bar
    exactly 60 min before B2 (outside the window), so it is absent and the vol is
    None -- the previous day's close is never carried forward.
    """
    instrument = create_instrument("BTCUSDT")
    minute = 60 * _NANOS_PER_S
    b = [_B0 - (3 - j) * _DAY_NS for j in range(4)]
    ts = [b[0], b[1] - 30 * minute, b[2], b[3]]
    closes = boundary_prices(_bars(ts, [100.0, 110.0, 99.0, 99.0]), b, instrument)
    assert closes == dict(zip(b, [100.0, 110.0, 99.0, 99.0], strict=True))
    assert trailing_vol(closes, _D, 3) == 0.10000000000000003

    gap = boundary_prices(_bars([b[0], b[1], b[2] - 60 * minute, b[3]], [100.0] * 4), b, instrument)
    assert b[2] not in gap
    assert trailing_vol(gap, _D, 3) is None

    # a return that overflows to inf (1e300 / 1e-300), and squared deviations that do
    assert trailing_vol(_closes(1e-300, 1e300, 1.0, 1.0), _D, 3) is None
    assert trailing_vol(_closes(1e-150, 1e150, 1e-150, 1e-150), _D, 3) is None
    # finite returns 1e308, -1, 1e308 whose plain fsum raises OverflowError
    with pytest.raises(OverflowError):
        math.fsum([1e308 - 1.0, -1.0, 1e308 - 1.0])
    assert trailing_vol(_closes(1e-300, 1e8, 1e-300, 1e8), _D, 3) is None


@pytest.mark.parametrize("lookback", [3, 12])
def test_trailing_vol_never_overflows_when_max_over_l_rounds_up(lookback: int) -> None:
    """R18a: float max / L rounds up for L = 3, 12; L squares just under it must not overflow fsum."""
    closes = {
        _B0 - j * _DAY_NS: (1.0 if j % 2 == 0 else 7.741001517595157e153)
        for j in range(lookback + 1)
    }
    vol = trailing_vol(closes, _D, lookback)  # must not raise OverflowError
    assert vol is None or math.isfinite(vol)


def test_inverse_vol_per_side() -> None:
    """R19: longs A .02 / B .04, shorts C .02 / D .04, gross 1.0 -> exact 1/3, 1/6 per side."""
    weights = {"A": 0.25, "B": 0.25, "C": -0.25, "D": -0.25, "E": 0.0}
    vols = {"A": 0.02, "B": 0.04, "C": 0.02, "D": 0.04}
    assert inverse_vol_weights(weights, vols, 1.0) == {
        "A": 0.3333333333333333,
        "B": 0.16666666666666666,
        "C": -0.3333333333333333,
        "D": -0.16666666666666666,
        "E": 0.0,
    }
    # a second, unrelated input: three longs, gross 2.0
    three = {"A": 2 / 6, "B": 2 / 6, "C": 2 / 6, "D": -1.0}
    got = inverse_vol_weights(three, {"A": 0.01, "B": 0.02, "C": 0.04, "D": 0.5}, 2.0)
    assert got == {
        "A": 0.5714285714285714,
        "B": 0.2857142857142857,
        "C": 0.14285714285714285,
        "D": -1.0,
    }
    assert math.fsum(got[s] for s in "ABC") == 1.0


def test_one_leg_side_unchanged() -> None:
    """R20: one leg per side keeps ``rank_weights`` bit for bit (gross 1.37 -> 0.685)."""
    weights = rank_weights({"A": 0.1, "B": 0.2, "C": 0.3}, 3, 1.37)
    assert weights == {"A": 0.685, "B": 0.0, "C": -0.685}
    assert inverse_vol_weights(weights, {"A": 0.01, "C": 0.9}, 1.37) == weights


@pytest.mark.parametrize(
    "vols",
    [None, {"A": 0.02}, {"A": 0.02, "B": 0.0}, {"A": 0.02, "B": -0.01}, {"A": 0.02, "B": math.nan}],
)
def test_inverse_vol_requires_vols(vols: dict[str, float] | None) -> None:
    """R22: a picked leg without a finite sigma > 0 is a programming error."""
    weights = {"A": 0.25, "B": 0.25, "C": -0.5}
    full = None if vols is None else {**vols, "C": 0.03}
    name = "A" if vols is None else "B"
    with pytest.raises(ValueError, match=f"inverse_vol weighting needs a vol for {name}$"):
        inverse_vol_weights(weights, full, 1.0)


def _vol_spec() -> CarrySpec:
    return CarrySpec.from_dates(
        _ELIG_SYMBOLS, "2022-06-01", "2026-09-28", weighting="inverse_vol", vol_lookback_days=3
    )


def test_decision_signals_no_vol() -> None:
    """A rankable symbol without a vol is ``no_vol``; earlier reasons win; vols keyed."""
    states = {
        "BNBUSDT": (False, True),  # vol ok
        "BTCUSDT": (True, True),  # funding_hole wins over a missing vol
        "ETHUSDT": (False, False),  # no_price wins over a missing vol
        "SOLUSDT": (False, True),  # no vol -> no_vol
    }
    settlements, ticks = _elig_inputs(states)
    vol_closes = {
        "BNBUSDT": _closes(100.0, 110.0, 99.0, 99.0),
        "BTCUSDT": {},
        "ETHUSDT": {},
        "SOLUSDT": _closes(5.0, 5.0, 5.0, 5.0),
    }

    got = decision_signals(_vol_spec(), settlements, _sizing(ticks, _D), _D, vol_closes=vol_closes)

    assert got.excluded == {
        "BTCUSDT": "funding_hole",
        "ETHUSDT": "no_price",
        "SOLUSDT": "no_vol",
    }
    assert got.signals == {
        "BNBUSDT": _ON_GRID_SUM,
        "BTCUSDT": None,
        "ETHUSDT": None,
        "SOLUSDT": None,
    }
    assert got.vols == {"BNBUSDT": 0.10000000000000003}
    assert got.prices == {"BNBUSDT": 100.0, "BTCUSDT": 200.0, "SOLUSDT": 400.0}
    # equal weighting: vols is None and the vol map is never needed
    assert decision_signals(_elig_spec(), settlements, _sizing(ticks, _D), _D).vols is None
    with pytest.raises(ValueError, match="vol_closes"):
        decision_signals(_vol_spec(), settlements, _sizing(ticks, _D), _D)


# ---------------------------------------------------------------------------
# liquidity filter (x0axf.14 T5: R27-R29, R35)
# ---------------------------------------------------------------------------


def _week(*sums: float, last: int = _B0 - _DAY_NS) -> dict[int, float]:
    """Daily quote-volume sums keyed by UTC day start, oldest first, the last at ``last``."""
    first = last - (len(sums) - 1) * _DAY_NS
    return {first + index * _DAY_NS: total for index, total in enumerate(sums)}


def _liq_spec(threshold: float, **over: object) -> CarrySpec:
    return CarrySpec.from_dates(
        _ELIG_SYMBOLS, "2022-06-01", "2026-09-28", min_quote_volume=threshold, **over
    )


def _liquid_inputs() -> tuple[dict[str, Settlements], dict[str, PriceTick | None]]:
    settlements, ticks = _elig_inputs(dict.fromkeys(_ELIG_SYMBOLS, (False, True)))
    return settlements, _sizing(ticks, _D)


def test_quote_volume_threshold() -> None:
    """R27: 7 x 1.0e8 at threshold 1e8 is liquid (inclusive); at nextafter(1e8) illiquid."""
    days = _week(*[1.0e8] * 7)
    assert trailing_quote_volume(days, _D) == 1.0e8
    settlements, sizing = _liquid_inputs()
    volume = dict.fromkeys(_ELIG_SYMBOLS, days)

    liquid = decision_signals(_liq_spec(1.0e8), settlements, sizing, _D, quote_volume=volume)
    above = math.nextafter(1.0e8, math.inf)
    assert above == 100000000.00000001
    illiquid = decision_signals(_liq_spec(above), settlements, sizing, _D, quote_volume=volume)

    assert liquid.excluded == {}
    assert all(value is not None for value in liquid.signals.values())
    assert illiquid.excluded == dict.fromkeys(_ELIG_SYMBOLS, "illiquid")
    assert illiquid.signals == dict.fromkeys(_ELIG_SYMBOLS, None)
    assert illiquid.prices == liquid.prices  # an illiquid symbol keeps its price
    # an unrelated mean: (1+2+...+7) e6 / 7 = 4e6
    assert trailing_quote_volume(_week(*[k * 1e6 for k in range(1, 8)]), _D) == 4.0e6
    # off: no map needed; on without the map: ValueError naming it
    assert decision_signals(_elig_spec(), settlements, sizing, _D).excluded == {}
    with pytest.raises(ValueError, match="needs CarryInputs.quote_volume"):
        decision_signals(_liq_spec(1.0e8), settlements, sizing, _D)


def test_quote_volume_mean_no_overflow() -> None:
    """R27a: 7 x 1.7e308 overflows fsum; the s/7 fallback gives a finite mean."""
    with pytest.raises(OverflowError):
        math.fsum([1.7e308] * 7)
    assert trailing_quote_volume(_week(*[1.7e308] * 7), _D) == 1.7e308
    # ordinary inputs take the plain fsum/7 path: for 0.1..0.7 it gives ...997, while
    # the s/7 fallback would give 0.4 -- so the fallback never runs without overflow
    tenths = _week(*[k / 10 for k in range(1, 8)])
    assert trailing_quote_volume(tenths, _D) == 0.39999999999999997
    assert math.fsum(k / 10 / 7 for k in range(1, 8)) == 0.4


def _archive_with(path: Path, rows: list[tuple[int, float]], symbol: str = "BTCUSDT") -> None:
    """1m klines at ``open_time_ms`` with the given quote volume in a tmp archive."""
    archive = RawDataArchive(path)
    archive.insert_klines(
        symbol,
        "1m",
        [(t, 1.0, 1.0, 1.0, 1.0, 1.0, t + 59_999, qv) for t, qv in rows],
        "test",
    )
    archive.close()


def _daily(path: Path, symbol: str = "BTCUSDT") -> dict[int, float]:
    archive = RawDataArchive(path)
    try:
        days = archive.get_daily_quote_volume(symbol, 0, 2**62 // 10**6)
    finally:
        archive.close()
    return {day: total for day, (total, _count) in days.items()}


def test_volume_window_edges(tmp_path: Path) -> None:
    """R28: a kline opening at B0 - 60 000 ms counts (day B0-1); one at B0 is never read."""
    b0_ms = _B0 // 10**6
    day_ms = _DAY_NS // 10**6
    rows = [(b0_ms - k * day_ms, 7.0e8) for k in range(1, 8)]  # 00:00 of each prior day
    _archive_with(tmp_path / "a.db", [*rows, (b0_ms - 60_000, 7.0e8)])
    _archive_with(tmp_path / "b.db", [*rows, (b0_ms, 7.0e10)])

    # a: the last minute of B0-1 adds to that day -> mean (7 x 7e8 + 7e8) / 7 = 8e8
    assert trailing_quote_volume(_daily(tmp_path / "a.db"), _D) == 8.0e8
    # b: the B0 minute lands on day B0, outside [B0 - 7 d, B0)
    assert trailing_quote_volume(_daily(tmp_path / "b.db"), _D) == 7.0e8


def test_volume_prelisting() -> None:
    """R29: a missing day is a volume_hole; the listing day counts its partial sum."""
    settlements, sizing = _liquid_inputs()
    listing = _B0 - 6 * _DAY_NS  # six days of data before B0: one short
    partial = {listing + k * _DAY_NS: (3.0e6 if k == 0 else 1.0e8) for k in range(6)}
    volume = dict.fromkeys(_ELIG_SYMBOLS, _week(*[1.0e9] * 7))
    volume["SOLUSDT"] = partial

    early = decision_signals(_liq_spec(1.0), settlements, sizing, _D, quote_volume=volume)
    assert early.excluded == {"SOLUSDT": "volume_hole"}
    assert trailing_quote_volume(partial, _D) is None

    # one day later the 7-day lookback holds L..L+6: partial L sum + six full days
    later = {**partial, _B0: 1.0e8}
    assert trailing_quote_volume(later, _D + _DAY_NS) == (3.0e6 + 6 * 1.0e8) / 7
    # a hole in the middle (not only before listing) is a hole too
    holed = _week(*[1.0e9] * 7)
    del holed[_B0 - 3 * _DAY_NS]
    assert trailing_quote_volume(holed, _D) is None


def test_exclusion_precedence() -> None:
    """R35: funding_hole > no_price > volume_hole > illiquid > no_vol."""
    states = {
        "BNBUSDT": (True, False),  # hole + no price + illiquid -> funding_hole
        "BTCUSDT": (False, False),  # no price + illiquid -> no_price
        "ETHUSDT": (False, True),  # volume_hole + no vol -> volume_hole
        "SOLUSDT": (False, True),  # illiquid + no vol -> illiquid
    }
    settlements, ticks = _elig_inputs(states)
    low = _week(*[1.0] * 7)
    volume = {"BNBUSDT": low, "BTCUSDT": low, "ETHUSDT": {}, "SOLUSDT": low}
    spec = _liq_spec(1.0e8, weighting="inverse_vol", vol_lookback_days=3)

    got = decision_signals(
        spec, settlements, _sizing(ticks, _D), _D, vol_closes={}, quote_volume=volume
    )

    assert got.excluded == {
        "BNBUSDT": "funding_hole",
        "BTCUSDT": "no_price",
        "ETHUSDT": "volume_hole",
        "SOLUSDT": "illiquid",
    }
    assert got.vols == {}
    # a liquid symbol without a vol still ends as no_vol
    liquid = {**volume, "SOLUSDT": _week(*[1.0e9] * 7)}
    got = decision_signals(
        spec, settlements, _sizing(ticks, _D), _D, vol_closes={}, quote_volume=liquid
    )
    assert got.excluded["SOLUSDT"] == "no_vol"


# ---------------------------------------------------------------------------
# trend signal: the stateless close-vs-SMA ensemble (clmmc.9 T2: R17-R19)
# ---------------------------------------------------------------------------

_TREND_D = 1_710_118_800 * _NANOS_PER_S  # 2024-03-11T01:00:00Z, a daily decision
_TREND_B0 = _TREND_D - _TREND_D % _DAY_NS


def _trend_closes(recent: float, older: float, cut: int = 20) -> dict[int, float]:
    """``C[B0 - j*DAY]`` = ``recent`` for j < ``cut``, ``older`` for cut <= j < 160."""
    return {_TREND_B0 - j * _DAY_NS: (recent if j < cut else older) for j in range(160)}


@pytest.mark.parametrize(
    ("closes", "expected"),
    [
        # SMA10 = SMA20 = 110 -> 0; SMA40 105, SMA80 102.5, SMA160 101.25 -> +1 each
        (_trend_closes(110.0, 100.0), 0.6),
        (_trend_closes(90.0, 100.0), -0.6),
        (_trend_closes(100.0, 100.0), 0.0),
        # C[B0] 50 below every SMA (the other 159 closes are 100)
        (_trend_closes(50.0, 100.0, cut=1), -1.0),
        # rising path: today above every trailing mean
        ({_TREND_B0 - j * _DAY_NS: 300.0 - j for j in range(160)}, 1.0),
    ],
)
def test_trend_signal_value(closes: dict[int, float], expected: float) -> None:
    """R18: fsum of the five sign terms / 5, exact."""
    assert trend_signal(closes, _TREND_D) == expected


def test_trend_signal_reads_only_b0_and_back() -> None:
    """A close after B0 or before B0 - 159 d never moves the signal."""
    closes = _trend_closes(110.0, 100.0)
    closes[_TREND_B0 + _DAY_NS] = 1.0
    closes[_TREND_B0 - 160 * _DAY_NS] = 1e9
    assert trend_signal(closes, _TREND_D) == 0.6
    assert trend_signal(closes, _TREND_B0 + 23 * 3_600 * _NANOS_PER_S) == 0.6


@pytest.mark.parametrize("bad", [None, 0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize("back", [0, 159])
def test_trend_signal_missing_or_bad_close(bad: float | None, back: int) -> None:
    """R19: any of the 160 closes missing, <= 0 or non-finite -> None."""
    closes = _trend_closes(110.0, 100.0)
    if bad is None:
        del closes[_TREND_B0 - back * _DAY_NS]
    else:
        closes[_TREND_B0 - back * _DAY_NS] = bad
    assert trend_signal(closes, _TREND_D) is None


def test_trend_signal_huge_closes_do_not_overflow() -> None:
    """160 closes near float max: the SMA sum falls back, no OverflowError."""
    closes = _trend_closes(1.7e308, 1.6e308, cut=1)
    assert trend_signal(closes, _TREND_D) == 1.0


def _trend_spec(**over: object) -> CarrySpec:
    kw: dict[str, object] = {
        "signal": "trend",
        "schedule": "daily",
        "weighting": "time_series",
        "direction": "long_flat",
        "vol_target": 0.4,
    }
    kw.update(over)
    return CarrySpec.from_dates(
        ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"), "2024-01-01", "2024-06-01", **kw
    )


def test_trend_decision_signals_precedence() -> None:
    """R19/R17: not_listed > no_history > no_price; prices keep every ticked symbol."""
    full = _trend_closes(110.0, 100.0)
    short = dict(full)
    del short[_TREND_B0 - 159 * _DAY_NS]
    tick = PriceTick(_TREND_D - 1_000_000, 123.0)
    result = decision_signals(
        _trend_spec(),
        {},
        {"BTCUSDT": None, "ETHUSDT": None, "SOLUSDT": None, "XRPUSDT": tick},
        _TREND_D,
        trend_closes={"BTCUSDT": full, "ETHUSDT": short, "SOLUSDT": full, "XRPUSDT": full},
        listings={"BTCUSDT": _TREND_D, "ETHUSDT": 0, "SOLUSDT": _TREND_D - 1},
    )
    assert result.excluded == {
        "BTCUSDT": "not_listed",  # D == listing (D <= listing)
        "ETHUSDT": "no_history",
        "SOLUSDT": "no_price",  # listed 1 ns before D
    }
    assert result.signals == {"BTCUSDT": None, "ETHUSDT": None, "SOLUSDT": None, "XRPUSDT": 0.6}
    assert result.prices == {"XRPUSDT": 123.0}
    assert result.vols is None
    assert result.regime is None


def test_trend_decision_signals_regime() -> None:
    """The filter symbol's raw signal is the regime, even when it has no sizing price."""
    tick = PriceTick(_TREND_D - 1_000_000, 50.0)
    closes = {
        "BTCUSDT": _trend_closes(90.0, 100.0),
        "ETHUSDT": _trend_closes(110.0, 100.0),
        "SOLUSDT": _trend_closes(100.0, 100.0),
        "XRPUSDT": _trend_closes(50.0, 100.0, cut=1),
    }
    result = decision_signals(
        _trend_spec(trend_filter="ETHUSDT"),
        {},
        {"BTCUSDT": tick, "ETHUSDT": None, "SOLUSDT": tick, "XRPUSDT": tick},
        _TREND_D,
        trend_closes=closes,
    )
    assert result.regime == 0.6
    assert result.excluded == {"ETHUSDT": "no_price"}
    assert result.signals == {"BTCUSDT": -0.6, "ETHUSDT": None, "SOLUSDT": 0.0, "XRPUSDT": -1.0}
    with pytest.raises(ValueError, match="trend_closes"):
        decision_signals(_trend_spec(), {}, {}, _TREND_D)


# --- clmmc.25: trend_signal is asked once per (coin, day) however many runs read it -----------


def test_trend_signal_is_computed_once_per_coin_and_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two runs over the same load (a null draw and the own book) read the same signals; the
    second asks nothing again, a later hour of the day included, and a coin whose map gained
    a day is read afresh."""
    import vibe_quant.portfolio.signal as signal_module

    asked: list[int] = []
    real = signal_module.trend_signal

    def counting(closes: dict[int, float], decision_ns: int) -> float | None:
        asked.append(id(closes))
        return real(closes, decision_ns)

    monkeypatch.setattr(signal_module, "trend_signal", counting)
    tick = PriceTick(_TREND_D - 1_000_000, 50.0)
    closes = {
        "BTCUSDT": _trend_closes(90.0, 100.0),
        "ETHUSDT": _trend_closes(110.0, 100.0),
        "SOLUSDT": _trend_closes(100.0, 100.0),
        "XRPUSDT": _trend_closes(50.0, 100.0, cut=1),
    }
    sizing = dict.fromkeys(closes, tick)

    def signals(decision_ns: int) -> dict[str, float | None]:
        return decision_signals(_trend_spec(), {}, sizing, decision_ns, trend_closes=closes).signals

    first = signals(_TREND_D)
    assert first == {"BTCUSDT": -0.6, "ETHUSDT": 0.6, "SOLUSDT": 0.0, "XRPUSDT": -1.0}
    assert len(asked) == 4
    assert signals(_TREND_D) == first
    assert signals(_TREND_B0 + 23 * 3_600 * _NANOS_PER_S) == first
    assert len(asked) == 4
    closes["BTCUSDT"][_TREND_B0 + _DAY_NS] = 1.0  # an appended day: the entry is dropped
    assert signals(_TREND_D)["BTCUSDT"] == -0.6
    assert len(asked) == 5


def test_trend_signal_memo_does_not_serve_another_load() -> None:
    """A different close map for the same coin and day (the lagged load) gets its own signal."""
    tick = PriceTick(_TREND_D - 1_000_000, 50.0)
    sizing = {"BTCUSDT": tick, "ETHUSDT": tick, "SOLUSDT": tick, "XRPUSDT": tick}

    def btc_signal(path: dict[int, float]) -> float | None:
        closes = {symbol: path for symbol in sizing}
        return decision_signals(_trend_spec(), {}, sizing, _TREND_D, trend_closes=closes).signals[
            "BTCUSDT"
        ]

    assert btc_signal(_trend_closes(110.0, 100.0)) == 0.6
    assert btc_signal(_trend_closes(90.0, 100.0)) == -0.6
    assert btc_signal({}) is None
