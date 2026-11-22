"""Decision-time 1m ticks, run inputs and the fail-closed preflight (task T2b).

Contract rows R15b (release staleness), R20b (ticks at/after ``end_ns``), R28
(preflight order) and R32 (listing mid-window) of the funding-carry semantics
contract (``data/swarm/20261015-funding-carry/design.md`` §1.3, §7, §8-T2b),
plus the design's ``select_ticks`` and ``boundary_prices`` tests.

Every window edge is pinned with a bar exactly on it -- opening ``D-60 min``,
``D-1 min``, ``D``, ``D+59 min``, a 1m bar at exactly ``B-60 min``/``B``, a
tick exactly at ``end_ns`` and a release exactly ``PRICE_WINDOW_NS`` after the
queue -- so flipping any ``<``/``<=`` fails a test. The project fill contract
stamps a 1m catalog bar ``ts_init = open + 59.999 s``.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import AggressorSide, PriceType
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

import vibe_quant.portfolio.__main__ as carry_cli
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import create_instrument, get_bar_type
from vibe_quant.data.fill_ticks import FILL_TICK_SIZE
from vibe_quant.data.mark_bars import MarkDataMissing
from vibe_quant.metrics import BarSeries
from vibe_quant.portfolio.signal import Settlements
from vibe_quant.portfolio.spec import (
    PRICE_WINDOW_NS,
    CarryDataError,
    CarryExecutionError,
    CarrySpec,
    decision_times,
)
from vibe_quant.portfolio.ticks import (
    PriceTick,
    boundary_prices,
    check_release,
    load_carry_inputs,
    release_tick,
    select_ticks,
    sizing_price,
    to_trade_ticks,
    trend_closes,
)
from vibe_quant.validation.funding import FundingCalculator

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

MIN = 60_000_000_000
HOUR = 60 * MIN
DAY = 86_400_000_000_000
_WEEK = 7 * DAY
INIT = 59_999_000_000  # 1m bar ts_init = open + 59.999 s


def _ns(date: str) -> int:
    """``YYYY-MM-DD`` 00:00 UTC -> epoch ns."""
    return int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp()) * 10**9


START = _ns("2024-06-03")  # a Monday 00:00 UTC
END = _ns("2024-07-01")
D = START + 5 * MIN  # the first Monday 00:05 decision


def _series(opens: Sequence[int], closes: Sequence[float]) -> BarSeries:
    """1m LAST bars at the given opens (ts_init = open + 59.999 s)."""
    return _series_at([open_ns + INIT for open_ns in opens], closes)


def _series_at(ts: Sequence[int], closes: Sequence[float]) -> BarSeries:
    values = np.array(closes, dtype=np.float64)
    return BarSeries(
        ts=np.array(ts, dtype=np.int64),
        open=values,
        high=values,
        low=values,
        close=values,
    )


# ---------------------------------------------------------------------------
# select_ticks / sizing_price / boundary_prices (design T2b)
# ---------------------------------------------------------------------------


def test_select_ticks_sizing_and_fill_bars() -> None:
    """The bar opening ``D-1 min`` is the sizing tick, the one opening ``D`` the fill."""
    bars = _series(
        [D - 60 * MIN, D - MIN, D, D + 59 * MIN, D + 60 * MIN, D + 61 * MIN],
        [51000.0, 51010.0, 51020.0, 51030.0, 51040.0, 51050.0],
    )
    assert select_ticks(bars, [D], create_instrument("BTCUSDT"), D + DAY) == (
        PriceTick(D - 1_000_000, 51010.0),
        PriceTick(D + INIT, 51020.0),
    )


def test_select_ticks_multi_decision_is_ascending() -> None:
    """Two weeks of ticks come back in time order, sizing before its own fill."""
    bars = _series([D - MIN, D, D + _WEEK - MIN, D + _WEEK], [1.0, 2.0, 3.0, 4.0])
    ticks = select_ticks(bars, [D, D + _WEEK], create_instrument("BTCUSDT"), D + 2 * _WEEK)
    assert [tick.ts for tick in ticks] == [
        D - 1_000_000,
        D + INIT,
        D + _WEEK - 1_000_000,
        D + _WEEK + INIT,
    ]


def test_select_ticks_sizing_window_edges() -> None:
    """A bar opening exactly ``D-60 min`` still sizes; ``D-61 min`` does not."""
    instrument = create_instrument("BTCUSDT")
    edges = _series([D - 60 * MIN, D], [1.0, 2.0])
    assert select_ticks(edges, [D], instrument, D + DAY) == (
        PriceTick(D - 60 * MIN + INIT, 1.0),
        PriceTick(D + INIT, 2.0),
    )
    outside = _series([D - 61 * MIN, D], [1.0, 2.0])
    assert select_ticks(outside, [D], instrument, D + DAY) == (PriceTick(D + INIT, 2.0),)
    last = _series([D - 60 * MIN, D - 30 * MIN, D - MIN], [1.0, 2.0, 3.0])
    assert select_ticks(last, [D], instrument, D + 2 * DAY)[0] == PriceTick(D - 1_000_000, 3.0)


def test_select_ticks_fill_window_edges_and_holes() -> None:
    """A bar opening exactly ``D+59 min`` fills; ``D+60 min`` is past the window."""
    instrument = create_instrument("BTCUSDT")
    upper_edge = _series([D - MIN, D + 59 * MIN, D + 60 * MIN], [1.0, 2.0, 3.0])
    assert select_ticks(upper_edge, [D], instrument, D + DAY) == (
        PriceTick(D - 1_000_000, 1.0),
        PriceTick(D + 59 * MIN + INIT, 2.0),
    )
    after_window = _series([D - MIN, D + 60 * MIN, D + 61 * MIN], [1.0, 2.0, 3.0])
    assert select_ticks(after_window, [D], instrument, D + DAY) == (PriceTick(D - 1_000_000, 1.0),)
    hole = _series([D - MIN, D + 5 * MIN, D + 6 * MIN], [1.0, 2.0, 3.0])
    assert select_ticks(hole, [D], instrument, D + DAY) == (
        PriceTick(D - 1_000_000, 1.0),
        PriceTick(D + 5 * MIN + INIT, 2.0),
    )


def test_ticks_before_end_only() -> None:
    """A tick at or after ``end_ns`` is dropped, one ns before it is kept."""
    instrument = create_instrument("BTCUSDT")
    bars = _series([D - MIN, D], [1.0, 2.0])
    fill_ts = D + INIT
    assert select_ticks(bars, [D], instrument, fill_ts + 1) == (
        PriceTick(D - 1_000_000, 1.0),
        PriceTick(fill_ts, 2.0),
    )
    assert select_ticks(bars, [D], instrument, fill_ts) == (PriceTick(D - 1_000_000, 1.0),)
    assert select_ticks(bars, [D], instrument, D - 1_000_000) == ()


def test_sizing_price_window_edges() -> None:
    """The last tick in ``[D-60 min, D)``: both edges and the empty case."""
    ticks = (
        PriceTick(D - 60 * MIN, 1.0),
        PriceTick(D - 30 * MIN, 2.0),
        PriceTick(D - 1_000_000, 3.0),
    )
    assert sizing_price(ticks, D) == PriceTick(D - 1_000_000, 3.0)
    assert sizing_price(ticks, D + 30 * MIN) == PriceTick(D - 1_000_000, 3.0)
    assert sizing_price((PriceTick(D - 60 * MIN, 1.0),), D) == PriceTick(D - 60 * MIN, 1.0)
    assert sizing_price((PriceTick(D - 60 * MIN - 1, 1.0),), D) is None
    assert sizing_price((PriceTick(D, 1.0),), D) is None
    assert sizing_price((), D) is None


def test_boundary_prices_window() -> None:
    """§4: close of the 1m bar with ``ts_init`` in ``(B-60 min, B]``; absent = no price."""
    instrument = create_instrument("BTCUSDT")
    boundary = D
    assert boundary_prices(
        _series_at(
            [boundary - 60 * MIN, boundary - 60 * MIN + 1, boundary - 1, boundary, boundary + 1],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ),
        [boundary],
        instrument,
    ) == {boundary: 4.0}
    # exactly at the lower edge -> out; 1 ns later -> in
    assert boundary_prices(_series_at([boundary - 60 * MIN], [1.0]), [boundary], instrument) == {}
    assert boundary_prices(_series_at([boundary - 60 * MIN + 1], [1.0]), [boundary], instrument) == {
        boundary: 1.0
    }
    # the last bar at or before B wins; a bar after B is not eligible (but is for the next B)
    two = [boundary, boundary + 1]
    assert boundary_prices(_series_at(two, [7.0, 9.0]), [boundary], instrument) == {boundary: 7.0}
    assert boundary_prices(_series_at(two, [7.0, 9.0]), [boundary + 1], instrument) == {
        boundary + 1: 9.0
    }
    # no bar in a boundary's window -> absent from the mapping
    one = _series_at([boundary - 1], [5.0])
    assert boundary_prices(one, [boundary, boundary + DAY], instrument) == {boundary: 5.0}


def test_boundary_prices_uses_the_instrument_tick() -> None:
    """Boundary closes are the instrument's tick-rounded price, like ``PriceTick.price``."""
    instrument = create_instrument("BTCUSDT")
    assert boundary_prices(_series_at([D - 1, D], [51000.04, 51000.09]), [D], instrument) == {
        D: 51000.1
    }
    assert boundary_prices(_series_at([D - 1], [51000.04]), [D], instrument) == {D: 51000.0}


# ---------------------------------------------------------------------------
# to_trade_ticks
# ---------------------------------------------------------------------------


def test_to_trade_ticks_identity_price_and_size() -> None:
    """TradeId ``C{ts}``; ts_event = ts_init = the tick's ts; size = FILL_TICK_SIZE."""
    instrument = create_instrument("BTCUSDT")
    ticks = (PriceTick(D, 51000.0), PriceTick(D + MIN, 51010.0))
    out = to_trade_ticks(instrument, ticks)
    assert [tick.trade_id.value for tick in out] == [f"C{D}", f"C{D + MIN}"]
    assert [tick.ts_event for tick in out] == [D, D + MIN]
    assert [tick.ts_init for tick in out] == [D, D + MIN]
    assert [float(tick.price) for tick in out] == [51000.0, 51010.0]
    assert all(float(tick.size) == FILL_TICK_SIZE for tick in out)
    assert all(tick.instrument_id == instrument.id for tick in out)
    assert all(tick.aggressor_side == AggressorSide.NO_AGGRESSOR for tick in out)
    assert [float(tick.size) for tick in to_trade_ticks(instrument, ticks[:1], size=0.001)] == [0.001]


# ---------------------------------------------------------------------------
# Release staleness
# ---------------------------------------------------------------------------


def test_check_release_limit() -> None:
    """``release_ns - queue_ns > PRICE_WINDOW_NS`` is stale; the equality is not."""
    check_release("BTCUSDT", D, D + PRICE_WINDOW_NS)
    with pytest.raises(CarryExecutionError, match="stale release BTCUSDT"):
        check_release("BTCUSDT", D, D + PRICE_WINDOW_NS + 1)


def test_release_tick_is_the_first_tick_after_the_queue_ts() -> None:
    """Strict ``>`` on the queue ts; ``None`` = no later own tick."""
    ticks = (PriceTick(D - 1, 1.0), PriceTick(D, 2.0), PriceTick(D + MIN, 3.0))
    assert release_tick("BTCUSDT", ticks, D) == PriceTick(D + MIN, 3.0)
    assert release_tick("BTCUSDT", ticks, D + MIN) is None
    assert release_tick("BTCUSDT", (), D) is None


def test_release_tick_applies_check_release_to_the_chosen_tick() -> None:
    """The first later tick past the window raises; an earlier in-window tick wins."""
    window = PRICE_WINDOW_NS
    assert release_tick("BTCUSDT", (PriceTick(D + window, 1.0),), D) == PriceTick(D + window, 1.0)
    with pytest.raises(CarryExecutionError, match="stale release BTCUSDT"):
        release_tick("BTCUSDT", (PriceTick(D + window + 1, 1.0),), D)
    both = (PriceTick(D + 2, 1.0), PriceTick(D + window + 1, 2.0))
    assert release_tick("BTCUSDT", both, D) == PriceTick(D + 2, 1.0)


# ---------------------------------------------------------------------------
# Catalog fixtures for load_carry_inputs
# ---------------------------------------------------------------------------


def _spec(symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")) -> CarrySpec:
    return CarrySpec.from_dates(symbols, "2024-06-03", "2024-07-01")


def _write_instrument(root: Path, symbol: str) -> None:
    ParquetDataCatalog(str(root)).write_data([create_instrument(symbol)])


def _write_minute_bars(root: Path, symbol: str, prices: Mapping[int, float]) -> None:
    """1m LAST bars at the given opens, one price each (ts_init = open + 59.999 s)."""
    bar_type = get_bar_type(symbol, "1m")
    precision = create_instrument(symbol).price_precision
    bars = [
        Bar(
            bar_type,
            Price(price, precision),
            Price(price, precision),
            Price(price, precision),
            Price(price, precision),
            Quantity(1, 0),
            open_ns,
            open_ns + INIT,
        )
        for open_ns, price in prices.items()
    ]
    ParquetDataCatalog(str(root)).write_data(bars)


def _write_mark_bars(root: Path, symbol: str, opens: Sequence[int]) -> None:
    """MARK 1m bars at the given openings (``ts_event`` = the bar's open)."""
    bar_type = get_bar_type(symbol, "1m", price_type=PriceType.MARK)
    price = Price(100.0, create_instrument(symbol).price_precision)
    bars = [
        Bar(bar_type, price, price, price, price, Quantity(1, 0), open_ns, open_ns + MIN)
        for open_ns in opens
    ]
    ParquetDataCatalog(str(root)).write_data(bars)


def _mark_opens(start_ns: int, end_ns: int) -> list[int]:
    """MARK openings every 30 min from ``start_ns`` (inside the 60 min coverage gap)."""
    return [*range(start_ns, end_ns, 30 * MIN), end_ns - MIN]


def _minute_bars_for_run(start_ns: int, end_ns: int) -> dict[int, float]:
    """One 1m bar per minute a tick or a daily boundary close needs."""
    opens = {start_ns - MIN}
    for decision_ns in decision_times(start_ns, end_ns):
        opens.add(decision_ns - MIN)
        opens.add(decision_ns)
    for boundary_ns in range(start_ns, end_ns + DAY, DAY):
        opens.add(boundary_ns - MIN)
    return {open_ns: 100.0 + index for index, open_ns in enumerate(sorted(opens))}


def _expected_ticks(decisions: Sequence[int], prices: Mapping[int, float]) -> tuple[PriceTick, ...]:
    return tuple(
        tick
        for decision_ns in decisions
        for tick in (
            PriceTick(decision_ns - MIN + INIT, prices[decision_ns - MIN]),
            PriceTick(decision_ns + INIT, prices[decision_ns]),
        )
    )


def _expected_boundaries(
    start_ns: int, end_ns: int, prices: Mapping[int, float]
) -> dict[int, float]:
    return {boundary: prices[boundary - MIN] for boundary in range(start_ns, end_ns + DAY, DAY)}


def _settlements_from(first_ns: int, end_ns: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    times = tuple(range(first_ns, end_ns, 8 * HOUR))
    return times, tuple([0.0001] * len(times))


def _stub_settlements(
    monkeypatch: pytest.MonkeyPatch,
    by_symbol: Mapping[str, tuple[tuple[int, ...], tuple[float, ...]]],
    hours: tuple[int, ...] | None = None,
) -> None:
    """Stub ``FundingCalculator.settlements`` with fixed per-symbol series.

    ``hours`` also stubs ``settlement_hours`` (all symbols), which the real
    method would answer from the (empty) archive as ``()`` -> ``hours=None``.
    """

    def settlements(
        self: FundingCalculator, symbol: str
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        return by_symbol.get(symbol, ((), ()))

    monkeypatch.setattr(FundingCalculator, "settlements", settlements)
    if hours is not None:

        def settlement_hours(self: FundingCalculator, symbol: str) -> tuple[int, ...]:
            return hours

        monkeypatch.setattr(FundingCalculator, "settlement_hours", settlement_hours)


def _valid_catalog(root: Path, prices: Mapping[int, float], mark_from: int) -> None:
    """Both spec symbols: instrument + 1m bars + MARK from ``mark_from``."""
    for symbol in ("BTCUSDT", "ETHUSDT"):
        _write_instrument(root, symbol)
        _write_minute_bars(root, symbol, prices)
        _write_mark_bars(root, symbol, _mark_opens(mark_from, END))


def _archive(tmp_path: Path) -> Path:
    path = tmp_path / "raw_data.db"
    path.write_bytes(b"")
    return path


# ---------------------------------------------------------------------------
# load_carry_inputs: assembly, R28 preflight, R32 listing
# ---------------------------------------------------------------------------


def test_load_carry_inputs_assembles_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loaded inputs: schedule, per-symbol ticks, settlements, marks, boundaries."""
    spec = _spec()
    prices = _minute_bars_for_run(START, END)
    rates = _settlements_from(START, END)
    _stub_settlements(monkeypatch, dict.fromkeys(spec.symbols, rates))
    root = tmp_path / "catalog"
    _valid_catalog(root, prices, START)

    inputs = load_carry_inputs(spec, root, _archive(tmp_path))

    decisions = tuple(decision_times(START, END))
    assert inputs.spec is spec
    assert decisions == (D, D + _WEEK, D + 2 * _WEEK, D + 3 * _WEEK)
    assert inputs.decisions == decisions
    assert sorted(inputs.instruments) == ["BTCUSDT", "ETHUSDT"]
    assert inputs.specs["BTCUSDT"].step == Decimal("0.001")
    assert inputs.specs["BTCUSDT"].min_notional == 50.0
    assert inputs.ticks["BTCUSDT"] == _expected_ticks(decisions, prices)
    assert inputs.ticks["ETHUSDT"] == _expected_ticks(decisions, prices)
    assert inputs.boundary_prices["BTCUSDT"] == _expected_boundaries(START, END, prices)
    assert inputs.settlements["BTCUSDT"] == Settlements(times_ns=rates[0], rates=rates[1])
    assert isinstance(inputs.calc, FundingCalculator)
    assert inputs.marks["BTCUSDT"].ts[0] == START


def test_load_carry_inputs_vol_closes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """inverse_vol loads daily closes from ``start - L days`` (boundary rule); equal loads none."""
    older = {START - back * DAY - MIN: 90.0 + back for back in (3, 2, 1)}
    prices = {**older, **_minute_bars_for_run(START, END)}  # ascending opens
    rates = _settlements_from(START, END)
    root = tmp_path / "catalog"
    _valid_catalog(root, prices, START)
    _stub_settlements(monkeypatch, {"BTCUSDT": rates, "ETHUSDT": rates})

    equal = load_carry_inputs(_spec(), root, _archive(tmp_path))
    spec = CarrySpec.from_dates(
        ("BTCUSDT", "ETHUSDT"), "2024-06-03", "2024-07-01",
        weighting="inverse_vol", vol_lookback_days=3,
    )
    varied = load_carry_inputs(spec, root, _archive(tmp_path))

    assert equal.vol_closes is None
    assert varied.vol_closes is not None
    expected = {
        START - 3 * DAY: 93.0,
        START - 2 * DAY: 92.0,
        START - DAY: 91.0,
        **_expected_boundaries(START, END, prices),
    }
    assert varied.vol_closes["BTCUSDT"] == expected
    assert varied.vol_closes["ETHUSDT"] == expected
    assert varied.boundary_prices == equal.boundary_prices  # metric marks unchanged


def test_load_carry_inputs_carries_non_8h_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-8h ``settlement_hours`` of matching length reaches ``Settlements``."""
    spec = _spec()
    prices = _minute_bars_for_run(START, END)
    rates = _settlements_from(START, END)
    hours = (2,) * len(rates[0])
    _stub_settlements(monkeypatch, dict.fromkeys(spec.symbols, rates), hours=hours)
    root = tmp_path / "catalog"
    _valid_catalog(root, prices, START)

    inputs = load_carry_inputs(spec, root, _archive(tmp_path))

    assert inputs.settlements["BTCUSDT"] == Settlements(
        times_ns=rates[0], rates=rates[1], hours=hours
    )
    assert inputs.settlements["BTCUSDT"].hours == hours


def test_preflight_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``ValueError`` for a bad spec first, then ``CarryDataError`` + symbol/reason."""
    spec = _spec()
    prices = _minute_bars_for_run(START, END)
    rates = _settlements_from(START, END)
    archive = _archive(tmp_path)
    _stub_settlements(monkeypatch, dict.fromkeys(spec.symbols, rates))

    # 1. bad spec -> ValueError, before any path or symbol is touched
    with pytest.raises(ValueError, match="CarrySpec"):
        load_carry_inputs({"symbols": ("BTCUSDT", "ETHUSDT")}, tmp_path / "nope", tmp_path / "no.db")

    # 2. missing archive -> CarryDataError naming the archive (before any symbol)
    root2 = tmp_path / "c2"
    root2.mkdir()
    with pytest.raises(CarryDataError, match=r"funding archive .*raw_data.db"):
        load_carry_inputs(spec, root2, tmp_path / "no-archive" / "raw_data.db")

    # 3. symbol not in the catalog -> named, even though its 1m bars are present
    root3 = tmp_path / "c3"
    _write_minute_bars(root3, "BTCUSDT", prices)
    with pytest.raises(CarryDataError, match=r"BTCUSDT: no instrument in catalog"):
        load_carry_inputs(spec, root3, archive)

    # 4. no 1m bars -> named, with the instrument present
    root4 = tmp_path / "c4"
    _write_instrument(root4, "BTCUSDT")
    with pytest.raises(CarryDataError, match=r"BTCUSDT: no 1m bars"):
        load_carry_inputs(spec, root4, archive)

    # 5. no settlement in [start - 7d, end) -> named; one before the lookback is not enough
    root5 = tmp_path / "c5"
    _write_instrument(root5, "BTCUSDT")
    _write_minute_bars(root5, "BTCUSDT", prices)
    _write_mark_bars(root5, "BTCUSDT", _mark_opens(START, END))
    _stub_settlements(monkeypatch, {"BTCUSDT": ((), ()), "ETHUSDT": rates})
    with pytest.raises(CarryDataError, match=r"BTCUSDT: no funding settlement"):
        load_carry_inputs(spec, root5, archive)
    _stub_settlements(monkeypatch, {"BTCUSDT": ((START - 8 * DAY,), (0.0001,)), "ETHUSDT": rates})
    with pytest.raises(CarryDataError, match=r"BTCUSDT: no funding settlement"):
        load_carry_inputs(spec, root5, archive)
    # ... and `end` itself is outside the half-open window `[start - 7d, end)`
    _stub_settlements(monkeypatch, {"BTCUSDT": ((END,), (0.0001,)), "ETHUSDT": rates})
    with pytest.raises(CarryDataError, match=r"BTCUSDT: no funding settlement"):
        load_carry_inputs(spec, root5, archive)

    # 6. MARK gap / absent MARK -> the wrapped MarkDataMissing reason, naming the symbol
    _stub_settlements(monkeypatch, dict.fromkeys(spec.symbols, rates))
    root6 = tmp_path / "c6"
    _write_instrument(root6, "BTCUSDT")
    _write_minute_bars(root6, "BTCUSDT", prices)
    _write_mark_bars(root6, "BTCUSDT", _mark_opens(START + DAY, END))
    with pytest.raises(CarryDataError, match=r"BTCUSDT: MARK coverage starts") as starts:
        load_carry_inputs(spec, root6, archive)
    assert isinstance(starts.value.__cause__, MarkDataMissing)
    assert str(starts.value) == str(starts.value.__cause__)
    root7 = tmp_path / "c7"
    _write_instrument(root7, "BTCUSDT")
    _write_minute_bars(root7, "BTCUSDT", prices)
    with pytest.raises(CarryDataError, match=r"BTCUSDT: no MARK price bars") as absent:
        load_carry_inputs(spec, root7, archive)
    assert isinstance(absent.value.__cause__, MarkDataMissing)
    assert str(absent.value) == str(absent.value.__cause__)

    # 7. a fully valid pair loads; nothing raised before any engine
    root8 = tmp_path / "c8"
    _valid_catalog(root8, prices, START)
    assert load_carry_inputs(spec, root8, archive).decisions == tuple(decision_times(START, END))


def test_preflight_accepts_settlement_exactly_at_lookback_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A settlement exactly at ``start - 7d`` is inside ``[start-7d, end)``."""
    spec = _spec()
    prices = _minute_bars_for_run(START, END)
    edge = ((START - _WEEK,), (0.0001,))
    _stub_settlements(monkeypatch, dict.fromkeys(spec.symbols, edge))
    root = tmp_path / "catalog"
    _valid_catalog(root, prices, START)

    inputs = load_carry_inputs(spec, root, _archive(tmp_path))

    assert inputs.settlements["BTCUSDT"].times_ns == (START - _WEEK,)
    assert inputs.marks["BTCUSDT"].ts[0] == START


def test_preflight_fails_closed_when_mark_ends_before_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MARK must reach ``end``; coverage ending a day early fails closed."""
    spec = _spec()
    prices = _minute_bars_for_run(START, END)
    rates = _settlements_from(START, END)
    _stub_settlements(monkeypatch, dict.fromkeys(spec.symbols, rates))
    root = tmp_path / "catalog"
    for symbol in spec.symbols:
        _write_instrument(root, symbol)
        _write_minute_bars(root, symbol, prices)
        _write_mark_bars(root, symbol, _mark_opens(START, END - DAY))

    with pytest.raises(CarryDataError, match=r"BTCUSDT: MARK coverage ends") as ends:
        load_carry_inputs(spec, root, _archive(tmp_path))
    assert isinstance(ends.value.__cause__, MarkDataMissing)
    assert str(ends.value) == str(ends.value.__cause__)


def test_listing_mid_window_mark_from_first_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MARK is required only from a mid-window listing's first settlement."""
    listing = _ns("2024-06-17") + 8 * HOUR
    assert START < listing < END
    spec = _spec()
    prices = _minute_bars_for_run(START, END)
    listed = _settlements_from(listing, END)
    _stub_settlements(monkeypatch, dict.fromkeys(spec.symbols, listed))
    archive = _archive(tmp_path)

    root = tmp_path / "catalog"
    _valid_catalog(root, prices, listing)
    inputs = load_carry_inputs(spec, root, archive)
    assert inputs.settlements["BTCUSDT"].times_ns[0] == listing
    assert inputs.marks["BTCUSDT"].ts[0] == listing

    late = tmp_path / "late"
    for symbol in spec.symbols:
        _write_instrument(late, symbol)
        _write_minute_bars(late, symbol, prices)
        _write_mark_bars(late, symbol, _mark_opens(listing + MIN, END))
    with pytest.raises(CarryDataError, match=r"BTCUSDT: MARK coverage starts"):
        load_carry_inputs(spec, late, archive)


def test_load_carry_inputs_on_real_data() -> None:
    """Smoke on the real catalog/archive: BTCUSDT+ETHUSDT over three weeks."""
    repo_root = Path(__file__).resolve().parents[2]
    catalog = repo_root / "data" / "catalog"
    archive = repo_root / "data" / "archive" / "raw_data.db"
    if not (catalog / "data" / "bar").is_dir() or not archive.is_file():
        pytest.skip("real catalog/archive not available")
    spec = CarrySpec.from_dates(("BTCUSDT", "ETHUSDT"), "2025-05-05", "2025-05-26")
    inputs = load_carry_inputs(spec, catalog, archive)  # real FundingCalculator.settlements
    assert len(inputs.decisions) == 3
    for symbol in spec.symbols:
        ticks = inputs.ticks[symbol]
        assert len(ticks) == 6  # two per decision
        assert all(tick.ts % MIN == INIT for tick in ticks)  # 1m bar ts_init
        assert sum(1 for tick in ticks if tick.ts > inputs.decisions[-1]) == 1  # last fill tick
        assert inputs.marks[symbol] is not None
        prices = inputs.boundary_prices[symbol]
        assert len(prices) == 22  # 2025-05-05 .. 2025-05-26 daily boundaries
        assert all(price > 0.0 for price in prices.values())


# ---------------------------------------------------------------------------
# Liquidity filter inputs (x0axf.14 T5: R30-R33)
# ---------------------------------------------------------------------------

_QV = 1.0e9
_DAY_MS = DAY // 10**6
_VOLUME_DAYS = range(START - 7 * DAY, END, DAY)


def _liquid_spec() -> CarrySpec:
    return CarrySpec.from_dates(
        ("BTCUSDT", "ETHUSDT"), "2024-06-03", "2024-07-01", min_quote_volume=1.0e8
    )


def _volume_archive(
    tmp_path: Path,
    extra: Sequence[tuple[str, str, int, object]] = (),
    *,
    skip_eth_from: int | None = None,
) -> Path:
    """A tmp archive: one 1m row (quote volume 1e9) per UTC day of ``[start - 7 d, end)``.

    ``extra`` adds ``(symbol, interval, open_time_ms, quote_volume)`` rows;
    ``skip_eth_from`` drops ETHUSDT's 1m rows from that day (ns) on.
    """
    path = tmp_path / "raw_data.db"
    archive = RawDataArchive(path)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        days = [
            day // 10**6
            for day in _VOLUME_DAYS
            if not (symbol == "ETHUSDT" and skip_eth_from is not None and day >= skip_eth_from)
        ]
        archive.insert_klines(
            symbol, "1m", [(t, 1.0, 1.0, 1.0, 1.0, 1.0, t + 59_999, _QV) for t in days], "test"
        )
    for symbol, interval, open_ms, volume in extra:
        archive.insert_klines(
            symbol,
            interval,
            [(open_ms, 1.0, 1.0, 1.0, 1.0, 1.0, open_ms + 59_999, volume)],
            "test",
        )
    archive.close()
    return path


def _volume_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A valid catalog with stubbed valid funding (funding preflight always passes)."""
    rates = _settlements_from(START, END)
    _stub_settlements(monkeypatch, {"BTCUSDT": rates, "ETHUSDT": rates})
    root = tmp_path / "catalog"
    _valid_catalog(root, _minute_bars_for_run(START, END), START)
    return root


def _cli_run(root: Path, archive: Path, tmp_path: Path) -> int:
    return carry_cli.main(
        [
            "run", "--symbols", "BTCUSDT", "ETHUSDT", "--start", "2024-06-03",
            "--end", "2024-07-01", "--engine", "reference", "--min-quote-volume", "1e8",
            "--catalog", str(root), "--archive", str(archive),
            "--out", str(tmp_path / "out"), "--trials", str(tmp_path / "trials.jsonl"),
        ]
    )


def test_volume_loaded_with_1m_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Daily sums keyed by UTC day ns; a same-minute 1d row adds neither volume nor coverage."""
    root = _volume_catalog(tmp_path, monkeypatch)
    same_minute = START // 10**6 + 3 * _DAY_MS
    archive = _volume_archive(
        tmp_path,
        [
            ("BTCUSDT", "1d", same_minute, 1.0e12),
            ("BTCUSDT", "1m", same_minute + 60_000, 2.5e8),
            ("BTCUSDT", "1d", END // 10**6 + 5 * _DAY_MS, 1.0e12),  # a day with no 1m row
        ],
    )

    inputs = load_carry_inputs(_liquid_spec(), root, archive)

    assert inputs.quote_volume is not None
    expected = dict.fromkeys(_VOLUME_DAYS, _QV)
    assert inputs.quote_volume["ETHUSDT"] == expected
    assert inputs.quote_volume["BTCUSDT"] == {**expected, START + 3 * DAY: _QV + 2.5e8}


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(60_000, None)], r"BTCUSDT: 1 malformed 1m quote_volume row\(s\) on 2024-06-06 UTC"),
        ([(60_000, "abc")], r"BTCUSDT: 1 malformed 1m quote_volume row\(s\) on 2024-06-06 UTC"),
        (
            [(60_000, -1.0), (120_000, 5.0e9)],
            r"BTCUSDT: 1 malformed 1m quote_volume row\(s\) on 2024-06-06 UTC",
        ),
        (
            [(60_000, 1.7e308), (120_000, 1.7e308)],
            r"BTCUSDT: non-finite 1m quote_volume sum on 2024-06-06 UTC",
        ),
    ],
)
def test_volume_malformed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rows: list[tuple[int, object]],
    message: str,
) -> None:
    """R30: a bad 1m row or a non-finite day sum -> CarryDataError (symbol + UTC day); CLI exit 2."""
    root = _volume_catalog(tmp_path, monkeypatch)
    day_ms = START // 10**6 + 3 * _DAY_MS
    archive = _volume_archive(
        tmp_path, [("BTCUSDT", "1m", day_ms + offset, volume) for offset, volume in rows]
    )

    with pytest.raises(CarryDataError, match=f"^{message}$"):
        load_carry_inputs(_liquid_spec(), root, archive)

    capsys.readouterr()
    assert _cli_run(root, archive, tmp_path) == 2
    assert "CarryDataError: BTCUSDT: " in capsys.readouterr().err
    assert not (tmp_path / "trials.jsonl").exists()
    assert not (tmp_path / "out").exists()


def test_volume_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R31: a failing quote-volume read is ``archive read failed``, never a volume_hole."""
    root = _volume_catalog(tmp_path, monkeypatch)
    archive = _volume_archive(tmp_path)
    trials = tmp_path / "trials.jsonl"
    # valid inputs load: the funding preflight passes, so only the injected read can fail
    assert load_carry_inputs(_liquid_spec(), root, archive).quote_volume is not None

    def broken(self: RawDataArchive, symbol: str, start_ms: int, end_ms: int) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(RawDataArchive, "get_daily_quote_volume", broken)
    with pytest.raises(CarryDataError, match=r"^archive read failed: BTCUSDT: disk I/O error$"):
        load_carry_inputs(_liquid_spec(), root, archive)

    trials.write_bytes(b"")
    capsys.readouterr()
    assert _cli_run(root, archive, tmp_path) == 2
    assert "archive read failed: BTCUSDT: disk I/O error" in capsys.readouterr().err
    assert trials.read_bytes() == b""
    assert not (tmp_path / "out").exists()


def test_volume_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R32: a symbol with 0 1m klines in [start, end) (1d rows only) fails at preflight."""
    root = _volume_catalog(tmp_path, monkeypatch)
    archive = _volume_archive(
        tmp_path,
        [("ETHUSDT", "1d", day // 10**6, _QV) for day in range(START, END, DAY)],
        skip_eth_from=START,  # ETH keeps only its lookback-week 1m rows
    )
    with pytest.raises(CarryDataError, match=r"^ETHUSDT: no 1m klines in archive "):
        load_carry_inputs(_liquid_spec(), root, archive)
    # one 1m row on the last day of the window is enough coverage
    edge_dir = tmp_path / "edge"
    edge_dir.mkdir()
    edge = _volume_archive(
        edge_dir,
        [("ETHUSDT", "1m", (END - DAY) // 10**6 + 60_000, _QV)],
        skip_eth_from=START,
    )
    loaded = load_carry_inputs(_liquid_spec(), root, edge)
    assert loaded.quote_volume is not None
    assert loaded.quote_volume["ETHUSDT"][END - DAY] == _QV


def test_no_volume_query_when_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R33: min_quote_volume None -> quote_volume None and no archive volume read."""
    root = _volume_catalog(tmp_path, monkeypatch)
    archive = _volume_archive(tmp_path)
    calls: list[str] = []
    real = RawDataArchive.get_daily_quote_volume

    def spy(
        self: RawDataArchive, symbol: str, start_ms: int, end_ms: int
    ) -> dict[int, tuple[float, int]]:
        calls.append(symbol)
        return real(self, symbol, start_ms, end_ms)

    monkeypatch.setattr(RawDataArchive, "get_daily_quote_volume", spy)

    assert load_carry_inputs(_spec(), root, archive).quote_volume is None
    assert calls == []
    assert load_carry_inputs(_liquid_spec(), root, archive).quote_volume is not None
    assert calls == ["BTCUSDT", "ETHUSDT"]


# ---------------------------------------------------------------------------
# T1 speed (arch-c9-2): bisect lookups over strictly ascending ticks
# ---------------------------------------------------------------------------


def _linear_sizing(ticks: Sequence[PriceTick], decision_ns: int) -> PriceTick | None:
    """The pre-T1 full scan of ``sizing_price``."""
    latest: PriceTick | None = None
    for tick in ticks:
        if decision_ns - PRICE_WINDOW_NS <= tick.ts < decision_ns and (
            latest is None or tick.ts > latest.ts
        ):
            latest = tick
    return latest


def _linear_release(ticks: Sequence[PriceTick], queue_ns: int) -> PriceTick | None:
    """The pre-T1 full scan of ``release_tick`` (before ``check_release``)."""
    first: PriceTick | None = None
    for tick in ticks:
        if tick.ts > queue_ns and (first is None or tick.ts < first.ts):
            first = tick
    return first


def test_sizing_and_release_lookups_equal_the_full_scan() -> None:
    """The bisect lookups pick the full scan's tick on strictly ascending ticks, at every edge.

    Probes sit on, and 1 ns either side of, every tick ts and every ``ts + PRICE_WINDOW_NS``;
    a stale release raises the same ``CarryExecutionError`` text.
    """
    import random

    rng = random.Random(9)
    checked = 0
    for _ in range(60):
        stamps = sorted(set(rng.randrange(0, 6 * HOUR, MIN // 4) for _ in range(rng.randrange(0, 30))))
        ticks = tuple(PriceTick(ts, float(index)) for index, ts in enumerate(stamps))
        probes = {0, 7 * HOUR}
        for ts in stamps:
            for edge in (ts, ts + PRICE_WINDOW_NS, ts - PRICE_WINDOW_NS):
                probes.update({edge - 1, edge, edge + 1})
        for at_ns in sorted(probes):
            assert sizing_price(ticks, at_ns) == _linear_sizing(ticks, at_ns)
            expected = _linear_release(ticks, at_ns)
            if expected is not None and expected.ts - at_ns > PRICE_WINDOW_NS:
                with pytest.raises(CarryExecutionError, match="stale release BTCUSDT") as stale:
                    release_tick("BTCUSDT", ticks, at_ns)
                assert f"released at {expected.ts}" in str(stale.value)
            else:
                assert release_tick("BTCUSDT", ticks, at_ns) == expected
            checked += 1
    assert checked > 1_000


@pytest.mark.parametrize(
    ("second_ts", "ascending"),
    [(D + INIT + 1, True), (D + INIT, False), (D + INIT - 1, False)],
)
def test_preflight_requires_strictly_ascending_ticks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second_ts: int, ascending: bool
) -> None:
    """The bisect lookups need strictly ascending ticks: preflight fails closed otherwise."""
    spec = _spec()
    prices = _minute_bars_for_run(START, END)
    _stub_settlements(monkeypatch, dict.fromkeys(spec.symbols, _settlements_from(START, END)))
    root = tmp_path / "catalog"
    _valid_catalog(root, prices, START)
    crafted = (PriceTick(D + INIT, 1.0), PriceTick(second_ts, 2.0))
    monkeypatch.setattr(
        "vibe_quant.portfolio.ticks.select_ticks", lambda *args, **kwargs: crafted
    )

    if ascending:
        assert load_carry_inputs(spec, root, _archive(tmp_path)).ticks["BTCUSDT"] == crafted
    else:
        with pytest.raises(CarryDataError, match="BTCUSDT: decision ticks are not strictly ascending"):
            load_carry_inputs(spec, root, _archive(tmp_path))


# ---------------------------------------------------------------------------
# arch-c9-2 1h tier: bar_ns clock (R13), exact daily close (R20), 1h release (R15),
# 1m trend inputs (R47)
# ---------------------------------------------------------------------------

DT = _ns("2024-03-11") + HOUR  # a daily 01:00 decision
MS = 1_000_000


def _bars_every(bar_ns: int, first_ns: int, end_ns: int, skip: Sequence[int] = ()) -> BarSeries:
    """Bars opening every ``bar_ns`` (ts = open + bar_ns - 1 ms), close = 100 + index."""
    opens = [o for o in range(first_ns, end_ns, bar_ns) if o not in skip]
    return _series_at([o + bar_ns - MS for o in opens], [100.0 + i for i in range(len(opens))])


@pytest.mark.parametrize(("bar_ns", "fill_offset"), [(MIN, 59_999 * MS), (HOUR, 3_599_999 * MS)])
def test_select_ticks_bar_ns(bar_ns: int, fill_offset: int) -> None:
    """R13: sizing = the bar ending D - 1 ms, fill = the bar opening D, at either bar size."""
    bars = _bars_every(bar_ns, DT - 2 * HOUR, DT + 2 * HOUR)
    sizing, fill = select_ticks(bars, [DT], create_instrument("BTCUSDT"), DT + DAY, bar_ns=bar_ns)
    assert (sizing.ts - DT, fill.ts - DT) == (-MS, fill_offset)
    opens = [int(t) + MS - bar_ns for t in bars.ts]
    assert sizing.price == 100.0 + opens.index(DT - bar_ns)
    assert fill.price == 100.0 + opens.index(DT)


def test_select_ticks_hour_window_edges() -> None:
    """At 1h the sizing window holds only the bar opening D - 1h and the fill window only
    the one opening D: a missing bar is a missing tick, never a neighbour."""
    instrument = create_instrument("BTCUSDT")
    no_sizing = _bars_every(HOUR, DT - 3 * HOUR, DT + 3 * HOUR, skip=[DT - HOUR])
    ticks = select_ticks(no_sizing, [DT], instrument, DT + DAY, bar_ns=HOUR)
    assert [t.ts for t in ticks] == [DT + HOUR - MS]
    no_fill = _bars_every(HOUR, DT - 3 * HOUR, DT + 3 * HOUR, skip=[DT])
    ticks = select_ticks(no_fill, [DT], instrument, DT + DAY, bar_ns=HOUR)
    assert [t.ts for t in ticks] == [DT - MS]


def test_hour_fill_bar_missing_is_a_stale_release() -> None:
    """R15 (rule level): the 1h fill tick is released inside the window (D + 3,599,999 ms);
    without it the next own tick is the next day's sizing tick -> stale, fail closed."""
    instrument = create_instrument("BTCUSDT")
    decisions = [DT, DT + DAY]
    full = _bars_every(HOUR, DT - 2 * HOUR, DT + DAY + 2 * HOUR)
    ticks = select_ticks(full, decisions, instrument, DT + 2 * DAY, bar_ns=HOUR)
    assert release_tick("BTCUSDT", ticks, DT) == ticks[1]
    assert ticks[1].ts - DT == 3_599_999 * MS
    holed = _bars_every(HOUR, DT - 2 * HOUR, DT + DAY + 2 * HOUR, skip=[DT])
    ticks = select_ticks(holed, decisions, instrument, DT + 2 * DAY, bar_ns=HOUR)
    with pytest.raises(CarryExecutionError, match=r"^stale release BTCUSDT: queued at "):
        release_tick("BTCUSDT", ticks, DT)


@pytest.mark.parametrize(("bar_ns", "last"), [(MIN, "23:59"), (HOUR, "23:00")])
def test_trend_close_exact(bar_ns: int, last: str) -> None:
    """R20: C[B] is the close of the bar ending exactly B - 1 ms; with that bar missing it
    is absent (never the previous bar's close, which boundary_prices would take)."""
    instrument = create_instrument("BTCUSDT")
    b1, b2 = _ns("2024-03-11"), _ns("2024-03-12")
    missing = _ns("2024-03-11") + int(last[:2]) * HOUR + int(last[3:]) * MIN
    bars = _bars_every(bar_ns, b1 - 2 * bar_ns, b2 + bar_ns, skip=[missing])
    closes = trend_closes(bars, [b1, b2], instrument)
    opens = [int(t) + MS - bar_ns for t in bars.ts]
    assert closes == {b1: 100.0 + opens.index(b1 - bar_ns)}
    if bar_ns == MIN:  # the as-of rule (ts in (B - 60 min, B]) takes the 23:58 neighbour
        assert b2 in boundary_prices(bars, [b2], instrument)
    assert trend_closes(bars, [b1 + 1], instrument) == {}  # an off-boundary key is exact too


def test_load_trend_1m_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R47: a 1m trend run keeps catalog bars/instruments and 1m MARK; closes by R20's rule."""
    spec = CarrySpec.from_dates(
        ("BTCUSDT", "ETHUSDT"),
        "2024-06-03",
        "2024-07-01",
        signal="trend",
        schedule="daily",
        weighting="time_series",
        direction="long_flat",
        vol_target=0.4,
    )
    decisions = decision_times(START, END, "daily")
    prices: dict[int, float] = {}
    for decision_ns in decisions:
        prices[decision_ns - MIN] = 100.0
        prices[decision_ns] = 101.0
    boundaries = list(range(START - 160 * DAY, END + DAY, DAY))
    for index, boundary in enumerate(boundaries):
        if index != 5:  # one missing 23:59 bar
            prices[boundary - MIN] = 200.0 + index
    prices[boundaries[5] - 2 * MIN] = 1.0  # its 23:58 neighbour
    prices = dict(sorted(prices.items()))  # the catalog needs ascending ts_init
    _stub_settlements(monkeypatch, dict.fromkeys(spec.symbols, _settlements_from(START, END)))
    root = tmp_path / "catalog"
    _valid_catalog(root, prices, START)

    inputs = load_carry_inputs(spec, root, _archive(tmp_path))

    assert inputs.decisions == tuple(decisions)
    assert inputs.price_source == "catalog_1m"
    assert inputs.mark_source == "catalog_1m_mark"
    assert inputs.marks["BTCUSDT"].ts[0] == START
    assert inputs.hour_bars is None
    assert inputs.instruments_source is None
    assert inputs.trend_closes is not None
    expected = {b: 200.0 + i for i, b in enumerate(boundaries) if i != 5 and b <= END}
    assert inputs.trend_closes["BTCUSDT"] == expected
    assert inputs.ticks["BTCUSDT"][:2] == (
        PriceTick(decisions[0] - MS, 100.0),
        PriceTick(decisions[0] + INIT, 101.0),
    )
    assert inputs.listings == dict.fromkeys(spec.symbols, boundaries[0] - MIN)


def test_funding_inputs_carry_no_trend_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy funding path is untouched: no trend closes, listings or 1h fields."""
    spec = _spec()
    _stub_settlements(monkeypatch, dict.fromkeys(spec.symbols, _settlements_from(START, END)))
    root = tmp_path / "catalog"
    _valid_catalog(root, _minute_bars_for_run(START, END), START)
    inputs = load_carry_inputs(spec, root, _archive(tmp_path))
    assert (inputs.trend_closes, inputs.listings, inputs.hour_bars) == (None, None, None)
    assert (inputs.price_source, inputs.mark_source) == ("catalog_1m", "catalog_1m_mark")
