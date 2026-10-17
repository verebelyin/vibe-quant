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
    """R20b: a tick at or after ``end_ns`` is dropped, one ns before it is kept."""
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
# R15b: release staleness
# ---------------------------------------------------------------------------


def test_check_release_limit() -> None:
    """R15b: ``release_ns - queue_ns > PRICE_WINDOW_NS`` is stale; the equality is not."""
    check_release("BTCUSDT", D, D + PRICE_WINDOW_NS)
    with pytest.raises(CarryExecutionError, match="stale release BTCUSDT"):
        check_release("BTCUSDT", D, D + PRICE_WINDOW_NS + 1)


def test_release_tick_is_the_first_tick_after_the_queue_ts() -> None:
    """R15b/R15c: strict ``>`` on the queue ts; ``None`` = no later own tick."""
    ticks = (PriceTick(D - 1, 1.0), PriceTick(D, 2.0), PriceTick(D + MIN, 3.0))
    assert release_tick("BTCUSDT", ticks, D) == PriceTick(D + MIN, 3.0)
    assert release_tick("BTCUSDT", ticks, D + MIN) is None
    assert release_tick("BTCUSDT", (), D) is None


def test_release_tick_applies_check_release_to_the_chosen_tick() -> None:
    """R15b: the first later tick past the window raises; an earlier in-window tick wins."""
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


def test_load_carry_inputs_carries_non_8h_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R35b: a non-8h ``settlement_hours`` of matching length reaches ``Settlements``."""
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
    """R28: ``ValueError`` for a bad spec first, then ``CarryDataError`` + symbol/reason."""
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
    """R28/R2: a settlement exactly at ``start - 7d`` is inside ``[start-7d, end)``."""
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
    """R28/R32: MARK must reach ``end``; coverage ending a day early fails closed."""
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
    """R32: MARK is required only from a mid-window listing's first settlement."""
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
