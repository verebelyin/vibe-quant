"""Decision-time 1m ticks, run inputs and the fail-closed preflight (design §1.3, §8-T2b).

The book decides once a week and needs only three prices per symbol-week: the
close known at the decision (the *sizing* tick), the fill price (the *fill*
tick) and the daily closes equity is marked at. All three come from the 1m LAST
catalog bars, which the project stamps ``ts_init = open + 59.999 s``:

* sizing tick = close of the LAST 1m bar opening in ``[D-60 min, D-1 min]``;
* fill tick = close of the FIRST 1m bar opening in ``[D, D+59 min]``;
* daily boundary close = close of the 1m bar with ``ts_init`` in ``(B-60 min, B]``.

Orders queued at ``D`` are released by the symbol's own next tick, and every
release is checked against the 60-minute price window (:func:`check_release`,
R15b). :func:`load_carry_inputs` assembles and preflights the whole input set
before any engine exists, failing closed with the symbol and the reason (R28).

The trend signal (arch-c9-2) adds the exact daily close (:func:`trend_closes`, the
bar ending ``B - 1 ms``) and a 1h research tier: ``fill_granularity="1h"`` takes
bars from the archive only (:mod:`~vibe_quant.portfolio.bars`) and instruments
from the frozen snapshot (:mod:`~vibe_quant.portfolio.instruments`), on the same
clock with ``bar_ns`` = 1h (R13).
"""

from __future__ import annotations

import bisect
import sqlite3
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.identifiers import TradeId

from vibe_quant.data.catalog import get_bar_type
from vibe_quant.data.fill_ticks import FILL_TICK_SIZE
from vibe_quant.portfolio.bars import (
    HOUR_NS,
    CarryNotListed,
    HourBars,
    load_hour_bars,
    warmup_start,
)
from vibe_quant.portfolio.signal import QUOTE_VOLUME_DAYS, Settlements, carry_hours
from vibe_quant.portfolio.sizing import InstrumentSpec
from vibe_quant.portfolio.spec import (
    DAY_NS,
    LOOKBACK_NS,
    PRICE_WINDOW_NS,
    TREND_LOOKBACKS_DAYS,
    CarryDataError,
    CarryExecutionError,
    CarrySpec,
    decision_times,
)
from vibe_quant.validation.funding import FundingCalculator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nautilus_trader.model.instruments import CryptoPerpetual

    from vibe_quant.data.mark_bars import MarkSeries
    from vibe_quant.metrics import BarSeries

_MINUTE_NS = 60 * 10**9
_MS_NS = 10**6  # a bar's ts_init = open + bar_ns - 1 ms (catalog 1m: + 59.999 s)


@dataclass(frozen=True)
class PriceTick:
    """One 1m bar close the engine may fill on, stamped at the bar's ``ts_init``."""

    ts: int
    price: float


@dataclass(frozen=True)
class CarryInputs:
    """Everything a carry run needs, preflighted by :func:`load_carry_inputs`."""

    spec: CarrySpec
    decisions: tuple[int, ...]
    instruments: dict[str, CryptoPerpetual]
    specs: dict[str, InstrumentSpec]
    ticks: dict[str, tuple[PriceTick, ...]]
    settlements: dict[str, Settlements]
    marks: dict[str, MarkSeries]
    calc: FundingCalculator
    boundary_prices: dict[str, dict[int, float]]
    # inverse_vol only: daily closes from start - L days (boundary_prices' rule)
    vol_closes: dict[str, dict[int, float]] | None = None
    # min_quote_volume only: UTC day start (ns) -> archived 1m quote-volume sum
    quote_volume: dict[str, dict[int, float]] | None = None
    # trend only: day boundary B -> the close of the bar ending exactly B - 1 ms (R20)
    trend_closes: dict[str, dict[int, float]] | None = None
    # trend only: open of each symbol's first bar (1h: first archived 1h bar, R42)
    listings: dict[str, int] | None = None
    # the tier's clock and sources (R13, R43, R54)
    bar_ns: int = _MINUTE_NS
    price_source: str = "catalog_1m"
    mark_source: str = "catalog_1m_mark"
    # 1h only: validated archive bars (marks, frozen-hour counts) and the snapshot used
    hour_bars: dict[str, HourBars] | None = None
    instruments_source: dict[str, str] | None = None


def _price(instrument: CryptoPerpetual, close: float) -> float:
    """The instrument's tick-rounded price (what NT fills at)."""
    return float(instrument.make_price(close))


def select_ticks(
    bars: BarSeries | HourBars,
    decisions: Sequence[int],
    instrument: CryptoPerpetual,
    end_ns: int,
    *,
    bar_ns: int = _MINUTE_NS,
) -> tuple[PriceTick, ...]:
    """The decision ticks of a symbol, ascending, all with ``ts < end_ns`` (R20b, R13).

    Per decision: the close of the LAST bar opening in ``[D-60 min, D-bar_ns]``
    (sizing) then of the FIRST one opening in ``[D, D+60 min-bar_ns]`` (fill), a
    bar's open being ``ts - (bar_ns - 1 ms)``. At 1m that is ``[D-60 min, D-1 min]``
    and ``[D, D+59 min]``; at 1h only the bar ending ``D - 1 ms`` sizes and only the
    bar opening ``D`` fills. A window with no bar contributes no tick -- a hole is
    not a shifted window, so a symbol may legitimately have only one tick (or none).
    """
    ts = np.asarray(bars.ts, dtype=np.int64)
    close = bars.close
    opens = ts - (bar_ns - _MS_NS)
    ticks: list[PriceTick] = []
    for decision_ns in decisions:
        sizing = int(np.searchsorted(opens, decision_ns - bar_ns, side="right")) - 1
        if sizing >= 0 and int(opens[sizing]) >= decision_ns - PRICE_WINDOW_NS:
            ticks.append(PriceTick(int(ts[sizing]), _price(instrument, float(close[sizing]))))
        fill = int(np.searchsorted(opens, decision_ns, side="left"))
        if fill < len(opens) and int(opens[fill]) <= decision_ns + PRICE_WINDOW_NS - bar_ns:
            ticks.append(PriceTick(int(ts[fill]), _price(instrument, float(close[fill]))))
    return tuple(tick for tick in ticks if tick.ts < end_ns)


def trend_closes(
    bars: BarSeries | HourBars,
    boundaries: Sequence[int],
    instrument: CryptoPerpetual,
) -> dict[int, float]:
    """Close of the bar ending EXACTLY ``B - 1 ms`` per boundary (R20, both tiers).

    The 23:59 1m bar or the 23:00 1h bar: the last trade before ``B``. A boundary
    without that exact bar is absent -- never the previous bar's close, unlike
    :func:`boundary_prices`' as-of window -- so the signal sees a hole as a hole.
    """
    ts = np.asarray(bars.ts, dtype=np.int64)
    close = bars.close
    closes: dict[int, float] = {}
    for boundary_ns in boundaries:
        index = int(np.searchsorted(ts, boundary_ns - _MS_NS, side="left"))
        if index < len(ts) and int(ts[index]) == boundary_ns - _MS_NS:
            closes[int(boundary_ns)] = _price(instrument, float(close[index]))
    return closes


def boundary_prices(
    bars: BarSeries | HourBars,
    boundaries: Sequence[int],
    instrument: CryptoPerpetual,
) -> dict[int, float]:
    """Close of the 1m bar with ``ts_init`` in ``(B-60 min, B]`` per boundary (§4).

    A boundary without such a bar is absent from the mapping; whether that is
    fatal (a held symbol) is the caller's decision, not this one's.
    """
    ts = np.asarray(bars.ts, dtype=np.int64)
    close = bars.close
    prices: dict[int, float] = {}
    for boundary_ns in boundaries:
        index = int(np.searchsorted(ts, boundary_ns, side="right")) - 1
        if index >= 0 and int(ts[index]) > boundary_ns - PRICE_WINDOW_NS:
            prices[int(boundary_ns)] = _price(instrument, float(close[index]))
    return prices


def to_trade_ticks(
    instrument: CryptoPerpetual,
    ticks: Sequence[PriceTick],
    size: float = FILL_TICK_SIZE,
) -> list[TradeTick]:
    """The ticks as in-memory ``TradeTick``s: TradeId ``C{ts}``, ts_event = ts_init.

    ``size`` is L1 liquidity in NT's matching engine, hence huge by default (a
    small one causes partial fills); R29's test asks for a partial fill with a
    small ``size``.
    """
    quantity = instrument.make_qty(size)
    return [
        TradeTick(
            instrument.id,
            instrument.make_price(tick.price),
            quantity,
            AggressorSide.NO_AGGRESSOR,
            TradeId(f"C{tick.ts}"),
            tick.ts,
            tick.ts,
        )
        for tick in ticks
    ]


def _tick_ts(tick: PriceTick) -> int:
    return tick.ts


def sizing_price(ticks: Sequence[PriceTick], decision_ns: int) -> PriceTick | None:
    """The last tick in ``[D-60 min, D)``: the price the decision sizes against (R9).

    ``ticks`` are strictly ascending (``load_carry_inputs`` checks it), so a bisect finds it.
    """
    index = bisect.bisect_left(ticks, decision_ns, key=_tick_ts) - 1
    if index >= 0 and ticks[index].ts >= decision_ns - PRICE_WINDOW_NS:
        return ticks[index]
    return None


def check_release(symbol: str, queue_ns: int, release_ns: int) -> None:
    """Fail closed when a queued order is released more than 60 min late (R15b).

    The boundary is inclusive: exactly ``PRICE_WINDOW_NS`` after the queue is
    still the decision's own price window (the fill tick is ``D`` + 59.999 s).
    """
    age_ns = release_ns - queue_ns
    if age_ns > PRICE_WINDOW_NS:
        raise CarryExecutionError(
            f"stale release {symbol}: queued at {queue_ns}, released at {release_ns} "
            f"({age_ns} ns after the decision, limit {PRICE_WINDOW_NS} ns)"
        )


def unreleased_error(symbol: str, queue_ns: int) -> CarryExecutionError:
    """R15c: an order of ``symbol`` queued at ``queue_ns`` with no own tick after it.

    One text for both executors (C9-1): the reference raises it at the decision, NT
    at the next decision or at the run end, whichever finds the order still queued.
    """
    return CarryExecutionError(
        f"unreleased orders for {symbol} at {queue_ns}: "
        "no own tick after the decision before the run ends"
    )


def release_tick(
    symbol: str,
    ticks: Sequence[PriceTick],
    queue_ns: int,
) -> PriceTick | None:
    """The first tick strictly after ``queue_ns``, with :func:`check_release` applied.

    ``None`` means the symbol has no later tick, so the order can never be
    released (the caller raises R15c). Release is per instrument: another
    symbol's tick never releases this order. ``ticks`` are strictly ascending.
    """
    index = bisect.bisect_right(ticks, queue_ns, key=_tick_ts)
    if index == len(ticks):
        return None
    first = ticks[index]
    check_release(symbol, queue_ns, first.ts)
    return first


def _settlements(
    calc: FundingCalculator, symbol: str
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """The calculator's archived ``(times_ns, rates)`` for a symbol."""
    return calc.settlements(symbol)


def _require_settlement(
    symbol: str, times: Sequence[int], lookback_ns: int, end_ns: int, *, waived: bool = False
) -> None:
    """Preflight shared by both tiers: a settlement in ``[start - 7 d, end)`` unless waived."""
    if not waived and not any(lookback_ns <= time_ns < end_ns for time_ns in times):
        raise CarryDataError(f"{symbol}: no funding settlement in [{lookback_ns}, {end_ns})")


def _ascending(symbol: str, ticks: tuple[PriceTick, ...]) -> tuple[PriceTick, ...]:
    """Preflight shared by both tiers: decision ticks strictly ascending (bisect needs it)."""
    if any(later.ts <= tick.ts for tick, later in pairwise(ticks)):
        raise CarryDataError(f"{symbol}: decision ticks are not strictly ascending")
    return ticks


def _catalog_instruments(catalog: Path) -> dict[str, CryptoPerpetual]:
    """The catalog's instruments by raw symbol (the precisions NT checks ticks against)."""
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    return {
        str(instrument.raw_symbol): instrument
        for instrument in ParquetDataCatalog(str(catalog)).instruments()
    }


def _quote_volume(spec: CarrySpec, archive: Path, symbol: str) -> dict[int, float]:
    """The symbol's daily 1m quote volume over ``[start - 7 d, end)``, preflighted.

    A read failure is ``archive read failed`` (never a hole); a malformed day
    or a symbol with no 1m row in ``[start, end)`` is a ``CarryDataError``.
    """
    from vibe_quant.data.archive import RawDataArchive

    raw = RawDataArchive(archive)
    try:
        days = raw.get_daily_quote_volume(
            symbol,
            (spec.start_ns - QUOTE_VOLUME_DAYS * DAY_NS) // 10**6,
            spec.end_ns // 10**6,
        )
    except sqlite3.Error as error:
        raise CarryDataError(f"archive read failed: {symbol}: {error}") from error
    except ValueError as error:
        raise CarryDataError(str(error)) from error
    finally:
        raw.close()
    if not any(spec.start_ns <= day_ns < spec.end_ns for day_ns in days):
        raise CarryDataError(
            f"{symbol}: no 1m klines in archive {archive} for [{spec.start_ns}, {spec.end_ns})"
        )
    return {day_ns: total for day_ns, (total, _count) in days.items()}


def load_carry_inputs(
    spec: CarrySpec,
    catalog_path: Path | str,
    archive_path: Path | str,
) -> CarryInputs:
    """Build and preflight a run's inputs before any engine exists (R28, R32).

    Order: the spec first (``ValueError``), then the archive/catalog paths, then
    per symbol -- instrument, 1m bars, a settlement in ``[start-7d, end)`` and
    MARK coverage -- every failure a ``CarryDataError`` naming the symbol and
    the reason (``MarkDataMissing`` wrapped). MARK is required only from
    ``max(start, the symbol's first settlement)``, so a symbol listed mid-window
    is not asked for prices it never had (R32). With ``min_quote_volume`` set,
    each symbol's daily 1m quote volume is read from the archive (only then)
    and must have a 1m row in ``[start, end)``. The spec's own ranges are
    validated at ``CarrySpec`` construction, so a non-``CarrySpec`` is the only
    spec failure reachable here.
    """
    if not isinstance(spec, CarrySpec):
        raise ValueError(f"spec must be a CarrySpec, got {type(spec).__name__}")
    catalog = Path(catalog_path)
    archive = Path(archive_path)
    if not archive.is_file():
        raise CarryDataError(f"funding archive {archive} does not exist")
    if spec.fill_granularity == "1h":
        return _load_hour_inputs(spec, archive)
    if not catalog.is_dir():
        raise CarryDataError(f"catalog {catalog} does not exist")

    from vibe_quant.data.mark_bars import MarkDataMissing, load_mark_series, require_mark_coverage
    from vibe_quant.validation.extraction import load_catalog_bars

    calc = FundingCalculator(archive)
    decisions = tuple(decision_times(spec.start_ns, spec.end_ns, spec.schedule))
    boundaries = range(spec.start_ns, spec.end_ns + DAY_NS, DAY_NS)
    lookback_ns = spec.start_ns - LOOKBACK_NS
    available = _catalog_instruments(catalog)
    instruments: dict[str, CryptoPerpetual] = {}
    specs: dict[str, InstrumentSpec] = {}
    ticks: dict[str, tuple[PriceTick, ...]] = {}
    settlements: dict[str, Settlements] = {}
    marks: dict[str, MarkSeries] = {}
    daily: dict[str, dict[int, float]] = {}
    vol_boundaries = range(
        spec.start_ns - spec.vol_lookback_days * DAY_NS, spec.end_ns + DAY_NS, DAY_NS
    )
    vol_closes: dict[str, dict[int, float]] | None = (
        {} if spec.weighting == "inverse_vol" else None
    )
    quote_volume: dict[str, dict[int, float]] | None = (
        None if spec.min_quote_volume is None else {}
    )
    trend = spec.signal == "trend"
    closes: dict[str, dict[int, float]] | None = {} if trend else None
    listings: dict[str, int] | None = {} if trend else None
    for symbol in spec.symbols:
        instrument = available.get(symbol)
        if instrument is None:
            raise CarryDataError(f"{symbol}: no instrument in catalog {catalog}")
        bars = load_catalog_bars(catalog, str(get_bar_type(symbol, "1m")))
        if bars is None or len(bars.ts) == 0:
            raise CarryDataError(f"{symbol}: no 1m bars in catalog {catalog}")
        times, rates = _settlements(calc, symbol)
        _require_settlement(symbol, times, lookback_ns, spec.end_ns)
        series = load_mark_series(catalog, symbol)
        try:
            require_mark_coverage(series, symbol, max(spec.start_ns, times[0]), spec.end_ns)
        except MarkDataMissing as error:
            raise CarryDataError(str(error)) from error
        assert series is not None  # require_mark_coverage raised for an absent series
        instruments[symbol] = instrument
        specs[symbol] = InstrumentSpec.from_instrument(instrument)
        ticks[symbol] = _ascending(symbol, select_ticks(bars, decisions, instrument, spec.end_ns))
        settlements[symbol] = Settlements(
            times_ns=tuple(times),
            rates=tuple(rates),
            hours=carry_hours(calc.settlement_hours(symbol)),
        )
        marks[symbol] = series
        daily[symbol] = boundary_prices(bars, boundaries, instrument)
        if vol_closes is not None:
            vol_closes[symbol] = boundary_prices(bars, vol_boundaries, instrument)
        if quote_volume is not None:
            quote_volume[symbol] = _quote_volume(spec, archive, symbol)
        if closes is not None and listings is not None:
            closes[symbol] = trend_closes(bars, _trend_boundaries(spec), instrument)
            listings[symbol] = int(bars.ts[0]) - (_MINUTE_NS - _MS_NS)
    return CarryInputs(
        spec=spec,
        decisions=decisions,
        instruments=instruments,
        specs=specs,
        ticks=ticks,
        settlements=settlements,
        marks=marks,
        calc=calc,
        boundary_prices=daily,
        vol_closes=vol_closes,
        quote_volume=quote_volume,
        trend_closes=closes,
        listings=listings,
    )


def _trend_boundaries(spec: CarrySpec) -> range:
    """Every day boundary a trend decision reads: ``start - 160 d`` .. ``end``."""
    first = spec.start_ns - TREND_LOOKBACKS_DAYS[-1] * DAY_NS
    return range(first, spec.end_ns + DAY_NS, DAY_NS)


def _load_hour_inputs(spec: CarrySpec, archive: Path) -> CarryInputs:
    """The 1h research tier: archive bars, snapshot instruments, no catalog (R41-R45).

    Per symbol: the snapshot instrument (absent -> ``CarryDataError``), the 1h bars
    from ``start - 161 d`` (validated, frozen ones dropped) and the listing evidence.
    A symbol first listed at or after ``end`` is ``not_listed`` throughout and needs
    nothing else; a listed one needs a funding settlement in ``[start - 7 d, end)``
    unless it was listed in the last 8 h (its first settlement is not due before end).
    Its settlements are those before ``end`` (nothing later is ever read), and the
    sizing/fill ticks use the 1h clock. When no symbol is listed before ``end`` the
    run is refused here, before any engine (:class:`CarryNotListed`, C9-4).
    """
    from vibe_quant.portfolio.instruments import (
        load_snapshot,
        onboard_ms,
        price_increment,
        research_instrument,
    )

    snapshot = load_snapshot()
    calc = FundingCalculator(archive)
    decisions = tuple(decision_times(spec.start_ns, spec.end_ns, spec.schedule))
    boundaries = range(spec.start_ns, spec.end_ns + DAY_NS, DAY_NS)
    lookback_ns = spec.start_ns - LOOKBACK_NS
    instruments: dict[str, CryptoPerpetual] = {}
    specs: dict[str, InstrumentSpec] = {}
    ticks: dict[str, tuple[PriceTick, ...]] = {}
    settlements: dict[str, Settlements] = {}
    daily: dict[str, dict[int, float]] = {}
    closes: dict[str, dict[int, float]] = {}
    listings: dict[str, int] = {}
    hour_bars: dict[str, HourBars] = {}
    for symbol in spec.symbols:
        instrument = research_instrument(snapshot, symbol)
        bars = load_hour_bars(
            archive,
            symbol,
            warmup_start(spec.start_ns),
            spec.end_ns,
            price_increment(snapshot, symbol),
            onboard_ms=onboard_ms(snapshot, symbol),
        )
        instruments[symbol] = instrument
        specs[symbol] = InstrumentSpec.from_instrument(instrument)
        listings[symbol] = bars.listing_ns
        hour_bars[symbol] = bars
        if bars.listing_ns >= spec.end_ns:
            ticks[symbol], closes[symbol], daily[symbol] = (), {}, {}
            settlements[symbol] = Settlements(times_ns=(), rates=())
            continue
        times, rates = _settlements(calc, symbol)
        # listed in the last 8 h: no settlement is due before end yet (ineligible, no_history)
        late = bars.listing_ns >= spec.end_ns - 8 * HOUR_NS
        _require_settlement(symbol, times, lookback_ns, spec.end_ns, waived=late)
        keep = bisect.bisect_left(times, spec.end_ns)
        settlements[symbol] = Settlements(
            times_ns=times[:keep],
            rates=rates[:keep],
            hours=carry_hours(calc.settlement_hours(symbol)[:keep]),
        )
        ticks[symbol] = _ascending(
            symbol, select_ticks(bars, decisions, instrument, spec.end_ns, bar_ns=HOUR_NS)
        )
        closes[symbol] = trend_closes(bars, _trend_boundaries(spec), instrument)
        daily[symbol] = boundary_prices(bars, boundaries, instrument)
    if all(listing >= spec.end_ns for listing in listings.values()):
        raise CarryNotListed(spec.end_ns, listings)
    return CarryInputs(
        spec=spec,
        decisions=decisions,
        instruments=instruments,
        specs=specs,
        ticks=ticks,
        settlements=settlements,
        marks={},
        calc=calc,
        boundary_prices=daily,
        trend_closes=closes,
        listings=listings,
        bar_ns=HOUR_NS,
        price_source="archive_1h",
        mark_source="archive_1h_last_close",
        hour_bars=hour_bars,
        instruments_source={"path": snapshot.path.name, "sha256": snapshot.sha256},
    )
