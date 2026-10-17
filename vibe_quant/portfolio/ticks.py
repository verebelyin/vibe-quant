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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.identifiers import TradeId

from vibe_quant.data.catalog import get_bar_type
from vibe_quant.data.fill_ticks import FILL_TICK_SIZE
from vibe_quant.portfolio.signal import Settlements, carry_hours
from vibe_quant.portfolio.sizing import InstrumentSpec
from vibe_quant.portfolio.spec import (
    DAY_NS,
    LOOKBACK_NS,
    PRICE_WINDOW_NS,
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
_BAR_TS_INIT_OFFSET_NS = 59_999_000_000  # catalog 1m bar: ts_init = open + 59.999 s


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


def _price(instrument: CryptoPerpetual, close: float) -> float:
    """The instrument's tick-rounded price (what NT fills at)."""
    return float(instrument.make_price(close))


def select_ticks(
    bars: BarSeries,
    decisions: Sequence[int],
    instrument: CryptoPerpetual,
    end_ns: int,
) -> tuple[PriceTick, ...]:
    """The decision ticks of a symbol, ascending, all with ``ts < end_ns`` (R20b).

    Per decision: the close of the LAST 1m bar opening in ``[D-60 min, D-1 min]``
    (sizing) then of the FIRST one opening in ``[D, D+59 min]`` (fill). A window
    with no bar contributes no tick -- a hole is not a shifted window, so a
    symbol may legitimately have only one tick (or none) that week.
    """
    ts = np.asarray(bars.ts, dtype=np.int64)
    close = bars.close
    opens = ts - _BAR_TS_INIT_OFFSET_NS
    ticks: list[PriceTick] = []
    for decision_ns in decisions:
        sizing = int(np.searchsorted(opens, decision_ns - _MINUTE_NS, side="right")) - 1
        if sizing >= 0 and int(opens[sizing]) >= decision_ns - PRICE_WINDOW_NS:
            ticks.append(PriceTick(int(ts[sizing]), _price(instrument, float(close[sizing]))))
        fill = int(np.searchsorted(opens, decision_ns, side="left"))
        if fill < len(opens) and int(opens[fill]) <= decision_ns + PRICE_WINDOW_NS - _MINUTE_NS:
            ticks.append(PriceTick(int(ts[fill]), _price(instrument, float(close[fill]))))
    return tuple(tick for tick in ticks if tick.ts < end_ns)


def boundary_prices(
    bars: BarSeries,
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


def sizing_price(ticks: Sequence[PriceTick], decision_ns: int) -> PriceTick | None:
    """The last tick in ``[D-60 min, D)``: the price the decision sizes against (R9)."""
    lower_ns = decision_ns - PRICE_WINDOW_NS
    latest: PriceTick | None = None
    for tick in ticks:
        if lower_ns <= tick.ts < decision_ns and (latest is None or tick.ts > latest.ts):
            latest = tick
    return latest


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


def release_tick(
    symbol: str,
    ticks: Sequence[PriceTick],
    queue_ns: int,
) -> PriceTick | None:
    """The first tick strictly after ``queue_ns``, with :func:`check_release` applied.

    ``None`` means the symbol has no later tick, so the order can never be
    released (the caller raises R15c). Release is per instrument: another
    symbol's tick never releases this order.
    """
    first: PriceTick | None = None
    for tick in ticks:
        if tick.ts > queue_ns and (first is None or tick.ts < first.ts):
            first = tick
    if first is not None:
        check_release(symbol, queue_ns, first.ts)
    return first


def _settlements(
    calc: FundingCalculator, symbol: str
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """The calculator's archived ``(times_ns, rates)`` for a symbol."""
    return calc.settlements(symbol)


def _catalog_instruments(catalog: Path) -> dict[str, CryptoPerpetual]:
    """The catalog's instruments by raw symbol (the precisions NT checks ticks against)."""
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    return {
        str(instrument.raw_symbol): instrument
        for instrument in ParquetDataCatalog(str(catalog)).instruments()
    }


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
    is not asked for prices it never had (R32). The spec's own ranges are
    validated at ``CarrySpec`` construction, so a non-``CarrySpec`` is the only
    spec failure reachable here.
    """
    if not isinstance(spec, CarrySpec):
        raise ValueError(f"spec must be a CarrySpec, got {type(spec).__name__}")
    catalog = Path(catalog_path)
    archive = Path(archive_path)
    if not archive.is_file():
        raise CarryDataError(f"funding archive {archive} does not exist")
    if not catalog.is_dir():
        raise CarryDataError(f"catalog {catalog} does not exist")

    from vibe_quant.data.mark_bars import MarkDataMissing, load_mark_series, require_mark_coverage
    from vibe_quant.validation.extraction import load_catalog_bars

    calc = FundingCalculator(archive)
    decisions = tuple(decision_times(spec.start_ns, spec.end_ns))
    boundaries = range(spec.start_ns, spec.end_ns + DAY_NS, DAY_NS)
    lookback_ns = spec.start_ns - LOOKBACK_NS
    available = _catalog_instruments(catalog)
    instruments: dict[str, CryptoPerpetual] = {}
    specs: dict[str, InstrumentSpec] = {}
    ticks: dict[str, tuple[PriceTick, ...]] = {}
    settlements: dict[str, Settlements] = {}
    marks: dict[str, MarkSeries] = {}
    daily: dict[str, dict[int, float]] = {}
    for symbol in spec.symbols:
        instrument = available.get(symbol)
        if instrument is None:
            raise CarryDataError(f"{symbol}: no instrument in catalog {catalog}")
        bars = load_catalog_bars(catalog, str(get_bar_type(symbol, "1m")))
        if bars is None or len(bars.ts) == 0:
            raise CarryDataError(f"{symbol}: no 1m bars in catalog {catalog}")
        times, rates = _settlements(calc, symbol)
        if not any(lookback_ns <= time_ns < spec.end_ns for time_ns in times):
            raise CarryDataError(
                f"{symbol}: no funding settlement in [{lookback_ns}, {spec.end_ns})"
            )
        series = load_mark_series(catalog, symbol)
        try:
            require_mark_coverage(series, symbol, max(spec.start_ns, times[0]), spec.end_ns)
        except MarkDataMissing as error:
            raise CarryDataError(str(error)) from error
        assert series is not None  # require_mark_coverage raised for an absent series
        instruments[symbol] = instrument
        specs[symbol] = InstrumentSpec.from_instrument(instrument)
        ticks[symbol] = select_ticks(bars, decisions, instrument, spec.end_ns)
        settlements[symbol] = Settlements(
            times_ns=tuple(times),
            rates=tuple(rates),
            hours=carry_hours(calc.settlement_hours(symbol)),
        )
        marks[symbol] = series
        daily[symbol] = boundary_prices(bars, boundaries, instrument)
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
    )
