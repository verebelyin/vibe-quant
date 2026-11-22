"""Archive-direct 1h LAST bars of the research tier (arch-c9-2 R21, R41-R44, dreview C9-4).

One source per tier: a 1h run reads ``raw_klines`` interval '1h' from the archive
(read-only) and never the catalog. Every row is validated before any engine
exists; a frozen bar (zero volume, O=H=L=C) is dropped and counted. Listing
evidence comes from queries outside the bounded window read: the first archived
1h bar of the symbol, checked against its first funding row and the snapshot's
onboard date (a funding row before a first bar that opens after the onboard hour,
or more than 48 h before it without an onboard date = incomplete download). A
symbol whose first bar is at or after ``end`` is ``not_listed`` at every
decision; a run where every symbol is is refused before any engine is built (:class:`CarryNotListed`).

The helpers here (tmp archive, tmp instrument snapshot) are shared with
``test_carry_ticks`` and ``test_carry_reference``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vibe_quant.data.archive import RawDataArchive
from vibe_quant.portfolio import instruments as instruments_module
from vibe_quant.portfolio import ticks as ticks_module
from vibe_quant.portfolio.bars import CarryNotListed, HourBars, load_hour_bars
from vibe_quant.portfolio.instruments import build_snapshot, load_snapshot, research_instrument
from vibe_quant.portfolio.signal import decision_signals
from vibe_quant.portfolio.spec import DAY_NS, CarryDataError, CarrySpec, decision_times
from vibe_quant.portfolio.ticks import load_carry_inputs, trend_closes
from vibe_quant.validation.funding import clear_rate_cache

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

HOUR_MS = 3_600_000
HOUR_NS = HOUR_MS * 1_000_000
DAY_MS = 86_400_000
SYMBOL = "BTCUSDT"


def ns(stamp: str) -> int:
    """ISO UTC timestamp -> epoch ns."""
    return int(datetime.fromisoformat(stamp).replace(tzinfo=UTC).timestamp()) * 10**9


def ms(stamp: str) -> int:
    return ns(stamp) // 1_000_000


def kline(
    open_ms: int,
    close: float,
    *,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1.0,
    close_time: int | None = None,
) -> tuple[float | int, ...]:
    """One archive kline row (open_time, o, h, l, c, volume, close_time)."""
    o = close if open_ is None else open_
    return (
        open_ms,
        o,
        max(o, close) if high is None else high,
        min(o, close) if low is None else low,
        close,
        volume,
        open_ms + HOUR_MS - 1 if close_time is None else close_time,
    )


def hourly(first_ms: int, end_ms: int, price: float = 100.0) -> list[tuple[float | int, ...]]:
    """Hourly bars from ``first_ms`` to before ``end_ms``, close rising 0.1 per hour."""
    return [
        kline(open_ms, round(price + 0.1 * (index % 500), 1))
        for index, open_ms in enumerate(range(first_ms, end_ms, HOUR_MS))
    ]


def write_archive(
    path: Path,
    klines: Mapping[str, Iterable[tuple[float | int, ...]]],
    funding_from: Mapping[str, int] | None = None,
    funding_to_ms: int | None = None,
) -> Path:
    """A tmp archive: 1h klines per symbol, 8h funding rows from ``funding_from[symbol]`` (ms)."""
    archive = RawDataArchive(path)
    assert archive.conn is not None  # creates the schema even with no rows
    for symbol, rows in klines.items():
        archive.insert_klines(symbol, "1h", list(rows), "test")
    for symbol, first_ms in (funding_from or {}).items():
        last_ms = funding_to_ms if funding_to_ms is not None else first_ms + 400 * DAY_MS
        archive.insert_funding_rates(
            symbol, [(t + 2, 0.0001) for t in range(first_ms, last_ms, 8 * HOUR_MS)], "test"
        )
    archive.close()
    return path


def write_payload(
    path: Path,
    filters: Mapping[str, Mapping[str, str]],
    onboard: Mapping[str, int] | None = None,
) -> Path:
    """A minimal exchangeInfo payload: one PERPETUAL entry per symbol (onboard ms per symbol)."""
    symbols = []
    for symbol, values in filters.items():
        entry: dict[str, object] = {
            "symbol": symbol,
            "baseAsset": symbol.removesuffix("USDT"),
            "quoteAsset": "USDT",
            "contractType": "PERPETUAL",
            "onboardDate": (onboard or {}).get(symbol, 1_567_965_300_000),
            "requiredMarginPercent": "5.0000",
            "maintMarginPercent": "2.5000",
            "filters": [],
        }
        filter_list: list[dict[str, str]] = []
        if "tickSize" in values:
            filter_list.append({"filterType": "PRICE_FILTER", "tickSize": values["tickSize"]})
        filter_list.append(
            {
                "filterType": "LOT_SIZE",
                "stepSize": values.get("stepSize", "0.001"),
                "minQty": values.get("minQty", "0.001"),
            }
        )
        filter_list.append({"filterType": "MIN_NOTIONAL", "notional": values.get("notional", "5")})
        entry["filters"] = filter_list
        symbols.append(entry)
    path.write_text(json.dumps({"serverTime": 1_795_245_209_074, "symbols": symbols}))
    return path


DEFAULT_FILTERS = {"BTCUSDT": {"tickSize": "0.10"}, "ETHUSDT": {"tickSize": "0.01"}}


def use_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive: Path,
    filters: Mapping[str, Mapping[str, str]] | None = None,
    cutoff: str = "2026-11-22",
    onboard: Mapping[str, int] | None = None,
) -> Path:
    """Build a snapshot from a tmp payload + ``archive`` and point the loader at it."""
    chosen = DEFAULT_FILTERS if filters is None else filters
    payload = write_payload(tmp_path / "exchangeinfo.json", chosen, onboard)
    out = tmp_path / "research_instruments.json"
    out.write_bytes(build_snapshot(payload, archive, sorted(chosen), ns(cutoff)))
    monkeypatch.setattr(instruments_module, "SNAPSHOT_PATH", out)
    return out


def trend_spec(
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    start: str = "2024-03-01",
    end: str = "2024-03-20",
    tier: str = "1h",
) -> CarrySpec:
    return CarrySpec.from_dates(
        symbols,
        start,
        end,
        signal="trend",
        schedule="daily",
        weighting="time_series",
        direction="long_flat",
        vol_target=0.4,
        fill_granularity=tier,
    )


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_rate_cache()


def _increment() -> Decimal:
    return Decimal("0.1")


# ---------------------------------------------------------------------------
# R41: every row validated before any engine; frozen bars dropped and counted
# ---------------------------------------------------------------------------

_T = ms("2024-03-05T00:00:00")


@pytest.mark.parametrize(
    ("rows", "reason_open", "reason"),
    [
        ([kline(_T + 60_000, 100.0)], _T + 60_000, "open not on the hour"),
        (
            [kline(_T, 100.0, close_time=_T + HOUR_MS)],
            _T,
            f"close_time {_T + HOUR_MS} is not open + 3599999 ms",
        ),
        (
            [kline(_T, 100.0), kline(_T + HOUR_MS, 100.0, close_time=_T + HOUR_MS - 1)],
            _T + HOUR_MS,
            f"duplicate close_time {_T + HOUR_MS - 1}",
        ),
        # SQLite stores NaN as NULL (refused by NOT NULL), so inf is the non-finite probe
        ([kline(_T, 100.0, open_=float("inf"))], _T, "non-finite or non-positive OHLC"),
        ([kline(_T, 100.0, low=0.0)], _T, "non-finite or non-positive OHLC"),
        ([kline(_T, 100.0, open_=99.0, high=99.9)], _T, "high below open/close"),
        ([kline(_T, 100.0, open_=99.0, low=99.1)], _T, "low above open/close"),
        ([kline(_T, 100.05)], _T, "close 100.05 off the price grid 0.1"),
    ],
)
def test_bad_rows_fail_closed(
    tmp_path: Path, rows: list[tuple[float | int, ...]], reason_open: int, reason: str
) -> None:
    archive = write_archive(tmp_path / "a.db", {SYMBOL: rows})
    with pytest.raises(CarryDataError) as error:
        load_hour_bars(archive, SYMBOL, _T * 10**6, (_T + DAY_MS) * 10**6, _increment())
    assert str(error.value) == f"{SYMBOL} 1h row {reason_open}: {reason}"


def test_edges_of_row_checks_pass(tmp_path: Path) -> None:
    """high == max(o, c), low == min(o, c), close on the grid: accepted, ts = open + 1h - 1 ms."""
    rows = [kline(_T, 100.0, open_=99.0, high=100.0, low=99.0), kline(_T + HOUR_MS, 100.1)]
    archive = write_archive(tmp_path / "a.db", {SYMBOL: rows})
    bars = load_hour_bars(archive, SYMBOL, _T * 10**6, (_T + DAY_MS) * 10**6, _increment())
    assert bars.ts.tolist() == [(_T + HOUR_MS - 1) * 10**6, (_T + 2 * HOUR_MS - 1) * 10**6]
    assert bars.close.tolist() == [100.0, 100.1]
    assert bars.frozen_hours == 0
    assert bars.listing_ns == _T * 10**6


def test_frozen_bars_dropped_and_counted(tmp_path: Path) -> None:
    """Zero volume AND O=H=L=C is frozen (dropped); either alone is a real bar."""
    rows = [
        kline(_T, 100.0, volume=0.0),  # frozen
        kline(_T + HOUR_MS, 100.0, open_=99.9, volume=0.0),  # zero volume, moved: kept
        kline(_T + 2 * HOUR_MS, 100.0, volume=5.0),  # flat with volume: kept
        kline(_T + 3 * HOUR_MS, 100.0, volume=0.0),  # frozen
    ]
    archive = write_archive(tmp_path / "a.db", {SYMBOL: rows})
    bars = load_hour_bars(archive, SYMBOL, _T * 10**6, (_T + DAY_MS) * 10**6, _increment())
    assert bars.ts.tolist() == [(_T + 2 * HOUR_MS - 1) * 10**6, (_T + 3 * HOUR_MS - 1) * 10**6]
    assert bars.frozen_hours == 2


# ---------------------------------------------------------------------------
# R42 / C9-4: listing evidence outside the bounded read; all-prelisting outcome
# ---------------------------------------------------------------------------


def test_listing_evidence(tmp_path: Path) -> None:
    """Listing = the first archived 1h bar, even far before the window; no 1h rows at all
    is a named failure (the incomplete-download rule: ``test_incomplete_download_rule``)."""
    early = ms("2020-01-01T00:00:00")
    rows = [kline(early, 100.0), *hourly(_T, _T + DAY_MS)]
    ok = write_archive(tmp_path / "ok.db", {SYMBOL: rows}, {SYMBOL: early})
    bars = load_hour_bars(ok, SYMBOL, _T * 10**6, (_T + DAY_MS) * 10**6, _increment())
    assert bars.listing_ns == early * 10**6
    assert len(bars.ts) == 24

    empty = write_archive(tmp_path / "empty.db", {"ETHUSDT": hourly(_T, _T + DAY_MS)})
    with pytest.raises(CarryDataError) as error:
        load_hour_bars(empty, SYMBOL, _T * 10**6, (_T + DAY_MS) * 10**6, _increment())
    assert str(error.value) == f"{SYMBOL}: no 1h klines in archive {empty}"


def _first_funding_archive(path: Path, first_bar_ms: int, first_funding_ms: int) -> Path:
    """One day of 1h bars from ``first_bar_ms`` and one funding row at ``first_funding_ms``."""
    write_archive(path, {SYMBOL: hourly(first_bar_ms, first_bar_ms + DAY_MS)})
    archive = RawDataArchive(path)
    archive.insert_funding_rates(SYMBOL, [(first_funding_ms, 0.0001)], "test")
    archive.close()
    return path


_GAP_48H = 48 * HOUR_MS


@pytest.mark.parametrize(
    ("gap_ms", "onboard_after_bar_ms", "refused"),
    [
        # no onboard date: funding up to 48 h before the first bar is a normal listing
        (15 * HOUR_MS, None, False),  # SOL: funding 2020-09-13 16:00, first kline 09-14 07:00
        (16 * HOUR_MS, None, False),  # NEAR: 10-14 16:00 vs 10-15 08:00
        (24 * HOUR_MS, None, False),  # ZEC: 02-04 08:00 vs 02-05 08:00
        (_GAP_48H, None, False),
        (_GAP_48H + 1, None, True),
        (72 * HOUR_MS, None, True),
        (-1, None, False),  # funding after the first bar
        # onboard date (listing evidence): the first bar must open in the onboard hour or before
        (15 * HOUR_MS, 0, False),  # SOL: onboard 2020-09-14 07:00 = first kline
        (90 * 60_000, 30 * 60_000, False),  # AKE: onboard 12:30, first 1h bar 12:00
        (15 * HOUR_MS, HOUR_MS - 1, False),  # onboard in the first bar's hour
        (15 * HOUR_MS, -1, True),  # onboard in the hour before: bars missing
        (_GAP_48H, -_GAP_48H, True),  # no 48 h tolerance once the onboard date is known
        (72 * HOUR_MS, -72 * HOUR_MS, True),
    ],
)
def test_incomplete_download_rule(
    tmp_path: Path, gap_ms: int, onboard_after_bar_ms: int | None, refused: bool
) -> None:
    """R42 as ruled (R-c9t3-2, rev S2): a funding row before the first 1h bar is an
    incomplete download only when the first bar opens after the onboard date's hour, or,
    without an onboard date, more than 48 h after that funding row."""
    archive = _first_funding_archive(tmp_path / "a.db", _T, _T - gap_ms)
    onboard = None if onboard_after_bar_ms is None else _T + onboard_after_bar_ms
    window = (_T * 10**6, (_T + DAY_MS) * 10**6)
    if refused:
        with pytest.raises(CarryDataError) as error:
            load_hour_bars(archive, SYMBOL, *window, _increment(), onboard_ms=onboard)
        assert str(error.value) == f"{SYMBOL}: 1h bars start after funding (incomplete download)"
    else:
        bars = load_hour_bars(archive, SYMBOL, *window, _increment(), onboard_ms=onboard)
        assert bars.listing_ns == _T * 10**6
        assert len(bars) == 24


def test_onboard_date_reaches_the_rule_from_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 1h loader forwards the snapshot's ``onboard_ms``: funding 15 h before the first
    bar loads with onboard = first bar (SOL) and is refused with onboard one day earlier."""
    spec = trend_spec(symbols=("BTCUSDT",))
    first = ms("2023-09-01T00:00:00")
    archive = write_archive(
        tmp_path / "a.db",
        {"BTCUSDT": hourly(first, ms("2024-03-20T00:00:00"))},
        {"BTCUSDT": first - 15 * HOUR_MS},
    )
    use_snapshot(tmp_path, monkeypatch, archive, onboard={"BTCUSDT": first})
    assert load_carry_inputs(spec, tmp_path / "no-catalog", archive).listings == {
        "BTCUSDT": first * 10**6
    }
    use_snapshot(tmp_path, monkeypatch, archive, onboard={"BTCUSDT": first - DAY_MS})
    with pytest.raises(CarryDataError) as error:
        load_carry_inputs(spec, tmp_path / "no-catalog", archive)
    assert str(error.value) == "BTCUSDT: 1h bars start after funding (incomplete download)"


@pytest.mark.parametrize(
    ("listed_hours_before_end", "refused"), [(5, False), (8, False), (9, True)]
)
def test_listed_in_the_last_8h_needs_no_funding_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, listed_hours_before_end: int, refused: bool
) -> None:
    """rev nit: ETH listed at most 8 h before end has no settlement due before end, so it
    loads (empty settlements, ineligible as ``no_history``) instead of failing the run; listed
    earlier with no funding row before end is still a data error."""
    spec = trend_spec()
    first, end = ms("2023-09-01T00:00:00"), ms("2024-03-20T00:00:00")
    eth_first = end - listed_hours_before_end * HOUR_MS
    archive = write_archive(
        tmp_path / "a.db",
        {"BTCUSDT": hourly(first, end), "ETHUSDT": hourly(eth_first, end + DAY_MS)},
        {"BTCUSDT": first, "ETHUSDT": end},
    )
    use_snapshot(tmp_path, monkeypatch, archive)
    if refused:
        with pytest.raises(CarryDataError) as error:
            load_carry_inputs(spec, tmp_path / "no-catalog", archive)
        assert str(error.value).startswith("ETHUSDT: no funding settlement in [")
        return
    inputs = load_carry_inputs(spec, tmp_path / "no-catalog", archive)
    assert inputs.listings == {"BTCUSDT": first * 10**6, "ETHUSDT": eth_first * 10**6}
    assert inputs.settlements["ETHUSDT"].times_ns == ()
    assert inputs.settlements["BTCUSDT"].times_ns


def test_listed_after_end_is_not_listed_and_needs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mixed cohort runs: ETH's first bar is at end, so it is ``not_listed`` at every D,
    has no ticks or closes, and needs no funding row; BTC is loaded as usual."""
    spec = trend_spec()
    first = ms("2023-09-01T00:00:00")
    archive = write_archive(
        tmp_path / "a.db",
        {
            "BTCUSDT": hourly(first, ms("2024-03-20T00:00:00")),
            "ETHUSDT": hourly(ms("2024-03-20T00:00:00"), ms("2024-03-22T00:00:00")),
        },
        {"BTCUSDT": first},
    )
    use_snapshot(tmp_path, monkeypatch, archive)
    inputs = load_carry_inputs(spec, tmp_path / "no-catalog", archive)

    assert inputs.listings == {"BTCUSDT": first * 10**6, "ETHUSDT": spec.end_ns}
    assert inputs.ticks["ETHUSDT"] == ()
    assert inputs.trend_closes is not None
    assert inputs.trend_closes["ETHUSDT"] == {}
    assert len(inputs.ticks["BTCUSDT"]) == 2 * len(inputs.decisions)
    for decision_ns in inputs.decisions:
        sizing = {s: ticks_module.sizing_price(inputs.ticks[s], decision_ns) for s in spec.symbols}
        signals = decision_signals(
            spec,
            inputs.settlements,
            sizing,
            decision_ns,
            trend_closes=inputs.trend_closes,
            listings=inputs.listings,
        )
        assert signals.excluded["ETHUSDT"] == "not_listed"


def test_all_prelisting_is_refused_before_any_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C9-4: every symbol listed at/after end -> CarryNotListed (0 covered coins), no inputs."""
    spec = trend_spec()
    later = ms("2024-03-20T00:00:00")
    archive = write_archive(
        tmp_path / "a.db",
        {s: hourly(later, later + 2 * DAY_MS) for s in spec.symbols},
        dict.fromkeys(spec.symbols, later),
    )
    use_snapshot(tmp_path, monkeypatch, archive)
    with pytest.raises(CarryNotListed) as error:
        load_carry_inputs(spec, tmp_path / "no-catalog", archive)
    assert str(error.value) == (
        f"no symbol listed before end {spec.end_ns}: 0 covered coins "
        f"(BTCUSDT listed {later * 10**6}, ETHUSDT listed {later * 10**6})"
    )
    assert error.value.listings == dict.fromkeys(spec.symbols, later * 10**6)
    assert isinstance(error.value, CarryDataError)


# ---------------------------------------------------------------------------
# R43 / R44: archive only, warm-up from start - 161 days
# ---------------------------------------------------------------------------


def test_hour_run_never_reads_the_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R43: catalog readers raise if touched; the run loads from the archive alone."""
    import vibe_quant.data.mark_bars as mark_bars
    import vibe_quant.validation.extraction as extraction

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("catalog read in a 1h run")

    monkeypatch.setattr(extraction, "load_catalog_bars", boom)
    monkeypatch.setattr(mark_bars, "load_mark_series", boom)
    monkeypatch.setattr(ticks_module, "_catalog_instruments", boom)
    spec = trend_spec()
    first = ms("2023-09-01T00:00:00")
    archive = write_archive(
        tmp_path / "a.db",
        {s: hourly(first, ms("2024-03-20T00:00:00")) for s in spec.symbols},
        dict.fromkeys(spec.symbols, first),
    )
    snapshot = use_snapshot(tmp_path, monkeypatch, archive)
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    inputs = load_carry_inputs(spec, catalog, archive)
    assert inputs.price_source == "archive_1h"
    assert inputs.mark_source == "archive_1h_last_close"
    assert inputs.marks == {}
    assert inputs.instruments_source == {
        "path": "research_instruments.json",
        "sha256": load_snapshot(snapshot).sha256,
    }
    assert inputs.instruments["BTCUSDT"] == research_instrument(load_snapshot(snapshot), "BTCUSDT")
    assert inputs.specs["BTCUSDT"].taker_fee == inputs.instruments["BTCUSDT"].taker_fee


def test_warmup_reads_from_start_minus_161_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R44: the bar opening exactly start - 161 d is loaded, the one an hour earlier is not."""
    spec = trend_spec()
    warm = spec.start_ns // 10**6 - 161 * DAY_MS
    archive = write_archive(
        tmp_path / "a.db",
        {s: hourly(warm - 5 * HOUR_MS, ms("2024-03-20T00:00:00")) for s in spec.symbols},
        dict.fromkeys(spec.symbols, warm - 5 * HOUR_MS),
    )
    use_snapshot(tmp_path, monkeypatch, archive)
    inputs = load_carry_inputs(spec, tmp_path / "no-catalog", archive)
    assert inputs.hour_bars is not None
    assert int(inputs.hour_bars["BTCUSDT"].ts[0]) == (warm + HOUR_MS - 1) * 10**6
    assert inputs.listings is not None
    assert inputs.listings["BTCUSDT"] == (warm - 5 * HOUR_MS) * 10**6


# ---------------------------------------------------------------------------
# R21: first eligible decision of a coin listed mid-history
# ---------------------------------------------------------------------------


def test_first_eligible(tmp_path: Path) -> None:
    """Listing 2024-03-10T05:00 (first 1h bar): first close key 2024-03-11T00:00,
    D 2024-08-16T01:00 is ``no_history``, the first eligible D is 2024-08-17T01:00."""
    listing = ms("2024-03-10T05:00:00")
    end = ms("2024-08-20T00:00:00")
    archive = write_archive(
        tmp_path / "a.db", {SYMBOL: hourly(listing, end)}, {SYMBOL: listing + HOUR_MS}
    )
    bars = load_hour_bars(archive, SYMBOL, ns("2024-01-01T00:00:00"), end * 10**6, _increment())
    assert isinstance(bars, HourBars)
    from vibe_quant.data.catalog import create_instrument

    instrument = create_instrument(SYMBOL)
    boundaries = range(ns("2024-03-01T00:00:00"), end * 10**6 + DAY_NS, DAY_NS)
    closes = trend_closes(bars, boundaries, instrument)
    assert min(closes) == ns("2024-03-11T00:00:00")

    spec = trend_spec((SYMBOL,), "2024-08-15", "2024-08-20")
    eligible = []
    for decision_ns in decision_times(spec.start_ns, spec.end_ns, "daily"):
        sizing = {SYMBOL: ticks_module.PriceTick(decision_ns - 10**6, 100.0)}
        result = decision_signals(
            spec,
            {},
            sizing,
            decision_ns,
            trend_closes={SYMBOL: closes},
            listings={SYMBOL: bars.listing_ns},
        )
        eligible.append((decision_ns, result.excluded.get(SYMBOL)))
    assert eligible[:3] == [
        (ns("2024-08-15T01:00:00"), "no_history"),
        (ns("2024-08-16T01:00:00"), "no_history"),
        (ns("2024-08-17T01:00:00"), None),
    ]
    assert np.all(np.diff(bars.ts) > 0)
