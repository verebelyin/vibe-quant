"""Non-OHLCV context for ``needs_context`` indicators (funding, taker flow, regime).

Runners call :func:`configure` in-process before building the engine (no
StrategyConfig fields -- NT 1.226+ rejects unknown ones) and :func:`preflight`
with the run window. Generated strategies hand each ``needs_context``
``compute_fn`` a DataFrame whose ``attrs`` carry ``bar_close_ns`` (int64 close
time per row) and ``symbol`` (instrument id).

Funding comes from the archive via :class:`FundingCalculator`'s loader and its
module-level ``_RATE_CACHE`` (one cache: ``clear_rate_cache`` covers both). The
only derived state (snapped series, z/mean series) is validated against the
identity of the cached source tuple, so a cache clear invalidates it.

Taker flow reads the archived 1m klines once per (absolute archive path,
symbol) and keeps integer running sums (volumes scaled by 10^p, p <= 4 chosen
per symbol), so a window value is a difference of two int64s and one float
division -- independent of where the query series starts.

Positioning indicators read archived series once per (absolute archive path,
source, column): OI_CHANGE / LS_RATIO_Z the Binance Vision 5-minute ``raw_metrics``
rows, DVOL_Z the Deribit hourly ``raw_dvol`` candles. Each derives one value per ROW
on the full series (row i reads rows <= i only) and looks it up as-of each bar
close with a publication lag (``METRICS_LAG_NS`` / ``DVOL_LAG_NS``).

Causality: archived settlement times carry ms jitter (``...08:00:00.004``), so
each is snapped to the 8h boundary when it lies within +-60s
(``_SETTLEMENT_TOLERANCE_NS``); settlements further off a boundary (symbols on
4h/1h funding) keep their archived time and are NOT snapped. A bar sees every
settlement with ``snapped_ts <= bar_close``: the 08:00 bar sees the 08:00
settlement, a bar closing 07:59:59.999 sees 00:00.

Failure model: NT logs and survives exceptions raised inside ``on_bar``, which
would turn missing data into a silent zero-trade run. So failures are owned by
the run-level :func:`preflight` (raises :class:`AuxDataUnavailableError`); the
per-bar lookups never raise on window problems and return NaN instead --
before the first settlement, and when the last settlement is more than two
funding periods old (archive hole; stale rates are never forward-filled).
"""

from __future__ import annotations

import logging
import os
from bisect import bisect_left
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.errors import DataUnavailableError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import pandas as pd


logger = logging.getLogger(__name__)


class AuxDataUnavailableError(DataUnavailableError):
    """Aux data (funding archive) missing or not covering the run window."""


_REFRESH_HINT = "— refresh funding data (Data Management → update)"

# The archive may begin this long after the run start (z-score / first-bar warmup).
PREFLIGHT_START_MARGIN_NS = 3 * 24 * 3_600 * 1_000_000_000

_archive_path: Path | None = None
_catalog_path: Path | None = None

# (archive path, symbol) -> (source tuple from _RATE_CACHE, snapped ts, rates)
_SNAPPED: dict[tuple[str, str], tuple[object, np.ndarray, np.ndarray]] = {}
# (archive path, symbol, kind, period) -> (source tuple, derived series)
_DERIVED: dict[tuple[str, str, str, int], tuple[object, np.ndarray]] = {}
# (archive path, symbol) -> (times list from _RATE_CACHE, minute-rounded times, interval hours)
_GRIDS: dict[tuple[str, str], tuple[object, list[int], list[int]]] = {}
# (archive path, symbol) pairs already warned about falling back to 8h (cleared by configure)
_GRID_WARNED: set[tuple[str, str]] = set()

_HOUR_NS = 3_600 * 1_000_000_000
_MINUTE_NS = 60 * 1_000_000_000
_DEFAULT_INTERVAL_H = 8


def configure(archive_path: Path | str | None, catalog_path: Path | str | None = None) -> None:
    """Point aux-data lookups at the run's archive / catalog (None = default)."""
    global _archive_path, _catalog_path, _TAKER_HOT, _METRICS_HOT
    _TAKER_HOT = None
    _METRICS_HOT = None
    _archive_path = Path(archive_path) if archive_path else None
    _catalog_path = Path(catalog_path) if catalog_path else None
    # Reference bars are resolved once per run (preflight / first use), not per bar.
    _REF.clear()
    _REF_DERIVED.clear()
    _GRID_WARNED.clear()  # a new run may warn about a missing grid again


def archive_path() -> Path | None:
    """The configured funding archive path (None = default)."""
    return _archive_path


def _resolved_archive() -> Path:
    from vibe_quant.data.archive import DEFAULT_ARCHIVE_PATH

    return _archive_path or DEFAULT_ARCHIVE_PATH


def _source(symbol: str) -> tuple[Path, tuple[list[int], list[float]]]:
    from vibe_quant.validation.funding import FundingCalculator

    path = _resolved_archive()
    if not path.exists():
        msg = f"FUNDING indicators need the funding archive, but {path} does not exist"
        raise AuxDataUnavailableError(msg)
    return path, FundingCalculator(path)._load_rates(symbol)


def settlement_series(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """Snapped settlement times (ns, int64) and rates (float64) for ``symbol``.

    Raises:
        AuxDataUnavailableError: archive missing or without funding for ``symbol``.
    """
    from vibe_quant.validation import funding as f

    path, source = _source(symbol)
    key = (str(path), symbol)
    hit = _SNAPPED.get(key)
    if hit is not None and hit[0] is source:
        return hit[1], hit[2]
    times, rates = source
    if not times:
        msg = f"No funding rates archived for {symbol} in {path}; download funding data first"
        raise AuxDataUnavailableError(msg)
    ts = np.asarray(times, dtype=np.int64)
    period = f._FUNDING_PERIOD_NS
    boundary = ((ts + period // 2) // period) * period
    snapped = np.where(np.abs(ts - boundary) <= f._SETTLEMENT_TOLERANCE_NS, boundary, ts)
    r = np.asarray(rates, dtype=np.float64)
    _SNAPPED[key] = (source, snapped, r)
    return snapped, r


def _settlement_grid(path: Path, symbol: str) -> tuple[list[int], list[int]] | None:
    """(minute-rounded settlement times ns, interval hours) of ``symbol``, None if none archived.

    Cached per (archive, symbol) on the IDENTITY of the ``_RATE_CACHE`` times list,
    so ``clear_rate_cache`` invalidates it like the snapped series above.
    """
    from vibe_quant.validation.funding import FundingCalculator

    calc = FundingCalculator(path)
    times, _ = calc._load_rates(symbol)
    if not times:
        return None
    key = (str(path), symbol)
    hit = _GRIDS.get(key)
    if hit is not None and hit[0] is times:
        return hit[1], hit[2]
    rounded = [(t + _MINUTE_NS // 2) // _MINUTE_NS * _MINUTE_NS for t in times]
    hours = calc._load_intervals(symbol)
    _GRIDS[key] = (times, rounded, hours)
    return rounded, hours


def _log_grid_fallback_once(path: Path, symbol: str, why: str) -> None:
    key = (str(path), symbol)
    if key not in _GRID_WARNED:
        _GRID_WARNED.add(key)
        logger.warning(
            "%s: %s; funding avoidance uses the %dh grid (00:00, 08:00, 16:00 UTC)",
            symbol,
            why,
            _DEFAULT_INTERVAL_H,
        )


def is_funding_settlement(instrument_id: str, hour_ns: int) -> bool:
    """Whether the whole UTC hour ``hour_ns`` (ns) is a funding settlement of the symbol.

    ``S(s)`` iff ``s % (H(s) * 1h) == 0``, where ``H(s)`` is the archived settlement
    interval (hours) of the first row whose minute-rounded time is ``>= s``. That row is
    used only when it is at most one of its own intervals away (past the last row: the
    last row's). Across a wider archive hole the previous row's interval is used
    (causal: a rate archived later cannot change an earlier decision), and before the
    first row the 8h fallback applies. Archive file missing, symbol without rows, a
    pre-coverage hour or any lookup failure: the 8h rule, with one WARNING per
    (archive, symbol) until :func:`configure` runs again.

    Never raises: NT swallows exceptions inside ``on_bar``, which would turn missing
    data into a silent zero-trade run (see the module docstring).
    """
    path = _resolved_archive()
    symbol = instrument_id
    interval = _DEFAULT_INTERVAL_H
    try:
        from vibe_quant.validation.funding import FundingCalculator

        symbol = FundingCalculator.symbol_from_instrument_id(instrument_id)
        if not path.exists():
            _log_grid_fallback_once(path, symbol, f"funding archive {path} not found")
        else:
            grid = _settlement_grid(path, symbol)
            if grid is None:
                _log_grid_fallback_once(path, symbol, f"no funding intervals archived in {path}")
            else:
                rounded, hours = grid
                i = bisect_left(rounded, hour_ns)
                if i == 0 and hour_ns < rounded[0]:
                    _log_grid_fallback_once(
                        path, symbol, f"no funding settlement archived at or before {_iso(hour_ns)}"
                    )
                elif i == len(rounded):
                    interval = hours[-1]
                elif rounded[i] - hour_ns > hours[i] * _HOUR_NS:
                    interval = hours[i - 1]
                else:
                    interval = hours[i]
        return hour_ns % (interval * _HOUR_NS) == 0
    except Exception as e:
        _log_grid_fallback_once(
            path, symbol, f"funding interval lookup failed ({type(e).__name__})"
        )
        return hour_ns % (_DEFAULT_INTERVAL_H * _HOUR_NS) == 0


def _date_ns(value: str) -> int:
    return int(datetime.fromisoformat(value[:10]).replace(tzinfo=UTC).timestamp()) * 1_000_000_000


def preflight(
    indicator_types: list[str], symbols: list[str], start_date: str, end_date: str
) -> None:
    """Fail before the engine starts if context indicators lack aux data for the window.

    Funding: raises :class:`AuxDataUnavailableError` when the archive is missing,
    has no funding for a symbol, its first settlement is more than
    ``PREFLIGHT_START_MARGIN_NS`` (3 days, warmup allowance) after ``start_date``,
    or its last settlement is before ``end_date - 16h``.

    Taker flow: raises when the 1m klines do not cover
    ``[start_date - 1 day, end_date]`` or a kline in that range has a NULL
    ``taker_buy_volume``.

    Positioning metrics (OI_CHANGE / LS_RATIO_Z): raises when the archive has no
    metrics table or no non-NULL rows of the column for a symbol, its first row is
    more than ``PREFLIGHT_START_MARGIN_NS`` after ``start_date``, or its last row is
    before ``end_date - 1 day``. DVOL_Z: the same checks on the hourly DVOL candles of
    each symbol's currency (``dvol_currency``). Interior holes are NaN at lookup time.
    """
    from vibe_quant.dsl.indicators import indicator_registry
    from vibe_quant.validation.funding import _FUNDING_PERIOD_NS

    start_ns, end_ns = _date_ns(start_date), _date_ns(end_date)
    _preflight_regime(indicator_types, symbols, start_ns, end_ns, start_date, end_date)
    _preflight_taker(indicator_types, symbols, start_ns, end_ns, start_date, end_date)
    _preflight_metrics(indicator_types, symbols, start_ns, end_ns, start_date, end_date)
    _preflight_dvol(indicator_types, symbols, start_ns, end_ns, start_date, end_date)
    if not any(
        (spec := indicator_registry.get(t)) is not None
        and spec.needs_context
        and t not in REGIME_TYPES
        and t not in TAKER_TYPES
        and t not in POSITIONING_TYPES
        for t in indicator_types
    ):
        return
    for symbol in symbols:
        snapped, _ = settlement_series(symbol)
        if int(snapped[0]) > start_ns + PREFLIGHT_START_MARGIN_NS:
            msg = (
                f"Funding archive for {symbol} starts after the requested window "
                f"(first settlement {_iso(int(snapped[0]))}, window start {start_date}) "
                f"{_REFRESH_HINT}"
            )
            raise AuxDataUnavailableError(msg)
        if int(snapped[-1]) < end_ns - 2 * _FUNDING_PERIOD_NS:
            msg = (
                f"Funding archive for {symbol} ends before the requested window "
                f"(last settlement {_iso(int(snapped[-1]))}, window end {end_date}) "
                f"{_REFRESH_HINT}"
            )
            raise AuxDataUnavailableError(msg)


def _preflight_taker(
    indicator_types: list[str],
    symbols: list[str],
    start_ns: int,
    end_ns: int,
    start_date: str,
    end_date: str,
) -> None:
    if not TAKER_TYPES.intersection(indicator_types):
        return
    lo_ns = start_ns - _TAKER_PREFLIGHT_MARGIN_NS
    for symbol in symbols:
        close_ns, _, _, present_cum = _taker_klines(symbol, revalidate=True)
        if int(close_ns[0]) > lo_ns:
            msg = (
                f"1m klines for {symbol} start after the requested window "
                f"(first kline close {_iso(int(close_ns[0]))}, needed from {_iso(lo_ns)} "
                f"for window start {start_date}) — download {symbol} 1m data"
            )
            raise AuxDataUnavailableError(msg)
        if int(close_ns[-1]) < end_ns:
            msg = (
                f"1m klines for {symbol} end before the requested window "
                f"(last kline close {_iso(int(close_ns[-1]))}, window end {end_date}) "
                f"— download {symbol} 1m data"
            )
            raise AuxDataUnavailableError(msg)
        lo = int(np.searchsorted(close_ns, lo_ns, side="left"))
        hi = int(np.searchsorted(close_ns, end_ns, side="right"))
        if int(present_cum[hi] - present_cum[lo]) != hi - lo:
            msg = (
                f"1m klines for {symbol} have NULL taker_buy_volume inside the "
                f"requested window — re-download {symbol} 1m data"
            )
            raise AuxDataUnavailableError(msg)


def _iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).strftime("%Y-%m-%d %H:%M")


def _asof(snapped: np.ndarray, series: np.ndarray, bar_close_ns: np.ndarray) -> np.ndarray:
    """As-of lookup; NaN before the first settlement or when it is > 2 periods stale."""
    from vibe_quant.validation.funding import _FUNDING_PERIOD_NS

    close = np.asarray(bar_close_ns, dtype=np.int64)
    idx = np.searchsorted(snapped, close, side="right") - 1
    safe = np.clip(idx, 0, None)
    ok = (idx >= 0) & (close - snapped[safe] <= 2 * _FUNDING_PERIOD_NS)
    return np.where(ok, series[safe], np.nan)


def _derived(symbol: str, kind: str, period: int) -> np.ndarray | None:
    path, source = _source(symbol)
    hit = _DERIVED.get((str(path), symbol, kind, period))
    if hit is not None and hit[0] is source:
        return hit[1]
    return None


def _store(symbol: str, kind: str, period: int, arr: np.ndarray) -> np.ndarray:
    path, source = _source(symbol)
    _DERIVED[(str(path), symbol, kind, period)] = (source, arr)
    return arr


def funding_asof(symbol: str, bar_close_ns: np.ndarray, period: int = 1) -> np.ndarray:
    """Last settled funding rate (fraction) as of each bar close.

    ``period > 1`` averages the last ``period`` settlements (NaN until that many exist).
    NaN before the first settlement and across archive holes (see module docstring).
    """
    snapped, rates = settlement_series(symbol)
    series = rates
    if period > 1:
        cached = _derived(symbol, "mean", period)
        if cached is None:
            series = np.full(rates.size, np.nan)
            if rates.size >= period:
                from numpy.lib.stride_tricks import sliding_window_view

                series[period - 1 :] = sliding_window_view(rates, period).mean(axis=1)
            cached = _store(symbol, "mean", period, series)
        series = cached
    return _asof(snapped, series, bar_close_ns)


def funding_z_asof(symbol: str, bar_close_ns: np.ndarray, period: int) -> np.ndarray:
    """Z-score of the rate over the last ``period`` SETTLEMENTS, as of each bar close.

    Computed causally on the full settlement series (ddof=1, trailing window
    including the current settlement), cached per (archive, symbol, period),
    then looked up as-of -- never over the bar buffer. NaN until ``period``
    settlements exist or when the window is flat.
    """
    snapped, rates = settlement_series(symbol)
    z = _derived(symbol, "z", period)
    if z is None:
        z = np.full(rates.size, np.nan)
        if period >= 2 and rates.size >= period:
            from numpy.lib.stride_tricks import sliding_window_view

            win = sliding_window_view(rates, period)
            mean = win.mean(axis=1)
            std = win.std(axis=1, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                z[period - 1 :] = np.where(std > 1e-15, (rates[period - 1 :] - mean) / std, np.nan)
        z = _store(symbol, "z", period, z)
    return _asof(snapped, z, bar_close_ns)


# -- taker flow (TAKER_IMBALANCE) ------------------------------------------------------

TAKER_TYPES = frozenset({"TAKER_IMBALANCE"})
_TAKER_KLINE_NS = 60 * 1_000_000_000
# A period-N window needs N minutes of history; one day of slack covers the
# largest GA period (240m) with room for a partial first bar.
_TAKER_PREFLIGHT_MARGIN_NS = 86_400 * 1_000_000_000
_TAKER_MAX_SCALE = 4
# A scaled value must stay an exactly representable float integer (< 2^53) and the
# running sum of one column must stay far inside int64 (the float sum used to
# check it may be off by a few ulps, hence 2^62 rather than 2^63).
_TAKER_MAX_SCALED = 2.0**53
_TAKER_MAX_SUM = 2.0**62

# (absolute archive path, symbol) -> (source stamp, entry); entry =
#   (kline close ns, volume cumsum, taker cumsum, non-NULL-taker cumsum)
_TakerEntry = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
_TAKER_KLINES: dict[tuple[str, str], tuple[tuple[object, object], _TakerEntry]] = {}


def _taker_archive_path() -> Path:
    return Path(os.path.abspath(_resolved_archive()))  # noqa: PTH100


def _stamp(path: Path) -> tuple[int, int] | None:
    if not os.path.lexists(path):  # noqa: PTH110
        return None
    st = os.stat(path)  # noqa: PTH116
    return st.st_size, st.st_mtime_ns


def _taker_source_stamp(path: Path) -> tuple[object, object]:
    """(size, mtime_ns) of the archive file and of its ``-wal`` file (None if absent)."""
    return _stamp(path), _stamp(path.with_name(path.name + "-wal"))


def _scaled_exact(values: np.ndarray, p: int) -> np.ndarray | None:
    """``values * 10^p`` as exact integers, or None unless EVERY value is ``n / 10^p``.

    A float ``v`` is the decimal ``n / 10^p`` iff ``float(round(v * 10^p)) / 10^p``
    gives back ``v`` (division by an exact power of ten is correctly rounded), so
    float representation error passes and a real extra decimal digit never does.
    """
    scale = 10.0**p
    scaled = values * scale
    if float(np.max(np.abs(scaled))) >= _TAKER_MAX_SCALED:
        return None
    rounded = np.rint(scaled)
    if not np.array_equal(rounded / scale, values):
        return None
    if float(np.sum(np.abs(rounded))) >= _TAKER_MAX_SUM:
        return None
    return rounded


def _taker_scale(
    volumes: np.ndarray, takers: np.ndarray, symbol: str, path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Integer volume / taker series at the smallest ``p`` in 0..4 that represents BOTH.

    One scale serves both columns, so the window ratio is a ratio of two integers
    scaled alike. Any ``p`` that represents the rows yields the same ratio (the
    integers differ only by an exact factor 10^k), so a value does not depend on
    which other rows were loaded, as long as scaling is exact.
    """
    for name, col in (("volume", volumes), ("taker_buy_volume", takers)):
        if not bool(np.all(np.isfinite(col))):
            msg = f"1m {name} for {symbol} in {path} is not finite — archive malformed"
            raise AuxDataUnavailableError(msg)
    for p in range(_TAKER_MAX_SCALE + 1):
        vol_int = _scaled_exact(volumes, p)
        if vol_int is None:
            continue
        taker_int = _scaled_exact(takers, p)
        if taker_int is not None:
            return vol_int.astype(np.int64), taker_int.astype(np.int64)
    msg = (
        f"1m volume / taker_buy_volume for {symbol} in {path} is not integral at any "
        f"scale up to 10^{_TAKER_MAX_SCALE} (or exceeds the int64 range) — archive malformed"
    )
    raise AuxDataUnavailableError(msg)


def _taker_klines(symbol: str, *, revalidate: bool = False) -> _TakerEntry:
    """Cached integer kline series of ``symbol`` (close ns + running sums).

    ``revalidate`` (preflight, i.e. every launch check) re-stats the archive and
    its ``-wal`` and reloads when they changed, so a refreshed archive is never
    served from a stale cache. Per-bar lookups skip the stat (~7 us per call)
    and reuse the entry loaded for this process.
    """
    global _TAKER_HOT
    path = _taker_archive_path()
    if revalidate:
        _TAKER_HOT = None
    else:
        cached = _TAKER_KLINES.get((str(path), symbol))
        if cached is not None:
            return cached[1]
    stamp = _taker_source_stamp(path)
    if stamp[0] is None or not path.is_file():
        msg = f"TAKER_IMBALANCE indicators need the 1m kline archive, but {path} is not a file"
        raise AuxDataUnavailableError(msg)
    key = (str(path), symbol)
    hit = _TAKER_KLINES.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]

    import sqlite3
    from array import array

    open_ms = array("q")
    volumes = array("d")
    takers = array("d")
    taker_present = array("b")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        for t, volume, taker in conn.execute(
            "SELECT open_time, volume, taker_buy_volume FROM raw_klines "
            "WHERE symbol = ? AND interval = '1m' ORDER BY open_time",
            (symbol,),
        ):
            open_ms.append(int(t))
            volumes.append(float(volume))
            taker_present.append(0 if taker is None else 1)
            takers.append(0.0 if taker is None else float(taker))
    finally:
        conn.close()
    if not open_ms:
        msg = f"No 1m klines archived for {symbol} in {path} — download {symbol} 1m data"
        raise AuxDataUnavailableError(msg)

    vol_int, taker_int = _taker_scale(
        np.asarray(volumes, dtype=np.float64), np.asarray(takers, dtype=np.float64), symbol, path
    )
    close_ns = np.asarray(open_ms, dtype=np.int64) * 1_000_000 + _TAKER_KLINE_NS
    zero = np.zeros(1, dtype=np.int64)
    entry = (
        close_ns,
        np.concatenate((zero, np.cumsum(vol_int, dtype=np.int64))),
        np.concatenate((zero, np.cumsum(taker_int, dtype=np.int64))),
        np.concatenate((zero, np.cumsum(np.asarray(taker_present, dtype=np.int64)))),
    )
    _TAKER_KLINES[key] = (stamp, entry)
    return entry


# (configured archive, default archive, resolved path) of the run in progress. The path is
# resolved (abspath, i.e. against the cwd) once per run: configure() and every launch check
# (``_taker_klines(revalidate=True)``) drop it, and a patched DEFAULT_ARCHIVE_PATH or a new
# configured path object misses it. A chdir in the middle of a run is not followed.
_TAKER_HOT: tuple[Path | None, Path, str] | None = None


def _taker_entry_hot(symbol: str) -> _TakerEntry:
    """Per-bar form of ``_taker_klines(symbol)``: no ``getcwd``/``abspath`` per bar.

    Returns the entry the cold path returns (it reads the same ``_TAKER_KLINES`` dict, so a
    preflight reload is seen); anything not already cached takes the cold path.
    """
    global _TAKER_HOT
    from vibe_quant.data.archive import DEFAULT_ARCHIVE_PATH

    hot = _TAKER_HOT
    if hot is not None and hot[0] is _archive_path and hot[1] is DEFAULT_ARCHIVE_PATH:
        cached = _TAKER_KLINES.get((hot[2], symbol))
        if cached is not None:
            return cached[1]
    entry = _taker_klines(symbol)
    _TAKER_HOT = (_archive_path, DEFAULT_ARCHIVE_PATH, str(_taker_archive_path()))
    return entry


def taker_imbalance_at(symbol: str, close_ns: int, period: int) -> float:
    """:func:`taker_imbalance_asof` for ONE bar close, without array or DataFrame allocation.

    Bit-equal to the vector form: the same integer window sums and the same single
    float division, NaN under the same conditions (see the vector form's docstring).
    """
    close_arr, vol_cum, taker_cum, present_cum = _taker_entry_hot(symbol)
    hi = int(close_arr.searchsorted(close_ns, side="right")) - 1
    lo = hi - period + 1
    if hi < 0 or lo < 0:
        return float("nan")
    safe_lo = min(lo, close_arr.size - 1)
    if close_arr[hi] != close_ns or close_arr[safe_lo] != close_ns - (period - 1) * _TAKER_KLINE_NS:
        return float("nan")
    if present_cum[hi + 1] - present_cum[safe_lo] != period:
        return float("nan")
    vol_sum = vol_cum[hi + 1] - vol_cum[safe_lo]
    if vol_sum <= 0:
        return float("nan")
    return float(taker_cum[hi + 1] - taker_cum[safe_lo]) / float(vol_sum) - 0.5


def taker_imbalance_asof(symbol: str, bar_close_ns: np.ndarray, period: int) -> np.ndarray:
    """Taker-buy share of volume minus 0.5 over the last ``period`` MINUTES.

    The window is ``(t - period*60s, t]`` of archived 1m klines; a kline is
    closed at ``open_time + 60s``, so a kline opening at ``t`` is excluded.
    NaN unless exactly ``period`` consecutive klines end at ``t``, every
    ``taker_buy_volume`` in them is non-NULL, and their volume sum is > 0.
    Never carried forward: every bar is computed from its own window.
    """
    close_ns, vol_cum, taker_cum, present_cum = _taker_klines(symbol)
    t = np.asarray(bar_close_ns, dtype=np.int64)
    hi = np.searchsorted(close_ns, t, side="right") - 1
    lo = hi - period + 1
    safe_hi = np.clip(hi, 0, close_ns.size - 1)
    safe_lo = np.clip(lo, 0, close_ns.size - 1)
    span = (period - 1) * _TAKER_KLINE_NS
    ok = (hi >= 0) & (close_ns[safe_hi] == t) & (lo >= 0) & (close_ns[safe_lo] == t - span)
    vol_sum = vol_cum[safe_hi + 1] - vol_cum[safe_lo]
    taker_sum = taker_cum[safe_hi + 1] - taker_cum[safe_lo]
    ok &= present_cum[safe_hi + 1] - present_cum[safe_lo] == period
    ok &= vol_sum > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        value = taker_sum / vol_sum - 0.5  # one float division per window
    return np.where(ok, value, np.nan)


# -- positioning: Binance Vision metrics (OI_CHANGE / LS_RATIO_Z), Deribit DVOL (DVOL_Z) --

METRICS_TYPES = frozenset({"LS_RATIO_Z", "OI_CHANGE"})
DVOL_TYPES = frozenset({"DVOL_Z"})
# The one rule bankroll and shadow refuse by (their snapshots have no metrics or DVOL).
POSITIONING_TYPES = METRICS_TYPES | DVOL_TYPES
METRICS_STEP_NS = 300 * 10**9
# A row stamped ``s`` is published 7-174 s after ``s`` (MEASURED 2026-11-27); a bar may use
# it from close ``s + METRICS_LAG_NS`` and while the close is less than
# ``METRICS_MAX_STALE_NS`` past that (one missing newest row is bridged, two are not).
METRICS_LAG_NS = 600 * 10**9
METRICS_MAX_STALE_NS = 600 * 10**9
# An hourly DVOL candle opened at ``o`` is complete at ``o + 1 h`` (the downloader stores
# only finished candles); usable from close ``o + DVOL_LAG_NS`` while the close is less
# than ``DVOL_MAX_STALE_NS`` past that (one missing candle is bridged, two are not).
DVOL_STEP_NS = 3600 * 10**9
DVOL_LAG_NS = 3600 * 10**9
DVOL_MAX_STALE_NS = 7200 * 10**9
_POSITIONING_END_MARGIN_NS = 86_400 * 10**9
_OI_COLUMN = "sum_open_interest"
_LS_COLUMN = "sum_toptrader_long_short_ratio"
_METRICS_COLUMNS = {"OI_CHANGE": _OI_COLUMN, "LS_RATIO_Z": _LS_COLUMN}
_DVOL_COLUMN = "dvol_close"  # cache tag of raw_dvol.close (never a metrics column name)

# (absolute archive path, symbol or currency, column) -> (source stamp, (stamp ns, values));
# NULL fields are left out of the column's series (holes).
_MetricsEntry = tuple[np.ndarray, np.ndarray]
_METRICS: dict[tuple[str, str, str], tuple[tuple[object, object], _MetricsEntry]] = {}
# (symbol or currency, column, kind, period) -> (source entry identity, value per row),
# least recently used first. Bounded by bytes, not entries: a GA run draws a new period per
# genome (277 LS_RATIO_Z periods at BTC size held 0.87 GiB, MEASURED), while one run reads
# symbols x indicators keys every bar, which an entry cap below that count re-derives per
# read. 1 GiB holds ~200 BTC-size arrays; the newest entry stays even when it alone is over.
DERIVED_CACHE_BYTES = 1 << 30
_METRICS_DERIVED: dict[tuple[str, str, str, int], tuple[object, np.ndarray]] = {}
# Per-run resolved archive path, like ``_TAKER_HOT`` (configure() and preflight drop it).
_METRICS_HOT: tuple[Path | None, Path, str] | None = None


def positioning_types(types: Iterable[str]) -> list[str]:
    """The positioning indicator types among ``types``, sorted (bankroll/shadow refuse them)."""
    return sorted(POSITIONING_TYPES.intersection(types))


def dvol_currency(symbol: str) -> str:
    """Deribit DVOL index a symbol's DVOL_Z reads: ``ETH`` for ETHUSDT, else ``BTC``."""
    return "ETH" if symbol == "ETHUSDT" else "BTC"


def _no_metrics(symbol: str, path: Path) -> AuxDataUnavailableError:
    return AuxDataUnavailableError(
        f"No Binance metrics archived for {symbol} in {path} — "
        f"python -m vibe_quant.data download-metrics --symbols {symbol}"
    )


def _no_dvol(currency: str, path: Path) -> AuxDataUnavailableError:
    return AuxDataUnavailableError(
        f"No Deribit {currency} DVOL archived in {path} — "
        f"python -m vibe_quant.data download-dvol --currencies {currency}"
    )


def _archive_series(
    source: tuple[str, str],
    sql: str,
    missing: Callable[[Path], AuxDataUnavailableError],
    revalidate: bool,
) -> _MetricsEntry:
    """(stamp ns int64, value float64) rows of ``sql`` (ms stamp, value; arg ``source[0]``).

    Read-only; cached per (absolute archive path, *source). ``revalidate`` (preflight)
    re-stats the archive and its ``-wal`` and reloads when they changed; per-bar lookups
    reuse the entry loaded for this process (see ``_taker_klines``). ``missing`` builds
    the error for a missing archive file or table, or no rows.
    """
    global _METRICS_HOT
    path = _taker_archive_path()
    key = (str(path), *source)
    if revalidate:
        _METRICS_HOT = None
    else:
        cached = _METRICS.get(key)
        if cached is not None:
            return cached[1]
    stamp = _taker_source_stamp(path)
    if stamp[0] is None or not path.is_file():
        raise missing(path)
    hit = _METRICS.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]

    import sqlite3

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(sql, (source[0],)).fetchall()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            raise missing(path) from None
        raise
    finally:
        conn.close()
    if not rows:
        raise missing(path)
    entry = (
        np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows)) * 1_000_000,
        np.fromiter((r[1] for r in rows), dtype=np.float64, count=len(rows)),
    )
    _METRICS[key] = (stamp, entry)
    return entry


def metrics_series(
    symbol: str, column: str, *, revalidate: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """(create_time ns int64, value float64) of ``symbol``'s non-NULL ``column`` rows, ascending.

    Cached and revalidated like :func:`_archive_series`.

    Raises:
        AuxDataUnavailableError: archive file or metrics table missing, or no rows.
    """
    if column not in _METRICS_COLUMNS.values():
        msg = f"Unknown metrics column {column!r}"
        raise ValueError(msg)
    sql = (
        f"SELECT create_time, {column} FROM raw_metrics "  # noqa: S608  (whitelisted column)
        f"WHERE symbol = ? AND {column} IS NOT NULL ORDER BY create_time"
    )
    return _archive_series((symbol, column), sql, lambda p: _no_metrics(symbol, p), revalidate)


def dvol_series(currency: str, *, revalidate: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(open_time ns int64, close float64) of ``currency``'s hourly DVOL candles, ascending.

    Cached and revalidated like :func:`_archive_series`; NULL closes are holes.

    Raises:
        AuxDataUnavailableError: archive file or DVOL table missing, or no candles.
    """
    sql = (
        "SELECT open_time, close FROM raw_dvol WHERE currency = ? AND resolution_s = 3600 "
        "AND close IS NOT NULL ORDER BY open_time"
    )
    return _archive_series(
        (currency, _DVOL_COLUMN), sql, lambda p: _no_dvol(currency, p), revalidate
    )


def _load_series(source: tuple[str, str]) -> _MetricsEntry:
    if source[1] == _DVOL_COLUMN:
        return dvol_series(source[0])
    return metrics_series(*source)


def _series_hot(source: tuple[str, str]) -> _MetricsEntry:
    """Per-bar form of ``_load_series(source)``: no ``getcwd``/``abspath`` per bar."""
    global _METRICS_HOT
    from vibe_quant.data.archive import DEFAULT_ARCHIVE_PATH

    hot = _METRICS_HOT
    if hot is not None and hot[0] is _archive_path and hot[1] is DEFAULT_ARCHIVE_PATH:
        cached = _METRICS.get((hot[2], *source))
        if cached is not None:
            return cached[1]
    entry = _load_series(source)
    _METRICS_HOT = (_archive_path, DEFAULT_ARCHIVE_PATH, str(_taker_archive_path()))
    return entry


def _asof_index(
    stamp_ns: np.ndarray, close_ns: np.ndarray, lag_ns: int, max_stale_ns: int
) -> np.ndarray:
    """Row usable at each close (-1 = none): the last ``stamp <= close - lag``, kept only
    while ``close - (stamp + lag) < max_stale`` (integer ns)."""
    close = np.asarray(close_ns, dtype=np.int64)
    idx = np.searchsorted(stamp_ns, close - lag_ns, side="right") - 1
    safe = np.clip(idx, 0, None)
    ok = (idx >= 0) & (close - (stamp_ns[safe] + lag_ns) < max_stale_ns)
    return np.where(ok, idx, -1)


def _asof_index_at(stamp_ns: np.ndarray, close_ns: int, lag_ns: int, max_stale_ns: int) -> int:
    """:func:`_asof_index` for ONE close."""
    i = int(stamp_ns.searchsorted(close_ns - lag_ns, side="right")) - 1
    if i >= 0 and close_ns - (int(stamp_ns[i]) + lag_ns) < max_stale_ns:
        return i
    return -1


def _oi_change_rows(ts: np.ndarray, x: np.ndarray, period: int) -> np.ndarray:
    """``x[i] / x[j] - 1`` where row j is stamped exactly ``ts[i] - period*STEP`` (endpoint
    rule: holes between the endpoints do not matter); NaN if j is absent or ``x[j] <= 0``."""
    out = np.full(ts.size, np.nan)
    if period < 1:
        return out  # a non-positive period would read the current or a future row
    target = ts - period * METRICS_STEP_NS
    j = np.searchsorted(ts, target, side="left")
    safe = np.clip(j, 0, ts.size - 1)
    ok = (ts[safe] == target) & (x[safe] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, x / x[safe] - 1.0, out)


# Window elements per ``_z_rows`` chunk: bounds the std temporaries to ~32 MiB each
# (unchunked, one LS_RATIO_Z(288) at 650k rows peaked +1.42 GiB, MEASURED).
_Z_CHUNK_ELEMS = 1 << 22


def _z_rows(
    ts: np.ndarray, y: np.ndarray, period: int, step_ns: int = METRICS_STEP_NS
) -> np.ndarray:
    """``(w[-1] - mean(w)) / std(w, ddof=1)`` over the ``period`` rows ending at each row,
    defined only when they are contiguous (``ts[i] - ts[i-p+1] == (p-1)*step``, no holes)
    and ``std > 1e-15``."""
    out = np.full(ts.size, np.nan)
    if period < 2 or ts.size < period:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(y, period)
    contiguous = ts[period - 1 :] - ts[: ts.size - period + 1] == (period - 1) * step_ns
    last = y[period - 1 :]
    rows = max(1, _Z_CHUNK_ELEMS // period)  # win.std makes chunk x period temporaries
    for a in range(0, win.shape[0], rows):
        b = a + rows
        mean = win[a:b].mean(axis=1)
        std = win[a:b].std(axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[period - 1 + a : period - 1 + b] = np.where(
                contiguous[a:b] & (std > 1e-15), (last[a:b] - mean) / std, np.nan
            )
    return out


def _dvol_z_rows(ts: np.ndarray, y: np.ndarray, period: int) -> np.ndarray:
    return _z_rows(ts, y, period, DVOL_STEP_NS)


# kind -> (column, per-row derivation, lag, max stale past usable)
_KINDS: dict[str, tuple[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray], int, int]] = {
    "oi_change": (_OI_COLUMN, _oi_change_rows, METRICS_LAG_NS, METRICS_MAX_STALE_NS),
    "ls_z": (_LS_COLUMN, _z_rows, METRICS_LAG_NS, METRICS_MAX_STALE_NS),
    "dvol_z": (_DVOL_COLUMN, _dvol_z_rows, DVOL_LAG_NS, DVOL_MAX_STALE_NS),
}


def _source_of(symbol: str, column: str) -> tuple[str, str]:
    """Cache source of ``column`` for ``symbol``: DVOL is per currency, metrics per symbol."""
    return (dvol_currency(symbol), column) if column == _DVOL_COLUMN else (symbol, column)


def _positioning_derived(
    entry: _MetricsEntry, source: tuple[str, str], kind: str, period: int
) -> np.ndarray:
    """Per-row value of ``kind`` on the FULL archived series (row i reads rows <= i only).

    LRU over ``DERIVED_CACHE_BYTES`` of arrays: a hit moves to the newest end, the oldest
    go until the rest fit.
    """
    key = (*source, kind, period)
    hit = _METRICS_DERIVED.pop(key, None)
    if hit is not None and hit[0] is entry:
        _METRICS_DERIVED[key] = hit
        return hit[1]
    derived = _KINDS[kind][1](entry[0], entry[1], period)
    _METRICS_DERIVED[key] = (entry, derived)
    total = sum(v[1].nbytes for v in _METRICS_DERIVED.values())
    while total > DERIVED_CACHE_BYTES and len(_METRICS_DERIVED) > 1:
        total -= _METRICS_DERIVED.pop(next(iter(_METRICS_DERIVED)))[1].nbytes
    return derived


def _positioning_asof(symbol: str, kind: str, bar_close_ns: np.ndarray, period: int) -> np.ndarray:
    column, _, lag_ns, max_stale_ns = _KINDS[kind]
    source = _source_of(symbol, column)
    entry = _load_series(source)
    derived = _positioning_derived(entry, source, kind, period)
    idx = _asof_index(entry[0], bar_close_ns, lag_ns, max_stale_ns)
    return np.where(idx >= 0, derived[np.clip(idx, 0, None)], np.nan)


def _positioning_at(symbol: str, kind: str, close_ns: int, period: int) -> float:
    column, _, lag_ns, max_stale_ns = _KINDS[kind]
    source = _source_of(symbol, column)
    entry = _series_hot(source)
    i = _asof_index_at(entry[0], close_ns, lag_ns, max_stale_ns)
    if i < 0:
        return float("nan")
    return float(_positioning_derived(entry, source, kind, period)[i])


def oi_change_asof(symbol: str, bar_close_ns: np.ndarray, period: int) -> np.ndarray:
    """Open-interest change (contracts) over ``period`` 5-minute ROWS, as of each bar close.

    The as-of row i is the last one usable at the close (see ``METRICS_LAG_NS``); the
    value is ``x[i] / x[j] - 1`` with j the row stamped ``ts[i] - period * 5m``. NaN
    when no row is usable, j is absent or ``x[j] <= 0``.
    """
    return _positioning_asof(symbol, "oi_change", bar_close_ns, period)


def oi_change_at(symbol: str, close_ns: int, period: int) -> float:
    """:func:`oi_change_asof` for ONE bar close (bit-equal, no array allocation)."""
    return _positioning_at(symbol, "oi_change", close_ns, period)


def ls_ratio_z_asof(symbol: str, bar_close_ns: np.ndarray, period: int) -> np.ndarray:
    """Z-score of the top-trader long/short position ratio over ``period`` contiguous
    5-minute ROWS ending at the as-of row (strict: a hole in the window is NaN)."""
    return _positioning_asof(symbol, "ls_z", bar_close_ns, period)


def ls_ratio_z_at(symbol: str, close_ns: int, period: int) -> float:
    """:func:`ls_ratio_z_asof` for ONE bar close (bit-equal, no array allocation)."""
    return _positioning_at(symbol, "ls_z", close_ns, period)


def dvol_z_asof(symbol: str, bar_close_ns: np.ndarray, period: int) -> np.ndarray:
    """Z-score of the Deribit DVOL close (``dvol_currency(symbol)``) over ``period``
    contiguous hourly candles ending at the as-of candle (see ``DVOL_LAG_NS``; strict:
    a missing candle in the window is NaN)."""
    return _positioning_asof(symbol, "dvol_z", bar_close_ns, period)


def dvol_z_at(symbol: str, close_ns: int, period: int) -> float:
    """:func:`dvol_z_asof` for ONE bar close (bit-equal, no array allocation)."""
    return _positioning_at(symbol, "dvol_z", close_ns, period)


def _preflight_metrics(
    indicator_types: list[str],
    symbols: list[str],
    start_ns: int,
    end_ns: int,
    start_date: str,
    end_date: str,
) -> None:
    columns = sorted({_METRICS_COLUMNS[t] for t in METRICS_TYPES.intersection(indicator_types)})
    for symbol in symbols:
        for column in columns:
            ts, _ = metrics_series(symbol, column, revalidate=True)
            if int(ts[0]) > start_ns + PREFLIGHT_START_MARGIN_NS:
                msg = (
                    f"Binance metrics for {symbol} start after the requested window "
                    f"(first row {_iso(int(ts[0]))}, window start {start_date}) — download-metrics"
                )
                raise AuxDataUnavailableError(msg)
            if int(ts[-1]) < end_ns - _POSITIONING_END_MARGIN_NS:
                msg = (
                    f"Binance metrics for {symbol} end before the requested window "
                    f"(last row {_iso(int(ts[-1]))}, window end {end_date}) — download-metrics"
                )
                raise AuxDataUnavailableError(msg)


def _preflight_dvol(
    indicator_types: list[str],
    symbols: list[str],
    start_ns: int,
    end_ns: int,
    start_date: str,
    end_date: str,
) -> None:
    if not DVOL_TYPES.intersection(indicator_types):
        return
    for currency in sorted({dvol_currency(s) for s in symbols}):
        ts, _ = dvol_series(currency, revalidate=True)
        if int(ts[0]) > start_ns + PREFLIGHT_START_MARGIN_NS:
            msg = (
                f"Deribit {currency} DVOL starts after the requested window "
                f"(first candle {_iso(int(ts[0]))}, window start {start_date}) — download-dvol"
            )
            raise AuxDataUnavailableError(msg)
        if int(ts[-1]) < end_ns - _POSITIONING_END_MARGIN_NS:
            msg = (
                f"Deribit {currency} DVOL ends before the requested window "
                f"(last candle {_iso(int(ts[-1]))}, window end {end_date}) — download-dvol"
            )
            raise AuxDataUnavailableError(msg)


# -- cross-asset regime references (BTC_TREND / BTC_ROC / OWN_TREND) -------------------

REGIME_TYPES = frozenset({"BTC_TREND", "BTC_ROC", "OWN_TREND"})
REF_SYMBOL = "BTCUSDT"
_MS_NS = 1_000_000
_REF_STALE_NS = 2 * 86_400 * 1_000_000_000
_TF_NS = {"1h": 3_600 * 10**9, "4h": 14_400 * 10**9, "1d": 86_400 * 10**9}

# (catalog path, symbol, tf) -> (BarSeries identity, rounded close ts, close)
_REF: dict[tuple[str, str, str], tuple[object, np.ndarray, np.ndarray]] = {}
# (catalog path, symbol, tf, kind, period) -> (BarSeries identity, derived series)
_REF_DERIVED: dict[tuple[str, str, str, str, int], tuple[object, np.ndarray]] = {}


def _resolved_catalog() -> Path:
    from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

    # Absolute: ``_REF``/``_REF_DERIVED`` are process-wide, so a relative catalog
    # ("data/catalog") would serve one root's bars under another cwd's spelling
    # of the same key. abspath, not resolve(): a symlinked data/catalog must keep
    # working and keep its own key.
    return Path(os.path.abspath(_catalog_path or DEFAULT_CATALOG_PATH))  # noqa: PTH100


def _ref_bar_type(symbol: str, tf: str) -> str:
    unit = {"1h": "1-HOUR", "4h": "4-HOUR", "1d": "1-DAY"}.get(tf)
    if unit is None:
        msg = f"Unsupported reference timeframe {tf!r}"
        raise AuxDataUnavailableError(msg)
    return f"{symbol}-PERP.BINANCE-{unit}-LAST-EXTERNAL"


def _ref_bars(symbol: str, tf: str, refresh: bool = False) -> tuple[object, np.ndarray, np.ndarray]:
    """Resolved reference bars; per-call path is a dict lookup unless ``refresh``.

    ``refresh`` (preflight) re-reads the catalog (itself signature-cached) and
    swaps the entry if the files changed; :func:`configure` clears the cache.
    """
    catalog = _resolved_catalog()
    key = (str(catalog), symbol, tf)
    if not refresh:
        fast = _REF.get(key)
        if fast is not None:
            return fast
    from vibe_quant.validation.extraction import load_catalog_bars

    bar_type = _ref_bar_type(symbol, tf)
    bars = load_catalog_bars(catalog, bar_type)
    if bars is None or bars.ts.size == 0:
        msg = (
            f"Regime indicators need {tf} bars for {symbol}, but none found in {catalog} "
            f"({bar_type}) — download {symbol} {tf} data first"
        )
        raise AuxDataUnavailableError(msg)
    hit = _REF.get(key)
    if hit is not None and hit[0] is bars:
        return hit
    step = _TF_NS[tf]
    raw = bars.ts.astype(np.int64)
    # A complete bar's ts_init is its close (boundary) or 1 ms before it. Anything
    # else is an in-progress bar (the current day is archived as a partial 1d bar):
    # never a valid close, so it is dropped rather than rounded onto a boundary.
    complete = (raw % step == 0) | ((raw + _MS_NS) % step == 0)
    ts = ((raw[complete] + step - 1) // step) * step  # ceil to the close boundary
    close = np.asarray(bars.close, dtype=np.float64)[complete]
    if ts.size == 0:
        msg = f"No complete {tf} bars for {symbol} in {catalog}"
        raise AuxDataUnavailableError(msg)
    entry = (bars, ts, close)
    _REF[key] = entry
    return entry


def ref_close_series(symbol: str, tf: str = "1d") -> tuple[np.ndarray, np.ndarray]:
    """(close time ns, close) of ``symbol``'s ``tf`` catalog bars; cached per process.

    Close time is ``ts_init`` rounded to the timeframe boundary (a daily bar's
    23:59:59.999 becomes the next midnight).

    Raises:
        AuxDataUnavailableError: bar type missing or empty in the catalog.
    """
    _, ts, close = _ref_bars(symbol, tf)
    return ts, close


def _ref_derived(symbol: str, tf: str, kind: str, period: int) -> tuple[np.ndarray, np.ndarray]:
    """(close ts, derived series) computed once on the FULL reference history."""
    import pandas as pd

    bars, ts, close = _ref_bars(symbol, tf)
    key = (str(_resolved_catalog()), symbol, tf, kind, period)
    hit = _REF_DERIVED.get(key)
    if hit is not None and hit[0] is bars:
        return ts, hit[1]
    if kind == "trend":
        ema = pd.Series(close).ewm(span=period, adjust=False).mean().to_numpy()
        out = close / ema - 1.0
        out[: max(period - 1, 0)] = np.nan  # EMA warmup is not a trend value
    else:  # roc
        out = np.full(close.size, np.nan)
        if 0 < period < close.size:
            out[period:] = close[period:] / close[:-period] - 1.0
    _REF_DERIVED[key] = (bars, out)
    return ts, out


def _ref_lookup(ts: np.ndarray, series: np.ndarray, close_ns: np.ndarray) -> np.ndarray:
    """As-of: latest ref bar whose close <= t; NaN if none or > 2 days stale."""
    t = np.asarray(close_ns, dtype=np.int64)
    idx = np.searchsorted(ts, t, side="right") - 1
    safe = np.clip(idx, 0, None)
    ok = (idx >= 0) & (t - ts[safe] <= _REF_STALE_NS)
    return np.where(ok, series[safe], np.nan)


def ref_close_asof(symbol: str, close_ns_array: np.ndarray, tf: str = "1d") -> np.ndarray:
    """Close of the latest ``tf`` bar closed at or before each time (NaN if none/stale)."""
    ts, close = ref_close_series(symbol, tf)
    return _ref_lookup(ts, close, close_ns_array)


def ref_trend_asof(
    symbol: str, close_ns_array: np.ndarray, period: int, tf: str = "1d"
) -> np.ndarray:
    """close / EMA(close, span=period) - 1 on the reference series, as of each time."""
    ts, series = _ref_derived(symbol, tf, "trend", period)
    return _ref_lookup(ts, series, close_ns_array)


def ref_roc_asof(
    symbol: str, close_ns_array: np.ndarray, period: int, tf: str = "1d"
) -> np.ndarray:
    """close[i] / close[i-period] - 1 on the reference series, as of each time."""
    ts, series = _ref_derived(symbol, tf, "roc", period)
    return _ref_lookup(ts, series, close_ns_array)


def _preflight_regime(
    indicator_types: list[str],
    symbols: list[str],
    start_ns: int,
    end_ns: int,
    start_date: str,
    end_date: str,
) -> None:
    used = REGIME_TYPES.intersection(indicator_types)
    if not used:
        return
    refs: list[str] = []
    if used & {"BTC_TREND", "BTC_ROC"}:
        refs.append(REF_SYMBOL)
    if "OWN_TREND" in used:
        refs.extend(s for s in symbols if s not in refs)
    for symbol in refs:
        _, ts, _ = _ref_bars(symbol, "1d", refresh=True)
        if int(ts[0]) > start_ns - PREFLIGHT_START_MARGIN_NS:
            msg = (
                f"{symbol} daily bars start after the requested window "
                f"(first close {_iso(int(ts[0]))}, needed from "
                f"{_iso(start_ns - PREFLIGHT_START_MARGIN_NS)} for window start {start_date}) "
                f"— download {symbol} 1d data"
            )
            raise AuxDataUnavailableError(msg)
        if int(ts[-1]) < end_ns - 2 * 86_400 * 10**9:
            msg = (
                f"{symbol} daily bars end before the requested window "
                f"(last close {_iso(int(ts[-1]))}, window end {end_date}) — download {symbol} 1d data"
            )
            raise AuxDataUnavailableError(msg)


def context_of(df: pd.DataFrame) -> tuple[str, np.ndarray]:
    """(symbol, bar close ns) from a context DataFrame."""
    from vibe_quant.validation.funding import FundingCalculator

    try:
        ns = np.asarray(df.attrs["bar_close_ns"], dtype=np.int64)
        instrument = str(df.attrs["symbol"])
    except KeyError as e:
        msg = f"needs_context indicator called without df.attrs[{e.args[0]!r}]"
        raise ValueError(msg) from e
    if ns.size != len(df.index):
        msg = f"bar_close_ns has {ns.size} entries for {len(df.index)} rows"
        raise ValueError(msg)
    return FundingCalculator.symbol_from_instrument_id(instrument), ns
