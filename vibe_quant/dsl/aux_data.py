"""Non-OHLCV context for ``needs_context`` indicators (funding rates today).

Runners call :func:`configure` in-process before building the engine (no
StrategyConfig fields -- NT 1.226+ rejects unknown ones) and :func:`preflight`
with the run window. Generated strategies hand each ``needs_context``
``compute_fn`` a DataFrame whose ``attrs`` carry ``bar_close_ns`` (int64 close
time per row) and ``symbol`` (instrument id).

Funding comes from the archive via :class:`FundingCalculator`'s loader and its
module-level ``_RATE_CACHE`` (one cache: ``clear_rate_cache`` covers both). The
only derived state (snapped series, z/mean series) is validated against the
identity of the cached source tuple, so a cache clear invalidates it.

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

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vibe_quant.errors import DataUnavailableError

if TYPE_CHECKING:
    import pandas as pd


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


def configure(archive_path: Path | str | None, catalog_path: Path | str | None = None) -> None:
    """Point aux-data lookups at the run's archive / catalog (None = default)."""
    global _archive_path, _catalog_path
    _archive_path = Path(archive_path) if archive_path else None
    _catalog_path = Path(catalog_path) if catalog_path else None
    # Reference bars are resolved once per run (preflight / first use), not per bar.
    _REF.clear()
    _REF_DERIVED.clear()


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


def _date_ns(value: str) -> int:
    return int(datetime.fromisoformat(value[:10]).replace(tzinfo=UTC).timestamp()) * 1_000_000_000


def preflight(
    indicator_types: list[str], symbols: list[str], start_date: str, end_date: str
) -> None:
    """Fail before the engine starts if context indicators lack aux data for the window.

    Raises :class:`AuxDataUnavailableError` when the archive is missing, has no
    funding for a symbol, its first settlement is more than
    ``PREFLIGHT_START_MARGIN_NS`` (3 days, warmup allowance) after ``start_date``,
    or its last settlement is before ``end_date - 16h``.
    """
    from vibe_quant.dsl.indicators import indicator_registry
    from vibe_quant.validation.funding import _FUNDING_PERIOD_NS

    start_ns, end_ns = _date_ns(start_date), _date_ns(end_date)
    _preflight_regime(indicator_types, symbols, start_ns, end_ns, start_date, end_date)
    if not any(
        (spec := indicator_registry.get(t)) is not None
        and spec.needs_context
        and t not in REGIME_TYPES
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


def _iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).strftime("%Y-%m-%d %H:%M")


def _asof(
    snapped: np.ndarray, series: np.ndarray, bar_close_ns: np.ndarray
) -> np.ndarray:
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

    return _catalog_path or DEFAULT_CATALOG_PATH


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


def _ref_lookup(
    ts: np.ndarray, series: np.ndarray, close_ns: np.ndarray
) -> np.ndarray:
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
