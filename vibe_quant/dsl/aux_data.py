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

    if not any(
        (spec := indicator_registry.get(t)) is not None and spec.needs_context
        for t in indicator_types
    ):
        return
    start_ns, end_ns = _date_ns(start_date), _date_ns(end_date)
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
