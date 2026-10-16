"""MARK-price 1m bars: catalog materialisation + a fast memmap loader.

The liquidation scanner (``vibe_quant/liquidation/``) needs the *mark* price —
the price Binance actually liquidates against — as a plain numpy series, and it
runs in every discovery worker. Two things live here:

* :func:`build_mark_bars` — writes the archive's ``raw_mark_klines`` into the
  ParquetDataCatalog as ``<SYMBOL>-PERP.BINANCE-1-MINUTE-MARK-EXTERNAL`` bars.
  Mark klines carry no volume, so a zero volume is fed through the shared
  ``klines_to_bars`` converter (its LAST behaviour is untouched). Prices are
  formatted to the instrument tick, i.e. an 8-decimal mark price is rounded:
  the error is <= half a tick, which is far below the liquidation threshold the
  scanner tests, and it is the price the venue itself quotes in ticks.

* :func:`load_mark_series` — decodes those parquet files into the five compact
  arrays the scanner reads (ts/open/low/high/close; the open feeds its fill
  rule) and caches them as ``.npy`` under ``<catalog>/cache/mark/<SYMBOL>/``
  (``data/catalog`` is gitignored; disk is ~97 % full, so only these arrays are
  stored). The cache is a set of immutable generation directories named by the
  sha256 of the cache layout version plus the parquet files'
  ``(name, mtime_ns, size)`` signature (a layout change, e.g. the 4-array
  generations before ``open``, is rebuilt, never trusted): a writer fills a private temp dir and
  publishes it with one atomic ``os.rename``; the loser of a publish race keeps
  the winner's directory. Parallel discovery workers share the pages through
  ``np.load(mmap_mode="r")``; a generation whose arrays disagree with its
  ``meta.json`` is never trusted.

The price decoder mirrors ``vibe_quant.validation.extraction._decode_price_column``
(NT fixed-point columns). It is duplicated rather than imported so this
data-layer module does not depend on the validation package; dedupe it into a
shared helper when that file is next in scope.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.objects import FIXED_PRECISION

from vibe_quant.data.catalog import (
    CatalogManager,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence

    from vibe_quant.data.archive import RawDataArchive

logger = logging.getLogger(__name__)

_NS_PER_MINUTE = 60_000_000_000
_CACHE_ARRAYS = ("ts", "open", "low", "high", "close")
_CACHE_DTYPES = {
    "ts": np.int64,
    "open": np.float64,
    "low": np.float64,
    "high": np.float64,
    "close": np.float64,
}
_CACHE_LAYOUT = 2  # 1: ts/low/high/close (no open); bump on any array/semantics change
_MAX_LOAD_ATTEMPTS = 3
_TMP_PREFIX = ".tmp-"
_STALE_TMP_SECONDS = 3600


class MarkSeries(NamedTuple):
    """Mark-price bars: ``ts`` is the bar OPEN time in ns, strictly increasing."""

    ts: np.ndarray  # int64 ns
    low: np.ndarray  # float64
    high: np.ndarray  # float64
    close: np.ndarray  # float64
    open: np.ndarray  # float64


class MarkDataMissing(RuntimeError):
    """The MARK series does not cover the requested window (fail closed)."""


def mark_bar_type(symbol: str) -> str:
    """Catalog bar type string of a symbol's MARK 1m bars."""
    return str(get_bar_type(symbol, "1m", price_type=PriceType.MARK))


def build_mark_bars(
    catalog: CatalogManager,
    archive: RawDataArchive,
    symbol: str,
    interval: str = "1m",
) -> int:
    """Write a symbol's archived mark klines into the catalog as MARK bars.

    Clears the existing MARK bar directory first (idempotent re-runs) and
    returns the number of bars written; 0 when the archive holds no mark rows
    for the symbol.
    """
    rows = archive.get_mark_klines(symbol, interval)
    if not rows:
        return 0

    instrument = create_instrument(symbol)
    bar_type = get_bar_type(symbol, interval, price_type=PriceType.MARK)
    # Mark klines have no volume column; feed zero through the shared converter.
    kline_like = cast(
        "Sequence[sqlite3.Row]",
        [
            {
                "open_time": r["open_time"],
                "close_time": r["close_time"],
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": 0.0,
            }
            for r in rows
        ],
    )
    bars = klines_to_bars(
        kline_like, instrument.id, bar_type, instrument.size_precision, instrument.price_precision
    )

    catalog.write_instrument(instrument)
    catalog.clear_bar_data(symbol, interval, PriceType.MARK)
    catalog.write_bars(bars)
    return len(bars)


def load_mark_series(catalog_path: Path | str, symbol: str) -> MarkSeries | None:
    """Load a symbol's MARK 1m bars as compact arrays, or None if absent.

    Arrays are served from an immutable ``.npy`` generation (memmap) keyed by
    the parquet signature. On a miss the parquet is decoded; if its signature
    changed while decoding (a concurrent ``build_mark_bars``) the decode is
    discarded and repeated, so a cached generation always matches the exact
    parquet it was keyed by.

    Raises:
        RuntimeError: if the parquet keeps changing across the retry budget.
    """
    root = Path(catalog_path)
    bar_dir = root / "data" / "bar" / mark_bar_type(symbol)
    cache_root = root / "cache" / "mark" / symbol

    for _ in range(_MAX_LOAD_ATTEMPTS):
        try:
            files, signature = _snapshot(bar_dir)
            if not files:
                return None
            cached = _load_generation(cache_root, signature)
            if cached is not None:
                # Generations are immutable per signature; confirm the parquet set
                # (re-globbed, so added/removed files count) is still that signature.
                if _snapshot(bar_dir)[1] != signature:
                    continue
                return cached

            series = _decode_mark_bars(files)
            if _snapshot(bar_dir)[1] != signature:
                continue  # parquet set changed mid-decode: the rows may mix versions
        except FileNotFoundError:
            continue  # build_mark_bars swapped the parquet files under us: retry

        try:
            return _publish_generation(cache_root, signature, series) or series
        except OSError as e:
            # The cache is an optimisation: decoded arrays are still correct.
            logger.warning(
                "Mark series cache write failed for %s (%s); using decoded arrays", symbol, e
            )
            return series
    raise RuntimeError(
        f"{symbol}: MARK parquet changed during {_MAX_LOAD_ATTEMPTS} consecutive loads"
    )


def require_mark_coverage(
    series: MarkSeries | None,
    symbol: str,
    start_ns: int,
    end_ns: int,
    max_gap_minutes: int = 60,
) -> None:
    """Fail closed unless MARK bars cover the half-open window ``[start_ns, end_ns)``.

    Coverage means every minute ``m`` with ``start_ns <= m < end_ns`` is either
    a bar (``ts`` holds bar OPEN times) or lies inside a gap of at most
    ``max_gap_minutes`` between consecutive bars. So the last bar may open at
    ``end_ns - 60 s`` (it covers the final minute) and the bar opening at
    ``end_ns`` itself is not required. Raises :class:`MarkDataMissing` naming
    the symbol and the first missing month/gap when

    * the series is absent or empty,
    * the first bar opens after ``start_ns``,
    * the last bar opens before ``end_ns - 60 s``, or
    * any gap between consecutive bars longer than ``max_gap_minutes``
      overlaps the window: wholly inside it, straddling either boundary, or
      the window lying entirely inside the gap. A gap between bars at ``a``
      and ``b`` spans ``(a + 60 s, b)``, compared at ns resolution so even a
      sub-minute window sliver counts; a gap with ``b == start_ns`` or
      ``a + 60 s == end_ns`` misses nothing in-window.
    """
    if series is None or len(series.ts) == 0:
        raise MarkDataMissing(
            f"{symbol}: no MARK price bars available for {_fmt_ns(start_ns)}..{_fmt_ns(end_ns)}"
        )

    ts = series.ts
    if ts[0] > start_ns:
        raise MarkDataMissing(
            f"{symbol}: MARK coverage starts {_fmt_ns(int(ts[0]))} (missing {_month_label(start_ns)}"
            f"..) after requested start {_fmt_ns(start_ns)}"
        )

    if ts[-1] < end_ns - _NS_PER_MINUTE:
        raise MarkDataMissing(
            f"{symbol}: MARK coverage ends {_fmt_ns(int(ts[-1]))} (missing"
            f" {_month_label(end_ns)}) before requested end {_fmt_ns(end_ns)}"
        )

    limit = max_gap_minutes * _NS_PER_MINUTE
    diffs = np.diff(ts)
    for i in np.nonzero(diffs > limit)[0]:
        gap_start = int(ts[i])
        gap_end = int(ts[i + 1])
        # The gap spans (gap_start + 60 s, gap_end) in ns; if it intersects
        # [start_ns, end_ns) at all - even by a sub-minute sliver - it is a hole.
        if gap_start + _NS_PER_MINUTE < end_ns and gap_end > start_ns:
            minutes = (gap_end - gap_start) // _NS_PER_MINUTE
            raise MarkDataMissing(
                f"{symbol}: MARK gap {_fmt_ns(gap_start)} -> {_fmt_ns(gap_end)}"
                f" ({_month_label(gap_start)}, {minutes} min) exceeds {max_gap_minutes} min"
            )


def check_mark_coverage(
    catalog_path: Path | str,
    symbols: Sequence[str],
    start_ns: int,
    end_ns: int,
) -> None:
    """Fail closed unless every symbol's MARK 1m bars cover ``[start_ns, end_ns)``.

    Loads each symbol's series in the given order and raises the first
    :class:`MarkDataMissing` reported by :func:`require_mark_coverage` (which
    owns the coverage/gap rules). A mode-on launch calls this before any run
    row exists; the runtime keeps its own per-symbol loop because it needs the
    loaded series.
    """
    for symbol in symbols:
        require_mark_coverage(load_mark_series(catalog_path, symbol), symbol, start_ns, end_ns)


def _snapshot(bar_dir: Path) -> tuple[list[Path], tuple[tuple[str, int, int], ...]]:
    """Glob the complete parquet set and sign it (raises FileNotFoundError if it moves)."""
    files = sorted(bar_dir.glob("*.parquet")) if bar_dir.is_dir() else []
    return files, _parquet_signature(files)


def _parquet_signature(files: list[Path]) -> tuple[tuple[str, int, int], ...]:
    return tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files)


def _decode_price_column(column: Any) -> np.ndarray:
    """NT fixed-point price column (fixed_size_binary 8/16) -> float64."""
    arr = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    width = arr.type.byte_width
    buf = arr.buffers()[1]
    raw = np.frombuffer(buf, dtype=np.uint8)[arr.offset * width : (arr.offset + len(arr)) * width]
    if width == 16:
        words = raw.view("<u8").reshape(-1, 2)
        value = words[:, 1].view("<i8").astype(np.float64) * 18446744073709551616.0 + words[
            :, 0
        ].astype(np.float64)
        return np.asarray(value / 10.0 ** int(FIXED_PRECISION), dtype=np.float64)
    if width == 8:
        return np.asarray(raw.view("<i8").astype(np.float64) / 1e9, dtype=np.float64)
    raise ValueError(f"Unsupported NT price encoding: {width}-byte fixed binary")


def _decode_mark_bars(files: list[Path]) -> MarkSeries:
    import pyarrow.parquet as pq

    prices = ("low", "high", "close", "open")
    parts: list[tuple[np.ndarray, ...]] = []
    for f in files:
        table = pq.read_table(  # type: ignore[no-untyped-call]
            f, columns=[*prices, "ts_event"]
        )
        cols = [_decode_price_column(table.column(name)) for name in prices]
        ts = table.column("ts_event").to_numpy().astype(np.int64)
        parts.append((ts, *cols))

    ts_all = np.concatenate([p[0] for p in parts])
    order = np.argsort(ts_all, kind="stable")
    ts_sorted = ts_all[order]
    keep = np.concatenate(([True], ts_sorted[1:] != ts_sorted[:-1]))
    sel = order[keep]
    return MarkSeries(
        ts=ts_all[sel],
        low=np.concatenate([p[1] for p in parts])[sel],
        high=np.concatenate([p[2] for p in parts])[sel],
        close=np.concatenate([p[3] for p in parts])[sel],
        open=np.concatenate([p[4] for p in parts])[sel],
    )


def _signature_json(signature: tuple[tuple[str, int, int], ...]) -> list[list[Any]]:
    return [[name, mtime, size] for name, mtime, size in signature]


def _signature_key(signature: tuple[tuple[str, int, int], ...]) -> str:
    keyed = {"layout": _CACHE_LAYOUT, "signature": _signature_json(signature)}
    return hashlib.sha256(json.dumps(keyed).encode()).hexdigest()


def _write_generation(
    target: Path, signature: tuple[tuple[str, int, int], ...], series: MarkSeries
) -> None:
    """Fill a private (not yet published) generation directory."""
    for name in _CACHE_ARRAYS:
        np.save(target / f"{name}.npy", getattr(series, name))
    meta = {
        "layout": _CACHE_LAYOUT,
        "signature": _signature_json(signature),
        "length": int(len(series.ts)),
    }
    (target / "meta.json").write_text(json.dumps(meta))


def _publish_generation(
    cache_root: Path, signature: tuple[tuple[str, int, int], ...], series: MarkSeries
) -> MarkSeries | None:
    """Publish ``series`` as the generation for ``signature``; return it memmapped.

    The generation is built in a unique temp dir inside ``cache_root`` and
    published with one atomic ``os.rename`` onto the signature-named directory,
    so readers never see a partial generation and writers never share files. If
    the target already exists a concurrent writer won: ours is discarded and
    theirs is used, unless theirs fails validation (then it is replaced once).
    Returns None when the published generation cannot be re-read (caller falls
    back to the decoded arrays).
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    gen_dir = cache_root / _signature_key(signature)
    tmp = Path(tempfile.mkdtemp(prefix=_TMP_PREFIX, dir=cache_root))
    try:
        _write_generation(tmp, signature, series)
        try:
            os.rename(tmp, gen_dir)
        except OSError:
            if not gen_dir.is_dir():
                raise
            existing = _load_generation(cache_root, signature)
            if existing is not None:
                _prune_cache(cache_root, keep=gen_dir.name)
                return existing
            # Corrupt generation (e.g. interrupted disk write): replace it.
            shutil.rmtree(gen_dir, ignore_errors=True)
            os.rename(tmp, gen_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _prune_cache(cache_root, keep=gen_dir.name)
    return _load_generation(cache_root, signature)


def _prune_cache(cache_root: Path, keep: str) -> None:
    """Best-effort removal of superseded generations, legacy files and stale temp dirs."""
    try:
        entries = list(cache_root.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name == keep:
            continue
        try:
            if entry.is_dir():
                live_tmp = (
                    entry.name.startswith(_TMP_PREFIX)
                    and time.time() - entry.stat().st_mtime < _STALE_TMP_SECONDS
                )
                if not live_tmp:  # a young temp dir may belong to a live writer
                    shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
        except OSError:
            continue


def _load_generation(
    cache_root: Path, signature: tuple[tuple[str, int, int], ...]
) -> MarkSeries | None:
    """Open the generation for ``signature``; None unless every check passes.

    Checks: meta.json layout is ``_CACHE_LAYOUT`` and its signature equals the
    current parquet signature, all five arrays exist, are 1-D, have the expected dtype and exactly ``meta.length``
    elements. Anything inconsistent is "no cache" - the caller rebuilds.
    """
    gen_dir = cache_root / _signature_key(signature)
    try:
        meta = json.loads((gen_dir / "meta.json").read_text())
        if meta.get("layout") != _CACHE_LAYOUT or meta["signature"] != _signature_json(signature):
            return None
        length = int(meta["length"])
        arrays = {
            name: np.load(gen_dir / f"{name}.npy", mmap_mode="r") for name in _CACHE_ARRAYS
        }
    except (OSError, ValueError, KeyError, TypeError) as e:
        if gen_dir.exists():
            logger.warning("Mark series cache read failed at %s (%s)", gen_dir, e)
        return None
    for name, arr in arrays.items():
        if arr.shape != (length,) or arr.dtype != _CACHE_DTYPES[name]:
            logger.warning(
                "Mark series cache %s: %s.npy has shape %s dtype %s, expected (%d,) %s",
                gen_dir,
                name,
                arr.shape,
                arr.dtype,
                length,
                np.dtype(_CACHE_DTYPES[name]),
            )
            return None
    return MarkSeries(
        ts=arrays["ts"],
        low=arrays["low"],
        high=arrays["high"],
        close=arrays["close"],
        open=arrays["open"],
    )


def _fmt_ns(ts_ns: int) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ts_ns / 1e9, tz=UTC).strftime("%Y-%m-%dT%H:%MZ")


def _month_label(ts_ns: int) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(ts_ns / 1e9, tz=UTC).strftime("%Y-%m")
