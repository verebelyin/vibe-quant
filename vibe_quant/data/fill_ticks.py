"""Immutable per-symbol "fill tick" sets derived from the 1m archive.

Screening releases a strategy's queued orders on these ticks so they fill at
the same price and time validation fills them (the close of the 1m bars at
the strategy-bar boundary T and T+1m). For each UTC-aligned boundary T of a
timeframe coarser than 1m we emit two ``TradeTick``s: the close of the 1m bar
opening at T and of the 1m bar opening at T+1m, each stamped at that bar's
``ts_init`` (bar open + 59.999s).

Sets live in immutable directories
``<catalog.parent>/fill_ticks/<FILL_TICK_VERSION>/<tf>/<symbol>/<fingerprint>/``
(a ParquetDataCatalog root + ``meta.json``). The fingerprint hashes the 1m bar
and instrument parquet file names, sizes and mtimes, so refreshed 1m data or a
changed instrument (copied into the dir) builds a NEW dir; an
existing dir is never rewritten (builders run under an fcntl lock into a tmp
dir that is renamed into place).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from nautilus_trader.model.data import TradeTick
    from nautilus_trader.model.instruments import CryptoPerpetual

# Bump when the tick construction changes (new dir tree, old sets ignored).
FILL_TICK_VERSION = "v3"

# Huge on purpose: a tick's size is L1 liquidity in NT's matching engine, so a
# small size causes partial fills / REDUCE_ONLY rejects.
FILL_TICK_SIZE = 1_000_000

_MINUTE_NS = 60_000_000_000
# 1m bar ts_init = open + 59.999s
_BAR_INIT_OFFSET_NS = 59_999_000_000
_TF_NS: dict[str, int] = {
    "5m": 300 * 1_000_000_000,
    "15m": 900 * 1_000_000_000,
    "1h": 3_600 * 1_000_000_000,
    "4h": 14_400 * 1_000_000_000,
    "1d": 86_400 * 1_000_000_000,
}


@dataclass(frozen=True)
class FillTickSet:
    """Resolved fill-tick catalogs: ``paths[symbol]`` is a ParquetDataCatalog root.

    ``start_ns``/``end_ns`` is the window the coverage check passed for; a run
    may only use the set inside it (None = unknown, never accepted by a run).
    """

    timeframe: str
    paths: Mapping[str, str]
    missing_boundaries: Mapping[str, int]
    start_ns: int | None = None
    end_ns: int | None = None


def _timeframe_ns(timeframe: str) -> int:
    if timeframe == "1m":
        raise ValueError("fill ticks are for timeframes coarser than 1m")
    try:
        return _TF_NS[timeframe]
    except KeyError:
        raise ValueError(f"unsupported fill-tick timeframe: {timeframe!r}") from None


def _scan(bars_1m: Any, step: int) -> tuple[np.ndarray, int, int]:
    """Tick selection: (bar indices, boundaries in the data range, gaps).

    A tick is emitted for every 1m bar opening at a boundary T or at T+1m. A
    gap is a boundary with NO tick at all (a boundary with one of the two is
    still released, on that tick).
    """
    ts = np.asarray(bars_1m.ts, dtype=np.int64)
    if ts.size == 0:
        return np.empty(0, dtype=np.int64), 0, 0
    opens = ts - _BAR_INIT_OFFSET_NS
    rem = opens % step
    idx = np.flatnonzero((rem == 0) | (rem == _MINUTE_NS))
    first_b = (int(opens[0]) // step) * step
    if int(opens[0]) - first_b > _MINUTE_NS:  # data starts mid-boundary: not a gap
        first_b += step
    last_b = (int(opens[-1]) // step) * step
    n_boundaries = (last_b - first_b) // step + 1
    have = np.unique((opens[idx] // step) * step)
    return idx, n_boundaries, n_boundaries - int(have.size)


def build_fill_ticks(
    bars_1m: Any, timeframe: str, instrument: CryptoPerpetual
) -> tuple[list[TradeTick], int]:
    """Up to two ticks per strategy-bar boundary from decoded 1m bars.

    Args:
        bars_1m: ``BarSeries``-like (``ts`` = bar ts_init ns, ``close``).
        timeframe: Strategy timeframe (coarser than 1m).
        instrument: The MAIN catalog's instrument (its precisions are what the
            engine checks ticks against; ``create_instrument`` can differ).

    Returns:
        (ticks sorted by time, number of boundaries without any tick).
    """
    from nautilus_trader.model.data import TradeTick
    from nautilus_trader.model.enums import AggressorSide
    from nautilus_trader.model.identifiers import TradeId

    step = _timeframe_ns(timeframe)
    idx, _, gaps = _scan(bars_1m, step)
    size = instrument.make_qty(FILL_TICK_SIZE)
    ts = bars_1m.ts
    close = bars_1m.close
    ticks = [
        TradeTick(
            instrument.id,
            instrument.make_price(float(close[i])),
            size,
            AggressorSide.NO_AGGRESSOR,
            TradeId(f"F{int(ts[i])}"),
            int(ts[i]),
            int(ts[i]),
        )
        for i in idx
    ]
    return ticks, gaps


def _fingerprint(bar_dir: Path, instrument_dir: Path) -> str:
    """Hash of the 1m bar and instrument parquet stats (both feed the built dir)."""
    h = hashlib.sha256()
    for label, src in (("bar", bar_dir), ("instrument", instrument_dir)):
        for f in sorted(src.glob("*.parquet")):
            st = f.stat()
            h.update(f"{label}|{f.name}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def _to_ns(value: datetime | str | int) -> int:
    import pandas as pd

    if isinstance(value, int):
        return value
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.value)


def _build_one(symbol: str, timeframe: str, catalog_path: Path, root: Path) -> Path:
    """Return the immutable dir for the symbol's current 1m data, building if absent."""
    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    from vibe_quant.data.catalog import get_bar_type
    from vibe_quant.screening.nt_runner import MissingBarDataError
    from vibe_quant.validation.extraction import load_catalog_bars

    bar_type = str(get_bar_type(symbol, "1m"))
    bar_dir = catalog_path / "data" / "bar" / bar_type
    if not bar_dir.is_dir() or not any(bar_dir.glob("*.parquet")):
        raise MissingBarDataError(f"no 1m bars for {symbol} in {catalog_path}")
    sym_dir = root / FILL_TICK_VERSION / timeframe / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    with open(sym_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            instrument_id = bar_type.rsplit("-", 4)[0]
            instrument_dir = catalog_path / "data" / "crypto_perpetual" / instrument_id
            final = sym_dir / _fingerprint(bar_dir, instrument_dir)
            if (final / "meta.json").is_file():
                return final
            bars = load_catalog_bars(catalog_path, bar_type)
            if bars is None or len(bars.ts) == 0:
                raise MissingBarDataError(f"no 1m bars for {symbol} in {catalog_path}")
            main_cat = ParquetDataCatalog(str(catalog_path))
            found = main_cat.instruments(instrument_ids=[instrument_id])
            if not found:
                raise MissingBarDataError(f"no instrument for {symbol} in {catalog_path}")
            instrument = found[0]
            ticks, gaps = build_fill_ticks(bars, timeframe, instrument)
            # orphans of a killed builder (we hold the lock, so none is live)
            for orphan in sym_dir.glob(".tmp-*"):
                shutil.rmtree(orphan, ignore_errors=True)
            tmp = sym_dir / f".tmp-{uuid.uuid4().hex}"
            try:
                dst = ParquetDataCatalog(str(tmp))
                dst.write_data([instrument])
                if ticks:
                    dst.write_data(ticks)
                meta = {
                    "version": FILL_TICK_VERSION,
                    "timeframe": timeframe,
                    "symbol": symbol,
                    "n_ticks": len(ticks),
                    "missing_boundaries": gaps,
                }
                (tmp / "meta.json").write_text(json.dumps(meta))
                os.rename(tmp, final)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            return final
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _tick_files(path: Path) -> list[Path]:
    return sorted((path / "data" / "trade_tick").rglob("*.parquet"))


def _tick_timestamps(path: Path) -> np.ndarray:
    """ts_init of every tick actually on disk (sorted)."""
    import pyarrow.parquet as pq

    cols = [
        pq.read_table(f, columns=["ts_init"]).column("ts_init").to_numpy()  # type: ignore[no-untyped-call]
        for f in _tick_files(path)
    ]
    if not cols:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(cols).astype(np.int64))


def fill_tick_dir_signature(path: Path | str) -> tuple[tuple[str, int, int], ...]:
    """(relative path, size, mtime_ns) of every file in a built dir.

    NT writes epoch parquet into a catalog on dispose; callers that hand a set
    to an engine can compare this before/after to prove the dir is untouched.
    """
    root = Path(path)
    return tuple(
        (str(f.relative_to(root)), f.stat().st_size, f.stat().st_mtime_ns)
        for f in sorted(root.rglob("*"))
        if f.is_file()
    )


def resolve_fill_ticks(
    symbols: Sequence[str],
    timeframe: str,
    start: datetime | str | int,
    end: datetime | str | int,
    catalog_path: Path | str,
) -> FillTickSet:
    """Build (if absent) and coverage-check the fill-tick sets for ``symbols``.

    Raises:
        ValueError: ``timeframe`` is '1m' or unsupported.
        MissingBarDataError: a symbol has no 1m bars, or its set does not cover
            [start, end] (first tick > start+tf, last tick < end-tf, no tick inside
            the window, or ticks on disk disagree with the build's count).
    """
    from vibe_quant.screening.nt_runner import MissingBarDataError

    tf_ns = _timeframe_ns(timeframe)
    cat = Path(catalog_path)
    root = cat.parent / "fill_ticks"
    start_ns, end_ns = _to_ns(start), _to_ns(end)
    paths: dict[str, str] = {}
    missing: dict[str, int] = {}
    for symbol in sorted(symbols):
        path = _build_one(symbol, timeframe, cat, root)
        meta = json.loads((path / "meta.json").read_text())
        ts = _tick_timestamps(path)
        in_win = ts[(ts >= start_ns) & (ts <= end_ns)]
        problem = None
        if ts.size != meta["n_ticks"]:
            problem = f"{ts.size} ticks on disk != {meta['n_ticks']} built"
        elif ts.size == 0 or in_win.size == 0:
            problem = "no ticks in the window"
        elif int(ts[0]) > start_ns + tf_ns:
            problem = f"first tick {int(ts[0])} > start+tf"
        elif int(ts[-1]) < end_ns - tf_ns:
            problem = f"last tick {int(ts[-1])} < end-tf"
        if problem:
            raise MissingBarDataError(
                f"fill ticks for {symbol} {timeframe} do not cover the run window: {problem}"
            )
        # boundaries T in [start, end) with no tick (1m holes inside the window)
        lo = -(-start_ns // tf_ns) * tf_ns
        hi = ((end_ns - 1) // tf_ns) * tf_ns
        n_window = max(0, (hi - lo) // tf_ns + 1)
        have = np.unique(((in_win - _BAR_INIT_OFFSET_NS) // tf_ns) * tf_ns)
        have = have[(have >= lo) & (have <= hi)]
        paths[symbol] = str(path)
        missing[symbol] = n_window - int(have.size)
    return FillTickSet(
        timeframe=timeframe,
        paths=paths,
        missing_boundaries=missing,
        start_ns=start_ns,
        end_ns=end_ns,
    )
