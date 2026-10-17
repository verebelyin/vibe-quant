"""Plant a known causal drift edge into a copy of BTCUSDT 1m klines (power test, L2).

At each 4h close the RSI(14) of the PLANTED 4h closes is updated; a trigger is
``rsi < 30 and prev >= 30`` (``crosses_below``). The 1,440 minutes (6 4h bars) that
start at the trigger bar's close get an extra +delta log-drift per minute (overlapping
triggers union, never 2*delta). A constant -kappa per minute over the plant window keeps
the end level near the input (``solve_kappa``). Everything is causal: the drift at a
minute depends only on triggers whose 4h bar closed at or before it.

Level model (contiguous 1m rows assumed, as in the BTCUSDT archive): with
m = minutes since ``plant_start_ms`` (clipped to the window), the log level is
``L = -kappa*m + delta*D`` where D counts drift minutes with start < open_time < end;
open/high/low/close are multiplied by ``exp(L)``, volume and times are untouched.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from nautilus_trader.indicators import RelativeStrengthIndex

if TYPE_CHECKING:
    from pathlib import Path

MIN_MS = 60_000
BAR_MIN = 240  # 4h
DRIFT_MIN = 6 * BAR_MIN  # 1,440 minutes
RSI_PERIOD = 14
RSI_LEVEL = 30.0

SYMBOL = "BTCUSDT"
PLANT_START = "2022-06-01"
PLANT_END = "2026-06-30"


@dataclass(frozen=True)
class Klines:
    """1m klines as parallel arrays (times in ms)."""

    open_time: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    close_time: np.ndarray


@dataclass(frozen=True)
class PlantReport:
    n_triggers: int
    trigger_close_ms: tuple[int, ...]  # 4h close of each trigger bar
    drift_minutes: int  # D: minutes with +delta applied (union, inside the window)
    l_end: float  # log level at the end of the plant window
    l_max_abs: float


@dataclass(frozen=True)
class PlantResult:
    klines: Klines
    report: PlantReport


def crosses_below(prev: float | None, cur: float | None, level: float = RSI_LEVEL) -> bool:
    """True iff prev >= level and cur < level; an uninitialized value (None) never triggers."""
    if prev is None or cur is None:
        return False
    return cur < level <= prev


def plant_drift(
    k: Klines,
    *,
    delta: float,
    kappa: float,
    plant_start_ms: int,
    plant_end_ms: int,
    price_decimals: int | None = None,
) -> PlantResult:
    """Return planted klines + report. ``kappa`` is explicit; this function never iterates."""
    t = k.open_time
    n = len(t)
    m_max = (plant_end_ms - plant_start_ms) // MIN_MS - 1
    keys = t // (BAR_MIN * MIN_MS)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(keys)) + 1, [n])) if n else np.array([0])

    open_ = k.open.copy()
    high = k.high.copy()
    low = k.low.copy()
    close = k.close.copy()
    rsi = RelativeStrengthIndex(RSI_PERIOD)
    prev_rsi: float | None = None
    drift_until = -1
    d_total = 0
    level = 0.0
    l_max = 0.0
    triggers: list[int] = []

    for gi in range(len(starts) - 1):
        a, b = int(starts[gi]), int(starts[gi + 1])
        tg = t[a:b]
        inwin = (tg > plant_start_ms) & (tg < plant_end_ms)
        active = inwin & (tg < drift_until)
        m = np.clip((tg - plant_start_ms) // MIN_MS, 0, max(m_max, 0))
        d_seg = d_total + np.cumsum(active)
        lvl = delta * d_seg - kappa * m
        d_total = int(d_seg[-1])
        level = float(lvl[-1])
        l_max = max(l_max, float(np.max(np.abs(lvl))))
        moved = lvl != 0.0
        if moved.any():
            f = np.exp(lvl)
            for src, dst in ((k.open, open_), (k.high, high), (k.low, low), (k.close, close)):
                new = src[a:b] * f
                if price_decimals is not None:
                    new = np.round(new, price_decimals)
                dst[a:b] = np.where(moved, new, src[a:b])
        if b - a != BAR_MIN:
            continue  # incomplete 4h bar: no RSI update
        bar_close_ms = int(keys[a]) * BAR_MIN * MIN_MS + BAR_MIN * MIN_MS
        rsi.update_raw(float(close[b - 1]))
        cur: float | None = float(rsi.value) * 100.0 if rsi.initialized else None
        if (
            crosses_below(prev_rsi, cur)
            and int(keys[a]) * BAR_MIN * MIN_MS >= plant_start_ms
            and bar_close_ms <= plant_end_ms
        ):
            triggers.append(bar_close_ms)
            drift_until = max(drift_until, bar_close_ms + DRIFT_MIN * MIN_MS)
        prev_rsi = cur

    planted = Klines(
        open_time=k.open_time.copy(),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=k.volume.copy(),
        close_time=k.close_time.copy(),
    )
    report = PlantReport(len(triggers), tuple(triggers), d_total, level, l_max)
    return PlantResult(planted, report)


def solve_kappa(
    k: Klines,
    delta: float,
    *,
    plant_start_ms: int,
    plant_end_ms: int,
    price_decimals: int | None = None,
    max_passes: int = 5,
    tol: float = 0.002,
) -> tuple[float, int, float]:
    """Fixed point kappa with |L_end| <= tol; returns (kappa, passes, L_end of that kappa).

    Each pass calls ``plant_drift`` with the candidate kappa and moves it by L_end / M
    (M = window minutes), since L_end = delta*D - kappa*M. Stops after ``max_passes``
    even when not converged (the caller checks L_end).
    """
    window_min = (plant_end_ms - plant_start_ms) // MIN_MS - 1
    kappa = 0.0
    l_end = math.inf
    passes = 0
    while passes < max_passes:
        res = plant_drift(
            k,
            delta=delta,
            kappa=kappa,
            plant_start_ms=plant_start_ms,
            plant_end_ms=plant_end_ms,
            price_decimals=price_decimals,
        )
        passes += 1
        l_end = res.report.l_end
        if abs(l_end) <= tol or passes == max_passes:
            break
        kappa += l_end / window_min
    return kappa, passes, l_end


def _ms(day: str) -> int:
    from datetime import UTC, datetime

    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1000)


def load_klines(
    archive_path: Path,
    symbol: str = SYMBOL,
    start: str = PLANT_START,
    end: str = PLANT_END,
) -> Klines:
    """Read 1m klines for days [start, end] (end day inclusive), archive opened read-only."""
    conn = sqlite3.connect(f"file:{archive_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT open_time, open, high, low, close, volume, close_time FROM raw_klines "
            "WHERE symbol = ? AND interval = '1m' AND open_time >= ? AND open_time <= ? "
            "ORDER BY open_time",
            (symbol, _ms(start), _ms(end) + 86_400_000 - 1),
        ).fetchall()
    finally:
        conn.close()
    a = np.array(rows, dtype=np.float64)
    return Klines(
        open_time=a[:, 0].astype(np.int64),
        open=a[:, 1],
        high=a[:, 2],
        low=a[:, 3],
        close=a[:, 4],
        volume=a[:, 5],
        close_time=a[:, 6].astype(np.int64),
    )


def write_scratch(root: Path, planted: Klines, source_archive: Path, symbol: str = SYMBOL) -> None:
    """Write the scratch catalog (1m..1d) + scratch archive (funding rows only) under ``root``.

    Uses the repo's catalog functions; ``source_archive`` is opened read-only. Raises if
    ``root`` already holds a catalog (each build gets a fresh root).
    """
    from vibe_quant.data.archive import RawDataArchive
    from vibe_quant.data.catalog import (
        CatalogManager,
        aggregate_bars,
        create_instrument,
        get_bar_type,
        klines_to_bars,
    )
    from vibe_quant.data.ingest import AGGREGATION_INTERVALS, _interval_to_minutes

    cat_dir = root / "data" / "catalog"
    if cat_dir.exists():
        raise FileExistsError(f"scratch catalog already exists: {cat_dir}")
    catalog = CatalogManager(cat_dir)
    instrument = create_instrument(symbol)
    catalog.write_instrument(instrument)
    rows: list[dict[str, Any]] = [
        {
            "open_time": int(planted.open_time[i]),
            "close_time": int(planted.close_time[i]),
            "open": float(planted.open[i]),
            "high": float(planted.high[i]),
            "low": float(planted.low[i]),
            "close": float(planted.close[i]),
            "volume": float(planted.volume[i]),
        }
        for i in range(len(planted.open_time))
    ]
    sp, pp = instrument.size_precision, instrument.price_precision
    bars_1m = klines_to_bars(rows, instrument.id, get_bar_type(symbol, "1m"), sp, pp)  # type: ignore[arg-type]
    catalog.write_bars(bars_1m)
    for interval in AGGREGATION_INTERVALS:
        agg = aggregate_bars(
            bars_1m, get_bar_type(symbol, interval), _interval_to_minutes(interval), sp, pp
        )
        catalog.write_bars(agg)

    arch_dir = root / "data" / "archive"
    arch_dir.mkdir(parents=True, exist_ok=True)
    scratch = RawDataArchive(arch_dir / "raw_data.db")
    src = sqlite3.connect(f"file:{source_archive}?mode=ro", uri=True)
    try:
        rates = src.execute(
            "SELECT funding_time, funding_rate, mark_price FROM raw_funding_rates WHERE symbol = ?",
            (symbol,),
        ).fetchall()
        ivals = src.execute(
            "SELECT funding_time, interval_hours FROM raw_funding_intervals WHERE symbol = ?",
            (symbol,),
        ).fetchall()
    finally:
        src.close()
    scratch.insert_funding_rates(symbol, rates, "power_scratch")
    scratch.insert_funding_intervals(symbol, ivals, "power_scratch")
    scratch.close()
