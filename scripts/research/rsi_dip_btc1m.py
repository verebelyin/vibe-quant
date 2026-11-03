"""Quick look: buy BTCUSDT 1m when RSI drops below a threshold (user idea, 2026-11-03).

Scratch research, NOT evidence, NOT in any DSR count. BTC 1m has been looked at
many times before (exactness strategy 239, sweeps), so nothing here is unseen.

Frozen before the first run:
- Data: catalog BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL, 2020-01-01..2026-10-20.
- RSI: Wilder smoothing (ewm alpha=1/n, adjust=False) on close, periods 7 and 14.
- Signal: RSI crosses below thr at the close of bar t (RSI[t-1] >= thr > RSI[t]),
  thr in 10, 15, 20, 25, 30. Long entry at the OPEN of bar t+1. One position at a time.
- Exits: hold 15 / 60 / 240 bars (exit at the open N bars after entry), or
  RSI crosses back above 50 (exit at the next bar's open, max 1440 bars).
  A trade whose exit is more than N+10 minutes after entry (data gap) is dropped.
- Cost: 0.10% round trip (taker both sides) is the decision number; gross and
  0.04% (maker both sides) are report-only. No funding (holds <= 1 day).
- Baseline: mean gross return of the same hold from EVERY bar's open (drift).
- Pass bar per cell (40 cells): >= 100 trades, net mean > 0, one-sided t-test
  p < 0.05/40 = 0.00125, and net mean > 0 in each period 2020-22, 2023-24, 2025-26.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from vibe_quant.validation.extraction import load_catalog_bars

PERIODS = (7, 14)
THRESHOLDS = (10, 15, 20, 25, 30)
HOLDS: tuple[int | str, ...] = (15, 60, 240, "rsi50")
MAX_HOLD = 1440
COST = 0.0010
COST_MAKER = 0.0004
ERAS = {"2020-22": (2020, 2022), "2023-24": (2023, 2024), "2025-26": (2025, 2026)}
P_BAR = 0.05 / 40


def rsi(close: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + up / dn)
    r[dn == 0] = 100.0
    r[:n] = np.nan
    return r


def simulate(
    op: np.ndarray, ts_min: np.ndarray, r: np.ndarray, thr: float, hold: int | str
) -> list[tuple[int, int]]:
    sig = np.flatnonzero((r[:-1] >= thr) & (r[1:] < thr)) + 1  # bar t of the cross
    above50 = np.flatnonzero((r[:-1] <= 50) & (r[1:] > 50)) + 1
    trades: list[tuple[int, int]] = []
    free_from = 0
    last = len(op) - 1
    for t in sig:
        ent = t + 1
        if ent < free_from or ent > last:
            continue
        if hold == "rsi50":
            k = np.searchsorted(above50, ent)
            cross = above50[k] if k < len(above50) else last
            ex = min(cross + 1, ent + MAX_HOLD)
            n_exp = ex - ent
        else:
            assert isinstance(hold, int)
            ex = ent + hold
            n_exp = hold
        if ex > last:
            break
        free_from = ex
        if ts_min[ex] - ts_min[ent] > n_exp + 10:
            continue
        trades.append((ent, ex))
    return trades


def main(out_dir: str) -> None:
    bars = load_catalog_bars("data/catalog", "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL")
    assert bars is not None
    op = bars.open
    ts_min = (bars.ts // 60_000_000_000).astype(np.int64)
    years = pd.to_datetime(bars.ts, unit="ns").year.to_numpy()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline: dict[str, float] = {}
    for h in (15, 60, 240):
        g = op[h:] / op[:-h] - 1
        ok = (ts_min[h:] - ts_min[:-h]) <= h + 10
        baseline[str(h)] = float(g[ok].mean())

    rows = []
    for n in PERIODS:
        r = rsi(bars.close, n)
        for thr in THRESHOLDS:
            for hold in HOLDS:
                tr = simulate(op, ts_min, r, thr, hold)
                ent = np.array([a for a, _ in tr], dtype=np.int64)
                ex = np.array([b for _, b in tr], dtype=np.int64)
                gross = op[ex] / op[ent] - 1 if len(tr) else np.array([])
                net = gross - COST
                row: dict[str, object] = {
                    "rsi": n, "thr": thr, "hold": hold, "trades": len(tr),
                    "gross_mean": float(gross.mean()) if len(tr) else None,
                    "net_mean": float(net.mean()) if len(tr) else None,
                    "net_maker_mean": float((gross - COST_MAKER).mean()) if len(tr) else None,
                    "win_rate_net": float((net > 0).mean()) if len(tr) else None,
                    "net_compound": float(np.prod(1 + net) - 1) if len(tr) else None,
                    "baseline_gross": baseline.get(str(hold)),
                    "avg_bars": float((ex - ent).mean()) if len(tr) else None,
                }
                p = 1.0
                if len(tr) >= 2:
                    t_stat, p_two = stats.ttest_1samp(net, 0.0)
                    p = float(p_two / 2 if t_stat > 0 else 1 - p_two / 2)
                row["p_one_sided"] = p
                era_ok = True
                eras: dict[str, object] = {}
                for name, (y0, y1) in ERAS.items():
                    m = (years[ent] >= y0) & (years[ent] <= y1) if len(tr) else np.array([])
                    k = int(m.sum()) if len(tr) else 0
                    mean = float(net[m].mean()) if k else None
                    eras[name] = {"n": k, "net_mean": mean}
                    era_ok = era_ok and mean is not None and mean > 0
                row["eras"] = eras
                row["pass"] = bool(
                    len(tr) >= 100 and (row["net_mean"] or 0) > 0 and p < P_BAR and era_ok
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
    (out / "results.json").write_text(json.dumps({"baseline": baseline, "cells": rows}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
