"""RSI-dip buy sweep on BTCUSDT over timeframes and RSI settings (user, 2026-11-03).

Follow-up to rsi_dip_btc1m.py (0/40 at 1m). Scratch research, NOT evidence, NOT in
any DSR count; BTC bars were seen before.

Frozen before the first run:
- Data: catalog BTCUSDT-PERP.BINANCE-{1,5,15}-MINUTE / 1-HOUR / 4-HOUR -LAST-EXTERNAL.
- RSI Wilder (ewm alpha=1/n, adjust=False) on close, n in 2,3,4,5,7,9,14,21,28.
- Signal: RSI crosses below thr at bar t close, thr in 5,10,15,20,25,30,35;
  filter none or close[t] > SMA200(close)[t] (Connors-style). Long at open of t+1,
  one position at a time.
- Exit: hold 1,3,6,12,24,48,96 bars (exit at open), or RSI crossing above 50 / 70
  (exit next open, max 200 bars). Trade dropped if its wall time exceeds
  1.1 * expected + 10 min (data gap).
- Cost 0.10% round trip (decision); 0.04% maker report-only.
- Train = entries 2020-01-01..2023-12-31, test = 2024-01-01..2026-10-20.
- Selection on TRAIN only: trades >= 100, net mean > 0, one-sided t p < 0.05 / N_cells.
- Test once on the selected cells: pass = test trades >= 50, net mean > 0,
  one-sided t p < 0.05 / max(1, n_selected).
- Also report the 15 best train cells by train t-stat (>= 100 trades) with their test,
  report-only.
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from vibe_quant.validation.extraction import load_catalog_bars

TFS = {"1m": ("1-MINUTE", 1), "5m": ("5-MINUTE", 5), "15m": ("15-MINUTE", 15),
       "1h": ("1-HOUR", 60), "4h": ("4-HOUR", 240)}
PERIODS = (2, 3, 4, 5, 7, 9, 14, 21, 28)
THRESHOLDS = (5, 10, 15, 20, 25, 30, 35)
FILTERS = ("none", "sma200")
HOLDS: tuple[int | str, ...] = (1, 3, 6, 12, 24, 48, 96, "rsi50", "rsi70")
MAX_HOLD = 200
COST = 0.0010
COST_MAKER = 0.0004
SPLIT_NS = pd.Timestamp("2024-01-01").value
N_CELLS = len(TFS) * len(PERIODS) * len(THRESHOLDS) * len(FILTERS) * len(HOLDS)


def rsi(close: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + up / dn)
    r[dn == 0] = 100.0
    r[: n + 1] = np.nan
    return r


def simulate(op, ts_min, tf_min, sig, cross_up, hold):  # type: ignore[no-untyped-def]
    ents: list[int] = []
    exs: list[int] = []
    free_from = 0
    last = len(op) - 1
    for t in sig:
        ent = t + 1
        if ent < free_from:
            continue
        if ent > last:
            break
        if isinstance(hold, int):
            ex = ent + hold
        else:
            k = np.searchsorted(cross_up, ent)
            c = cross_up[k] if k < len(cross_up) else last
            ex = min(c + 1, ent + MAX_HOLD)
        if ex > last:
            break
        free_from = ex
        if ts_min[ex] - ts_min[ent] > 1.1 * (ex - ent) * tf_min + 10:
            continue
        ents.append(ent)
        exs.append(ex)
    return np.array(ents, dtype=np.int64), np.array(exs, dtype=np.int64)


def p_one(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or x.std() == 0:
        return 0.0, 1.0
    t, p2 = stats.ttest_1samp(x, 0.0)
    return float(t), float(p2 / 2 if t > 0 else 1 - p2 / 2)


def summary(gross: np.ndarray) -> dict[str, float | int | None]:
    net = gross - COST
    t, p = p_one(net)
    n = len(gross)
    return {"n": n, "gross": float(gross.mean()) if n else None,
            "net": float(net.mean()) if n else None,
            "net_maker": float((gross - COST_MAKER).mean()) if n else None,
            "t": t, "p": p}


def run_tf(tf: str) -> list[dict[str, object]]:
    name, tf_min = TFS[tf]
    bars = load_catalog_bars("data/catalog", f"BTCUSDT-PERP.BINANCE-{name}-LAST-EXTERNAL")
    assert bars is not None
    op, close, ts = bars.open, bars.close, bars.ts
    ts_min = (ts // 60_000_000_000).astype(np.int64)
    sma = pd.Series(close).rolling(200).mean().to_numpy()
    above = close > sma  # NaN -> False
    rows: list[dict[str, object]] = []
    for n in PERIODS:
        r = rsi(close, n)
        up50 = np.flatnonzero((r[:-1] <= 50) & (r[1:] > 50)) + 1
        up70 = np.flatnonzero((r[:-1] <= 70) & (r[1:] > 70)) + 1
        for thr in THRESHOLDS:
            raw = np.flatnonzero((r[:-1] >= thr) & (r[1:] < thr)) + 1
            for flt in FILTERS:
                sig = raw[above[raw]] if flt == "sma200" else raw
                for hold in HOLDS:
                    cu = up70 if hold == "rsi70" else up50
                    ent, ex = simulate(op, ts_min, tf_min, sig, cu, hold)
                    gross = op[ex] / op[ent] - 1
                    tr = ts[ent] < SPLIT_NS
                    rows.append({"tf": tf, "rsi": n, "thr": thr, "filter": flt, "hold": hold,
                                 "train": summary(gross[tr]), "test": summary(gross[~tr])})
    print(f"{tf} done {len(rows)}", file=sys.stderr, flush=True)
    return rows


def main(out_dir: str) -> None:
    with Pool(len(TFS)) as pool:
        rows = [r for part in pool.map(run_tf, list(TFS)) for r in part]
    assert len(rows) == N_CELLS
    sel_bar = 0.05 / N_CELLS
    selected = [r for r in rows if r["train"]["n"] >= 100 and (r["train"]["net"] or 0) > 0  # type: ignore[index,operator]
                and r["train"]["p"] < sel_bar]  # type: ignore[index,operator]
    test_bar = 0.05 / max(1, len(selected))
    for r in selected:
        te = r["test"]
        r["pass"] = bool(te["n"] >= 50 and (te["net"] or 0) > 0 and te["p"] < test_bar)  # type: ignore[index,operator]
    top = sorted((r for r in rows if r["train"]["n"] >= 100), key=lambda r: -r["train"]["t"])[:15]  # type: ignore[index,operator]
    res = {"n_cells": N_CELLS, "select_p_bar": sel_bar, "n_selected": len(selected),
           "test_p_bar": test_bar, "n_pass": sum(bool(r.get("pass")) for r in selected),
           "selected": selected, "top15_train": top}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells.json").write_text(json.dumps(rows))
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("selected", "top15_train")}))


if __name__ == "__main__":
    main(sys.argv[1])
