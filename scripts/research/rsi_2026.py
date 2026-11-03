"""RSI rules on 2026 data only, 8 perps, 1m..1d (user, 2026-11-03).

Exploratory, NOT pre-registered, NOT in any DSR count: 2026 bars were already
inside earlier RSI tests (rsi_ideas_test.py, the BTC sweeps).

Window: entries 2026-01-01..2026-10-20 (indicators warmed up on all earlier bars).
Coins: BTC ETH SOL BNB NEAR SUI ZEC HYPE (USDT perps, catalog LAST bars, archive
funding). Cost 0.05% per side. A long pays funding at settlements in (entry, exit].

U (user rule): RSI(n) crosses below 15 at bar t close -> long at open t+1, 5x
isolated, all equity per trade; exit at the open after the first close with
RSI >= 30. Liquidation if a bar low from entry to the exit signal bar is
<= entry * (1 - 1/5 + 0.005): the trade returns -100% of its margin.
Margin return = 5 * (exit/entry - 1 - 0.001 - funding). n = 14 (user default),
also 7 and 2. Also reported at 1x.
A: RSI(2) crosses below 30 and close > SMA200 -> long at next open, hold 12 bars, 1x.
B: RSI(14) > 50 long/flat. C: 60 <= RSI(14) <= 80 long/flat. Nulls B&H and
close > SMA50, all 1x, equal-weight across coins.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from vibe_quant.validation.extraction import load_catalog_bars

# Sibling script import: runpy (tests/unit/test_scripts_import_smoke.py) does not put the
# script's own directory on sys.path, so add it explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rsi_ideas_test import fund_sum, funding, rsi  # noqa: E402

COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "NEARUSDT", "SUIUSDT", "ZECUSDT",
         "HYPEUSDT")
TFS = {"1m": ("1-MINUTE", 525_600), "5m": ("5-MINUTE", 105_120), "15m": ("15-MINUTE", 35_040),
       "1h": ("1-HOUR", 8_760), "4h": ("4-HOUR", 2_190), "1d": ("1-DAY", 365)}
START = pd.Timestamp("2026-01-01").value
END = pd.Timestamp("2026-10-21").value
LEV = 5
LIQ = 1 - 1 / LEV + 0.005
COST = 0.001


def load(sym: str, tf: str):  # type: ignore[no-untyped-def]
    b = load_catalog_bars("data/catalog", f"{sym}-PERP.BINANCE-{tf}-LAST-EXTERNAL")
    assert b is not None
    k = b.ts < END
    return b.ts[k], b.open[k], b.low[k], b.close[k]


def user_rule(ts, op, lo, close, ft, fc, n):  # type: ignore[no-untyped-def]
    r = rsi(close, n)
    sig = np.flatnonzero((r[:-1] >= 15) & (r[1:] < 15)) + 1
    sig = sig[ts[sig] >= START]
    ge30 = np.flatnonzero(r >= 30)
    gross, liq, bars_held = [], [], []
    free, last = 0, len(op) - 1
    for t in sig:
        e = t + 1
        if e < free:
            continue
        k = np.searchsorted(ge30, e)
        if k >= len(ge30) or ge30[k] + 1 > last:
            break
        u = ge30[k]
        x = u + 1
        free = x
        f = float(fund_sum(ft, fc, ts[e - 1 : e], ts[x - 1 : x])[0])
        gross.append(op[x] / op[e] - 1 - COST - f)
        liq.append(bool(lo[e : u + 1].min() <= op[e] * LIQ))
        bars_held.append(x - e)
    return np.array(gross), np.array(liq, dtype=bool), np.array(bars_held)


def rule_a(ts, op, close, ft, fc):  # type: ignore[no-untyped-def]
    r2 = rsi(close, 2)
    sma = pd.Series(close).rolling(200).mean().to_numpy()
    sig = np.flatnonzero((r2[:-1] >= 30) & (r2[1:] < 30)) + 1
    sig = sig[(close[sig] > sma[sig]) & (ts[sig] >= START)]
    nets = []
    free, last = 0, len(op) - 1
    for t in sig:
        e = t + 1
        if e < free:
            continue
        if e + 12 > last:
            break
        free = e + 12
        f = float(fund_sum(ft, fc, ts[e - 1 : e], ts[e + 11 : e + 12])[0])
        nets.append(op[e + 12] / op[e] - 1 - COST - f)
    m = ts[:-12] >= START
    drift = float((op[12:][m] / op[:-12][m] - 1).mean())
    return np.array(nets), drift


def positions(ts, close, ft, fc):  # type: ignore[no-untyped-def]
    r14 = rsi(close, 14)
    ret = np.zeros(len(close))
    ret[1:] = close[1:] / close[:-1] - 1
    fund = np.zeros(len(close))
    fund[1:] = fund_sum(ft, fc, ts[:-1], ts[1:])
    sma50 = pd.Series(close).rolling(50).mean().to_numpy()
    pos = {"B_rsi50": r14 > 50, "C_band": (r14 >= 60) & (r14 <= 80), "null_sma50": close > sma50}
    cols = {"bh": ret - fund}
    for k, p in pos.items():
        held = np.concatenate(([0.0], p[:-1].astype(float)))
        chg = np.abs(np.diff(np.concatenate(([0.0], held))))
        cols[k] = held * (ret - fund) - COST / 2 * chg
    df = pd.DataFrame(cols, index=ts)
    return df[df.index >= START]


def stats_u(g: np.ndarray, liq: np.ndarray, lev: int) -> dict[str, object]:
    if len(g) == 0:
        return {"n": 0}
    m = np.where(liq, -1.0, lev * g) if lev > 1 else g
    return {"n": len(g), "win": float((m > 0).mean()), "mean": float(m.mean()),
            "liqs": int(liq.sum()) if lev > 1 else 0,
            "compound": float(np.prod(1 + np.maximum(m, -1)) - 1)}


def run_tf(tf: str) -> dict[str, object]:
    name, per_year = TFS[tf]
    con = sqlite3.connect("file:data/archive/raw_data.db?mode=ro", uri=True)
    out: dict[str, object] = {"U": {}, "A": {}}
    frames = {}
    for sym in COINS:
        ts, op, lo, close = load(sym, name)
        ft, fc = funding(con, sym)
        m = ts >= START
        bh = float(close[m][-1] / op[m][0] - 1)
        u_res = {}
        for n in (14, 7, 2):
            g, liq, held = user_rule(ts, op, lo, close, ft, fc, n)
            u_res[f"rsi{n}"] = {"x5": stats_u(g, liq, LEV), "x1": stats_u(g, liq, 1),
                                "avg_bars": float(held.mean()) if len(held) else None}
        out["U"][sym] = {"bh_2026": bh, **u_res}  # type: ignore[index]
        nets, drift = rule_a(ts, op, close, ft, fc)
        out["A"][sym] = {"n": len(nets), "mean_net": float(nets.mean()) if len(nets) else None,  # type: ignore[index]
                         "drift": drift}
        frames[sym] = positions(ts, close, ft, fc)
    port = pd.concat(frames, axis=1)
    cols = ["bh", "B_rsi50", "C_band", "null_sma50"]
    pm = pd.DataFrame({c: port.xs(c, axis=1, level=1).mean(axis=1) for c in cols}).dropna()
    out["BC"] = {c: {"sharpe": float(pm[c].mean() / pm[c].std(ddof=1) * np.sqrt(per_year)),
                     "ret": float(np.prod(1 + pm[c]) - 1)} for c in cols}
    print(f"{tf} done", file=sys.stderr, flush=True)
    return {"tf": tf, **out}


def main(out_dir: str) -> None:
    with Pool(len(TFS)) as pool:
        res = pool.map(run_tf, list(TFS))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
