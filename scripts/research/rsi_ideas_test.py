"""Pre-registered test of three RSI ideas on 8 Binance perps (2026-11-03).

Sources: rsi_dip_btc_sweep.py lead (Connors-style 4h RSI(2) + SMA200, picked on BTC)
and Zatwarnicki et al. 2023, Sensors, PMC9920669 (RSI(14) > 50 and 60..80 band,
long-only, crypto daily, sample 2018..2022). Scratch, NOT in any DSR count. Rules
below are frozen before the first run of this file.

Data: catalog 1-DAY and 4-HOUR LAST bars and archive raw_funding_rates for BTC, ETH,
SOL, BNB, NEAR, SUI, ZEC, HYPE (USDT perps), up to 2026-10-20. RSI = Wilder
(ewm alpha=1/n, adjust=False) on close. Cost 0.05% per side (0.10% round trip).
Funding: a long pays rate_s at each settlement s in (entry time, exit time].

A. Connors 4h (out-of-sample on the 7 coins other than BTC, each coin's full range):
   RSI(2) crosses below 30 at bar t close and close[t] > SMA200(close)[t]; buy at the
   open of t+1, sell at the open 12 bars later; one position at a time; a trade whose
   wall time exceeds 12 bars + 1 h is dropped. net = gross - 0.001 - funding.
   drift_h = mean gross of a long from EVERY bar's open held 12 bars (same coin).
   edge = net - drift_h. PASS A: pooled mean net > 0, pooled mean edge > 0 with
   month-cluster bootstrap p < 0.05 (10,000 draws, default_rng(20261103); p = share
   of resampled means <= 0), and mean edge > 0 on >= 5 of 7 coins. Report-only: hold 24.

B. RSI(14) > 50 long/flat, daily. C. 60 <= RSI(14) <= 80 long/flat, daily.
   Position for day t+1 (close t -> close t+1) is set from day t's close; cost 0.0005
   per unit of position change; funding over (close t, close t+1].
   Nulls: buy-and-hold (always long, funding paid, no cost), close > SMA20 and
   close > SMA50 long/flat (same cost and funding rules).
   Decision window: days 2023-01-01..2026-10-20 (after the paper's sample). Portfolio
   = each day, the mean daily return over coins with a bar that day and >= 60 prior
   bars. Sharpe = mean/std(ddof=1)*sqrt(365).
   PASS (each of B, C): portfolio Sharpe > B&H, > SMA20 null and > SMA50 null
   portfolio Sharpe; coin Sharpe > coin B&H Sharpe on >= 6 of 8 coins; and the
   portfolio Sharpe-difference vs B&H has a 20-day moving-block bootstrap
   p < 0.05 (5,000 draws, default_rng(20261104); p = share of resampled
   differences <= 0). Report-only: full range 2020+, and the same rules on 4h bars.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vibe_quant.validation.extraction import load_catalog_bars

COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "NEARUSDT", "SUIUSDT", "ZECUSDT",
         "HYPEUSDT")
SIDE_COST = 0.0005
DEC_START = pd.Timestamp("2023-01-01").value
END = pd.Timestamp("2026-10-21").value


def rsi(close: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + up / dn)
    r[dn == 0] = 100.0
    r[: n + 1] = np.nan
    return r


def load(sym: str, tf: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = load_catalog_bars("data/catalog", f"{sym}-PERP.BINANCE-{tf}-LAST-EXTERNAL")
    assert b is not None
    keep = b.ts < END
    return b.ts[keep], b.open[keep], b.close[keep]


def funding(con: sqlite3.Connection, sym: str) -> tuple[np.ndarray, np.ndarray]:
    rows = con.execute(
        "select funding_time, funding_rate from raw_funding_rates where symbol=? "
        "order by funding_time", (sym,)).fetchall()
    t = np.array([r[0] for r in rows], dtype=np.int64) * 1_000_000
    cum = np.concatenate(([0.0], np.cumsum([r[1] for r in rows])))
    return t, cum


def fund_sum(ft: np.ndarray, fcum: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sum of rates with a < s <= b (ns)."""
    return fcum[np.searchsorted(ft, b, "right")] - fcum[np.searchsorted(ft, a, "right")]


def cluster_p(x: np.ndarray, months: np.ndarray, rng: np.random.Generator) -> float:
    ids = np.unique(months)
    groups = [x[months == m] for m in ids]
    sums = np.array([g.sum() for g in groups])
    cnts = np.array([len(g) for g in groups])
    draws = rng.integers(0, len(ids), size=(10_000, len(ids)))
    means = sums[draws].sum(1) / cnts[draws].sum(1)
    return float((means <= 0).mean())


def test_a(con: sqlite3.Connection, hold: int) -> dict[str, object]:
    edges, nets, months = [], [], []
    per: dict[str, object] = {}
    for sym in COINS[1:]:
        ts, op, close = load(sym, "4-HOUR")
        ft, fcum = funding(con, sym)
        r2 = rsi(close, 2)
        sma = pd.Series(close).rolling(200).mean().to_numpy()
        sig = np.flatnonzero((r2[:-1] >= 30) & (r2[1:] < 30)) + 1
        sig = sig[close[sig] > sma[sig]]
        ent_l: list[int] = []
        free, last = 0, len(op) - 1
        for t in sig:
            e = t + 1
            if e < free:
                continue
            if e + hold > last:
                break
            free = e + hold
            if ts[e + hold] - ts[e] > (hold * 4 + 1) * 3_600_000_000_000:
                continue
            ent_l.append(e)
        ent = np.array(ent_l, dtype=np.int64)
        ex = ent + hold
        gross = op[ex] / op[ent] - 1
        fund = fund_sum(ft, fcum, ts[ent - 1], ts[ex - 1])  # open of bar k = close of k-1
        net = gross - 2 * SIDE_COST - fund
        drift = float((op[hold:] / op[:-hold] - 1).mean())
        edge = net - drift
        per[sym] = {"n": len(ent), "mean_net": float(net.mean()), "drift": drift,
                    "mean_edge": float(edge.mean()), "mean_funding": float(fund.mean())}
        edges.append(edge)
        nets.append(net)
        months.append(pd.to_datetime(ts[ent]).to_period("M").astype(str).to_numpy())
    e_all, n_all, m_all = np.concatenate(edges), np.concatenate(nets), np.concatenate(months)
    p = cluster_p(e_all, m_all, np.random.default_rng(20261103))
    coins_pos = sum(v["mean_edge"] > 0 for v in per.values())  # type: ignore[index]
    return {"hold": hold, "n": len(e_all), "mean_net": float(n_all.mean()),
            "mean_edge": float(e_all.mean()), "p_edge": p, "coins_edge_pos": coins_pos,
            "per_coin": per,
            "pass": bool(n_all.mean() > 0 and e_all.mean() > 0 and p < 0.05 and coins_pos >= 5)}


def daily_series(con: sqlite3.Connection, tf: str) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in COINS:
        ts, _, close = load(sym, tf)
        ft, fcum = funding(con, sym)
        r14 = rsi(close, 14)
        ret = np.full(len(close), np.nan)
        ret[1:] = close[1:] / close[:-1] - 1
        fund = np.zeros(len(close))
        fund[1:] = fund_sum(ft, fcum, ts[:-1], ts[1:])
        pos = {
            "B_rsi50": (r14 > 50).astype(float),
            "C_band60_80": ((r14 >= 60) & (r14 <= 80)).astype(float),
            "null_sma20": (close > pd.Series(close).rolling(20).mean().to_numpy()).astype(float),
            "null_sma50": (close > pd.Series(close).rolling(50).mean().to_numpy()).astype(float),
        }
        cols = {"bh": ret - fund}
        for k, p in pos.items():
            held = np.concatenate(([0.0], p[:-1]))  # set at close t, earns bar t+1
            chg = np.abs(np.diff(np.concatenate(([0.0], held))))
            cols[k] = held * (ret - fund) - SIDE_COST * chg
        df = pd.DataFrame(cols, index=ts)
        df = df.iloc[60:]  # >= 60 prior bars
        out[sym] = df.dropna()
    return out


def sharpe(x: np.ndarray, per_year: float) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(per_year))


def block_p(diff_a: np.ndarray, diff_b: np.ndarray, per_year: float) -> float:
    rng = np.random.default_rng(20261104)
    n, blk = len(diff_a), 20
    starts = rng.integers(0, n - blk + 1, size=(5_000, int(np.ceil(n / blk))))
    idx = (starts[:, :, None] + np.arange(blk)).reshape(5_000, -1)[:, :n]
    a, b = diff_a[idx], diff_b[idx]
    d = (a.mean(1) / a.std(1, ddof=1) - b.mean(1) / b.std(1, ddof=1)) * np.sqrt(per_year)
    return float((d <= 0).mean())


def test_bc(con: sqlite3.Connection, tf: str, per_year: float, start: int) -> dict[str, object]:
    series = daily_series(con, tf)
    cols = ["bh", "B_rsi50", "C_band60_80", "null_sma20", "null_sma50"]
    port = pd.concat({s: d[cols] for s, d in series.items()}, axis=1)
    port = port[port.index >= start]
    pm = pd.DataFrame({c: port.xs(c, axis=1, level=1).mean(axis=1) for c in cols}).dropna()
    res: dict[str, object] = {"days": len(pm)}
    psh = {c: sharpe(pm[c].to_numpy(), per_year) for c in cols}
    res["portfolio_sharpe"] = psh
    res["portfolio_ann_return"] = {c: float(pm[c].mean() * per_year) for c in cols}
    coin = {}
    for s, d in series.items():
        d = d[d.index >= start]
        coin[s] = {c: sharpe(d[c].to_numpy(), per_year) for c in cols} if len(d) > 30 else None
    res["coin_sharpe"] = coin
    for k in ("B_rsi50", "C_band60_80"):
        beats = sum(1 for v in coin.values() if v and v[k] > v["bh"])
        p = block_p(pm[k].to_numpy(), pm["bh"].to_numpy(), per_year)
        res[k] = {"coins_beat_bh": beats, "p_vs_bh": p, "pass": bool(
            psh[k] > psh["bh"] and psh[k] > psh["null_sma20"] and psh[k] > psh["null_sma50"]
            and beats >= 6 and p < 0.05)}
    return res


def main(out_dir: str) -> None:
    con = sqlite3.connect("file:data/archive/raw_data.db?mode=ro", uri=True)
    res = {
        "A_hold12": test_a(con, 12),
        "A_hold24_report": test_a(con, 24),
        "BC_daily_2023": test_bc(con, "1-DAY", 365, DEC_START),
        "BC_daily_full_report": test_bc(con, "1-DAY", 365, 0),
        "BC_4h_2023_report": test_bc(con, "4-HOUR", 365 * 6, DEC_START),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
