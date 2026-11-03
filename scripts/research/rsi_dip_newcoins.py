"""Pre-registered: user RSI dip rule on 30 coins it was never run on (2026-11-03).

Frozen before any of these coins' 15m data was downloaded. Scratch research, NOT in
any DSR count.

Origin: user rule "RSI drops below 15 -> long, sell at RSI 30", tried at 5x then 2x
on BTC ETH SOL BNB NEAR SUI ZEC HYPE (rsi_2026.py + journal 2026-11-03). On those 8
coins, exits RSI >= 40..60 looked better than 30 at 15m; exit 50 is the primary
here because it was the only exit with all 8 coins up in 2026 (picked after seeing
data, which is why the test is on new coins).

Universe (rule fixed before these coins' 15m data was used; amended 2026-11-03 by the
user to rank on 2026 volume, before any run): top 30 USDT perps by mean daily quote
volume 2026-01..2026-09 (>= 250 days of 2026 data, data in 2026-09) over local 1d
files (rsi_newcoins_universe.py), excluding the 8 coins above, TradFi / pre-market
(stage2_exclusions.json + exchangeInfo) and stablecoin / index symbols:
XRP DOGE 1000PEPE RIVER ADA LAB WLD TAO ENA LINK RAVE AVAX BCH UNI SIREN PUMP TRUMP
FIL PIPPIN AAVE LTC ASTER DASH XLM DOT AKE BEAT ONDO BANK ARB (+USDT).

Data: Binance Vision UM monthly 15m klines + fundingRate, every listed month up to
2026-09 (fetch_newcoins.py). Rows parsed as in stage 1 (header skipped, first row per
open_time wins).

Rule: RSI(14) Wilder (ewm alpha=1/14, adjust=False) on 15m close. Signal when RSI
crosses below 15 at bar t close; long at open of t+1; exit at the open after the first
close (from the entry bar on) with RSI >= X; one position at a time per coin. A trade
whose wall time from entry to exit exceeds (bars + 2) * 15 min is dropped (data gap).
2x isolated, all equity of the coin's own account per trade; liquidation if any bar
low from the entry bar to the exit-signal bar is <= entry * 0.505 -> trade = -100%.
Margin return = 2 * (exit/entry - 1 - 0.001 - funding), funding = sum of rates at
settlements in (entry time, exit time].
Windows by entry time: OLD = coin start..2025-12-31, NEW = 2026-01-01..2026-09-30.

PASS (X = 50): pooled per-trade margin mean > 0 in OLD and in NEW; month-cluster
bootstrap over OLD+NEW trades (10,000 draws, default_rng(20261105), p = share of
resampled means <= 0) < 0.05; in NEW >= 60% of coins with trades end with a
positive account and the equal-split account (mean of coin account returns) > 0.
Report-only: X = 30, 40, 60, 70; 1h bars resampled from 15m (open first, low min,
close last, only full 4-bar hours); equal-weight 2026 B&H of the 30 coins.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COINS = ("XRPUSDT", "DOGEUSDT", "1000PEPEUSDT", "RIVERUSDT", "ADAUSDT", "LABUSDT", "WLDUSDT",
         "TAOUSDT", "ENAUSDT", "LINKUSDT", "RAVEUSDT", "AVAXUSDT", "BCHUSDT",
         "UNIUSDT", "SIRENUSDT", "PUMPUSDT", "TRUMPUSDT", "FILUSDT", "PIPPINUSDT",
         "AAVEUSDT", "LTCUSDT", "ASTERUSDT", "DASHUSDT", "XLMUSDT", "DOTUSDT",
         "AKEUSDT", "BEATUSDT", "ONDOUSDT", "BANKUSDT", "ARBUSDT")
NEW_START = int(pd.Timestamp("2026-01-01").value // 1_000_000)
NEW_END = int(pd.Timestamp("2026-10-01").value // 1_000_000)
EXITS = (50, 30, 40, 60, 70)
LEV = 2
LIQ = 1 - 1 / LEV + 0.005
COST = 0.001


def read_klines(kdir: Path, sym: str) -> pd.DataFrame:
    rows: dict[int, tuple[float, float, float]] = {}
    for f in sorted(kdir.glob(f"{sym}-????-??.csv")):
        with f.open(encoding="utf-8") as fh:
            for r in csv.reader(fh):
                if not r or not r[0].isdigit():
                    continue
                t = int(r[0])
                if t not in rows:
                    rows[t] = (float(r[1]), float(r[3]), float(r[4]))
    t = np.array(sorted(rows), dtype=np.int64)
    v = np.array([rows[k] for k in t])
    return pd.DataFrame({"open": v[:, 0], "low": v[:, 1], "close": v[:, 2]}, index=t)


def read_funding(fdir: Path, sym: str) -> tuple[np.ndarray, np.ndarray]:
    rows: dict[int, float] = {}
    for f in sorted(fdir.glob(f"{sym}-????-??.csv")):
        with f.open(encoding="utf-8") as fh:
            for r in csv.reader(fh):
                if not r or not r[0].isdigit():
                    continue
                rows.setdefault(int(r[0]), float(r[2]))
    t = np.array(sorted(rows), dtype=np.int64)
    return t, np.concatenate(([0.0], np.cumsum([rows[k] for k in t])))


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + up / dn)
    r[dn == 0] = 100.0
    r[: n + 1] = np.nan
    return r


def to_1h(df: pd.DataFrame) -> pd.DataFrame:
    h = df.index // 3_600_000 * 3_600_000
    g = df.groupby(h)
    out = pd.DataFrame({"open": g["open"].first(), "low": g["low"].min(),
                        "close": g["close"].last(), "n": g.size()})
    return out[out["n"] == 4].drop(columns="n")


def trades(df: pd.DataFrame, ft: np.ndarray, fc: np.ndarray, x_thr: int, bar_ms: int):  # type: ignore[no-untyped-def]
    ts = df.index.to_numpy()
    op, lo, cl = (df[c].to_numpy() for c in ("open", "low", "close"))
    r = rsi(cl)
    sig = np.flatnonzero((r[:-1] >= 15) & (r[1:] < 15)) + 1
    ge = np.flatnonzero(r >= x_thr)
    out: list[tuple[int, float]] = []
    free, last = 0, len(op) - 1
    for t in sig:
        e = t + 1
        if e < free:
            continue
        k = np.searchsorted(ge, e)
        if k >= len(ge) or ge[k] + 1 > last:
            break
        u = ge[k]
        x = u + 1
        free = x
        if ts[x] - ts[e] > (x - e + 2) * bar_ms:
            continue
        f = fc[np.searchsorted(ft, ts[x], "right")] - fc[np.searchsorted(ft, ts[e], "right")]
        g = op[x] / op[e] - 1 - COST - f
        m = -1.0 if lo[e : u + 1].min() <= op[e] * LIQ else LEV * g
        out.append((int(ts[e]), float(m)))
    return out


def summarize(per: dict[str, list[tuple[int, float]]], lo_t: int, hi_t: int) -> dict[str, object]:
    accts, allm = {}, []
    for sym, tr in per.items():
        m = np.array([v for t, v in tr if lo_t <= t < hi_t])
        if len(m):
            accts[sym] = float(np.prod(1 + m) - 1)
            allm.append(m)
    a = np.concatenate(allm) if allm else np.array([])
    return {"n": len(a), "mean": float(a.mean()) if len(a) else None,
            "win": float((a > 0).mean()) if len(a) else None, "liqs": int((a == -1).sum()),
            "coins": len(accts), "coins_up": sum(v > 0 for v in accts.values()),
            "split_account": float(np.mean(list(accts.values()))) if accts else None,
            "accounts": accts}


def cluster_p(per: dict[str, list[tuple[int, float]]]) -> float:
    rows = [(t, v) for tr in per.values() for t, v in tr if t < NEW_END]
    months = pd.to_datetime([t for t, _ in rows], unit="ms").to_period("M").astype(str)
    x = np.array([v for _, v in rows])
    ids, inv = np.unique(np.asarray(months), return_inverse=True)
    sums = np.bincount(inv, weights=x)
    cnts = np.bincount(inv)
    rng = np.random.default_rng(20261105)
    d = rng.integers(0, len(ids), size=(10_000, len(ids)))
    return float(((sums[d].sum(1) / cnts[d].sum(1)) <= 0).mean())


def main(raw: str, out_path: str) -> None:
    root = Path(raw)
    res: dict[str, object] = {}
    bh = {}
    data = {}
    for sym in COINS:
        df = read_klines(root / "klines", sym)
        ft, fc = read_funding(root / "funding", sym)
        data[sym] = (df, ft, fc)
        new = df[(df.index >= NEW_START) & (df.index < NEW_END)]
        if len(new):
            bh[sym] = float(new["close"].iloc[-1] / new["open"].iloc[0] - 1)
    res["bh_2026"] = {"per_coin": bh, "equal_weight_mean": float(np.mean(list(bh.values())))}
    for tf, bar_ms in (("15m", 900_000), ("1h", 3_600_000)):
        for x in EXITS:
            per = {}
            for sym, (df, ft, fc) in data.items():
                d = df if tf == "15m" else to_1h(df)
                per[sym] = trades(d, ft, fc, x, bar_ms)
            old = summarize(per, 0, NEW_START)
            new = summarize(per, NEW_START, NEW_END)
            cell: dict[str, object] = {"old": old, "new": new, "p_cluster": cluster_p(per)}
            if tf == "15m" and x == 50:
                cell["pass"] = bool(
                    (old["mean"] or 0) > 0 and (new["mean"] or 0) > 0  # type: ignore[operator]
                    and cell["p_cluster"] < 0.05  # type: ignore[operator]
                    and new["coins_up"] >= 0.6 * new["coins"]  # type: ignore[operator]
                    and (new["split_account"] or 0) > 0)  # type: ignore[operator]
            res[f"{tf}_exit{x}"] = cell
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
