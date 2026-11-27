"""H-11 (pre-registered ee9f376): cross-sectional funding carry, out-of-universe coins, 2022-06..2026-09.
Usage: h11_funding_carry.py KLINES_DIR FUNDING_DIR START_CAL WINDOW_START WINDOW_END"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KDIR, FDIR = Path(sys.argv[1]), Path(sys.argv[2])
CAL0, W0, W1 = sys.argv[3], sys.argv[4], sys.argv[5]
FEE, LIQ, CAP = 0.0005, 10_000_000, 0.001
cols = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "qv",
    "n",
    "tbv",
    "tbqv",
    "ignore",
]
cal = pd.date_range(CAL0, W1, freq="D", tz="UTC")

close, qv, fund = {}, {}, {}
for sym in KDIR.joinpath("UNIVERSE.txt").read_text().split():
    parts = [
        p
        for f in sorted(KDIR.glob(f"{sym}-*.csv"))
        if len(p := pd.read_csv(f, header=None, names=cols))
    ]
    if not parts:
        continue
    d = pd.concat(parts)
    d = d[pd.to_numeric(d["open_time"], errors="coerce").notna()]
    idx = pd.DatetimeIndex(
        pd.to_datetime(d["open_time"].astype("int64"), unit="ms", utc=True)
    ).normalize()
    c = pd.Series(d["close"].astype(float).values, index=idx)
    v = pd.Series(d["volume"].astype(float).values, index=idx)
    q = pd.Series(d["qv"].astype(float).values, index=idx)
    keep = ~idx.duplicated()
    c, v, q = c[keep], v[keep], q[keep]
    close[sym] = c.where(v > 0)  # zero-volume day = no data (delisted frozen bars)
    qv[sym] = q.where(v > 0)
    fparts = [pd.read_csv(f) for f in sorted(FDIR.glob(f"{sym}-*.csv"))]
    if fparts:
        fd = pd.concat(fparts)
        fd = fd[pd.to_numeric(fd["calc_time"], errors="coerce").notna()]
        ts = pd.to_datetime(fd["calc_time"].astype("int64"), unit="ms", utc=True)
        fund[sym] = pd.Series(fd["last_funding_rate"].astype(float).values, index=ts).sort_index()

syms = list(close)
px = pd.DataFrame(close).reindex(cal)
ret = px.pct_change(fill_method=None)
liquid = pd.DataFrame(qv).reindex(cal).rolling(7, min_periods=7).mean() >= LIQ
# daily funding sum per coin (settlements on that UTC day), raw and capped
fday = (
    pd.DataFrame({s: f.groupby(f.index.normalize()).sum() for s, f in fund.items()})
    .reindex(cal, columns=syms)
    .fillna(0.0)
)
fday_cap = (
    pd.DataFrame({s: f.clip(-CAP, CAP).groupby(f.index.normalize()).sum() for s, f in fund.items()})
    .reindex(cal, columns=syms)
    .fillna(0.0)
)
# signal at Monday t: settlements in [t-6d 00:00, t+1d 00:00) = days t-6..t
sig = (
    fday.rolling(7, min_periods=7)
    .sum()
    .where(px.notna().rolling(7, min_periods=7).sum() == 7)
    .where(liquid)
)

w = pd.DataFrame(0.0, index=cal, columns=syms)
cur = pd.Series(0.0, index=syms)
rankable = []
for t in cal:
    if t.dayofweek == 0:
        m = sig.loc[t].dropna()
        if t >= pd.Timestamp(W0, tz="UTC"):
            rankable.append(len(m))
        k = len(m) // 3
        if k >= 1:
            r = m.sort_values()
            cur = pd.Series(0.0, index=syms)
            cur[r.index[:k]] = 0.5 / k  # lowest funding: long
            cur[r.index[-k:]] = -0.5 / k  # highest funding: short
    cur = cur.where(px.loc[t].notna(), 0.0)
    w.loc[t] = cur
held = w.shift(1).fillna(0.0)
price = (held * ret.fillna(0.0)).sum(axis=1)
funding = -(held * fday).sum(axis=1)
funding_cap = -(held * fday_cap).sum(axis=1)
cost = FEE * (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
bh = ret.mean(axis=1, skipna=True)


def stats(r: pd.Series, label: str) -> None:
    r = r.loc[W0:W1]
    sh = float(r.mean() / r.std(ddof=1) * np.sqrt(365))
    eq = (1 + r).cumprod()
    b = bh.reindex(r.index).fillna(0.0)
    beta = float(np.cov(r, b, ddof=1)[0, 1] / np.var(b, ddof=1))
    print(
        f"{label}: sharpe={sh} total_return={float(eq.iloc[-1] - 1)} maxDD={float((1 - eq / eq.cummax()).max())} beta={beta}"
    )


stats(price + funding - cost, "NET (price+funding-cost)")
stats(price + funding_cap - cost, "NET capped funding +-0.1%")
stats(price - cost, "price-only net")
stats(funding, "funding-only")
print(
    "annual funding carry",
    float(funding.loc[W0:W1].sum() / ((pd.Timestamp(W1) - pd.Timestamp(W0)).days / 365)),
    "rankable mean",
    float(np.mean(rankable)),
    "min",
    int(np.min(rankable)),
    "coins",
    len(syms),
    "with funding",
    len(fund),
)
