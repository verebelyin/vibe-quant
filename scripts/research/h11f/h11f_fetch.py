"""H-11F forward data: Binance public REST (no key) -> Binance-Vision-format CSVs for the frozen universe.
Usage: h11f_fetch.py OUT_KLINES_DIR OUT_FUNDING_DIR START(YYYY-MM-DD) END(YYYY-MM-DD) UNIVERSE_FILE"""

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

KO, FO, START, END, UNI = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    sys.argv[3],
    sys.argv[4],
    Path(sys.argv[5]),
)
KO.mkdir(parents=True, exist_ok=True)
FO.mkdir(parents=True, exist_ok=True)


def ms(d: str) -> int:
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


t0, t1 = ms(START), ms(END) + 86_400_000 - 1
API = "https://fapi.binance.com"
c = httpx.Client(timeout=30)
syms = UNI.read_text().split()
(KO / "UNIVERSE.txt").write_text("\n".join(syms) + "\n")
kcols = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
for s in syms:
    k = c.get(
        f"{API}/fapi/v1/klines",
        params={"symbol": s, "interval": "1d", "startTime": t0, "endTime": t1, "limit": 1500},
    )
    k.raise_for_status()
    kd = pd.DataFrame(k.json(), columns=kcols)
    kd = kd[kd["close_time"].astype("int64") <= int(time.time() * 1000)]  # closed bars only
    rows, start = [], t0
    while True:
        f = c.get(
            f"{API}/fapi/v1/fundingRate",
            params={"symbol": s, "startTime": start, "endTime": t1, "limit": 1000},
        )
        f.raise_for_status()
        j = f.json()
        rows += j
        if len(j) < 1000:
            break
        start = j[-1]["fundingTime"] + 1
    fd = pd.DataFrame(
        {
            "calc_time": [r["fundingTime"] for r in rows],
            "funding_interval_hours": "",
            "last_funding_rate": [r["fundingRate"] for r in rows],
        }
    )
    for ym, g in kd.groupby(
        pd.to_datetime(kd["open_time"].astype("int64"), unit="ms", utc=True).dt.strftime("%Y-%m")
    ):
        g.to_csv(KO / f"{s}-{ym}.csv", index=False)
    if len(fd):
        for ym, g in fd.groupby(
            pd.to_datetime(fd["calc_time"].astype("int64"), unit="ms", utc=True).dt.strftime(
                "%Y-%m"
            )
        ):
            g.to_csv(FO / f"{s}-{ym}.csv", index=False)
    print(s, len(kd), "bars", len(fd), "funding")
