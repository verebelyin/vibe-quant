"""Universe rule for rsi_dip_newcoins.py (run with -I, arg = repo root).

Top 30 USDT perps by mean daily quote volume in 2026-01..2026-09 over local 1d files
(user 2026-11-03: rank on 2026 Binance volume; first version used 2025), needing
>= 250 days of 2026 data and data in 2026-09; excludes the 8 coins already tested,
TradFi / pre-market symbols and stablecoin / index symbols.
"""

import collections
import csv
import glob
import json
import os
import re
import sys

STABLE = re.compile(
    r"^(USDC|FDUSD|TUSD|BUSD|USDP|DAI|USDE|RLUSD|USD1|XUSD|BTCDOM|DEFI|PAXG|XAUT)USDT$"
)
OURS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "NEARUSDT", "SUIUSDT", "ZECUSDT",
        "HYPEUSDT"}
NON_COIN = ("EQUITY", "HK_EQUITY", "KR_EQUITY", "CN_EQUITY", "COMMODITY", "FX", "INDEX",
            "PREMARKET")


def main(root: str) -> None:
    dirs = [f"{root}/data/swarm/20261011-h9-raw", f"{root}/data/swarm/20261011-h10b-raw",
            f"{root}/data/swarm/20261102-listing-s2/raw/klines"]
    with open(f"{root}/scripts/research/listing_fade/stage2_exclusions.json") as fh:
        excl = json.load(fh)
    bad = set(excl["tradfi"]) | set(excl["pre_market"])
    with open(f"{root}/data/swarm/20261102-listing-s2/coverage/exchangeInfo.json") as fh:
        info = json.load(fh)
    for s in info.get("symbols", []):
        if s.get("contractType") == "TRADIFI_PERPETUAL" or s.get("underlyingType") in NON_COIN:
            bad.add(s["symbol"])
    vol: dict[str, dict[int, float]] = collections.defaultdict(dict)
    last: dict[str, str] = collections.defaultdict(str)
    for d in dirs:
        for f in glob.glob(d + "/*.csv"):
            m = re.match(r"(.+)-(\d{4})-(\d{2})\.csv$", os.path.basename(f))
            if not m:
                continue
            sym, y, mo = m.group(1), m.group(2), m.group(3)
            last[sym] = max(last[sym], y + mo)
            if y != "2026":
                continue
            with open(f, encoding="utf-8") as fh:
                for row in csv.reader(fh):
                    if row and row[0].isdigit():
                        vol[sym][int(row[0]) // 86_400_000] = float(row[7])
    rank = []
    for sym, dv in vol.items():
        if sym in OURS or sym in bad or STABLE.match(sym) or "_" in sym:
            continue
        if len(dv) >= 250 and last[sym] >= "202609":
            rank.append((sum(dv.values()) / len(dv), sym))
    rank.sort(reverse=True)
    print(json.dumps({"coins": [s for _, s in rank[:30]],
                      "vol_musd": [round(v / 1e6) for v, _ in rank[:30]]}))


if __name__ == "__main__":
    main(sys.argv[1])
