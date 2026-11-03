"""Download Binance Vision UM monthly 15m klines + fundingRate for rsi_dip_newcoins.py.

    .venv/bin/python -I scripts/research/fetch_newcoins.py <coverage vision_files.json> <out>

Months = the coin's listed 1d-kline / funding months up to 2026-09 (from the stage-2
coverage file). Downloaded content is data only: never executed or imported.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "listing_fade"))
from fetch_stage2 import VISION, _get  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rsi_dip_newcoins import COINS  # noqa: E402

LAST = "2026-09"


def main(cov: str, out: str) -> None:
    files = json.loads(Path(cov).read_text())
    root = Path(out)
    jobs = []
    for sym in COINS:
        for kind in ("klines", "funding"):
            (root / kind).mkdir(parents=True, exist_ok=True)
            for m in files[sym][kind]:
                if m <= LAST:
                    jobs.append((kind, sym, m))

    def one(job: tuple[str, str, str]) -> str:
        kind, sym, m = job
        dest = root / kind / f"{sym}-{m}.csv"
        if dest.exists():
            return "cached"
        q = urllib.parse.quote(sym)
        url = (f"{VISION}data/futures/um/monthly/klines/{q}/15m/{q}-15m-{m}.zip" if kind == "klines"
               else f"{VISION}data/futures/um/monthly/fundingRate/{q}/{q}-fundingRate-{m}.zip")
        try:
            raw = _get(url)
        except urllib.error.HTTPError as e:
            return f"http {e.code} {sym} {kind} {m}"
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist() if n.endswith(".csv") and "/" not in n]
            if len(names) != 1:
                return f"zip members {sym} {m} {z.namelist()}"
            data = z.read(names[0])
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(dest)
        return "ok"

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, jobs))
    summary = {"jobs": len(jobs), "ok": res.count("ok"), "cached": res.count("cached"),
               "errors": [r for r in res if r not in ("ok", "cached")]}
    (root / "download_log.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
