"""R1 metrics: Binance Vision USDT-M daily metrics zips -> one Parquet per symbol + coverage.json.
Usage: fetch_metrics.py OUT_DIR START(YYYY-MM-DD) END(YYYY-MM-DD) SYM [SYM ...]

URL: https://data.binance.vision/data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-{YYYY-MM-DD}.zip
Each zip holds one 5-minute CSV (288 rows/day). A missing day is a 404: recorded, not fatal.
Raw zips are cached under OUT_DIR/raw/{SYM}/ so a re-run is idempotent (no network for cached days).
"""

import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

NUM = [
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]
COLS = ["create_time", "symbol"] + NUM

OUT, START, END, SYMS = (
    Path(sys.argv[1]),
    date.fromisoformat(sys.argv[2]),
    date.fromisoformat(sys.argv[3]),
    sys.argv[4:],
)
BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
RAW = OUT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
DAYS = [START + timedelta(days=i) for i in range((END - START).days + 1)]
CLIENT = httpx.Client(timeout=60, follow_redirects=True, limits=httpx.Limits(max_connections=32))


def is_ts(cell: str) -> bool:
    try:
        pd.Timestamp(cell.strip())
        return True
    except Exception:
        return False


def download(sym: str, day: date) -> tuple[date, str]:
    """Fetch one day's zip into the cache. Returns (day, status) with status 'ok' | '404' | 'error:...'."""
    d = day.isoformat()
    dst = RAW / sym / f"{sym}-metrics-{d}.zip"
    if dst.exists() and dst.stat().st_size > 0:
        return day, "ok"
    dst.parent.mkdir(parents=True, exist_ok=True)
    err = ""
    for attempt in range(3):
        try:
            r = CLIENT.get(f"{BASE}/{sym}/{sym}-metrics-{d}.zip")
            if r.status_code == 404:
                return day, "404"
            if r.status_code == 200:
                tmp = dst.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.replace(dst)
                return day, "ok"
            err = f"HTTP {r.status_code}"
        except Exception as e:  # network error / timeout
            err = f"{type(e).__name__}: {e}"
        if attempt < 2:
            time.sleep(1.5 * (2**attempt))
    return day, f"error:{err}"


def read_zip(path: Path) -> pd.DataFrame:
    """Read one cached zip. Header presence is detected from the first cell; empty cells stay NaN."""
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8-sig", errors="replace")
    header = None if is_ts(raw.split("\n", 1)[0].split(",")[0]) else 0
    df = pd.read_csv(io.StringIO(raw), header=header, names=COLS, usecols=range(len(COLS)))
    df["ts"] = pd.to_datetime(df["create_time"], errors="coerce", utc=True).astype(
        "datetime64[ns, UTC]"
    )
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df[["ts"] + NUM].dropna(subset=["ts"])


def rows_per_day_stats(df: pd.DataFrame) -> dict:
    counts = df.groupby(df["ts"].dt.date).size()
    return {"min": int(counts.min()), "median": float(counts.median()), "max": int(counts.max())}


def coverage(sym: str, df: pd.DataFrame, days404: list[str], dropped: int) -> dict:
    ts = df["ts"].reset_index(drop=True)
    counts = df.groupby(df["ts"].dt.date).size()
    short = counts[counts < 288]
    diff = ts.diff()
    gi = list((diff > pd.Timedelta(hours=1)).to_numpy().nonzero()[0])
    biggest = sorted(gi, key=lambda i: diff.iloc[i], reverse=True)[:10]
    return {
        "symbol": sym,
        "first_ts": ts.iloc[0].isoformat() if len(ts) else None,
        "last_ts": ts.iloc[-1].isoformat() if len(ts) else None,
        "days_requested": len(DAYS),
        "days_404": days404,
        "rows": int(len(df)),
        "duplicate_rows_dropped": dropped,
        "rows_per_day": rows_per_day_stats(df) if len(df) else {},
        "days_with_fewer_than_288_rows": {
            "count": int(len(short)),
            "first_20_dates": [d.isoformat() for d in short.index[:20]],
        },
        "nan_count": {c: int(df[c].isna().sum()) for c in NUM},
        "zero_count": {c: int((df[c] == 0).sum()) for c in NUM},
        "gaps_over_1h": {
            "count": int(len(gi)),
            "largest": [
                {"start": ts.iloc[i - 1].isoformat(), "end": ts.iloc[i].isoformat()}
                for i in biggest
            ],
        },
    }


def build_symbol(sym: str) -> dict:
    statuses = list(ThreadPoolExecutor(max_workers=16).map(lambda d: download(sym, d), DAYS))
    ok = [d for d, s in statuses if s == "ok"]
    d404 = [d.isoformat() for d, s in statuses if s == "404"]
    errors = [(d.isoformat(), s) for d, s in statuses if s.startswith("error")]

    frames = []
    for day in ok:
        p = RAW / sym / f"{sym}-metrics-{day.isoformat()}.zip"
        try:
            frames.append(read_zip(p))
        except Exception as e:
            errors.append((day.isoformat(), f"parse:{type(e).__name__}: {e}"))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ts"] + NUM)
    dropped = int(df["ts"].duplicated().sum()) if len(df) else 0
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="first").reset_index(drop=True)

    df.to_parquet(OUT / f"{sym}.parquet", index=False)
    cov = coverage(sym, df, d404, dropped)
    print(
        f"{sym}: rows={cov['rows']} first={cov['first_ts']} last={cov['last_ts']} "
        f"days_404={len(d404)} dup_dropped={dropped} short_days={cov['days_with_fewer_than_288_rows']['count']} "
        f"gaps>1h={cov['gaps_over_1h']['count']}"
    )
    return {"coverage": cov, "errors": errors}


def main() -> None:
    t0 = time.time()
    out, errors = {}, {}
    for sym in SYMS:
        r = build_symbol(sym)
        out[sym] = r["coverage"]
        if r["errors"]:
            errors[sym] = r["errors"]
    (OUT / "coverage.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}/coverage.json ({len(out)} symbols) in {time.time() - t0:.1f}s")
    if errors:
        print(f"FAILED days (non-404): {json.dumps(errors)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
