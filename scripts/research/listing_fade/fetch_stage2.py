"""Stage-2 data fetch for H-20261029-1 (new-listing fade), approved by the user 2026-11-02.

Two steps, run with `python -I`:
  coverage <out_dir>  list Binance Vision UM monthly 1d-kline and fundingRate files per
                      USDT symbol (file names only) and the Binance CMS new-listing titles.
                      No prices are read.
  exclusions <cov> <dest>  write the frozen tradfi / pre_market symbol sets.
  download <cov> <excl> <out>  download the files SPEC_IMPL_STAGE2.md names into
                      <out>/{klines,funding,klines_fill}/ (unzipped CSVs).

Downloaded content is data only: it is never executed or imported.
"""

from __future__ import annotations

import io
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi

CTX = ssl.create_default_context(cafile=certifi.where())
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
VISION = "https://data.binance.vision/"
CMS = (
    "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
    "?type=1&catalogId=48&pageNo={page}&pageSize=50"
)
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
LAST_MONTH = "2026-09"
UA = {"User-Agent": "vibe-quant-research/1.0"}


def _get(url: str, tries: int = 8) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=60, context=CTX
            ) as r:
                return bytes(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if k == tries - 1:
                raise
        except (urllib.error.URLError, OSError):
            if k == tries - 1:
                raise
        time.sleep(1.5 * (k + 1))
    raise RuntimeError("unreachable")


def _s3_list(prefix: str, delimiter: bool) -> tuple[list[str], list[str]]:
    """Return (common prefixes, keys) under prefix, following pagination."""
    prefixes: list[str] = []
    keys: list[str] = []
    marker = ""
    while True:
        url = f"{S3}?prefix={urllib.parse.quote(prefix)}" + ("&delimiter=/" if delimiter else "")
        if marker:
            url += f"&marker={urllib.parse.quote(marker)}"
        root = ET.fromstring(_get(url))
        prefixes += [p.text or "" for p in root.findall("s3:CommonPrefixes/s3:Prefix", NS)]
        keys += [k.text or "" for k in root.findall("s3:Contents/s3:Key", NS)]
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=NS)) == "true"
        if not truncated:
            return prefixes, keys
        marker = root.findtext("s3:NextMarker", default="", namespaces=NS) or (keys or prefixes)[-1]


MONTH_RE = re.compile(r"-(\d{4}-\d{2})\.zip$")


def _months(keys: list[str]) -> list[str]:
    return sorted({m.group(1) for k in keys if (m := MONTH_RE.search(k))})


def coverage(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    prefixes, _ = _s3_list("data/futures/um/monthly/klines/", delimiter=True)
    symbols = sorted(
        p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes if p.rstrip("/").endswith("USDT")
    )
    symbols = [s for s in symbols if "_" not in s]

    cache = out / "vision_files.partial.jsonl"
    done: dict[str, dict[str, list[str]]] = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            row = json.loads(line)
            done[row["sym"]] = {"klines": row["klines"], "funding": row["funding"]}

    def one(sym: str) -> tuple[str, list[str], list[str]]:
        _, kk = _s3_list(f"data/futures/um/monthly/klines/{sym}/1d/", delimiter=False)
        _, fk = _s3_list(f"data/futures/um/monthly/fundingRate/{sym}/", delimiter=False)
        return sym, _months(kk), _months(fk)

    with ThreadPoolExecutor(8) as ex, cache.open("a") as fh:
        for sym, k, f in ex.map(one, [s for s in symbols if s not in done]):
            done[sym] = {"klines": k, "funding": f}
            fh.write(json.dumps({"sym": sym, "klines": k, "funding": f}) + "\n")
            fh.flush()
    files = {s: done[s] for s in symbols}
    (out / "vision_files.json").write_text(json.dumps(files, indent=1, sort_keys=True) + "\n")

    articles: list[dict[str, object]] = []
    for page in range(1, 21):
        body = json.loads(_get(CMS.format(page=page)))
        catalogs = (body.get("data") or {}).get("catalogs") or []
        got = [a for c in catalogs for a in (c.get("articles") or [])]
        if not got:
            break
        articles += [
            {"code": a.get("code"), "title": a.get("title"), "releaseDate": a.get("releaseDate")}
            for a in got
        ]
        time.sleep(0.5)
    (out / "cms_titles.json").write_text(json.dumps(articles, indent=1) + "\n")
    print(json.dumps({"symbols": len(symbols), "articles": len(articles)}))


TOKEN_RE = re.compile(r"[0-9A-Z]+USDT")
TRADFI_TITLE = ("tradfi", "pre-ipo", "composite index", "equity", "index perpetual")
NON_COIN = {"EQUITY", "HK_EQUITY", "KR_EQUITY", "CN_EQUITY", "COMMODITY", "FX", "INDEX"}


def _month_shift(m: str, k: int) -> str:
    y, mm = map(int, m.split("-"))
    t = y * 12 + mm - 1 + k
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def exclusions(cov: Path, dest: Path) -> None:
    """Frozen non-crypto / pre-market symbol sets for stage 2 (rule in SPEC_IMPL_STAGE2.md)."""
    import datetime as dt

    files = json.loads((cov / "vision_files.json").read_text())
    arts = json.loads((cov / "cms_exclusion_articles.json").read_text())
    info = json.loads((cov / "exchangeInfo.json").read_text())["symbols"]
    tradfi: set[str] = set()
    pre_market: set[str] = set()
    for a in arts.values():
        title = str(a["title"])
        low = title.lower()
        if "futures" not in low:
            continue
        rm = dt.datetime.fromtimestamp(int(a["releaseDate"]) / 1000, dt.UTC).strftime("%Y-%m")
        named = {t for t in TOKEN_RE.findall(title) if t in files}
        if not named:
            named = {t for t in TOKEN_RE.findall(str(a["body"])) if t in files}
        named = {
            t for t in named if _month_shift(rm, -1) <= files[t]["klines"][0] <= _month_shift(rm, 1)
        }
        if any(k in low for k in TRADFI_TITLE):
            tradfi |= named
        elif "pre-market" in low:
            pre_market |= named
    for row in info:
        sym = row["symbol"]
        if sym not in files:
            continue
        if row.get("contractType") == "TRADIFI_PERPETUAL" or row.get("underlyingType") in NON_COIN:
            tradfi.add(sym)
        elif row.get("underlyingType") == "PREMARKET":
            pre_market.add(sym)
    pre_market -= tradfi
    dest.write_text(
        json.dumps(
            {"tradfi": sorted(tradfi), "pre_market": sorted(pre_market)},
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps({"tradfi": len(tradfi), "pre_market": len(pre_market)}))


LOCAL_KLINES = ("data/swarm/20261011-h9-raw", "data/swarm/20261011-h10b-raw")
FILE_RE = re.compile(r"^(.+)-(\d{4})-(\d{2})\.csv$")


def _plan(cov: Path, excl: Path, repo: Path) -> dict[str, dict[str, list[str]]]:
    """S2 (2024-01+ first file, not local, not tradfi) and P (pre-2024, not local)."""
    files = json.loads((cov / "vision_files.json").read_text())
    tradfi = set(json.loads(excl.read_text())["tradfi"])
    local = {
        m.group(1)
        for d in LOCAL_KLINES
        for n in (repo / d).iterdir()
        if (m := FILE_RE.match(n.name))
    }
    plan: dict[str, dict[str, list[str]]] = {"klines": {}, "funding": {}}
    for sym, v in sorted(files.items()):
        first = v["klines"][0]
        if sym in local or first > LAST_MONTH:
            continue
        if first >= "2024-01" and sym in tradfi:
            continue
        plan["klines"][sym] = [m for m in v["klines"] if m <= LAST_MONTH]
        plan["funding"][sym] = [m for m in v["funding"] if m <= LAST_MONTH]
    return plan


def _fetch_jobs(
    out: Path, jobs: list[tuple[str, str, str]]
) -> list[tuple[tuple[str, str, str], str]]:
    def one(job: tuple[str, str, str]) -> tuple[tuple[str, str, str], str]:
        kind, sym, m = job
        dest = out / kind / f"{sym}-{m}.csv"
        if dest.exists():
            return job, "cached"
        q = urllib.parse.quote(sym)
        if kind == "klines":
            url = f"{VISION}data/futures/um/monthly/klines/{q}/1d/{q}-1d-{m}.zip"
        elif kind == "funding":
            url = f"{VISION}data/futures/um/monthly/fundingRate/{q}/{q}-fundingRate-{m}.zip"
        else:
            url = f"{VISION}data/futures/um/daily/klines/{q}/1d/{q}-1d-{m}.zip"
        try:
            raw = _get(url)
        except urllib.error.HTTPError as e:
            return job, f"http {e.code}"
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist() if n.endswith(".csv") and "/" not in n]
            if len(names) != 1:
                return job, f"zip members {z.namelist()}"
            data = z.read(names[0])
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(dest)
        return job, "ok"

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(one, jobs))


def _interior_missing(kdir: Path) -> list[tuple[str, str]]:
    import datetime as dt

    days: dict[str, set[int]] = {}
    for f in kdir.iterdir():
        m = FILE_RE.match(f.name)
        if m is None:
            continue
        got = days.setdefault(m.group(1), set())
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            first = line.split(",", 1)[0]
            if first.isdigit():
                t = int(first)
                got.add((t // 1000 if t >= 100_000_000_000_000 else t) // 86_400_000)
    out: list[tuple[str, str]] = []
    for sym, ds in sorted(days.items()):
        if not ds:
            continue
        for d in range(min(ds) + 1, max(ds)):
            if d not in ds:
                out.append((sym, (dt.date(1970, 1, 1) + dt.timedelta(days=d)).isoformat()))
    return out


def download(cov: Path, excl: Path, out: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    plan = _plan(cov, excl, repo)
    jobs = [
        (kind, sym, m)
        for kind in ("klines", "funding")
        for sym, ms in plan[kind].items()
        for m in ms
    ]
    for kind in ("klines", "funding", "klines_fill"):
        (out / kind).mkdir(parents=True, exist_ok=True)
    res = _fetch_jobs(out, jobs)
    missing = _interior_missing(out / "klines")
    fill = _fetch_jobs(out, [("klines_fill", sym, d) for sym, d in missing])
    bad = [[*j, s] for j, s in res if s not in ("ok", "cached")]
    log = {
        "symbols": len(plan["klines"]),
        "jobs": len(jobs),
        "bad": bad,
        "interior_missing_days": len(missing),
        "fill": {
            "ok": sum(1 for _, s in fill if s in ("ok", "cached")),
            "missing": [[*j, s] for j, s in fill if s not in ("ok", "cached")],
        },
    }
    (out / "download_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {k: (len(v) if isinstance(v, list) else v) for k, v in log.items() if k != "fill"}
            | {"fill_ok": log["fill"]["ok"], "fill_missing": len(log["fill"]["missing"])}
        )
    )


if __name__ == "__main__":
    if sys.argv[1] == "coverage":
        coverage(Path(sys.argv[2]))
    elif sys.argv[1] == "exclusions":
        exclusions(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "download":
        download(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit(
            "usage: coverage <out> | exclusions <cov> <dest> | download <cov> <exclusions.json> <out>"
        )
