"""Research-tier download: 1h/1d LAST klines + funding for universe coins only.

Every archive is a tmp file and every request goes to an in-process fake of
Binance Vision + the futures REST API (``httpx.MockTransport``), so the tests
see exactly which URLs the command asks for and which rows it writes.
"""

from __future__ import annotations

import io
import json
import sqlite3
import zipfile
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from vibe_quant.data import archive as archive_mod
from vibe_quant.data import downloader, ingest, universe
from vibe_quant.data.archive import RawDataArchive

if TYPE_CHECKING:
    from pathlib import Path

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
STEP_MS = {"1h": HOUR_MS, "1d": DAY_MS}
GB = 1024**3


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _day_floor(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


NOW = datetime.now(UTC)
# XRP: 70 days of history; NEW: 40 days ("short history"); both start mid-month.
LISTINGS = {
    "XRPUSDT": _ms(_day_floor(NOW - timedelta(days=70))),
    "NEWUSDT": _ms(_day_floor(NOW - timedelta(days=40))),
    "BTCUSDT": _ms(datetime(2019, 9, 8, tzinfo=UTC)),
}


def _kline_row(t: int, step: int, price: float = 100.0, volume: float = 5.0) -> list[Any]:
    return [t, price, price + 1, price - 1, price + 0.5, volume, t + step - 1, 500.0, 7, 2.0, 200.0]


class FakeBinance:
    """Vision monthly zips + REST klines/funding + exchangeInfo for LISTINGS."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.fail: dict[str, int] = {}  # URL substring -> status served on every attempt

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        for needle, status in self.fail.items():
            if needle in url:
                return httpx.Response(status)
        path = request.url.path
        if path == "/fapi/v1/exchangeInfo":
            info = {
                "symbols": [
                    {
                        "symbol": s,
                        "contractType": "PERPETUAL",
                        "status": "TRADING",
                        "baseAsset": s.removesuffix("USDT"),
                        "quoteAsset": "USDT",
                        "underlyingType": "COIN",
                        "onboardDate": ms,
                    }
                    for s, ms in LISTINGS.items()
                ]
            }
            return httpx.Response(200, content=json.dumps(info).encode())
        if path == "/fapi/v1/klines":
            p = request.url.params
            return self._rest_klines(
                p["symbol"], p["interval"], int(p["startTime"]), int(p["endTime"]), int(p["limit"])
            )
        if path == "/fapi/v1/fundingRate":
            p = request.url.params
            rows = [
                {"symbol": p["symbol"], "fundingTime": t, "fundingRate": "0.0001", "markPrice": "1"}
                for t in self._funding_times(p["symbol"], int(p["startTime"]), int(p["endTime"]))
            ][: int(p["limit"])]
            return httpx.Response(200, content=json.dumps(rows).encode())
        if "/monthly/klines/" in path:
            return self._vision_klines(path)
        if "/monthly/fundingRate/" in path:
            return self._vision_funding(path)
        return httpx.Response(404)

    @staticmethod
    def _bars(symbol: str, interval: str, start: int, end: int) -> list[list[Any]]:
        """Bars with open_time in [start, end] from the listing, up to the live bar."""
        step = STEP_MS[interval]
        t = max(start, LISTINGS[symbol])
        t = -(-t // step) * step
        now = _ms(datetime.now(UTC))
        rows = []
        while t <= end and t <= now:
            rows.append(_kline_row(t, step))
            t += step
        return rows

    def _rest_klines(
        self, symbol: str, interval: str, start: int, end: int, limit: int
    ) -> httpx.Response:
        rows = self._bars(symbol, interval, start, end)[:limit]
        return httpx.Response(200, content=json.dumps(rows).encode())

    @staticmethod
    def _month(name: str) -> tuple[int, int]:
        year, month = name.rsplit(".", 1)[0].split("-")[-2:]
        return int(year), int(month)

    @staticmethod
    def _month_bounds(year: int, month: int) -> tuple[int, int]:
        start = datetime(year, month, 1, tzinfo=UTC)
        nxt = datetime(year + month // 12, month % 12 + 1, 1, tzinfo=UTC)
        return _ms(start), _ms(nxt) - 1

    @staticmethod
    def _zip(name: str, body: str) -> httpx.Response:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(name.replace(".zip", ".csv"), body)
        return httpx.Response(200, content=buf.getvalue())

    def _vision_klines(self, path: str) -> httpx.Response:
        _, symbol, interval, name = path.rsplit("/", 3)
        lo, hi = self._month_bounds(*self._month(name))
        if hi < LISTINGS[symbol]:
            return httpx.Response(404)
        rows = self._bars(symbol, interval, lo, hi)
        body = "open_time,o,h,l,c,v,close_time,qv,n,tbv,tbqv,ignore\n" + "".join(
            ",".join(str(x) for x in r) + ",0\n" for r in rows
        )
        return self._zip(name, body)

    def _funding_times(self, symbol: str, start: int, end: int) -> list[int]:
        step = 8 * HOUR_MS
        t = -(-max(start, LISTINGS[symbol]) // step) * step
        now = _ms(datetime.now(UTC))
        out = []
        while t <= end and t <= now:
            out.append(t)
            t += step
        return out

    def _vision_funding(self, path: str) -> httpx.Response:
        _, symbol, name = path.rsplit("/", 2)
        lo, hi = self._month_bounds(*self._month(name))
        times = self._funding_times(symbol, lo, hi)
        if not times:
            return httpx.Response(404)
        body = "calc_time,funding_interval_hours,last_funding_rate\n" + "".join(
            f"{t},8,0.0001\n" for t in times
        )
        return self._zip(name, body)

    def asked(self, needle: str) -> list[str]:
        return [str(r.url) for r in self.requests if needle in str(r.url)]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeBinance:
    server = FakeBinance()
    real_client = httpx.Client

    def client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(server.handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.setattr(downloader.time, "sleep", lambda _s: None)
    uni = tmp_path / "universe_top20.txt"
    uni.write_text(
        "# vibe-quant research universe\n# as-of: 2026-11-21\nBTCUSDT\nXRPUSDT\nNEWUSDT\n"
    )
    monkeypatch.setattr(universe, "UNIVERSE_FILE", uni)
    monkeypatch.setattr(archive_mod, "DEFAULT_ARCHIVE_PATH", tmp_path / "raw_data.db")
    return server


@pytest.fixture
def seeded(fake: FakeBinance, tmp_path: Path) -> Path:
    """Archive holding a full-tier BTCUSDT (1m klines, mark, funding)."""
    path = tmp_path / "raw_data.db"
    arc = RawDataArchive(path)
    t0 = _ms(datetime(2024, 1, 1, tzinfo=UTC))
    arc.insert_klines(
        "BTCUSDT", "1m", [_kline_row(t0 + i * 60_000, 60_000) for i in range(50)], "seed"
    )
    arc.insert_mark_klines("BTCUSDT", "1m", [_kline_row(t0, 60_000)], "seed")
    arc.insert_funding_rates("BTCUSDT", [(t0, 0.0001, 1.0)], "seed")
    arc.close()
    return path


def _snapshot(path: Path) -> dict[str, Any]:
    """Row counts + max(id) per table and per (symbol, interval)."""
    if not path.exists():
        return {}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        snap: dict[str, Any] = {}
        for table in (
            "raw_klines",
            "raw_mark_klines",
            "raw_funding_rates",
            "raw_funding_intervals",
        ):
            snap[table] = con.execute(f"SELECT COUNT(*), MAX(id) FROM {table}").fetchone()
        snap["by_key"] = con.execute(
            "SELECT symbol, interval, COUNT(*), COUNT(DISTINCT open_time), MIN(open_time), "
            "MAX(open_time), MAX(id) FROM raw_klines GROUP BY 1, 2 ORDER BY 1, 2"
        ).fetchall()
        snap["funding"] = con.execute(
            "SELECT symbol, COUNT(*), COUNT(DISTINCT funding_time) FROM raw_funding_rates "
            "GROUP BY 1 ORDER BY 1"
        ).fetchall()
        snap["intervals"] = con.execute(
            "SELECT symbol, COUNT(*) FROM raw_funding_intervals GROUP BY 1 ORDER BY 1"
        ).fetchall()
        return snap
    finally:
        con.close()


def _key(snap: dict[str, Any], symbol: str, interval: str) -> tuple[Any, ...] | None:
    return next((r for r in snap["by_key"] if r[0] == symbol and r[1] == interval), None)


def _download_requests(fake: FakeBinance) -> list[str]:
    return [u for u in (str(r.url) for r in fake.requests) if "exchangeInfo" not in u]


def test_dry_run_prints_estimate_and_writes_nothing(
    seeded: Path, fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _snapshot(seeded)
    assert ingest.main(["download-research", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert _snapshot(seeded) == before
    assert _download_requests(fake) == []
    assert "estimate" in out.lower() and "free disk" in out.lower()
    assert "selection-biased" in out
    btc = next(ln for ln in out.splitlines() if "BTCUSDT" in ln)
    assert "full tier" in btc
    new = next(ln for ln in out.splitlines() if "NEWUSDT" in ln)
    assert "short history" in new
    xrp = next(ln for ln in out.splitlines() if "XRPUSDT" in ln)
    assert "short history" in xrp  # 70 days is short too (< 1 year)


@pytest.mark.parametrize("answer", ["n", "", "no", "yes please", "Y "])
def test_answer_other_than_yes_aborts_without_writing(
    seeded: Path,
    fake: FakeBinance,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
) -> None:
    before = _snapshot(seeded)
    monkeypatch.setattr("builtins.input", lambda _prompt="": answer)
    code = ingest.main(["download-research"])
    out = capsys.readouterr().out
    if answer.strip().lower() in {"y", "yes"}:
        assert code == 0
        assert _snapshot(seeded) != before
        return
    assert code == 0
    assert "aborted" in out
    assert _snapshot(seeded) == before
    assert _download_requests(fake) == []


def test_missing_archive_dry_run_creates_no_file(fake: FakeBinance, tmp_path: Path) -> None:
    assert ingest.main(["download-research", "--dry-run"]) == 0
    assert not (tmp_path / "raw_data.db").exists()


def _estimate_bytes() -> int:
    return sum(
        p["est_bytes"]
        for p in ingest.plan_research_tier(
            ["BTCUSDT", "XRPUSDT", "NEWUSDT"],
            {s: LISTINGS[s] for s in ("XRPUSDT", "NEWUSDT", "BTCUSDT")},
            archive_mod.DEFAULT_ARCHIVE_PATH,
            datetime.now(UTC),
        )
    )


def test_free_disk_below_estimate_plus_headroom_refuses(
    seeded: Path,
    fake: FakeBinance,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    est = _estimate_bytes()
    assert est > 0
    before = _snapshot(seeded)
    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda _p: (10 * GB, 0, est + 5 * GB - 1))
    assert ingest.main(["download-research", "--yes"]) == 1
    assert "refus" in capsys.readouterr().err.lower()
    assert _snapshot(seeded) == before
    assert _download_requests(fake) == []

    # exactly estimate + 5 GB free is enough
    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda _p: (10 * GB, 0, est + 5 * GB))
    assert ingest.main(["download-research", "--yes"]) == 0


def test_download_writes_1h_1d_funding_from_listing_and_skips_full_tier(
    seeded: Path, fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _snapshot(seeded)
    started = datetime.now(UTC)
    assert ingest.main(["download-research", "--yes"]) == 0
    out = capsys.readouterr().out
    snap = _snapshot(seeded)

    # Full-tier BTC: no request, rows and ids untouched.
    assert fake.asked("BTCUSDT") == []
    for interval in ("1m",):
        assert _key(snap, "BTCUSDT", interval) == _key(before, "BTCUSDT", interval)
    assert snap["raw_mark_klines"] == before["raw_mark_klines"]

    last_closed_hour = _ms(started.replace(minute=0, second=0, microsecond=0)) - HOUR_MS
    for sym in ("XRPUSDT", "NEWUSDT"):
        listing = LISTINGS[sym]
        h = _key(snap, sym, "1h")
        d = _key(snap, sym, "1d")
        assert h is not None and d is not None
        # contiguous from the listing bar to the last CLOSED bar, no duplicates
        assert h[4] == listing and h[2] == h[3] == (h[5] - listing) // HOUR_MS + 1
        assert h[5] >= last_closed_hour
        assert h[5] + HOUR_MS <= _ms(datetime.now(UTC))  # live bar dropped
        assert d[4] == listing and d[2] == d[3] == (d[5] - listing) // DAY_MS + 1
        assert d[5] + DAY_MS <= _ms(datetime.now(UTC))
        assert _key(snap, sym, "1m") is None
        assert _key(snap, sym, "5m") is None
    # no MARK anywhere for research coins
    assert fake.asked("markPrice") == [] and fake.asked("/1m/") == []
    assert {r[0] for r in snap["funding"]} == {"BTCUSDT", "XRPUSDT", "NEWUSDT"}
    xrp_funding = next(r for r in snap["funding"] if r[0] == "XRPUSDT")
    assert xrp_funding[1] == xrp_funding[2] >= 70 * 3
    assert {r[0] for r in snap["intervals"]} >= {"XRPUSDT", "NEWUSDT"}
    assert "measured" in out.lower()

    # Rerun: nothing new, nothing duplicated.
    assert ingest.main(["download-research", "--yes"]) == 0
    again = _snapshot(seeded)
    for sym in ("XRPUSDT", "NEWUSDT"):
        for interval in ("1h", "1d"):
            row, row2 = _key(snap, sym, interval), _key(again, sym, interval)
            assert row is not None and row2 is not None
            # same first bar + same max id for old rows; count == distinct
            assert row2[4] == row[4] and row2[2] == row2[3]
    assert again["funding"] == [(s, c, c2) for s, c, c2 in again["funding"] if c == c2]


def test_include_full_plans_full_tier_coins_for_download(
    seeded: Path, fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ingest.main(["download-research", "--dry-run"]) == 0
    assert "BTCUSDT          full tier: skipped" in capsys.readouterr().out
    before = _snapshot(seeded)
    assert ingest.main(["download-research", "--dry-run", "--include-full"]) == 0
    out = capsys.readouterr().out
    btc = next(line for line in out.splitlines() if "BTCUSDT" in line)
    assert "listed" in btc and "estimate" in btc and "full tier" not in btc
    assert _snapshot(seeded) == before
    assert _download_requests(fake) == []


@pytest.mark.parametrize("status", [429, 503])
def test_vision_errors_retry_then_resumable_failure(
    seeded: Path,
    fake: FakeBinance,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    listing = datetime.fromtimestamp(LISTINGS["XRPUSDT"] / 1000, UTC)
    month = f"XRPUSDT-1h-{listing.year}-{listing.month:02d}.zip"
    fake.fail[month] = status
    assert ingest.main(["download-research", "--yes"]) == 1
    captured = capsys.readouterr()
    assert "re-run" in (captured.out + captured.err).lower()
    assert len(fake.asked(month)) == len(downloader.MONTHLY_RETRY_DELAYS) + 1
    snap = _snapshot(seeded)
    con = sqlite3.connect(f"file:{seeded}?mode=ro", uri=True)
    lo = LISTINGS["XRPUSDT"]
    nxt = datetime(listing.year + listing.month // 12, listing.month % 12 + 1, 1, tzinfo=UTC)
    in_month = con.execute(
        "SELECT COUNT(*) FROM raw_klines WHERE symbol='XRPUSDT' AND interval='1h' "
        "AND open_time >= ? AND open_time < ?",
        (lo, _ms(nxt)),
    ).fetchone()[0]
    con.close()
    assert in_month == 0  # no partial month
    assert _key(snap, "NEWUSDT", "1h") is not None  # other coins still done

    # Rerun after the outage completes and leaves no duplicates.
    fake.fail.clear()
    assert ingest.main(["download-research", "--yes"]) == 0
    done = _snapshot(seeded)
    h = _key(done, "XRPUSDT", "1h")
    assert h is not None
    assert h[4] == lo and h[2] == h[3] == (h[5] - lo) // HOUR_MS + 1


def test_funding_rest_failure_writes_no_partial_funding(
    seeded: Path, fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.fail["fundingRate?symbol=XRPUSDT"] = 429
    assert ingest.main(["download-research", "--yes"]) == 1
    snap = _snapshot(seeded)
    assert "XRPUSDT" not in {r[0] for r in snap["funding"]}
    assert len(fake.asked("fundingRate?symbol=XRPUSDT")) == len(downloader.MONTHLY_RETRY_DELAYS) + 1
    fake.fail.clear()
    assert ingest.main(["download-research", "--yes"]) == 0
    xrp = next(r for r in _snapshot(seeded)["funding"] if r[0] == "XRPUSDT")
    assert xrp[1] == xrp[2] >= 70 * 3


def test_update_requests_no_1m_for_a_research_coin(
    fake: FakeBinance, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Reason this passes on base too: it pins that SUPPORTED_SYMBOLS stays the
    # full-tier list after the research tier lands.
    from vibe_quant.data import catalog as catalog_mod

    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", tmp_path / "catalog")
    seeded = tmp_path / "raw_data.db"
    arc = RawDataArchive(seeded)
    t0 = LISTINGS["XRPUSDT"]
    arc.insert_klines(
        "XRPUSDT", "1h", [_kline_row(t0 + i * HOUR_MS, HOUR_MS) for i in range(5)], "seed"
    )
    arc.insert_funding_rates("XRPUSDT", [(t0, 0.0001, 1.0)], "seed")
    arc.close()
    ingest.update_all(verbose=False)
    assert fake.asked("XRPUSDT") == []
    assert "XRPUSDT" not in downloader.SUPPORTED_SYMBOLS
    assert _key(_snapshot(seeded), "XRPUSDT", "1m") is None


def _seed_research(
    path: Path,
    symbol: str,
    days: int,
    *,
    frozen: set[int],
    gap_hours: set[int],
    jump_day: int | None = None,
    daily_days: int | None = None,
    funding_skip: set[int] | None = None,
    intervals: bool = True,
) -> None:
    """1h + 1d klines from 2025-01-01, 8h funding over the same span, Jan intervals."""
    arc = RawDataArchive(path)
    t0 = _ms(datetime(2025, 1, 1, tzinfo=UTC))
    hours = [_kline_row(t0 + i * HOUR_MS, HOUR_MS) for i in range(days * 24) if i not in gap_hours]
    arc.insert_klines(symbol, "1h", hours, "seed")
    daily = []
    for i in range(days if daily_days is None else daily_days):
        price = 100.0 if jump_day is None or i < jump_day else 1000.0
        if i in frozen:
            row = [
                t0 + i * DAY_MS,
                price,
                price,
                price,
                price,
                0.0,
                t0 + (i + 1) * DAY_MS - 1,
                0.0,
                0,
                0.0,
                0.0,
            ]
        else:
            row = _kline_row(t0 + i * DAY_MS, DAY_MS, price=price)
            row[1] = row[4] = price  # open == previous close
        daily.append(row)
    arc.insert_klines(symbol, "1d", daily, "seed")
    skip = funding_skip or set()
    settlements = [t0 + i * 8 * HOUR_MS for i in range(days * 3) if i not in skip]
    arc.insert_funding_rates(symbol, [(t, 0.0001, 1.0) for t in settlements], "seed")
    if intervals:
        arc.insert_funding_intervals(symbol, [(t0, 8)], "seed")
    arc.close()


def test_verify_reports_frozen_daily_bars_as_no_data_not_an_error(
    seeded: Path, fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_research(seeded, "XRPUSDT", 10, frozen={3, 4}, gap_hours=set())
    assert ingest.main(["verify", "--symbols", "XRPUSDT"]) == 0
    out = capsys.readouterr().out
    assert "research tier" in out
    assert "NO DATA" in out and "2025-01-04" in out and "2025-01-05" in out
    assert "symbol not found" not in out


def test_verify_flags_research_gaps_and_ticker_reuse(
    seeded: Path, fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_research(seeded, "XRPUSDT", 10, frozen=set(), gap_hours={30, 31})
    _seed_research(seeded, "NEWUSDT", 10, frozen=set(), gap_hours=set(), jump_day=6)
    assert ingest.main(["verify", "--symbols", "XRPUSDT"]) == 1
    out = capsys.readouterr().out
    assert "GAPS" in out and "2025-01-02T05:00:00+00:00" in out
    assert ingest.main(["verify", "--symbols", "NEWUSDT"]) == 1
    out = capsys.readouterr().out
    assert "TICKER REUSE" in out and "2025-01-07" in out


def test_status_lists_research_coins_with_ranges(
    seeded: Path,
    fake: FakeBinance,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibe_quant.data import catalog as catalog_mod

    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", tmp_path / "catalog")
    _seed_research(seeded, "XRPUSDT", 3, frozen=set(), gap_hours=set())
    assert ingest.main(["status"]) == 0
    out = capsys.readouterr().out
    block = out.split("XRPUSDT:")[1]
    assert "research tier" in block
    assert "1h: 72" in block and "1d: 3" in block and "funding: 9" in block
    assert "2025-01-01" in block and "2025-01-03T23:00:00" in block


# --------------------------------------------------------------- coverage (B2)

B2_LISTING = datetime(2025, 1, 1, tzinfo=UTC)
B2_NOW = datetime(2025, 4, 2, tzinfo=UTC)
FEB_1 = _ms(datetime(2025, 2, 1, tzinfo=UTC))
MAR_1 = _ms(datetime(2025, 3, 1, tzinfo=UTC))
B2_DAYS = 91  # 2025-01-01 .. 2025-04-01, the last closed daily bar


def _candles(lo: int, hi: int, step: int) -> list[tuple[Any, ...]]:
    return [tuple(_kline_row(t, step)) for t in range(lo, hi, step)]


def _run_research(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    *,
    feb_vision_404: bool = False,
    rest_serves_feb: bool = True,
) -> tuple[dict[str, Any], list[int], int, list[tuple[str, int]]]:
    """ingest_research_symbol with Vision months and REST served from a clean grid."""
    rest_calls: list[tuple[str, int]] = []

    def monthly(_symbol: str, interval: str, year: int, month: int, **_kw: Any) -> Any:
        if feb_vision_404 and month == 2:
            return None
        lo = _ms(datetime(year, month, 1, tzinfo=UTC))
        return _candles(lo, _ms(datetime(year, month + 1, 1, tzinfo=UTC)), STEP_MS[interval])

    def rest(_symbol: str, interval: str, lo: int, hi: int) -> list[tuple[Any, ...]]:
        rest_calls.append((interval, lo))
        rows = _candles(lo, hi, STEP_MS[interval])
        if not rest_serves_feb:
            rows = [r for r in rows if not FEB_1 <= r[0] < MAR_1]
        return rows

    monkeypatch.setattr(ingest, "download_monthly_klines", monthly)
    monkeypatch.setattr(ingest, "download_recent_klines", rest)
    monkeypatch.setattr(ingest, "download_funding_rates", lambda *_a, **_k: [])
    monkeypatch.setattr(ingest, "ingest_funding_intervals", lambda *_a, **_k: 0)
    arc = RawDataArchive(path)
    try:
        result = ingest.ingest_research_symbol(
            "XRPUSDT", _ms(B2_LISTING), arc, now=B2_NOW, verbose=False
        )
        days = [int(r["open_time"]) for r in arc.get_klines("XRPUSDT", "1d")]
        hours = len(arc.get_klines("XRPUSDT", "1h"))
    finally:
        arc.close()
    return result, days, hours, rest_calls


@pytest.mark.parametrize("hole", ["vision_404", "past_coverage_threshold"])
def test_interior_hole_is_backfilled_from_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hole: str
) -> None:
    path = tmp_path / "raw.db"
    if hole == "past_coverage_threshold":
        # 27 of 28 February daily bars: the month clears the 0.9 coverage skip
        arc = RawDataArchive(path)
        arc.insert_klines("XRPUSDT", "1d", _candles(FEB_1 + DAY_MS, MAR_1, DAY_MS), "seed")
        arc.close()
    result, days, hours, rest_calls = _run_research(
        monkeypatch, path, feb_vision_404=hole == "vision_404"
    )
    assert result["failed"] == []
    assert days == [_ms(B2_LISTING) + i * DAY_MS for i in range(B2_DAYS)]
    assert hours == B2_DAYS * 24
    assert ("1d", FEB_1) in rest_calls

    # rerun: complete, nothing missing, so REST is asked for nothing
    result, days2, _hours, rest_calls = _run_research(monkeypatch, path)
    assert result["failed"] == [] and days2 == days and rest_calls == []


def test_hole_rest_cannot_serve_fails_resumably(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "raw.db"
    for _run in range(2):  # a rerun reports the same hole again, never silently
        result, days, _hours, _calls = _run_research(
            monkeypatch, path, feb_vision_404=True, rest_serves_feb=False
        )
        assert "1d missing 2025-02-01..2025-02-28" in result["failed"]
        assert "1h missing 2025-02-01T00:00..2025-02-28T23:00" in result["failed"]
        assert FEB_1 not in days


# ------------------------------------------------------------ verify (B3, B5)


def test_verify_fails_when_required_streams_are_absent(
    seeded: Path, fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    arc = RawDataArchive(seeded)
    t0 = _ms(datetime(2025, 1, 1, tzinfo=UTC))
    arc.insert_klines(
        "XRPUSDT", "1h", [_kline_row(t0 + i * HOUR_MS, HOUR_MS) for i in range(72)], "s"
    )
    arc.close()
    assert ingest.main(["verify", "--symbols", "XRPUSDT"]) == 1
    out = capsys.readouterr().out
    for stream in ("1d klines", "funding rates", "funding intervals"):
        assert f"MISSING {stream}" in out
    assert "ALL CHECKS PASSED" not in out


@pytest.mark.parametrize(
    ("defect", "kwargs", "marker"),
    [
        ("funding gap", {"funding_skip": {5, 6}}, "funding gap 2025-01-02T08:00"),
        ("funding late", {"funding_skip": {0, 1, 2, 3}}, "funding starts 2025-01-02T08:00"),
        ("no intervals", {"intervals": False}, "MISSING funding intervals"),
        ("1d short", {"daily_days": 7}, "1d ends 2025-01-08"),
        ("1h late", {"gap_hours": set(range(24))}, "1h starts 2025-01-02"),
    ],
)
def test_verify_flags_uncovered_price_and_funding_spans(
    seeded: Path,
    fake: FakeBinance,
    capsys: pytest.CaptureFixture[str],
    defect: str,
    kwargs: dict[str, Any],
    marker: str,
) -> None:
    seed: dict[str, Any] = {"frozen": set(), "gap_hours": set(), **kwargs}
    _seed_research(seeded, "XRPUSDT", 10, **seed)
    assert ingest.main(["verify", "--symbols", "XRPUSDT"]) == 1, defect
    assert marker in capsys.readouterr().out, defect


def test_verify_clean_research_coin_passes(
    seeded: Path, fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_research(seeded, "XRPUSDT", 10, frozen=set(), gap_hours=set())
    assert ingest.main(["verify", "--symbols", "XRPUSDT"]) == 0
    assert "ALL CHECKS PASSED" in capsys.readouterr().out


def test_no_data_days_are_the_actual_frozen_dates_across_a_gap(
    seeded: Path, fake: FakeBinance, capsys: pytest.CaptureFixture[str]
) -> None:
    from vibe_quant.data.verify import detect_flat_zero_volume_runs

    t0 = _ms(datetime(2025, 1, 1, tzinfo=UTC))
    arc = RawDataArchive(seeded)
    flat = [t0, 1.0, 1.0, 1.0, 1.0, 0.0, t0 + DAY_MS - 1, 0.0, 0, 0.0, 0.0]
    jan3 = [t0 + 2 * DAY_MS, *flat[1:6], t0 + 3 * DAY_MS - 1, *flat[7:]]
    arc.insert_klines("XRPUSDT", "1d", [flat, jan3], "seed")
    arc.close()
    ingest.main(["verify", "--symbols", "XRPUSDT"])
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "NO DATA" in ln)
    assert "2025-01-01" in line and "2025-01-03" in line and "2025-01-02" not in line
    # the shared scanner splits a flat run at every gap (1m path too)
    m = [
        {"open_time": t, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}
        for t in (0, 60_000, 180_000)
    ]
    assert detect_flat_zero_volume_runs(m) == [(0, 60_000, 2), (180_000, 180_000, 1)]


# ------------------------------------------------------------------ budget (B7)


def test_research_estimate_above_the_2gb_budget_refuses_before_writing(
    seeded: Path,
    fake: FakeBinance,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert ingest.RESEARCH_BUDGET_BYTES == 2 * GB
    est = _estimate_bytes()
    before = _snapshot(seeded)
    monkeypatch.setattr(ingest, "RESEARCH_BUDGET_BYTES", est - 1)
    assert ingest.main(["download-research", "--yes"]) == 1
    assert "budget" in capsys.readouterr().err.lower()
    assert _snapshot(seeded) == before
    assert _download_requests(fake) == []

    # equality proceeds (growth pinned at 0 so the run-time guard stays quiet)
    monkeypatch.setattr(ingest, "RESEARCH_BUDGET_BYTES", est)
    monkeypatch.setattr(ingest, "_archive_bytes", lambda _p: 0)
    assert ingest.main(["download-research", "--yes"]) == 0


def test_measured_growth_over_budget_stops_before_the_next_coin(
    seeded: Path,
    fake: FakeBinance,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(ingest, "RESEARCH_BUDGET_BYTES", _estimate_bytes())
    sizes = iter([0, 3 * GB, 3 * GB, 3 * GB])
    monkeypatch.setattr(ingest, "_archive_bytes", lambda _p: next(sizes))
    assert ingest.main(["download-research", "--yes"]) == 1
    err = capsys.readouterr().err
    assert "budget" in err.lower() and "NEWUSDT" in err
    assert fake.asked("XRPUSDT") != []
    assert fake.asked("NEWUSDT") == []
