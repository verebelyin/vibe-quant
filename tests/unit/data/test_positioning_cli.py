"""CLI for positioning data: ``download-metrics``, ``download-dvol`` and ``status``.

Every archive is a tmp file; Binance Vision (S3 listing + zip host) and
Deribit are ``httpx.MockTransport`` fakes, installed by replacing
``httpx.Client`` so the commands run through ``ingest.main`` unchanged.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from vibe_quant.data import archive as archive_mod
from vibe_quant.data import catalog as catalog_mod
from vibe_quant.data import downloader, dvol, ingest, universe
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.downloader import SUPPORTED_SYMBOLS

if TYPE_CHECKING:
    from pathlib import Path

GB = 1024**3
S3_HOST = "s3-ap-northeast-1.amazonaws.com"
PREFIX = "data/futures/um/daily/metrics/"
HEADER = (
    "create_time,symbol,sum_open_interest,sum_open_interest_value,"
    "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
    "count_long_short_ratio,sum_taker_long_short_vol_ratio"
)
DAY1, DAY2 = date(2024, 1, 1), date(2024, 1, 2)
TWO_DAYS = ["--start", "2024-01-01", "--end", "2024-01-02"]
HOUR_MS = 3_600_000
JAN1_MS = 1704067200000  # 2024-01-01 00:00 UTC


def _zip(symbol: str, day: date, row_symbol: str | None = None) -> bytes:
    t0 = datetime(day.year, day.month, day.day, tzinfo=UTC)
    rows = [
        f"{(t0 + timedelta(minutes=5 * k)):%Y-%m-%d %H:%M:%S},{row_symbol or symbol},"
        f"{1000 + k},{(1000 + k) * 40000},1.1,{1 + k / 1000},1.3,0.7"
        for k in range(288)
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{symbol}-metrics-{day}.csv", "\n".join([HEADER, *rows]) + "\n")
    return buf.getvalue()


class FakeServers:
    """Vision metrics (any symbol) + Deribit DVOL; records every request."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, tuple[bytes, str]]] = {}
        self.listed: list[str] = []
        self.gets: list[str] = []
        self.listing_status = 200
        self.candles: dict[str, list[list[float]]] = {}
        self.dvol_error: set[str] = set()

    def put(self, symbol: str, day: date, content: bytes | None = None, etag: str = "") -> None:
        body = content if content is not None else _zip(symbol, day)
        md5 = etag or hashlib.md5(body, usedforsecurity=False).hexdigest()
        self.files.setdefault(symbol, {})[f"{symbol}-metrics-{day}.zip"] = (body, md5)

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == S3_HOST:
            return self._listing(request)
        if request.url.host == "www.deribit.com":
            return self._dvol(request)
        _, symbol, name = request.url.path.rsplit("/", 2)
        self.gets.append(name)
        return httpx.Response(200, content=self.files[symbol][name][0])

    def _listing(self, request: httpx.Request) -> httpx.Response:
        if self.listing_status != 200:
            return httpx.Response(self.listing_status)
        symbol = request.url.params["prefix"].removeprefix(PREFIX).rstrip("/")
        self.listed.append(symbol)
        contents = "".join(
            f"<Contents><Key>{PREFIX}{symbol}/{name}</Key>"
            f"<LastModified>2026-11-20T01:02:03.000Z</LastModified>"
            f"<ETag>&quot;{md5}&quot;</ETag><Size>{len(body)}</Size></Contents>"
            for name, (body, md5) in sorted(self.files.get(symbol, {}).items())
        )
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<IsTruncated>false</IsTruncated>{contents}</ListBucketResult>"
        )
        return httpx.Response(200, content=xml.encode())

    def _dvol(self, request: httpx.Request) -> httpx.Response:
        currency = request.url.params["currency"]
        if currency in self.dvol_error:
            return httpx.Response(
                400, json={"error": {"code": 10001, "message": "boom"}, "usOut": 0}
            )
        lo, hi = (
            int(request.url.params["start_timestamp"]),
            int(request.url.params["end_timestamp"]),
        )
        data = [c for c in self.candles.get(currency, []) if lo <= c[0] <= hi]
        now_us = int(datetime.now(UTC).timestamp() * 1_000_000)
        return httpx.Response(
            200, json={"result": {"data": data, "continuation": None}, "usOut": now_us}
        )


@pytest.fixture
def servers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeServers:
    fake = FakeServers()
    real_client = httpx.Client

    def client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(fake.handler), **kwargs)

    def no_prompt(_prompt: str = "") -> str:
        raise AssertionError("prompted")

    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.setattr(downloader, "MONTHLY_RETRY_DELAYS", (0.0, 0.0, 0.0))
    monkeypatch.setattr(dvol.time, "sleep", lambda _s: None)
    monkeypatch.setattr(archive_mod, "DEFAULT_ARCHIVE_PATH", tmp_path / "raw_data.db")
    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", tmp_path / "catalog")
    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda _p: (100 * GB, 0, 50 * GB))
    monkeypatch.setattr("builtins.input", no_prompt)
    uni = tmp_path / "universe_top20.txt"
    uni.write_text("# as-of: 2026-11-21\nBTCUSDT\nXRPUSDT\n")
    monkeypatch.setattr(universe, "UNIVERSE_FILE", uni)
    return fake


def _metrics_count(path: Path) -> int:
    arc = RawDataArchive(path)
    try:
        return int(arc.conn.execute("SELECT COUNT(*) FROM raw_metrics").fetchone()[0])
    finally:
        arc.close()


def test_disk_refusal(
    servers: FakeServers,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "raw_data.db"
    servers.put("BTCUSDT", DAY1)
    assert ingest.main(["download-metrics", "--symbols", "BTCUSDT", *TWO_DAYS, "--yes"]) == 0
    servers.put("BTCUSDT", DAY2)
    capsys.readouterr()
    mtime, gets = path.stat().st_mtime_ns, list(servers.gets)
    estimate = 288 * 80  # DAY1 is archived: only DAY2 counts

    monkeypatch.setattr(
        ingest.shutil, "disk_usage", lambda _p: (100 * GB, 0, estimate + 5 * GB - 1)
    )
    assert ingest.main(["download-metrics", "--symbols", "BTCUSDT", *TWO_DAYS, "--yes"]) == 1
    out, err = capsys.readouterr()
    assert "BTCUSDT          1 file(s), 288 rows, 22.5 KB" in out
    assert "Estimate: 22.5 KB for 1 file(s)" in out
    assert "Error: refusing to download: free disk is below estimate + 5 GB headroom" in err
    assert path.stat().st_mtime_ns == mtime
    assert servers.gets == gets
    assert _metrics_count(path) == 288

    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda _p: (100 * GB, 0, estimate + 5 * GB))
    assert ingest.main(["download-metrics", "--symbols", "BTCUSDT", *TWO_DAYS, "--yes"]) == 0
    assert servers.gets == [*gets, "BTCUSDT-metrics-2024-01-02.zip"]
    assert _metrics_count(path) == 576


def test_confirm_and_dry_run(
    servers: FakeServers,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "raw_data.db"
    for day in (DAY1, DAY2):
        servers.put("BTCUSDT", day)
    servers.put("XRPUSDT", DAY1)

    assert ingest.main(["download-metrics", "--dry-run", *TWO_DAYS]) == 0
    out = capsys.readouterr().out
    assert servers.listed == [*SUPPORTED_SYMBOLS, "XRPUSDT"]  # universe adds XRP, BTC once
    assert "BTCUSDT          2 file(s), 576 rows, 45.0 KB" in out
    assert "XRPUSDT          1 file(s), 288 rows, 22.5 KB" in out
    assert "Estimate: 67.5 KB for 3 file(s)" in out
    assert "Dry run: nothing written." in out
    assert (servers.gets, path.exists()) == ([], False)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    assert ingest.main(["download-metrics", "--symbols", "BTCUSDT", *TWO_DAYS]) == 0
    assert "aborted: nothing written" in capsys.readouterr().out
    assert (servers.gets, path.exists()) == ([], False)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    assert ingest.main(["download-metrics", "--symbols", "XRPUSDT", *TWO_DAYS]) == 0
    assert servers.gets == ["XRPUSDT-metrics-2024-01-01.zip"]

    def no_prompt(_prompt: str = "") -> str:
        raise AssertionError("prompted")

    monkeypatch.setattr("builtins.input", no_prompt)
    assert ingest.main(["download-metrics", "--symbols", "BTCUSDT", *TWO_DAYS, "--yes"]) == 0
    out = capsys.readouterr().out
    assert "BTCUSDT: 2 file(s) ok, 576 rows inserted" in out
    assert _metrics_count(path) == 864


def test_status_shows_metrics_range(
    servers: FakeServers, capsys: pytest.CaptureFixture[str]
) -> None:
    for day in (DAY1, DAY2):
        servers.put("BTCUSDT", day)
    servers.put("ETHUSDT", date(2023, 6, 5))
    assert ingest.main(["download-metrics", "--symbols", "BTCUSDT", *TWO_DAYS, "--yes"]) == 0
    assert ingest.main(["download-metrics", "--symbols", "ETHUSDT", "--yes"]) == 0
    capsys.readouterr()

    assert ingest.main(["status"]) == 0
    out = capsys.readouterr().out
    btc = out.index("BTCUSDT:")
    eth = out.index("ETHUSDT:")
    assert (
        out.index("Metrics: 576 rows 2024-01-01T00:00:00+00:00 to 2024-01-02T23:55:00+00:00", btc)
        < eth
    )
    assert "Metrics: 288 rows 2023-06-05T00:00:00+00:00 to 2023-06-05T23:55:00+00:00" in out[eth:]
    assert "DVOL" not in out


def test_download_dvol_and_status(
    servers: FakeServers, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    servers.candles["BTC"] = [[JAN1_MS + h * HOUR_MS, 50, 51, 49, 50 + h] for h in range(3)]
    servers.candles["ETH"] = [[JAN1_MS + h * HOUR_MS, 70, 71, 69, 70 + h] for h in range(2)]

    assert ingest.main(["download-dvol", "--start", "2024-01-01", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "DVOL BTC: 3 new candle(s)" in out and "DVOL ETH: 2 new candle(s)" in out
    arc = RawDataArchive(tmp_path / "raw_data.db")
    stored = list(arc.conn.execute("SELECT currency, open_time, close FROM raw_dvol ORDER BY 1, 2"))
    arc.close()
    assert [tuple(r) for r in stored] == [
        ("BTC", JAN1_MS, 50.0),
        ("BTC", JAN1_MS + HOUR_MS, 51.0),
        ("BTC", JAN1_MS + 2 * HOUR_MS, 52.0),
        ("ETH", JAN1_MS, 70.0),
        ("ETH", JAN1_MS + HOUR_MS, 71.0),
    ]

    assert ingest.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "DVOL BTC: 3 candles 2024-01-01T00:00:00+00:00 to 2024-01-01T02:00:00+00:00" in out
    assert "DVOL ETH: 2 candles 2024-01-01T00:00:00+00:00 to 2024-01-01T01:00:00+00:00" in out

    servers.dvol_error.add("ETH")
    servers.candles["BTC"].append([JAN1_MS + 3 * HOUR_MS, 50, 51, 49, 53])
    assert ingest.main(["download-dvol", "--currencies", "ETH,BTC", "--yes"]) == 1
    out, err = capsys.readouterr()
    assert "Deribit DVOL ETH failed" in err and "re-run download-dvol to resume" in err
    assert "DVOL BTC: 1 new candle(s)" in out  # one currency failing does not stop the next


def test_download_metrics_outcomes_and_exit_codes(
    servers: FakeServers, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "raw_data.db"
    servers.listing_status = 403
    assert ingest.main(["download-metrics", "--symbols", "BTCUSDT", *TWO_DAYS, "--yes"]) == 1
    assert "Vision metrics listing BTCUSDT" in capsys.readouterr().err
    assert not path.exists()

    servers.listing_status = 200
    servers.put("BTCUSDT", DAY1, _zip("BTCUSDT", DAY1, row_symbol="ETHUSDT"))
    assert ingest.main(["download-metrics", "--symbols", "BTCUSDT", *TWO_DAYS, "--yes"]) == 0
    out = capsys.readouterr().out
    assert "MALFORMED BTCUSDT 2024-01-01 (symbol ETHUSDT != BTCUSDT)" in out
    assert "run verify" in out

    servers.put("BTCUSDT", DAY2, etag="0" * 32)  # md5 never matches: the day fails
    assert ingest.main(["download-metrics", "--symbols", "BTCUSDT", *TWO_DAYS, "--yes"]) == 1
    out, err = capsys.readouterr()
    assert "MALFORMED" not in out  # recorded by the earlier run, not this one
    assert "BTCUSDT: 1 day(s) failed" in err and "re-run download-metrics" in err


# ----------------------------------------------------------- verify + ack-metrics

MIN_MS = 60_000
DAY_MS = 86_400_000
DAY3 = date(2024, 1, 3)


def _variant(
    symbol: str,
    day: date,
    *,
    changed: int = 0,
    drop: frozenset[int] = frozenset(),
    shift: frozenset[int] = frozenset(),
    blank: frozenset[int] = frozenset(),
    comment: bytes = b"",
) -> bytes:
    """``_zip`` with the first *changed* rows' OI bumped, rows dropped/shifted 30 s/blanked."""
    t0 = datetime(day.year, day.month, day.day, tzinfo=UTC)
    rows = []
    for k in range(288):
        if k in drop:
            continue
        stamp = t0 + timedelta(minutes=5 * k, seconds=30 if k in shift else 0)
        oi = "" if k in blank else str(1000 + k + (7 if k < changed else 0))
        rows.append(
            f"{stamp:%Y-%m-%d %H:%M:%S},{symbol},{oi},{(1000 + k) * 40000},1.1,"
            f"{1 + k / 1000},1.3,0.7"
        )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{symbol}-metrics-{day}.csv", "\n".join([HEADER, *rows]) + "\n")
        zf.comment = comment
    return buf.getvalue()


def _seed_full_tier(path: Path, symbol: str) -> None:
    """One clean UTC day of 1m klines (2024-01-01): the kline checks pass."""
    arc = RawDataArchive(path)
    arc.insert_klines(
        symbol,
        "1m",
        [
            [t, 100.0, 101.0, 99.0, 100.5, 5.0, t + MIN_MS - 1, 500.0, 7, 2.0, 200.0]
            for t in range(JAN1_MS, JAN1_MS + DAY_MS, MIN_MS)
        ],
        "seed",
    )
    arc.close()


def _seed_research_coin(path: Path, symbol: str) -> None:
    """2024-01-01..02 1h + 1d klines, 8h funding, January intervals: research checks pass."""
    arc = RawDataArchive(path)
    for interval, step, n in (("1h", HOUR_MS, 48), ("1d", DAY_MS, 2)):
        rows = [
            [t, 100.0, 101.0, 99.0, 100.5, 5.0, t + step - 1, 500.0, 7, 2.0, 200.0]
            for t in range(JAN1_MS, JAN1_MS + n * step, step)
        ]
        arc.insert_klines(symbol, interval, rows, "seed")
    arc.insert_funding_rates(
        symbol, [(JAN1_MS + i * 8 * HOUR_MS, 0.0001, 1.0) for i in range(6)], "seed"
    )
    arc.insert_funding_intervals(symbol, [(JAN1_MS, 8)], "seed")
    arc.close()


def _verify(symbols: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    capsys.readouterr()
    rc = ingest.main(["verify", "--symbols", symbols])
    return rc, capsys.readouterr().out


def _section(out: str, symbol: str) -> str:
    """The verify lines of *symbol*: from its header to the next blank line."""
    start = out.index(f"\n{symbol}: ")
    end = out.find("\n\n", start + 1)
    return out[start : end if end != -1 else len(out)]


def _acks(path: Path) -> list[tuple[Any, ...]]:
    arc = RawDataArchive(path)
    try:
        return [
            tuple(r)
            for r in arc.conn.execute("SELECT file_id, md5, detail, reason FROM raw_metrics_acks")
        ]
    finally:
        arc.close()


def _download(*args: str) -> int:
    return ingest.main(["download-metrics", "--symbols", "BTCUSDT", "--yes", *args])


def test_verify_drift_until_acknowledged(
    servers: FakeServers, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "raw_data.db"
    servers.put("BTCUSDT", DAY1)
    servers.put("BTCUSDT", DAY2, _zip("BTCUSDT", DAY2, row_symbol="ETHUSDT"))
    assert _download(*TWO_DAYS) == 0
    _seed_full_tier(path, "BTCUSDT")

    rc, out = _verify("BTCUSDT", capsys)
    assert rc == 1 and "ISSUES FOUND" in out
    assert "  METRICS MALFORMED BTCUSDT 2024-01-02 (symbol ETHUSDT != BTCUSDT)" in out

    # Re-served valid (new md5): the newest file of the day is ok, the malformed one superseded
    servers.put("BTCUSDT", DAY2)
    assert _download(*TWO_DAYS) == 0
    rc, out = _verify("BTCUSDT", capsys)
    assert rc == 0 and "ALL CHECKS PASSED" in out
    assert "MALFORMED" not in out
    assert "  Metrics: 576 rows 2024-01-01 to 2024-01-02, 3 file(s)" in _section(out, "BTCUSDT")

    # DAY1 regenerated with 3 changed rows: drift fails verify until acknowledged
    servers.put("BTCUSDT", DAY1, _variant("BTCUSDT", DAY1, changed=3))
    assert _download(*TWO_DAYS, "--recheck") == 0
    rc, out = _verify("BTCUSDT", capsys)
    assert rc == 1 and "ISSUES FOUND" in out
    assert "  METRICS DRIFT BTCUSDT 2024-01-01 (3 rows differ)" in out

    capsys.readouterr()
    assert ingest.main(["ack-metrics", "BTCUSDT", "2024-01-01", "--reason", "regen wave"]) == 0
    rc, out = _verify("BTCUSDT", capsys)
    assert rc == 0 and "ALL CHECKS PASSED" in out
    assert "  metrics drift acknowledged: 2024-01-01 (3 rows differ; regen wave)" in out
    assert "METRICS DRIFT" not in out

    # DAY2 regenerated with identical values (only the zip bytes differ): info, exit 0
    servers.put("BTCUSDT", DAY2, _variant("BTCUSDT", DAY2, comment=b"regenerated"))
    assert _download(*TWO_DAYS, "--recheck") == 0
    rc, out = _verify("BTCUSDT", capsys)
    assert rc == 0 and "ALL CHECKS PASSED" in out
    assert "  metrics: 1 regenerated file(s), values identical" in out

    # A newer drift of the acknowledged day is a new issue
    servers.put("BTCUSDT", DAY1, _variant("BTCUSDT", DAY1, changed=5))
    assert _download(*TWO_DAYS, "--recheck") == 0
    rc, out = _verify("BTCUSDT", capsys)
    assert rc == 1
    assert "  METRICS DRIFT BTCUSDT 2024-01-01 (5 rows differ)" in out
    assert "acknowledged" not in out


def test_verify_gaps_informational(
    servers: FakeServers, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "raw_data.db"
    servers.put("BTCUSDT", DAY1, _variant("BTCUSDT", DAY1, blank=frozenset({4})))
    servers.put(
        "BTCUSDT",
        DAY3,
        _variant("BTCUSDT", DAY3, drop=frozenset({10, 11}), shift=frozenset({1})),
    )
    assert _download("--start", "2024-01-01", "--end", "2024-01-03") == 0
    _seed_full_tier(path, "BTCUSDT")
    arc = RawDataArchive(path)
    arc.insert_dvol(
        "BTC",
        3600,
        [(JAN1_MS + h * HOUR_MS, 50.0, 51.0, 49.0, 50.0) for h in (0, 1, 4)],
        "deribit",
    )
    arc.close()

    rc, out = _verify("BTCUSDT", capsys)
    assert rc == 0 and "ALL CHECKS PASSED" in out
    btc = _section(out, "BTCUSDT")
    assert "  Metrics: 574 rows 2024-01-01 to 2024-01-03, 2 file(s)" in btc
    assert "  metrics gaps: 1 missing day (2024-01-02)" in btc
    assert "  metrics missing rows: 2024-01-03 3" in btc  # 2 dropped + 1 off the grid
    assert "  metrics misaligned: 1 row(s) off the 5-minute grid" in btc
    assert "  metrics null values: sum_open_interest 1" in btc
    assert "DVOL BTC: 3 candles 2024-01-01T00:00:00+00:00 to 2024-01-01T04:00:00+00:00" in out
    assert "  dvol gaps: 2 missing candle(s) (2024-01-01T02:00:00+00:00 2)" in out
    assert "DVOL ETH: none archived (download-dvol)" in out


def test_verify_none_archived_is_info(
    servers: FakeServers, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "raw_data.db"
    _seed_full_tier(path, "BTCUSDT")
    _seed_research_coin(path, "XRPUSDT")

    rc, out = _verify("BTCUSDT,XRPUSDT", capsys)
    assert rc == 0 and "ALL CHECKS PASSED" in out
    assert "research tier" in _section(out, "XRPUSDT")
    for symbol in ("BTCUSDT", "XRPUSDT"):
        assert "  Metrics: none archived (download-metrics)" in _section(out, symbol)
    assert out.count("Metrics: none archived") == 2
    assert out.count("DVOL BTC: none archived (download-dvol)") == 1
    assert out.count("DVOL ETH: none archived (download-dvol)") == 1


def test_ack_metrics(
    servers: FakeServers, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "raw_data.db"

    def ack(*args: str) -> tuple[int, str, str]:
        capsys.readouterr()
        rc = ingest.main(["ack-metrics", *args])
        out, err = capsys.readouterr()
        return rc, out, err

    rc, _out, err = ack("BTCUSDT", "2024-01-01", "--reason", "bad wave")
    assert rc == 1
    assert "Error: no unacknowledged drift or malformed metrics file for BTCUSDT 2024-01-01" in err
    assert not path.exists()

    servers.put("BTCUSDT", DAY1)
    malformed = _zip("BTCUSDT", DAY2, row_symbol="ETHUSDT")
    servers.put("BTCUSDT", DAY2, malformed)
    assert _download(*TWO_DAYS) == 0
    md5 = hashlib.md5(malformed, usedforsecurity=False).hexdigest()

    rc, _out, err = ack("BTCUSDT", "2024-01-01", "--reason", "bad wave")  # day 1 is ok
    assert rc == 1 and "no unacknowledged drift or malformed metrics file" in err
    rc, _out, err = ack("BTCUSDT", "2024-01-02", "--reason", "  ")
    assert rc == 1 and "Error: --reason must not be blank" in err
    servers.put("BTCUSDT", DAY2, _variant("BTCUSDT", DAY2, comment=b"again"))
    rc, _out, err = ack("BTCUSDT", "2024-01-02", "--reason", "bad wave")
    assert rc == 1
    assert (
        "Error: BTCUSDT 2024-01-02 changed again since it was recorded; "
        "run download-metrics --recheck first"
    ) in err
    servers.put("BTCUSDT", DAY2, malformed)
    servers.listing_status = 503
    rc, _out, err = ack("BTCUSDT", "2024-01-02", "--reason", "bad wave")
    assert rc == 1 and "Error: Vision metrics listing BTCUSDT" in err
    assert _acks(path) == []

    servers.listing_status = 200
    file_id = next(
        int(r["id"])
        for r in RawDataArchive(path).get_metrics_files("BTCUSDT")
        if r["status"] == "malformed"
    )
    rc, out, _err = ack("btcusdt", "2024-01-02", "--reason", "bad wave")
    assert rc == 0
    assert (
        f"ACKNOWLEDGED METRICS MALFORMED BTCUSDT 2024-01-02 file #{file_id} "
        "(symbol ETHUSDT != BTCUSDT) (bad wave)"
    ) in out
    assert _acks(path) == [(file_id, md5, "symbol ETHUSDT != BTCUSDT", "bad wave")]

    listed = list(servers.listed)
    rc, out, _err = ack("BTCUSDT", "2024-01-02", "--reason", "other")
    assert rc == 0
    assert f"already acknowledged: METRICS MALFORMED BTCUSDT 2024-01-02 file #{file_id}" in out
    assert _acks(path) == [(file_id, md5, "symbol ETHUSDT != BTCUSDT", "bad wave")]
    assert servers.listed == listed

    _seed_full_tier(path, "BTCUSDT")
    rc, out = _verify("BTCUSDT", capsys)
    assert rc == 0
    assert (
        "  metrics malformed acknowledged: 2024-01-02 (symbol ETHUSDT != BTCUSDT; bad wave)"
    ) in out


def _file_ids(path: Path, day: date) -> dict[str, list[int]]:
    """files row ids of BTCUSDT *day* by status, oldest first."""
    arc = RawDataArchive(path)
    try:
        ids: dict[str, list[int]] = {}
        for r in arc.get_metrics_files("BTCUSDT"):
            if r["day"] == day.isoformat():
                ids.setdefault(r["status"], []).append(int(r["id"]))
        return ids
    finally:
        arc.close()


def test_ack_metrics_targets_newest_file_of_day(
    servers: FakeServers, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "raw_data.db"
    servers.put("BTCUSDT", DAY1)
    servers.put("BTCUSDT", DAY2)
    assert _download(*TWO_DAYS) == 0
    servers.put("BTCUSDT", DAY1, _variant("BTCUSDT", DAY1, changed=3))
    assert _download(*TWO_DAYS, "--recheck") == 0
    newest_body = _variant("BTCUSDT", DAY1, changed=5)
    servers.put("BTCUSDT", DAY1, newest_body)
    assert _download(*TWO_DAYS, "--recheck") == 0
    _older, newest = _file_ids(path, DAY1)["drift"]
    newest_md5 = hashlib.md5(newest_body, usedforsecurity=False).hexdigest()
    _seed_full_tier(path, "BTCUSDT")

    capsys.readouterr()
    assert ingest.main(["ack-metrics", "BTCUSDT", "2024-01-01", "--reason", "wave 2"]) == 0
    out = capsys.readouterr().out
    assert f"ACKNOWLEDGED METRICS DRIFT BTCUSDT 2024-01-01 file #{newest} (5 rows differ)" in out
    assert _acks(path) == [(newest, newest_md5, "5 rows differ", "wave 2")]
    rc, out = _verify("BTCUSDT", capsys)
    assert rc == 0 and "ALL CHECKS PASSED" in out
    assert "  metrics drift acknowledged: 2024-01-01 (5 rows differ; wave 2)" in out

    listed = list(servers.listed)
    assert ingest.main(["ack-metrics", "BTCUSDT", "2024-01-01", "--reason", "again"]) == 0
    out = capsys.readouterr().out
    assert f"already acknowledged: METRICS DRIFT BTCUSDT 2024-01-01 file #{newest}" in out
    assert _acks(path) == [(newest, newest_md5, "5 rows differ", "wave 2")]
    assert servers.listed == listed


def test_ack_metrics_original_reserved_after_drift(
    servers: FakeServers, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "raw_data.db"
    original = _zip("BTCUSDT", DAY1)
    same = _variant("BTCUSDT", DAY2, comment=b"regenerated")
    servers.put("BTCUSDT", DAY1, original)
    servers.put("BTCUSDT", DAY2)
    assert _download(*TWO_DAYS) == 0
    servers.put("BTCUSDT", DAY2, same)
    assert _download(*TWO_DAYS, "--recheck") == 0  # DAY2: same row
    first_drift = _variant("BTCUSDT", DAY1, changed=3)
    servers.put("BTCUSDT", DAY1, first_drift)
    servers.put("BTCUSDT", DAY2, _variant("BTCUSDT", DAY2, changed=2))
    assert _download(*TWO_DAYS, "--recheck") == 0
    servers.put("BTCUSDT", DAY1, _variant("BTCUSDT", DAY1, changed=4))
    assert _download(*TWO_DAYS, "--recheck") == 0  # DAY1: drift, drift (newest)
    _seed_full_tier(path, "BTCUSDT")

    def ack(day: str) -> tuple[int, str, str]:
        capsys.readouterr()
        rc = ingest.main(["ack-metrics", "BTCUSDT", day, "--reason", "original back"])
        out, err = capsys.readouterr()
        return rc, out, err

    # Vision re-serves an earlier DRIFT file of DAY1: not an ok/same file, refused
    servers.put("BTCUSDT", DAY1, first_drift)
    rc, _out, err = ack("2024-01-01")
    assert rc == 1 and "BTCUSDT 2024-01-01 changed again since it was recorded" in err
    assert _acks(path) == []

    # Vision re-serves the original files (DAY1 the ok one, DAY2 the same one)
    servers.put("BTCUSDT", DAY1, original)
    servers.put("BTCUSDT", DAY2, same)
    assert _download(*TWO_DAYS, "--recheck") == 0  # md5 already archived: nothing new
    rc, out = _verify("BTCUSDT", capsys)
    assert rc == 1
    assert "  METRICS DRIFT BTCUSDT 2024-01-01 (4 rows differ)" in out
    assert "  METRICS DRIFT BTCUSDT 2024-01-02 (2 rows differ)" in out

    day1, day2 = _file_ids(path, DAY1), _file_ids(path, DAY2)
    rc, out, _err = ack("2024-01-01")
    assert rc == 0
    assert f"ACKNOWLEDGED METRICS DRIFT BTCUSDT 2024-01-01 file #{day1['drift'][-1]}" in out
    rc, out, _err = ack("2024-01-02")
    assert rc == 0
    assert f"ACKNOWLEDGED METRICS DRIFT BTCUSDT 2024-01-02 file #{day2['drift'][-1]}" in out
    same_md5 = hashlib.md5(same, usedforsecurity=False).hexdigest()
    original_md5 = hashlib.md5(original, usedforsecurity=False).hexdigest()
    assert _acks(path) == [
        (day1["drift"][-1], original_md5, "4 rows differ", "original back"),
        (day2["drift"][-1], same_md5, "2 rows differ", "original back"),
    ]
    rc, out = _verify("BTCUSDT", capsys)
    assert rc == 0 and "ALL CHECKS PASSED" in out
    assert "METRICS DRIFT" not in out
    assert "  metrics drift acknowledged: 2024-01-01 (4 rows differ; original back)" in out
