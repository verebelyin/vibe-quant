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
