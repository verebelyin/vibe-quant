"""Binance Vision daily ``metrics`` download: parse, archive, rerun and recheck rules.

All HTTP is stubbed with ``httpx.MockTransport`` (an S3 listing that pages two
keys at a time + the zip host); every archive is a tmp file.
"""

from __future__ import annotations

import hashlib
import io
import random
import threading
import zipfile
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import httpx
import pytest

from vibe_quant.data import downloader
from vibe_quant.data.archive import MetricsFileMeta, RawDataArchive
from vibe_quant.data.downloader import DownloadError
from vibe_quant.data.metrics import ingest_metrics, parse_metrics_zip

if TYPE_CHECKING:
    from pathlib import Path

HEADER = (
    "create_time,symbol,sum_open_interest,sum_open_interest_value,"
    "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
    "count_long_short_ratio,sum_taker_long_short_vol_ratio"
)
S3_HOST = "s3-ap-northeast-1.amazonaws.com"
DAY1 = date(2024, 1, 1)
DAY1_MS = 1704067200000  # 2024-01-01 00:00 UTC


def _rows(day: date, symbol: str = "BTCUSDT", n: int = 288) -> list[list[str]]:
    """``n`` rows on the 5-minute grid of ``day``; open interest 1000 + k."""
    t0 = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return [
        [
            (t0 + timedelta(minutes=5 * k)).strftime("%Y-%m-%d %H:%M:%S"),
            symbol,
            f"{1000 + k:.8f}",
            f"{(1000 + k) * 40000:.8f}",
            "1.17547937",
            f"{1 + k / 1000:.8f}",
            "1.35731217",
            "0.78373373",
        ]
        for k in range(n)
    ]


def _name(day: date, symbol: str = "BTCUSDT") -> str:
    return f"{symbol}-metrics-{day.isoformat()}.zip"


def _zip(filename: str, rows: list[list[str]], header: str = HEADER) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        body = "\n".join([header, *(",".join(r) for r in rows)]) + "\n"
        zf.writestr(filename.removesuffix(".zip") + ".csv", body)
    return buf.getvalue()


def _md5(content: bytes) -> str:
    return hashlib.md5(content, usedforsecurity=False).hexdigest()


class FakeVision:
    """S3 listing (two keys per page, ``.CHECKSUM`` keys interleaved) + zip host."""

    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self.symbol = symbol
        self.files: dict[str, tuple[bytes, str]] = {}
        self.gets: list[str] = []
        self.checksums: dict[str, bytes] = {}
        self.listing_status = 200

    def put(self, filename: str, content: bytes, etag: str | None = None) -> None:
        self.files[filename] = (content, etag or _md5(content))

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == S3_HOST:
            return self._listing(request)
        name = request.url.path.rsplit("/", 1)[-1]
        assert request.url.path == f"/data/futures/um/daily/metrics/{self.symbol}/{name}"
        self.gets.append(name)
        if name.endswith(".CHECKSUM"):
            return httpx.Response(200, content=self.checksums[name])
        return httpx.Response(200, content=self.files[name][0])

    def _listing(self, request: httpx.Request) -> httpx.Response:
        if self.listing_status != 200:
            return httpx.Response(self.listing_status)
        prefix = request.url.params["prefix"]
        assert prefix == f"data/futures/um/daily/metrics/{self.symbol}/"
        marker = request.url.params.get("marker", "")
        entries: list[tuple[str, str]] = []
        for name, (_, etag) in self.files.items():
            entries += [(prefix + name, etag), (prefix + name + ".CHECKSUM", "c" * 32)]
        after = sorted(e for e in entries if e[0] > marker)
        page, truncated = after[:2], len(after) > 2
        contents = "".join(
            f"<Contents><Key>{key}</Key><LastModified>2026-11-20T01:02:03.000Z</LastModified>"
            f"<ETag>&quot;{etag}&quot;</ETag><Size>12345</Size></Contents>"
            for key, etag in page
        )
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<Name>data.binance.vision</Name><Prefix>{prefix}</Prefix>"
            f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>{contents}"
            "</ListBucketResult>"
        )
        return httpx.Response(200, content=xml.encode())


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloader, "MONTHLY_RETRY_DELAYS", (0.0, 0.0, 0.0))


def _files(archive: RawDataArchive) -> list[tuple[str, str, int, int, str | None]]:
    return [
        (r["day"], r["status"], r["csv_rows"], r["rows"], r["detail"])
        for r in archive.get_metrics_files("BTCUSDT")
    ]


def _metrics_rows(archive: RawDataArchive) -> list[tuple[object, ...]]:
    return [
        tuple(r)
        for r in archive.conn.execute("SELECT * FROM raw_metrics ORDER BY symbol, create_time")
    ]


def test_parse_unsorted_and_blank() -> None:
    rows = _rows(DAY1)
    rows[5][5] = ""  # blank sum_toptrader_long_short_ratio at 00:25
    random.Random(7).shuffle(rows)
    assert rows[0][0] != "2024-01-01 00:00:00"  # really unsorted

    parsed = parse_metrics_zip(_zip(_name(DAY1), rows), _name(DAY1), "BTCUSDT")

    assert (parsed.status, parsed.csv_rows, parsed.detail) == ("ok", 288, None)
    assert len(parsed.rows) == 288
    stamps = [r[0] for r in parsed.rows]
    assert stamps == [DAY1_MS + 300_000 * k for k in range(288)]
    assert parsed.rows[0] == (DAY1_MS, 1000.0, 40000000.0, 1.17547937, 1.0, 1.35731217, 0.78373373)
    assert parsed.rows[5] == (
        DAY1_MS + 1_500_000,
        1005.0,
        40200000.0,
        1.17547937,
        None,
        1.35731217,
        0.78373373,
    )
    assert parsed.rows[287][1:3] == (1287.0, 51480000.0)


def test_duplicates_and_malformed(tmp_path: Path) -> None:
    day = date(2020, 9, 1)
    base = _rows(day)
    differing = [*base, [*base[0][:2], "1.00000000", *base[0][3:]]]
    next_day = [*base, ["2020-09-02 00:00:00", *base[0][1:]]]
    wrong_symbol = [*base[:-1], [base[-1][0], "ETHUSDT", *base[-1][2:]]]
    cases = {
        "doubled": ([r for row in base for r in (row, list(row))], ("ok", 576, 288, None)),
        "differing": (
            differing,
            (
                "malformed",
                289,
                0,
                "duplicate create_time 2020-09-01 00:00:00 with different values",
            ),
        ),
        "next_day": (
            next_day,
            ("malformed", 289, 0, "create_time 2020-09-02 00:00:00 outside 2020-09-01"),
        ),
        "symbol": (wrong_symbol, ("malformed", 288, 0, "symbol ETHUSDT != BTCUSDT")),
    }
    for label, (rows, (status, csv_rows, stored, detail)) in cases.items():
        archive = RawDataArchive(tmp_path / f"{label}.db")
        vision = FakeVision()
        vision.put(_name(day), _zip(_name(day), rows))

        counts = ingest_metrics("BTCUSDT", archive, recheck=False, client=vision.client())

        assert _files(archive) == [("2020-09-01", status, csv_rows, stored, detail)], label
        assert archive.get_metrics_range("BTCUSDT") == (
            (288, 1598918400000, 1598918400000 + 287 * 300_000) if stored else None
        ), label
        assert counts["malformed"] == (status == "malformed"), label
        assert counts["rows_inserted"] == stored, label

    bad_header = parse_metrics_zip(
        _zip(_name(day), base, header=HEADER.replace("symbol", "pair")), _name(day), "BTCUSDT"
    )
    assert (bad_header.status, bad_header.rows) == ("malformed", [])
    assert bad_header.detail is not None and bad_header.detail.startswith("bad header")


def test_md5_mismatch_fails_day(tmp_path: Path) -> None:
    archive = RawDataArchive(tmp_path / "archive.db")
    vision = FakeVision()
    content = _zip(_name(DAY1), _rows(DAY1))
    vision.put(_name(DAY1), content, etag="0" * 32)

    counts = ingest_metrics("BTCUSDT", archive, recheck=False, client=vision.client())

    assert counts["failed"] == 1
    assert counts["files_ok"] == 0
    assert vision.gets == [_name(DAY1)] * 3
    assert archive.get_metrics_files("BTCUSDT") == []
    assert archive.get_metrics_range("BTCUSDT") is None

    vision.put(_name(DAY1), content)  # listing ETag now matches the bytes
    counts = ingest_metrics("BTCUSDT", archive, recheck=False, client=vision.client())

    assert (counts["failed"], counts["files_ok"], counts["rows_inserted"]) == (0, 1, 288)
    assert vision.gets == [_name(DAY1)] * 4
    assert _files(archive) == [("2024-01-01", "ok", 288, 288, None)]


def test_multipart_etag_falls_back_to_checksum_sha256(tmp_path: Path) -> None:
    archive = RawDataArchive(tmp_path / "archive.db")
    vision = FakeVision()
    day2 = DAY1 + timedelta(days=1)
    good = _zip(_name(DAY1), _rows(DAY1))
    bad = _zip(_name(day2), _rows(day2))
    vision.put(_name(DAY1), good, etag="a" * 32 + "-2")
    vision.put(_name(day2), bad, etag="b" * 32 + "-2")
    sha = hashlib.sha256(good).hexdigest()
    vision.checksums[_name(DAY1) + ".CHECKSUM"] = f"{sha}  {_name(DAY1)}\n".encode()
    vision.checksums[_name(day2) + ".CHECKSUM"] = f"{sha}  {_name(day2)}\n".encode()

    counts = ingest_metrics("BTCUSDT", archive, recheck=False, client=vision.client())

    assert (counts["files_ok"], counts["rows_inserted"], counts["failed"]) == (1, 288, 1)
    assert vision.gets.count(_name(day2)) == 3
    assert _files(archive) == [("2024-01-01", "ok", 288, 288, None)]


def test_rerun_skips_archived_days(tmp_path: Path) -> None:
    archive = RawDataArchive(tmp_path / "archive.db")
    vision = FakeVision()
    days = [DAY1 + timedelta(days=i) for i in range(3)]
    for day in days:
        vision.put(_name(day), _zip(_name(day), _rows(day)))

    counts = ingest_metrics("BTCUSDT", archive, recheck=False, client=vision.client())

    assert (counts["files_ok"], counts["rows_inserted"]) == (3, 864)
    assert archive.get_metrics_range("BTCUSDT") == (864, DAY1_MS, 1704326100000)

    # R24: archived ok days are never fetched again.
    vision.gets.clear()
    counts = ingest_metrics("BTCUSDT", archive, recheck=False, client=vision.client())
    assert vision.gets == []
    assert (counts["skipped"], counts["files_ok"]) == (3, 0)

    # R24b: a malformed day with the same ETag is not fetched again.
    day4 = DAY1 + timedelta(days=3)
    vision.put(_name(day4), _zip(_name(day4), _rows(day4), header="garbage"))
    counts = ingest_metrics("BTCUSDT", archive, recheck=False, client=vision.client())
    assert (vision.gets, counts["malformed"]) == ([_name(day4)], 1)
    vision.gets.clear()
    counts = ingest_metrics("BTCUSDT", archive, recheck=False, client=vision.client())
    assert (vision.gets, counts["skipped"], counts["malformed"]) == ([], 4, 0)
    assert [f[:2] for f in _files(archive) if f[0] == "2024-01-04"] == [("2024-01-04", "malformed")]

    # R24c: the listing ETag changes and the new file is valid -> fetched and stored.
    vision.put(_name(day4), _zip(_name(day4), _rows(day4)))
    counts = ingest_metrics("BTCUSDT", archive, recheck=False, client=vision.client())
    assert vision.gets == [_name(day4)]
    assert (counts["files_ok"], counts["rows_inserted"]) == (1, 288)
    assert [f[1] for f in _files(archive) if f[0] == "2024-01-04"] == ["malformed", "ok"]
    assert archive.get_metrics_range("BTCUSDT") == (1152, DAY1_MS, 1704412500000)


def test_recheck_records_drift_never_overwrites(tmp_path: Path) -> None:
    archive = RawDataArchive(tmp_path / "archive.db")
    vision = FakeVision()
    days = [DAY1 + timedelta(days=i) for i in range(4)]
    for day in days:
        vision.put(_name(day), _zip(_name(day), _rows(day)))
    ingest_metrics("BTCUSDT", archive, recheck=False, client=vision.client())
    before = _metrics_rows(archive)
    assert len(before) == 1152

    # Regeneration wave: day 1 re-served with equal values (rows reordered, so
    # new bytes / ETag); day 2 with 3 changed open-interest values; day 3 drops
    # its last stamp; day 4 gains an off-grid stamp. Day 5 was never archived:
    # a recheck does not download it.
    vision.put(_name(days[0]), _zip(_name(days[0]), _rows(days[0])[::-1]))
    changed = _rows(days[1])
    for k in (0, 100, 287):
        changed[k][2] = "1.00000000"
    vision.put(_name(days[1]), _zip(_name(days[1]), changed))
    vision.put(_name(days[2]), _zip(_name(days[2]), _rows(days[2])[:-1]))
    extra = _rows(days[3])
    extra.append(["2024-01-04 23:57:00", *extra[-1][1:]])
    vision.put(_name(days[3]), _zip(_name(days[3]), extra))
    day5 = DAY1 + timedelta(days=4)
    vision.put(_name(day5), _zip(_name(day5), _rows(day5)))
    vision.gets.clear()

    counts = ingest_metrics("BTCUSDT", archive, recheck=True, client=vision.client())

    assert sorted(vision.gets) == sorted(_name(d) for d in days)
    assert (counts["same"], counts["drift"], counts["files_ok"]) == (1, 3, 0)
    assert _files(archive) == [
        ("2024-01-01", "ok", 288, 288, None),
        ("2024-01-01", "same", 288, 0, None),
        ("2024-01-02", "ok", 288, 288, None),
        ("2024-01-02", "drift", 288, 0, "3 rows differ"),
        ("2024-01-03", "ok", 288, 288, None),
        ("2024-01-03", "drift", 287, 0, "1 rows differ"),
        ("2024-01-04", "ok", 288, 288, None),
        ("2024-01-04", "drift", 289, 0, "1 rows differ"),
    ]
    assert _metrics_rows(archive) == before

    # R25b: a second recheck of the same wave fetches nothing.
    vision.gets.clear()
    counts = ingest_metrics("BTCUSDT", archive, recheck=True, client=vision.client())
    assert vision.gets == []
    assert len(archive.get_metrics_files("BTCUSDT")) == 8
    assert _metrics_rows(archive) == before


def test_insert_same_md5_is_a_no_op(tmp_path: Path) -> None:
    archive = RawDataArchive(tmp_path / "archive.db")
    content = _zip(_name(DAY1), _rows(DAY1))
    parsed = parse_metrics_zip(content, _name(DAY1), "BTCUSDT")
    meta = MetricsFileMeta(
        filename=_name(DAY1),
        md5=_md5(content),
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        vision_last_modified="2026-11-20T01:02:03.000Z",
        csv_rows=parsed.csv_rows,
        status="ok",
        detail=None,
    )

    assert archive.insert_metrics_file("BTCUSDT", DAY1, meta, parsed.rows) == 288
    assert archive.insert_metrics_file("BTCUSDT", DAY1, meta, parsed.rows) == 0

    assert len(archive.get_metrics_files("BTCUSDT")) == 1
    assert archive.get_metrics_range("BTCUSDT") == (288, DAY1_MS, DAY1_MS + 287 * 300_000)


def test_listing_failure_writes_nothing(tmp_path: Path) -> None:
    vision = FakeVision()
    vision.put(_name(DAY1), _zip(_name(DAY1), _rows(DAY1)))
    vision.listing_status = 503
    path = tmp_path / "fresh.db"

    with pytest.raises(DownloadError, match="BTCUSDT"):
        ingest_metrics("BTCUSDT", RawDataArchive(path), recheck=False, client=vision.client())
    assert not path.exists()
    assert vision.gets == []

    # An existing archive is left byte-identical.
    existing = tmp_path / "existing.db"
    archive = RawDataArchive(existing)
    vision.listing_status = 200
    ingest_metrics("BTCUSDT", archive, recheck=False, client=vision.client())
    archive.close()
    digest = hashlib.sha256(existing.read_bytes()).hexdigest()
    day2 = DAY1 + timedelta(days=1)
    vision.put(_name(day2), _zip(_name(day2), _rows(day2)))
    vision.listing_status = 503

    with pytest.raises(DownloadError):
        ingest_metrics("BTCUSDT", RawDataArchive(existing), recheck=True, client=vision.client())
    assert hashlib.sha256(existing.read_bytes()).hexdigest() == digest
    assert vision.gets == [_name(DAY1)]


def test_days_are_fetched_in_parallel_and_stored_in_day_order(tmp_path: Path) -> None:
    """Zips download concurrently (a 4-party barrier only opens with 4 GETs in flight);
    the archive still records the days in listing order."""
    days = [DAY1 + timedelta(days=i) for i in range(8)]
    barrier = threading.Barrier(4, timeout=5)

    class ParallelVision(FakeVision):
        def _handle(self, request: httpx.Request) -> httpx.Response:
            if request.url.host != S3_HOST:
                barrier.wait()  # raises BrokenBarrierError when GETs run one at a time
            return super()._handle(request)

    archive = RawDataArchive(tmp_path / "archive.db")
    vision = ParallelVision()
    for day in days:
        vision.put(_name(day), _zip(_name(day), _rows(day)))

    counts = ingest_metrics("BTCUSDT", archive, recheck=False, client=vision.client())

    assert (counts["files_ok"], counts["rows_inserted"], counts["failed"]) == (8, 8 * 288, 0)
    stored = archive.get_metrics_files("BTCUSDT")
    assert [row["day"] for row in sorted(stored, key=lambda r: r["id"])] == [
        d.isoformat() for d in days
    ]
