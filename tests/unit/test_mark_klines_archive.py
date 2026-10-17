"""Mark-price 1m klines archive + Vision download (vibe-quant-2gyk1.1.1).

Liquidation triggers on the mark price, which Binance publishes as
``markPriceKlines`` monthly zips under data.binance.vision. They are archived
append-only in the new ``raw_mark_klines`` table (OHLC + timestamps only: mark
rows carry zero volume) so the ``raw_klines`` path that feeds the catalog is
untouched. 404 = month not published (unavailable, exit 0); a transient error
that survives the retry schedule is a hole (failed, exit 1).
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from vibe_quant.data import downloader, ingest
from vibe_quant.data.archive import RawDataArchive

if TYPE_CHECKING:
    from pathlib import Path

MARK_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)

# First row of each archive, verbatim: 2024-01 ships a header row, 2020-01 does
# not, and both must parse through the same CSV reader.
MARK_2024_01 = (
    1704067200000,
    42313.90000000,
    42328.34025532,
    42289.70000000,
    42326.90000000,
    0,
    1704067259999,
    0,
    60,
    0,
    0,
    0,
)
MARK_2020_01 = (
    1577836800000,
    7195.36581933,
    7195.36581933,
    7185.82196770,
    7186.83293715,
    0,
    1577836859999,
    0,
    60,
    0,
    0,
    0,
)


def _mark_zip(filename: str, rows: tuple[tuple[Any, ...], ...], header: bool) -> bytes:
    body = (f"{MARK_HEADER}\n" if header else "") + "".join(
        ",".join(str(v) for v in row) + "\n" for row in rows
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, body)
    return buf.getvalue()


JAN_2024_ZIP = _mark_zip("BTCUSDT-1m-2024-01.csv", (MARK_2024_01,), header=True)
JAN_2020_ZIP = _mark_zip("BTCUSDT-1m-2020-01.csv", (MARK_2020_01,), header=False)


def _mark_row(open_ms: int) -> tuple[Any, ...]:
    """A mark-price 1m row in Binance order (volume columns 0, count 60)."""
    return (open_ms, 1.0, 2.0, 0.5, 1.5, 0.0, open_ms + 59_999, 0.0, 60, 0.0, 0.0)


def _patch_mark_http(
    monkeypatch: pytest.MonkeyPatch,
    served: dict[str, bytes | Exception],
) -> list[str]:
    """Serve mark-price zips by filename; any other URL is a 404.

    Returns the requested URL paths so a test can assert the base URL.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        item = served.get(request.url.path.rsplit("/", 1)[-1])
        if item is None:
            return httpx.Response(404)
        if isinstance(item, Exception):
            raise item
        return httpx.Response(200, content=item)

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler))
    )
    return seen


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)
    return delays


@pytest.fixture
def archive_path(tmp_path: Path) -> Path:
    return tmp_path / "mark_archive.db"


@pytest.fixture
def archive(archive_path: Path) -> RawDataArchive:
    return RawDataArchive(archive_path)


def _one_month(year: int, month: int) -> tuple[datetime, datetime]:
    """A range covering exactly (year, month): first to last day."""
    start = datetime(year, month, 1, tzinfo=UTC)
    end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=UTC) - timedelta(days=1)
    return start, end


def _monthly(seen: list[str]) -> list[str]:
    return [p.rsplit("/", 1)[-1] for p in seen if "/monthly/" in p]


def _daily(seen: list[str]) -> list[str]:
    return [p.rsplit("/", 1)[-1] for p in seen if "/daily/" in p]


def _daily_urls(start: date, end: date) -> list[str]:
    out: list[str] = []
    cur = start
    while cur <= end:
        out.append(f"BTCUSDT-1m-{cur.isoformat()}.zip")
        cur += timedelta(days=1)
    return out


def _ingest(archive: RawDataArchive, start: datetime, end: datetime) -> dict[str, Any]:
    return ingest.ingest_mark_klines(
        "BTCUSDT", start_date=start, end_date=end, archive=archive, verbose=False
    )


class TestMarkZipParsing:
    def test_mark_zip_parsed_into_raw_mark_klines(
        self, archive: RawDataArchive, archive_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _patch_mark_http(
            monkeypatch,
            {
                "BTCUSDT-1m-2024-01.zip": JAN_2024_ZIP,
                "BTCUSDT-1m-2020-01.zip": JAN_2020_ZIP,
            },
        )
        counts = _ingest(archive, *_one_month(2024, 1))
        assert counts["mark_klines_fetched"] == 1
        assert counts["mark_klines_inserted"] == 1
        assert counts["failed_months"] == []
        assert counts["unavailable_months"] == []
        # The fixture monthly zip holds one minute, so every later day of the
        # range is a zero-row day past the coverage end: the tail pass walks
        # them from the DAILY zips (all 404 here) — contract M12.
        assert _monthly(seen) == ["BTCUSDT-1m-2024-01.zip"]
        assert _daily(seen) == _daily_urls(date(2024, 1, 2), date(2024, 1, 31))

        # Read through a fresh connection: an uncommitted write is invisible to it.
        archive.close()
        archive = RawDataArchive(archive_path)
        rows = archive.get_mark_klines("BTCUSDT", "1m")
        assert len(rows) == 1
        assert rows[0]["downloaded_at"] is not None
        assert (
            rows[0]["open_time"],
            rows[0]["open"],
            rows[0]["high"],
            rows[0]["low"],
            rows[0]["close"],
            rows[0]["close_time"],
            rows[0]["source"],
        ) == (
            1704067200000,
            42313.9,
            42328.34025532,
            42289.7,
            42326.9,
            1704067259999,
            "binance_vision",
        )

        # 2020 files have no header row; the same parser must handle them.
        counts_2020 = _ingest(archive, *_one_month(2020, 1))
        assert counts_2020["mark_klines_inserted"] == 1
        assert archive.get_mark_date_range("BTCUSDT", "1m") == (1577836800000, 1704067200000)

        # The catalog input path must stay empty.
        assert archive.get_kline_count("BTCUSDT", "1m") == 0
        assert archive.conn.execute("SELECT COUNT(*) FROM raw_klines").fetchone()[0] == 0


class TestUnavailableMonth:
    def test_404_month_is_unavailable_not_failed(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_mark_http(monkeypatch, {})
        counts = _ingest(archive, *_one_month(2024, 1))
        assert counts["unavailable_months"] == ["2024-01"]
        assert counts["failed_months"] == []
        assert counts["mark_klines_inserted"] == 0
        assert archive.get_mark_date_range("BTCUSDT", "1m") is None

    def test_cli_exit_0_when_only_unavailable(
        self,
        archive: RawDataArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _patch_mark_http(monkeypatch, {})
        monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: archive)
        rc = ingest.main(
            [
                "download-mark",
                "--symbols",
                "BTCUSDT",
                "--start",
                "2024-01-01",
                "--end",
                "2024-02-15",
            ]
        )
        assert rc == 0
        assert "unavailable" in capsys.readouterr().out


class TestPersistentFailure:
    def test_failure_retried_then_recorded_and_cli_exits_1(
        self,
        archive: RawDataArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        no_sleep: list[float],
    ) -> None:
        seen = _patch_mark_http(
            monkeypatch, {"BTCUSDT-1m-2024-01.zip": httpx.ConnectError("reset")}
        )
        counts = _ingest(archive, *_one_month(2024, 1))
        assert counts["failed_months"] == ["2024-01"]
        assert counts["unavailable_months"] == []
        attempts = len(downloader.MONTHLY_RETRY_DELAYS) + 1
        # the monthly zip is retried to exhaustion; with nothing archived the
        # whole range is then a tail the daily pass walks (and fails too)
        assert len(_monthly(seen)) == attempts
        assert no_sleep[: len(downloader.MONTHLY_RETRY_DELAYS)] == list(
            downloader.MONTHLY_RETRY_DELAYS
        )

        monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: archive)
        rc = ingest.main(
            [
                "download-mark",
                "--symbols",
                "BTCUSDT",
                "--start",
                "2024-01-01",
                "--end",
                "2024-02-15",
            ]
        )
        assert rc == 1
        assert "BTCUSDT: 2024-01" in capsys.readouterr().err


class TestIdempotent:
    def test_second_run_inserts_nothing_new(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_mark_http(monkeypatch, {"BTCUSDT-1m-2024-01.zip": JAN_2024_ZIP})
        first = _ingest(archive, *_one_month(2024, 1))
        second = _ingest(archive, *_one_month(2024, 1))
        assert first["mark_klines_inserted"] == 1
        assert second["mark_klines_inserted"] == 0
        assert archive.get_month_mark_kline_count("BTCUSDT", "1m", 2024, 1) == 1

    def test_replayed_key_never_rewrites_history(
        self, archive: RawDataArchive, archive_path: Path
    ) -> None:
        """Append-only: a duplicate key with different values changes nothing."""
        archive.insert_mark_klines("BTCUSDT", "1m", [MARK_2024_01], "binance_vision")
        archive.conn.execute("UPDATE raw_mark_klines SET downloaded_at = '2024-02-01 00:00:00'")
        archive.conn.commit()
        original = dict(archive.get_mark_klines("BTCUSDT", "1m")[0])

        changed = (MARK_2024_01[0], 1.0, 2.0, 0.5, 1.5, 0, MARK_2024_01[6], 0, 60, 0, 0, 0)
        assert archive.insert_mark_klines("BTCUSDT", "1m", [changed], "binance_api") == 0
        archive.close()

        reopened = RawDataArchive(archive_path)
        rows = reopened.get_mark_klines("BTCUSDT", "1m")
        assert [dict(r) for r in rows] == [original]
        assert original["downloaded_at"] == "2024-02-01 00:00:00"
        assert original["source"] == "binance_vision"
        assert original["close"] == 42326.9


class TestMonthCoverage:
    def test_mark_coverage_threshold_and_skip(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        jan1 = 1704067200000  # 2024-01-01 00:00:00 UTC
        expected = 31 * 24 * 60  # 44640 minutes in January
        below = int(expected * 0.9) - 1  # 40175
        archive.insert_mark_klines(
            "BTCUSDT", "1m", [_mark_row(jan1 + i * 60_000) for i in range(below)], "test"
        )
        assert archive.get_month_mark_kline_count("BTCUSDT", "1m", 2024, 1) == below
        assert archive.has_month_mark_coverage("BTCUSDT", "1m", 2024, 1) is False
        # close_time comes from tuple index 6, never the index-5 volume (0).
        assert archive.get_mark_klines("BTCUSDT", "1m")[0]["close_time"] == jan1 + 59_999

        archive.insert_mark_klines("BTCUSDT", "1m", [_mark_row(jan1 + below * 60_000)], "test")
        assert archive.has_month_mark_coverage("BTCUSDT", "1m", 2024, 1) is True

        # A covered month is skipped: its monthly zip is not requested. The
        # archived rows stop on Jan 28, so the tail pass still walks the
        # zero-row days after them (all 404 here) — contract M12.
        seen = _patch_mark_http(monkeypatch, {"BTCUSDT-1m-2024-01.zip": JAN_2024_ZIP})
        counts = _ingest(archive, *_one_month(2024, 1))
        assert counts["months_skipped"] == 1
        assert counts["mark_klines_inserted"] == 0
        assert _monthly(seen) == []
        assert _daily(seen) == _daily_urls(date(2024, 1, 29), date(2024, 1, 31))


class TestFinalPartialMonth:
    def test_historical_same_month_range_is_downloaded(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """2024-01-01..2024-01-31 requests January; no REST path exists to fill it."""
        seen = _patch_mark_http(monkeypatch, {"BTCUSDT-1m-2024-01.zip": JAN_2024_ZIP})
        counts = _ingest(
            archive, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC)
        )
        # Jan 1 is the only archived minute, so the tail pass requests Jan 2-31
        # from the daily zips (all 404 here) — contract M12.
        assert _monthly(seen) == ["BTCUSDT-1m-2024-01.zip"]
        assert _daily(seen) == _daily_urls(date(2024, 1, 2), date(2024, 1, 31))
        assert counts["mark_klines_inserted"] == 1
        assert counts["unavailable_months"] == []

    def test_range_end_month_requested_and_404_classified_unavailable(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _patch_mark_http(monkeypatch, {"BTCUSDT-1m-2024-01.zip": JAN_2024_ZIP})
        counts = _ingest(
            archive, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 15, tzinfo=UTC)
        )
        assert _monthly(seen) == [
            "BTCUSDT-1m-2024-01.zip",
            "BTCUSDT-1m-2024-02.zip",
        ]
        # coverage ends on Jan 1 (the monthly fixture's only minute), so the
        # tail runs from Jan 2 to the range end, Feb's days included
        assert _daily(seen) == _daily_urls(date(2024, 1, 2), date(2024, 2, 15))
        assert counts["mark_klines_inserted"] == 1
        assert counts["unavailable_months"] == ["2024-02"]
        assert counts["failed_months"] == []

    def test_current_month_only_cli_reports_gap_and_coverage_end(
        self,
        archive: RawDataArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        sep_30 = int(datetime(2026, 9, 30, 23, 59, tzinfo=UTC).timestamp() * 1000)
        archive.insert_mark_klines("BTCUSDT", "1m", [_mark_row(sep_30)], "binance_vision")
        seen = _patch_mark_http(monkeypatch, {})
        monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: archive)
        rc = ingest.main(
            [
                "download-mark",
                "--symbols",
                "BTCUSDT",
                "--start",
                "2026-10-01",
                "--end",
                "2026-10-12",
            ]
        )
        assert rc == 0  # unavailable is not a failure, but it must be visible
        assert _monthly(seen) == ["BTCUSDT-1m-2026-10.zip"]
        assert _daily(seen) == _daily_urls(date(2026, 10, 1), date(2026, 10, 12))
        out = capsys.readouterr().out
        assert "2026-10: unavailable (not yet published) - mark coverage ends 2026-09-30" in out
        assert "1 unavailable" in out


class TestMonthEnumerationBoundaries:
    """Endpoints are normalised to UTC and months enumerated on midnight boundaries."""

    def test_late_start_time_of_day_still_requests_the_inclusive_end_month(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _patch_mark_http(monkeypatch, {"BTCUSDT-1m-2024-01.zip": JAN_2024_ZIP})
        counts = _ingest(
            archive,
            datetime(2024, 1, 31, 23, 59, tzinfo=UTC),
            datetime(2024, 2, 1, 0, 0, tzinfo=UTC),
        )
        assert _monthly(seen) == [
            "BTCUSDT-1m-2024-01.zip",
            "BTCUSDT-1m-2024-02.zip",
        ]
        # the tail starts right after the Jan 1 coverage end, so Jan 31 is in it
        assert _daily(seen) == _daily_urls(date(2024, 1, 31), date(2024, 2, 1))
        assert counts["unavailable_months"] == ["2024-02"]

    def test_non_utc_interval_is_normalised_to_utc_months(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 2024-02-01 00:30 +01:00 == 2024-01-31 23:30 UTC; end 2024-02-01 01:30 +01:00
        # == 2024-02-01 00:30 UTC. Local month is February only, UTC spans Jan + Feb.
        plus_one = timezone(timedelta(hours=1))
        seen = _patch_mark_http(monkeypatch, {"BTCUSDT-1m-2024-01.zip": JAN_2024_ZIP})
        _ingest(
            archive,
            datetime(2024, 2, 1, 0, 30, tzinfo=plus_one),
            datetime(2024, 2, 1, 1, 30, tzinfo=plus_one),
        )
        assert _monthly(seen) == [
            "BTCUSDT-1m-2024-01.zip",
            "BTCUSDT-1m-2024-02.zip",
        ]
        # the range starts 2024-01-31 23:30 UTC, after the Jan 1 coverage end
        assert _daily(seen) == _daily_urls(date(2024, 1, 31), date(2024, 2, 1))
