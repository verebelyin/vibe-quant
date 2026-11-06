"""Interior missing-day backfill from Binance Vision DAILY zips.

Binance Vision monthly zips silently omit whole UTC days (SOLUSDT 1m 2022-02
ships 25/28 days) and the >=90% month-coverage smart skip hides that hole, so a
re-ingest never fills it: ``has_month_coverage`` still says "covered".

After the monthly pass each symbol's *interior* missing 1m days — zero rows
strictly between its first and last archived day — are fetched from the daily
zips and archived with source ``binance_vision_daily`` (both raw_klines and
raw_mark_klines). A 404/failed daily zip is recorded in
``counts['missing_days']`` and makes the download session ``partial``, but is
not fatal. ``verify`` reports the same interior holes.

All HTTP is stubbed with a MockTransport; every archive is a tmp_path db.

regression: vibe-quant-cekup
"""

from __future__ import annotations

import calendar
import io
import json
import zipfile
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from vibe_quant.data import downloader, ingest
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.data.downloader import (
    DailyDownloadError,
    download_daily_klines,
    download_daily_mark_klines,
)
from vibe_quant.data.verify import interior_missing_days

if TYPE_CHECKING:
    from pathlib import Path

DAY_MS = 86_400_000
JAN_HOLE = date(2024, 1, 4)
JAN_FIRST = datetime(2024, 1, 1, tzinfo=UTC)
FEB_FIRST = datetime(2024, 2, 1, tzinfo=UTC)

HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)


def _day_start_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


def _days(start: date, end: date) -> list[date]:
    out: list[date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _kline(open_ms: int, price: float = 100.0) -> tuple[Any, ...]:
    return (
        open_ms,
        price,
        price + 1.0,
        price - 1.0,
        price + 0.5,
        10.0,
        open_ms + 59_999,
        15.0,
        3,
        5.0,
        7.0,
    )


def _mark(open_ms: int) -> tuple[Any, ...]:
    return (open_ms, 1.0, 2.0, 0.5, 1.5, 0.0, open_ms + 59_999, 0.0, 60, 0.0, 0.0)


def _month_day_rows(
    year: int, month: int, *, skip: set[date] | None = None, mark: bool = False
) -> list[tuple[Any, ...]]:
    """One row at 00:00 UTC on every day of the month except *skip*.

    Day granularity is what ``missing_days`` counts, so one row per day is a
    "complete" month for the purposes of this test (and keeps zips tiny).
    """
    _, ndays = calendar.monthrange(year, month)
    rows: list[tuple[Any, ...]] = []
    for day_num in range(1, ndays + 1):
        day = date(year, month, day_num)
        if skip and day in skip:
            continue
        open_ms = _day_start_ms(day)
        rows.append(_mark(open_ms) if mark else _kline(open_ms))
    return rows


def _full_day_rows(day: date, *, mark: bool = False, n: int = 1440) -> list[tuple[Any, ...]]:
    base = _day_start_ms(day)
    if mark:
        return [_mark(base + i * 60_000) for i in range(n)]
    return [_kline(base + i * 60_000, 100.0 + i * 0.01) for i in range(n)]


def _zip_bytes(filename: str, rows: list[tuple[Any, ...]], *, header: bool = False) -> bytes:
    body = (f"{HEADER}\n" if header else "") + "".join(
        ",".join(str(v) for v in row) + "\n" for row in rows
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, body)
    return buf.getvalue()


def _monthly_zip(symbol: str, year: int, month: int, rows: list[tuple[Any, ...]]) -> bytes:
    return _zip_bytes(f"{symbol}-1m-{year}-{month:02d}.csv", rows, header=True)


def _daily_zip(symbol: str, day: date, rows: list[tuple[Any, ...]]) -> bytes:
    # 2020 Vision files have no header, 2023+ daily mark files do; the parser
    # must accept either through the same reader.
    return _zip_bytes(f"{symbol}-1m-{day.isoformat()}.csv", rows, header=False)


def _patch_http(
    monkeypatch: pytest.MonkeyPatch, served: dict[str, bytes | Exception]
) -> list[str]:
    """Serve Vision zips by filename; any other URL is a 404.

    Returns the requested URL paths so a test can assert which zips were hit.
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


def _daily(seen: list[str]) -> list[str]:
    return [p.rsplit("/", 1)[-1] for p in seen if "/daily/" in p]


def _daily_paths(seen: list[str]) -> list[str]:
    return [p for p in seen if "/daily/" in p]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)


class _FakeInstrument:
    id = "BTCUSDT-PERP.BINANCE"
    size_precision = 8
    price_precision = 2


class _FakeCatalog:
    """No-op catalog: keeps the D1-D6 ingest tests off the parquet path."""

    def write_instrument(self, *a: object, **k: object) -> None: ...
    def clear_bar_data(self, *a: object, **k: object) -> None: ...
    def write_bars(self, *a: object, **k: object) -> None: ...


@pytest.fixture
def archive(tmp_path: Path) -> RawDataArchive:
    arc = RawDataArchive(tmp_path / "raw.db")
    yield arc
    arc.close()


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the bar/REST machinery so ingest_symbol runs without a real catalog."""
    monkeypatch.setattr(ingest, "create_instrument", lambda _s: _FakeInstrument())
    monkeypatch.setattr(ingest, "klines_to_bars", lambda *a, **k: [])
    monkeypatch.setattr(ingest, "aggregate_bars", lambda *a, **k: [])
    monkeypatch.setattr(ingest, "get_bar_type", lambda *a, **k: None)
    monkeypatch.setattr(ingest, "download_recent_klines", lambda *a, **k: [])


def _run_symbol(archive: RawDataArchive, **kw: Any) -> dict[str, Any]:
    return ingest.ingest_symbol(
        "BTCUSDT",
        archive=archive,
        catalog=_FakeCatalog(),  # type: ignore[arg-type]
        verbose=False,
        start_date=kw.pop("start_date", JAN_FIRST),
        end_date=kw.pop("end_date", FEB_FIRST),
        **kw,
    )


def _run_mark(archive: RawDataArchive, **kw: Any) -> dict[str, Any]:
    return ingest.ingest_mark_klines(
        "BTCUSDT",
        archive=archive,
        verbose=False,
        start_date=kw.pop("start_date", JAN_FIRST),
        end_date=kw.pop("end_date", datetime(2024, 1, 31, tzinfo=UTC)),
        **kw,
    )


class TestD1CompleteMonthlyMonth:
    def test_no_daily_request_and_second_run_is_idempotent(
        self, archive: RawDataArchive, fast: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _month_day_rows(2024, 1)
        seen = _patch_http(monkeypatch, {"BTCUSDT-1m-2024-01.zip": _monthly_zip("BTCUSDT", 2024, 1, rows)})

        first = _run_symbol(archive)
        assert first["klines_inserted"] == 31
        assert first["missing_days"] == []
        assert _daily(seen) == []
        assert archive.get_kline_count("BTCUSDT", "1m") == 31

        seen.clear()
        second = _run_symbol(archive)
        assert second["klines_inserted"] == 0
        assert second["missing_days"] == []
        assert _daily(seen) == []
        assert archive.get_kline_count("BTCUSDT", "1m") == 31


class TestD2InteriorHoleFilledFromDaily:
    def test_one_daily_request_and_daily_source(
        self, archive: RawDataArchive, fast: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _month_day_rows(2024, 1, skip={JAN_HOLE})
        seen = _patch_http(
            monkeypatch,
            {
                "BTCUSDT-1m-2024-01.zip": _monthly_zip("BTCUSDT", 2024, 1, rows),
                "BTCUSDT-1m-2024-01-04.zip": _daily_zip(
                    "BTCUSDT", JAN_HOLE, _full_day_rows(JAN_HOLE)
                ),
            },
        )

        counts = _run_symbol(archive)

        assert counts["missing_days"] == []
        assert counts["klines_inserted"] == 30 + 1440
        assert _daily(seen) == ["BTCUSDT-1m-2024-01-04.zip"]
        assert _daily_paths(seen)[0].startswith("/data/futures/um/daily/klines/BTCUSDT/1m/")
        assert archive.get_month_kline_count("BTCUSDT", "1m", 2024, 1) == 1470

        lo = _day_start_ms(JAN_HOLE)
        sources = {
            row[0]
            for row in archive.conn.execute(
                "SELECT DISTINCT source FROM raw_klines WHERE open_time >= ? AND open_time < ?",
                (lo, lo + DAY_MS),
            )
        }
        assert sources == {"binance_vision_daily"}

    def test_daily_zip_404_is_not_fatal(
        self, archive: RawDataArchive, fast: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _month_day_rows(2024, 1, skip={JAN_HOLE})
        _patch_http(
            monkeypatch, {"BTCUSDT-1m-2024-01.zip": _monthly_zip("BTCUSDT", 2024, 1, rows)}
        )

        counts = _run_symbol(archive)
        assert counts["missing_days"] == ["2024-01-04"]
        assert archive.get_month_kline_count("BTCUSDT", "1m", 2024, 1) == 30

    def test_failed_daily_download_is_recorded_not_raised(
        self, archive: RawDataArchive, fast: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _month_day_rows(2024, 1, skip={JAN_HOLE})
        _patch_http(
            monkeypatch,
            {
                "BTCUSDT-1m-2024-01.zip": _monthly_zip("BTCUSDT", 2024, 1, rows),
                "BTCUSDT-1m-2024-01-04.zip": httpx.ConnectError("reset"),
            },
        )

        counts = _run_symbol(archive)
        assert counts["missing_days"] == ["2024-01-04"]


class TestD3CoveredMonthStillFillsHole:
    def test_monthly_skipped_but_daily_hole_fetched(
        self, archive: RawDataArchive, fast: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hole = date(2024, 1, 15)
        archive.insert_klines(
            "BTCUSDT",
            "1m",
            [
                _kline(_day_start_ms(day) + minute * 60_000)
                for day in _days(date(2024, 1, 1), date(2024, 1, 31))
                if day != hole
                for minute in range(1440)
            ],
            "test",
        )
        assert archive.has_month_coverage("BTCUSDT", "1m", 2024, 1) is True

        seen = _patch_http(
            monkeypatch,
            {
                "BTCUSDT-1m-2024-01-15.zip": _daily_zip(
                    "BTCUSDT", hole, _full_day_rows(hole)
                )
            },
        )

        counts = _run_symbol(archive)

        assert counts["months_skipped"] == 1
        assert counts["months_downloaded"] == 0
        assert _daily(seen) == ["BTCUSDT-1m-2024-01-15.zip"]  # no monthly zip at all
        assert "BTCUSDT-1m-2024-01.zip" not in seen
        assert counts["missing_days"] == []
        assert counts["klines_inserted"] == 1440
        assert archive.get_month_kline_count("BTCUSDT", "1m", 2024, 1) == 44_640
        assert archive.has_month_coverage("BTCUSDT", "1m", 2024, 1) is True


class TestD4SessionStatus:
    def test_missing_days_make_session_partial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "raw.db"
        monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: RawDataArchive(db))
        monkeypatch.setattr(ingest, "CatalogManager", lambda *a, **k: _FakeCatalog())
        monkeypatch.setattr(ingest, "create_instrument", lambda _s: _FakeInstrument())
        monkeypatch.setattr(ingest, "klines_to_bars", lambda *a, **k: [])
        monkeypatch.setattr(ingest, "aggregate_bars", lambda *a, **k: [])
        monkeypatch.setattr(ingest, "get_bar_type", lambda *a, **k: None)
        monkeypatch.setattr(ingest, "download_recent_klines", lambda *a, **k: [])
        monkeypatch.setattr(ingest, "ingest_funding_rates", lambda *a, **k: 0)

        rows = _month_day_rows(2024, 1, skip={JAN_HOLE})
        _patch_http(
            monkeypatch, {"BTCUSDT-1m-2024-01.zip": _monthly_zip("BTCUSDT", 2024, 1, rows)}
        )

        results = ingest.ingest_all(
            symbols=["BTCUSDT"], start_date=JAN_FIRST, end_date=FEB_FIRST, verbose=False
        )
        assert results["BTCUSDT"]["missing_days"] == ["2024-01-04"]

        session = RawDataArchive(db)
        try:
            row = session.get_download_sessions(1)[0]
        finally:
            session.close()
        assert row["status"] == "partial"
        assert json.loads(row["error_message"])["BTCUSDT"]["missing_days"] == ["2024-01-04"]

    def test_cli_exit_stays_0_for_missing_days_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A missing day is a recorded hole, not a failure: only failed_months /
        # failed_ranges are fatal for the ingest CLI.
        monkeypatch.setattr(
            ingest,
            "ingest_all",
            lambda **k: {
                "BTCUSDT": {
                    "failed_months": [],
                    "failed_ranges": [],
                    "missing_days": ["2024-01-04"],
                }
            },
        )
        assert ingest.main(["ingest", "--symbols", "BTCUSDT"]) == 0


class TestD5InteriorBoundaries:
    def test_pre_listing_and_post_last_days_not_requested(
        self, archive: RawDataArchive, fast: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        present = [day for day in _days(date(2024, 1, 10), date(2024, 1, 20)) if day.day != 15]
        archive.insert_klines(
            "BTCUSDT", "1m", [_kline(_day_start_ms(d)) for d in present], "test"
        )
        hole = date(2024, 1, 15)
        seen = _patch_http(
            monkeypatch,
            {"BTCUSDT-1m-2024-01-15.zip": _daily_zip("BTCUSDT", hole, _full_day_rows(hole))},
        )

        counts = _run_symbol(
            archive,
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, tzinfo=UTC),
        )

        assert _daily(seen) == ["BTCUSDT-1m-2024-01-15.zip"]
        assert counts["missing_days"] == []

    def test_range_beyond_last_archived_day_requests_nothing(
        self, archive: RawDataArchive, fast: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive.insert_klines(
            "BTCUSDT",
            "1m",
            [_kline(_day_start_ms(date(2024, 1, d))) for d in range(1, 6)],
            "test",
        )
        seen = _patch_http(monkeypatch, {})

        counts = _run_symbol(
            archive,
            start_date=datetime(2024, 1, 10, tzinfo=UTC),
            end_date=datetime(2024, 1, 20, tzinfo=UTC),
        )

        assert _daily(seen) == []
        assert counts["missing_days"] == []


class TestD6MarkKlines:
    def test_complete_mark_month_no_daily_request(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _month_day_rows(2024, 1, mark=True)
        seen = _patch_http(
            monkeypatch, {"BTCUSDT-1m-2024-01.zip": _monthly_zip("BTCUSDT", 2024, 1, rows)}
        )
        counts = _run_mark(archive)
        assert counts["mark_klines_inserted"] == 31
        assert counts["missing_days"] == []
        assert _daily(seen) == []

    def test_interior_mark_hole_filled_from_daily(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _month_day_rows(2024, 1, skip={JAN_HOLE}, mark=True)
        seen = _patch_http(
            monkeypatch,
            {
                "BTCUSDT-1m-2024-01.zip": _monthly_zip("BTCUSDT", 2024, 1, rows),
                "BTCUSDT-1m-2024-01-04.zip": _daily_zip(
                    "BTCUSDT", JAN_HOLE, _full_day_rows(JAN_HOLE, mark=True)
                ),
            },
        )

        counts = _run_mark(archive)

        assert counts["missing_days"] == []
        assert counts["mark_klines_inserted"] == 30 + 1440
        assert _daily(seen) == ["BTCUSDT-1m-2024-01-04.zip"]
        assert _daily_paths(seen)[0].startswith(
            "/data/futures/um/daily/markPriceKlines/BTCUSDT/1m/"
        )

        lo = _day_start_ms(JAN_HOLE)
        sources = {
            row[0]
            for row in archive.conn.execute(
                "SELECT DISTINCT source FROM raw_mark_klines WHERE open_time >= ? AND open_time < ?",
                (lo, lo + DAY_MS),
            )
        }
        assert sources == {"binance_vision_daily"}
        # mark rows must never leak into the catalog input table
        assert archive.get_kline_count("BTCUSDT", "1m") == 0

    def test_missing_mark_day_404_recorded(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _month_day_rows(2024, 1, skip={JAN_HOLE}, mark=True)
        _patch_http(
            monkeypatch, {"BTCUSDT-1m-2024-01.zip": _monthly_zip("BTCUSDT", 2024, 1, rows)}
        )
        counts = _run_mark(archive)
        assert counts["missing_days"] == ["2024-01-04"]
        assert counts["failed_months"] == []


class TestDailyDownloaderContract:
    def test_404_returns_none_without_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(404)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        assert download_daily_klines("BTCUSDT", "1m", JAN_HOLE, client=client) is None
        assert len(calls) == 1

    def test_klines_and_mark_urls_are_daily_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _full_day_rows(JAN_HOLE)
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return httpx.Response(200, content=_daily_zip("BTCUSDT", JAN_HOLE, rows))

        client = httpx.Client(transport=httpx.MockTransport(handler))
        assert download_daily_klines("BTCUSDT", "1m", JAN_HOLE, client=client) is not None
        assert download_daily_mark_klines("BTCUSDT", "1m", JAN_HOLE, client=client) is not None
        assert seen == [
            "/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01-04.zip",
            "/data/futures/um/daily/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2024-01-04.zip",
        ]

    def test_transient_failure_is_retried_then_raises_daily_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(503)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(DailyDownloadError):
            download_daily_klines("BTCUSDT", "1m", JAN_HOLE, client=client)
        assert len(calls) == len(downloader.MONTHLY_RETRY_DELAYS) + 1


class TestD7VerifyReportsMissingDays:
    def _seed(self, archive: RawDataArchive) -> None:
        archive.insert_klines(
            "BTCUSDT",
            "1m",
            [_kline(_day_start_ms(d)) for d in (date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4))],
            "test",
        )
        archive.insert_mark_klines(
            "BTCUSDT",
            "1m",
            [_mark(_day_start_ms(d)) for d in (date(2024, 1, 1), date(2024, 1, 4))],
            "test",
        )

    def test_interior_missing_days_function(self, archive: RawDataArchive) -> None:
        self._seed(archive)
        assert interior_missing_days(archive, "BTCUSDT") == {
            "klines": ["2024-01-03"],
            "mark_klines": ["2024-01-02", "2024-01-03"],
        }

    def test_cli_lists_dates_and_fails(
        self,
        archive: RawDataArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._seed(archive)
        monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: archive)

        rc = ingest.main(["verify", "--symbols", "BTCUSDT"])
        out = capsys.readouterr().out

        assert rc == 1
        assert "MISSING 1m DAYS" in out
        assert "2024-01-03" in out  # klines hole
        assert "2024-01-02" in out  # mark hole
        assert "interior missing" in out

    def test_clean_archive_reports_no_missing_days(
        self,
        archive: RawDataArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Full contiguous days: no minute gaps and no day holes -> verify passes.
        archive.insert_klines(
            "BTCUSDT",
            "1m",
            [
                row
                for day in _days(date(2024, 1, 1), date(2024, 1, 3))
                for row in _full_day_rows(day)
            ],
            "test",
        )
        monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: archive)

        rc = ingest.main(["verify", "--symbols", "BTCUSDT"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Missing 1m days: None" in out
        assert "ALL CHECKS PASSED" in out


class TestD8CatalogGetsFilledDay:
    def test_daily_filled_day_reaches_catalog(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _month_day_rows(2024, 1, skip={JAN_HOLE})
        _patch_http(
            monkeypatch,
            {
                "BTCUSDT-1m-2024-01.zip": _monthly_zip("BTCUSDT", 2024, 1, rows),
                "BTCUSDT-1m-2024-01-04.zip": _daily_zip(
                    "BTCUSDT", JAN_HOLE, _full_day_rows(JAN_HOLE)
                ),
            },
        )

        arc = RawDataArchive(tmp_path / "raw.db")
        cm = CatalogManager(tmp_path / "catalog")
        try:
            counts = ingest.ingest_symbol(
                "BTCUSDT",
                archive=arc,
                catalog=cm,
                verbose=False,
                start_date=JAN_FIRST,
                end_date=FEB_FIRST,
            )
            assert counts["bars_1m"] == 30 + 1440
            assert cm.get_bar_count("BTCUSDT", "1m") == 1470

            day_bars = cm.get_bars(
                "BTCUSDT",
                "1m",
                start=datetime(2024, 1, 4, tzinfo=UTC),
                end=datetime(2024, 1, 5, tzinfo=UTC),
            )
            assert len(day_bars) == 1440
        finally:
            arc.close()


def _patch_ingest_all(
    monkeypatch: pytest.MonkeyPatch, db: Path, rest_rows: list[tuple[Any, ...]]
) -> None:
    """Run the real ingest_all/session persistence on a tmp db; REST serves *rest_rows*."""
    monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: RawDataArchive(db))
    monkeypatch.setattr(ingest, "CatalogManager", lambda *a, **k: _FakeCatalog())
    monkeypatch.setattr(ingest, "create_instrument", lambda _s: _FakeInstrument())
    monkeypatch.setattr(ingest, "klines_to_bars", lambda *a, **k: [])
    monkeypatch.setattr(ingest, "aggregate_bars", lambda *a, **k: [])
    monkeypatch.setattr(ingest, "get_bar_type", lambda *a, **k: None)
    monkeypatch.setattr(ingest, "download_recent_klines", lambda *a, **k: list(rest_rows))
    monkeypatch.setattr(ingest, "ingest_funding_rates", lambda *a, **k: 0)


def _last_session(db: Path) -> dict[str, Any]:
    session = RawDataArchive(db)
    try:
        return dict(session.get_download_sessions(1)[0])
    finally:
        session.close()


class TestR9FinalMissingDaysAfterAllFillPasses:
    """counts['missing_days'] = unresolved interior days AFTER daily AND REST passes."""

    def test_rest_repairs_day_daily_404_session_completed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "raw.db"
        seed = RawDataArchive(db)
        seed.insert_klines(
            "BTCUSDT", "1m", [_kline(_day_start_ms(date(2024, 1, d))) for d in (1, 3)], "test"
        )
        seed.close()
        # Daily zip for Jan 2 is a 404; REST then fills the day.
        _patch_ingest_all(monkeypatch, db, [_kline(_day_start_ms(date(2024, 1, 2)))])
        seen = _patch_http(monkeypatch, {})

        results = ingest.ingest_all(
            symbols=["BTCUSDT"],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 5, tzinfo=UTC),
            verbose=False,
        )

        assert _daily(seen) == ["BTCUSDT-1m-2024-01-02.zip"]  # daily was tried and 404'd
        check = RawDataArchive(db)
        try:
            assert check.missing_days("BTCUSDT", "1m") == []  # REST really repaired it
        finally:
            check.close()
        assert results["BTCUSDT"]["missing_days"] == []
        row = _last_session(db)
        assert row["status"] == "completed"
        assert row["error_message"] is None

    def test_rest_exposes_hole_session_partial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "raw.db"
        seed = RawDataArchive(db)
        seed.insert_klines(
            "BTCUSDT", "1m", [_kline(_day_start_ms(date(2024, 1, 1)))], "test"
        )
        seed.close()
        # One archived day -> no interior hole before REST; REST adds Jan 3 and
        # thereby makes Jan 2 an interior hole nobody has tried to fill.
        _patch_ingest_all(monkeypatch, db, [_kline(_day_start_ms(date(2024, 1, 3)))])
        _patch_http(monkeypatch, {})

        results = ingest.ingest_all(
            symbols=["BTCUSDT"],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 5, tzinfo=UTC),
            verbose=False,
        )

        assert results["BTCUSDT"]["missing_days"] == ["2024-01-02"]
        row = _last_session(db)
        assert row["status"] == "partial"
        assert json.loads(row["error_message"])["BTCUSDT"]["missing_days"] == ["2024-01-02"]


class TestRequestedRangeBoundsMissingDays:
    def test_holes_before_start_and_after_end_are_not_requested(
        self, archive: RawDataArchive, fast: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Holes: 3, 6, 8, 11. Archive spans Jan 1..12, so all four are interior;
        # only 6 and 8 lie inside the requested [Jan 5, Jan 8].
        present = [1, 2, 4, 5, 7, 9, 10, 12]
        archive.insert_klines(
            "BTCUSDT",
            "1m",
            [_kline(_day_start_ms(date(2024, 1, d))) for d in present],
            "test",
        )
        served = {
            f"BTCUSDT-1m-2024-01-{d:02d}.zip": _daily_zip(
                "BTCUSDT", date(2024, 1, d), _full_day_rows(date(2024, 1, d))
            )
            for d in (3, 6, 8, 11)
        }
        seen = _patch_http(monkeypatch, served)

        counts = _run_symbol(
            archive,
            start_date=datetime(2024, 1, 5, tzinfo=UTC),
            end_date=datetime(2024, 1, 8, tzinfo=UTC),
        )

        assert sorted(_daily(seen)) == [
            "BTCUSDT-1m-2024-01-06.zip",
            "BTCUSDT-1m-2024-01-08.zip",
        ]
        assert counts["missing_days"] == []


class TestCoveredMarkMonthStillFillsHole:
    def test_mark_month_over_coverage_threshold_with_interior_hole(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hole = date(2024, 1, 15)
        archive.insert_mark_klines(
            "BTCUSDT",
            "1m",
            [
                _mark(_day_start_ms(day) + minute * 60_000)
                for day in _days(date(2024, 1, 1), date(2024, 1, 31))
                if day != hole
                for minute in range(1440)
            ],
            "test",
        )
        assert archive.has_month_mark_coverage("BTCUSDT", "1m", 2024, 1) is True
        seen = _patch_http(
            monkeypatch,
            {
                "BTCUSDT-1m-2024-01-15.zip": _daily_zip(
                    "BTCUSDT", hole, _full_day_rows(hole, mark=True)
                )
            },
        )

        counts = _run_mark(archive)

        assert counts["months_skipped"] == 1
        assert counts["months_downloaded"] == 0
        assert _daily(seen) == ["BTCUSDT-1m-2024-01-15.zip"]
        assert "BTCUSDT-1m-2024-01.zip" not in seen
        assert counts["missing_days"] == []
        assert counts["mark_klines_inserted"] == 1440


class TestVerifyMarkOnlyHoles:
    def test_clean_raw_with_mark_only_hole_exits_1(
        self,
        archive: RawDataArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        archive.insert_klines(
            "BTCUSDT",
            "1m",
            [row for day in _days(date(2024, 1, 1), date(2024, 1, 3)) for row in _full_day_rows(day)],
            "test",
        )
        archive.insert_mark_klines(
            "BTCUSDT",
            "1m",
            [_mark(_day_start_ms(d)) for d in (date(2024, 1, 1), date(2024, 1, 3))],
            "test",
        )
        monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: archive)

        rc = ingest.main(["verify", "--symbols", "BTCUSDT"])
        out = capsys.readouterr().out

        assert rc == 1
        assert "MISSING 1m DAYS (mark klines)" in out
        assert "2024-01-02" in out
        assert "ISSUES FOUND" in out

    def test_default_verify_covers_mark_only_symbol(
        self,
        archive: RawDataArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # ETH has raw klines only; BTC exists only in raw_mark_klines (a fresh
        # archive populated through download-mark) and has an interior hole.
        archive.insert_klines(
            "ETHUSDT",
            "1m",
            [row for day in _days(date(2024, 1, 1), date(2024, 1, 2)) for row in _full_day_rows(day)],
            "test",
        )
        archive.insert_mark_klines(
            "BTCUSDT",
            "1m",
            [_mark(_day_start_ms(d)) for d in (date(2024, 1, 1), date(2024, 1, 3))],
            "test",
        )
        monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: archive)

        rc = ingest.main(["verify"])
        out = capsys.readouterr().out

        assert "No symbols to verify" not in out
        assert "ETHUSDT" in out
        assert "BTCUSDT" in out
        assert "MISSING 1m DAYS (mark klines)" in out
        assert "2024-01-02" in out
        assert rc == 1
