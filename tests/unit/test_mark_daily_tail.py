"""MARK 1m tail backfill from daily Vision zips (vibe-quant-6n2jo).

The monthly ``markPriceKlines`` zip for the current, not-yet-published month 404s,
so ``download-mark`` used to stop at the last published month. The interior pass
cannot repair the tail either: ``RawDataArchive.missing_days`` clamps to the
symbol's own last archived day, so days past the coverage end are never reported.

``ingest_mark_klines`` now fills every zero-row day of an *unavailable* month from
the daily zips (published ~a day after each UTC day closes), before the interior
pass, and never requests the current UTC day. A 404 day lands in
``counts['unpublished_days']`` and a failed download in ``counts['failed_days']``
(both ISO, oldest first) — holes, not failures. ``now`` is injectable.

All HTTP is stubbed with a MockTransport; every archive is a tmp_path db.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from vibe_quant.data import downloader, ingest
from vibe_quant.data.archive import RawDataArchive

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
START = datetime(2026, 10, 1, tzinfo=UTC)
SEP_30 = date(2026, 9, 30)

HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
)


def _day_start_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


def _mark(open_ms: int) -> tuple[Any, ...]:
    return (open_ms, 1.0, 2.0, 0.5, 1.5, 0.0, open_ms + 59_999, 0.0, 60, 0.0, 0.0)


def _zip_bytes(filename: str, rows: list[tuple[Any, ...]], *, header: bool) -> bytes:
    body = (f"{HEADER}\n" if header else "") + "".join(
        ",".join(str(v) for v in row) + "\n" for row in rows
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, body)
    return buf.getvalue()


def _daily_zip(day: date, *, rows: int = 1) -> bytes:
    base = _day_start_ms(day)
    body = [_mark(base + i * 60_000) for i in range(rows)]
    return _zip_bytes(f"BTCUSDT-1m-{day.isoformat()}.csv", body, header=False)


def _monthly_zip(days: list[date], *, rows: int = 1) -> bytes:
    body = [
        _mark(_day_start_ms(day) + i * 60_000) for day in days for i in range(rows)
    ]
    return _zip_bytes("BTCUSDT-1m-2026-10.csv", body, header=True)


def _oct_days(first: int, last: int) -> list[date]:
    return [date(2026, 10, d) for d in range(first, last + 1)]


def _oct_urls(days: list[date]) -> list[str]:
    return [f"BTCUSDT-1m-{day.isoformat()}.zip" for day in days]


def _daily(seen: list[str]) -> list[str]:
    return [p.rsplit("/", 1)[-1] for p in seen if "/daily/" in p]


def _monthly(seen: list[str]) -> list[str]:
    return [p.rsplit("/", 1)[-1] for p in seen if "/monthly/" in p]


def _patch_http(
    monkeypatch: pytest.MonkeyPatch, served: dict[str, bytes | Exception]
) -> list[str]:
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
def archive(tmp_path: Path) -> RawDataArchive:
    return RawDataArchive(tmp_path / "mark_tail.db")


def _seed_sep30(archive: RawDataArchive) -> int:
    open_ms = _day_start_ms(SEP_30) + 23 * 3_600_000
    archive.insert_mark_klines("BTCUSDT", "1m", [_mark(open_ms)], "binance_vision")
    return open_ms


def _run(
    archive: RawDataArchive,
    *,
    start: datetime = START,
    end: datetime | None = datetime(2026, 10, 31, tzinfo=UTC),
    now: datetime = NOW,
    verbose: bool = False,
) -> dict[str, Any]:
    return ingest.ingest_mark_klines(
        "BTCUSDT",
        start_date=start,
        end_date=end,
        archive=archive,
        verbose=verbose,
        now=now,
    )


class TestM1TailFilledFromDaily:
    def test_unpublished_month_days_fetched_until_yesterday(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sep30 = _seed_sep30(archive)
        wanted = _oct_days(1, 16)
        served = {f"BTCUSDT-1m-{d.isoformat()}.zip": _daily_zip(d) for d in wanted}
        seen = _patch_http(monkeypatch, served)

        counts = _run(archive)

        assert counts["unavailable_months"] == ["2026-10"]
        assert _monthly(seen) == ["BTCUSDT-1m-2026-10.zip"]  # monthly still tried
        assert _daily(seen) == _oct_urls(wanted)
        assert counts["mark_klines_inserted"] == 16
        assert counts["unpublished_days"] == []
        assert counts["failed_days"] == []
        assert counts["missing_days"] == []  # coverage extended, no interior hole
        assert archive.get_mark_date_range("BTCUSDT", "1m") == (
            sep30,
            _day_start_ms(date(2026, 10, 16)),
        )

        last = _day_start_ms(date(2026, 10, 16))
        sources = {
            row[0]
            for row in archive.conn.execute(
                "SELECT DISTINCT source FROM raw_mark_klines WHERE open_time >= ? AND open_time < ?",
                (last, last + 86_400_000),
            )
        }
        assert sources == {"binance_vision_daily"}

    def test_start_mid_month_days_before_start_not_requested(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_sep30(archive)
        wanted = _oct_days(5, 16)
        served = {
            f"BTCUSDT-1m-{d.isoformat()}.zip": _daily_zip(d)
            for d in _oct_days(1, 16)
        }
        seen = _patch_http(monkeypatch, served)

        _run(archive, start=datetime(2026, 10, 5, tzinfo=UTC))

        assert _daily(seen) == _oct_urls(wanted)

    def test_day_already_holding_rows_is_not_requested(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_sep30(archive)
        archive.insert_mark_klines(
            "BTCUSDT", "1m", [_mark(_day_start_ms(date(2026, 10, 5)))], "binance_vision"
        )
        seen = _patch_http(
            monkeypatch,
            {f"BTCUSDT-1m-{d.isoformat()}.zip": _daily_zip(d) for d in _oct_days(1, 6)},
        )

        counts = _run(
            archive,
            end=datetime(2026, 10, 6, tzinfo=UTC),
            now=datetime(2026, 10, 7, tzinfo=UTC),
        )

        wanted = [date(2026, 10, d) for d in (1, 2, 3, 4, 6)]
        assert _daily(seen) == _oct_urls(wanted)
        assert counts["mark_klines_inserted"] == 5
        assert counts["missing_days"] == []


class TestM4TodayNeverRequested:
    def test_todays_utc_day_is_not_requested_even_when_in_range(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_sep30(archive)
        # end is 2026-10-31 and now is 2026-10-17: the clamp to yesterday, not
        # the range end, is what bounds the tail.
        seen = _patch_http(
            monkeypatch,
            {f"BTCUSDT-1m-{d.isoformat()}.zip": _daily_zip(d) for d in _oct_days(1, 17)},
        )

        counts = _run(archive, end=datetime(2026, 10, 31, tzinfo=UTC))

        assert _oct_urls(_oct_days(1, 16)) == _daily(seen)
        assert "BTCUSDT-1m-2026-10-17.zip" not in _daily(seen)
        assert counts["mark_klines_inserted"] == 16

    def test_now_shifts_the_tail_boundary(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_sep30(archive)
        seen = _patch_http(
            monkeypatch,
            {f"BTCUSDT-1m-{d.isoformat()}.zip": _daily_zip(d) for d in _oct_days(1, 5)},
        )

        counts = _run(archive, now=datetime(2026, 10, 5, tzinfo=UTC))

        assert _daily(seen) == _oct_urls(_oct_days(1, 4))
        assert counts["mark_klines_inserted"] == 4


class TestM2UnpublishedDayIsNotFatal:
    def test_404_days_recorded_oldest_first(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_sep30(archive)
        seen = _patch_http(monkeypatch, {})

        counts = _run(archive)  # must not raise

        assert _monthly(seen) == ["BTCUSDT-1m-2026-10.zip"]
        assert _daily(seen) == _oct_urls(_oct_days(1, 16))
        assert counts["unpublished_days"] == [d.isoformat() for d in _oct_days(1, 16)]
        assert counts["failed_days"] == []
        assert counts["failed_months"] == []
        assert counts["mark_klines_inserted"] == 0
        assert counts["missing_days"] == []


class TestM3FailedDayIsRecordedAndOthersContinue:
    def test_error_day_goes_to_failed_days_others_still_fetched(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _seed_sep30(archive)
        served = {
            f"BTCUSDT-1m-{d.isoformat()}.zip": (
                httpx.ConnectError("reset") if d == date(2026, 10, 3) else _daily_zip(d)
            )
            for d in _oct_days(1, 16)
        }
        seen = _patch_http(monkeypatch, served)

        with caplog.at_level(logging.ERROR, logger="vibe_quant.data.ingest"):
            counts = _run(archive)

        daily = _daily(seen)
        # every day is attempted by the tail pass; the failed one is retried by
        # the interior pass too, because the days after it extended coverage and
        # thereby made it an interior hole. Each pass retries it to exhaustion.
        assert sorted(set(daily)) == sorted(_oct_urls(_oct_days(1, 16)))
        attempts = len(downloader.MONTHLY_RETRY_DELAYS) + 1
        assert daily.count("BTCUSDT-1m-2026-10-03.zip") == 2 * attempts
        assert counts["failed_days"] == ["2026-10-03"]
        assert counts["unpublished_days"] == []
        assert counts["mark_klines_inserted"] == 15
        assert "daily 2026-10-03: FAILED" in caplog.text


class TestM5PublishedMonthHasNoDailyCalls:
    def test_monthly_rows_present_no_tail(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sep30 = _seed_sep30(archive)
        seen = _patch_http(monkeypatch, {"BTCUSDT-1m-2026-10.zip": _monthly_zip(_oct_days(1, 16))})

        counts = _run(archive)

        assert _monthly(seen) == ["BTCUSDT-1m-2026-10.zip"]
        assert _daily(seen) == []
        assert counts["unavailable_months"] == []
        assert counts["failed_days"] == []
        assert archive.get_mark_date_range("BTCUSDT", "1m") == (
            sep30,
            _day_start_ms(date(2026, 10, 16)),
        )


class TestM6RepublishedMonthlyInsertsNoDuplicates:
    def test_daily_then_monthly_same_minutes_insert_zero(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_sep30(archive)
        days = _oct_days(1, 5)
        _patch_http(
            monkeypatch, {f"BTCUSDT-1m-{d.isoformat()}.zip": _daily_zip(d) for d in days}
        )
        first = _run(
            archive,
            end=datetime(2026, 10, 5, tzinfo=UTC),
            now=datetime(2026, 10, 6, tzinfo=UTC),
        )
        assert first["mark_klines_inserted"] == 5

        # Later the monthly zip is published and downloaded: same minutes -> 0 new.
        _patch_http(monkeypatch, {"BTCUSDT-1m-2026-10.zip": _monthly_zip(days)})
        second = _run(
            archive,
            end=datetime(2026, 10, 5, tzinfo=UTC),
            now=datetime(2026, 10, 6, tzinfo=UTC),
        )

        assert second["mark_klines_inserted"] == 0
        assert second["failed_days"] == []
        assert archive.get_month_mark_kline_count("BTCUSDT", "1m", 2026, 10) == 5


class TestM7TailRunsBeforeInteriorPass:
    def test_interior_pass_sees_the_tail_filled(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_sep30(archive)
        _patch_http(
            monkeypatch,
            {f"BTCUSDT-1m-{d.isoformat()}.zip": _daily_zip(d) for d in _oct_days(1, 16)},
        )
        coverage_seen: list[Any] = []
        real_interior = ingest._fill_interior_missing_days

        def spy(*args: Any, **kwargs: Any) -> list[str]:
            coverage_seen.append(archive.get_mark_date_range("BTCUSDT", "1m"))
            return real_interior(*args, **kwargs)

        monkeypatch.setattr(ingest, "_fill_interior_missing_days", spy)

        counts = _run(archive)

        # When the interior pass ran, the tail was already archived.
        assert coverage_seen == [(_seed_sep30_open_ms(), _day_start_ms(date(2026, 10, 16)))]
        assert counts["missing_days"] == []


def _seed_sep30_open_ms() -> int:
    return _day_start_ms(SEP_30) + 23 * 3_600_000


class TestM8VerboseOutput:
    def test_day_lines_summary_and_new_coverage_end(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _seed_sep30(archive)
        served = {
            f"BTCUSDT-1m-{d.isoformat()}.zip": _daily_zip(d)
            for d in _oct_days(1, 16)
            if d != date(2026, 10, 3)
        }
        _patch_http(monkeypatch, served)

        _run(archive, verbose=True)
        out = capsys.readouterr().out

        assert "    2026-10-01: 1 rows from daily zip (1 new)" in out
        assert "    2026-10-03: daily zip unavailable (404)" in out
        assert "  Unpublished-month tail: 16 days requested, 15 filled, 1 unpublished, 0 failed" in out
        assert "mark coverage now ends 2026-10-16" in out
        # the original unavailable-month line is unchanged and still printed
        assert "2026-10: unavailable (not yet published) - mark coverage ends 2026-09-30" in out


class TestM10CliSummary:
    def _fix_now(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = ingest.ingest_mark_klines

        def fixed(
            symbol: str,
            start_date: datetime,
            end_date: datetime | None = None,
            archive: RawDataArchive | None = None,
            verbose: bool = True,
        ) -> dict[str, Any]:
            return real(
                symbol,
                start_date=start_date,
                end_date=end_date,
                archive=archive,
                verbose=verbose,
                now=NOW,
            )

        monkeypatch.setattr(ingest, "ingest_mark_klines", fixed)

    def test_summary_counts_unpublished_days(
        self,
        archive: RawDataArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _seed_sep30(archive)
        _patch_http(monkeypatch, {})
        monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: archive)
        self._fix_now(monkeypatch)

        rc = ingest.main(
            [
                "download-mark",
                "--symbols",
                "BTCUSDT",
                "--start",
                "2026-10-01",
                "--end",
                "2026-10-31",
            ]
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "16 unpublished days, 0 failed days" in out

    def test_failed_day_is_visible_in_summary(
        self,
        archive: RawDataArchive,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _seed_sep30(archive)
        served = {
            f"BTCUSDT-1m-{d.isoformat()}.zip": (
                httpx.ConnectError("reset") if d == date(2026, 10, 3) else _daily_zip(d)
            )
            for d in _oct_days(1, 16)
        }
        _patch_http(monkeypatch, served)
        monkeypatch.setattr(ingest, "RawDataArchive", lambda *a, **k: archive)
        self._fix_now(monkeypatch)

        rc = ingest.main(
            [
                "download-mark",
                "--symbols",
                "BTCUSDT",
                "--start",
                "2026-10-01",
                "--end",
                "2026-10-31",
            ]
        )

        assert rc == 0  # a failed day is a hole to re-run, not fatal
        out = capsys.readouterr().out
        assert "1 failed days" in out
        assert "FAILED days: 2026-10-03" in out


def _seed_full_days(archive: RawDataArchive, days: list[date]) -> None:
    archive.insert_mark_klines(
        "BTCUSDT",
        "1m",
        [
            _mark(_day_start_ms(day) + i * 60_000)
            for day in days
            for i in range(1440)
        ],
        "binance_vision_daily",
    )


class TestM12TailCoversAMonthTheSkipPassed:
    """A month can be *skipped* rather than *unavailable*.

    28 full days of daily rows (40320/44640 = 90.3%) clear
    ``has_month_mark_coverage``'s 0.9 bar, so ``download_monthly_mark_klines`` is
    never called and the month never lands in ``unavailable_months`` — a tail
    pass that walks only that list fetches nothing, while ``missing_days`` clamps
    to the coverage end (Oct 28) and so never reports the closed days after it.
    The tail pass must cover the zero-row days after the coverage end whatever
    the month's status (contract M12).
    """

    def test_skip_passed_month_still_gets_its_tail(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_full_days(archive, _oct_days(1, 28))
        # This is the skip the bug hides behind: the month looks complete enough.
        assert archive.has_month_mark_coverage("BTCUSDT", "1m", 2026, 10)
        served = {"BTCUSDT-1m-2026-10-29.zip": _daily_zip(date(2026, 10, 29), rows=1440)}
        seen = _patch_http(monkeypatch, served)

        counts = _run(archive, now=datetime(2026, 10, 30, tzinfo=UTC))

        assert _monthly(seen) == []  # month skipped: the monthly zip is never tried
        assert _daily(seen) == ["BTCUSDT-1m-2026-10-29.zip"]  # Oct 29 (closed) IS requested
        assert "BTCUSDT-1m-2026-10-30.zip" not in _daily(seen)  # today is not
        assert counts["unavailable_months"] == []
        assert counts["mark_klines_inserted"] == 1440
        assert counts["unpublished_days"] == []
        assert counts["failed_days"] == []
        assert counts["missing_days"] == []
        coverage = archive.get_mark_date_range("BTCUSDT", "1m")
        assert coverage is not None
        assert datetime.fromtimestamp(coverage[0] / 1000, tz=UTC).date() == date(2026, 10, 1)
        assert datetime.fromtimestamp(coverage[1] / 1000, tz=UTC).date() == date(2026, 10, 29)

    def test_second_run_after_the_tail_is_filled_requests_nothing(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_full_days(archive, _oct_days(1, 28))
        _patch_http(
            monkeypatch,
            {"BTCUSDT-1m-2026-10-29.zip": _daily_zip(date(2026, 10, 29), rows=1440)},
        )
        first = _run(archive, now=datetime(2026, 10, 30, tzinfo=UTC))
        assert first["mark_klines_inserted"] == 1440

        seen = _patch_http(monkeypatch, {})
        second = _run(archive, now=datetime(2026, 10, 30, tzinfo=UTC))

        assert _monthly(seen) == []
        assert _daily(seen) == []  # coverage now ends Oct 29; nothing left to fetch
        assert second["mark_klines_inserted"] == 0
        assert second["unpublished_days"] == []
        assert second["failed_days"] == []


class TestM11FailedTailDayIsTheFinalDbState:
    def test_failed_last_day_is_failed_not_missing(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same skip-passed month as M12: the only zero-row day is the old tail,
        # so a failure there has no later day to make it an interior hole.
        _seed_full_days(archive, _oct_days(1, 28))
        seen = _patch_http(
            monkeypatch,
            {"BTCUSDT-1m-2026-10-29.zip": httpx.ConnectError("reset")},
        )

        counts = _run(archive, now=datetime(2026, 10, 30, tzinfo=UTC))

        assert counts["failed_days"] == ["2026-10-29"]
        assert counts["unpublished_days"] == []
        # missing_days is the final DB state: coverage still ends Oct 28, so the
        # failed day is past the clamp and is not an interior hole (M11).
        assert counts["missing_days"] == []
        assert counts["mark_klines_inserted"] == 0
        # no later day extended coverage, so only the tail pass retried it
        assert _daily(seen).count("BTCUSDT-1m-2026-10-29.zip") == (
            len(downloader.MONTHLY_RETRY_DELAYS) + 1
        )
