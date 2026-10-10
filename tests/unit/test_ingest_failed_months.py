"""Ingest must not silently leave holes (vibe-quant-yul7u.19).

404 = month not published (unavailable, skipped); transient errors retry with
bounded backoff; persistent failures are recorded and make the CLI exit 1.
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from vibe_quant.data import downloader, ingest
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.downloader import MonthlyDownloadError, download_monthly_klines

if TYPE_CHECKING:
    from pathlib import Path


def _zip_bytes(filename: str = "BTCUSDT-1m-2024-01.csv") -> bytes:
    row = "1704067200000,1,2,0.5,1.5,10,1704067259999,15,3,5,7,0\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, row)
    return buf.getvalue()


def _client(responses: list[httpx.Response | Exception]) -> tuple[httpx.Client, list[int]]:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        item = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)
    return delays


class TestMonthlyRetry:
    def test_500_then_success_is_retried(self, no_sleep: list[float]) -> None:
        client, calls = _client([httpx.Response(500), httpx.Response(200, content=_zip_bytes())])
        klines = download_monthly_klines("BTCUSDT", "1m", 2024, 1, client=client)
        assert klines is not None
        assert len(klines) == 1
        assert len(calls) == 2
        assert no_sleep == [1.0]

    def test_connection_reset_then_success(self) -> None:
        client, calls = _client(
            [httpx.ConnectError("reset"), httpx.Response(200, content=_zip_bytes())]
        )
        assert download_monthly_klines("BTCUSDT", "1m", 2024, 1, client=client)
        assert len(calls) == 2

    def test_persistent_500_raises_after_bounded_attempts(self, no_sleep: list[float]) -> None:
        client, calls = _client([httpx.Response(503)])
        with pytest.raises(MonthlyDownloadError):
            download_monthly_klines("BTCUSDT", "1m", 2024, 1, client=client)
        assert len(calls) == len(downloader.MONTHLY_RETRY_DELAYS) + 1
        assert no_sleep == list(downloader.MONTHLY_RETRY_DELAYS)

    def test_429_then_success(self) -> None:
        client, calls = _client([httpx.Response(429), httpx.Response(200, content=_zip_bytes())])
        assert download_monthly_klines("BTCUSDT", "1m", 2024, 1, client=client)
        assert len(calls) == 2

    def test_truncated_zip_then_success(self) -> None:
        good = _zip_bytes()
        client, calls = _client(
            [httpx.Response(200, content=good[:20]), httpx.Response(200, content=good)]
        )
        assert download_monthly_klines("BTCUSDT", "1m", 2024, 1, client=client)
        assert len(calls) == 2

    def test_corrupt_deflate_then_success(self) -> None:
        good = _zip_bytes()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "BTCUSDT-1m-2024-01.csv",
                "1704067200000,1,2,0.5,1.5,10,1704067259999,15,3,5,7,0\n" * 50,
            )
        deflated = bytearray(buf.getvalue())
        off = 30 + len("BTCUSDT-1m-2024-01.csv")  # first byte of the deflate stream
        deflated[off : off + 10] = b"\xff" * 10  # invalid deflate block -> zlib.error
        client, calls = _client(
            [httpx.Response(200, content=bytes(deflated)), httpx.Response(200, content=good)]
        )
        assert download_monthly_klines("BTCUSDT", "1m", 2024, 1, client=client)
        assert len(calls) == 2

    def test_404_is_none_without_retry(self, no_sleep: list[float]) -> None:
        client, calls = _client([httpx.Response(404)])
        assert download_monthly_klines("BTCUSDT", "1m", 2024, 1, client=client) is None
        assert len(calls) == 1
        assert no_sleep == []

    def test_403_not_retried_but_raises(self) -> None:
        client, calls = _client([httpx.Response(403)])
        with pytest.raises(MonthlyDownloadError):
            download_monthly_klines("BTCUSDT", "1m", 2024, 1, client=client)
        assert len(calls) == 1


def _rest_row(open_ms: int) -> list[Any]:
    return [open_ms, "1", "2", "0.5", "1.5", "10", open_ms + 59_999, "15", 3, "5", "7", "0"]


class TestRestRetry:
    def test_mid_paging_429_then_success_completes(
        self, no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page1 = [_rest_row(1_000 + i * 60_000) for i in range(1500)]
        page2 = [_rest_row(1_000 + (1500 + i) * 60_000) for i in range(3)]
        seq: list[httpx.Response] = [
            httpx.Response(200, json=page1),
            httpx.Response(429),
            httpx.Response(200, json=page2),
            httpx.Response(200, json=[]),
        ]
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return seq[len(calls) - 1]

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **k: real_client(transport=httpx.MockTransport(handler)),
        )
        out = downloader.download_recent_klines("BTCUSDT", "1m", 0, 10**13)
        assert len(out) == 1503
        assert len(calls) == 4
        assert no_sleep == [1.0]

    def test_persistent_500_raises_with_partial(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page1 = [_rest_row(1_000 + i * 60_000) for i in range(1500)]
        n = {"c": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            n["c"] += 1
            return httpx.Response(200, json=page1) if n["c"] == 1 else httpx.Response(500)

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **k: real_client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(downloader.RestDownloadError) as ei:
            downloader.download_recent_klines("BTCUSDT", "1m", 0, 10**13)
        assert len(ei.value.partial) == 1500
        assert ei.value.failed_from == 1_000 + 1499 * 60_000 + 1


class _FakeInstrument:
    id = "BTCUSDT-PERP.BINANCE"
    size_precision = 8
    price_precision = 2


class _FakeCatalog:
    def write_instrument(self, *a: object, **k: object) -> None: ...
    def clear_bar_data(self, *a: object, **k: object) -> None: ...
    def write_bars(self, *a: object, **k: object) -> None: ...


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RawDataArchive:
    monkeypatch.setattr(ingest, "create_instrument", lambda _s: _FakeInstrument())
    monkeypatch.setattr(ingest, "klines_to_bars", lambda *a, **k: [])
    monkeypatch.setattr(ingest, "aggregate_bars", lambda *a, **k: [])
    monkeypatch.setattr(ingest, "get_bar_type", lambda *a, **k: None)
    return RawDataArchive(tmp_path / "raw.db")


def _kline(open_ms: int) -> tuple[Any, ...]:
    return (open_ms, 1.0, 2.0, 0.5, 1.5, 10.0, open_ms + 59_999, 15.0, 3, 5.0, 7.0)


def _run(archive: RawDataArchive, **kw: Any) -> dict[str, Any]:
    return ingest.ingest_symbol(
        "BTCUSDT",
        archive=archive,
        catalog=_FakeCatalog(),  # type: ignore[arg-type]
        verbose=False,
        **kw,
    )


class TestIngestSymbolMonths:
    def test_failed_and_unavailable_months_recorded(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake(symbol: str, interval: str, year: int, month: int, **k: Any) -> Any:
            if month == 1:
                raise MonthlyDownloadError("boom")
            if month == 2:
                return None
            return [_kline(1_709_251_200_000)]

        monkeypatch.setattr(ingest, "download_monthly_klines", fake)
        monkeypatch.setattr(ingest, "download_recent_klines", lambda *a, **k: [])
        counts = _run(
            archive,
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 4, 15, tzinfo=UTC),
        )
        assert counts["failed_months"] == ["2024-01"]
        assert counts["unavailable_months"] == ["2024-02"]
        assert counts["klines_fetched"] == 1  # March only; others skipped

    def test_range_inside_one_month_fetches_via_rest(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[int, int]] = []

        def fake_recent(symbol: str, interval: str, start: int, end: int, **k: Any) -> list[Any]:
            calls.append((start, end))
            return [_kline(start)]

        monkeypatch.setattr(ingest, "download_recent_klines", fake_recent)
        monkeypatch.setattr(
            ingest,
            "download_monthly_klines",
            lambda *a, **k: pytest.fail("no monthly zip applies"),
        )
        start = datetime(2024, 3, 5, tzinfo=UTC)
        end = datetime(2024, 3, 20, tzinfo=UTC)
        counts = _run(archive, start_date=start, end_date=end)
        assert calls == [(int(start.timestamp() * 1000), int(end.timestamp() * 1000))]
        assert counts["klines_inserted"] == 1


class TestRestFailureRecorded:
    def test_rest_tail_failure_recorded_and_partial_archived(
        self, archive: RawDataArchive, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(symbol: str, interval: str, start: int, end: int, **k: Any) -> list[Any]:
            raise downloader.RestDownloadError("500", [_kline(start)], start + 60_000)

        monkeypatch.setattr(ingest, "download_recent_klines", boom)
        counts = _run(
            archive,
            start_date=datetime(2024, 3, 5, tzinfo=UTC),
            end_date=datetime(2024, 3, 20, tzinfo=UTC),
        )
        assert len(counts["failed_ranges"]) == 1
        assert counts["failed_ranges"][0].startswith("2024-03-05T00:01:00")
        assert counts["klines_inserted"] == 1


class TestCliExit:
    def test_exit_1_on_failed_rest_range(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            ingest,
            "ingest_all",
            lambda **k: {"BTCUSDT": {"failed_months": [], "failed_ranges": ["a..b"]}},
        )
        assert ingest.main(["ingest", "--symbols", "BTCUSDT"]) == 1
        assert "a..b" in capsys.readouterr().err

    def _patch(self, monkeypatch: pytest.MonkeyPatch, failed: list[str]) -> None:
        monkeypatch.setattr(
            ingest,
            "ingest_all",
            lambda **k: {"BTCUSDT": {"failed_months": failed, "unavailable_months": []}},
        )

    def test_exit_1_and_lists_failed_months(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._patch(monkeypatch, ["2024-01"])
        assert ingest.main(["ingest", "--symbols", "BTCUSDT"]) == 1
        assert "BTCUSDT: 2024-01" in capsys.readouterr().err

    def test_exit_0_when_clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, [])
        assert ingest.main(["ingest", "--symbols", "BTCUSDT"]) == 0
