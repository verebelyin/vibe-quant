"""Deribit DVOL download: in-progress filter, resumable failures, append-only archive.

All HTTP is stubbed with ``httpx.MockTransport`` (a fake
``public/get_volatility_index_data`` that pages backward); every archive is a
tmp file.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import httpx
import pytest

from vibe_quant.data import downloader, dvol
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.downloader import DownloadError
from vibe_quant.data.dvol import DvolDownloadError, ingest_dvol

if TYPE_CHECKING:
    from pathlib import Path

S0 = 1704067200000  # 2024-01-01 00:00 UTC
HOUR = 3_600_000
LATE_US = (S0 + 48 * HOUR) * 1000  # usOut long after every fixture candle closed


def _h(hours: int) -> int:
    return S0 + hours * HOUR


class FakeDeribit:
    """Backward-paging DVOL endpoint; ``fail`` maps a request index to 503, 429 or a JSON-RPC error.

    Like the real one it floors ``start_timestamp`` to the candle grid and
    serves the in-progress candle; ``usOut`` is ``us_out``.
    """

    def __init__(self, candles: dict[int, float], us_out: int = LATE_US, page: int = 3) -> None:
        self.candles = candles  # open_ms -> close
        self.us_out = us_out
        self.page = page
        self.fail: dict[int, str] = {}
        self.requests: list[dict[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v2/public/get_volatility_index_data"
        params = dict(request.url.params)
        index = len(self.requests)
        self.requests.append(params)
        failure = self.fail.get(index)
        if failure == "503":
            return httpx.Response(503, text="Service Unavailable")
        if failure == "429":
            return httpx.Response(429, text="Too Many Requests")
        if failure == "rpc":
            error = {"code": -32602, "message": "Invalid params"}
            return httpx.Response(400, json={"jsonrpc": "2.0", "error": error})
        start = int(params["start_timestamp"]) // HOUR * HOUR
        end = int(params["end_timestamp"])
        opens = sorted(t for t in self.candles if start <= t <= end)
        served = opens[-self.page :]
        rest = opens[: -self.page]
        data = [[t, 1.0, 2.0, 0.5, self.candles[t]] for t in served]
        result = {"data": data, "continuation": served[0] - 1 if rest else None}
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "result": result, "usIn": 0, "usOut": self.us_out}
        )

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def _archived(archive: RawDataArchive) -> list[tuple[object, ...]]:
    return [
        tuple(row)
        for row in archive.conn.execute(
            "SELECT currency, resolution_s, open_time, open, high, low, close, source"
            " FROM raw_dvol ORDER BY currency, open_time"
        )
    ]


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloader, "MONTHLY_RETRY_DELAYS", (0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("currency", "us_out", "stored"),
    [
        ("BTC", (_h(12) + 30_000) * 1000, [_h(10)]),  # 11:00 candle ends 12:00, +60 s not reached
        ("ETH", (_h(12) + 59_999) * 1000 + 999, [_h(10)]),  # 12:00:59.999999: still in progress
        ("ETH", (_h(12) + 60_000) * 1000, [_h(10), _h(11)]),  # 12:01:00 == end + 60 s: stored
    ],
)
def test_in_progress_candle_dropped(
    tmp_path: Path, currency: str, us_out: int, stored: list[int]
) -> None:
    fake = FakeDeribit({_h(10): 50.0, _h(11): 52.0}, us_out=us_out)
    archive = RawDataArchive(tmp_path / "archive.db")
    with fake.client() as client:
        counts = ingest_dvol(currency, archive, start=date(2024, 1, 1), client=client, sleep=0.0)

    assert [r[2] for r in _archived(archive)] == stored
    assert {r[0] for r in _archived(archive)} == {currency}
    assert {r[1] for r in _archived(archive)} == {3600}
    assert fake.requests[0]["currency"] == currency
    assert fake.requests[0]["resolution"] == "3600"
    assert fake.requests[0]["start_timestamp"] == str(S0)
    assert counts == {"inserted": len(stored), "first": _h(10), "last": stored[-1]}


def test_failure_is_resumable(tmp_path: Path) -> None:
    candles = {_h(k): 50.0 + k for k in range(6)}
    fake = FakeDeribit(candles)
    fake.fail = {1: "503", 2: "429", 3: "503", 4: "503"}  # page 2: every retry attempt
    archive = RawDataArchive(tmp_path / "archive.db")
    with fake.client() as client, pytest.raises(DvolDownloadError) as excinfo:
        ingest_dvol("BTC", archive, start=date(2024, 1, 1), client=client, sleep=0.0)

    message = str(excinfo.value)
    assert isinstance(excinfo.value, DownloadError)
    assert message.startswith("Deribit DVOL BTC failed: ")
    assert "503" in message
    assert message.endswith(
        "; archived through 2024-01-01T05:00:00+00:00; re-run download-dvol to resume"
    )
    assert len(fake.requests) == 5
    assert fake.requests[1]["end_timestamp"] == str(_h(3) - 1)
    assert [r[2] for r in _archived(archive)] == [_h(3), _h(4), _h(5)]  # page 1 kept

    fake.fail = {}
    fake.requests = []
    with fake.client() as client:
        counts = ingest_dvol("BTC", archive, start=date(2024, 1, 1), client=client, sleep=0.0)

    range_2 = [r for r in fake.requests if r["start_timestamp"] == str(S0)]
    assert fake.requests[0]["start_timestamp"] == str(_h(5) + 1)  # range 1: (B, now]
    assert range_2[0]["end_timestamp"] == str(_h(3) - 1)  # range 2: [start, A)
    assert counts == {"inserted": 3, "first": _h(0), "last": _h(5)}
    assert [r[6] for r in _archived(archive)] == [50.0, 51.0, 52.0, 53.0, 54.0, 55.0]


@pytest.mark.parametrize(
    "fail",
    [{1: "503", 2: "503", 3: "503", 4: "503"}, {1: "rpc"}],
    ids=["503-retries-exhausted", "rpc-error"],
)
def test_incremental_failure_hole_filled_by_rerun(tmp_path: Path, fail: dict[int, str]) -> None:
    archive = RawDataArchive(tmp_path / "archive.db")
    fake = FakeDeribit({_h(k): 50.0 + k for k in range(3)})
    with fake.client() as client:
        ingest_dvol("BTC", archive, start=date(2024, 1, 1), client=client, sleep=0.0)
    fake.candles = {_h(k): 50.0 + k for k in range(9)}  # 03:00..08:00 appeared since
    fake.requests = []
    fake.fail = fail  # range 1 page 1 (06..08) archived, page 2 (03..05) fails
    with fake.client() as client, pytest.raises(DvolDownloadError) as excinfo:
        ingest_dvol("BTC", archive, start=date(2024, 1, 1), client=client, sleep=0.0)

    assert [_archived(archive)[i][2] for i in range(6)] == [_h(k) for k in (0, 1, 2, 6, 7, 8)]
    assert str(excinfo.value).endswith(
        "; archived through 2024-01-01T02:00:00+00:00; re-run download-dvol to resume"
    )  # names the contiguous coverage end, not the newest candle

    fake.fail = {}
    fake.requests = []
    with fake.client() as client:
        counts = ingest_dvol("BTC", archive, start=date(2024, 1, 1), client=client, sleep=0.0)

    assert fake.requests[0]["start_timestamp"] == str(_h(2) + 1)  # resumes at the hole
    assert counts == {"inserted": 3, "first": _h(0), "last": _h(8)}
    assert [r[6] for r in _archived(archive)] == [50.0 + k for k in range(9)]


@pytest.mark.parametrize(
    ("reply", "field"),
    [
        ({"result": {"data": [], "continuation": None}}, "usOut"),
        ({"usOut": LATE_US}, "result"),
        ({"result": {"continuation": None}, "usOut": LATE_US}, "data"),
        ({"result": {"data": [[S0, 1.0]], "continuation": None}, "usOut": LATE_US}, "data"),
    ],
    ids=["no-usOut", "no-result", "no-data", "short-candle"],
)
def test_malformed_reply_names_field(tmp_path: Path, reply: dict[str, object], field: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"jsonrpc": "2.0", **reply})

    archive = RawDataArchive(tmp_path / "archive.db")
    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(DvolDownloadError, match=f"reply field {field!r}"),
    ):
        ingest_dvol("BTC", archive, start=date(2024, 1, 1), client=client, sleep=0.0)

    assert len(requests) == 1  # a malformed reply is not retried
    assert _archived(archive) == []


def test_rpc_error_on_empty_archive_reports_nothing(tmp_path: Path) -> None:
    fake = FakeDeribit({_h(0): 50.0})
    fake.fail = {0: "rpc"}
    archive = RawDataArchive(tmp_path / "archive.db")
    with fake.client() as client, pytest.raises(DvolDownloadError) as excinfo:
        ingest_dvol("ETH", archive, start=date(2024, 1, 1), client=client, sleep=0.0)

    message = str(excinfo.value)
    assert message.startswith("Deribit DVOL ETH failed: ")
    assert "Invalid params" in message
    assert message.endswith("; archived through nothing; re-run download-dvol to resume")
    assert len(fake.requests) == 1  # a JSON-RPC error is not retried
    assert _archived(archive) == []


def test_reserved_candle_keeps_archived_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeDeribit({_h(0): 50.0, _h(1): 52.0})
    archive = RawDataArchive(tmp_path / "archive.db")
    with fake.client() as client:
        ingest_dvol("BTC", archive, start=date(2024, 1, 1), client=client, sleep=0.0)

    fake.candles = {_h(0): 50.0, _h(1): 99.0}  # 01:00 re-served with a different close
    sleeps: list[float] = []
    monkeypatch.setattr(dvol.time, "sleep", sleeps.append)
    with fake.client() as client:
        counts = ingest_dvol("BTC", archive, start=date(2024, 1, 1), client=client)

    assert fake.requests[-1]["start_timestamp"] == str(_h(1) + 1)  # floored to 01:00: re-served
    assert counts == {"inserted": 0, "first": _h(0), "last": _h(1)}
    assert _archived(archive)[1] == ("BTC", 3600, _h(1), 1.0, 2.0, 0.5, 52.0, "deribit")
    assert sleeps == [0.1]  # one page, default pacing (<= 10 req/s)
