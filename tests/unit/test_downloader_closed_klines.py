"""``fetch_closed_klines``: room-bounded REST klines as raw decimal strings.

Contract rows R13/R14: the still-forming bar (``close_time >= end_ms``) is
dropped, prices/volumes survive as the strings Binance sent (paper builds
``Price.from_str``/``Quantity.from_str`` from them, so no float round trip), and
paging past the 1500-row REST limit is shared with ``download_recent_klines``.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from vibe_quant.data.downloader import (
    RestDownloadError,
    download_recent_klines,
    fetch_closed_klines,
)

_STEP_MS = 900_000  # 15m
_START = 1_700_000_000_000


def _rest_row(
    open_ms: int,
    open_: str = "100.10000000",
    high: str = "101.50000000",
    low: str = "99.90000000",
    close: str = "100.75000000",
    volume: str = "12.34000000",
) -> list[Any]:
    """One Binance kline row; prices/volumes are what the API sends (strings)."""
    return [
        open_ms,
        open_,
        high,
        low,
        close,
        volume,
        open_ms + _STEP_MS - 1,
        "1234.50000000",
        7,
        "6.17000000",
        "617.25000000",
        "0",
    ]


class _Server:
    """MockTransport handler that serves *rows* in ``limit``-sized pages."""

    def __init__(self, rows: list[list[Any]], fail_from_call: int | None = None) -> None:
        self.rows = rows
        self.fail_from_call = fail_from_call
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        self.requests.append(params)
        if self.fail_from_call is not None and len(self.requests) >= self.fail_from_call:
            return httpx.Response(500)
        start = int(params["startTime"])
        limit = int(params["limit"])
        return httpx.Response(200, json=[r for r in self.rows if r[0] >= start][:limit])


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)
    return delays


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _serve(rows: list[list[Any]], fail_from_call: int | None = None) -> _Server:
        server = _Server(rows, fail_from_call)
        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **k: real_client(transport=httpx.MockTransport(server)),
        )
        return server

    return _serve


def test_forming_bar_dropped(serve: Any) -> None:
    forming = _START + _STEP_MS
    serve([_rest_row(_START), _rest_row(forming, close="100.99000000")])
    # end_ms is exactly the forming bar's close_time: >= means still forming.
    out = fetch_closed_klines("BTCUSDT", "15m", _START, forming + _STEP_MS - 1)
    assert [r[0] for r in out] == [_START]


def test_returns_raw_strings(serve: Any) -> None:
    serve(
        [
            _rest_row(_START, "0.10000000", "0.20000000", "0.05000000", "0.15000000", "3.50000000"),
            _rest_row(
                _START + _STEP_MS,
                "0.15000000",
                "0.40000000",
                "0.01000000",
                "0.37000000",
                "8.12500000",
            ),
        ]
    )
    out = fetch_closed_klines("BTCUSDT", "15m", _START, _START + 2 * _STEP_MS)
    assert out == [
        (
            _START,
            "0.10000000",
            "0.20000000",
            "0.05000000",
            "0.15000000",
            "3.50000000",
            _START + _STEP_MS - 1,
        ),
        (
            _START + _STEP_MS,
            "0.15000000",
            "0.40000000",
            "0.01000000",
            "0.37000000",
            "8.12500000",
            _START + 2 * _STEP_MS - 1,
        ),
    ]
    assert isinstance(out[0][0], int)
    assert isinstance(out[0][6], int)


def test_pages_past_1500(serve: Any) -> None:
    rows = [_rest_row(_START + i * _STEP_MS) for i in range(3_200)]
    server = serve(rows)
    out = fetch_closed_klines("BTCUSDT", "15m", _START, _START + 3_200 * _STEP_MS)
    assert len(out) == 3_200
    assert [r[0] for r in out] == [r[0] for r in rows]
    assert len({r[0] for r in out}) == 3_200
    # Page 2 starts exactly one millisecond past page 1's last open_time.
    assert int(server.requests[1]["startTime"]) == rows[1_499][0] + 1
    assert len(server.requests) == 4  # 1500 + 1500 + 200, then the empty page


def test_page_failure_raises_rest_download_error(serve: Any, no_sleep: list[float]) -> None:
    rows = [_rest_row(_START + i * _STEP_MS) for i in range(3_200)]
    serve(rows, fail_from_call=2)
    with pytest.raises(RestDownloadError) as ei:
        fetch_closed_klines("BTCUSDT", "15m", _START, _START + 3_200 * _STEP_MS)
    assert len(ei.value.partial) == 1_500
    assert ei.value.partial[0][0] == _START
    assert ei.value.partial[-1][0] == rows[1_499][0]
    assert ei.value.failed_from == rows[1_499][0] + 1
    assert no_sleep == [1.0, 2.0, 4.0]


def test_empty_page_stops(serve: Any) -> None:
    server = serve([])
    assert fetch_closed_klines("BTCUSDT", "15m", _START, _START + _STEP_MS) == []
    assert len(server.requests) == 1


def test_empty_window_makes_no_request(serve: Any) -> None:
    server = serve([_rest_row(_START)])
    assert fetch_closed_klines("BTCUSDT", "15m", _START, _START) == []
    assert server.requests == []


def test_recent_klines_keep_float_tuple_shape(serve: Any) -> None:
    serve([_rest_row(_START)])
    assert download_recent_klines("BTCUSDT", "15m", _START, _START + _STEP_MS) == [
        (
            _START,
            100.1,
            101.5,
            99.9,
            100.75,
            12.34,
            _START + _STEP_MS - 1,
            1234.5,
            7,
            6.17,
            617.25,
        )
    ]


def test_failure_partial_drops_the_forming_bar(serve: Any, no_sleep: list[float]) -> None:
    """The partial a failed page carries is filtered like a success: no bar closing at or past end_ms."""
    rows = [_rest_row(_START + i * _STEP_MS) for i in range(10)]
    serve(rows, fail_from_call=2)
    end_ms = _START + 9 * _STEP_MS + _STEP_MS // 2  # bar 9 is still forming
    with pytest.raises(RestDownloadError) as ei:
        fetch_closed_klines("BTCUSDT", "15m", _START, end_ms)
    assert [r[0] for r in ei.value.partial] == [r[0] for r in rows[:9]]
    assert ei.value.failed_from == rows[9][0] + 1
