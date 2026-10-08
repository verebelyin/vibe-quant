"""Tests for funding-rate parsing in vibe_quant.data.downloader."""

from __future__ import annotations

import logging

import pytest

from vibe_quant.data import downloader

# Binance returns "" (empty string) for markPrice/fundingRate on older 2022 records.
FUNDING_PAGE: list[dict[str, object]] = [
    {
        "symbol": "BTCUSDT",
        "fundingTime": 1640995200000,
        "fundingRate": "0.0001",
        "markPrice": "",
    },
    {
        "symbol": "BTCUSDT",
        "fundingTime": 1641024000000,
        "fundingRate": "",
        "markPrice": "47000.1",
    },
    {
        "symbol": "BTCUSDT",
        "fundingTime": 1641052800000,
        "fundingRate": "-0.0002",
        "markPrice": "46900",
    },
]

START_MS = 1640995200000
# Well past the last record's fundingTime: only a short page (< limit) can stop pagination.
END_MS = 1641081600000


def test_download_funding_rates_skips_blank_funding_rate(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    requests: list[dict[str, object]] = []

    class FakeResponse:
        def raise_for_status(self) -> None: ...

        def json(self) -> list[dict[str, object]]:
            return FUNDING_PAGE

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *exc_info: object) -> None: ...

        def get(self, url: str, params: dict[str, object] | None = None) -> FakeResponse:
            requests.append(dict(params or {}))
            return FakeResponse()

    monkeypatch.setattr(downloader.httpx, "Client", FakeClient)

    with caplog.at_level(logging.WARNING, logger="vibe_quant.data.downloader"):
        rates = downloader.download_funding_rates("BTCUSDT", START_MS, END_MS)

    # The mock returns the same 3-record page (< limit=1000) on every call,
    # so a single request proves pagination stops on a short page.
    assert len(requests) == 1

    assert len(rates) == 2
    assert rates[0][0] == 1640995200000
    assert rates[0][1] == 0.0001
    assert rates[0][2] == 0.0
    assert rates[1][0] == 1641052800000
    assert rates[1][1] == -0.0002
    assert rates[1][2] == 46900.0

    assert "skipped 1 funding records with empty fundingRate" in caplog.text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.0001", 0.0001),
        ("-0.0002", -0.0002),
        ("46900", 46900.0),
        (42, 42.0),
        (None, 0.0),
        ("", 0.0),
        ("   ", 0.0),
    ],
)
def test_to_float_parses_numbers_and_defaults_blanks(value: object, expected: float) -> None:
    assert downloader._to_float(value) == expected


def test_to_float_custom_default() -> None:
    assert downloader._to_float("", default=-1.0) == -1.0
