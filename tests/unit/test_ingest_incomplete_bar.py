"""Guard: never archive the currently-forming candle.

Binance REST returns the in-progress candle regardless of ``endTime``. Because
the next incremental fetch starts at ``last_open_time + interval`` and never
re-pulls a bar it already stored, archiving that partial candle freezes its
wrong (mid-formation) OHLC permanently. ``_drop_unclosed_klines`` excludes any
kline whose ``close_time`` (tuple index 6) has not yet passed, and it must be
wired into every REST ingest path.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from vibe_quant.data import ingest
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.ingest import (
    _drop_unclosed_klines,
    ingest_detail_data,
    update_symbol,
)

if TYPE_CHECKING:
    from pathlib import Path

# Kline tuple layout (see downloader): index 0 = open_time, 6 = close_time.
_NOW = 1_700_000_000_000


def _kline(open_time: int, close_time: int, *, close: float = 1.0) -> tuple[object, ...]:
    return (open_time, 1.0, 2.0, 0.5, close, 10.0, close_time)


class TestDropUnclosedKlines:
    def test_drops_trailing_in_progress_candle(self) -> None:
        closed = _kline(_NOW - 120_000, _NOW - 60_001)
        forming = _kline(_NOW, _NOW + 59_999)  # close_time in the future
        assert _drop_unclosed_klines([closed, forming], _NOW) == [closed]

    def test_keeps_fully_closed_final_candle(self) -> None:
        """Guard isn't over-eager: a bar whose close_time already passed stays."""
        closed = _kline(_NOW - 60_000, _NOW - 1)  # closed exactly one ms ago
        assert _drop_unclosed_klines([closed], _NOW) == [closed]

    def test_close_time_equal_to_now_is_unclosed(self) -> None:
        """A candle is closed only once now has moved *past* its close_time."""
        boundary = _kline(_NOW - 60_000 + 1, _NOW)
        assert _drop_unclosed_klines([boundary], _NOW) == []

    def test_empty_input(self) -> None:
        assert _drop_unclosed_klines([], _NOW) == []


class _FakeInstrument:
    """Stand-in so behavioral tests don't need the NT runtime/catalog."""

    id = "BTCUSDT-PERP.BINANCE"
    size_precision = 8
    price_precision = 2


class _FakeCatalog:
    def write_instrument(self, *args: object, **kwargs: object) -> None: ...
    def clear_bar_data(self, *args: object, **kwargs: object) -> None: ...
    def write_bars(self, *args: object, **kwargs: object) -> None: ...


@pytest.fixture
def stub_catalog_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the post-insert catalog rebuild (NT + parquet) so the test
    isolates the archive-insert behavior."""
    monkeypatch.setattr(ingest, "create_instrument", lambda _symbol: _FakeInstrument())
    monkeypatch.setattr(ingest, "klines_to_bars", lambda *a, **k: [])
    monkeypatch.setattr(ingest, "aggregate_bars", lambda *a, **k: [])
    monkeypatch.setattr(ingest, "get_bar_type", lambda *a, **k: None)


def test_incremental_update_excludes_in_progress_candle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_catalog_layer: None
) -> None:
    """update_symbol (incremental REST path) must archive the closed bar the
    mock returns but never the trailing forming bar."""
    now_ms = int(time.time() * 1000)
    seed_open = now_ms - 600_000
    closed_open = now_ms - 180_000
    forming_open = now_ms + 30_000  # close_time far in the future -> unclosed

    archive = RawDataArchive(tmp_path / "incr.db")
    try:
        archive.insert_klines(
            "BTCUSDT", "1m", [_kline(seed_open, seed_open + 59_999)], "seed"
        )
        monkeypatch.setattr(
            ingest,
            "download_recent_klines",
            lambda *a, **k: [
                _kline(closed_open, closed_open + 59_999, close=2.0),
                _kline(forming_open, forming_open + 59_999, close=9.0),
            ],
        )

        update_symbol("BTCUSDT", archive=archive, catalog=_FakeCatalog(), verbose=False)

        open_times = {row["open_time"] for row in archive.get_klines("BTCUSDT", "1m")}
        assert closed_open in open_times  # closed bar archived
        assert forming_open not in open_times  # forming bar dropped
    finally:
        archive.close()


def test_detail_ingest_excludes_in_progress_candle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_catalog_layer: None
) -> None:
    """ingest_detail_data (sub-minute REST path) applies the same guard."""
    now_ms = int(time.time() * 1000)
    closed_open = now_ms - 10_000
    forming_open = now_ms  # 5s candle still forming

    archive = RawDataArchive(tmp_path / "detail.db")
    try:
        monkeypatch.setattr(
            ingest,
            "download_recent_klines",
            lambda *a, **k: [
                _kline(closed_open, closed_open + 4_999, close=2.0),
                _kline(forming_open, forming_open + 4_999, close=9.0),
            ],
        )

        ingest_detail_data(
            "BTCUSDT",
            interval="5s",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            archive=archive,
            catalog=_FakeCatalog(),
            verbose=False,
        )

        open_times = {row["open_time"] for row in archive.get_klines("BTCUSDT", "5s")}
        assert closed_open in open_times
        assert forming_open not in open_times
    finally:
        archive.close()
