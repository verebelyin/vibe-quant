"""update_symbol refreshes funding; 1d bars are built from 1m (vibe-quant-c8va4)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vibe_quant.data import ingest
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.data.catalog import (
    CatalogManager,
    create_instrument,
    get_bar_type,
    klines_to_bars,
)
from vibe_quant.data.ingest import update_symbol

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_DAY = 86_400_000


def _kline(open_time: int, price: float = 100.0) -> tuple[object, ...]:
    return (open_time, price, price + 1, price - 1, price + 0.5, 10.0, open_time + 59_999)


class _FakeInstrument:
    id = "BTCUSDT-PERP.BINANCE"
    size_precision = 8
    price_precision = 2


class _FakeCatalog:
    def write_instrument(self, *a: object, **k: object) -> None: ...
    def clear_bar_data(self, *a: object, **k: object) -> None: ...
    def write_bars(self, *a: object, **k: object) -> None: ...


def test_update_fetches_funding_since_last_archived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now_ms = int(time.time() * 1000)
    last_funding = now_ms - 3 * 8 * 3_600_000
    new_funding = now_ms - 8 * 3_600_000
    archive = RawDataArchive(tmp_path / "a.db")
    try:
        # klines already current -> klines branch returns early; funding must still run
        archive.insert_klines("BTCUSDT", "1m", [_kline(now_ms - 30_000)], "seed")
        archive.insert_funding_rates("BTCUSDT", [(last_funding, 0.0001, 50000.0)], "seed")
        calls: list[tuple[str, int, int]] = []

        def fake_funding(symbol: str, start: int, end: int, *a: object, **k: object):  # type: ignore[no-untyped-def]
            calls.append((symbol, start, end))
            return [(new_funding, 0.0002, 51000.0)]

        monkeypatch.setattr(ingest, "download_funding_rates", fake_funding)
        monkeypatch.setattr(ingest, "download_recent_klines", lambda *a, **k: [])

        counts = update_symbol("BTCUSDT", archive=archive, catalog=_FakeCatalog(), verbose=False)  # type: ignore[arg-type]

        assert len(calls) == 1
        assert calls[0][0] == "BTCUSDT"
        assert calls[0][1] == last_funding + 1
        assert counts["new_funding_rates"] == 1
        times = [r["funding_time"] for r in archive.get_funding_rates("BTCUSDT")]
        assert times == [last_funding, new_funding]
    finally:
        archive.close()


def test_1d_in_aggregation_path(tmp_path: Path) -> None:
    """Two UTC days of 1m klines -> 2 daily bars, OHLCV right, ts_event=open, ts_init=close."""
    assert "1d" in ingest.AGGREGATION_INTERVALS
    day0 = 1_704_067_200_000  # 2024-01-01T00:00Z
    klines = []
    for i in range(2 * 1440):
        t = day0 + i * 60_000
        klines.append(
            {
                "open_time": t, "open": 100 + i, "high": 200 + i, "low": 50 + i,
                "close": 101 + i, "volume": 1.0, "close_time": t + 59_999,
            }
        )
    inst = create_instrument("BTCUSDT")
    bars_1m = klines_to_bars(
        klines, inst.id, get_bar_type("BTCUSDT", "1m"), inst.size_precision, inst.price_precision
    )
    from vibe_quant.data.catalog import aggregate_bars

    daily = aggregate_bars(
        bars_1m, get_bar_type("BTCUSDT", "1d"), ingest._interval_to_minutes("1d"),
        inst.size_precision, inst.price_precision,
    )
    assert len(daily) == 2
    assert daily[0].ts_event == day0 * 1_000_000
    assert daily[1].ts_event == (day0 + _DAY) * 1_000_000
    assert float(daily[0].open) == 100.0
    assert float(daily[0].close) == 101 + 1439
    assert float(daily[0].high) == 200 + 1439
    assert float(daily[0].low) == 50.0
    assert float(daily[0].volume) == 1440.0
    assert daily[0].ts_init == (day0 + _DAY - 1) * 1_000_000  # close of last 1m bar
    # round-trips through a tmp catalog
    cm = CatalogManager(tmp_path / "cat")
    cm.write_instrument(inst)
    cm.write_bars(daily)
    assert cm.get_bar_count("BTCUSDT", "1d") == 2


def test_funding_network_error_does_not_abort_klines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    now_ms = int(time.time() * 1000)
    seed = now_ms - 600_000
    new_open = now_ms - 180_000
    archive = RawDataArchive(tmp_path / "b.db")
    try:
        archive.insert_klines("BTCUSDT", "1m", [_kline(seed)], "seed")
        archive.insert_funding_rates("BTCUSDT", [(now_ms - 10**9, 0.0001, 1.0)], "seed")

        def boom(*a: object, **k: object) -> list[object]:
            raise httpx.ConnectError("down")

        monkeypatch.setattr(ingest, "download_funding_rates", boom)
        monkeypatch.setattr(ingest, "download_recent_klines", lambda *a, **k: [_kline(new_open)])
        monkeypatch.setattr(ingest, "create_instrument", lambda _s: _FakeInstrument())
        monkeypatch.setattr(ingest, "klines_to_bars", lambda *a, **k: [])
        monkeypatch.setattr(ingest, "aggregate_bars", lambda *a, **k: [])
        monkeypatch.setattr(ingest, "get_bar_type", lambda *a, **k: None)

        counts = update_symbol("BTCUSDT", archive=archive, catalog=_FakeCatalog(), verbose=False)  # type: ignore[arg-type]
        assert counts["new_klines"] == 1
        assert counts["new_funding_rates"] == 0
    finally:
        archive.close()
