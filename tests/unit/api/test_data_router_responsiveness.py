"""Responsiveness and response-equality tests for the data router.

Two contracts are pinned here:

1. Blocking sync work in a data endpoint must not stall the API event loop:
   while a slow ``GET /api/data/coverage`` is in flight, an unrelated cheap
   request (``/health``) must still complete promptly.
2. ``GET /api/data/coverage`` output is unchanged after the batched archive
   read: same JSON for seeded archives, including the empty archive and the
   "symbol with no catalog bars" edges.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from pathlib import Path

import vibe_quant.api.routers.data as data_router
from vibe_quant.api.app import create_app
from vibe_quant.data.archive import RawDataArchive

_MS_2024_01_01 = 1_704_067_200_000
_MS_2024_01_02 = 1_704_153_600_000


class _FakeCatalog:
    """CatalogManager stand-in returning fixed parquet bar counts."""

    def __init__(self, bar_counts: dict[str, int] | None = None) -> None:
        self._bar_counts = bar_counts or {}

    def get_bar_count(self, symbol: str, interval: str) -> int:
        return self._bar_counts.get(symbol, 0)


class _GatedArchive:
    """Archive stand-in whose coverage read blocks until ``gate`` is set.

    ``entered`` is set once the read is in flight. ``released_by_gate`` records
    whether the read returned because the gate was set (True) or because its
    10 s safety timeout fired (False). No wall-clock bound is involved.
    """

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate
        self.entered = threading.Event()
        self.released_by_gate: bool | None = None

    def get_coverage_rows(self, interval: str = "1m") -> list[tuple[object, ...]]:
        self.entered.set()
        self.released_by_gate = self._gate.wait(timeout=10)
        return []

    def close(self) -> None:
        pass


def _client(app) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


async def test_slow_coverage_does_not_block_the_event_loop(monkeypatch) -> None:
    """A blocking ``/api/data/coverage`` must run off the event loop.

    Ordering handshake instead of a timing bound: coverage is held open by the
    gate until this test releases it. ``/health`` must complete while coverage
    is still pending. If coverage blocked the loop, ``/health`` could not run
    until the gate's 10 s timeout, so the pending check and the release check fail.
    """
    gate = threading.Event()
    archive = _GatedArchive(gate)
    monkeypatch.setattr(data_router, "_get_archive", lambda: archive)

    app = create_app()
    app.state.catalog_manager = _FakeCatalog()

    async with _client(app) as client:
        coverage_task = asyncio.create_task(client.get("/api/data/coverage"))
        await asyncio.wait_for(asyncio.to_thread(archive.entered.wait, 10), 15)
        health = await asyncio.wait_for(client.get("/health"), 15)
        coverage_pending = not coverage_task.done()
        gate.set()
        coverage = await asyncio.wait_for(coverage_task, 15)

    assert health.status_code == 200
    assert coverage_pending, "/api/data/coverage finished before /health (event loop blocked)"
    assert archive.released_by_gate is True, (
        "coverage read was released by timeout, not by the test"
    )
    assert coverage.status_code == 200
    assert coverage.json() == {"coverage": []}


def _seed_archive(db_path: Path) -> None:
    archive = RawDataArchive(db_path)
    archive.insert_klines(
        "AAAUSDT",
        "1m",
        [
            (_MS_2024_01_01, 1.0, 1.0, 1.0, 1.0, 1.0, _MS_2024_01_01 + 59_999),
            (_MS_2024_01_02, 2.0, 2.0, 2.0, 2.0, 2.0, _MS_2024_01_02 + 59_999),
        ],
        "binance_vision",
    )
    archive.insert_klines(
        "BBBUSDT",
        "1m",
        [(_MS_2024_01_01, 3.0, 3.0, 3.0, 3.0, 3.0, _MS_2024_01_01 + 59_999)],
        "binance_vision",
    )
    # Only 4h bars: present in raw_klines but has no 1m coverage.
    archive.insert_klines(
        "DDDUSDT",
        "4h",
        [(_MS_2024_01_01, 4.0, 4.0, 4.0, 4.0, 4.0, _MS_2024_01_01 + 14_399_999)],
        "binance_vision",
    )
    archive.insert_funding_rates("AAAUSDT", [(_MS_2024_01_01, 0.0001, 100.0)], "binance_vision")
    # Funding-only symbol: not in raw_klines, so it must not surface.
    archive.insert_funding_rates("CCCUSDT", [(_MS_2024_01_01, 0.0002, 50.0)], "binance_vision")
    archive.close()


async def test_coverage_response_is_unchanged_after_batched_read(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive(db_path)
    monkeypatch.setattr("vibe_quant.data.archive.DEFAULT_ARCHIVE_PATH", db_path)

    app = create_app()
    app.state.catalog_manager = _FakeCatalog({"AAAUSDT": 7})

    async with _client(app) as client:
        resp = await client.get("/api/data/coverage")

    assert resp.status_code == 200
    assert resp.json() == {
        "coverage": [
            {
                "symbol": "AAAUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
                "kline_count": 2,
                "bar_count": 7,
                "funding_rate_count": 1,
                "tier": "full",
            },
            {
                "symbol": "BBBUSDT",
                "start_date": "2024-01-01",
                "end_date": "2024-01-01",
                "kline_count": 1,
                "bar_count": 0,
                "funding_rate_count": 0,
                "tier": "full",
            },
            {
                "symbol": "DDDUSDT",
                "start_date": "",
                "end_date": "",
                "kline_count": 0,
                "bar_count": 0,
                "funding_rate_count": 0,
                "tier": "research",
            },
        ]
    }


async def test_coverage_empty_archive(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "empty_archive.db"
    RawDataArchive(db_path).close()
    monkeypatch.setattr("vibe_quant.data.archive.DEFAULT_ARCHIVE_PATH", db_path)

    app = create_app()
    app.state.catalog_manager = _FakeCatalog()

    async with _client(app) as client:
        resp = await client.get("/api/data/coverage")

    assert resp.status_code == 200
    assert resp.json() == {"coverage": []}
