"""Symbol tiers: a research-tier coin (1h/1d, no 1m bars) never reaches a launch picker.

``full`` = the symbol has 1m klines in the archive (screening, validation and
discovery run on them); ``research`` = archived bars but no 1m. ``GET
/api/data/symbols`` keeps listing every archived symbol unless ``?tier=`` asks
for one tier; ``GET /api/data/coverage`` labels each row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from vibe_quant.api.app import create_app
from vibe_quant.data.archive import RawDataArchive

if TYPE_CHECKING:
    from pathlib import Path

_MS = 1_704_067_200_000  # 2024-01-01


class _FakeCatalog:
    def get_bar_count(self, symbol: str, interval: str) -> int:
        return 0


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    db = tmp_path / "archive.db"
    archive = RawDataArchive(db)
    archive.insert_klines("BTCUSDT", "1m", [(_MS, 1.0, 1.0, 1.0, 1.0, 1.0, _MS + 59_999)], "test")
    archive.insert_klines(
        "BTCUSDT", "1h", [(_MS, 1.0, 1.0, 1.0, 1.0, 1.0, _MS + 3_599_999)], "test"
    )
    archive.insert_klines(
        "DOGEUSDT", "1h", [(_MS, 1.0, 1.0, 1.0, 1.0, 1.0, _MS + 3_599_999)], "test"
    )
    archive.insert_klines(
        "DOGEUSDT", "1d", [(_MS, 1.0, 1.0, 1.0, 1.0, 1.0, _MS + 86_399_999)], "test"
    )
    archive.close()
    monkeypatch.setattr("vibe_quant.data.archive.DEFAULT_ARCHIVE_PATH", db)
    app = create_app()
    app.state.catalog_manager = _FakeCatalog()
    return TestClient(app)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("", ["BTCUSDT", "DOGEUSDT"]),
        ("?tier=full", ["BTCUSDT"]),
        ("?tier=research", ["DOGEUSDT"]),
    ],
)
def test_symbols_filter_by_tier(client: TestClient, query: str, expected: list[str]) -> None:
    r = client.get(f"/api/data/symbols{query}")
    assert r.status_code == 200
    assert r.json() == expected


def test_symbols_reject_an_unknown_tier(client: TestClient) -> None:
    assert client.get("/api/data/symbols?tier=1h").status_code == 422


def test_coverage_labels_each_symbol_with_its_tier(client: TestClient) -> None:
    r = client.get("/api/data/coverage")
    assert r.status_code == 200
    assert {c["symbol"]: c["tier"] for c in r.json()["coverage"]} == {
        "BTCUSDT": "full",
        "DOGEUSDT": "research",
    }
