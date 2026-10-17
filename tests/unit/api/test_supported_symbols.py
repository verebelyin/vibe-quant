"""GET /api/data/symbols/supported lists every ingestable symbol.

The Data page ingest form must offer a brand-new coin (one with no archived
data yet) so its first ingest can be started from the UI. The downloader's
``SUPPORTED_SYMBOLS`` is the source of truth for that choice; the
archive-scoped ``GET /api/data/symbols`` cannot supply it.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from vibe_quant.api.app import create_app
from vibe_quant.api.routers.data import list_supported_symbols
from vibe_quant.data.downloader import SUPPORTED_SYMBOLS

# A literal copy of downloader.SUPPORTED_SYMBOLS, deliberately not derived from
# it: the response must carry these exact strings in this exact order.
EXPECTED_SUPPORTED_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "SUIUSDT",
    "ZECUSDT",
    "NEARUSDT",
    "HYPEUSDT",
]


def test_supported_symbols_endpoint_returns_the_exact_ordered_list() -> None:
    client = TestClient(create_app())
    response = client.get("/api/data/symbols/supported")
    assert response.status_code == 200
    assert response.json() == EXPECTED_SUPPORTED_SYMBOLS


def test_supported_symbols_endpoint_matches_the_downloader_constant() -> None:
    client = TestClient(create_app())
    response = client.get("/api/data/symbols/supported")
    assert response.json() == list(SUPPORTED_SYMBOLS)


def test_supported_symbols_response_is_a_copy_not_the_constant() -> None:
    """Mutating a caller's list must not corrupt the downloader constant."""
    symbols = list_supported_symbols()
    symbols.append("FAKEUSDT")
    assert "FAKEUSDT" not in SUPPORTED_SYMBOLS
    assert list_supported_symbols() == EXPECTED_SUPPORTED_SYMBOLS
