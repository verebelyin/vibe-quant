"""Pytest fixtures for vibe-quant tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.fixtures.known_results import KnownResult, load_known_results

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from vibe_quant.data.archive import RawDataArchive
    from vibe_quant.db import StateManager


# Re-export for convenience
__all__ = ["KnownResult", "load_known_results"]


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Create temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def tmp_archive(tmp_path: Path) -> Generator[RawDataArchive]:
    """Create temporary RawDataArchive for testing."""
    from vibe_quant.data.archive import RawDataArchive

    archive_path = tmp_path / "archive.db"
    archive = RawDataArchive(archive_path)
    yield archive
    archive.close()


@pytest.fixture
def tmp_state_manager(tmp_path: Path) -> Generator[StateManager]:
    """Create temporary StateManager for testing."""
    from vibe_quant.db import StateManager

    db_path = tmp_path / "state.db"
    manager = StateManager(db_path)
    yield manager
    manager.close()


@pytest.fixture
def ohlc_aggregation_cases() -> list[KnownResult]:
    """Load OHLC aggregation known results."""
    return load_known_results("ohlc_aggregation")


@pytest.fixture
def sample_klines() -> list[tuple[int, float, float, float, float, float, int]]:
    """Sample kline data for testing.

    Returns list of (open_time, open, high, low, close, volume, close_time).
    """
    base_time = 1704067200000  # 2024-01-01 00:00:00 UTC
    return [
        (base_time, 42000.0, 42100.0, 41900.0, 42050.0, 100.0, base_time + 59999),
        (base_time + 60000, 42050.0, 42200.0, 42000.0, 42150.0, 120.0, base_time + 119999),
        (base_time + 120000, 42150.0, 42300.0, 42100.0, 42250.0, 80.0, base_time + 179999),
        (base_time + 180000, 42250.0, 42350.0, 42200.0, 42300.0, 150.0, base_time + 239999),
        (base_time + 240000, 42300.0, 42400.0, 42250.0, 42350.0, 90.0, base_time + 299999),
    ]


@pytest.fixture
def sample_dsl_config() -> dict[str, object]:
    """Sample DSL configuration for testing."""
    return {
        "name": "test_strategy",
        "description": "Test strategy",
        "version": 1,
        "timeframe": "5m",
        "indicators": {
            "rsi": {"type": "RSI", "period": 14, "source": "close"},
        },
        "entry_conditions": {
            "long": ["rsi < 30"],
            "short": ["rsi > 70"],
        },
        "exit_conditions": {
            "long": ["rsi > 50"],
            "short": ["rsi < 50"],
        },
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }
