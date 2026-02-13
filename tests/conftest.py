"""Pytest fixtures for vibe-quant tests."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.fixtures.known_results import KnownResult, load_known_results

if TYPE_CHECKING:
    from collections.abc import Generator

    from vibe_quant.data.archive import RawDataArchive
    from vibe_quant.db import StateManager


_NT_EAGER_IMPORT_TEST_MODULES = frozenset({
    "test_data_catalog.py",
    "test_ethereal_instruments.py",
    "test_ethereal_venue.py",
    "test_validation_venue.py",
})
_NT_RUNTIME_TEST_PREFIXES = ("test_validation_",)


@lru_cache(maxsize=1)
def _nt_runtime_preflight() -> tuple[bool, str]:
    """Check that NautilusTrader can be imported and basic C-extension types instantiate."""
    try:
        from nautilus_trader.model.objects import Price, Quantity

        Price.from_str("1.0")
        Quantity.from_str("1")
    except Exception as exc:  # pragma: no cover - environment-dependent
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _requires_nt_runtime(path: Path) -> bool:
    """Return True for tests that should be skipped when NT runtime is incompatible."""
    filename = path.name
    return (
        filename in _NT_EAGER_IMPORT_TEST_MODULES
        or filename.startswith(_NT_RUNTIME_TEST_PREFIXES)
    )


def _nt_skip_reason() -> str:
    _, reason = _nt_runtime_preflight()
    return f"Skipping NT-dependent tests: NautilusTrader runtime preflight failed ({reason})"


def pytest_report_header(config: pytest.Config) -> str:  # pragma: no cover - pytest hook
    """Print NT compatibility status once per test session."""
    compatible, reason = _nt_runtime_preflight()
    if compatible:
        return "NT runtime preflight: OK"
    return f"NT runtime preflight: FAILED ({reason})"


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """Avoid importing modules with top-level NT imports when runtime is incompatible."""
    compatible, _ = _nt_runtime_preflight()
    if compatible or collection_path.suffix != ".py":
        return False
    return collection_path.name in _NT_EAGER_IMPORT_TEST_MODULES


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark NT-dependent tests as skipped when preflight fails."""
    compatible, _ = _nt_runtime_preflight()
    if compatible:
        return

    marker = pytest.mark.skip(reason=_nt_skip_reason())
    for item in items:
        item_path = Path(str(getattr(item, "path", item.fspath)))
        if _requires_nt_runtime(item_path):
            item.add_marker(marker)


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
