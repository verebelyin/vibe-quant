"""Tests for batch validation orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from vibe_quant.validation import batch


def test_parse_strategy_ids_deduplicates_and_preserves_order() -> None:
    """Parser should keep the first occurrence of each strategy ID."""
    assert batch.parse_strategy_ids("212, 220,212") == [212, 220]


def test_ensure_data_window_ingests_missing_coverage(monkeypatch: Any) -> None:
    """Missing archive coverage should trigger ingest then re-check."""
    responses = [["BTCUSDT"], []]
    ingest_calls: list[dict[str, object]] = []

    def fake_validate_archive_window(
        symbols: list[str],
        *,
        start_date: str,
        end_date: str,
    ) -> list[str]:
        assert symbols == ["BTCUSDT"]
        assert start_date == "2024-10-01"
        assert end_date == "2024-12-31"
        return responses.pop(0)

    def fake_ingest_all(
        *,
        symbols: list[str],
        years: int = 2,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        verbose: bool = True,
    ) -> dict[str, dict[str, object]]:
        del years
        ingest_calls.append(
            {
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
                "verbose": verbose,
            }
        )
        return {}

    monkeypatch.setattr(batch, "validate_archive_window", fake_validate_archive_window)
    monkeypatch.setattr(batch, "ingest_all", fake_ingest_all)

    batch.ensure_data_window(
        ["BTCUSDT"],
        start_date="2024-10-01",
        end_date="2024-12-31",
        ingest_missing=True,
        verbose=False,
    )

    assert ingest_calls == [
        {
            "symbols": ["BTCUSDT"],
            "start_date": datetime(2024, 10, 1, tzinfo=UTC),
            "end_date": datetime(2024, 12, 31, tzinfo=UTC),
            "verbose": False,
        }
    ]


def test_parse_strategy_ids_rejects_empty_input() -> None:
    """Parser should fail fast when no IDs are provided."""
    with pytest.raises(ValueError, match="At least one strategy ID is required"):
        batch.parse_strategy_ids(" , ")
