"""Tests for batch validation orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
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


def test_run_validation_batch_creates_runs_and_closes_resources(monkeypatch: Any) -> None:
    """Batch runner should create one validation run per strategy and close resources."""
    calls: dict[str, object] = {"created_runs": [], "runner_closed": 0, "state_closed": 0}

    def fake_ensure_data_window(
        symbols: list[str],
        *,
        start_date: str,
        end_date: str,
        ingest_missing: bool,
        verbose: bool,
    ) -> None:
        assert symbols == ["BTCUSDT"]
        assert start_date == "2024-10-01"
        assert end_date == "2024-12-31"
        assert ingest_missing is False
        assert verbose is False

    class FakeStateManager:
        def __init__(self, db_path: Any) -> None:
            assert db_path is None

        def get_strategy(self, strategy_id: int) -> dict[str, object] | None:
            return {"id": strategy_id, "name": f"strategy_{strategy_id}"}

        def create_backtest_run(
            self,
            *,
            strategy_id: int | None,
            run_mode: str,
            symbols: list[str],
            timeframe: str,
            start_date: str,
            end_date: str,
            parameters: dict[str, object],
            latency_preset: str | None = None,
            sizing_config_id: int | None = None,
            risk_config_id: int | None = None,
        ) -> int:
            del sizing_config_id, risk_config_id
            created_runs = calls["created_runs"]
            assert isinstance(created_runs, list)
            created_runs.append(
                {
                    "strategy_id": strategy_id,
                    "run_mode": run_mode,
                    "symbols": symbols,
                    "timeframe": timeframe,
                    "start_date": start_date,
                    "end_date": end_date,
                    "parameters": parameters,
                    "latency_preset": latency_preset,
                }
            )
            return 900 + int(strategy_id or 0)

        def close(self) -> None:
            calls["state_closed"] = int(calls["state_closed"]) + 1

    class FakeValidationRunner:
        def __init__(self, db_path: Any) -> None:
            assert db_path is None

        def run(self, run_id: int, latency_preset: str | None = None) -> Any:
            strategy_id = run_id - 900
            assert latency_preset == "retail"
            return SimpleNamespace(
                strategy_name=f"strategy_{strategy_id}",
                total_return=0.10 + strategy_id / 1000,
                sharpe_ratio=1.0 + strategy_id / 1000,
                max_drawdown=0.05,
                total_trades=20 + strategy_id,
                win_rate=0.60,
                profit_factor=1.40,
            )

        def close(self) -> None:
            calls["runner_closed"] = int(calls["runner_closed"]) + 1

    monkeypatch.setattr(batch, "ensure_data_window", fake_ensure_data_window)
    monkeypatch.setattr(batch, "StateManager", FakeStateManager)
    monkeypatch.setattr(batch, "ValidationRunner", FakeValidationRunner)

    results = batch.run_validation_batch(
        [212, 220],
        symbol="BTCUSDT",
        timeframe="1m",
        start_date="2024-10-01",
        end_date="2024-12-31",
        latency_preset="retail",
        ensure_data=False,
        verbose=False,
    )

    assert [result.strategy_id for result in results] == [212, 220]
    assert [result.run_id for result in results] == [1112, 1120]
    assert calls["created_runs"] == [
        {
            "strategy_id": 212,
            "run_mode": "validation",
            "symbols": ["BTCUSDT"],
            "timeframe": "1m",
            "start_date": "2024-10-01",
            "end_date": "2024-12-31",
            "parameters": {},
            "latency_preset": "retail",
        },
        {
            "strategy_id": 220,
            "run_mode": "validation",
            "symbols": ["BTCUSDT"],
            "timeframe": "1m",
            "start_date": "2024-10-01",
            "end_date": "2024-12-31",
            "parameters": {},
            "latency_preset": "retail",
        },
    ]
    assert calls["runner_closed"] == 1
    assert calls["state_closed"] == 1


def test_format_batch_results_markdown_formats_fraction_metrics() -> None:
    """Markdown formatter should display returns and win rates as percentages."""
    markdown = batch.format_batch_results_markdown(
        [
            batch.ValidationBatchResult(
                strategy_id=212,
                strategy_name="champion",
                run_id=901,
                total_return=0.1234,
                sharpe_ratio=1.23,
                max_drawdown=0.0456,
                total_trades=55,
                win_rate=0.678,
                profit_factor=1.89,
            )
        ]
    )

    assert "12.34%" in markdown
    assert "4.56%" in markdown
    assert "67.8%" in markdown


def test_parse_strategy_ids_rejects_empty_input() -> None:
    """Parser should fail fast when no IDs are provided."""
    with pytest.raises(ValueError, match="At least one strategy ID is required"):
        batch.parse_strategy_ids(" , ")
