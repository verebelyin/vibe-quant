"""Tests for top-level vibe-quant CLI helpers."""

from __future__ import annotations

import argparse
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from vibe_quant import __main__ as root_cli


class _FakeJobManager:
    """Stub job manager for CLI tests."""

    def mark_completed(self, run_id: int, error: str | None = None) -> None:
        pass

    def close(self) -> None:
        pass


def _install_fake_validation_runner_module(
    monkeypatch: Any,
    runner_cls: type[Any],
    runner_error_cls: type[Exception],
    list_validation_runs_fn: Any | None = None,
) -> None:
    """Inject a fake validation.runner module for deterministic CLI tests."""
    module = ModuleType("vibe_quant.validation.runner")
    module.ValidationRunner = runner_cls
    module.ValidationRunnerError = runner_error_cls
    if list_validation_runs_fn is not None:
        module.list_validation_runs = list_validation_runs_fn
    monkeypatch.setitem(sys.modules, "vibe_quant.validation.runner", module)

    # Mock run_with_heartbeat to return no-op job manager and stop function
    jobs_module = ModuleType("vibe_quant.jobs.manager")
    jobs_module.run_with_heartbeat = lambda run_id, db_path: (_FakeJobManager(), lambda: None)
    monkeypatch.setitem(sys.modules, "vibe_quant.jobs.manager", jobs_module)


def _install_forward_target(
    monkeypatch: Any,
    module_name: str,
    main_fn: Any,
) -> None:
    """Install a fake forwarding target module exposing main(argv)."""
    module = ModuleType(module_name)
    module.main = main_fn
    monkeypatch.setitem(sys.modules, module_name, module)


def test_cmd_validation_run_closes_runner_on_success(monkeypatch: Any, capsys: Any) -> None:
    """Validation runner must always close after successful execution."""
    state = {"closed": 0}

    class FakeValidationRunner:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, run_id: int, latency_preset: str | None = None) -> Any:
            assert run_id == 123
            assert latency_preset is None
            return SimpleNamespace(
                strategy_name="test_strategy",
                total_return=0.12,
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                max_drawdown=0.08,
                total_trades=42,
                win_rate=0.55,
                profit_factor=1.8,
                total_fees=12.34,
                total_funding=1.23,
                total_slippage=4.56,
                execution_time_seconds=0.78,
            )

        def close(self) -> None:
            state["closed"] += 1

    class FakeValidationRunnerError(Exception):
        pass

    _install_fake_validation_runner_module(
        monkeypatch,
        FakeValidationRunner,
        FakeValidationRunnerError,
    )

    args = argparse.Namespace(run_id=123, latency=None)
    assert root_cli.cmd_validation_run(args) == 0
    assert state["closed"] == 1
    output = capsys.readouterr().out
    assert "Total Return: 12.00%" in output
    assert "Max Drawdown: 8.00%" in output
    assert "Win Rate: 55.0%" in output


def test_cmd_validation_run_closes_runner_on_validation_error(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """Validation runner must close even when run() raises ValidationRunnerError."""
    state = {"closed": 0}

    class FakeValidationRunnerError(Exception):
        pass

    class FakeValidationRunner:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, run_id: int, latency_preset: str | None = None) -> Any:
            assert run_id == 999
            assert latency_preset == "retail"
            raise FakeValidationRunnerError("boom")

        def close(self) -> None:
            state["closed"] += 1

    _install_fake_validation_runner_module(
        monkeypatch,
        FakeValidationRunner,
        FakeValidationRunnerError,
    )

    args = argparse.Namespace(run_id=999, latency="retail")
    assert root_cli.cmd_validation_run(args) == 1
    assert state["closed"] == 1
    assert "Error: boom" in capsys.readouterr().err


def test_cmd_validation_list_formats_fraction_as_percent(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """Validation list output should display fractional returns as percentages."""

    class FakeValidationRunnerError(Exception):
        pass

    class FakeValidationRunner:
        def run(self, run_id: int, latency_preset: str | None = None) -> Any:
            raise NotImplementedError

        def close(self) -> None:
            return None

    def fake_list_validation_runs(*, limit: int) -> list[dict[str, object]]:
        assert limit == 5
        return [
            {
                "run_id": 7,
                "strategy_id": 1,
                "status": "completed",
                "sharpe_ratio": 1.23,
                "total_return": 0.056,
                "total_trades": 12,
                "created_at": "2026-02-13 10:30:00",
            }
        ]

    _install_fake_validation_runner_module(
        monkeypatch,
        FakeValidationRunner,
        FakeValidationRunnerError,
        list_validation_runs_fn=fake_list_validation_runs,
    )

    args = argparse.Namespace(limit=5)
    assert root_cli.cmd_validation_list(args) == 0
    output = capsys.readouterr().out
    assert "5.6%" in output


def test_cmd_validation_batch_formats_markdown_results(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """Validation batch output should render a markdown summary table."""

    module = ModuleType("vibe_quant.validation.batch")

    class FakeBatchResult(SimpleNamespace):
        pass

    def fake_parse_strategy_ids(raw_value: str) -> list[int]:
        assert raw_value == "212,220"
        return [212, 220]

    def fake_run_validation_batch(  # noqa: PLR0913
        strategy_ids: list[int],
        *,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        db_path: Any,
        latency_preset: str | None,
        ensure_data: bool,
        verbose: bool,
    ) -> list[FakeBatchResult]:
        assert strategy_ids == [212, 220]
        assert symbol == "BTCUSDT"
        assert timeframe == "1m"
        assert start_date == "2024-10-01"
        assert end_date == "2024-12-31"
        assert latency_preset == "retail"
        assert ensure_data is True
        assert verbose is True
        return [
            FakeBatchResult(
                strategy_id=212,
                strategy_name="champion",
                run_id=900,
                total_return=0.123,
                sharpe_ratio=1.23,
                max_drawdown=0.045,
                total_trades=50,
                win_rate=0.55,
                profit_factor=1.7,
            )
        ]

    def fake_format_batch_results_markdown(results: list[FakeBatchResult]) -> str:
        assert len(results) == 1
        return "| Strategy ID | Return |\n| --- | --- |\n| 212 | 12.30% |"

    module.parse_strategy_ids = fake_parse_strategy_ids
    module.run_validation_batch = fake_run_validation_batch
    module.format_batch_results_markdown = fake_format_batch_results_markdown
    monkeypatch.setitem(sys.modules, "vibe_quant.validation.batch", module)

    args = argparse.Namespace(
        strategy_ids="212,220",
        symbol="BTCUSDT",
        timeframe="1m",
        start_date="2024-10-01",
        end_date="2024-12-31",
        latency="retail",
        ensure_data=True,
        db=None,
    )
    assert root_cli.cmd_validation_batch(args) == 0
    output = capsys.readouterr().out
    assert "| 212 | 12.30% |" in output


def test_cmd_data_forwards_argv_without_mutating_sys_argv(monkeypatch: Any) -> None:
    """Root data command should forward argv directly and keep sys.argv untouched."""
    captured: dict[str, list[str]] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = [] if argv is None else argv
        return 0

    _install_forward_target(monkeypatch, "vibe_quant.data.ingest", fake_main)

    original_argv = list(sys.argv)
    assert root_cli.cmd_data(argparse.Namespace(), extra=["ingest", "--years", "1"]) == 0
    assert captured["argv"] == ["ingest", "--years", "1"]
    assert sys.argv == original_argv


def test_cmd_screening_forwards_argv_without_mutating_sys_argv(monkeypatch: Any) -> None:
    """Root screening command should forward argv directly and keep sys.argv untouched."""
    captured: dict[str, list[str]] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = [] if argv is None else argv
        return 0

    _install_forward_target(monkeypatch, "vibe_quant.screening.__main__", fake_main)

    original_argv = list(sys.argv)
    assert root_cli.cmd_screening(argparse.Namespace(), extra=["list", "--limit", "5"]) == 0
    assert captured["argv"] == ["list", "--limit", "5"]
    assert sys.argv == original_argv


def test_cmd_overfitting_forwards_argv_without_mutating_sys_argv(monkeypatch: Any) -> None:
    """Root overfitting command should forward argv directly and keep sys.argv untouched."""
    captured: dict[str, list[str]] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        captured["argv"] = [] if argv is None else argv
        return 0

    _install_forward_target(monkeypatch, "vibe_quant.overfitting.__main__", fake_main)

    original_argv = list(sys.argv)
    assert root_cli.cmd_overfitting(argparse.Namespace(), extra=["report", "--run-id", "10"]) == 0
    assert captured["argv"] == ["report", "--run-id", "10"]
    assert sys.argv == original_argv
