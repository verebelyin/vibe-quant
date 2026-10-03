"""Validation data window + detail resolution (bd vibe-quant-e70tl.13).

A run window ending past the last 1m bar used to drop the 1m detail data
(INFO log) while keeping latency -> fills at the next 4h close. 5m strategies
asked for 5s detail (never present) and sub-bar runs deferred only 30% of
fills (70% filled at the signal bar's own close).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.results import ValidationResult
from vibe_quant.validation.runner import ValidationRunner, ValidationRunnerError

if TYPE_CHECKING:
    from pathlib import Path

DSL = {
    "name": "window_probe",
    "timeframe": "4h",
    "indicators": {"rsi": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi < 30"]},
    "exit_conditions": {"long": ["rsi > 70"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 3.0},
}

# 1m data: first bar 2024-01-01 00:00, last bar 2026-03-17 00:00 (opens)
COVERAGE = {
    ("BTCUSDT", "1m"): (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2026, 3, 17, 0, 0, tzinfo=UTC),
    ),
    ("ETHUSDT", "1m"): (
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 2, 28, 16, 0, tzinfo=UTC),
    ),
}


@pytest.fixture(autouse=True)
def _fake_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    def date_range(
        self: CatalogManager, symbol: str, interval: str
    ) -> tuple[datetime, datetime] | None:
        return COVERAGE.get((symbol, interval))

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", date_range)


@pytest.fixture
def runner(tmp_path: Path) -> ValidationRunner:
    r = ValidationRunner(db_path=tmp_path / "s.db", logs_path=tmp_path / "logs")
    yield r
    r.close()


def _cfg(symbols: list[str], start: str, end: str) -> dict[str, object]:
    return {"symbols": json.dumps(symbols), "start_date": start, "end_date": end}


class TestClampRunWindow:
    def test_end_past_last_1m_bar_is_clamped_and_recorded(
        self, runner: ValidationRunner, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = _cfg(["BTCUSDT"], "2025-01-01", "2026-10-02")
        with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.runner"):
            clamped, note = runner._clamp_run_window(cfg, "1m")
        assert clamped["end_date"] == "2026-03-17"
        assert clamped["start_date"] == "2025-01-01"
        assert note is not None
        assert note["requested_end"] == "2026-10-02"
        assert note["effective_end"] == "2026-03-17"
        assert "clamped" in caplog.text

    def test_multi_symbol_uses_shortest_coverage(self, runner: ValidationRunner) -> None:
        clamped, note = runner._clamp_run_window(
            _cfg(["BTCUSDT", "ETHUSDT"], "2025-01-01", "2026-03-17"), "1m"
        )
        # ETH last bar opens 2026-02-28 16:00 -> last fully covered day ends 02-28
        assert clamped["end_date"] == "2026-02-28"
        assert note is not None

    def test_start_before_data_is_clamped_forward(self, runner: ValidationRunner) -> None:
        clamped, note = runner._clamp_run_window(_cfg(["BTCUSDT"], "2023-06-01", "2024-06-01"), "1m")
        assert clamped["start_date"] == "2024-01-01"
        assert note is not None

    def test_fully_covered_window_unchanged(self, runner: ValidationRunner) -> None:
        cfg = _cfg(["BTCUSDT"], "2024-01-01", "2026-03-17")
        clamped, note = runner._clamp_run_window(cfg, "1m")
        assert clamped is cfg
        assert note is None

    def test_symbol_without_1m_data_fails_loudly(self, runner: ValidationRunner) -> None:
        with pytest.raises(ValidationRunnerError, match="No 1m bars for SOLUSDT"):
            runner._clamp_run_window(_cfg(["BTCUSDT", "SOLUSDT"], "2025-01-01", "2025-02-01"), "1m")

    def test_no_overlap_fails(self, runner: ValidationRunner) -> None:
        with pytest.raises(ValidationRunnerError, match="does not overlap"):
            runner._clamp_run_window(_cfg(["BTCUSDT"], "2026-06-01", "2026-07-01"), "1m")


class TestDetailResolution:
    @pytest.mark.parametrize("tf", ["5m", "15m", "1h", "4h"])
    def test_coarser_than_1m_uses_1m_detail(self, runner: ValidationRunner, tf: str) -> None:
        assert runner._resolve_detail_timeframe(
            _cfg(["BTCUSDT"], "2025-01-01", "2025-02-01"), tf, None
        ) == "1m"

    def test_1m_strategy_without_5s_data_has_no_detail(self, runner: ValidationRunner) -> None:
        assert runner._resolve_detail_timeframe(
            _cfg(["BTCUSDT"], "2025-01-01", "2025-02-01"), "1m", None
        ) is None

    def test_no_detail_sub_bar_defers_every_fill(self, runner: ValidationRunner) -> None:
        params = runner._augment_strategy_params_for_validation(
            {}, timeframe="1m", has_detail_data=False
        )
        assert params["execution_delay_probability"] == 1.0


def _validation_run(state: StateManager, symbols: list[str], end: str) -> int:
    sid = state.create_strategy(name="window_probe", dsl_config=DSL)
    return state.create_backtest_run(
        strategy_id=sid, run_mode="validation", symbols=symbols, timeframe="4h",
        start_date="2025-01-01", end_date=end, parameters={},
    )


def test_run_uses_clamped_window_and_persists_note(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    state = StateManager(db)
    run_id = _validation_run(state, ["BTCUSDT"], "2026-10-02")
    state.close()

    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    seen: dict[str, object] = {}

    def fake_backtest(**kwargs: object) -> ValidationResult:
        seen.update(kwargs)
        return ValidationResult(run_id=run_id, strategy_name="window_probe", total_trades=3)

    runner._run_backtest = fake_backtest  # type: ignore[assignment,method-assign]
    result = runner.run(run_id=run_id)
    runner.close()

    run_config = seen["run_config"]
    assert isinstance(run_config, dict)
    assert run_config["end_date"] == "2026-03-17"
    assert seen["detail_timeframe"] == "1m"
    assert result.notes["data_window"]["effective_end"] == "2026-03-17"  # type: ignore[index]

    state = StateManager(db)
    stored = state.get_backtest_result(run_id)
    state.close()
    assert stored is not None
    notes = json.loads(str(stored["notes"]))
    assert notes["data_window"]["requested_end"] == "2026-10-02"


def test_run_without_1m_data_marks_failed(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    state = StateManager(db)
    run_id = _validation_run(state, ["SOLUSDT"], "2025-02-01")
    state.close()

    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    with pytest.raises(ValidationRunnerError, match="No 1m bars for SOLUSDT"):
        runner.run(run_id=run_id)
    runner.close()

    state = StateManager(db)
    run = state.get_backtest_run(run_id)
    state.close()
    assert run is not None
    assert run["status"] == "failed"
    assert "No 1m bars" in str(run["error_message"])
