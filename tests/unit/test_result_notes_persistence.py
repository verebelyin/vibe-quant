"""Persisted ``backtest_results.notes`` bytes and reader outcomes (bead vibe-quant-rabmo.8, T13).

The three writers outside discovery and :mod:`vibe_quant.db.result_notes` -- the
screening compiler-version stamp, the validation single-window dumps and the
walk-forward aggregate dump -- must store exactly the bytes they stored before
the codec move. ``persisted_notes_golden.json`` was captured ONCE on the unmoved
base (d6e00dbd) from the real writers; a mismatch means the refactor moved a
note. The reader tests pin the same base behaviour for the two notes readers
(``_reference_from_discovery_notes``, ``_mtm_columns_from_notes``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.test_dsr_cumulative_n import _series
from tests.unit.test_multi_seed_mtm_notes import (
    _champion_entry_notes,
    _pooled_multi_seed_result,
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.overfitting.pipeline import OverfittingPipeline
from vibe_quant.screening.__main__ import _store_compiler_version
from vibe_quant.validation.consistency import (
    ScreeningReference,
    _reference_from_discovery_notes,
)
from vibe_quant.validation.runner import TradeRecord, ValidationResult, ValidationRunner

GOLDEN = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "result_notes"
        / "persisted_notes_golden.json"
    ).read_text(encoding="utf-8")
)

DSL: dict[str, object] = {
    "name": "test_strategy",
    "description": "Test strategy for validation",
    "version": 1,
    "timeframe": "5m",
    "additional_timeframes": [],
    "indicators": {
        "rsi": {"type": "RSI", "period": 14, "source": "close"},
        "ema": {"type": "EMA", "period": 20, "source": "close"},
    },
    "entry_conditions": {"long": ["rsi < 30"], "short": ["rsi > 70"]},
    "exit_conditions": {"long": ["rsi > 50"], "short": ["rsi < 50"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}

#: Mixed key order, a float sum, a non-ASCII string and a nested list: every
#: json.dumps default the codec must keep (separators, key order, escapes).
LITERAL_NOTES = (
    '{"zeta": [1, {"nested": ["a", "b"]}], "alpha": 0.30000000000000004, '
    '"unicode": "caf\\u00e9 \\u2014 \\u4e2d\\u6587", "mid": true}'
)


def _mock_result(run_id: int, strategy_name: str) -> ValidationResult:
    """Deterministic result: fixed values only, so ``notes`` cannot carry wall-clock noise."""
    return ValidationResult(
        run_id=run_id,
        strategy_name=strategy_name,
        starting_balance=1000.0,
        total_return=0.052,
        sharpe_ratio=1.4,
        sortino_ratio=1.8,
        max_drawdown=0.08,
        total_trades=15,
        winning_trades=9,
        losing_trades=6,
        win_rate=0.6,
        profit_factor=1.5,
        total_fees=45.0,
        total_funding=12.0,
        total_slippage=8.0,
        trades=[
            TradeRecord(
                symbol="BTCUSDT-PERP",
                direction="LONG",
                leverage=10,
                entry_time="2025-01-10T08:00:00Z",
                exit_time="2025-01-10T14:00:00Z",
                entry_price=42000.0,
                exit_price=42500.0,
                quantity=0.1,
                entry_fee=2.1,
                exit_fee=2.125,
                net_pnl=45.775,
                gross_pnl=50.0,
                exit_reason="take_profit",
            )
        ],
    )


def _strategy_and_run(db_path: Path) -> int:
    state = StateManager(db_path)
    strategy_id = state.create_strategy(
        name="test_strategy", dsl_config=DSL, description="Test"
    )
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="validation",
        symbols=["BTCUSDT-PERP"],
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-31",
        parameters={"rsi_period": 14, "execution_delay_probability": 0.5},
        latency_preset="retail",
    )
    state.close()
    return run_id


def _stored_notes(db_path: Path, run_id: int) -> str:
    state = StateManager(db_path)
    stored = state.get_backtest_result(run_id)
    state.close()
    assert stored is not None
    return str(stored["notes"])


@pytest.fixture
def no_catalog_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fakes run no engine, so skip the catalog coverage clamp (as the runner tests do)."""
    monkeypatch.setattr(
        ValidationRunner, "_clamp_run_window", lambda self, run_config, tf: (run_config, None)
    )


class TestPersistedNotesBytes:
    def test_screening_compiler_version_keeps_the_stored_notes_bytes(
        self, tmp_path: Path
    ) -> None:
        """``_store_compiler_version`` rewrites the notes JSON byte for byte as before."""
        db_path = tmp_path / "screening.db"
        state = StateManager(db_path)
        strategy_id = state.create_strategy(name="test_strategy", dsl_config=DSL)
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT-PERP"],
            timeframe="5m",
            start_date="2025-01-01",
            end_date="2025-01-31",
            parameters={},
        )
        state.save_backtest_result(run_id, {"notes": LITERAL_NOTES})

        _store_compiler_version(state, run_id, "abc123def456")
        stored = state.get_backtest_result(run_id)
        state.close()

        assert stored is not None
        assert stored["notes"] == GOLDEN["screening_compiler_version"]

    def test_single_window_validation_keeps_the_persisted_notes_bytes(
        self, tmp_path: Path, no_catalog_clamp: None
    ) -> None:
        db_path = tmp_path / "single.db"
        run_id = _strategy_and_run(db_path)
        runner = ValidationRunner(db_path=db_path, logs_path=tmp_path / "logs_single")
        runner._run_backtest = lambda **kw: _mock_result(  # type: ignore[assignment]
            int(kw["run_id"]), str(kw["strategy_name"])
        )
        runner.run(run_id=run_id)
        runner.close()

        assert _stored_notes(db_path, run_id) == GOLDEN["validation_single_window"]

    def test_walk_forward_aggregate_keeps_the_persisted_notes_bytes(
        self, tmp_path: Path, no_catalog_clamp: None
    ) -> None:
        db_path = tmp_path / "walk.db"
        run_id = _strategy_and_run(db_path)
        runner = ValidationRunner(db_path=db_path, logs_path=tmp_path / "logs_walk")
        windows: list[int] = []

        def _fake_backtest(**kw: object) -> ValidationResult:
            windows.append(1)
            result = _mock_result(int(kw["run_id"]), str(kw["strategy_name"]))
            result.total_return = 0.01 * len(windows)
            result.sharpe_ratio = 1.0 + len(windows)
            return result

        runner._run_backtest = _fake_backtest  # type: ignore[assignment]
        runner.run_walk_forward(run_id=run_id, train_days=7, test_days=7, step_days=7)
        runner.close()

        assert len(windows) == 3
        assert _stored_notes(db_path, run_id) == GOLDEN["validation_walk_forward"]


DISCOVERY_NOTES = (
    '{"top_strategies": [{"dsl": {"name": "genome_abc123"}, '
    '"sharpe": 1.8, "trades": 64}], "sizing": {"risk_pct": 1.0}}'
)


def _discovery_row(state: StateManager, notes: str | None) -> int:
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2025-01-01",
        parameters={},
    )
    state.update_backtest_run_status(run_id, "completed")
    state.save_backtest_result(run_id, {"sharpe_ratio": 0.0, "notes": notes})
    return run_id


def _reference_with_bad_newest_row(tmp_path: Path, bad_notes: str | None) -> ScreeningReference | None:
    """A DB whose newest discovery row is ``bad_notes`` and whose older row is a match."""
    state = StateManager(tmp_path / "consistency.db")
    state.create_strategy("genome_abc123", {"name": "genome_abc123"})
    _discovery_row(state, DISCOVERY_NOTES)
    _discovery_row(state, bad_notes)
    try:
        return _reference_from_discovery_notes(
            state,
            "genome_abc123",
            strategy_dsl={"name": "genome_abc123"},
        )
    finally:
        state.close()


class TestDiscoveryNotesReader:
    def test_matches_the_champion_in_literal_notes(self, tmp_path: Path) -> None:
        state = StateManager(tmp_path / "good.db")
        state.create_strategy("genome_abc123", {"name": "genome_abc123"})
        run_id = _discovery_row(state, DISCOVERY_NOTES)
        reference = _reference_from_discovery_notes(
            state, "genome_abc123", strategy_dsl={"name": "genome_abc123"}
        )
        state.close()

        assert reference == ScreeningReference(
            sharpe=1.8,
            trades=64,
            source=f"discovery_run:{run_id}",
            start_date="2024-01-01",
            end_date="2025-01-01",
        )

    def test_skips_malformed_notes(self, tmp_path: Path) -> None:
        reference = _reference_with_bad_newest_row(tmp_path, "{not json")

        assert reference == ScreeningReference(
            sharpe=1.8,
            trades=64,
            source="discovery_run:1",
            start_date="2024-01-01",
            end_date="2025-01-01",
        )

    def test_skips_a_null_notes_cell(self, tmp_path: Path) -> None:
        reference = _reference_with_bad_newest_row(tmp_path, None)

        assert reference == ScreeningReference(
            sharpe=1.8,
            trades=64,
            source="discovery_run:1",
            start_date="2024-01-01",
            end_date="2025-01-01",
        )

    def test_non_object_notes_raise_attribute_error(self, tmp_path: Path) -> None:
        """Preserved base behaviour: a JSON list parses, then ``payload.get`` raises."""
        with pytest.raises(AttributeError, match="'list' object has no attribute 'get'"):
            _reference_with_bad_newest_row(tmp_path, "[1]")


MTM_NONE: dict[str, float | int | None] = {
    "mtm_sharpe_daily": None,
    "mtm_skewness": None,
    "mtm_kurtosis": None,
    "mtm_days": None,
}


def _mtm_notes(record: dict[str, object]) -> str:
    return json.dumps({"top_strategies": [{"dsr": record}]})


class TestMtmNotesReader:
    @pytest.mark.parametrize(
        ("record", "expected"),
        [
            (
                {
                    "input": "pooled_mtm_daily_returns",
                    "sharpe_daily": 0.123,
                    "skewness": -0.5,
                    "kurtosis": 4.25,
                    "observations": 730,
                },
                {
                    "mtm_sharpe_daily": 0.123,
                    "mtm_skewness": -0.5,
                    "mtm_kurtosis": 4.25,
                    "mtm_days": 730,
                },
            ),
            (
                {
                    "input": "pooled_mtm_daily_returns",
                    "sharpe_daily": -1.75,
                    "skewness": 2.0,
                    "kurtosis": 9.5,
                    "observations": 42,
                },
                {
                    "mtm_sharpe_daily": -1.75,
                    "mtm_skewness": 2.0,
                    "mtm_kurtosis": 9.5,
                    "mtm_days": 42,
                },
            ),
        ],
    )
    def test_reads_the_four_columns_of_a_literal_record(
        self,
        tmp_path: Path,
        record: dict[str, object],
        expected: dict[str, float | int | None],
    ) -> None:
        pipeline = OverfittingPipeline(tmp_path / "mtm.db")

        assert pipeline._mtm_columns_from_notes(_mtm_notes(record)) == expected

    @pytest.mark.parametrize("notes", ["{broken", "", None])
    def test_fails_closed_on_unreadable_notes(self, tmp_path: Path, notes: str | None) -> None:
        pipeline = OverfittingPipeline(tmp_path / "mtm.db")

        assert pipeline._mtm_columns_from_notes(notes) == MTM_NONE

    def test_reads_the_multi_seed_producers_record(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The real producer's pooled note keeps the record the reader consumes."""
        result = _pooled_multi_seed_result(monkeypatch, _series(5.0))
        notes_json = json.dumps(_champion_entry_notes(result))
        pipeline = OverfittingPipeline(tmp_path / "mtm.db")

        columns = pipeline._mtm_columns_from_notes(notes_json)

        assert all(value is not None for value in columns.values())
