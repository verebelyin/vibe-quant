"""Collapse-check reference row selection (bd vibe-quant-e70tl.15, consistency part).

``find_screening_reference`` took an arbitrary ``LIMIT 1`` sweep row, so
strategies 70/71/72 were compared against (sharpe=-inf, trades=0) and the
collapse flag could never fire.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.consistency import assess_consistency, find_screening_reference

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def state(tmp_path: Path) -> StateManager:
    s = StateManager(db_path=tmp_path / "t.db")
    yield s
    s.close()


def _screening_run(state: StateManager, rows: list[tuple[dict[str, object], float, int]]) -> tuple[int, int]:
    sid = state.create_strategy("s1", {"name": "s1"})
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="screening", symbols=["BTCUSDT"], timeframe="4h",
        start_date="2024-01-01", end_date="2025-01-01", parameters={},
    )
    state.update_backtest_run_status(run_id, "completed")
    for params, sharpe, trades in rows:
        state.conn.execute(
            "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades)"
            " VALUES (?, ?, ?, ?)",
            (run_id, json.dumps(params), sharpe, trades),
        )
    state.conn.commit()
    return sid, run_id


ROWS = [
    ({"ema_fast.period": 5}, float("-inf"), 0),  # junk row stored first
    ({"ema_fast.period": 10}, 2.0, 50),
    ({"ema_fast.period": 20}, 3.0, 40),
    ({"ema_fast.period": 30}, float("nan"), 12),
]


def test_picks_row_matching_validated_params(state: StateManager) -> None:
    sid, run_id = _screening_run(state, ROWS)
    # validation params use config underscore keys + run-level knobs
    ref = find_screening_reference(
        state, sid, "s1", validated_params={"ema_fast_period": 10, "initial_balance": 1000}
    )
    assert ref is not None
    assert (ref.sharpe, ref.trades) == (2.0, 50)
    assert ref.source == f"screening_run:{run_id}"


def test_falls_back_to_best_finite_sharpe_with_trades(state: StateManager) -> None:
    sid, run_id = _screening_run(state, ROWS)
    ref = find_screening_reference(state, sid, "s1", validated_params={})
    assert ref is not None
    assert (ref.sharpe, ref.trades) == (3.0, 40)
    assert ref.source == f"screening_run:{run_id}:best"


def test_junk_reference_no_longer_masks_collapse(state: StateManager) -> None:
    """Validation Sharpe -1 vs a real screening row now flags the collapse."""
    sid, _ = _screening_run(state, ROWS)
    ref = find_screening_reference(state, sid, "s1", validated_params={"ema_fast.period": 20})
    assert ref is not None
    report = assess_consistency(ref, val_sharpe=-1.0, val_trades=38)
    assert any("collapse" in f for f in report.flags)


def test_only_junk_rows_fall_through_to_discovery(state: StateManager) -> None:
    sid, _ = _screening_run(state, [({"x": 1}, float("-inf"), 0), ({"x": 2}, 1.5, 0)])
    assert find_screening_reference(state, sid, "s1", validated_params={"x": 1}) is None


def test_discovery_reference_prefers_full_range_metrics(state: StateManager) -> None:
    sid = state.create_strategy("genome_abc123", {"name": "genome_abc123"})
    disc_id = state.create_backtest_run(
        strategy_id=None, run_mode="discovery", symbols=["BTCUSDT"], timeframe="4h",
        start_date="2024-01-01", end_date="2025-01-01", parameters={},
    )
    state.update_backtest_run_status(disc_id, "completed")
    notes = json.dumps({"top_strategies": [{
        "dsl": {"name": "genome_abc123"},
        "sharpe": 2.4, "trades": 90,  # mean of 3 sub-windows
        "full_range_sharpe": 1.1, "full_range_trades": 88,
    }]})
    state.save_backtest_result(disc_id, {"sharpe_ratio": 2.4, "notes": notes})
    ref = find_screening_reference(state, sid, "genome_abc123")
    assert ref is not None
    assert (ref.sharpe, ref.trades) == (1.1, 88)
