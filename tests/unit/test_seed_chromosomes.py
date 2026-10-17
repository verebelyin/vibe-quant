"""Unit tests for the shared warm-start seed-chromosome loader.

``vibe_quant.discovery.seed.load_seed_chromosomes`` is the single loader the
launch API and the discovery subprocess both call, so its "nothing to seed
from" contract (``None``) is pinned here at the unit level.
"""

from __future__ import annotations

import json

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.genome import chromosome_to_serializable
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.seed import load_seed_chromosomes


@pytest.fixture()
def state(tmp_path):
    mgr = StateManager(db_path=tmp_path / "seed.db")
    _ = mgr.conn
    yield mgr
    mgr.close()


def _create_run(state: StateManager) -> int:
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2025-01-01",
        end_date="2025-04-01",
        parameters={},
    )
    state.update_backtest_run_status(run_id, "completed")
    return run_id


def _chromosome() -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=3.0,
        direction=Direction.LONG,
    )


def test_returns_none_when_run_has_no_result_row(state: StateManager) -> None:
    assert load_seed_chromosomes(state, _create_run(state)) is None


def test_returns_none_when_notes_is_empty(state: StateManager) -> None:
    run_id = _create_run(state)
    state.save_backtest_result(run_id, {"notes": ""})
    assert load_seed_chromosomes(state, run_id) is None


def test_returns_none_when_notes_is_unparseable(state: StateManager) -> None:
    run_id = _create_run(state)
    state.save_backtest_result(run_id, {"notes": "not-json{"})
    assert load_seed_chromosomes(state, run_id) is None


def test_returns_none_when_top_strategies_is_empty(state: StateManager) -> None:
    run_id = _create_run(state)
    state.save_backtest_result(run_id, {"notes": json.dumps({"top_strategies": []})})
    assert load_seed_chromosomes(state, run_id) is None


def test_returns_none_when_entries_carry_no_chromosome(state: StateManager) -> None:
    run_id = _create_run(state)
    state.save_backtest_result(
        run_id,
        {
            "notes": json.dumps(
                {"top_strategies": [{"strategy_id": 1}, {"chromosome": None}]}
            )
        },
    )
    assert load_seed_chromosomes(state, run_id) is None


def test_round_trips_a_persisted_chromosome(state: StateManager) -> None:
    run_id = _create_run(state)
    state.save_backtest_result(
        run_id,
        {
            "notes": json.dumps(
                {"top_strategies": [{"chromosome": chromosome_to_serializable(_chromosome())}]}
            )
        },
    )

    loaded = load_seed_chromosomes(state, run_id)
    assert loaded is not None
    assert len(loaded) == 1
    gene = loaded[0].entry_genes[0]
    assert gene.indicator_type == "RSI"
    assert gene.condition == ConditionType.LT
    assert gene.threshold == 30.0
    assert loaded[0].direction == Direction.LONG
    assert loaded[0].stop_loss_pct == 2.0
