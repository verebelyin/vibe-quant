"""Unit tests for the shared warm-start seed-chromosome loader.

``vibe_quant.discovery.seed.load_seed_chromosomes`` is the single loader the
launch API and the discovery subprocess both call, so its "nothing to seed
from" contract (``None``) is pinned here at the unit level.
"""

from __future__ import annotations

import json
import logging

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.genome import (
    chromosome_to_dsl,
    chromosome_to_serializable,
)
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


# ---------------------------------------------------------------------------
# R7 / R7b / R7d: legacy direction=both seeds load as long (inydc T1)
# ---------------------------------------------------------------------------


def _legacy_both_blob(**overrides: object) -> dict[str, object]:
    blob: dict[str, object] = {
        "entry_genes": [
            {
                "indicator_type": "RSI",
                "parameters": {"period": 14.0},
                "condition": "crosses_below",
                "threshold": 30.0,
                "sub_value": None,
            }
        ],
        "exit_genes": [
            {
                "indicator_type": "RSI",
                "parameters": {"period": 14.0},
                "condition": "crosses_above",
                "threshold": 60.0,
                "sub_value": None,
            }
        ],
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
        "direction": "both",
    }
    blob.update(overrides)
    return blob


def _save_chromosomes(state: StateManager, run_id: int, blobs: list[dict[str, object]]) -> None:
    state.save_backtest_result(
        run_id,
        {
            "notes": json.dumps(
                {"top_strategies": [{"chromosome": b} for b in blobs]}
            )
        },
    )


def test_r7_legacy_both_seed_loads_as_long_with_long_sl_tp(
    state: StateManager, caplog: pytest.LogCaptureFixture
) -> None:
    run_id = _create_run(state)
    _save_chromosomes(
        state,
        run_id,
        [_legacy_both_blob(stop_loss_long_pct=1.09, take_profit_long_pct=17.13)],
    )

    with caplog.at_level(logging.WARNING):
        loaded = load_seed_chromosomes(state, run_id)

    assert loaded is not None and len(loaded) == 1
    chrom = loaded[0]
    assert chrom.direction is Direction.LONG
    assert chrom.stop_loss_pct == 1.09
    assert chrom.take_profit_pct == 17.13
    assert chrom.stop_loss_long_pct is None
    assert chrom.stop_loss_short_pct is None
    assert chrom.take_profit_long_pct is None
    assert chrom.take_profit_short_pct is None

    # The DSL's long lists are what the legacy BOTH chromosome traded.
    dsl = chromosome_to_dsl(chrom)
    assert dsl["entry_conditions"] == {"long": ["rsi_entry_0 crosses_below 30"]}
    assert dsl["exit_conditions"] == {"long": ["rsi_exit_0 crosses_above 60"]}
    assert "legacy direction=both" in caplog.text


def test_r7b_legacy_both_without_long_override_keeps_generic_sl(state: StateManager) -> None:
    run_id = _create_run(state)
    _save_chromosomes(state, run_id, [_legacy_both_blob()])

    loaded = load_seed_chromosomes(state, run_id)
    assert loaded is not None
    chrom = loaded[0]
    assert chrom.direction is Direction.LONG
    assert chrom.stop_loss_pct == 2.0
    assert chrom.take_profit_pct == 4.0


def test_r7d_legacy_both_conversion_warns_once(
    state: StateManager, caplog: pytest.LogCaptureFixture
) -> None:
    run_id = _create_run(state)
    _save_chromosomes(state, run_id, [_legacy_both_blob(), _legacy_both_blob()])

    with caplog.at_level(logging.WARNING):
        loaded = load_seed_chromosomes(state, run_id)

    assert loaded is not None and len(loaded) == 2
    warnings = [r for r in caplog.records if "legacy direction=both" in r.getMessage()]
    assert len(warnings) == 1
    assert "2 legacy direction=both" in warnings[0].getMessage()


def test_new_both_seed_with_short_genes_stays_both(state: StateManager) -> None:
    run_id = _create_run(state)
    both = StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        short_entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        short_exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.BOTH,
    )
    _save_chromosomes(state, run_id, [chromosome_to_serializable(both)])

    loaded = load_seed_chromosomes(state, run_id)
    assert loaded is not None and len(loaded) == 1
    assert loaded[0].direction is Direction.BOTH
    assert loaded[0].short_entry_genes[0].threshold == 70.0
    assert loaded[0].short_exit_genes[0].threshold == 30.0
