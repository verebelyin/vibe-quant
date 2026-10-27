"""Validation consistency matches screening rows through the resolver's alias mapping.

``_normalize_params`` folded dots to underscores only, so a validation param spelled the
way screening's ONE resolver accepts it (``take_profit.risk_reward_ratio``) never matched
the ``take_profit_risk_reward`` field a sweep row stores, and ``find_screening_reference``
fell back to the best-Sharpe row -- comparing the validation against a different parameter
set (vibe-quant-xh4xn). Both sides now resolve to compiled config fields, and two keys of
one side naming the same field fail the exact-row comparison instead of silently keeping
one.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.translator import translate_dsl_config
from vibe_quant.validation.consistency import assess_consistency, find_screening_reference

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

Row = tuple[dict[str, object], float, int]

# rsi_period + take_profit_risk_reward are the compiled config fields of this DSL; a
# take_profit.risk_reward_ratio override resolves onto take_profit_risk_reward.
DSL_NAME = "consistency_alias_probe"


def _dsl() -> dict[str, object]:
    return {
        "name": DSL_NAME,
        "timeframe": "1m",
        "indicators": {"rsi": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi < 30"]},
        "exit_conditions": {"long": ["rsi > 70"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "risk_reward", "risk_reward_ratio": 2.5},
    }


@pytest.fixture
def state(tmp_path: Path) -> Iterator[StateManager]:
    s = StateManager(db_path=tmp_path / "t.db")
    yield s
    s.close()


def _screening(state: StateManager, rows: list[Row]) -> tuple[int, int]:
    sid = state.create_strategy(DSL_NAME, _dsl())
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="screening", symbols=["BTCUSDT"], timeframe="1m",
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


# The reviewer's repro: rsi 12 at Sharpe 1 / 10 trades, rsi 14 at Sharpe 3 / 20 trades.
REPRO_ROWS: list[Row] = [
    ({"rsi_period": 12, "take_profit_risk_reward": 2.5}, 1.0, 10),
    ({"rsi_period": 14, "take_profit_risk_reward": 2.5}, 3.0, 20),
]


def test_alias_validation_key_selects_the_exact_row(state: StateManager) -> None:
    """Repro: ``take_profit.risk_reward_ratio`` matches the stored ``take_profit_risk_reward``."""
    sid, run_id = _screening(state, REPRO_ROWS)
    ref = find_screening_reference(
        state,
        sid,
        DSL_NAME,
        validated_params={"rsi.period": 12, "take_profit.risk_reward_ratio": 2.5},
    )
    assert ref is not None
    assert (ref.sharpe, ref.trades) == (1.0, 10), "fell back to the best-Sharpe row"
    assert ref.source == f"screening_run:{run_id}"


def test_alias_spelling_stored_on_the_row_matches_the_canonical_validation_key(
    state: StateManager,
) -> None:
    """The alias may sit on either side: rows store the raw sweep spelling too."""
    rows: list[Row] = [
        ({"rsi.period": 12, "take_profit.risk_reward_ratio": 2.5}, 1.0, 10),
        ({"rsi.period": 14, "take_profit.risk_reward_ratio": 2.5}, 3.0, 20),
    ]
    sid, run_id = _screening(state, rows)
    ref = find_screening_reference(
        state,
        sid,
        DSL_NAME,
        validated_params={"rsi.period": 12, "take_profit_risk_reward": 2.5},
    )
    assert ref is not None
    assert (ref.sharpe, ref.trades) == (1.0, 10)
    assert ref.source == f"screening_run:{run_id}"


def test_canonical_validation_key_still_matches_exactly(state: StateManager) -> None:
    """The exact match that worked before the fix is unchanged."""
    sid, run_id = _screening(state, REPRO_ROWS)
    ref = find_screening_reference(
        state, sid, DSL_NAME, validated_params={"rsi.period": 12, "take_profit.risk_reward": 2.5}
    )
    assert ref is not None
    assert (ref.sharpe, ref.trades) == (1.0, 10)
    assert ref.source == f"screening_run:{run_id}"


def test_key_outside_the_compiled_config_keeps_its_dot_folded_spelling(
    state: StateManager, caplog: pytest.LogCaptureFixture
) -> None:
    """A key the resolver does not know is compared by spelling, not dropped."""
    rows: list[Row] = [
        ({"rsi_period": 12, "mystery_knob": 7}, 1.0, 10),
        ({"rsi_period": 14}, 3.0, 20),
    ]
    sid, run_id = _screening(state, rows)
    with caplog.at_level(logging.DEBUG, logger="vibe_quant.validation.consistency"):
        ref = find_screening_reference(
            state, sid, DSL_NAME, validated_params={"rsi.period": 12, "mystery_knob": 7}
        )
    assert ref is not None
    assert (ref.sharpe, ref.trades) == (1.0, 10)
    assert ref.source == f"screening_run:{run_id}"
    assert "Unknown strategy parameter" not in caplog.text, "the resolver's error text leaked"
    assert "Traceback" not in caplog.text


def test_uncompilable_dsl_keeps_the_dot_folded_comparison(
    state: StateManager, caplog: pytest.LogCaptureFixture
) -> None:
    """No config fields (DSL does not compile): the old exact match still works, quietly."""
    sid = state.create_strategy("s1", {"name": "s1"})
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="screening", symbols=["BTCUSDT"], timeframe="1m",
        start_date="2024-01-01", end_date="2025-01-01", parameters={},
    )
    state.update_backtest_run_status(run_id, "completed")
    state.conn.execute(
        "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades)"
        " VALUES (?, ?, ?, ?)",
        (run_id, json.dumps({"rsi.period": 12}), 1.0, 10),
    )
    state.conn.commit()
    with caplog.at_level(logging.DEBUG, logger="vibe_quant.validation.consistency"):
        ref = find_screening_reference(
            state, sid, "s1", validated_params={"rsi_period": 12}
        )
    assert ref is not None
    assert (ref.sharpe, ref.trades) == (1.0, 10)
    assert ref.source == f"screening_run:{run_id}"
    assert "Strategy validation failed" not in caplog.text, "the parser's error text leaked"
    assert "Traceback" not in caplog.text


def _ambiguity_notes(ref_notes: list[str]) -> list[str]:
    return [n for n in ref_notes if "ambiguous" in n.lower()]


def test_two_row_keys_naming_one_field_are_explained_in_the_consistency_result(
    state: StateManager,
) -> None:
    """An ambiguous row cannot claim the exact match; the report says which keys and why."""
    rows: list[Row] = [
        (
            {
                "rsi_period": 12,
                "take_profit_risk_reward": 2.5,
                "take_profit.risk_reward_ratio": 2.5,
            },
            3.0,
            20,
        ),
        ({"rsi_period": 14, "take_profit_risk_reward": 2.5}, 1.0, 10),
    ]
    sid, _ = _screening(state, rows)
    ref = find_screening_reference(
        state,
        sid,
        DSL_NAME,
        validated_params={"rsi.period": 12, "take_profit.risk_reward_ratio": 2.5},
    )
    assert ref is not None
    assert ref.source.endswith(":best"), "the ambiguous row must not claim the exact match"
    report = assess_consistency(ref, 1.0, 10).to_dict()
    notes = _ambiguity_notes(report["notes"])  # type: ignore[arg-type]
    assert len(notes) == 1, report
    assert "take_profit_risk_reward" in notes[0]
    assert "take_profit.risk_reward_ratio" in notes[0]
    assert "best" in notes[0].lower(), "the note must say what the reference fell back to"


def test_two_validation_keys_naming_one_field_are_explained_in_the_consistency_result(
    state: StateManager,
) -> None:
    """Same on the validation side: no exact row is claimed, and the report says why."""
    sid, _ = _screening(state, REPRO_ROWS)
    ref = find_screening_reference(
        state,
        sid,
        DSL_NAME,
        validated_params={
            "rsi.period": 12,
            "take_profit_risk_reward": 2.5,
            "take_profit.risk_reward_ratio": 2.5,
        },
    )
    assert ref is not None
    assert ref.source.endswith(":best")
    assert (ref.sharpe, ref.trades) == (3.0, 20)
    report = assess_consistency(ref, 1.0, 10).to_dict()
    notes = _ambiguity_notes(report["notes"])  # type: ignore[arg-type]
    assert len(notes) == 1, report
    assert "take_profit_risk_reward" in notes[0]
    assert "take_profit.risk_reward_ratio" in notes[0]


def test_unambiguous_lookup_carries_no_ambiguity_note(state: StateManager) -> None:
    sid, _ = _screening(state, REPRO_ROWS)
    ref = find_screening_reference(
        state, sid, DSL_NAME, validated_params={"rsi.period": 12, "take_profit.risk_reward": 2.5}
    )
    assert ref is not None
    assert ref.notes == []


# --- editor-format (frontend DslConfig) strategies, translated as both runners do ---

EDITOR_NAME = "editor_alias_probe"


def _editor_dsl() -> dict[str, object]:
    return {
        "general": {"strategy_type": "momentum", "symbols": ["BTCUSDT"], "timeframe": "1h"},
        "indicators": [{"type": "RSI", "params": {"period": 14}}],
        "conditions": {
            "entry": [{"left": "RSI(14)", "operator": "<", "right": "30"}],
            "exit": [{"left": "RSI(14)", "operator": ">", "right": "70", "logic": None}],
        },
        "risk": {
            "stop_loss": {"type": "fixed_pct", "value": 2},
            "take_profit": {"type": "risk_reward", "value": 2.5},
            "position_sizing": {"type": "percent_equity", "value": 10},
        },
        "time": {},
    }


def _editor_screening(
    state: StateManager, rows: list[Row], parameters: dict[str, object]
) -> tuple[int, int]:
    sid = state.create_strategy(EDITOR_NAME, _editor_dsl())
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="screening", symbols=["BTCUSDT"], timeframe="1h",
        start_date="2024-01-01", end_date="2025-01-01", parameters=parameters,
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


@pytest.mark.parametrize("with_translated_override", [False, True])
def test_editor_format_strategy_resolves_aliases_like_screening(
    state: StateManager, with_translated_override: bool
) -> None:
    """The stored editor DSL is translated before the field lookup, as the runners do; a
    screening run whose ``dsl_override`` is that translation resolves identically."""
    parameters: dict[str, object] = (
        {"dsl_override": translate_dsl_config(_editor_dsl(), strategy_name=EDITOR_NAME)}  # type: ignore[arg-type]
        if with_translated_override
        else {}
    )
    rows: list[Row] = [
        ({"rsi_14_period": 12, "take_profit_risk_reward": 2.5}, 1.0, 10),
        ({"rsi_14_period": 14, "take_profit_risk_reward": 2.5}, 3.0, 20),
    ]
    sid, run_id = _editor_screening(state, rows, parameters)
    ref = find_screening_reference(
        state,
        sid,
        EDITOR_NAME,
        validated_params={"rsi_14_period": 12, "take_profit.risk_reward_ratio": 2.5},
    )
    assert ref is not None
    assert (ref.sharpe, ref.trades) == (1.0, 10), "fell back to the best-Sharpe row"
    assert ref.source == f"screening_run:{run_id}"


def test_editor_format_strategy_that_does_not_translate_keeps_the_dot_folded_comparison(
    state: StateManager,
) -> None:
    """Genuinely invalid editor DSL: no fields, the old spelling comparison still works."""
    broken = _editor_dsl()
    broken["conditions"] = {"entry": [{"left": "RSI(14)", "operator": "??", "right": "x"}]}
    sid = state.create_strategy("editor_broken", broken)
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="screening", symbols=["BTCUSDT"], timeframe="1h",
        start_date="2024-01-01", end_date="2025-01-01", parameters={},
    )
    state.update_backtest_run_status(run_id, "completed")
    for params, sharpe, trades in [({"rsi.period": 12}, 1.0, 10), ({"rsi.period": 14}, 3.0, 20)]:
        state.conn.execute(
            "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades)"
            " VALUES (?, ?, ?, ?)",
            (run_id, json.dumps(params), sharpe, trades),
        )
    state.conn.commit()
    ref = find_screening_reference(
        state, sid, "editor_broken", validated_params={"rsi_period": 12}
    )
    assert ref is not None
    assert (ref.sharpe, ref.trades) == (1.0, 10)
    assert ref.source == f"screening_run:{run_id}"
