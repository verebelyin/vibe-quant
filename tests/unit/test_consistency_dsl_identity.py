"""A screening reference must come from a run that executed the SAME effective DSL.

A screening launch may carry ``parameters.dsl_override``: the CLI then runs that DSL
INSTEAD of the stored one (a discovery replay, ``screening/__main__.py``), so its sweep
rows describe a different strategy. The lookup resolved its config fields from the STORED
DSL and matched rows by parameters alone, so a validation run of the stored strategy was
compared against a champion of another strategy -- or, with no exact row, against that
run's best-Sharpe row (vibe-quant-e2af5, merged into vibe-quant-xh4xn).

Each candidate run's effective DSL (the override used RAW when present, else the stored
one) is compared to the validation run's effective DSL by the canonical body key
(``dsl_body_key``), not by text: a row of another strategy is refused with a reason, and
the lookup fails closed instead of falling back to it.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.consistency import find_screening_reference

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

Row = tuple[dict[str, object], float, int]

DSL_NAME = "consistency_dsl_probe"


def _stored_dsl() -> dict[str, object]:
    """The strategy row's DSL: RSI 14."""
    return {
        "name": DSL_NAME,
        "timeframe": "1m",
        "indicators": {"rsi": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi < 30"]},
        "exit_conditions": {"long": ["rsi > 70"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "risk_reward", "risk_reward_ratio": 2.5},
    }


def _override_dsl() -> dict[str, object]:
    """A different strategy body (RSI 7) a screening replay ran instead."""
    dsl = _stored_dsl()
    return {**dsl, "indicators": {"rsi": {"type": "RSI", "period": 7}}}


@pytest.fixture
def state(tmp_path: Path) -> Iterator[StateManager]:
    s = StateManager(db_path=tmp_path / "t.db")
    yield s
    s.close()


def _run(
    state: StateManager,
    sid: int,
    rows: list[Row],
    parameters: dict[str, object] | None = None,
) -> int:
    run_id = state.create_backtest_run(
        strategy_id=sid,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="1m",
        start_date="2024-01-01",
        end_date="2025-01-01",
        parameters=parameters or {},
    )
    state.update_backtest_run_status(run_id, "completed")
    for params, sharpe, trades in rows:
        state.conn.execute(
            "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades)"
            " VALUES (?, ?, ?, ?)",
            (run_id, json.dumps(params), sharpe, trades),
        )
    state.conn.commit()
    return run_id


def test_validation_of_the_stored_dsl_ignores_a_newer_override_run(
    state: StateManager,
) -> None:
    """The stored-DSL run B wins even though the override run A is newer and better."""
    sid = state.create_strategy(DSL_NAME, _stored_dsl())
    b = _run(state, sid, [({"rsi_period": 14}, 1.5, 20)])
    a = _run(state, sid, [({"rsi_period": 7}, 9.9, 30)], {"dsl_override": _override_dsl()})
    assert a > b, "the override run must be the newest so the old fallback picks it"

    ref = find_screening_reference(
        state, sid, DSL_NAME, validated_params={"rsi.period": 14}
    )

    assert ref is not None
    assert (ref.sharpe, ref.trades) == (1.5, 20), "compared against another strategy"
    assert ref.source == f"screening_run:{b}"


def test_validation_of_the_override_picks_the_override_run(state: StateManager) -> None:
    """The other direction: a validation carrying the override compares against run A."""
    sid = state.create_strategy(DSL_NAME, _stored_dsl())
    _run(state, sid, [({"rsi_period": 14}, 1.5, 20)])
    a = _run(state, sid, [({"rsi_period": 7}, 9.9, 30)], {"dsl_override": _override_dsl()})

    ref = find_screening_reference(
        state,
        sid,
        DSL_NAME,
        validated_params={"dsl_override": _override_dsl(), "rsi.period": 7},
    )

    assert ref is not None
    assert (ref.sharpe, ref.trades) == (9.9, 30)
    assert ref.source == f"screening_run:{a}"


def test_an_override_of_the_same_body_is_the_same_strategy(state: StateManager) -> None:
    """Identity is the canonical body, not "did this run carry an override"."""
    sid = state.create_strategy(DSL_NAME, _stored_dsl())
    _run(state, sid, [({"rsi_period": 14}, 1.5, 20)])
    # Same strategy body, keys inserted in another order: still the same DSL.
    reordered = {key: _stored_dsl()[key] for key in reversed(list(_stored_dsl()))}
    newest = _run(state, sid, [({"rsi_period": 14}, 2.4, 25)], {"dsl_override": reordered})

    ref = find_screening_reference(
        state, sid, DSL_NAME, validated_params={"rsi.period": 14}
    )

    assert ref is not None
    assert (ref.sharpe, ref.trades) == (2.4, 25)
    assert ref.source == f"screening_run:{newest}"


def test_only_rows_of_another_dsl_are_refused_with_a_reason(
    state: StateManager, caplog: pytest.LogCaptureFixture
) -> None:
    """No same-DSL screening run: no reference at all, never another strategy's row."""
    sid = state.create_strategy(DSL_NAME, _stored_dsl())
    a = _run(state, sid, [({"rsi_period": 7}, 9.9, 30)], {"dsl_override": _override_dsl()})

    with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.consistency"):
        ref = find_screening_reference(
            state, sid, DSL_NAME, validated_params={"rsi.period": 14}
        )

    assert ref is None, "fell back to a row of another strategy"
    assert f"screening run {a} executed a different effective DSL" in caplog.text


def test_the_best_row_fallback_stays_inside_the_same_dsl(
    state: StateManager,
) -> None:
    """A newer run of another DSL cannot supply the best-Sharpe fallback either."""
    sid = state.create_strategy(DSL_NAME, _stored_dsl())
    b = _run(state, sid, [({"rsi_period": 13}, 1.5, 20)])
    _run(state, sid, [({"rsi_period": 14}, 9.9, 30)], {"dsl_override": _override_dsl()})

    ref = find_screening_reference(
        state, sid, DSL_NAME, validated_params={"rsi.period": 14}
    )

    assert ref is not None
    assert (ref.sharpe, ref.trades) == (1.5, 20)
    assert ref.source == f"screening_run:{b}:best"


def test_unreadable_parameters_refuse_the_run_with_a_reason(
    state: StateManager, caplog: pytest.LogCaptureFixture
) -> None:
    """A run whose parameters column is corrupt has no known effective DSL: refuse it."""
    sid = state.create_strategy(DSL_NAME, _stored_dsl())
    run_id = _run(state, sid, [({"rsi_period": 14}, 9.9, 30)])
    state.conn.execute(
        "UPDATE backtest_runs SET parameters = ? WHERE id = ?", ("{not json", run_id)
    )
    state.conn.commit()

    with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.consistency"):
        ref = find_screening_reference(
            state, sid, DSL_NAME, validated_params={"rsi.period": 14}
        )

    assert ref is None
    assert f"screening run {run_id} has unreadable parameters" in caplog.text


def _validate(
    state: StateManager, sid: int, parameters: dict[str, object], sharpe: float, trades: int
) -> dict[str, object] | None:
    """The production path: the runner's check with the default (empty) sizing mapping."""
    from types import SimpleNamespace

    from vibe_quant.validation.runner import ValidationRunner

    runner = object.__new__(ValidationRunner)
    runner._state = state  # type: ignore[attr-defined]
    cfg: dict[str, object] = {
        "parameters": parameters,
        "symbols": ["BTCUSDT"],
        "start_date": "2024-01-01",
        "end_date": "2025-01-01",
    }
    return runner._check_screening_consistency(  # type: ignore[no-any-return]
        999,
        sid,
        DSL_NAME,
        SimpleNamespace(sharpe_ratio=sharpe, total_trades=trades),  # type: ignore[arg-type]
        run_config=cfg,
    )


def test_validation_with_only_other_dsl_runs_persists_a_refusal_flag(
    state: StateManager,
) -> None:
    sid = state.create_strategy(DSL_NAME, _stored_dsl())
    a = _run(state, sid, [({"rsi_period": 7}, 9.9, 30)], {"dsl_override": _override_dsl()})

    report = _validate(state, sid, {"rsi_period": 14}, -2.0, 20)

    assert report is not None
    assert report["flags"] == [
        "reference-refused: 1 screening run(s) of this strategy ran a different effective "
        f"DSL or unreadable parameters (ids {a}); metrics not compared"
    ]
    assert (report["screen_sharpe"], report["screen_trades"], report["screen_source"]) == (
        None,
        None,
        None,
    )
    assert "9.9" not in json.dumps(report)


def test_validation_without_any_screening_run_adds_no_report(state: StateManager) -> None:
    sid = state.create_strategy(DSL_NAME, _stored_dsl())

    assert _validate(state, sid, {"rsi_period": 14}, -2.0, 20) is None


def test_corrupt_parameters_with_production_sizing_are_refused_with_a_reason(
    state: StateManager, caplog: pytest.LogCaptureFixture
) -> None:
    sid = state.create_strategy(DSL_NAME, _stored_dsl())
    run_id = _run(state, sid, [({"rsi_period": 14}, 9.9, 30)])
    state.conn.execute("UPDATE backtest_runs SET parameters = ? WHERE id = ?", ("{oops", run_id))
    state.conn.commit()

    with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.consistency"):
        ref = find_screening_reference(
            state, sid, DSL_NAME, validated_params={"rsi.period": 14}, val_sizing={}
        )
        report = _validate(state, sid, {"rsi_period": 14}, -2.0, 20)

    assert ref is None
    assert f"screening run {run_id} has unreadable parameters" in caplog.text
    assert report is not None
    assert f"(ids {run_id})" in str(report["flags"])


def test_corrupt_newest_run_still_falls_back_to_an_older_valid_one(
    state: StateManager,
) -> None:
    sid = state.create_strategy(DSL_NAME, _stored_dsl())
    good = _run(state, sid, [({"rsi_period": 14}, 1.5, 20)])
    bad = _run(state, sid, [({"rsi_period": 14}, 9.9, 30)])
    state.conn.execute("UPDATE backtest_runs SET parameters = ? WHERE id = ?", ("{oops", bad))
    state.conn.commit()

    ref = find_screening_reference(
        state, sid, DSL_NAME, validated_params={"rsi.period": 14}, val_sizing={}
    )

    assert ref is not None
    assert (ref.sharpe, ref.source) == (1.5, f"screening_run:{good}")


def test_a_different_entry_rule_is_a_different_strategy(state: StateManager) -> None:
    """Same indicators, RSI < 20 vs RSI < 30: another strategy, never the reference."""
    sid = state.create_strategy(DSL_NAME, _stored_dsl())
    stricter = {**_stored_dsl(), "entry_conditions": {"long": ["rsi < 20"]}}
    _run(state, sid, [({"rsi_period": 14}, 9.9, 30)], {"dsl_override": stricter})

    ref = find_screening_reference(
        state, sid, DSL_NAME, validated_params={"rsi.period": 14}
    )

    assert ref is None, "RSI < 20 run was compared as the RSI < 30 strategy"


def test_distinct_invalid_dsl_bodies_do_not_collide(state: StateManager) -> None:
    """Bodies that do not compile are compared raw: {bad: 1} is not {bad: 2}."""
    sid = state.create_strategy(DSL_NAME, _stored_dsl())
    same = _run(state, sid, [({"x": 1}, 1.5, 20)], {"dsl_override": {"bad": 1}})
    _run(state, sid, [({"x": 1}, 9.9, 30)], {"dsl_override": {"bad": 2}})

    ref = find_screening_reference(
        state, sid, DSL_NAME, validated_params={"dsl_override": {"bad": 1}}
    )

    assert ref is not None
    assert (ref.sharpe, ref.source) == (1.5, f"screening_run:{same}:best")


def test_explicit_defaults_are_the_same_strategy(state: StateManager) -> None:
    sid = state.create_strategy(DSL_NAME, _stored_dsl())
    explicit = {**_stored_dsl(), "description": "", "version": 1, "additional_timeframes": []}
    newest = _run(state, sid, [({"rsi_period": 14}, 2.4, 25)], {"dsl_override": explicit})

    ref = find_screening_reference(
        state, sid, DSL_NAME, validated_params={"rsi.period": 14}
    )

    assert ref is not None
    assert ref.source == f"screening_run:{newest}"
