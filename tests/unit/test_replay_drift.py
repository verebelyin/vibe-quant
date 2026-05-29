"""Tests for post-promote replay drift check (bd-l6ml)."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TCH003

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.screening.replay_drift import (
    SHARPE_DRIFT_THRESHOLD,
    TRADE_DRIFT_THRESHOLD,
    check_replay_drift,
)


def _make_discovery_run(
    state: StateManager,
    *,
    sharpe: float,
    trades: int,
    strategy_index: int = 0,
    full_range_sharpe: float | None = None,
    full_range_trades: int | None = None,
) -> int:
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2025-09-01",
        end_date="2026-02-28",
        parameters={},
    )
    top: list[dict[str, object]] = [
        {"dsl": {"name": f"ga_{i}"}, "sharpe": 1.0, "trades": 10}
        for i in range(strategy_index + 1)
    ]
    entry: dict[str, object] = {
        "dsl": {"name": f"ga_{strategy_index}"},
        "sharpe": sharpe,
        "trades": trades,
    }
    # bd vibe-quant-rewru: post-rewru runs also carry a full-range headline; when
    # absent the drift check must fall back to the aggregate sharpe/trades above.
    if full_range_sharpe is not None:
        entry["full_range_sharpe"] = full_range_sharpe
    if full_range_trades is not None:
        entry["full_range_trades"] = full_range_trades
    top[strategy_index] = entry
    notes = json.dumps({"top_strategies": top})
    state.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)",
        (run_id, notes),
    )
    state.conn.commit()
    return run_id


def _make_promote_screening_run(
    state: StateManager,
    *,
    discovery_run_id: int,
    strategy_index: int,
    sweep_sharpe: float | None,
    sweep_trades: int | None,
) -> int:
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2025-09-01",
        end_date="2026-02-28",
        parameters={
            "promote_source": {
                "discovery_run_id": discovery_run_id,
                "strategy_index": strategy_index,
            }
        },
    )
    if sweep_sharpe is not None or sweep_trades is not None:
        state.conn.execute(
            """
            INSERT INTO sweep_results
                (run_id, parameters, sharpe_ratio, total_trades, is_pareto_optimal)
            VALUES (?, ?, ?, ?, 1)
            """,
            (run_id, json.dumps({}), sweep_sharpe, sweep_trades),
        )
        state.conn.commit()
    return run_id


@pytest.fixture()
def state(tmp_path: Path) -> StateManager:
    mgr = StateManager(db_path=tmp_path / "test.db")
    _ = mgr.conn
    yield mgr
    mgr.close()


def _load_drift_from_params(state: StateManager, run_id: int) -> dict[str, object]:
    row = state.conn.execute(
        "SELECT parameters FROM backtest_runs WHERE id = ?", (run_id,)
    ).fetchone()
    params = json.loads(row[0])
    assert isinstance(params, dict)
    drift = params.get("replay_drift")
    assert isinstance(drift, dict), "replay_drift should be persisted in parameters"
    return drift


def test_no_promote_source_is_noop(state: StateManager) -> None:
    """Regular screening runs (no promote_source) skip the drift check."""
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2025-09-01",
        end_date="2026-02-28",
        parameters={},
    )
    assert check_replay_drift(state, run_id) is None

    row = state.conn.execute(
        "SELECT parameters FROM backtest_runs WHERE id = ?", (run_id,)
    ).fetchone()
    params = json.loads(row[0])
    assert "replay_drift" not in params


def test_flags_large_sharpe_drift(state: StateManager) -> None:
    """Sharpe drop below 0.8x discovery sharpe → flagged=True."""
    disc = _make_discovery_run(state, sharpe=4.70, trades=79)
    run_id = _make_promote_screening_run(
        state,
        discovery_run_id=disc,
        strategy_index=0,
        sweep_sharpe=2.47,
        sweep_trades=75,
    )
    payload = check_replay_drift(state, run_id)
    assert payload is not None
    assert payload["flagged"] is True
    assert payload["discovery_sharpe"] == 4.70
    assert payload["screening_sharpe"] == 2.47
    assert payload["sharpe_ratio"] == pytest.approx(2.47 / 4.70)

    persisted = _load_drift_from_params(state, run_id)
    assert persisted["flagged"] is True


def test_flags_large_trade_drift(state: StateManager) -> None:
    """Trade count below 0.9x discovery count → flagged=True."""
    disc = _make_discovery_run(state, sharpe=2.0, trades=100)
    run_id = _make_promote_screening_run(
        state,
        discovery_run_id=disc,
        strategy_index=0,
        sweep_sharpe=1.95,
        sweep_trades=50,
    )
    payload = check_replay_drift(state, run_id)
    assert payload is not None
    assert payload["flagged"] is True
    assert payload["trade_ratio"] == pytest.approx(0.5)
    assert payload["sharpe_ratio"] == pytest.approx(1.95 / 2.0)


def test_clean_replay_not_flagged(state: StateManager) -> None:
    """Metrics within tolerance → flagged=False, still persisted."""
    disc = _make_discovery_run(state, sharpe=2.0, trades=100)
    run_id = _make_promote_screening_run(
        state,
        discovery_run_id=disc,
        strategy_index=0,
        sweep_sharpe=1.95,
        sweep_trades=98,
    )
    payload = check_replay_drift(state, run_id)
    assert payload is not None
    assert payload["flagged"] is False
    assert payload["trade_ratio"] == pytest.approx(0.98)

    persisted = _load_drift_from_params(state, run_id)
    assert persisted["flagged"] is False


def test_threshold_boundary_trade_ratio_exact(state: StateManager) -> None:
    """trades_ratio == 0.9 is NOT flagged (strict `<` comparison)."""
    disc = _make_discovery_run(state, sharpe=2.0, trades=100)
    run_id = _make_promote_screening_run(
        state,
        discovery_run_id=disc,
        strategy_index=0,
        sweep_sharpe=1.99,
        sweep_trades=90,
    )
    payload = check_replay_drift(state, run_id)
    assert payload is not None
    assert payload["trade_ratio"] == pytest.approx(TRADE_DRIFT_THRESHOLD)
    assert payload["flagged"] is False


def test_missing_sweep_results(state: StateManager) -> None:
    """If screening produced no sweep rows, both ratios are None, not flagged."""
    disc = _make_discovery_run(state, sharpe=2.0, trades=100)
    run_id = _make_promote_screening_run(
        state,
        discovery_run_id=disc,
        strategy_index=0,
        sweep_sharpe=None,
        sweep_trades=None,
    )
    payload = check_replay_drift(state, run_id)
    assert payload is not None
    assert payload["screening_sharpe"] is None
    assert payload["screening_trades"] is None
    assert payload["trade_ratio"] is None
    assert payload["sharpe_ratio"] is None
    assert payload["flagged"] is False


def test_zero_discovery_sharpe_skips_ratio(state: StateManager) -> None:
    """Zero denominator → ratio is None (no division-by-zero)."""
    disc = _make_discovery_run(state, sharpe=0.0, trades=0)
    run_id = _make_promote_screening_run(
        state,
        discovery_run_id=disc,
        strategy_index=0,
        sweep_sharpe=1.5,
        sweep_trades=50,
    )
    payload = check_replay_drift(state, run_id)
    assert payload is not None
    assert payload["sharpe_ratio"] is None
    assert payload["trade_ratio"] is None
    assert payload["flagged"] is False


def test_missing_discovery_result_returns_none(state: StateManager) -> None:
    """Broken promote_source (discovery run deleted) → no-op, no persisted payload."""
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2025-09-01",
        end_date="2026-02-28",
        parameters={
            "promote_source": {"discovery_run_id": 99999, "strategy_index": 0}
        },
    )
    assert check_replay_drift(state, run_id) is None

    row = state.conn.execute(
        "SELECT parameters FROM backtest_runs WHERE id = ?", (run_id,)
    ).fetchone()
    params = json.loads(row[0])
    assert "replay_drift" not in params


def test_prefers_full_range_headline_over_aggregate(state: StateManager) -> None:
    """bd vibe-quant-rewru: the run-812 case, fixed.

    Discovery's multi-window AGGREGATE was sharpe=4.70/trades=79, but a continuous
    full-range replay yields 2.80/46 -- and the screening replay (sweep) also
    produces 2.80/46. With the full-range headline as the baseline the drift check
    compares like-for-like (ratios ~1.0) and does NOT flag, whereas the old
    aggregate baseline would have flagged this perfectly-reproducing champion.
    """
    disc = _make_discovery_run(
        state,
        sharpe=4.70,
        trades=79,
        full_range_sharpe=2.80,
        full_range_trades=46,
    )
    run_id = _make_promote_screening_run(
        state,
        discovery_run_id=disc,
        strategy_index=0,
        sweep_sharpe=2.80,
        sweep_trades=46,
    )
    payload = check_replay_drift(state, run_id)
    assert payload is not None
    # Baseline is the full-range headline, NOT the multi-window aggregate (4.70/79).
    assert payload["discovery_sharpe"] == 2.80
    assert payload["discovery_trades"] == 46
    assert payload["sharpe_ratio"] == pytest.approx(1.0)
    assert payload["trade_ratio"] == pytest.approx(1.0)
    assert payload["flagged"] is False


def test_falls_back_to_aggregate_when_full_range_absent(state: StateManager) -> None:
    """Pre-rewru runs lack full_range_* -> baseline falls back to the aggregate.

    Same inputs as test_flags_large_sharpe_drift, but routed through the fallback
    path, proving discovery results persisted before rewru keep their original
    drift behaviour.
    """
    disc = _make_discovery_run(state, sharpe=4.70, trades=79)  # no full_range_*
    run_id = _make_promote_screening_run(
        state,
        discovery_run_id=disc,
        strategy_index=0,
        sweep_sharpe=2.47,
        sweep_trades=75,
    )
    payload = check_replay_drift(state, run_id)
    assert payload is not None
    assert payload["discovery_sharpe"] == 4.70  # aggregate, via fallback
    assert payload["discovery_trades"] == 79
    assert payload["flagged"] is True


def test_thresholds_reflect_bead_spec() -> None:
    """Guardrail ratios match bd-l6ml requirements (>10% trade, >20% Sharpe)."""
    assert TRADE_DRIFT_THRESHOLD == 0.9
    assert SHARPE_DRIFT_THRESHOLD == 0.8


def test_writes_into_backtest_results_notes_when_row_exists(state: StateManager) -> None:
    """When a backtest_results row exists, drift is merged into its notes JSON."""
    disc = _make_discovery_run(state, sharpe=4.7, trades=79)
    run_id = _make_promote_screening_run(
        state,
        discovery_run_id=disc,
        strategy_index=0,
        sweep_sharpe=2.47,
        sweep_trades=51,
    )
    # Simulate an earlier writer (e.g. compiler_version) seeding notes.
    state.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)",
        (run_id, json.dumps({"compiler_version": "abc123"})),
    )
    state.conn.commit()

    payload = check_replay_drift(state, run_id)
    assert payload is not None

    row = state.conn.execute(
        "SELECT notes FROM backtest_results WHERE run_id = ?", (run_id,)
    ).fetchone()
    notes = json.loads(row[0])
    assert notes["compiler_version"] == "abc123"
    assert notes["replay_drift"]["flagged"] is True
