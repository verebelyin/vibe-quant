"""Tests for discovery promote & replay endpoints."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TCH003

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
async def client(tmp_db: Path):
    app = create_app()
    state_mgr = StateManager(db_path=tmp_db)
    _ = state_mgr.conn

    job_mgr = BacktestJobManager(db_path=tmp_db)
    ws_mgr = ConnectionManager()
    await ws_mgr.start()

    app.state.state_manager = state_mgr
    app.state.job_manager = job_mgr
    app.state.ws_manager = ws_mgr

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, state_mgr

    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


def _create_discovery_run_with_results(
    state: StateManager,
    *,
    timeframe: str = "4h",
    symbols: list[str] | None = None,
    top_strategies: list[dict[str, object]] | None = None,
    notes_extra: dict[str, object] | None = None,
) -> int:
    """Create a completed discovery run with genome results in notes."""
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=symbols or ["BTCUSDT", "ETHUSDT"],
        timeframe=timeframe,
        start_date="2024-01-01",
        end_date="2025-01-01",
        parameters={"population": 20, "generations": 10},
    )
    state.update_backtest_run_status(run_id, "completed")

    # Insert discovery results with genome DSL
    genome_dsl = {
        "name": "ga_winner_1",
        "strategy_type": "momentum",
        "entry": {"conditions": [{"indicator": "RSI", "params": {"period": 14}, "operator": "<", "value": 30}]},
        "exit": {"conditions": [{"indicator": "RSI", "params": {"period": 14}, "operator": ">", "value": 70}]},
    }
    notes_payload: dict[str, object] = {
        "top_strategies": top_strategies
        or [
            {"dsl": genome_dsl, "score": 1.5, "trades": 42, "sharpe": 1.8},
            {"dsl": {**genome_dsl, "name": "ga_winner_2"}, "score": 1.2, "trades": 30, "sharpe": 1.5},
        ]
    }
    if notes_extra:
        notes_payload.update(notes_extra)

    notes = json.dumps(notes_payload)
    state.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)",
        (run_id, notes),
    )
    state.conn.commit()
    return run_id


# --- Promote tests ---


async def test_promote_creates_strategy_and_launches_screening(client: tuple[AsyncClient, StateManager]) -> None:
    ac, state = client
    run_id = _create_discovery_run_with_results(state)

    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "ga_winner_1"
    assert data["mode"] == "screening"
    assert data["strategy_id"] >= 1
    assert data["run_id"] >= 1

    # Verify strategy was created in DB
    row = state.conn.execute("SELECT name FROM strategies WHERE id = ?", (data["strategy_id"],)).fetchone()
    assert row is not None
    assert row[0] == "ga_winner_1"

    # Verify backtest run was created
    bt_run = state.get_backtest_run(data["run_id"])
    assert bt_run is not None
    assert bt_run["run_mode"] == "screening"
    assert bt_run["strategy_id"] == data["strategy_id"]


async def test_promote_with_validation_mode(client: tuple[AsyncClient, StateManager]) -> None:
    ac, state = client
    # Create a separate discovery run to avoid job conflicts
    run_id = _create_discovery_run_with_results(state)

    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0?mode=validation")
    if r.status_code != 201:
        # Debug: print response detail
        pytest.fail(f"Expected 201, got {r.status_code}: {r.text}")
    assert r.json()["mode"] == "validation"


async def test_promote_reuses_existing_strategy(client: tuple[AsyncClient, StateManager]) -> None:
    ac, state = client
    run_id = _create_discovery_run_with_results(state)

    # First promote creates the strategy
    r1 = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r1.status_code == 201
    sid1 = r1.json()["strategy_id"]

    # Second promote reuses same strategy
    r2 = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r2.status_code == 201
    assert r2.json()["strategy_id"] == sid1


async def test_promote_invalid_index(client: tuple[AsyncClient, StateManager]) -> None:
    ac, state = client
    run_id = _create_discovery_run_with_results(state)

    r = await ac.post(f"/api/discovery/results/{run_id}/promote/99")
    assert r.status_code == 404


async def test_promote_invalid_mode(client: tuple[AsyncClient, StateManager]) -> None:
    ac, state = client
    run_id = _create_discovery_run_with_results(state)

    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0?mode=invalid")
    assert r.status_code == 400


async def test_promote_non_discovery_run(client: tuple[AsyncClient, StateManager]) -> None:
    ac, state = client
    # Create a strategy first to satisfy FK constraint
    cursor = state.conn.execute(
        "INSERT INTO strategies (name, description, dsl_config, strategy_type) VALUES (?, ?, ?, ?)",
        ("test_strat", "test", "{}", "momentum"),
    )
    state.conn.commit()
    sid = cursor.lastrowid
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="screening", symbols=["BTCUSDT"],
        timeframe="4h", start_date="2024-01-01", end_date="2025-01-01", parameters={},
    )
    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r.status_code == 400


async def test_promote_blocks_1m_short_without_opposing_regime_pass(
    client: tuple[AsyncClient, StateManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ac, state = client
    short_entry = {
        "dsl": {
            "name": "ga_short_1m",
            "strategy_type": "momentum",
            "entry_conditions": {"short": ["rsi_entry < 30"]},
            "exit_conditions": {"short": ["rsi_exit > 70"]},
        },
        "chromosome": {"direction": "short"},
        "score": 2.4,
        "trades": 50,
        "sharpe": 2.1,
        "cross_window": {
            "windows_passed": 1,
            "total_windows": 2,
            "passed": False,
            "windows": [
                {"sharpe": 2.1, "return_pct": 0.18, "trades": 50},
                {"sharpe": -1.2, "return_pct": -0.09, "trades": 41},
            ],
        },
    }
    run_id = _create_discovery_run_with_results(
        state,
        timeframe="1m",
        symbols=["BTCUSDT"],
        top_strategies=[short_entry],
        notes_extra={"cross_window_months": [-15], "cross_window_min_sharpe": 0.5},
    )

    monkeypatch.setattr(
        "vibe_quant.api.routers.discovery._window_regime_sign",
        lambda _symbol, start, _end: -1 if start == "2024-01-01" else 1,
    )

    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r.status_code == 409
    assert "opposing-regime cross-window validation" in r.json()["detail"]


async def test_promote_allows_1m_short_with_opposing_regime_pass(
    client: tuple[AsyncClient, StateManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ac, state = client
    short_entry = {
        "dsl": {
            "name": "ga_short_1m_pass",
            "strategy_type": "momentum",
            "entry_conditions": {"short": ["rsi_entry < 30"]},
            "exit_conditions": {"short": ["rsi_exit > 70"]},
        },
        "chromosome": {"direction": "short"},
        "score": 2.7,
        "trades": 48,
        "sharpe": 2.3,
        "cross_window": {
            "windows_passed": 2,
            "total_windows": 2,
            "passed": True,
            "windows": [
                {"sharpe": 2.3, "return_pct": 0.16, "trades": 48},
                {"sharpe": 0.9, "return_pct": 0.04, "trades": 35},
            ],
        },
    }
    run_id = _create_discovery_run_with_results(
        state,
        timeframe="1m",
        symbols=["BTCUSDT"],
        top_strategies=[short_entry],
        notes_extra={"cross_window_months": [-15], "cross_window_min_sharpe": 0.5},
    )

    monkeypatch.setattr(
        "vibe_quant.api.routers.discovery._window_regime_sign",
        lambda _symbol, start, _end: -1 if start == "2024-01-01" else 1,
    )

    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r.status_code == 201
    assert r.json()["name"] == "ga_short_1m_pass"


# --- Replay tests ---


async def test_replay_creates_run_with_dsl_override(client: tuple[AsyncClient, StateManager]) -> None:
    ac, state = client
    run_id = _create_discovery_run_with_results(state)

    r = await ac.post(f"/api/discovery/results/{run_id}/replay/0")
    assert r.status_code == 201
    data = r.json()
    assert data["original_run_id"] == run_id
    assert data["replay_run_id"] >= 1

    # Verify backtest run has dsl_override in parameters
    bt_run = state.get_backtest_run(data["replay_run_id"])
    assert bt_run is not None
    assert bt_run["run_mode"] == "screening"
    assert bt_run["strategy_id"] is None
    params = bt_run.get("parameters", {})
    if isinstance(params, str):
        params = json.loads(params)
    assert "dsl_override" in params


async def test_replay_metrics_note_absent_for_single_window(
    client: tuple[AsyncClient, StateManager],
) -> None:
    """No eval_windows (or 1) — replay metrics comparable, no note."""
    ac, state = client
    run_id = _create_discovery_run_with_results(state)

    r = await ac.post(f"/api/discovery/results/{run_id}/replay/0")
    assert r.status_code == 201
    assert r.json()["metrics_note"] is None


async def test_replay_metrics_note_for_multi_window_fitness(
    client: tuple[AsyncClient, StateManager],
) -> None:
    """eval_windows > 1 — stored fitness is worst-of-N, replay is full-window."""
    ac, state = client
    run_id = _create_discovery_run_with_results(state)
    state.conn.execute(
        "UPDATE backtest_runs SET parameters = ? WHERE id = ?",
        (json.dumps({"population": 20, "generations": 10, "eval_windows": 3}), run_id),
    )
    state.conn.commit()

    r = await ac.post(f"/api/discovery/results/{run_id}/replay/0")
    assert r.status_code == 201
    note = r.json()["metrics_note"]
    assert note is not None
    assert "worst-of-3" in note

    bt_run = state.get_backtest_run(r.json()["replay_run_id"])
    assert bt_run is not None
    params = bt_run.get("parameters", {})
    if isinstance(params, str):
        params = json.loads(params)
    assert "worst-of-3" in str(params.get("replay_note"))


async def test_replay_invalid_index(client: tuple[AsyncClient, StateManager]) -> None:
    ac, state = client
    run_id = _create_discovery_run_with_results(state)

    r = await ac.post(f"/api/discovery/results/{run_id}/replay/99")
    assert r.status_code == 404


async def test_replay_not_found(client: tuple[AsyncClient, StateManager]) -> None:
    ac, _ = client
    r = await ac.post("/api/discovery/results/9999/replay/0")
    assert r.status_code == 404


# --- Screening CLI dsl_override ---


def test_screening_cmd_run_with_dsl_override(tmp_path: Path) -> None:
    """Verify screening CLI accepts dsl_override in run parameters."""
    from vibe_quant.db.state_manager import StateManager

    db_path = tmp_path / "test.db"
    state = StateManager(db_path=db_path)
    _ = state.conn

    dsl_config = {
        "name": "test_replay",
        "strategy_type": "momentum",
        "entry": {"conditions": [{"indicator": "RSI", "params": {"period": 14}, "operator": "<", "value": 30}]},
        "exit": {"conditions": [{"indicator": "RSI", "params": {"period": 14}, "operator": ">", "value": 70}]},
    }

    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2025-01-01",
        parameters={"dsl_override": dsl_config},
    )

    # Verify the run was created with dsl_override
    run = state.get_backtest_run(run_id)
    assert run is not None
    params = run.get("parameters", {})
    if isinstance(params, str):
        import json
        params = json.loads(params)
    assert "dsl_override" in params

    state.close()


# --- vibe-quant-e70tl.1 / .5: promote by DSL content, validate on holdout ---


def _genome_dsl(name: str, threshold: int) -> dict[str, object]:
    return {
        "name": name,
        "timeframe": "4h",
        "indicators": {"rsi_entry_0": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": [f"rsi_entry_0 < {threshold}"]},
        "exit_conditions": {"long": ["rsi_entry_0 > 70"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }


def _stored_dsl(state: StateManager, strategy_id: int) -> dict[str, object]:
    row = state.conn.execute(
        "SELECT dsl_config FROM strategies WHERE id = ?", (strategy_id,)
    ).fetchone()
    assert row is not None
    loaded: dict[str, object] = json.loads(row[0])
    return loaded


async def test_promote_same_name_different_dsl_creates_new_strategy(
    client: tuple[AsyncClient, StateManager],
) -> None:
    """#0 (elite) and #2 (its mutant) share genome_<uid> -> #2 must get its OWN row."""
    ac, state = client
    run_id = _create_discovery_run_with_results(
        state,
        top_strategies=[
            {"dsl": _genome_dsl("genome_abc123", 30), "score": 0.9},
            {"dsl": _genome_dsl("genome_other", 25), "score": 0.8},
            {"dsl": _genome_dsl("genome_abc123", 35), "score": 0.7},
        ],
    )

    r0 = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    r2 = await ac.post(f"/api/discovery/results/{run_id}/promote/2")
    assert r0.status_code == 201 and r2.status_code == 201
    sid0, sid2 = r0.json()["strategy_id"], r2.json()["strategy_id"]
    assert sid0 != sid2
    assert r2.json()["name"] == "genome_abc123_2"

    stored2 = _stored_dsl(state, sid2)
    expected2 = _genome_dsl("genome_abc123", 35)
    assert {k: v for k, v in stored2.items() if k != "name"} == {
        k: v for k, v in expected2.items() if k != "name"
    }
    assert stored2["name"] == "genome_abc123_2"
    # #0's row still holds #0's DSL
    assert _stored_dsl(state, sid0)["entry_conditions"] == {"long": ["rsi_entry_0 < 30"]}


async def test_promote_same_genome_twice_is_idempotent(
    client: tuple[AsyncClient, StateManager],
) -> None:
    ac, state = client
    run_id = _create_discovery_run_with_results(
        state, top_strategies=[{"dsl": _genome_dsl("genome_x1", 30), "score": 0.9}],
    )
    r1 = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    r2 = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r1.json()["strategy_id"] == r2.json()["strategy_id"]
    (count,) = state.conn.execute(
        "SELECT COUNT(*) FROM strategies WHERE name LIKE 'genome_x1%'"
    ).fetchone()
    assert count == 1


async def test_export_matches_by_dsl_not_name(
    client: tuple[AsyncClient, StateManager],
) -> None:
    ac, state = client
    run_id = _create_discovery_run_with_results(
        state,
        top_strategies=[
            {"dsl": _genome_dsl("genome_dup", 30), "score": 0.9},
            {"dsl": _genome_dsl("genome_dup", 40), "score": 0.8},
        ],
    )
    e0 = await ac.post(f"/api/discovery/results/{run_id}/export/0")
    e1 = await ac.post(f"/api/discovery/results/{run_id}/export/1")
    e0_again = await ac.post(f"/api/discovery/results/{run_id}/export/0")
    assert e0.json()["status"] == "created"
    assert e1.json()["status"] == "created"
    assert e1.json()["strategy_id"] != e0.json()["strategy_id"]
    assert e0_again.json() == {**e0.json(), "status": "exists"}


async def test_promote_validation_uses_holdout_range_by_default(
    client: tuple[AsyncClient, StateManager],
) -> None:
    ac, state = client
    run_id = _create_discovery_run_with_results(
        state,
        top_strategies=[{"dsl": _genome_dsl("genome_h1", 30), "score": 0.9}],
        notes_extra={
            "train_dates": ["2024-01-01", "2024-10-19"],
            "holdout_dates": ["2024-10-19", "2025-01-01"],
        },
    )
    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0?mode=validation")
    assert r.status_code == 201
    data = r.json()
    assert data["date_range"] == "holdout"
    assert (data["start_date"], data["end_date"]) == ("2024-10-19", "2025-01-01")
    bt_run = state.get_backtest_run(data["run_id"])
    assert bt_run is not None
    assert (bt_run["start_date"], bt_run["end_date"]) == ("2024-10-19", "2025-01-01")
    params = bt_run["parameters"]
    if isinstance(params, str):
        params = json.loads(params)
    assert params["promote_source"]["date_range"] == "holdout"


async def test_promote_validation_full_range_is_explicit_opt_in(
    client: tuple[AsyncClient, StateManager],
) -> None:
    ac, state = client
    run_id = _create_discovery_run_with_results(
        state,
        top_strategies=[{"dsl": _genome_dsl("genome_h2", 30), "score": 0.9}],
        notes_extra={"holdout_dates": ["2024-10-19", "2025-01-01"]},
    )
    r = await ac.post(
        f"/api/discovery/results/{run_id}/promote/0?mode=validation&validation_range=full"
    )
    assert r.status_code == 201
    assert r.json()["date_range"] == "full"
    assert (r.json()["start_date"], r.json()["end_date"]) == ("2024-01-01", "2025-01-01")
    bad = await ac.post(
        f"/api/discovery/results/{run_id}/promote/0?mode=validation&validation_range=x"
    )
    assert bad.status_code == 400


async def test_promote_validation_without_holdout_uses_full_range(
    client: tuple[AsyncClient, StateManager],
) -> None:
    ac, state = client
    run_id = _create_discovery_run_with_results(
        state, top_strategies=[{"dsl": _genome_dsl("genome_h3", 30), "score": 0.9}],
    )
    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0?mode=validation")
    assert r.status_code == 201
    assert r.json()["date_range"] == "full"


async def test_regime_gate_uses_persisted_window_dates_and_blocks_neutral_base(
    client: tuple[AsyncClient, StateManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New-format cross-window payload (shifted windows only, with dates).

    A neutral base regime no longer waves a 1m short through: it needs a pass
    on a BULL window (adverse for shorts).
    """
    ac, state = client
    entry = {
        "dsl": _genome_dsl("genome_short1m", 30),
        "chromosome": {"direction": "short"},
        "score": 1.0,
        "cross_window": {
            "windows_passed": 1, "total_windows": 1, "passed": True,
            "windows": [
                {"sharpe": 1.2, "return_pct": 0.05, "trades": 40,
                 "offset_months": 1, "dates": ["2024-02-01", "2024-10-19"]},
            ],
        },
    }
    run_id = _create_discovery_run_with_results(
        state, timeframe="1m", symbols=["BTCUSDT"], top_strategies=[entry],
        notes_extra={
            "cross_window_months": [1],
            "train_dates": ["2024-01-01", "2024-10-19"],
        },
    )
    seen: list[tuple[str, str]] = []

    def neutral_base_bear_window(_s: str, start: str, end: str) -> int:
        seen.append((start, end))
        return 0 if start == "2024-01-01" else -1

    monkeypatch.setattr(
        "vibe_quant.api.routers.discovery._window_regime_sign", neutral_base_bear_window,
    )
    r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r.status_code == 409
    # base = TRAIN range, shifted window = its persisted dates
    assert seen == [("2024-01-01", "2024-10-19"), ("2024-02-01", "2024-10-19")]

    monkeypatch.setattr(
        "vibe_quant.api.routers.discovery._window_regime_sign",
        lambda _s, start, _e: 0 if start == "2024-01-01" else 1,
    )
    r_ok = await ac.post(f"/api/discovery/results/{run_id}/promote/0")
    assert r_ok.status_code == 201


async def test_launch_defaults_to_holdout_and_passes_explicit_zero(
    client: tuple[AsyncClient, StateManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ac, _state = client
    commands: list[list[str]] = []

    def fake_start_job(
        self: object, run_id: int, job_type: str, command: list[str], **_: object
    ) -> int:
        commands.append(command)
        return 4242

    monkeypatch.setattr(BacktestJobManager, "start_job", fake_start_job)
    await ac.post("/api/discovery/launch", json={})
    await ac.post("/api/discovery/launch", json={"train_test_split": 0})
    assert len(commands) == 2
    i0 = commands[0].index("--train-test-split")
    assert commands[0][i0 + 1] == "0.8"
    i1 = commands[1].index("--train-test-split")
    assert commands[1][i1 + 1] == "0.0"
