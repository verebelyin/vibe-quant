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
