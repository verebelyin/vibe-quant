"""Tests for /api/reconciliation/{paper_session_id}."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TCH003
from unittest.mock import patch

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
def events_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the reconciliation event-log directory to a tmp dir."""
    d = tmp_path / "events"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("vibe_quant.logging.query._DEFAULT_BASE_PATH", d)
    monkeypatch.setattr("vibe_quant.reconciliation._DEFAULT_BASE_PATH", d)
    return d


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
        yield ac, state_mgr, job_mgr

    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


def _ensure_strategy(state: StateManager, strategy_id: int) -> None:
    """Create a minimal strategy row so FK constraints succeed."""
    row = state.conn.execute("SELECT id FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
    if row is not None:
        return
    state.conn.execute(
        """INSERT INTO strategies (id, name, dsl_config, version)
           VALUES (?, ?, '{}', 1)""",
        (strategy_id, f"strat_{strategy_id}"),
    )
    state.conn.commit()


def _insert_stopped_paper_job(
    state: StateManager,
    job_mgr: BacktestJobManager,
    strategy_id: int = 1,
) -> int:
    _ensure_strategy(state, strategy_id)
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="paper",
        symbols=[],
        timeframe="1m",
        start_date="",
        end_date="",
        parameters={},
    )
    job_mgr.conn.execute(
        """INSERT INTO background_jobs
           (run_id, pid, job_type, status, started_at, heartbeat_at)
           VALUES (?, ?, 'paper', 'completed', datetime('now'), datetime('now'))""",
        (run_id, 99000),
    )
    job_mgr.conn.commit()
    return run_id


def _insert_running_paper_job(
    state: StateManager,
    job_mgr: BacktestJobManager,
    strategy_id: int = 1,
) -> int:
    _ensure_strategy(state, strategy_id)
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="paper",
        symbols=[],
        timeframe="1m",
        start_date="",
        end_date="",
        parameters={},
    )
    job_mgr.conn.execute(
        """INSERT INTO background_jobs
           (run_id, pid, job_type, status, started_at, heartbeat_at)
           VALUES (?, ?, 'paper', 'running', datetime('now'), datetime('now'))""",
        (run_id, 99001),
    )
    job_mgr.conn.commit()
    return run_id


def _insert_validation_run(
    state: StateManager,
    strategy_id: int = 1,
) -> int:
    _ensure_strategy(state, strategy_id)
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1m",
        start_date="2025-01-01",
        end_date="2025-01-02",
        parameters={},
    )
    state.update_backtest_run_status(run_id, "completed")
    state.save_backtest_result(
        run_id,
        {
            "total_return": 0.1,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.05,
            "profit_factor": 1.8,
            "total_trades": 1,
        },
    )
    return run_id


def _write_event_log(
    events_dir: Path,
    run_label: str,
    *,
    entry_ts: str,
    exit_ts: str,
    symbol: str = "BTCUSDT",
    side: str = "LONG",
    entry_price: float = 50000.0,
    exit_price: float = 51000.0,
    quantity: float = 0.1,
    net_pnl: float = 100.0,
) -> None:
    """Write a minimal open/close event pair to logs/events/<label>.jsonl."""
    path = events_dir / f"{run_label}.jsonl"
    with path.open("w") as f:
        f.write(
            json.dumps(
                {
                    "ts": entry_ts,
                    "event": "POSITION_OPEN",
                    "data": {
                        "position_id": "p1",
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry_price,
                        "quantity": quantity,
                    },
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "ts": exit_ts,
                    "event": "POSITION_CLOSE",
                    "data": {
                        "position_id": "p1",
                        "symbol": symbol,
                        "exit_price": exit_price,
                        "net_pnl": net_pnl,
                        "gross_pnl": net_pnl + 2.0,
                        "exit_reason": "signal",
                    },
                }
            )
            + "\n"
        )


async def test_reconcile_400_when_session_running(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, state, job_mgr = client
    with patch.object(job_mgr, "is_process_alive", return_value=True):
        paper_id = _insert_running_paper_job(state, job_mgr)
        r = await ac.get(f"/api/reconciliation/{paper_id}")
    assert r.status_code == 400
    assert "still running" in r.json()["detail"]


async def test_reconcile_404_when_session_missing(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, _state, _jobs = client
    r = await ac.get("/api/reconciliation/9999")
    assert r.status_code == 404


async def test_reconcile_404_when_no_validation_run(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    ac, state, job_mgr = client
    paper_id = _insert_stopped_paper_job(state, job_mgr, strategy_id=7)
    r = await ac.get(f"/api/reconciliation/{paper_id}")
    assert r.status_code == 404
    assert "No completed validation run" in r.json()["detail"]


async def test_reconcile_happy_path(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    events_dir: Path,
) -> None:
    ac, state, job_mgr = client
    paper_id = _insert_stopped_paper_job(state, job_mgr, strategy_id=5)
    val_id = _insert_validation_run(state, strategy_id=5)

    entry = "2026-01-01T00:00:00+00:00"
    exit_ = "2026-01-01T00:10:00+00:00"
    _write_event_log(
        events_dir,
        f"paper_{paper_id}",
        entry_ts=entry,
        exit_ts=exit_,
        entry_price=50010.0,
        net_pnl=95.0,
    )
    _write_event_log(
        events_dir,
        str(val_id),
        entry_ts=entry,
        exit_ts=exit_,
        entry_price=50000.0,
        net_pnl=100.0,
    )

    r = await ac.get(f"/api/reconciliation/{paper_id}")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["paper_run"] == f"paper_{paper_id}"
    assert data["validation_run"] == str(val_id)
    assert len(data["paired_trades"]) == 1
    pair = data["paired_trades"][0]
    assert pair["entry_slippage"] == pytest.approx(10.0)
    assert pair["pnl_delta"] == pytest.approx(-5.0)
    assert data["divergence_summary"]["matched"] == 1
    assert data["divergence_summary"]["parity_rate"] == pytest.approx(1.0)
