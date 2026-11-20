"""Tests for vibe_quant.utils.log_dir and the job-log leak fix.

regression: vibe-quant-yul7u.21
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from pathlib import Path


async def test_api_launch_writes_job_log_under_env_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Launching screening + validation via the API writes logs under env dir.

    subprocess.Popen is patched, but BacktestJobManager.start_job is REAL — it
    opens the log file — so the assertion only passes if the router resolves the
    log path through ``log_dir()`` (a hardcoded ``logs/...`` would leak into the
    repo / cwd instead).
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(tmp_path / "joblogs"))

    db = tmp_path / "api.db"
    app = create_app()
    state = StateManager(db)
    jobs = BacktestJobManager(db)
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.catalog_manager = CatalogManager()
    app.state.ws_manager = ConnectionManager()

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            sid = state.create_strategy("guard", {"name": "guard"})
            body = {
                "strategy_id": sid,
                "symbols": ["BTCUSDT"],
                "timeframe": "4h",
                "start_date": "2025-01-01",
                "end_date": "2025-06-01",
                "parameters": {"leverage": 5, "initial_balance": 1000},
            }
            with patch("subprocess.Popen") as popen:
                popen.return_value.pid = 4242
                screening = await ac.post("/api/backtest/screening", json=body)
                validation = await ac.post("/api/backtest/validation", json=body)
    finally:
        jobs.close()
        state.close()

    assert screening.status_code == 201, screening.text
    assert validation.status_code == 201, validation.text

    joblogs = tmp_path / "joblogs"
    assert list(joblogs.glob("screening_*.log")), "screening job log missing from env dir"
    assert list(joblogs.glob("validation_*.log")), "validation job log missing from env dir"
    assert not (tmp_path / "logs").exists(), "repo logs/ must not be created"


def _write_one_trade_log(directory: Path, run_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{run_id}.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "POSITION_OPEN",
                        "ts": "2025-01-01T00:00:00+00:00",
                        "data": {
                            "position_id": "p1",
                            "symbol": "BTCUSDT",
                            "side": "LONG",
                            "entry_price": 100.0,
                            "quantity": 1.0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "POSITION_CLOSE",
                        "ts": "2025-01-01T01:00:00+00:00",
                        "data": {
                            "position_id": "p1",
                            "exit_price": 110.0,
                            "net_pnl": 10.0,
                            "gross_pnl": 10.0,
                            "exit_reason": "signal",
                        },
                    }
                ),
            ]
        )
    )


def test_load_trades_default_follows_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``load_trades`` with no base_path reads ``VIBE_QUANT_LOG_DIR/events``.

    Before this, ``reconciliation`` imported the frozen ``_DEFAULT_BASE_PATH``
    constant, so a direct caller ignored the env var the writer/reader honor.
    """
    from vibe_quant import reconciliation as recon_module

    target = tmp_path / "env-logs"
    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(target))
    _write_one_trade_log(target / "events", "ENV-RUN")

    trades = recon_module.load_trades("ENV-RUN")
    assert len(trades) == 1
    assert trades[0].position_id == "p1"
    assert trades[0].net_pnl == 10.0


def test_load_trades_default_unset_keeps_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vibe_quant import reconciliation as recon_module
    from vibe_quant.logging import query as query_module

    monkeypatch.delenv("VIBE_QUANT_LOG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        recon_module.load_trades("NOPE-UNSET")

    assert str(query_module._DEFAULT_BASE_PATH / "NOPE-UNSET.jsonl") in str(excinfo.value)
