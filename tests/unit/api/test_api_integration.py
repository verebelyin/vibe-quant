"""Integration tests for the FastAPI API endpoints."""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
async def client(tmp_db: Path):
    """Create app with temp DB, manually wire up lifespan deps."""
    app = create_app()

    state_mgr = StateManager(db_path=tmp_db)
    # Force schema init
    _ = state_mgr.conn

    job_mgr = BacktestJobManager()
    catalog_mgr = CatalogManager()
    ws_mgr = ConnectionManager()
    await ws_mgr.start()

    app.state.state_manager = state_mgr
    app.state.job_manager = job_mgr
    app.state.catalog_manager = catalog_mgr
    app.state.ws_manager = ws_mgr

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Strategies CRUD
# ---------------------------------------------------------------------------

_STRATEGY_BODY = {
    "name": "test_sma_cross",
    "dsl_config": {"entry": "sma_cross", "exit": "trailing_stop"},
    "description": "test strategy",
    "strategy_type": "technical",
}


async def test_create_strategy(client: AsyncClient) -> None:
    r = await client.post("/api/strategies", json=_STRATEGY_BODY)
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "test_sma_cross"
    assert data["id"] >= 1


async def test_list_strategies(client: AsyncClient) -> None:
    await client.post("/api/strategies", json=_STRATEGY_BODY)
    r = await client.get("/api/strategies")
    assert r.status_code == 200
    data = r.json()
    assert "strategies" in data
    assert len(data["strategies"]) >= 1


async def test_get_strategy(client: AsyncClient) -> None:
    cr = await client.post("/api/strategies", json=_STRATEGY_BODY)
    sid = cr.json()["id"]
    r = await client.get(f"/api/strategies/{sid}")
    assert r.status_code == 200
    assert r.json()["id"] == sid


async def test_get_strategy_not_found(client: AsyncClient) -> None:
    r = await client.get("/api/strategies/999999")
    assert r.status_code == 404


async def test_update_strategy(client: AsyncClient) -> None:
    cr = await client.post("/api/strategies", json=_STRATEGY_BODY)
    sid = cr.json()["id"]
    r = await client.put(
        f"/api/strategies/{sid}",
        json={"description": "updated desc"},
    )
    assert r.status_code == 200
    assert r.json()["description"] == "updated desc"


async def test_delete_strategy(client: AsyncClient) -> None:
    cr = await client.post("/api/strategies", json=_STRATEGY_BODY)
    sid = cr.json()["id"]
    r = await client.delete(f"/api/strategies/{sid}")
    assert r.status_code == 204


async def test_validate_strategy(client: AsyncClient) -> None:
    # Use a valid DSL config for validation (the default _STRATEGY_BODY has
    # a placeholder dsl_config that doesn't pass schema validation)
    valid_body = {
        "name": "test_rsi_valid",
        "description": "test strategy with valid DSL",
        "strategy_type": "technical",
        "dsl_config": {
            "name": "test_rsi_valid",
            "timeframe": "4h",
            "indicators": {"rsi1": {"type": "RSI", "period": 14}},
            "entry_conditions": {"long": ["rsi1 < 30"]},
            "exit_conditions": {"long": ["rsi1 > 70"]},
            "stop_loss": {"type": "fixed_pct", "percent": 5.0},
            "take_profit": {"type": "fixed_pct", "percent": 10.0},
        },
    }
    cr = await client.post("/api/strategies", json=valid_body)
    sid = cr.json()["id"]
    r = await client.post(f"/api/strategies/{sid}/validate")
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is True
    assert data["errors"] == []


async def test_list_templates(client: AsyncClient) -> None:
    r = await client.get("/api/strategies/templates")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

_SIZING_BODY = {
    "name": "test_sizing",
    "method": "fixed_fraction",
    "config": {"fraction": 0.02},
}

_RISK_BODY = {
    "name": "test_risk",
    "strategy_level": {"max_drawdown": 0.15},
    "portfolio_level": {"max_correlation": 0.7},
}


async def test_list_sizing_configs(client: AsyncClient) -> None:
    r = await client.get("/api/settings/sizing")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


async def test_create_sizing_config(client: AsyncClient) -> None:
    r = await client.post("/api/settings/sizing", json=_SIZING_BODY)
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "test_sizing"
    assert data["id"] >= 1


async def test_list_risk_configs(client: AsyncClient) -> None:
    r = await client.get("/api/settings/risk")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


async def test_create_risk_config(client: AsyncClient) -> None:
    r = await client.post("/api/settings/risk", json=_RISK_BODY)
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "test_risk"
    assert data["id"] >= 1


async def test_latency_presets(client: AsyncClient) -> None:
    r = await client.get("/api/settings/latency-presets")
    assert r.status_code == 200
    presets = r.json()
    assert len(presets) == 4
    names = {p["name"] for p in presets}
    assert "co_located" in names
    assert "retail" in names


async def test_system_info(client: AsyncClient) -> None:
    r = await client.get("/api/settings/system-info")
    assert r.status_code == 200
    data = r.json()
    assert "nt_version" in data
    assert "python_version" in data
    assert "table_counts" in data


async def test_database_info(client: AsyncClient) -> None:
    r = await client.get("/api/settings/database")
    assert r.status_code == 200
    data = r.json()
    assert "path" in data
    assert "tables" in data
    assert isinstance(data["tables"], list)


# ---------------------------------------------------------------------------
# Backtest / Jobs
# ---------------------------------------------------------------------------


async def test_list_jobs(client: AsyncClient) -> None:
    r = await client.get("/api/backtest/jobs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


async def test_cleanup_stale(client: AsyncClient) -> None:
    r = await client.post("/api/backtest/jobs/cleanup-stale")
    assert r.status_code == 200
    data = r.json()
    assert "cleaned" in data


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


async def test_list_runs(client: AsyncClient) -> None:
    r = await client.get("/api/results/runs")
    assert r.status_code == 200
    data = r.json()
    assert "runs" in data
    assert isinstance(data["runs"], list)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


async def test_data_symbols(client: AsyncClient) -> None:
    r = await client.get("/api/data/symbols")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


async def test_indicator_pool(client: AsyncClient) -> None:
    r = await client.get("/api/discovery/indicator-pool")
    assert r.status_code == 200
    pool = r.json()
    assert isinstance(pool, list)
    assert len(pool) > 0
    assert "name" in pool[0]


async def test_discovery_jobs(client: AsyncClient) -> None:
    r = await client.get("/api/discovery/jobs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ---------------------------------------------------------------------------
# Paper Trading
# ---------------------------------------------------------------------------


async def test_paper_status(client: AsyncClient) -> None:
    r = await client.get("/api/paper/status")
    assert r.status_code == 200
    data = r.json()
    assert "state" in data


async def test_paper_positions(client: AsyncClient) -> None:
    r = await client.get("/api/paper/positions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
