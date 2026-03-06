"""Regression tests for API-level bug fixes."""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.schemas.data import DataQualityResponse, IngestRequest
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


@pytest.fixture()
def state_mgr(tmp_db: Path) -> StateManager:
    """Bare StateManager for direct DB manipulation in tests."""
    mgr = StateManager(db_path=tmp_db)
    _ = mgr.conn
    return mgr


_STRATEGY_BODY = {
    "name": "regression_test",
    "dsl_config": {"entry": "sma_cross", "exit": "trailing_stop"},
    "description": "regression test strategy",
    "strategy_type": "technical",
}


# ---------------------------------------------------------------------------
# 1. Strategy name update persisted (bd-wt64)
# ---------------------------------------------------------------------------


async def test_strategy_name_update_persisted(client: AsyncClient) -> None:
    """PUT /api/strategies/{id} with new name must persist through GET."""
    cr = await client.post("/api/strategies", json=_STRATEGY_BODY)
    assert cr.status_code == 201
    sid = cr.json()["id"]

    r = await client.put(f"/api/strategies/{sid}", json={"name": "new_name"})
    assert r.status_code == 200
    assert r.json()["name"] == "new_name"

    # Re-fetch and verify persistence
    g = await client.get(f"/api/strategies/{sid}")
    assert g.status_code == 200
    assert g.json()["name"] == "new_name"


# ---------------------------------------------------------------------------
# 2. end_date filter includes runs on that date (bd-pyng)
# ---------------------------------------------------------------------------


async def test_end_date_filter_includes_runs_on_that_date(
    client: AsyncClient, state_mgr: StateManager
) -> None:
    """GET /api/results/runs?end_date=<date> must include runs created on that date."""
    # Create a strategy first
    cr = await client.post("/api/strategies", json=_STRATEGY_BODY)
    sid = cr.json()["id"]

    # Create a backtest run directly via state manager
    run_id = state_mgr.create_backtest_run(
        strategy_id=sid,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2025-01-01",
        end_date="2025-06-01",
        parameters={},
    )
    assert run_id > 0

    # Get the created_at date from the run
    run = state_mgr.get_backtest_run(run_id)
    assert run is not None
    created_date = run["created_at"][:10]  # "YYYY-MM-DD"

    # Filter by end_date = created_date; the run should be included
    r = await client.get(f"/api/results/runs?end_date={created_date}")
    assert r.status_code == 200
    run_ids = [rr["id"] for rr in r.json()["runs"]]
    assert run_id in run_ids


# ---------------------------------------------------------------------------
# 3. Equity curve uses actual starting balance (bd-bzp6)
# ---------------------------------------------------------------------------


async def test_equity_curve_uses_actual_starting_balance(
    client: AsyncClient, state_mgr: StateManager
) -> None:
    """Equity curve must start from the run's starting_balance, not hardcoded 10k."""
    cr = await client.post("/api/strategies", json=_STRATEGY_BODY)
    sid = cr.json()["id"]

    run_id = state_mgr.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["ETHUSDT"],
        timeframe="4h",
        start_date="2025-01-01",
        end_date="2025-06-01",
        parameters={},
    )

    # Save a backtest result with specific starting_balance
    state_mgr.save_backtest_result(run_id, {"starting_balance": 50_000.0})

    # Save a trade so equity curve has data
    state_mgr.save_trade(run_id, {
        "symbol": "ETHUSDT",
        "direction": "long",
        "leverage": 1,
        "entry_time": "2025-02-01T00:00:00",
        "exit_time": "2025-02-02T00:00:00",
        "entry_price": 3000.0,
        "exit_price": 3100.0,
        "quantity": 1.0,
        "net_pnl": 100.0,
    })

    r = await client.get(f"/api/results/runs/{run_id}/equity-curve")
    assert r.status_code == 200
    points = r.json()
    assert len(points) >= 2
    # First point should be the starting balance
    assert points[0]["equity"] == 50_000.0
    # Second point = starting_balance + net_pnl
    assert points[1]["equity"] == 50_100.0


# ---------------------------------------------------------------------------
# 4. Data quality response can have null score (bd-pafy)
# ---------------------------------------------------------------------------


def test_data_quality_response_allows_null_score() -> None:
    """DataQualityResponse schema must accept quality_score=None."""
    resp = DataQualityResponse(
        symbol="BTCUSDT",
        gaps=[],
        quality_score=None,
    )
    assert resp.quality_score is None


def test_data_quality_response_accepts_float_score() -> None:
    """DataQualityResponse schema must also accept a float score."""
    resp = DataQualityResponse(
        symbol="BTCUSDT",
        gaps=[],
        quality_score=0.95,
    )
    assert resp.quality_score == 0.95


# ---------------------------------------------------------------------------
# 5. IngestForm interval field accepted (bd-i7iy)
# ---------------------------------------------------------------------------


def test_ingest_request_accepts_interval_field() -> None:
    """IngestRequest schema must accept the interval field."""
    req = IngestRequest(
        symbols=["BTCUSDT"],
        start_date="2025-01-01",
        end_date="2025-06-01",
        interval="5m",
    )
    assert req.interval == "5m"


def test_ingest_request_interval_defaults_to_1m() -> None:
    """IngestRequest interval should default to '1m'."""
    req = IngestRequest(
        symbols=["BTCUSDT"],
        start_date="2025-01-01",
        end_date="2025-06-01",
    )
    assert req.interval == "1m"


# ---------------------------------------------------------------------------
# 6. CatalogManager DI in backtest router (bd-r2me)
# ---------------------------------------------------------------------------


async def test_validate_coverage_endpoint_smoke(client: AsyncClient) -> None:
    """POST /api/backtest/validate-coverage must not crash (DI wired correctly)."""
    r = await client.post(
        "/api/backtest/validate-coverage",
        json={
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "start_date": "2025-01-01",
            "end_date": "2025-06-01",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "coverage" in data
    assert "BTCUSDT" in data["coverage"]


# ---------------------------------------------------------------------------
# 7. Settings table counts uses whitelist (bd-zyry)
# ---------------------------------------------------------------------------


async def test_system_info_table_counts_no_error(client: AsyncClient) -> None:
    """GET /api/settings/system-info must return without error (whitelist-safe)."""
    r = await client.get("/api/settings/system-info")
    assert r.status_code == 200
    data = r.json()
    assert "table_counts" in data
    counts = data["table_counts"]
    # All returned tables must be from the known whitelist
    known = {
        "strategies",
        "sizing_configs",
        "risk_configs",
        "backtest_runs",
        "backtest_results",
        "trades",
        "sweep_results",
        "background_jobs",
        "consistency_checks",
    }
    for table_name in counts:
        assert table_name in known, f"Unexpected table '{table_name}' outside whitelist"
