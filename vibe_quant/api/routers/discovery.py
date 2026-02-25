"""Discovery router — GA strategy discovery launch & management."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException

from vibe_quant.api.deps import get_job_manager, get_state_manager, get_ws_manager
from vibe_quant.api.schemas.discovery import (
    DiscoveryJobResponse,
    DiscoveryLaunchRequest,
    DiscoveryResultResponse,
)
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from vibe_quant.jobs.manager import JobInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/discovery", tags=["discovery"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]
JobMgr = Annotated[BacktestJobManager, Depends(get_job_manager)]
WsMgr = Annotated[ConnectionManager, Depends(get_ws_manager)]

# Hardcoded indicator pool; replaced by DSL catalog when available
_DEFAULT_INDICATOR_POOL: list[dict[str, object]] = [
    {"name": "SMA", "params": {"period": {"min": 5, "max": 200, "step": 5}}},
    {"name": "EMA", "params": {"period": {"min": 5, "max": 200, "step": 5}}},
    {"name": "RSI", "params": {"period": {"min": 7, "max": 28, "step": 1}}},
    {"name": "MACD", "params": {"fast": {"min": 8, "max": 21}, "slow": {"min": 21, "max": 55}, "signal": {"min": 5, "max": 13}}},
    {"name": "BollingerBands", "params": {"period": {"min": 10, "max": 50, "step": 5}, "std_dev": {"min": 1.5, "max": 3.0, "step": 0.25}}},
    {"name": "ATR", "params": {"period": {"min": 7, "max": 28, "step": 1}}},
    {"name": "Stochastic", "params": {"k_period": {"min": 5, "max": 21}, "d_period": {"min": 3, "max": 9}}},
    {"name": "ADX", "params": {"period": {"min": 7, "max": 28, "step": 1}}},
    {"name": "CCI", "params": {"period": {"min": 10, "max": 40, "step": 5}}},
    {"name": "VWAP", "params": {}},
]


def _read_progress_file(run_id: int) -> dict[str, object] | None:
    """Read progress JSON written by the discovery pipeline subprocess."""
    import json
    from pathlib import Path

    path = Path(f"logs/discovery_{run_id}_progress.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())  # type: ignore[return-value]
    except (json.JSONDecodeError, OSError):
        return None


def _job_info_to_discovery_response(info: JobInfo) -> DiscoveryJobResponse:
    progress = _read_progress_file(info.run_id)
    return DiscoveryJobResponse(
        run_id=info.run_id,
        status=info.status.value,
        started_at=info.started_at.isoformat() if info.started_at else None,
        progress=progress,
    )


# --- Launch ---


@router.post("/launch", response_model=DiscoveryJobResponse, status_code=201)
async def launch_discovery(
    body: DiscoveryLaunchRequest,
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
) -> DiscoveryJobResponse:
    params: dict[str, object] = {
        "population": body.population,
        "generations": body.generations,
        "mutation_rate": body.mutation_rate,
        "crossover_rate": body.crossover_rate,
        "elite_count": body.elite_count,
        "tournament_size": body.tournament_size,
        "convergence_generations": body.convergence_generations,
    }
    if body.indicator_pool is not None:
        params["indicator_pool"] = body.indicator_pool

    # Use strategy_id=None for discovery (no pre-existing strategy)
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=body.symbols,
        timeframe=body.timeframes[0] if body.timeframes else "1h",
        start_date=body.start_date or "",
        end_date=body.end_date or "",
        parameters=params,
    )

    log_file = f"logs/discovery_{run_id}.log"
    symbols_str = ",".join(body.symbols)
    timeframe = body.timeframes[0] if body.timeframes else "4h"
    start_date = body.start_date or "2024-01-01"
    end_date = body.end_date or "2026-02-24"
    command = [
        sys.executable,
        "-m",
        "vibe_quant",
        "discovery",
        "--run-id",
        str(run_id),
        "--population-size",
        str(body.population),
        "--max-generations",
        str(body.generations),
        "--mutation-rate",
        str(body.mutation_rate),
        "--crossover-rate",
        str(body.crossover_rate),
        "--elite-count",
        str(body.elite_count),
        "--tournament-size",
        str(body.tournament_size),
        "--convergence-generations",
        str(body.convergence_generations),
        "--max-workers",
        "0",
        "--symbols",
        symbols_str,
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
    ]
    if body.indicator_pool is not None:
        command.extend(["--indicator-pool", ",".join(body.indicator_pool)])

    try:
        pid = jobs.start_job(run_id, "discovery", command, log_file=log_file)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    state.update_backtest_run_status(run_id, "running", pid=pid)
    logger.info("discovery job started run_id=%d pid=%d", run_id, pid)

    await ws.broadcast("jobs", {"type": "job_started", "run_id": run_id, "job_type": "discovery"})

    info = jobs.get_job_info(run_id)
    if info is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Job disappeared after creation")
    return _job_info_to_discovery_response(info)


# --- Job management ---


@router.get("/jobs", response_model=list[DiscoveryJobResponse])
async def list_discovery_jobs(jobs: JobMgr) -> list[DiscoveryJobResponse]:
    active = jobs.list_active_jobs()
    return [
        _job_info_to_discovery_response(j)
        for j in active
        if j.job_type == "discovery"
    ]


@router.delete("/jobs/{run_id}", status_code=204)
async def kill_discovery_job(run_id: int, jobs: JobMgr, ws: WsMgr) -> None:
    info = jobs.get_job_info(run_id)
    if info is None or info.job_type != "discovery":
        raise HTTPException(status_code=404, detail="Discovery job not found")
    killed = jobs.kill_job(run_id)
    if not killed:
        raise HTTPException(status_code=404, detail="Job not running")
    logger.info("discovery job killed run_id=%d", run_id)
    await ws.broadcast("jobs", {"type": "job_killed", "run_id": run_id})


# --- Results ---


def _load_discovery_strategies(state: StateManager, run_id: int) -> list[dict[str, object]]:
    """Load discovered strategies from backtest_results notes JSON."""
    import json

    result = state.get_backtest_result(run_id)
    if result is None:
        return []
    notes = result.get("notes", "")
    if not notes or not isinstance(notes, str):
        return []
    try:
        data = json.loads(notes)
        if isinstance(data, dict) and "top_strategies" in data:
            return data["top_strategies"]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


@router.get("/results/latest", response_model=DiscoveryResultResponse)
async def get_latest_results(state: StateMgr) -> DiscoveryResultResponse:
    # Find latest completed discovery run that has strategies with trades > 0
    conn = state.conn
    rows = conn.execute(
        "SELECT id FROM backtest_runs WHERE run_mode='discovery' AND status='completed' "
        "ORDER BY id DESC LIMIT 10"
    ).fetchall()
    for row in rows:
        strategies = _load_discovery_strategies(state, row[0])
        if strategies and any(s.get("trades", 0) > 0 for s in strategies):
            return DiscoveryResultResponse(strategies=strategies)
    # Fallback: return latest even if 0 trades
    if rows:
        return DiscoveryResultResponse(strategies=_load_discovery_strategies(state, rows[0][0]))
    return DiscoveryResultResponse(strategies=[])


@router.get("/results/{run_id}", response_model=DiscoveryResultResponse)
async def get_discovery_results(run_id: int, state: StateMgr) -> DiscoveryResultResponse:
    return DiscoveryResultResponse(strategies=_load_discovery_strategies(state, run_id))


@router.post("/results/{run_id}/export/{strategy_index}", status_code=201)
async def export_discovered_strategy(
    run_id: int,
    strategy_index: int,
    state: StateMgr,
) -> dict[str, object]:
    """Export a discovered strategy to the strategies table."""
    strategies = _load_discovery_strategies(state, run_id)
    if strategy_index < 0 or strategy_index >= len(strategies):
        raise HTTPException(status_code=404, detail="Strategy index out of range")

    entry = strategies[strategy_index]
    dsl = entry.get("dsl", {})
    name = dsl.get("name", f"discovery_{run_id}_{strategy_index}")

    # Check if strategy name already exists
    existing = state.conn.execute(
        "SELECT id FROM strategies WHERE name = ?", (name,)
    ).fetchone()
    if existing:
        return {"status": "exists", "strategy_id": existing[0], "name": name}

    import json

    cursor = state.conn.execute(
        "INSERT INTO strategies (name, description, dsl_config, strategy_type) VALUES (?, ?, ?, ?)",
        (
            name,
            f"Discovered via GA run {run_id} (score={entry.get('score', 0):.4f})",
            json.dumps(dsl),
            dsl.get("strategy_type", "momentum"),
        ),
    )
    state.conn.commit()
    return {"status": "created", "strategy_id": cursor.lastrowid, "name": name}


# --- Indicator pool ---


@router.get("/indicator-pool")
async def get_indicator_pool() -> list[dict[str, object]]:
    try:
        from vibe_quant.dsl.indicators import INDICATOR_CATALOG  # type: ignore[import-not-found]

        return INDICATOR_CATALOG  # type: ignore[return-value]
    except (ImportError, AttributeError):
        return _DEFAULT_INDICATOR_POOL
