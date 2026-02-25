"""Backtest launch & job management router."""

from __future__ import annotations

import logging
import sys
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from vibe_quant.api.deps import get_job_manager, get_state_manager, get_ws_manager
from vibe_quant.api.schemas.backtest import (
    BacktestLaunchRequest,
    BacktestRunResponse,
    CoverageCheckRequest,
    CoverageCheckResponse,
    JobStatusResponse,
)
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]
JobMgr = Annotated[BacktestJobManager, Depends(get_job_manager)]
WsMgr = Annotated[ConnectionManager, Depends(get_ws_manager)]


def _job_info_to_response(info: object) -> JobStatusResponse:
    from vibe_quant.jobs.manager import JobInfo

    if not isinstance(info, JobInfo):
        raise TypeError(f"Expected JobInfo, got {type(info).__name__}")
    return JobStatusResponse(
        run_id=info.run_id,
        pid=info.pid,
        job_type=info.job_type,
        status=info.status.value,
        heartbeat_at=info.heartbeat_at.isoformat() if info.heartbeat_at else None,
        started_at=info.started_at.isoformat() if info.started_at else None,
        completed_at=info.completed_at.isoformat() if info.completed_at else None,
        is_stale=info.is_stale,
    )


def _run_to_response(row: dict[str, object]) -> BacktestRunResponse:
    import json

    symbols = row["symbols"]
    if isinstance(symbols, str):
        symbols = json.loads(symbols)
    params = row["parameters"]
    if isinstance(params, str):
        params = json.loads(params)
    return BacktestRunResponse(
        id=row["id"],  # type: ignore[arg-type]
        strategy_id=row["strategy_id"],  # type: ignore[arg-type]
        run_mode=row["run_mode"],  # type: ignore[arg-type]
        symbols=symbols,  # type: ignore[arg-type]
        timeframe=row["timeframe"],  # type: ignore[arg-type]
        start_date=row["start_date"],  # type: ignore[arg-type]
        end_date=row["end_date"],  # type: ignore[arg-type]
        parameters=params,  # type: ignore[arg-type]
        status=row["status"],  # type: ignore[arg-type]
        started_at=row.get("started_at"),  # type: ignore[arg-type]
        completed_at=row.get("completed_at"),  # type: ignore[arg-type]
        error_message=row.get("error_message"),  # type: ignore[arg-type]
        created_at=row["created_at"],  # type: ignore[arg-type]
    )


# --- Launch endpoints ---


@router.post("/screening", response_model=BacktestRunResponse, status_code=201)
async def launch_screening(
    body: BacktestLaunchRequest,
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
) -> BacktestRunResponse:
    run_id = state.create_backtest_run(
        strategy_id=body.strategy_id,
        run_mode="screening",
        symbols=body.symbols,
        timeframe=body.timeframe,
        start_date=body.start_date,
        end_date=body.end_date,
        parameters=body.parameters,
        sizing_config_id=body.sizing_config_id,
        risk_config_id=body.risk_config_id,
        latency_preset=body.latency_preset,
    )

    log_file = f"logs/screening_{run_id}.log"
    command = [
        sys.executable,
        "-m",
        "vibe_quant",
        "screening",
        "run",
        "--run-id",
        str(run_id),
    ]

    try:
        pid = jobs.start_job(run_id, "screening", command, log_file=log_file)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    state.update_backtest_run_status(run_id, "running", pid=pid)
    logger.info("screening job started run_id=%d pid=%d", run_id, pid)

    await ws.broadcast("jobs", {"type": "job_started", "run_id": run_id, "job_type": "screening"})

    row = state.get_backtest_run(run_id)
    if row is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Run disappeared after creation")
    return _run_to_response(row)


@router.post("/validation", response_model=BacktestRunResponse, status_code=201)
async def launch_validation(
    body: BacktestLaunchRequest,
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
) -> BacktestRunResponse:
    run_id = state.create_backtest_run(
        strategy_id=body.strategy_id,
        run_mode="validation",
        symbols=body.symbols,
        timeframe=body.timeframe,
        start_date=body.start_date,
        end_date=body.end_date,
        parameters=body.parameters,
        sizing_config_id=body.sizing_config_id,
        risk_config_id=body.risk_config_id,
        latency_preset=body.latency_preset,
    )

    log_file = f"logs/validation_{run_id}.log"
    command = [
        sys.executable,
        "-m",
        "vibe_quant",
        "validation",
        "run",
        "--run-id",
        str(run_id),
    ]

    try:
        pid = jobs.start_job(run_id, "validation", command, log_file=log_file)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    state.update_backtest_run_status(run_id, "running", pid=pid)
    logger.info("validation job started run_id=%d pid=%d", run_id, pid)

    await ws.broadcast("jobs", {"type": "job_started", "run_id": run_id, "job_type": "validation"})

    row = state.get_backtest_run(run_id)
    if row is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Run disappeared after creation")
    return _run_to_response(row)


# --- Job management endpoints ---


@router.get("/jobs", response_model=list[JobStatusResponse])
async def list_jobs(jobs: JobMgr) -> list[JobStatusResponse]:
    active = jobs.list_active_jobs()
    return [_job_info_to_response(j) for j in active]


@router.get("/jobs/{run_id}", response_model=JobStatusResponse)
async def get_job(run_id: int, jobs: JobMgr) -> JobStatusResponse:
    info = jobs.get_job_info(run_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_info_to_response(info)


@router.delete("/jobs/{run_id}", status_code=204)
async def kill_job(run_id: int, jobs: JobMgr, ws: WsMgr) -> None:
    killed = jobs.kill_job(run_id)
    if not killed:
        raise HTTPException(status_code=404, detail="Job not found or not running")
    logger.info("job killed run_id=%d", run_id)
    await ws.broadcast("jobs", {"type": "job_killed", "run_id": run_id})


@router.post("/jobs/{run_id}/sync", response_model=JobStatusResponse)
async def sync_job(run_id: int, jobs: JobMgr, ws: WsMgr) -> JobStatusResponse:
    status = jobs.sync_job_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")

    await ws.broadcast("jobs", {"type": "job_synced", "run_id": run_id, "status": status.value})

    info = jobs.get_job_info(run_id)
    if info is None:  # pragma: no cover
        raise HTTPException(status_code=404, detail="Job disappeared after sync")
    return _job_info_to_response(info)


# --- Utility endpoints ---


@router.post("/validate-coverage", response_model=CoverageCheckResponse)
async def validate_coverage(body: CoverageCheckRequest) -> CoverageCheckResponse:
    from datetime import UTC, datetime

    from vibe_quant.data.catalog import CatalogManager

    try:
        req_start = datetime.fromisoformat(body.start_date).replace(tzinfo=UTC)
        req_end = datetime.fromisoformat(body.end_date).replace(tzinfo=UTC)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date format; use YYYY-MM-DD") from exc

    catalog = CatalogManager()
    coverage: dict[str, object] = {}

    for symbol in body.symbols:
        try:
            bars = catalog.get_bars(symbol, body.timeframe, start=req_start, end=req_end)
        except Exception:
            coverage[symbol] = {
                "has_data": False,
                "start_date": body.start_date,
                "end_date": body.end_date,
                "bars": 0,
                "message": "Symbol or timeframe not available in catalog",
            }
            continue

        if not bars:
            date_range = catalog.get_bar_date_range(symbol, body.timeframe)
            if date_range:
                avail_start, avail_end = date_range
                message = (
                    f"No data in requested range; catalog has "
                    f"{avail_start.date()} to {avail_end.date()}"
                )
            else:
                message = "No data for this symbol/timeframe in catalog"
            coverage[symbol] = {
                "has_data": False,
                "start_date": body.start_date,
                "end_date": body.end_date,
                "bars": 0,
                "message": message,
            }
        else:
            actual_start = datetime.fromtimestamp(bars[0].ts_event / 1e9, tz=UTC)
            actual_end = datetime.fromtimestamp(bars[-1].ts_event / 1e9, tz=UTC)
            coverage[symbol] = {
                "has_data": True,
                "start_date": actual_start.date().isoformat(),
                "end_date": actual_end.date().isoformat(),
                "bars": len(bars),
                "message": f"{len(bars):,} bars available",
            }

    return CoverageCheckResponse(coverage=coverage)


@router.post("/jobs/cleanup-stale")
async def cleanup_stale_jobs(jobs: JobMgr, ws: WsMgr) -> dict[str, int]:
    count = jobs.cleanup_stale_jobs()
    if count > 0:
        logger.info("cleaned up %d stale jobs", count)
        await ws.broadcast("jobs", {"type": "stale_cleanup", "count": count})
    return {"cleaned": count}
