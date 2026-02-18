"""Paper trading router (/api/paper)."""

from __future__ import annotations

import logging
import os
import signal
import sys
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from vibe_quant.api.deps import get_job_manager, get_state_manager, get_ws_manager
from vibe_quant.api.schemas.paper_trading import (
    CheckpointResponse,
    PaperPositionResponse,
    PaperStartRequest,
    PaperStatusResponse,
)
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/paper", tags=["paper"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]
JobMgr = Annotated[BacktestJobManager, Depends(get_job_manager)]
WsMgr = Annotated[ConnectionManager, Depends(get_ws_manager)]


def _find_active_paper_job(
    jobs: BacktestJobManager,
) -> tuple[int, int]:
    """Find running paper trading job. Returns (run_id, pid)."""
    active = jobs.list_active_jobs()
    for job in active:
        if job.job_type == "paper":
            return job.run_id, job.pid
    raise HTTPException(status_code=404, detail="No active paper trading session")


# --- Start / lifecycle ---


@router.post("/start", response_model=PaperStatusResponse, status_code=201)
async def start_paper(
    body: PaperStartRequest,
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
) -> PaperStatusResponse:
    params: dict[str, object] = {}
    if body.sizing_method is not None:
        params["sizing_method"] = body.sizing_method
    if body.max_leverage is not None:
        params["max_leverage"] = body.max_leverage
    if body.max_position_pct is not None:
        params["max_position_pct"] = body.max_position_pct
    if body.risk_per_trade is not None:
        params["risk_per_trade"] = body.risk_per_trade
    if body.max_drawdown_pct is not None:
        params["max_drawdown_pct"] = body.max_drawdown_pct
    if body.max_daily_loss_pct is not None:
        params["max_daily_loss_pct"] = body.max_daily_loss_pct
    if body.max_consecutive_losses is not None:
        params["max_consecutive_losses"] = body.max_consecutive_losses
    if body.max_position_count is not None:
        params["max_position_count"] = body.max_position_count

    run_id = state.create_backtest_run(
        strategy_id=body.strategy_id,
        run_mode="paper",
        symbols=[],
        timeframe="1m",
        start_date="",
        end_date="",
        parameters=params,
    )

    log_file = f"logs/paper_{run_id}.log"
    command = [
        sys.executable,
        "-m",
        "vibe_quant.paper.cli",
        "--strategy-id",
        str(body.strategy_id),
    ]
    if body.testnet:
        command.append("--testnet")

    try:
        pid = jobs.start_job(run_id, "paper", command, log_file=log_file)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    state.update_backtest_run_status(run_id, "running", pid=pid)
    logger.info("paper trading started run_id=%d pid=%d", run_id, pid)

    await ws.broadcast(
        "trading",
        {"type": "paper_started", "run_id": run_id, "strategy_id": body.strategy_id},
    )

    return PaperStatusResponse(state="running", pnl_metrics=None, trades_count=0)


@router.post("/halt", status_code=200)
async def halt_paper(jobs: JobMgr, ws: WsMgr) -> dict[str, str]:
    run_id, pid = _find_active_paper_job(jobs)
    try:
        os.kill(pid, signal.SIGUSR1)
    except ProcessLookupError as exc:
        raise HTTPException(status_code=410, detail="Process no longer running") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Signal failed: {exc}") from exc

    logger.info("paper trading halted run_id=%d pid=%d", run_id, pid)
    await ws.broadcast("trading", {"type": "paper_halted", "run_id": run_id})
    return {"status": "halted", "run_id": str(run_id)}


@router.post("/resume", status_code=200)
async def resume_paper(jobs: JobMgr, ws: WsMgr) -> dict[str, str]:
    run_id, pid = _find_active_paper_job(jobs)
    try:
        os.kill(pid, signal.SIGUSR2)
    except ProcessLookupError as exc:
        raise HTTPException(status_code=410, detail="Process no longer running") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Signal failed: {exc}") from exc

    logger.info("paper trading resumed run_id=%d pid=%d", run_id, pid)
    await ws.broadcast("trading", {"type": "paper_resumed", "run_id": run_id})
    return {"status": "resumed", "run_id": str(run_id)}


@router.post("/stop", status_code=200)
async def stop_paper(jobs: JobMgr, ws: WsMgr) -> dict[str, str]:
    run_id, pid = _find_active_paper_job(jobs)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # already dead, still mark killed
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Signal failed: {exc}") from exc

    logger.info("paper trading stopped run_id=%d pid=%d", run_id, pid)
    await ws.broadcast("trading", {"type": "paper_stopped", "run_id": run_id})
    return {"status": "stopped", "run_id": str(run_id)}


# --- Read-only queries ---


@router.get("/status", response_model=PaperStatusResponse)
async def get_status(jobs: JobMgr) -> PaperStatusResponse:
    try:
        run_id, _pid = _find_active_paper_job(jobs)
    except HTTPException:
        return PaperStatusResponse(state="idle", pnl_metrics=None, trades_count=0)

    info = jobs.get_job_info(run_id)
    state = info.status.value if info else "unknown"
    return PaperStatusResponse(state=state, pnl_metrics=None, trades_count=0)


@router.get("/positions", response_model=list[PaperPositionResponse])
async def get_positions() -> list[PaperPositionResponse]:
    # Stub -- real data streamed via /ws/trading
    return []


@router.get("/orders")
async def get_orders() -> list[dict[str, object]]:
    # Stub -- real data streamed via /ws/trading
    return []


@router.get("/checkpoints", response_model=list[CheckpointResponse])
async def get_checkpoints() -> list[CheckpointResponse]:
    try:
        from vibe_quant.paper.persistence import StatePersistence

        persistence = StatePersistence()
        # list_checkpoints needs a trader_id; without active session return empty
        _ = persistence
    except (ImportError, Exception):
        pass
    return []


@router.get("/sessions/{trader_id}", response_model=CheckpointResponse | None)
async def get_session(trader_id: str) -> CheckpointResponse | None:
    try:
        from vibe_quant.paper.persistence import StatePersistence

        persistence = StatePersistence()
        checkpoint = persistence.load_latest_checkpoint(trader_id)
        if checkpoint is not None:
            return CheckpointResponse(
                timestamp=str(checkpoint.timestamp),
                state=checkpoint.state,
                halt_reason=getattr(checkpoint, "halt_reason", None),
                error_message=getattr(checkpoint, "error_message", None),
            )
    except (ImportError, Exception):
        logger.debug("paper persistence not available for trader_id=%s", trader_id)
    return None
