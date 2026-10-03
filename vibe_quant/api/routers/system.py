"""System router (/api/system).

Portfolio-wide kill switch persisted in the state DB. Setting it
prevents starting new paper/live sessions and halts every active paper
node: each node polls the flag (and also gets a ``kill`` command on the
paper command queue), then flattens positions with reduce-only market
orders, cancels orders and stops its strategies. Clearing it requires an
explicit operator unlock — there is no auto-release, and unlocking does
not auto-resume sessions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from vibe_quant.api.deps import get_job_manager, get_state_manager, get_ws_manager
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.paper.config import default_paper_trader_id
from vibe_quant.paper.persistence import PaperCommandQueue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]
JobMgr = Annotated[BacktestJobManager, Depends(get_job_manager)]
WsMgr = Annotated[ConnectionManager, Depends(get_ws_manager)]


class KillRequest(BaseModel):
    """Request body for POST /api/system/kill."""

    reason: str = Field(..., min_length=1, max_length=500)
    killed_by: str | None = Field(default=None, max_length=100)


class UnlockRequest(BaseModel):
    """Request body for POST /api/system/unlock — operator attestation."""

    cleared_by: str | None = Field(default=None, max_length=100)
    # Operator must echo this to confirm they've resolved the cause
    acknowledge: bool = Field(..., description="Must be true to unlock")


class SystemStatusResponse(BaseModel):
    kill_switch: bool
    reason: str | None
    killed_at: str | None
    killed_by: str | None
    updated_at: str | None


@router.get("/status", response_model=SystemStatusResponse)
async def get_status(state: StateMgr) -> SystemStatusResponse:
    return SystemStatusResponse(**state.get_system_state())


@router.post("/kill", response_model=SystemStatusResponse, status_code=200)
async def kill(
    body: KillRequest,
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
) -> SystemStatusResponse:
    """Engage the system-wide kill switch.

    - Persists kill state to the DB so restarts stay halted (nodes poll it).
    - Queues a ``kill`` command for every active paper session (no signals:
      a recycled PID can never receive it).
    - Broadcasts `system_killed` over the trading WebSocket.
    """
    state.set_kill_switch(body.reason, body.killed_by)
    logger.warning(
        "system kill engaged reason=%r by=%r", body.reason, body.killed_by
    )

    # Fast-path cascade. Never raise here — the persisted flag is the source
    # of truth and every node polls it independently.
    db_row = state.conn.execute("PRAGMA database_list").fetchone()
    db_file = db_row["file"] if db_row is not None else ""
    queue = PaperCommandQueue(Path(db_file) if db_file else None)
    try:
        for job in jobs.list_active_jobs():
            if job.job_type != "paper":
                continue
            run = state.get_backtest_run(job.run_id) or {}
            raw_params = run.get("parameters")
            params: dict[str, object] = raw_params if isinstance(raw_params, dict) else {}
            trader_id = str(params.get("trader_id") or default_paper_trader_id(job.run_id))
            try:
                queue.enqueue(trader_id, "kill", {"message": f"kill switch: {body.reason}"})
                logger.info("queued kill for paper trader_id=%s run_id=%d", trader_id, job.run_id)
            except Exception as exc:
                logger.warning("kill cascade failed run_id=%d err=%s", job.run_id, exc)
    finally:
        queue.close()

    await ws.broadcast(
        "trading",
        {"type": "system_killed", "reason": body.reason, "killed_by": body.killed_by},
    )
    return SystemStatusResponse(**state.get_system_state())


@router.post("/unlock", response_model=SystemStatusResponse, status_code=200)
async def unlock(
    body: UnlockRequest,
    state: StateMgr,
    ws: WsMgr,
) -> SystemStatusResponse:
    """Clear the kill switch. Requires ``acknowledge=true``."""
    if not body.acknowledge:
        raise HTTPException(
            status_code=400,
            detail="unlock requires acknowledge=true — operator must confirm",
        )
    state.clear_kill_switch(body.cleared_by)
    logger.warning("system kill cleared by=%r", body.cleared_by)
    await ws.broadcast(
        "trading",
        {"type": "system_unlocked", "cleared_by": body.cleared_by},
    )
    return SystemStatusResponse(**state.get_system_state())
