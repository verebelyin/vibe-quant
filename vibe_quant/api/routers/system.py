"""System router (/api/system).

Portfolio-wide kill switch persisted in the state DB. Setting it
prevents starting new paper/live sessions and signals any active
paper job to halt. Clearing it requires an explicit operator unlock
— there is no auto-release.
"""

from __future__ import annotations

import logging
import os
import signal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from vibe_quant.api.deps import get_job_manager, get_state_manager, get_ws_manager
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

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

    - Persists kill state to the DB so restarts stay halted.
    - Sends SIGUSR1 to any active paper-trading PID (best-effort).
    - Broadcasts `system_killed` over the trading WebSocket.
    """
    state.set_kill_switch(body.reason, body.killed_by)
    logger.warning(
        "system kill engaged reason=%r by=%r", body.reason, body.killed_by
    )

    # Best-effort cascade to any live paper job. Never raise here —
    # the kill-switch flag is the source of truth regardless.
    for job in jobs.list_active_jobs():
        if job.job_type != "paper":
            continue
        try:
            os.kill(job.pid, signal.SIGUSR1)
            logger.info("cascaded kill SIGUSR1 to paper pid=%d", job.pid)
        except (ProcessLookupError, OSError) as exc:
            logger.warning("kill cascade failed pid=%d err=%s", job.pid, exc)

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
