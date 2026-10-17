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

import json
import logging
import sqlite3
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from vibe_quant.api.deps import get_job_manager, get_state_manager, get_ws_manager
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.connection import get_connection
from vibe_quant.db.schema import SystemStateMissingError
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.paper.config import default_paper_trader_id
from vibe_quant.paper.persistence import PaperCommandQueue

if TYPE_CHECKING:
    from pathlib import Path

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


def _paper_trader_id(db_path: Path | None, run_id: int) -> str:
    """The run's ``parameters.trader_id`` (default id if absent or unreadable).

    Reads through its own plain connection: ``StateManager`` may be exactly the
    thing that cannot initialise (the cascade must still run).
    """
    default = default_paper_trader_id(run_id)
    try:
        conn = get_connection(db_path)
        try:
            row = conn.execute(
                "SELECT parameters FROM backtest_runs WHERE id = ?", (run_id,)
            ).fetchone()
        finally:
            conn.close()
        params = json.loads(row[0]) if row is not None else None
    except (sqlite3.Error, TypeError, ValueError):
        return default
    if not isinstance(params, dict):
        return default
    return str(params.get("trader_id") or default)


@router.get("/status", response_model=SystemStatusResponse)
async def get_status(state: StateMgr) -> SystemStatusResponse:
    """Report the persisted kill state.

    A row that went missing is re-created ENGAGED by the read (a deleted
    kill-switch record is never reported as unlocked); an unreadable-and-
    unreseedable state is a 503, never a 200 default.
    """
    try:
        return SystemStatusResponse(**state.get_system_state())
    except SystemStateMissingError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"kill switch state unavailable: {exc}",
        ) from exc


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

    If persisting fails the cascade still runs (live sessions must stop even
    when the DB write does not land) and the response is a 503 naming the error
    class and how many sessions were told to kill — a kill that is not on disk
    must never look like a successful kill.
    """
    persist_error: Exception | None = None
    try:
        state.set_kill_switch(body.reason, body.killed_by)
    except (sqlite3.Error, SystemStateMissingError) as exc:
        persist_error = exc
        logger.error("system kill NOT persisted (%s); cascading kill anyway", type(exc).__name__)
    else:
        logger.warning("system kill engaged reason=%r by=%r", body.reason, body.killed_by)

    # Fast-path cascade. Never raise here — the persisted flag is the source
    # of truth and every node polls it independently.
    # Nothing below may touch ``state``'s connection: when the persist failed
    # because the StateManager's initializer fails, every ``state`` call would
    # re-run that failing initializer and abort the cascade (500, no commands).
    # The path comes from the configured value and trader ids from a plain
    # connection of their own.
    queue = PaperCommandQueue(state.db_path)
    sent = 0
    try:
        for job in jobs.list_active_jobs():
            if job.job_type != "paper":
                continue
            try:
                trader_id = _paper_trader_id(state.db_path, job.run_id)
                queue.enqueue(trader_id, "kill", {"message": f"kill switch: {body.reason}"})
                sent += 1
                logger.info("queued kill for paper trader_id=%s run_id=%d", trader_id, job.run_id)
            except Exception as exc:
                logger.warning("kill cascade failed run_id=%d err=%s", job.run_id, exc)
    finally:
        queue.close()

    await ws.broadcast(
        "trading",
        {"type": "system_killed", "reason": body.reason, "killed_by": body.killed_by},
    )

    if persist_error is not None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"kill switch NOT persisted ({type(persist_error).__name__}); "
                f"kill commands sent to {sent} paper session(s)"
            ),
        )
    return SystemStatusResponse(**state.get_system_state())


@router.post("/unlock", response_model=SystemStatusResponse, status_code=200)
async def unlock(
    body: UnlockRequest,
    state: StateMgr,
    ws: WsMgr,
) -> SystemStatusResponse:
    """Clear the kill switch. Requires ``acknowledge=true``.

    If the row went missing in the meantime the unlock is a 409: the row has
    just been re-created ENGAGED and the operator must unlock again.
    """
    if not body.acknowledge:
        raise HTTPException(
            status_code=400,
            detail="unlock requires acknowledge=true — operator must confirm",
        )
    try:
        state.clear_kill_switch(body.cleared_by)
    except SystemStateMissingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    logger.warning("system kill cleared by=%r", body.cleared_by)
    await ws.broadcast(
        "trading",
        {"type": "system_unlocked", "cleared_by": body.cleared_by},
    )
    return SystemStatusResponse(**state.get_system_state())
