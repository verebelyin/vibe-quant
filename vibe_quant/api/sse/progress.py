"""SSE progress streaming endpoints for backtest and data ingest jobs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from vibe_quant.api.deps import get_job_manager, get_state_manager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

router = APIRouter(tags=["progress"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]
JobMgr = Annotated[BacktestJobManager, Depends(get_job_manager)]

_TERMINAL_STATUSES = frozenset({"completed", "failed", "killed"})
_POLL_INTERVAL = 1.0


async def _tail_log(
    run_id: int,
    job_mgr: BacktestJobManager,
    state_mgr: StateManager,
    last_event_id: int = 0,
) -> AsyncIterator[dict[str, str]]:
    job = state_mgr.get_job(run_id)
    if not job or not job.get("log_file"):
        yield {"event": "error", "data": "No log file for this job", "id": "0"}
        return

    log_path = Path(job["log_file"])
    line_num = last_event_id

    while True:
        status = job_mgr.get_status(run_id)

        if log_path.exists():
            async with asyncio.Lock():
                with log_path.open() as f:
                    lines = f.readlines()

            for i, line in enumerate(lines[line_num:], start=line_num):
                yield {"event": "log", "data": line.rstrip(), "id": str(i)}
            line_num = len(lines)

        if status and status.value in _TERMINAL_STATUSES:
            yield {"event": "complete", "data": status.value, "id": str(line_num)}
            return

        await asyncio.sleep(_POLL_INTERVAL)


@router.get("/api/backtest/jobs/{run_id}/progress")
async def backtest_progress(
    run_id: int,
    request: Request,
    job_mgr: JobMgr,
    state_mgr: StateMgr,
) -> EventSourceResponse:
    last_id = int(request.headers.get("Last-Event-ID", "0"))
    return EventSourceResponse(
        _tail_log(run_id, job_mgr, state_mgr, last_event_id=last_id),
    )


@router.get("/api/data/ingest/{job_id}/progress")
async def data_ingest_progress(
    job_id: int,
    request: Request,
    job_mgr: JobMgr,
    state_mgr: StateMgr,
) -> EventSourceResponse:
    last_id = int(request.headers.get("Last-Event-ID", "0"))
    return EventSourceResponse(
        _tail_log(job_id, job_mgr, state_mgr, last_event_id=last_id),
    )
