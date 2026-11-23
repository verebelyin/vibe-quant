"""Shadow forward tracker router (/api/shadow), bead vibe-quant-clmmc.7.

Enrol a frozen strategy, list the rows with the unseen state-change badge, read a row's
forward days, mark state changes seen, and start an update job. The update job runs the
shadow CLI (``python -m vibe_quant.shadow update``), which only scores data already
downloaded; it never places orders. ``kill_switch_engaged`` is informational: the tracker
never reads the kill switch for control.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from vibe_quant.api.deps import get_job_manager, get_state_manager
from vibe_quant.db.state_manager import ShadowUpdateRunning, StateManager, shadow_update_lock
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.utils import log_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/shadow", tags=["shadow"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]
JobMgr = Annotated[BacktestJobManager, Depends(get_job_manager)]

_EPOCH_DAY = date(1970, 1, 1)


class ShadowEnrolRequest(BaseModel):
    """Mirrors ``tracker.EnrolRequest``; the tracker checks the values (422 text)."""

    name: str = Field(..., min_length=1)
    source_kind: Literal["strategy", "champion", "candidate"]
    dsl_json: str
    symbols: list[str]
    parameters: dict[str, object] = Field(default_factory=dict)
    pre_registration: str
    max_dd_limit: float
    warmup_days: int = 180
    source_ref: dict[str, object] = Field(default_factory=dict)
    strategy_id: int | None = None
    backdate_days: int | None = None

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be blank")
        return value

    @field_validator("dsl_json")
    @classmethod
    def _dsl_is_json_object(cls, value: str) -> str:
        try:
            parsed = json.loads(value)
        except ValueError as exc:
            raise ValueError(f"dsl_json is not valid JSON ({exc})") from exc
        if not isinstance(parsed, dict):
            raise ValueError("dsl_json must be a JSON object")
        return value


class ShadowStats(BaseModel):
    """Forward-window statistics (``estat.ForwardStats``)."""

    days: int
    total_return: float
    sharpe_ann: float | None
    se_ann: float | None
    ub95: float | None
    max_dd: float
    blocks: int
    e_value: float


class ShadowStrategy(BaseModel):
    """One enrolment row (its DSL and frozen identity stay in the DB)."""

    id: int
    name: str
    source_kind: str
    source_ref: dict[str, object]
    strategy_id: int | None
    dsl_sha256: str
    compiler_hash: str
    compiler_hash_last: str | None
    symbols: list[str]
    timeframe: str
    parameters: dict[str, object]
    enrolled_at: str
    first_day: str
    run_start: str
    is_test: bool
    family: str | None
    slot: int | None
    pre_registration: str
    max_dd_limit: float
    engine_epoch: int
    engine_epoch_last: int
    state: Literal["tracking", "stalled", "passed", "futile", "retired"]
    state_reason: str | None
    state_changed_at: str | None
    verdict_day: str | None
    proposal_stale: bool
    status_note: str | None
    committed_through: str | None
    last_update_at: str | None
    stats: ShadowStats | None


class ShadowListResponse(BaseModel):
    strategies: list[ShadowStrategy]
    unseen_changes: int
    kill_switch_engaged: bool


class ShadowDay(BaseModel):
    day: str
    ret: float


class ShadowSeenResponse(BaseModel):
    marked: int


class ShadowUpdateResponse(BaseModel):
    status: Literal["started"]
    job_id: int
    pid: int
    kill_switch_engaged: bool


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _kill_switch_engaged(state: StateManager) -> bool:
    """Read like paper does; a missing row is re-created ENGAGED."""
    return bool(state.get_system_state()["kill_switch"])


def _row(state: StateManager, shadow_id: int) -> ShadowStrategy | None:
    row = next((r for r in state.list_shadow_strategies() if r["id"] == shadow_id), None)
    return None if row is None else ShadowStrategy.model_validate(row)


@router.get("", response_model=ShadowListResponse)
def list_shadow(state: StateMgr) -> ShadowListResponse:
    return ShadowListResponse(
        strategies=[ShadowStrategy.model_validate(r) for r in state.list_shadow_strategies()],
        unseen_changes=state.count_unseen_shadow_changes(),
        kill_switch_engaged=_kill_switch_engaged(state),
    )


@router.get("/{shadow_id}/days", response_model=list[ShadowDay])
def shadow_days(shadow_id: int, state: StateMgr) -> list[ShadowDay]:
    if _row(state, shadow_id) is None:
        raise HTTPException(status_code=404, detail=f"shadow row {shadow_id} not found")
    return [
        ShadowDay(day=(_EPOCH_DAY + timedelta(days=d)).isoformat(), ret=r)
        for d, r in state.get_shadow_days(shadow_id)
    ]


@router.post("/enrol", response_model=ShadowStrategy, status_code=201)
def enrol_shadow(body: ShadowEnrolRequest, state: StateMgr) -> ShadowStrategy:
    from vibe_quant.shadow import tracker  # loads the engine seams + scipy on first use

    try:
        shadow_id = tracker.enrol(state, tracker.EnrolRequest(**body.model_dump()), now=_utcnow())
    except tracker.EnrolRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    row = _row(state, shadow_id)
    assert row is not None
    logger.info("shadow enrol id=%d name=%s slot=%s", shadow_id, row.name, row.slot)
    return row


@router.post("/events/seen", response_model=ShadowSeenResponse)
def mark_seen(state: StateMgr) -> ShadowSeenResponse:
    return ShadowSeenResponse(marked=state.mark_shadow_events_seen())


@router.post("/update", response_model=ShadowUpdateResponse, status_code=202)
def start_shadow_update(state: StateMgr, jobs: JobMgr) -> ShadowUpdateResponse:
    """Start ``python -m vibe_quant.shadow update`` as a job (R39).

    The lock pre-check is best effort: an update that starts between it and the job's
    own lock makes the job complete "skipped: shadow update already running".
    """
    db = jobs.db_path
    try:
        with shadow_update_lock(db):
            pass
    except ShadowUpdateRunning:
        raise HTTPException(status_code=409, detail="shadow update already running") from None
    engaged = _kill_switch_engaged(state)  # read before the job can change anything
    job_id = -int(time.time() * 1_000_000)  # negative: never a backtest run id
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    command = [
        sys.executable,
        "-m",
        "vibe_quant.shadow",
        "update",
        "--db",
        str(db),
        "--job-id",
        str(job_id),
    ]
    try:
        pid = jobs.start_job(
            job_id, "shadow_update", command, log_file=str(log_dir() / f"shadow_update_{stamp}.log")
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    logger.info("shadow update started job_id=%d pid=%d", job_id, pid)
    return ShadowUpdateResponse(
        status="started",
        job_id=job_id,
        pid=pid,
        kill_switch_engaged=engaged,
    )
