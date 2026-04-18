"""Reconciliation router — paper↔validation trade diff."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from vibe_quant.api.deps import get_job_manager, get_state_manager
from vibe_quant.api.schemas.reconciliation import (
    DivergenceSummary,
    PairedTrade,
    ReconciliationResponse,
    ReconciliationTrade,
)
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.reconciliation import Trade, load_trades, reconcile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reconciliation", tags=["reconciliation"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]
JobMgr = Annotated[BacktestJobManager, Depends(get_job_manager)]


def _trade_to_schema(t: Trade) -> ReconciliationTrade:
    return ReconciliationTrade(
        position_id=t.position_id,
        symbol=t.symbol,
        side=t.side,
        entry_time=t.entry_time.isoformat(),
        exit_time=t.exit_time.isoformat(),
        entry_price=t.entry_price,
        exit_price=t.exit_price,
        quantity=t.quantity,
        net_pnl=t.net_pnl,
        exit_reason=t.exit_reason,
    )


@router.get("/{paper_session_id}", response_model=ReconciliationResponse)
async def reconcile_paper_session(
    paper_session_id: int,
    state: StateMgr,
    jobs: JobMgr,
    validation_run_id: int | None = None,
    tolerance_seconds: int = 120,
) -> ReconciliationResponse:
    """Reconcile a stopped paper session against a validation run.

    V1: post-stop only. If the paper session is still running, returns 400.

    If `validation_run_id` is omitted, we try to pick the most recent
    successful validation run for the same strategy_id as the paper session.
    """
    info = jobs.get_job_info(paper_session_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Paper session {paper_session_id} not found")
    if info.status.value == "running":
        raise HTTPException(
            status_code=400,
            detail="Paper session still running — stop it before reconciling",
        )

    paper_run = state.get_backtest_run(paper_session_id)
    if paper_run is None:
        raise HTTPException(status_code=404, detail=f"Paper run {paper_session_id} not found")

    if validation_run_id is None:
        strategy_id = paper_run.get("strategy_id")
        if strategy_id is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Paper session has no strategy_id; pass validation_run_id "
                    "query param explicitly"
                ),
            )
        candidates = state.list_runs_with_results(
            strategy_id=strategy_id,
            run_mode="validation",
            status="completed",
        )
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail=f"No completed validation run found for strategy {strategy_id}",
            )
        validation_run_id = int(candidates[0]["run_id"])

    try:
        paper_trades, validation_trades = await asyncio.gather(
            asyncio.to_thread(load_trades, f"paper_{paper_session_id}"),
            asyncio.to_thread(load_trades, str(validation_run_id)),
        )
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(
            status_code=404, detail=f"Event log not found: {exc}"
        ) from exc

    report = reconcile(
        paper_trades,
        validation_trades,
        tolerance_seconds=tolerance_seconds,
        paper_run=f"paper_{paper_session_id}",
        validation_run=str(validation_run_id),
    )

    paired = [
        PairedTrade(
            paper=_trade_to_schema(m.paper),
            validation=_trade_to_schema(m.validation),
            entry_slippage=m.entry_slippage,
            pnl_delta=m.pnl_delta,
            side_agrees=m.side_agrees,
        )
        for m in report.matches
    ]
    return ReconciliationResponse(
        paper_run=report.paper_run,
        validation_run=report.validation_run,
        tolerance_seconds=report.tolerance_seconds,
        paired_trades=paired,
        unpaired_paper=[_trade_to_schema(t) for t in report.paper_only],
        unpaired_validation=[_trade_to_schema(t) for t in report.validation_only],
        divergence_summary=DivergenceSummary(
            matched=len(report.matches),
            paper_only=len(report.paper_only),
            validation_only=len(report.validation_only),
            parity_rate=report.parity_rate(),
            side_disagreements=report.side_disagreements(),
            mean_entry_slippage=report.mean_entry_slippage(),
            mean_pnl_delta=report.mean_pnl_delta(),
        ),
    )
