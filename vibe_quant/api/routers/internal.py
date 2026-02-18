"""Subprocess-facing internal endpoints (heartbeat, trades, sweep)."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from vibe_quant.api.deps import get_state_manager, get_ws_manager
from vibe_quant.api.schemas.backtest import (  # noqa: TCH001
    ParetoMarkRequest,
    SweepResultsBatchRequest,
    TradesBatchRequest,
)
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest/jobs", tags=["internal"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]
WsMgr = Annotated[ConnectionManager, Depends(get_ws_manager)]


@router.post("/{run_id}/heartbeat")
async def heartbeat(run_id: int, state: StateMgr, ws: WsMgr) -> dict[str, str]:
    state.update_heartbeat(run_id)
    state.update_job_heartbeat(run_id)
    await ws.broadcast("jobs", {"type": "heartbeat", "run_id": run_id})
    return {"status": "ok"}


@router.post("/{run_id}/trades")
async def save_trades(
    run_id: int, body: TradesBatchRequest, state: StateMgr,
) -> dict[str, str | int]:
    state.save_trades_batch(run_id, body.trades)
    logger.info("saved %d trades for run_id=%d", len(body.trades), run_id)
    return {"status": "ok", "count": len(body.trades)}


@router.post("/{run_id}/sweep-results")
async def save_sweep_results(
    run_id: int, body: SweepResultsBatchRequest, state: StateMgr,
) -> dict[str, str | int]:
    state.save_sweep_results_batch(run_id, body.results)
    logger.info("saved %d sweep results for run_id=%d", len(body.results), run_id)
    return {"status": "ok", "count": len(body.results)}


@router.post("/{run_id}/mark-pareto")
async def mark_pareto(
    run_id: int, body: ParetoMarkRequest, state: StateMgr,
) -> dict[str, str | int]:
    state.mark_pareto_optimal(body.result_ids)
    logger.info(
        "marked %d pareto-optimal results for run_id=%d",
        len(body.result_ids),
        run_id,
    )
    return {"status": "ok", "count": len(body.result_ids)}
