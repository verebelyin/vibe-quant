"""Results & analytics router."""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from vibe_quant.api.deps import get_state_manager
from vibe_quant.api.schemas.backtest import BacktestRunResponse
from vibe_quant.api.schemas.result import (
    BacktestResultResponse,
    ComparisonResponse,
    DrawdownPoint,
    EquityCurvePoint,
    MonthlyReturn,
    NotesUpdateRequest,
    RunListResponse,
    SweepResultResponse,
    TradeResponse,
)
from vibe_quant.db.state_manager import StateManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/results", tags=["results"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    mgr: StateMgr,
    status: str | None = None,
    strategy_id: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> RunListResponse:
    rows = mgr.list_backtest_runs(strategy_id=strategy_id, status=status)
    runs = [BacktestRunResponse(**r) for r in rows]
    if start_date:
        runs = [r for r in runs if r.created_at >= start_date]
    if end_date:
        runs = [r for r in runs if r.created_at <= end_date]
    return RunListResponse(runs=runs)


@router.get("/compare", response_model=ComparisonResponse)
async def compare_runs(
    mgr: StateMgr,
    run_ids: Annotated[str, Query(description="Comma-separated run IDs")] = "",
) -> ComparisonResponse:
    if not run_ids.strip():
        raise HTTPException(status_code=400, detail="run_ids query param required")
    id_list = [int(x.strip()) for x in run_ids.split(",") if x.strip()]
    results: list[BacktestResultResponse] = []
    for rid in id_list:
        row = mgr.get_backtest_result(rid)
        if row is not None:
            results.append(BacktestResultResponse(**row))
    return ComparisonResponse(runs=results)


@router.get("/runs/{run_id}", response_model=BacktestResultResponse)
async def get_run_summary(run_id: int, mgr: StateMgr) -> BacktestResultResponse:
    row = mgr.get_backtest_result(run_id)
    if row is not None:
        return BacktestResultResponse(**row)

    # Screening runs don't populate backtest_results — synthesize from best sweep
    sweeps = mgr.get_sweep_results(run_id, pareto_only=True)
    if not sweeps:
        sweeps = mgr.get_sweep_results(run_id)
    if not sweeps:
        # Ensure run exists at all
        run = mgr.get_backtest_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        raise HTTPException(status_code=404, detail="No results available for this run yet")

    best = sweeps[0]  # already ordered by sharpe DESC
    return BacktestResultResponse(
        id=best["id"],
        run_id=run_id,
        total_return=best.get("total_return"),
        cagr=None,
        sharpe_ratio=best.get("sharpe_ratio"),
        sortino_ratio=best.get("sortino_ratio"),
        calmar_ratio=None,
        max_drawdown=best.get("max_drawdown"),
        max_drawdown_duration_days=None,
        volatility_annual=None,
        total_trades=best.get("total_trades"),
        winning_trades=None,
        losing_trades=None,
        win_rate=best.get("win_rate"),
        profit_factor=best.get("profit_factor"),
        avg_win=None,
        avg_loss=None,
        largest_win=None,
        largest_loss=None,
        avg_trade_duration_hours=None,
        max_consecutive_wins=None,
        max_consecutive_losses=None,
        total_fees=best.get("total_fees"),
        total_funding=best.get("total_funding"),
        total_slippage=None,
        deflated_sharpe=None,
        walk_forward_efficiency=None,
        purged_kfold_mean_sharpe=None,
        execution_time_seconds=best.get("execution_time_seconds"),
        starting_balance=None,
        notes=None,
        created_at=None,
    )


@router.get("/runs/{run_id}/trades", response_model=list[TradeResponse])
async def get_trades(
    run_id: int,
    mgr: StateMgr,
    symbol: str | None = None,
    direction: str | None = None,
) -> list[TradeResponse]:
    _ensure_run_exists(mgr, run_id)
    rows = mgr.get_trades(run_id)
    trades = [TradeResponse(**r) for r in rows]
    if symbol:
        trades = [t for t in trades if t.symbol == symbol]
    if direction:
        trades = [t for t in trades if t.direction == direction]
    return trades


@router.get("/runs/{run_id}/sweeps", response_model=list[SweepResultResponse])
async def get_sweeps(
    run_id: int,
    mgr: StateMgr,
    pareto_only: bool = False,
) -> list[SweepResultResponse]:
    _ensure_run_exists(mgr, run_id)
    rows = mgr.get_sweep_results(run_id, pareto_only=pareto_only)
    return [SweepResultResponse(**r) for r in rows]


@router.get("/runs/{run_id}/equity-curve", response_model=list[EquityCurvePoint])
async def get_equity_curve(run_id: int, mgr: StateMgr) -> list[EquityCurvePoint]:
    _ensure_run_exists(mgr, run_id)
    # Stub -- will compute from trades later
    return []


@router.get("/runs/{run_id}/drawdown", response_model=list[DrawdownPoint])
async def get_drawdown(run_id: int, mgr: StateMgr) -> list[DrawdownPoint]:
    _ensure_run_exists(mgr, run_id)
    # Stub -- will compute from trades later
    return []


@router.get("/runs/{run_id}/monthly-returns", response_model=list[MonthlyReturn])
async def get_monthly_returns(run_id: int, mgr: StateMgr) -> list[MonthlyReturn]:
    _ensure_run_exists(mgr, run_id)
    # Stub -- will compute from trades later
    return []


@router.put("/runs/{run_id}/notes", response_model=BacktestResultResponse)
async def update_notes(
    run_id: int,
    body: NotesUpdateRequest,
    mgr: StateMgr,
) -> BacktestResultResponse:
    _ensure_run_exists(mgr, run_id)
    mgr.update_result_notes(run_id, body.notes)
    row = mgr.get_backtest_result(run_id)
    if row is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Result disappeared after update")
    return BacktestResultResponse(**row)


@router.get("/runs/{run_id}/export/csv")
async def export_csv(run_id: int, mgr: StateMgr) -> StreamingResponse:
    _ensure_run_exists(mgr, run_id)
    trades = mgr.get_trades(run_id)

    buf = io.StringIO()
    if trades:
        writer = csv.DictWriter(buf, fieldnames=list(trades[0].keys()))
        writer.writeheader()
        writer.writerows(trades)
    else:
        buf.write("")

    buf.seek(0)
    filename = f"run_{run_id}_trades_{datetime.now().strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _ensure_run_exists(mgr: StateManager, run_id: int) -> None:
    run = mgr.get_backtest_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
