"""Paper trading router (/api/paper)."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from vibe_quant.api.deps import get_job_manager, get_state_manager, get_ws_manager
from vibe_quant.api.schemas.paper_trading import (
    CheckpointResponse,
    PaperOrderResponse,
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
    sys_state = state.get_system_state()
    if sys_state.get("kill_switch"):
        # 423 Locked is the canonical code for resource-in-a-locked-state.
        raise HTTPException(
            status_code=423,
            detail=f"System kill switch engaged: {sys_state.get('reason') or 'no reason'}",
        )

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

    # Write config JSON for the CLI subprocess
    config_dir = Path("data/state/paper_configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"paper_{run_id}.json"
    config_data: dict[str, object] = {
        "trader_id": body.trader_id or f"paper_{run_id}",
        "strategy_id": body.strategy_id,
        "binance": {"testnet": body.testnet},
        "symbols": [],
    }
    if body.sizing_method is not None:
        config_data.setdefault("sizing", {})
        config_data["sizing"]["method"] = body.sizing_method  # type: ignore[index]
    if body.max_leverage is not None:
        config_data.setdefault("sizing", {})
        config_data["sizing"]["max_leverage"] = str(body.max_leverage)  # type: ignore[index]
    if body.max_position_pct is not None:
        config_data.setdefault("sizing", {})
        config_data["sizing"]["max_position_pct"] = str(body.max_position_pct)  # type: ignore[index]
    if body.risk_per_trade is not None:
        config_data.setdefault("sizing", {})
        config_data["sizing"]["risk_per_trade"] = str(body.risk_per_trade)  # type: ignore[index]
    if body.max_drawdown_pct is not None:
        config_data.setdefault("risk", {})
        config_data["risk"]["max_drawdown_pct"] = str(body.max_drawdown_pct)  # type: ignore[index]
    if body.max_daily_loss_pct is not None:
        config_data.setdefault("risk", {})
        config_data["risk"]["max_daily_loss_pct"] = str(body.max_daily_loss_pct)  # type: ignore[index]
    if body.max_consecutive_losses is not None:
        config_data.setdefault("risk", {})
        config_data["risk"]["max_consecutive_losses"] = body.max_consecutive_losses  # type: ignore[index]
    if body.max_position_count is not None:
        config_data.setdefault("risk", {})
        config_data["risk"]["max_position_count"] = body.max_position_count  # type: ignore[index]
    with config_path.open("w") as f:
        json.dump(config_data, f, indent=2)

    from datetime import UTC
    from datetime import datetime as dt

    _ts = dt.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/paper_{run_id}_{_ts}.log"
    command = [
        sys.executable,
        "-m",
        "vibe_quant.paper.cli",
        "start",
        "--config",
        str(config_path),
        "--run-id",
        str(run_id),
    ]

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


@router.post("/close-all-positions", status_code=200)
async def close_all_positions(jobs: JobMgr, ws: WsMgr) -> dict[str, str]:
    """Signal the paper trading process to close all open positions."""
    run_id, pid = _find_active_paper_job(jobs)
    try:
        os.kill(pid, signal.SIGWINCH)  # Use SIGWINCH as close-all signal
    except ProcessLookupError as exc:
        raise HTTPException(status_code=410, detail="Process no longer running") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Signal failed: {exc}") from exc

    logger.info("paper trading close-all-positions run_id=%d pid=%d", run_id, pid)
    await ws.broadcast("trading", {"type": "paper_close_all", "run_id": run_id})
    return {"status": "closing_positions", "run_id": str(run_id)}


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


def _load_latest_for_trader(trader_id: str | None) -> object:
    if not trader_id:
        return None
    try:
        from vibe_quant.paper.persistence import StatePersistence

        persistence = StatePersistence()
        return persistence.load_latest_checkpoint(trader_id)
    except ImportError:
        logger.debug("paper persistence not available")
        return None


@router.get("/positions", response_model=list[PaperPositionResponse])
async def get_positions(trader_id: str | None = None) -> list[PaperPositionResponse]:
    """Return open positions from the latest checkpoint for trader_id.

    WebSocket /ws/trading streams real-time updates; this endpoint provides
    a fallback snapshot when the UI first loads or the socket is down.
    """
    checkpoint = _load_latest_for_trader(trader_id)
    if checkpoint is None:
        return []
    positions = getattr(checkpoint, "positions", {}) or {}
    result: list[PaperPositionResponse] = []
    for pos in positions.values():
        if not isinstance(pos, dict):
            continue
        try:
            result.append(
                PaperPositionResponse(
                    symbol=str(pos.get("symbol", "")),
                    direction=str(pos.get("direction") or pos.get("side") or ""),
                    quantity=float(pos.get("quantity") or pos.get("qty") or 0.0),
                    entry_price=float(pos.get("entry_price") or pos.get("avg_px") or 0.0),
                    unrealized_pnl=float(pos.get("unrealized_pnl") or 0.0),
                    leverage=float(pos.get("leverage") or 1.0),
                )
            )
        except (TypeError, ValueError):
            continue
    return result


@router.get("/orders", response_model=list[PaperOrderResponse])
async def get_orders(trader_id: str | None = None) -> list[PaperOrderResponse]:
    """Return open orders from the latest checkpoint for trader_id."""
    checkpoint = _load_latest_for_trader(trader_id)
    if checkpoint is None:
        return []
    orders = getattr(checkpoint, "orders", {}) or {}
    result: list[PaperOrderResponse] = []
    for oid, order in orders.items():
        if not isinstance(order, dict):
            continue
        qty = order.get("quantity") or order.get("qty")
        px = order.get("price")
        result.append(
            PaperOrderResponse(
                order_id=str(oid),
                symbol=str(order.get("symbol", "")),
                side=order.get("side"),
                quantity=float(qty) if qty is not None else None,
                price=float(px) if px is not None else None,
                status=order.get("status"),
            )
        )
    return result


@router.get("/checkpoints", response_model=list[CheckpointResponse])
async def get_checkpoints(
    trader_id: str | None = None, limit: int = 50
) -> list[CheckpointResponse]:
    """Return checkpoint history for a trader, newest first."""
    if not trader_id:
        return []
    try:
        from vibe_quant.paper.persistence import StatePersistence

        persistence = StatePersistence()
        checkpoints = persistence.list_checkpoints(trader_id, limit=limit)
    except ImportError:
        return []
    result: list[CheckpointResponse] = []
    for cp in checkpoints:
        node_status = cp.node_status or {}
        result.append(
            CheckpointResponse(
                timestamp=str(cp.timestamp),
                state=node_status.get("state", "unknown"),
                halt_reason=node_status.get("halt_reason"),
                error_message=node_status.get("error_message"),
            )
        )
    return result


@router.get("/sessions/{trader_id}", response_model=CheckpointResponse | None)
async def get_session(trader_id: str) -> CheckpointResponse | None:
    try:
        from vibe_quant.paper.persistence import StatePersistence

        persistence = StatePersistence()
        checkpoint = persistence.load_latest_checkpoint(trader_id)
        if checkpoint is not None:
            node_status = checkpoint.node_status or {}
            return CheckpointResponse(
                timestamp=str(checkpoint.timestamp),
                state=node_status.get("state", "unknown"),
                halt_reason=node_status.get("halt_reason"),
                error_message=node_status.get("error_message"),
            )
    except ImportError:
        logger.debug("paper persistence not available for trader_id=%s", trader_id)
    return None
