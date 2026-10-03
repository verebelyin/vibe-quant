"""Paper trading router (/api/paper).

Control (halt / pause / resume / close-all) goes through the
``paper_commands`` queue in the state DB: the running node executes the command
and reports a result or an error, which the endpoint returns. Nothing is
fire-and-forget, and no POSIX signal can be confused with a terminal resize.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException

from vibe_quant.api.deps import get_job_manager, get_state_manager, get_ws_manager
from vibe_quant.api.schemas.paper_trading import (
    CheckpointResponse,
    PaperOrderResponse,
    PaperPositionResponse,
    PaperRestoreRequest,
    PaperStartRequest,
    PaperStatusResponse,
)
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.paper.config import (
    DEFAULT_PAPER_LOGS_PATH,
    credential_env_names,
    default_paper_trader_id,
    is_valid_trader_id,
)
from vibe_quant.paper.persistence import COMMAND_FAILED, PaperCommandQueue, StatePersistence

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/paper", tags=["paper"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]
JobMgr = Annotated[BacktestJobManager, Depends(get_job_manager)]
WsMgr = Annotated[ConnectionManager, Depends(get_ws_manager)]

#: Where /start writes the JSON config the paper subprocess reads.
PAPER_CONFIG_DIR = Path("data/state/paper_configs")
#: How long control endpoints wait for the node to acknowledge a command.
COMMAND_TIMEOUT_SECS = 15.0
#: How long close-all waits for every position to be flat.
CLOSE_ALL_WAIT_SECS = 10.0


def _state_db_path(state: StateManager) -> Path | None:
    row = state.conn.execute("PRAGMA database_list").fetchone()
    file = row["file"] if row is not None else ""
    return Path(file) if file else None


def _active_paper_jobs(jobs: BacktestJobManager) -> list[Any]:
    return [job for job in jobs.list_active_jobs() if job.job_type == "paper"]


def _find_active_paper_job(
    jobs: BacktestJobManager,
) -> tuple[int, int]:
    """Find running paper trading job. Returns (run_id, pid)."""
    for job in _active_paper_jobs(jobs):
        return job.run_id, job.pid
    raise HTTPException(status_code=404, detail="No active paper trading session")


def _session_params(state: StateManager, run_id: int) -> dict[str, Any]:
    run = state.get_backtest_run(run_id)
    params = run.get("parameters") if run else None
    return params if isinstance(params, dict) else {}


def _trader_id_for_run(state: StateManager, run_id: int) -> str:
    trader_id = _session_params(state, run_id).get("trader_id")
    return str(trader_id) if trader_id else default_paper_trader_id(run_id)


def _require_credentials(testnet: bool) -> None:
    key_var, secret_var = credential_env_names(testnet)
    missing = [v for v in (key_var, secret_var) if not os.environ.get(v)]
    if missing:
        env = "testnet" if testnet else "LIVE"
        raise HTTPException(
            status_code=400,
            detail=(
                f"Binance {env} credentials missing: set {', '.join(missing)} in the "
                "backend environment (keys are never accepted through the API)"
            ),
        )


def _refuse_if_killed(state: StateManager) -> None:
    sys_state = state.get_system_state()
    if sys_state.get("kill_switch"):
        # 423 Locked is the canonical code for resource-in-a-locked-state.
        raise HTTPException(
            status_code=423,
            detail=f"System kill switch engaged: {sys_state.get('reason') or 'no reason'}",
        )


def _refuse_if_active(jobs: BacktestJobManager) -> None:
    for job in _active_paper_jobs(jobs):
        raise HTTPException(
            status_code=409,
            detail=f"Paper session already running (run_id={job.run_id}); stop it first",
        )


def _launch(
    jobs: BacktestJobManager,
    state: StateManager,
    run_id: int,
    config_data: dict[str, Any],
    *,
    live: bool,
) -> int:
    PAPER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = PAPER_CONFIG_DIR / f"paper_{run_id}.json"
    with config_path.open("w") as f:
        json.dump(config_data, f, indent=2)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/paper_{run_id}_{ts}.log"
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
    if live:
        command.append("--live")

    try:
        pid = jobs.start_job(run_id, "paper", command, log_file=log_file)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    state.update_backtest_run_status(run_id, "running", pid=pid)
    return pid


# --- Start / lifecycle ---


@router.post("/start", response_model=PaperStatusResponse, status_code=201)
async def start_paper(
    body: PaperStartRequest,
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
) -> PaperStatusResponse:
    _refuse_if_killed(state)
    _refuse_if_active(jobs)

    if not body.testnet and not body.confirm_live:
        raise HTTPException(
            status_code=400,
            detail="testnet=false trades real funds: set confirm_live=true to proceed",
        )
    if body.trader_id is not None and not is_valid_trader_id(body.trader_id):
        raise HTTPException(
            status_code=422,
            detail=(
                f"trader_id {body.trader_id!r} is invalid: NautilusTrader needs NAME-TAG "
                "with a hyphen (e.g. PAPER-001)"
            ),
        )

    strategy_id = body.strategy_id
    requested = [s.strip().upper() for s in body.symbols or [] if s.strip()]
    if body.validation_run_id is not None:
        run = state.get_backtest_run(body.validation_run_id)
        if run is None:
            raise HTTPException(404, f"Validation run {body.validation_run_id} not found")
        if run.get("run_mode") != "validation" or run.get("status") != "completed":
            raise HTTPException(
                400,
                f"Run {body.validation_run_id} is not a completed validation run "
                f"(mode={run.get('run_mode')}, status={run.get('status')})",
            )
        if strategy_id is not None and strategy_id != run.get("strategy_id"):
            raise HTTPException(
                400,
                f"Validation run {body.validation_run_id} is for strategy "
                f"{run.get('strategy_id')}, not {strategy_id}",
            )
        strategy_id = int(run["strategy_id"])
        symbols = [str(s) for s in run.get("symbols") or []]
        if requested and sorted(requested) != sorted(symbols):
            raise HTTPException(
                400, f"symbols {requested} differ from the validation run's {symbols}"
            )
        timeframe = str(run.get("timeframe") or "")
    else:
        if strategy_id is None:
            raise HTTPException(400, "strategy_id or validation_run_id is required")
        strategy = state.get_strategy(strategy_id)
        if strategy is None:
            raise HTTPException(404, f"Strategy {strategy_id} not found")
        if not requested:
            raise HTTPException(
                400, "symbols are required when not starting from a validation run"
            )
        symbols = requested
        dsl = strategy.get("dsl_config")
        timeframe = str(dsl.get("timeframe", "")) if isinstance(dsl, dict) else ""
    if not symbols:
        raise HTTPException(400, "No symbols to trade")

    _require_credentials(body.testnet)

    sizing: dict[str, object] = {}
    if body.sizing_method is not None:
        sizing["method"] = body.sizing_method
    if body.max_leverage is not None:
        sizing["max_leverage"] = str(body.max_leverage)
    if body.max_position_pct is not None:
        sizing["max_position_pct"] = str(body.max_position_pct)
    if body.risk_per_trade is not None:
        sizing["risk_per_trade"] = str(body.risk_per_trade)
    risk: dict[str, object] = {}
    if body.max_drawdown_pct is not None:
        risk["max_drawdown_pct"] = str(body.max_drawdown_pct)
    if body.max_daily_loss_pct is not None:
        risk["max_daily_loss_pct"] = str(body.max_daily_loss_pct)
    if body.max_consecutive_losses is not None:
        risk["max_consecutive_losses"] = body.max_consecutive_losses
    if body.max_position_count is not None:
        risk["max_position_count"] = body.max_position_count

    params: dict[str, object] = {
        "testnet": body.testnet,
        "validation_run_id": body.validation_run_id,
        "logs_path": str(DEFAULT_PAPER_LOGS_PATH),
        "sizing": sizing,
        "risk": risk,
    }
    if body.trader_id:
        params["trader_id"] = body.trader_id

    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="paper",
        symbols=symbols,
        timeframe=timeframe,
        start_date="",
        end_date="",
        parameters=params,
    )
    trader_id = body.trader_id or default_paper_trader_id(run_id)
    db_path = _state_db_path(state)

    config_data: dict[str, Any] = {
        "trader_id": trader_id,
        "strategy_id": strategy_id,
        "validation_run_id": body.validation_run_id,
        "binance": {"testnet": body.testnet},
        "symbols": symbols,
        "logs_path": str(DEFAULT_PAPER_LOGS_PATH),
    }
    if db_path is not None:
        config_data["db_path"] = str(db_path)
    if sizing:
        config_data["sizing"] = sizing
    if risk:
        config_data["risk"] = risk

    pid = _launch(jobs, state, run_id, config_data, live=not body.testnet)
    logger.info(
        "paper trading started run_id=%d pid=%d trader_id=%s testnet=%s",
        run_id,
        pid,
        trader_id,
        body.testnet,
    )

    await ws.broadcast(
        "trading",
        {
            "type": "paper_started",
            "run_id": run_id,
            "strategy_id": strategy_id,
            "trader_id": trader_id,
        },
    )

    return PaperStatusResponse(
        state="starting",
        pnl_metrics=None,
        trades_count=0,
        run_id=run_id,
        trader_id=trader_id,
        testnet=body.testnet,
    )


def _find_latest_config_for_trader(trader_id: str) -> tuple[Path, dict[str, object]] | None:
    """Scan paper_configs/*.json newest-first for one matching trader_id."""
    if not PAPER_CONFIG_DIR.exists():
        return None
    candidates = sorted(
        PAPER_CONFIG_DIR.glob("paper_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            with path.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("trader_id") == trader_id:
            return path, data
    return None


@router.post("/restore", response_model=PaperStatusResponse, status_code=201)
async def restore_paper(
    body: PaperRestoreRequest,
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
) -> PaperStatusResponse:
    """Restart a paper session for an existing trader_id.

    Reuses the most-recent saved config (strategy, validation run, symbols,
    risk/sizing) for trader_id. On startup the node loads that trader's latest
    checkpoint and restores the risk bookkeeping (high water mark, UTC-day
    baseline, consecutive losses) so limits survive the restart. Open positions
    and orders are NOT taken from the checkpoint: NautilusTrader reconciles them
    from the venue and the strategy claims them (``external_order_claims``), so
    the restored session manages the existing position instead of opening a
    second one.
    """
    _refuse_if_killed(state)
    _refuse_if_active(jobs)

    found = _find_latest_config_for_trader(body.trader_id)
    if found is None:
        raise HTTPException(
            status_code=404,
            detail=f"No prior paper config found for trader_id={body.trader_id!r}",
        )
    _prior_path, prior_config = found

    strategy_id = prior_config.get("strategy_id")
    if not isinstance(strategy_id, int):
        raise HTTPException(
            status_code=400,
            detail="Prior config missing strategy_id — cannot restore",
        )
    binance = prior_config.get("binance")
    testnet = bool(binance.get("testnet", True)) if isinstance(binance, dict) else True
    if not testnet and not body.confirm_live:
        raise HTTPException(
            status_code=400,
            detail="This trader_id traded LIVE: set confirm_live=true to restore it",
        )
    _require_credentials(testnet)

    raw_symbols = prior_config.get("symbols")
    symbols = [str(s) for s in raw_symbols] if isinstance(raw_symbols, list) else []
    params: dict[str, object] = {
        "trader_id": body.trader_id,
        "testnet": testnet,
        "validation_run_id": prior_config.get("validation_run_id"),
        "logs_path": prior_config.get("logs_path", str(DEFAULT_PAPER_LOGS_PATH)),
        "sizing": prior_config.get("sizing") or {},
        "risk": prior_config.get("risk") or {},
        "restored_from_trader_id": body.trader_id,
    }

    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="paper",
        symbols=symbols,
        timeframe="",
        start_date="",
        end_date="",
        parameters=params,
    )

    new_config: dict[str, Any] = dict(prior_config)
    new_config["trader_id"] = body.trader_id
    db_path = _state_db_path(state)
    if db_path is not None:
        new_config["db_path"] = str(db_path)
    pid = _launch(jobs, state, run_id, new_config, live=not testnet)
    logger.info(
        "paper trading restored trader_id=%s run_id=%d pid=%d", body.trader_id, run_id, pid
    )

    await ws.broadcast(
        "trading",
        {"type": "paper_restored", "run_id": run_id, "trader_id": body.trader_id},
    )
    return PaperStatusResponse(
        state="starting",
        pnl_metrics=None,
        trades_count=0,
        run_id=run_id,
        trader_id=body.trader_id,
        testnet=testnet,
    )


async def _send_command(
    state: StateManager,
    jobs: BacktestJobManager,
    command: str,
    payload: dict[str, Any] | None = None,
    *,
    extra_timeout: float = 0.0,
    failure_status: int = 409,
) -> dict[str, object]:
    """Queue a command for the active node and return its acknowledged result."""
    timeout = COMMAND_TIMEOUT_SECS + extra_timeout
    run_id, _pid = _find_active_paper_job(jobs)
    trader_id = _trader_id_for_run(state, run_id)
    queue = PaperCommandQueue(_state_db_path(state))
    try:
        command_id = queue.enqueue(trader_id, command, payload)
        cmd = await queue.wait_for(command_id, timeout)
        if cmd is None:
            cancelled = queue.expire(command_id)
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Paper node {trader_id} did not acknowledge '{command}' within "
                    f"{timeout:.0f}s"
                    + (" (command cancelled)" if cancelled else " (still executing)")
                ),
            )
    finally:
        queue.close()
    if cmd.status == COMMAND_FAILED:
        raise HTTPException(
            status_code=failure_status,
            detail={"error": cmd.error, "result": cmd.result, "run_id": run_id},
        )
    return {"status": cmd.status, "run_id": run_id, "trader_id": trader_id, **(cmd.result or {})}


@router.post("/halt", status_code=200)
async def halt_paper(
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
    mode: Literal["halt", "pause"] = "halt",
) -> dict[str, object]:
    """``halt``: flatten (reduce-only), cancel orders, stop strategies.
    ``pause``: block new entries only; SL/TP and exits keep working."""
    result = await _send_command(state, jobs, mode)
    logger.info("paper trading %s run_id=%s", mode, result.get("run_id"))
    await ws.broadcast(
        "trading", {"type": f"paper_{'halted' if mode == 'halt' else 'paused'}", **result}
    )
    return result


@router.post("/resume", status_code=200)
async def resume_paper(state: StateMgr, jobs: JobMgr, ws: WsMgr) -> dict[str, object]:
    """Resume from pause or an operator-resumable halt (never while killed)."""
    _refuse_if_killed(state)
    result = await _send_command(state, jobs, "resume")
    logger.info("paper trading resumed run_id=%s", result.get("run_id"))
    await ws.broadcast("trading", {"type": "paper_resumed", **result})
    return result


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
async def close_all_positions(state: StateMgr, jobs: JobMgr, ws: WsMgr) -> dict[str, object]:
    """Close every open position with reduce-only market orders and wait until flat.

    502 with the node's error list when anything could not be closed.
    """
    result = await _send_command(
        state,
        jobs,
        "close_all",
        {"wait_secs": CLOSE_ALL_WAIT_SECS},
        extra_timeout=CLOSE_ALL_WAIT_SECS,
        failure_status=502,
    )
    logger.info("paper trading close-all run_id=%s", result.get("run_id"))
    await ws.broadcast("trading", {"type": "paper_close_all", **result})
    return result


# --- Read-only queries ---


def _load_latest_for_trader(trader_id: str | None, db_path: Path | None) -> Any:
    if not trader_id:
        return None
    persistence = StatePersistence(db_path)
    try:
        return persistence.load_latest_checkpoint(trader_id)
    finally:
        persistence.close()


@router.get("/status", response_model=PaperStatusResponse)
async def get_status(state: StateMgr, jobs: JobMgr) -> PaperStatusResponse:
    try:
        run_id, _pid = _find_active_paper_job(jobs)
    except HTTPException:
        return PaperStatusResponse(state="idle", pnl_metrics=None, trades_count=0)

    params = _session_params(state, run_id)
    trader_id = _trader_id_for_run(state, run_id)
    testnet = params.get("testnet")
    checkpoint = _load_latest_for_trader(trader_id, _state_db_path(state))
    if checkpoint is None:
        return PaperStatusResponse(
            state="starting",
            trades_count=0,
            run_id=run_id,
            trader_id=trader_id,
            testnet=testnet if isinstance(testnet, bool) else None,
        )
    node_status = checkpoint.node_status or {}
    risk = node_status.get("risk") if isinstance(node_status.get("risk"), dict) else None
    return PaperStatusResponse(
        state=str(node_status.get("state", "unknown")),
        pnl_metrics=risk,
        trades_count=int(node_status.get("trades_today") or 0),
        run_id=run_id,
        trader_id=trader_id,
        testnet=testnet if isinstance(testnet, bool) else None,
        halt_reason=node_status.get("halt_reason"),
        message=node_status.get("error_message"),
    )


def _num(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


@router.get("/positions", response_model=list[PaperPositionResponse])
async def get_positions(
    state: StateMgr, trader_id: str | None = None
) -> list[PaperPositionResponse]:
    """Return open positions from the latest checkpoint for trader_id.

    Checkpoints store NautilusTrader ``Position.to_dict()`` (instrument_id,
    side, quantity, avg_px_open) enriched with unrealized_pnl and leverage.
    """
    checkpoint = _load_latest_for_trader(trader_id, _state_db_path(state))
    if checkpoint is None:
        return []
    positions = getattr(checkpoint, "positions", {}) or {}
    result: list[PaperPositionResponse] = []
    for pos in positions.values():
        if not isinstance(pos, dict):
            continue
        result.append(
            PaperPositionResponse(
                symbol=str(pos.get("instrument_id") or pos.get("symbol") or ""),
                direction=str(pos.get("side") or pos.get("direction") or ""),
                quantity=_num(pos.get("quantity")) or 0.0,
                entry_price=_num(pos.get("avg_px_open") or pos.get("entry_price")) or 0.0,
                unrealized_pnl=_num(pos.get("unrealized_pnl")) or 0.0,
                leverage=_num(pos.get("leverage")) or 1.0,
            )
        )
    return result


@router.get("/orders", response_model=list[PaperOrderResponse])
async def get_orders(state: StateMgr, trader_id: str | None = None) -> list[PaperOrderResponse]:
    """Return open orders (``Order.to_dict()``) from the latest checkpoint."""
    checkpoint = _load_latest_for_trader(trader_id, _state_db_path(state))
    if checkpoint is None:
        return []
    orders = getattr(checkpoint, "orders", {}) or {}
    result: list[PaperOrderResponse] = []
    for oid, order in orders.items():
        if not isinstance(order, dict):
            continue
        price = _num(order.get("price"))
        if price is None:
            price = _num(order.get("trigger_price"))
        side = order.get("side")
        status = order.get("status")
        result.append(
            PaperOrderResponse(
                order_id=str(oid),
                symbol=str(order.get("instrument_id") or order.get("symbol") or ""),
                side=str(side) if side is not None else None,
                quantity=_num(order.get("quantity")),
                price=price,
                status=str(status) if status is not None else None,
            )
        )
    return result


@router.get("/checkpoints", response_model=list[CheckpointResponse])
async def get_checkpoints(
    state: StateMgr, trader_id: str | None = None, limit: int = 50
) -> list[CheckpointResponse]:
    """Return checkpoint history for a trader, newest first."""
    if not trader_id:
        return []
    persistence = StatePersistence(_state_db_path(state))
    try:
        checkpoints = persistence.list_checkpoints(trader_id, limit=limit)
    finally:
        persistence.close()
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
async def get_session(trader_id: str, state: StateMgr) -> CheckpointResponse | None:
    checkpoint = _load_latest_for_trader(trader_id, _state_db_path(state))
    if checkpoint is None:
        return None
    node_status = checkpoint.node_status or {}
    return CheckpointResponse(
        timestamp=str(checkpoint.timestamp),
        state=node_status.get("state", "unknown"),
        halt_reason=node_status.get("halt_reason"),
        error_message=node_status.get("error_message"),
    )
