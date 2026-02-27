"""Discovery router — GA strategy discovery launch & management."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException

from vibe_quant.api.deps import get_job_manager, get_state_manager, get_ws_manager
from vibe_quant.api.schemas.discovery import (
    DiscoveryJobResponse,
    DiscoveryLaunchRequest,
    DiscoveryResultResponse,
    PromoteResponse,
    ReplayResponse,
)
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from vibe_quant.jobs.manager import JobInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/discovery", tags=["discovery"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]
JobMgr = Annotated[BacktestJobManager, Depends(get_job_manager)]
WsMgr = Annotated[ConnectionManager, Depends(get_ws_manager)]

# Hardcoded indicator pool; replaced by DSL catalog when available
_DEFAULT_INDICATOR_POOL: list[dict[str, object]] = [
    {"name": "SMA", "params": {"period": {"min": 5, "max": 200, "step": 5}}},
    {"name": "EMA", "params": {"period": {"min": 5, "max": 200, "step": 5}}},
    {"name": "RSI", "params": {"period": {"min": 7, "max": 28, "step": 1}}},
    {
        "name": "MACD",
        "params": {
            "fast": {"min": 8, "max": 21},
            "slow": {"min": 21, "max": 55},
            "signal": {"min": 5, "max": 13},
        },
    },
    {
        "name": "BollingerBands",
        "params": {
            "period": {"min": 10, "max": 50, "step": 5},
            "std_dev": {"min": 1.5, "max": 3.0, "step": 0.25},
        },
    },
    {"name": "ATR", "params": {"period": {"min": 7, "max": 28, "step": 1}}},
    {
        "name": "Stochastic",
        "params": {"k_period": {"min": 5, "max": 21}, "d_period": {"min": 3, "max": 9}},
    },
    {"name": "ADX", "params": {"period": {"min": 7, "max": 28, "step": 1}}},
    {"name": "CCI", "params": {"period": {"min": 10, "max": 40, "step": 5}}},
    {"name": "VWAP", "params": {}},
]


def _read_progress_file(run_id: int) -> dict[str, object] | None:
    """Read progress JSON written by the discovery pipeline subprocess."""
    import json
    from pathlib import Path

    path = Path(f"logs/discovery_{run_id}_progress.json")
    if not path.exists():
        return None
    try:
        data: object = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _job_info_to_discovery_response(info: JobInfo) -> DiscoveryJobResponse:
    progress = _read_progress_file(info.run_id)
    return DiscoveryJobResponse(
        run_id=info.run_id,
        status=info.status.value,
        started_at=info.started_at.isoformat() if info.started_at else None,
        progress=progress,
    )


# --- Launch ---


@router.post("/launch", response_model=DiscoveryJobResponse, status_code=201)
async def launch_discovery(
    body: DiscoveryLaunchRequest,
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
) -> DiscoveryJobResponse:
    params: dict[str, object] = {
        "population": body.population,
        "generations": body.generations,
        "mutation_rate": body.mutation_rate,
        "crossover_rate": body.crossover_rate,
        "elite_count": body.elite_count,
        "tournament_size": body.tournament_size,
        "convergence_generations": body.convergence_generations,
    }
    if body.indicator_pool is not None:
        params["indicator_pool"] = body.indicator_pool

    # Use strategy_id=None for discovery (no pre-existing strategy)
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=body.symbols,
        timeframe=body.timeframes[0] if body.timeframes else "1h",
        start_date=body.start_date or "",
        end_date=body.end_date or "",
        parameters=params,
    )

    log_file = f"logs/discovery_{run_id}.log"
    symbols_str = ",".join(body.symbols)
    timeframe = body.timeframes[0] if body.timeframes else "4h"
    start_date = body.start_date or "2024-01-01"
    end_date = body.end_date or "2026-02-24"
    command = [
        sys.executable,
        "-m",
        "vibe_quant",
        "discovery",
        "--run-id",
        str(run_id),
        "--population-size",
        str(body.population),
        "--max-generations",
        str(body.generations),
        "--mutation-rate",
        str(body.mutation_rate),
        "--crossover-rate",
        str(body.crossover_rate),
        "--elite-count",
        str(body.elite_count),
        "--tournament-size",
        str(body.tournament_size),
        "--convergence-generations",
        str(body.convergence_generations),
        "--max-workers",
        "4",
        "--symbols",
        symbols_str,
        "--timeframe",
        timeframe,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
    ]
    if body.indicator_pool is not None:
        command.extend(["--indicator-pool", ",".join(body.indicator_pool)])
    if body.direction is not None:
        command.extend(["--direction", body.direction])

    try:
        pid = jobs.start_job(run_id, "discovery", command, log_file=log_file)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    state.update_backtest_run_status(run_id, "running", pid=pid)
    logger.info("discovery job started run_id=%d pid=%d", run_id, pid)

    await ws.broadcast("jobs", {"type": "job_started", "run_id": run_id, "job_type": "discovery"})

    info = jobs.get_job_info(run_id)
    if info is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Job disappeared after creation")
    return _job_info_to_discovery_response(info)


# --- Job management ---


@router.get("/jobs", response_model=list[DiscoveryJobResponse])
async def list_discovery_jobs(jobs: JobMgr) -> list[DiscoveryJobResponse]:
    all_jobs = jobs.list_all_jobs(job_type="discovery")
    return [_job_info_to_discovery_response(j) for j in all_jobs]


@router.get("/jobs/{run_id}/progress")
async def get_discovery_progress(run_id: int, jobs: JobMgr) -> dict[str, object]:
    """Get discovery job progress from file written by subprocess."""
    info = jobs.get_job_info(run_id)
    if info is None or info.job_type != "discovery":
        raise HTTPException(status_code=404, detail="Discovery job not found")
    progress = _read_progress_file(run_id)
    return {
        "run_id": run_id,
        "status": info.status.value,
        "progress": progress,
    }


@router.delete("/jobs/{run_id}", status_code=204)
async def kill_discovery_job(run_id: int, jobs: JobMgr, ws: WsMgr) -> None:
    info = jobs.get_job_info(run_id)
    if info is None or info.job_type != "discovery":
        raise HTTPException(status_code=404, detail="Discovery job not found")
    killed = jobs.kill_job(run_id)
    if not killed:
        raise HTTPException(status_code=404, detail="Job not running")
    logger.info("discovery job killed run_id=%d", run_id)
    await ws.broadcast("jobs", {"type": "job_killed", "run_id": run_id})


# --- Results ---


def _load_discovery_strategies(state: StateManager, run_id: int) -> list[dict[str, object]]:
    """Load discovered strategies from backtest_results notes JSON."""
    import json

    result = state.get_backtest_result(run_id)
    if result is None:
        return []
    notes = result.get("notes", "")
    if not notes or not isinstance(notes, str):
        return []
    try:
        data = json.loads(notes)
        if isinstance(data, dict) and "top_strategies" in data:
            strategies = data["top_strategies"]
            if isinstance(strategies, list):
                return strategies  # type narrowed to list
        return []
    except (json.JSONDecodeError, TypeError):
        pass
    return []


@router.get("/results/latest", response_model=DiscoveryResultResponse)
async def get_latest_results(state: StateMgr) -> DiscoveryResultResponse:
    import json as _json

    # Single query: join runs with results, get notes directly
    conn = state.conn
    rows = conn.execute(
        """SELECT br.notes FROM backtest_runs r
           JOIN backtest_results br ON br.run_id = r.id
           WHERE r.run_mode='discovery' AND r.status='completed'
             AND br.notes IS NOT NULL AND br.notes != ''
           ORDER BY r.id DESC LIMIT 10"""
    ).fetchall()
    for row in rows:
        try:
            data = _json.loads(row[0])
            if isinstance(data, dict) and "top_strategies" in data:
                strategies = data["top_strategies"]
                if strategies and any(s.get("trades", 0) > 0 for s in strategies):
                    return DiscoveryResultResponse(strategies=strategies)
        except (ValueError, TypeError):
            continue
    # Fallback: return first valid result even if 0 trades
    for row in rows:
        try:
            data = _json.loads(row[0])
            if isinstance(data, dict) and "top_strategies" in data:
                return DiscoveryResultResponse(strategies=data["top_strategies"])
        except (ValueError, TypeError):
            continue
    return DiscoveryResultResponse(strategies=[])


@router.get("/results/{run_id}", response_model=DiscoveryResultResponse)
async def get_discovery_results(run_id: int, state: StateMgr) -> DiscoveryResultResponse:
    return DiscoveryResultResponse(strategies=_load_discovery_strategies(state, run_id))


@router.post("/results/{run_id}/export/{strategy_index}", status_code=201)
async def export_discovered_strategy(
    run_id: int,
    strategy_index: int,
    state: StateMgr,
) -> dict[str, object]:
    """Export a discovered strategy to the strategies table."""
    strategies = _load_discovery_strategies(state, run_id)
    if strategy_index < 0 or strategy_index >= len(strategies):
        raise HTTPException(status_code=404, detail="Strategy index out of range")

    entry = strategies[strategy_index]
    dsl_raw = entry.get("dsl", {})
    dsl: dict[str, object] = dsl_raw if isinstance(dsl_raw, dict) else {}
    name = str(dsl.get("name", f"discovery_{run_id}_{strategy_index}"))

    # Check if strategy name already exists
    existing = state.conn.execute("SELECT id FROM strategies WHERE name = ?", (name,)).fetchone()
    if existing:
        return {"status": "exists", "strategy_id": existing[0], "name": name}

    import json

    _score_raw = entry.get("score", 0)
    _score: float = _score_raw if isinstance(_score_raw, (int, float)) else 0.0
    cursor = state.conn.execute(
        "INSERT INTO strategies (name, description, dsl_config, strategy_type) VALUES (?, ?, ?, ?)",
        (
            name,
            f"Discovered via GA run {run_id} (score={_score:.4f})",
            json.dumps(dsl),
            dsl.get("strategy_type", "momentum"),
        ),
    )
    state.conn.commit()
    return {"status": "created", "strategy_id": cursor.lastrowid, "name": name}


# --- Promote & Replay ---


def _get_discovery_run_config(state: StateManager, run_id: int) -> dict[str, object]:
    """Load discovery run and validate it exists with mode=discovery."""
    run = state.get_backtest_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Discovery run not found")
    if run.get("run_mode") != "discovery":
        raise HTTPException(status_code=400, detail="Run is not a discovery run")
    return run


def _get_genome_entry(
    state: StateManager, run_id: int, strategy_index: int
) -> dict[str, object]:
    """Load a specific genome entry from discovery results."""
    strategies = _load_discovery_strategies(state, run_id)
    if strategy_index < 0 or strategy_index >= len(strategies):
        raise HTTPException(status_code=404, detail="Strategy index out of range")
    return strategies[strategy_index]


def _launch_backtest_job(
    state: StateManager,
    jobs: BacktestJobManager,
    run_id: int,
    job_type: str,
) -> int:
    """Start a screening/validation subprocess for a backtest run."""
    log_file = f"logs/{job_type}_{run_id}.log"
    command = [
        sys.executable,
        "-m",
        "vibe_quant",
        job_type,
        "run",
        "--run-id",
        str(run_id),
    ]
    try:
        pid = jobs.start_job(run_id, job_type, command, log_file=log_file)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    state.update_backtest_run_status(run_id, "running", pid=pid)
    return pid


@router.post(
    "/results/{run_id}/promote/{strategy_index}",
    response_model=PromoteResponse,
    status_code=201,
)
async def promote_discovered_strategy(
    run_id: int,
    strategy_index: int,
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
    mode: str = "screening",
) -> PromoteResponse:
    """Export genome as strategy and launch screening/validation backtest."""
    if mode not in ("screening", "validation"):
        raise HTTPException(status_code=400, detail="mode must be 'screening' or 'validation'")

    discovery_run = _get_discovery_run_config(state, run_id)
    entry = _get_genome_entry(state, run_id, strategy_index)

    # Export genome to strategies table (reuse export logic)
    import json

    dsl_raw = entry.get("dsl", {})
    dsl: dict[str, object] = dsl_raw if isinstance(dsl_raw, dict) else {}
    name = str(dsl.get("name", f"discovery_{run_id}_{strategy_index}"))

    existing = state.conn.execute("SELECT id FROM strategies WHERE name = ?", (name,)).fetchone()
    if existing:
        strategy_id: int = existing[0]
    else:
        _score_raw = entry.get("score", 0)
        _score: float = _score_raw if isinstance(_score_raw, (int, float)) else 0.0
        cursor = state.conn.execute(
            "INSERT INTO strategies (name, description, dsl_config, strategy_type) VALUES (?, ?, ?, ?)",
            (
                name,
                f"Discovered via GA run {run_id} (score={_score:.4f})",
                json.dumps(dsl),
                dsl.get("strategy_type", "momentum"),
            ),
        )
        state.conn.commit()
        strategy_id = cursor.lastrowid or 0

    # Create backtest run using discovery run's symbols/timeframe/dates
    symbols_raw = discovery_run.get("symbols", [])
    symbols_list: list[str] = json.loads(symbols_raw) if isinstance(symbols_raw, str) else list(symbols_raw) if isinstance(symbols_raw, list) else []
    backtest_run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode=mode,
        symbols=symbols_list,
        timeframe=str(discovery_run.get("timeframe", "4h")),
        start_date=str(discovery_run.get("start_date", "")),
        end_date=str(discovery_run.get("end_date", "")),
        parameters={},
    )

    pid = _launch_backtest_job(state, jobs, backtest_run_id, mode)
    logger.info("promote: strategy=%d %s run=%d pid=%d", strategy_id, mode, backtest_run_id, pid)
    await ws.broadcast("jobs", {"type": "job_started", "run_id": backtest_run_id, "job_type": mode})

    return PromoteResponse(
        strategy_id=strategy_id,
        run_id=backtest_run_id,
        name=name,
        mode=mode,
    )


@router.post(
    "/results/{run_id}/replay/{strategy_index}",
    response_model=ReplayResponse,
    status_code=201,
)
async def replay_discovered_strategy(
    run_id: int,
    strategy_index: int,
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
) -> ReplayResponse:
    """Re-run genome through screening to verify discovery metrics match."""
    import json

    discovery_run = _get_discovery_run_config(state, run_id)
    entry = _get_genome_entry(state, run_id, strategy_index)

    dsl_raw = entry.get("dsl", {})
    dsl: dict[str, object] = dsl_raw if isinstance(dsl_raw, dict) else {}

    # Use dsl_override so screening CLI uses DSL directly (no strategy needed)
    symbols_raw = discovery_run.get("symbols", [])
    symbols_list: list[str] = json.loads(symbols_raw) if isinstance(symbols_raw, str) else list(symbols_raw) if isinstance(symbols_raw, list) else []

    from vibe_quant.dsl.translator import translate_dsl_config

    dsl_name = str(dsl.get("name", f"replay_{run_id}_{strategy_index}"))
    translated = translate_dsl_config(dsl, strategy_name=dsl_name)

    replay_run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="screening",
        symbols=symbols_list,
        timeframe=str(discovery_run.get("timeframe", "4h")),
        start_date=str(discovery_run.get("start_date", "")),
        end_date=str(discovery_run.get("end_date", "")),
        parameters={"dsl_override": translated},
    )

    pid = _launch_backtest_job(state, jobs, replay_run_id, "screening")
    logger.info("replay: discovery=%d genome=%d screening=%d pid=%d", run_id, strategy_index, replay_run_id, pid)
    await ws.broadcast("jobs", {"type": "job_started", "run_id": replay_run_id, "job_type": "screening"})

    return ReplayResponse(replay_run_id=replay_run_id, original_run_id=run_id)


# --- Indicator pool ---


@router.get("/indicator-pool")
async def get_indicator_pool() -> list[dict[str, object]]:
    try:
        from vibe_quant.dsl import indicators as _ind_mod

        catalog: list[dict[str, object]] = getattr(
            _ind_mod, "INDICATOR_CATALOG", _DEFAULT_INDICATOR_POOL
        )
        return catalog
    except ImportError:
        return _DEFAULT_INDICATOR_POOL
