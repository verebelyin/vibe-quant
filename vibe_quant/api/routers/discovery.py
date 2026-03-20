"""Discovery router — GA strategy discovery launch & management."""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime, timedelta
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

# Maximum concurrent discovery jobs. Prevents resource contention
# (each discovery run spawns --max-workers backtests in parallel).
_MAX_CONCURRENT_DISCOVERIES = 5
_REGIME_RETURN_THRESHOLD = 0.05

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


def _job_info_to_discovery_response(
    info: JobInfo, state: StateManager | None = None
) -> DiscoveryJobResponse:
    progress = _read_progress_file(info.run_id)
    resp = DiscoveryJobResponse(
        run_id=info.run_id,
        status=info.status.value,
        started_at=info.started_at.isoformat() if info.started_at else None,
        completed_at=info.completed_at.isoformat() if info.completed_at else None,
        progress=progress,
    )
    if state is not None:
        run = state.get_backtest_run(info.run_id)
        if run:
            resp.symbols = run.get("symbols")
            resp.timeframe = run.get("timeframe")
            params = run.get("parameters", {})
            resp.generations = params.get("generations")
            resp.population = params.get("population")
            resp.error_message = run.get("error_message")
            strategies = _load_discovery_strategies(state, info.run_id)
            resp.strategies_found = len(strategies) if strategies else None
    return resp


# --- Helpers ---


def _load_discovery_payload(state: StateManager, run_id: int) -> dict[str, object]:
    """Load raw discovery payload from backtest_results notes JSON."""
    import json

    result = state.get_backtest_result(run_id)
    if result is None:
        return {}

    notes = result.get("notes", "")
    if not notes or not isinstance(notes, str):
        return {}

    try:
        data = json.loads(notes)
    except (json.JSONDecodeError, TypeError):
        return {}

    return data if isinstance(data, dict) else {}


def _sync_discovery_statuses(jobs: BacktestJobManager) -> list[JobInfo]:
    """Sync status for all running discovery jobs and return the still-running ones.

    Detects dead processes (OOM-killed, crashed) that are still marked
    'running' in the DB and updates them to 'failed'.
    """
    all_discovery = jobs.list_all_jobs(job_type="discovery")
    running: list[JobInfo] = []
    for job in all_discovery:
        if job.status.value != "running":
            continue
        synced = jobs.sync_job_status(job.run_id)
        if synced == "running":
            running.append(job)
    return running


# --- Launch ---


@router.post("/launch", response_model=DiscoveryJobResponse, status_code=201)
async def launch_discovery(
    body: DiscoveryLaunchRequest,
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
) -> DiscoveryJobResponse:
    # Guard: refuse if too many discovery jobs already running
    running = _sync_discovery_statuses(jobs)
    if len(running) >= _MAX_CONCURRENT_DISCOVERIES:
        run_ids = [j.run_id for j in running]
        raise HTTPException(
            status_code=409,
            detail=(
                f"{len(running)} discovery jobs already running (run_ids={run_ids}). "
                f"Kill existing jobs via DELETE /api/discovery/jobs/{{run_id}} "
                f"or wait for them to complete."
            ),
        )

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
    if body.direction is not None:
        params["direction"] = body.direction
    if body.train_test_split > 0:
        params["train_test_split"] = body.train_test_split
    if body.cross_window_months:
        params["cross_window_months"] = body.cross_window_months
        params["cross_window_min_sharpe"] = body.cross_window_min_sharpe
    if body.num_seeds > 1:
        params["num_seeds"] = body.num_seeds
    if body.wfa_oos_step_days > 0:
        params["wfa_oos_step_days"] = body.wfa_oos_step_days
        params["wfa_min_consistency"] = body.wfa_min_consistency

    symbols_str = ",".join(body.symbols)
    timeframe = body.timeframes[0] if body.timeframes else "4h"
    today = datetime.now()
    start_date = body.start_date or (today - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = body.end_date or today.strftime("%Y-%m-%d")

    # Use strategy_id=None for discovery (no pre-existing strategy)
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=body.symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        parameters=params,
    )

    _ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/discovery_{run_id}_{_ts}.log"
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
    if body.train_test_split > 0:
        command.extend(["--train-test-split", str(body.train_test_split)])
    if body.cross_window_months:
        command.extend(["--cross-window-months", ",".join(str(m) for m in body.cross_window_months)])
        command.extend(["--cross-window-min-sharpe", str(body.cross_window_min_sharpe)])
    if body.num_seeds > 1:
        command.extend(["--num-seeds", str(body.num_seeds)])
    if body.wfa_oos_step_days > 0:
        command.extend(["--wfa-oos-step-days", str(body.wfa_oos_step_days)])
        command.extend(["--wfa-min-consistency", str(body.wfa_min_consistency)])

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
    return _job_info_to_discovery_response(info, state)


# --- Job management ---


@router.get("/jobs", response_model=list[DiscoveryJobResponse])
async def list_discovery_jobs(jobs: JobMgr, state: StateMgr) -> list[DiscoveryJobResponse]:
    # Sync running jobs to detect dead processes before returning status
    _sync_discovery_statuses(jobs)
    all_jobs = jobs.list_all_jobs(job_type="discovery")
    return [_job_info_to_discovery_response(j, state) for j in all_jobs]


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
async def kill_discovery_job(run_id: int, jobs: JobMgr, state: StateMgr, ws: WsMgr) -> None:
    info = jobs.get_job_info(run_id)
    if info is None or info.job_type != "discovery":
        raise HTTPException(status_code=404, detail="Discovery job not found")
    killed = jobs.kill_job(run_id)
    if not killed:
        raise HTTPException(status_code=404, detail="Job not running")
    state.update_backtest_run_status(run_id, "killed")
    logger.info("discovery job killed run_id=%d", run_id)
    await ws.broadcast("jobs", {"type": "job_killed", "run_id": run_id})


# --- Results ---


def _load_discovery_strategies(state: StateManager, run_id: int) -> list[dict[str, object]]:
    """Load discovered strategies from backtest_results notes JSON."""
    data = _load_discovery_payload(state, run_id)
    strategies = data.get("top_strategies")
    if isinstance(strategies, list):
        return strategies  # type narrowed to list
    return []


def _discovery_entry_is_short_only(entry: dict[str, object]) -> bool:
    """Return True when a discovery result only trades the short side."""
    chrom_raw = entry.get("chromosome")
    if isinstance(chrom_raw, dict):
        direction = chrom_raw.get("direction")
        if direction == "short":
            return True
        if direction in {"long", "both"}:
            return False

    dsl_raw = entry.get("dsl")
    if not isinstance(dsl_raw, dict):
        return False

    entry_conditions = dsl_raw.get("entry_conditions")
    if isinstance(entry_conditions, dict):
        has_long = bool(entry_conditions.get("long"))
        has_short = bool(entry_conditions.get("short"))
        return has_short and not has_long

    return False


def _window_regime_sign(symbol: str, start_date: str, end_date: str) -> int:
    """Classify a window as bull (+1), bear (-1), or neutral (0)."""
    from vibe_quant.validation.random_baseline import load_ohlc

    bars = load_ohlc(symbol, "1m", start_date, end_date)
    if len(bars) < 2 or bars[0].open <= 0:
        return 0

    total_return = (bars[-1].close - bars[0].open) / bars[0].open
    if abs(total_return) < _REGIME_RETURN_THRESHOLD:
        return 0
    return 1 if total_return > 0 else -1


def _shift_window(start_date: str, end_date: str, months: int) -> tuple[str, str]:
    """Shift a date window by N months."""
    from datetime import datetime as _dt

    from dateutil.relativedelta import relativedelta

    start = _dt.strptime(start_date, "%Y-%m-%d")
    end = _dt.strptime(end_date, "%Y-%m-%d")
    shifted_start = (start + relativedelta(months=months)).strftime("%Y-%m-%d")
    shifted_end = (end + relativedelta(months=months)).strftime("%Y-%m-%d")
    return shifted_start, shifted_end


def _passes_opposing_regime_gate(
    state: StateManager,
    run_id: int,
    discovery_run: dict[str, object],
    entry: dict[str, object],
) -> bool:
    """Check whether a 1m short strategy passed on at least one opposing regime."""
    payload = _load_discovery_payload(state, run_id)
    offsets_raw = payload.get("cross_window_months")
    cross_window_raw = entry.get("cross_window")

    if not isinstance(offsets_raw, list) or not offsets_raw:
        return False
    if not isinstance(cross_window_raw, dict):
        return False

    windows_raw = cross_window_raw.get("windows")
    if not isinstance(windows_raw, list) or len(windows_raw) < 2:
        return False

    symbol = ""
    symbols_raw = discovery_run.get("symbols", [])
    if isinstance(symbols_raw, list) and symbols_raw:
        symbol = str(symbols_raw[0])
    elif isinstance(symbols_raw, str) and symbols_raw:
        import json

        try:
            parsed = json.loads(symbols_raw)
        except (json.JSONDecodeError, TypeError):
            parsed = [symbols_raw]
        if isinstance(parsed, list) and parsed:
            symbol = str(parsed[0])
    if not symbol:
        return False

    start_date = str(discovery_run.get("start_date", ""))
    end_date = str(discovery_run.get("end_date", ""))
    if not start_date or not end_date:
        return False

    base_sign = _window_regime_sign(symbol, start_date, end_date)
    if base_sign == 0:
        return False

    min_sharpe_raw = payload.get("cross_window_min_sharpe", 0.5)
    try:
        min_sharpe = float(min_sharpe_raw)
    except (TypeError, ValueError):
        min_sharpe = 0.5

    max_idx = min(len(offsets_raw), len(windows_raw) - 1)
    for idx in range(max_idx):
        try:
            months = int(offsets_raw[idx])
        except (TypeError, ValueError):
            continue

        shifted_start, shifted_end = _shift_window(start_date, end_date, months)
        shifted_sign = _window_regime_sign(symbol, shifted_start, shifted_end)
        if shifted_sign == 0 or shifted_sign == base_sign:
            continue

        shifted_window = windows_raw[idx + 1]
        if not isinstance(shifted_window, dict):
            continue

        try:
            sharpe = float(shifted_window.get("sharpe", 0.0) or 0.0)
            total_return = float(shifted_window.get("return_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue

        if sharpe >= min_sharpe and total_return > 0:
            return True

    return False


def _enforce_short_1m_cross_regime_gate(
    state: StateManager,
    run_id: int,
    discovery_run: dict[str, object],
    entry: dict[str, object],
) -> None:
    """Block promotion of fragile 1m short champions without opposing-regime proof."""
    if discovery_run.get("timeframe") != "1m":
        return
    if not _discovery_entry_is_short_only(entry):
        return
    if _passes_opposing_regime_gate(state, run_id, discovery_run, entry):
        return

    raise HTTPException(
        status_code=409,
        detail=(
            "1m short champions must pass at least one opposing-regime cross-window "
            "validation before promotion. Re-run discovery with shifted windows "
            "covering an opposite BTC market regime."
        ),
    )


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
    _ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/{job_type}_{run_id}_{_ts}.log"
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
    _enforce_short_1m_cross_regime_gate(state, run_id, discovery_run, entry)

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
