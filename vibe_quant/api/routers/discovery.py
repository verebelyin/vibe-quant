"""Discovery router — GA strategy discovery launch & management."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, NoReturn, cast

from fastapi import APIRouter, Depends, HTTPException

from vibe_quant.api import data_precheck, mark_precheck
from vibe_quant.api.deps import get_job_manager, get_state_manager, get_ws_manager
from vibe_quant.api.schemas.discovery import (
    DiscoveryJobResponse,
    DiscoveryLaunchRequest,
    DiscoveryResultResponse,
    PromoteResponse,
    ReplayResponse,
)
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.result_notes import parse_notes
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import gate_chain
from vibe_quant.discovery.record import (
    ChampionBankrollInvalid,
    ChildParams,
    DiscoveryRecord,
    RecordUnreadable,
)
from vibe_quant.discovery.run_spec import DiscoveryRunSpec
from vibe_quant.discovery.seed import load_seed_chromosomes
from vibe_quant.dsl.identity import dsl_body_key
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.utils import log_dir

if TYPE_CHECKING:
    from vibe_quant.jobs.manager import JobInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/discovery", tags=["discovery"])

# Maximum concurrent discovery jobs. Prevents resource contention
# (each discovery run spawns --max-workers backtests in parallel).
_MAX_CONCURRENT_DISCOVERIES = 5

StateMgr = Annotated[StateManager, Depends(get_state_manager)]
JobMgr = Annotated[BacktestJobManager, Depends(get_job_manager)]
WsMgr = Annotated[ConnectionManager, Depends(get_ws_manager)]



def _read_progress_file(run_id: int) -> dict[str, object] | None:
    """Read progress JSON written by the discovery pipeline subprocess."""
    import json

    path = log_dir(create=False) / f"discovery_{run_id}_progress.json"
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
            resp.num_seeds = params.get("num_seeds")
            resp.wfa_oos_step_days = params.get("wfa_oos_step_days")
            resp.cross_window_months = params.get("cross_window_months")
            resp.eval_windows = params.get("eval_windows")
            resp.direction = params.get("direction")
            resp.error_message = run.get("error_message")
            record = DiscoveryRecord.load(state, info.run_id)
            resp.seed = record.seed()
            strategies = record.top_strategies()
            resp.strategies_found = len(strategies) if strategies else None
            if strategies:
                top = strategies[0]
                if isinstance(top, dict):
                    sharpe = top.get("sharpe")
                    ret_pct = top.get("return_pct")
                    resp.best_sharpe = float(sharpe) if isinstance(sharpe, (int, float)) else None
                    resp.best_return = (
                        float(ret_pct) if isinstance(ret_pct, (int, float)) else None
                    )
    return resp


# --- Helpers ---


def _validate_seed_run_compiler(state: StateManager, seed_run_id: int) -> None:
    """Reject warm-start when the seed run's compiler hash != current.

    Also rejects a seed run that has no loadable champions: launching it would
    create a run row and then fail 1 s later in the subprocess with a
    ``ValueError``, so the API refuses it up front (same 400 family as the
    sibling seed errors).
    """
    from vibe_quant.dsl.compiler import compiler_version_hash

    run = state.get_backtest_run(seed_run_id)
    if run is None:
        raise HTTPException(
            status_code=400,
            detail=f"Seed run {seed_run_id} not found",
        )
    params = run.get("parameters") or {}
    record = DiscoveryRecord.load(state, seed_run_id)
    seed_hash = params.get("compiler_version") or record.compiler_version()
    if not seed_hash:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Seed run {seed_run_id} has no recorded compiler_version — "
                "cannot verify warm-start compatibility"
            ),
        )
    current_hash = compiler_version_hash()
    if seed_hash != current_hash:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Seed run {seed_run_id} compiler_version={seed_hash} does not "
                f"match current {current_hash}; warm-start would be stale"
            ),
        )
    if load_seed_chromosomes(state, seed_run_id) is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Seed run {seed_run_id} has no persisted champions to seed "
                "from (guardrails may have rejected all candidates); launch "
                "without seed_run_id instead"
            ),
        )


class _CliArgsError(Exception):
    """Our own discovery argv failed to parse."""


def _precheck_discovery_config(cli_args: list[str]) -> None:
    """422 on a bad window/gate config; 500 if our own argv does not parse."""
    from vibe_quant.discovery.config_check import DiscoveryConfigError
    from vibe_quant.discovery.run_spec import build_parser

    parser = build_parser()

    def _raise(message: str) -> NoReturn:
        # argparse would print usage to the process-global stderr and exit
        raise _CliArgsError(message)

    parser.error = _raise  # type: ignore[method-assign]
    try:
        args = parser.parse_args(["--run-id", "0", *cli_args])
    except (_CliArgsError, SystemExit) as exc:
        # the API built a malformed command
        raise HTTPException(
            status_code=500,
            detail=f"Internal error building discovery command: {exc}",
        ) from None
    try:
        DiscoveryRunSpec.from_args(args).plan()
    except DiscoveryConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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


def _refuse_if_discovery_full(jobs: BacktestJobManager) -> None:
    """409 when the max concurrent discoveries are already running."""
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


@router.post("/launch", response_model=DiscoveryJobResponse, status_code=201)
async def launch_discovery(
    body: DiscoveryLaunchRequest,
    state: StateMgr,
    jobs: JobMgr,
    ws: WsMgr,
) -> DiscoveryJobResponse:
    # Guard: refuse if too many discovery jobs already running
    _refuse_if_discovery_full(jobs)

    # Warm-start: validate that the seed run's compiler hash matches the
    # current compiler. A mismatch means chromosomes would be evaluated by a
    # compiler they weren't selected under, silently invalidating warm-start.
    if body.seed_run_id is not None:
        _validate_seed_run_compiler(state, body.seed_run_id)

    # Built once: the stored params, the argv and the precheck all read these.
    settings = body.liquidation_settings()
    today = datetime.now()
    start_date = body.start_date or (today - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = body.end_date or today.strftime("%Y-%m-%d")
    spec = DiscoveryRunSpec.from_request(body, start_date=start_date, end_date=end_date)
    params = spec.to_params()
    timeframe = spec.timeframe

    # Precheck BEFORE any run row exists: parse the exact argv the subprocess
    # will get and derive its windows (same code path as the CLI).
    cli_args = spec.to_argv()
    # Refuse a pool the GA genome cannot use via the SAME validator the
    # pipeline runs, before any run row or job exists (vibe-quant-l42df).
    if body.indicator_pool is not None:
        from vibe_quant.discovery.operators import validate_indicator_pool

        try:
            validate_indicator_pool(body.indicator_pool)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    _precheck_discovery_config(cli_args)

    # The run fills on ticks for its timeframe and needs aux data for every
    # context indicator in its pool; refuse before any run row exists rather
    # than failing the run inside a worker (vibe-quant-b5xfj).
    await asyncio.to_thread(
        data_precheck.precheck_discovery_data,
        timeframe,
        body.symbols,
        body.indicator_pool,
        start_date,
        end_date,
    )

    # A mode-on run scans MARK 1m bars across every stage of the span; refuse
    # before any run row exists rather than failing the run in a worker
    # (vibe-quant-592lt).
    # A grid is always liquidation-on (its modes are isolated / cross only).
    grid_cells = body.bankroll_cells()
    if settings.on or grid_cells:
        await asyncio.to_thread(
            mark_precheck.precheck_discovery,
            settings.mode if settings.on else grid_cells[0].mode,
            body.symbols,
            start_date,
            end_date,
        )
    # The awaits above let another launch in: re-check capacity right before the
    # run row, with no await between here and start_job (M16).
    _refuse_if_discovery_full(jobs)

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
    log_file = str(log_dir() / f"discovery_{run_id}_{_ts}.log")
    command = [
        sys.executable,
        "-m",
        "vibe_quant",
        "discovery",
        "--run-id",
        str(run_id),
        *cli_args,
    ]

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


def _enforce_short_1m_cross_regime_gate(
    record: DiscoveryRecord,
    discovery_run: dict[str, object],
    entry: dict[str, object],
) -> None:
    """Block promotion of fragile 1m champions that can trade short without proof.

    Takes the request's ONE :class:`DiscoveryRecord`: a non-1m or long-only champion
    returns before the payload is ever read.
    """
    if discovery_run.get("timeframe") != "1m":
        return
    if not gate_chain.entry_trades_short(entry):
        return
    payload = cast("dict[str, object]", record.payload)
    if gate_chain.passes_opposing_regime_gate(
        payload, discovery_run, entry, regime_sign=gate_chain.window_regime_sign,
    ):
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
    # Single query: join runs with results, get notes directly
    notes_rows = state.list_recent_discovery_result_notes(limit=10)
    for raw in notes_rows:
        try:
            data = parse_notes(raw)
            if isinstance(data, dict) and "top_strategies" in data:
                strategies = data["top_strategies"]
                if strategies and any(s.get("trades", 0) > 0 for s in strategies):
                    return DiscoveryResultResponse(strategies=strategies)
        except (ValueError, TypeError):
            continue
    # Fallback: return first valid result even if 0 trades
    for raw in notes_rows:
        try:
            data = parse_notes(raw)
            if isinstance(data, dict) and "top_strategies" in data:
                return DiscoveryResultResponse(strategies=data["top_strategies"])
        except (ValueError, TypeError):
            continue
    return DiscoveryResultResponse(strategies=[])


@router.get("/results/{run_id}", response_model=DiscoveryResultResponse)
async def get_discovery_results(run_id: int, state: StateMgr) -> DiscoveryResultResponse:
    record = DiscoveryRecord.load(state, run_id)
    return DiscoveryResultResponse(
        strategies=record.top_strategies(),
        reason=record.reason(),
        guardrail_rejections=record.guardrail_rejections(),
    )


@router.post("/results/{run_id}/export/{strategy_index}", status_code=201)
async def export_discovered_strategy(
    run_id: int,
    strategy_index: int,
    state: StateMgr,
) -> dict[str, object]:
    """Export a discovered strategy to the strategies table."""
    strategies = DiscoveryRecord.load(state, run_id).top_strategies()
    if strategy_index < 0 or strategy_index >= len(strategies):
        raise HTTPException(status_code=404, detail="Strategy index out of range")

    entry = strategies[strategy_index]
    strategy_id, name, created = _find_or_create_discovered_strategy(
        state, run_id, strategy_index, entry,
    )
    return {
        "status": "created" if created else "exists",
        "strategy_id": strategy_id,
        "name": name,
    }


def _find_strategy_by_dsl(
    state: StateManager, dsl: dict[str, object]
) -> tuple[int, str] | None:
    """Active strategy whose DSL equals ``dsl`` (ignoring name), else None."""
    import json

    key = dsl_body_key(dsl)
    for strategy_id, name, dsl_config in state.list_active_strategy_dsl_rows():
        try:
            stored = json.loads(dsl_config) if isinstance(dsl_config, str) else dsl_config
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(stored, dict) and dsl_body_key(stored) == key:
            return int(strategy_id), str(name)
    return None


def _unique_strategy_name(state: StateManager, base: str) -> str:
    """``base`` if unused, else ``base_2``, ``base_3``, ... (names are UNIQUE)."""

    if not state.strategy_name_exists(base):
        return base
    k = 2
    while state.strategy_name_exists(f"{base}_{k}"):
        k += 1
    return f"{base}_{k}"


def _find_or_create_discovered_strategy(
    state: StateManager,
    run_id: int,
    strategy_index: int,
    entry: dict[str, object],
) -> tuple[int, str, bool]:
    """Strategy row for a discovered genome, matched by DSL content -- never by name.

    Genome names (``genome_<uid>``) used to collide between an elite and its
    mutant, so a by-name lookup could hand back ANOTHER genome's strategy and
    validate the wrong DSL (vibe-quant-e70tl.1). Same DSL -> reuse the row
    (idempotent re-promote); same name but different DSL -> new row with a
    disambiguated name.

    Returns:
        (strategy_id, name, created).
    """
    dsl_raw = entry.get("dsl", {})
    dsl: dict[str, object] = dict(dsl_raw) if isinstance(dsl_raw, dict) else {}
    if not dsl:
        raise HTTPException(status_code=422, detail="Discovery entry has no DSL")

    found = _find_strategy_by_dsl(state, dsl)
    if found is not None:
        return found[0], found[1], False

    base = str(dsl.get("name") or f"discovery_{run_id}_{strategy_index}")
    name = _unique_strategy_name(state, base)
    dsl["name"] = name  # strategy name and DSL name stay in sync

    _score_raw = entry.get("score", 0)
    _score: float = _score_raw if isinstance(_score_raw, (int, float)) else 0.0
    strategy_type = dsl.get("strategy_type", "momentum")
    strategy_id = state.create_strategy(
        name,
        dsl,
        description=f"Discovered via GA run {run_id} #{strategy_index} (score={_score:.4f})",
        strategy_type=strategy_type if isinstance(strategy_type, str) else None,
    )
    return strategy_id, name, True


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
    record: DiscoveryRecord, strategy_index: int
) -> dict[str, object]:
    """Bounds-check the record's stored strategies and return the indexed entry.

    The element itself is unchecked: an in-range null still reaches ``entry.get`` and
    raises ``AttributeError`` as it always did (chief ruling D5, R38d).
    """
    strategies = record.top_strategies()
    if strategy_index < 0 or strategy_index >= len(strategies):
        raise HTTPException(status_code=404, detail="Strategy index out of range")
    return strategies[strategy_index]


def _child_params(record: DiscoveryRecord, strategy_index: int) -> ChildParams:
    """The parent's inherited fee / liquidation / sizing, in that order.

    A record failure is the request's own 409 (unreadable notes) or 422 (a grid
    champion with no valid bankroll), with the detail text byte-identical.
    """
    try:
        return record.child_params(strategy_index)
    except RecordUnreadable as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    except ChampionBankrollInvalid as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc


def _launch_backtest_job(
    state: StateManager,
    jobs: BacktestJobManager,
    run_id: int,
    job_type: str,
) -> int:
    """Start a screening/validation subprocess for a backtest run."""
    _ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_file = str(log_dir() / f"{job_type}_{run_id}_{_ts}.log")
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
    validation_range: str = "holdout",
) -> PromoteResponse:
    """Export genome as strategy and launch screening/validation backtest.

    ``mode=validation`` validates on the discovery run's HOLDOUT (out-of-sample)
    range by default; ``validation_range=full`` opts into the full discovery
    range (mostly in-sample). Runs without a holdout always use the full range.
    ``mode=screening`` is a like-for-like replay of the full discovery range
    (the replay-drift check compares it with the full-range headline). The
    range used is recorded in the run's ``promote_source``.
    """
    if mode not in ("screening", "validation"):
        raise HTTPException(status_code=400, detail="mode must be 'screening' or 'validation'")
    if validation_range not in ("holdout", "full"):
        raise HTTPException(
            status_code=400, detail="validation_range must be 'holdout' or 'full'"
        )

    discovery_run = _get_discovery_run_config(state, run_id)
    record = DiscoveryRecord.load(state, run_id)
    entry = _get_genome_entry(record, strategy_index)
    _enforce_short_1m_cross_regime_gate(record, discovery_run, entry)

    # Resolve the parent's authoritative fee/liquidation notes BEFORE the
    # strategy row exists: an unreadable parent note fails closed (409) without
    # leaking a strategies row (vibe-quant-dtj5h). Sizing comes last, still
    # before the strategy row (S23b/S23d: an unreadable notes.sizing is a 409
    # with no strategies/backtest_runs row created).
    child = _child_params(record, strategy_index)

    # Export genome to strategies table (matched by DSL content, not name)
    import json

    strategy_id, name, _created = _find_or_create_discovered_strategy(
        state, run_id, strategy_index, entry,
    )

    # Create backtest run using discovery run's symbols/timeframe/dates
    symbols_raw = discovery_run.get("symbols", [])
    symbols_list: list[str] = json.loads(symbols_raw) if isinstance(symbols_raw, str) else list(symbols_raw) if isinstance(symbols_raw, list) else []

    start_date = str(discovery_run.get("start_date", ""))
    end_date = str(discovery_run.get("end_date", ""))
    date_range = "full"
    holdout_dates = record.holdout_dates()
    if mode == "validation" and validation_range == "holdout" and holdout_dates is not None:
        start_date, end_date = holdout_dates
        date_range = "holdout"

    # promote_source: lets the post-run drift check (bd-l6ml) find the genome
    # and records WHICH range was validated (OOS holdout vs full range).
    # The child inherits the parent's fee schedule (bd vibe-quant-2gyk1.4);
    # its child params were resolved above, before the strategy row.
    run_parameters: dict[str, object] = {
        "promote_source": {
            "discovery_run_id": run_id,
            "strategy_index": strategy_index,
            "date_range": date_range,
            "holdout_dates": list(holdout_dates) if holdout_dates else None,
            "fee_schedule_source": child.fee_source,
        },
        **child.run_parameters(),
    }
    backtest_run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode=mode,
        symbols=symbols_list,
        timeframe=str(discovery_run.get("timeframe", "4h")),
        start_date=start_date,
        end_date=end_date,
        parameters=run_parameters,
    )

    pid = _launch_backtest_job(state, jobs, backtest_run_id, mode)
    logger.info(
        "promote: strategy=%d %s run=%d range=%s %s..%s pid=%d",
        strategy_id, mode, backtest_run_id, date_range, start_date, end_date, pid,
    )
    await ws.broadcast("jobs", {"type": "job_started", "run_id": backtest_run_id, "job_type": mode})

    return PromoteResponse(
        strategy_id=strategy_id,
        run_id=backtest_run_id,
        name=name,
        mode=mode,
        date_range=date_range,
        start_date=start_date,
        end_date=end_date,
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
    record = DiscoveryRecord.load(state, run_id)
    entry = _get_genome_entry(record, strategy_index)

    dsl_raw = entry.get("dsl", {})
    dsl: dict[str, object] = dsl_raw if isinstance(dsl_raw, dict) else {}

    # Use dsl_override so screening CLI uses DSL directly (no strategy needed)
    symbols_raw = discovery_run.get("symbols", [])
    symbols_list: list[str] = json.loads(symbols_raw) if isinstance(symbols_raw, str) else list(symbols_raw) if isinstance(symbols_raw, list) else []

    from vibe_quant.dsl.translator import translate_dsl_config

    dsl_name = str(dsl.get("name", f"replay_{run_id}_{strategy_index}"))
    translated = translate_dsl_config(dsl, strategy_name=dsl_name)

    # Replay runs the FULL window; discovery with eval_windows > 1 stored
    # worst-of-N sub-window fitness (min Sharpe / min return / max DD / min
    # PF over the windows, trades summed), so metrics legitimately differ.
    metrics_note: str | None = None
    run_params_raw = discovery_run.get("parameters")
    if isinstance(run_params_raw, str):
        run_params: dict[str, object] = json.loads(run_params_raw)
    elif isinstance(run_params_raw, dict):
        run_params = run_params_raw
    else:
        run_params = {}
    eval_windows = run_params.get("eval_windows")
    # CLI-launched runs carry symbol_agg only in the result notes.
    symbol_agg = run_params.get("symbol_agg")
    if symbol_agg is None:
        symbol_agg = record.symbol_agg()
    worst = symbol_agg == "worst" and len(symbols_list) > 1
    multi_window = isinstance(eval_windows, int) and eval_windows > 1
    parts: list[str] = []
    if multi_window:
        parts.append(
            f"Discovery run {run_id} scored fitness as worst-of-{eval_windows} "
            "eval windows (min Sharpe, min return, max drawdown, min PF across "
            "the train sub-windows; trades summed)"
        )
    if worst:
        parts.append(
            ("Each genome was" if not multi_window else "and each genome was")
            + f" scored by its WORST symbol in run {run_id} (symbol_agg=worst: "
            "per-symbol backtests; min Sharpe/return/PF, max drawdown, MIN trades "
            "across symbols)"
        )
    if parts:
        metrics_note = (
            "; ".join(parts)
            + "; this replay is one continuous shared-account portfolio backtest "
            "over the full range, so metrics differ from the stored champion "
            "fitness by design."
        )
        if not worst:
            metrics_note += " Compare against the champion's full_range_* headline instead."
        else:
            metrics_note += (
                " The full_range_* headline is also worst-of-symbols, so compare "
                "trades against full_range_trades_sum and read Sharpe as the worst symbol."
            )

    # Resolved BEFORE the run row: an unreadable parent note is a 409/422 with no
    # backtest_runs row created (S23b/S23d).
    child = _child_params(record, strategy_index)
    replay_run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="screening",
        symbols=symbols_list,
        timeframe=str(discovery_run.get("timeframe", "4h")),
        start_date=str(discovery_run.get("start_date", "")),
        end_date=str(discovery_run.get("end_date", "")),
        parameters={
            "dsl_override": translated,
            "replay_note": metrics_note,
            "fee_schedule_source": child.fee_source,
            **child.run_parameters(),
        },
    )

    pid = _launch_backtest_job(state, jobs, replay_run_id, "screening")
    logger.info("replay: discovery=%d genome=%d screening=%d pid=%d", run_id, strategy_index, replay_run_id, pid)
    await ws.broadcast("jobs", {"type": "job_started", "run_id": replay_run_id, "job_type": "screening"})

    return ReplayResponse(
        replay_run_id=replay_run_id,
        original_run_id=run_id,
        metrics_note=metrics_note,
    )


# --- Indicator pool ---


@router.get("/indicator-pool")
async def get_indicator_pool() -> list[dict[str, object]]:
    """Serve the GA genome's indicator pool (the launch-time source of truth).

    Names must match ``discovery.operators.INDICATOR_POOL`` exactly — a
    display-name mismatch here fails the run at launch (vibe-quant-rubp1).
    """
    from vibe_quant.discovery.operators import INDICATOR_POOL, _ensure_pool

    _ensure_pool()
    return [
        {
            "name": name,
            "params": {
                param: {"min": lo, "max": hi} for param, (lo, hi) in ranges.items()
            },
        }
        for name, ranges in INDICATOR_POOL.items()
    ]
