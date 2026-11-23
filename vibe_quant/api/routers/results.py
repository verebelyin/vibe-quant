"""Results & analytics router."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import signal
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from vibe_quant.api.deps import get_job_manager, get_state_manager
from vibe_quant.api.schemas.backtest import BacktestRunResponse
from vibe_quant.api.schemas.result import (
    BacktestResultResponse,
    BankrollReportLaunchResponse,
    BankrollReportRequest,
    BankrollReportResponse,
    ComparisonResponse,
    DrawdownPoint,
    EquityCurvePoint,
    FrozenDslResponse,
    MonthlyReturn,
    NotesUpdateRequest,
    RunListResponse,
    RunSummaryItem,
    RunSummaryResponse,
    SweepResultResponse,
    TradeResponse,
)
from vibe_quant.balance_settings import recorded_balance
from vibe_quant.bankroll.report import default_scratch_base, dsl_hasher
from vibe_quant.bankroll.snapshot import reclaim_abandoned
from vibe_quant.db.result_notes import notes_mapping, parse_notes
from vibe_quant.db.state_manager import (
    BANKROLL_JOB_TYPE,
    BankrollAdmissionError,
    StateManager,
)
from vibe_quant.jobs.manager import BacktestJobManager, process_start_time
from vibe_quant.utils import log_dir
from vibe_quant.validation.venue import DEFAULT_STARTING_BALANCE_USDT

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _enrich_result_with_notes(row: dict[str, Any]) -> dict[str, Any]:
    """Parse `notes` JSON and merge overfitting/cross-window fields into row.

    The discovery pipeline stores a structured JSON payload in
    backtest_results.notes including top_strategies[0] with nested
    `cross_window` and `wfa` sub-objects. Surface these as typed fields so
    the UI can render them without re-parsing JSON.

    Defensive: any parse error or missing key → field stays None.
    """
    enriched = dict(row)
    notes = enriched.get("notes")
    if not notes or not isinstance(notes, str):
        return enriched
    try:
        payload = parse_notes(notes)
    except (json.JSONDecodeError, TypeError):
        return enriched
    if not isinstance(payload, dict):
        return enriched

    top = payload.get("top_strategies")
    if not isinstance(top, list) or not top or not isinstance(top[0], dict):
        return enriched
    best = top[0]

    cw = best.get("cross_window")
    if isinstance(cw, dict):
        windows = cw.get("windows")
        offsets = payload.get("cross_window_months")
        min_sharpe = payload.get("cross_window_min_sharpe", 0.5)
        if isinstance(windows, list):
            mapped: list[dict[str, Any]] = []
            for i, w in enumerate(windows):
                if not isinstance(w, dict):
                    continue
                sharpe = w.get("sharpe")
                ret = w.get("return_pct")  # fraction despite the name
                # windows[0] is the original (in-sample) window; windows[i>=1]
                # are shifted by cross_window_months[i-1] (discovery pipeline).
                if i == 0:
                    offset: int | None = 0
                elif isinstance(offsets, list) and i - 1 < len(offsets):
                    offset = offsets[i - 1]
                else:
                    offset = None
                # Same rule as DiscoveryPipeline._evaluate_cross_windows.
                passed = (
                    isinstance(sharpe, (int, float))
                    and isinstance(ret, (int, float))
                    and isinstance(min_sharpe, (int, float))
                    and ret > 0
                    and sharpe >= min_sharpe
                )
                mapped.append(
                    {
                        "offset": offset,
                        "in_sample": i == 0,
                        "sharpe": sharpe,
                        "return_pct": ret,
                        "max_dd": w.get("max_dd"),
                        "trades": w.get("trades"),
                        "passed": passed,
                    }
                )
            enriched["cross_window_results"] = mapped

        if isinstance(cw.get("passed"), bool):
            enriched["cross_window_passed"] = cw["passed"]

    wfa = best.get("wfa")
    if isinstance(wfa, dict):
        sc = wfa.get("sharpe_consistency")
        if isinstance(sc, (int, float)):
            enriched["wfa_sharpe_consistency"] = float(sc)
        # The discovery WFA gate is the return-based consistency, not Sharpe's.
        rc = wfa.get("consistency")
        if isinstance(rc, (int, float)):
            enriched["wfa_consistency"] = float(rc)
        if isinstance(wfa.get("passed"), bool):
            enriched["wfa_passed"] = wfa["passed"]

    # Effective bootstrap floor the run was gated with (timeframe default or override).
    bms = payload.get("bootstrap_min_sharpe")
    if isinstance(bms, (int, float)):
        enriched["bootstrap_min_sharpe"] = float(bms)

    bootstrap = payload.get("bootstrap_ci") or best.get("bootstrap_ci")
    if isinstance(bootstrap, dict):
        lo = bootstrap.get("lower")
        hi = bootstrap.get("upper")
        lvl = bootstrap.get("level")
        if isinstance(lo, (int, float)):
            enriched["bootstrap_sharpe_lower"] = float(lo)
        if isinstance(hi, (int, float)):
            enriched["bootstrap_sharpe_upper"] = float(hi)
        if isinstance(lvl, (int, float)):
            enriched["bootstrap_ci_level"] = float(lvl)

    cross_regime = payload.get("cross_regime_results")
    if isinstance(cross_regime, list):
        enriched["cross_regime_results"] = cross_regime

    rsb = payload.get("random_short_baseline_pct")
    if isinstance(rsb, (int, float)):
        enriched["random_short_baseline_pct"] = float(rsb)

    return enriched

router = APIRouter(prefix="/api/results", tags=["results"])

StateMgr = Annotated[StateManager, Depends(get_state_manager)]
JobMgr = Annotated[BacktestJobManager, Depends(get_job_manager)]


_VALID_RUN_MODES = {"screening", "validation", "discovery"}
_VALID_STATUSES = {"pending", "running", "completed", "failed", "killed"}


@router.get("/runs/summary", response_model=RunSummaryResponse)
async def list_runs_summary(
    mgr: StateMgr,
    strategy_id: int | None = None,
    run_mode: str | None = None,
    status: str | None = None,
) -> RunSummaryResponse:
    if run_mode and run_mode not in _VALID_RUN_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid run_mode '{run_mode}'. Must be one of: {', '.join(_VALID_RUN_MODES)}",
        )
    if status and status not in _VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{status}'. Must be one of: {', '.join(_VALID_STATUSES)}",
        )
    rows = mgr.list_runs_with_results(strategy_id=strategy_id, run_mode=run_mode, status=status)
    runs = [RunSummaryItem(**r) for r in rows]
    return RunSummaryResponse(runs=runs)


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    mgr: StateMgr,
    status: str | None = None,
    strategy_id: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> RunListResponse:
    if status and status not in _VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{status}'. Must be one of: {', '.join(_VALID_STATUSES)}",
        )
    rows = mgr.list_backtest_runs(strategy_id=strategy_id, status=status)
    runs = [BacktestRunResponse(**r) for r in rows]
    if start_date:
        runs = [r for r in runs if r.created_at and r.created_at[:10] >= start_date[:10]]
    if end_date:
        runs = [r for r in runs if r.created_at and r.created_at[:10] <= end_date[:10]]
    return RunListResponse(runs=runs)


@router.get("/compare", response_model=ComparisonResponse)
async def compare_runs(
    mgr: StateMgr,
    run_ids: Annotated[str, Query(description="Comma-separated run IDs")] = "",
) -> ComparisonResponse:
    if not run_ids.strip():
        raise HTTPException(status_code=400, detail="run_ids query param required")
    parts = [x.strip() for x in run_ids.split(",") if x.strip()]
    bad = [x for x in parts if not x.isdigit()]
    if bad:
        raise HTTPException(status_code=422, detail=f"run_ids must be integers, got {bad}")
    id_list = [int(x) for x in parts]
    results: list[BacktestResultResponse] = []
    for rid in id_list:
        row = mgr.get_backtest_result(rid)
        if row is not None:
            results.append(BacktestResultResponse(**_enrich_result_with_notes(row)))
    return ComparisonResponse(runs=results)


@router.get("/runs/{run_id}/meta", response_model=BacktestRunResponse)
async def get_run_meta(run_id: int, mgr: StateMgr) -> BacktestRunResponse:
    """Get run metadata (strategy_id, symbols, timeframe, dates)."""
    row = mgr.get_backtest_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return BacktestRunResponse(**row)


_REVALIDATE = "re-validate first"


@router.get("/runs/{run_id}/frozen-dsl", response_model=FrozenDslResponse)
def get_frozen_dsl(run_id: int, mgr: StateMgr) -> FrozenDslResponse:
    """The DSL a completed validation run executed, for shadow enrolment.

    The run stores only ``notes.validation_input.dsl_sha256``; the strategy's stored DSL
    is returned while it still hashes to that value (validation never runs a
    ``parameters.dsl_override``). Anything else is 422 with its reason.
    """
    from vibe_quant.dsl.translator import translate_dsl_config

    _ensure_run_exists(mgr, run_id)
    run = mgr.get_backtest_run(run_id)
    assert run is not None  # checked above
    if run["run_mode"] != "validation":
        raise HTTPException(
            status_code=422, detail=f"Run {run_id} is a {run['run_mode']} run, not a validation run"
        )
    if run["status"] != "completed":
        raise HTTPException(
            status_code=422, detail=f"Run {run_id} is {run['status']}, not completed"
        )
    result = mgr.get_backtest_result(run_id)
    identity = notes_mapping(result["notes"] if result else None).get("validation_input")
    recorded = identity.get("dsl_sha256") if isinstance(identity, dict) else None
    if not isinstance(recorded, str):
        raise HTTPException(
            status_code=422, detail=f"Run {run_id} recorded no validation inputs: {_REVALIDATE}"
        )
    sid = run["strategy_id"]
    strategy = mgr.get_strategy(int(sid)) if sid is not None else None
    if strategy is None:
        # Nothing exists that could match the recorded hash, so "changed" would mislead.
        missing = (
            f"strategy {sid} no longer exists" if sid is not None else "strategy no longer exists"
        )
        raise HTTPException(status_code=422, detail=missing)
    if dsl_hasher(strategy["name"])(strategy["dsl_config"]) != recorded:
        raise HTTPException(
            status_code=422, detail=f"strategy changed since this run: {_REVALIDATE}"
        )
    frozen = translate_dsl_config(strategy["dsl_config"], strategy_name=strategy["name"])
    return FrozenDslResponse(
        run_id=run_id,
        strategy_id=int(strategy["id"]),
        strategy_name=str(strategy["name"]),
        dsl_json=json.dumps(frozen),
        dsl_sha256=recorded,
    )


@router.get("/runs/{run_id}", response_model=BacktestResultResponse)
async def get_run_summary(run_id: int, mgr: StateMgr) -> BacktestResultResponse:
    row = mgr.get_backtest_result(run_id)
    if row is not None:
        return BacktestResultResponse(**_enrich_result_with_notes(row))

    # Screening runs don't populate backtest_results — synthesize from best sweep
    best = _best_sweep_row(mgr, run_id)
    if best is None:
        # Ensure run exists at all
        run = mgr.get_backtest_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        raise HTTPException(status_code=404, detail="No results available for this run yet")
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
        # The balance the screening run actually used, recorded on every sweep row
        # (R12). A row written before the column existed is NULL and reads as the
        # default through the ONE shared rule (R13b) — never the requested
        # initial_balance, and never a hard-coded fallback. The UI derives net PnL
        # and fee ratios from it.
        starting_balance=recorded_balance(best.get("starting_balance")),
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
    rows = mgr.get_trades(run_id)
    if not rows:
        return []
    # Sort by exit_time, skip open trades
    closed = sorted(
        (r for r in rows if r.get("exit_time")),
        key=lambda r: r["exit_time"],
    )
    if not closed:
        return []
    equity = _get_starting_balance(mgr, run_id)
    points = [EquityCurvePoint(timestamp=closed[0]["entry_time"], equity=equity)]
    for t in closed:
        equity += float(t.get("net_pnl") or 0)
        points.append(EquityCurvePoint(timestamp=t["exit_time"], equity=round(equity, 2)))
    return points


@router.get("/runs/{run_id}/drawdown", response_model=list[DrawdownPoint])
async def get_drawdown(run_id: int, mgr: StateMgr) -> list[DrawdownPoint]:
    _ensure_run_exists(mgr, run_id)
    rows = mgr.get_trades(run_id)
    if not rows:
        return []
    closed = sorted(
        (r for r in rows if r.get("exit_time")),
        key=lambda r: r["exit_time"],
    )
    if not closed:
        return []
    equity = _get_starting_balance(mgr, run_id)
    peak = equity
    points: list[DrawdownPoint] = []
    for t in closed:
        equity += float(t.get("net_pnl") or 0)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        points.append(DrawdownPoint(timestamp=t["exit_time"], drawdown=round(dd, 6)))
    return points


@router.get("/runs/{run_id}/monthly-returns", response_model=list[MonthlyReturn])
async def get_monthly_returns(run_id: int, mgr: StateMgr) -> list[MonthlyReturn]:
    _ensure_run_exists(mgr, run_id)
    rows = mgr.get_trades(run_id)
    if not rows:
        return []
    closed = sorted(
        (r for r in rows if r.get("exit_time")),
        key=lambda r: r["exit_time"],
    )
    if not closed:
        return []
    monthly: dict[tuple[int, int], float] = {}
    for t in closed:
        ts = t["exit_time"][:7]  # "YYYY-MM"
        parts = ts.split("-")
        if len(parts) >= 2:
            key = (int(parts[0]), int(parts[1]))
            monthly[key] = monthly.get(key, 0.0) + float(t.get("net_pnl") or 0)
    # Convert to percentage of starting equity
    starting = _get_starting_balance(mgr, run_id)
    return [
        MonthlyReturn(year=y, month=m, return_pct=round(pnl / starting * 100, 2))
        for (y, m), pnl in sorted(monthly.items())
    ]


@router.put("/runs/{run_id}/notes", response_model=BacktestResultResponse)
async def update_notes(
    run_id: int,
    body: NotesUpdateRequest,
    mgr: StateMgr,
) -> BacktestResultResponse:
    """Save the user's free-text notes (``user_notes``).

    Never touches ``notes``, which holds machine JSON (discovery results,
    consistency flags) that promote/export depend on.
    """
    _ensure_run_exists(mgr, run_id)
    if not mgr.update_result_user_notes(run_id, body.notes):
        raise HTTPException(
            status_code=404, detail="No results for this run — notes require completed results"
        )
    row = mgr.get_backtest_result(run_id)
    if row is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Result disappeared after update")
    return BacktestResultResponse(**_enrich_result_with_notes(row))


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


# --- Bankroll sizing report (bead vibe-quant-2gyk1.3) ---

_ADMISSION_STATUS = {
    "run_missing": 404,
    "not_validation": 422,
    "not_completed": 422,
    "inputs_unverified": 422,
    "stale_parent": 422,
    "report_running": 409,
    "job_open": 409,
    "job_id_taken": 409,
}


def _bankroll_command(report_id: int, job_id: int, db_path: Path) -> list[str]:
    return [
        sys.executable, "-m", "vibe_quant.bankroll", "report",
        "--report-id", str(report_id), "--job-id", str(job_id), "--db", str(db_path),
    ]  # fmt: skip


def _strategy_name(mgr: StateManager, run_id: int) -> str:
    """Current name of the run's strategy ("" if none): the DSL hash needs it.

    A rename between this read and the admission transaction only makes the hash
    differ, which refuses (or marks stale), never admits a changed strategy.
    """
    try:
        run = mgr.get_backtest_run(run_id)
    except OverflowError:  # an id outside SQLite's INTEGER range matches no row
        return ""
    strategy = mgr.get_strategy(int(run["strategy_id"])) if run and run["strategy_id"] else None
    return str(strategy["name"]) if strategy else ""


def _job_alive(jobs: BacktestJobManager, pid: int, identity: str | None) -> bool:
    """The job's process is alive and, when recorded, still the same process."""
    if not jobs.is_process_alive(pid):
        return False
    return identity is None or process_start_time(pid) == identity


def _reconcile_bankroll(mgr: StateManager, jobs: BacktestJobManager) -> None:
    """Fail dead, stale or cancelled report jobs; reclaim scratch roots they left."""
    flipped = mgr.reconcile_bankroll_reports(
        job_alive=lambda pid, identity: _job_alive(jobs, pid, identity),
        terminate=lambda job_id: jobs.signal_job(job_id, signal.SIGTERM),
    )
    if flipped:
        reclaim_abandoned(default_scratch_base(), mgr.list_running_bankroll_report_ids())


def _estimated_seconds(
    mgr: StateManager, run_id: int, body: BankrollReportRequest
) -> int | None:
    """Upper estimate: the baseline plus every cell, in waves of ``workers`` engines.

    Scaled by the parent's own validation time; None when it recorded none.
    """
    result = mgr.get_backtest_result(run_id)
    seconds = result.get("execution_time_seconds") if result else None
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return None
    cells = len({*body.multiples, 1.0})
    return round(seconds * (1 + math.ceil(cells / body.workers)))


@router.post(
    "/runs/{run_id}/bankroll-report",
    response_model=BankrollReportLaunchResponse,
    status_code=202,
)
def launch_bankroll_report(
    run_id: int, body: BankrollReportRequest, mgr: StateMgr, jobs: JobMgr
) -> BankrollReportLaunchResponse:
    """Admit a bankroll sizing report on a completed validation run and start its job.

    One report runs at a time (409). A parent that recorded no validation inputs, or
    whose strategy changed since, is refused with "re-validate first" (422).
    """
    _reconcile_bankroll(mgr, jobs)
    try:
        report_id, job_id = mgr.admit_bankroll_report(
            run_id,
            body.model_dump(mode="json"),
            dsl_hash=dsl_hasher(_strategy_name(mgr, run_id)),
            job_alive=lambda pid, identity: _job_alive(jobs, pid, identity),
            terminate=lambda j: jobs.signal_job(j, signal.SIGTERM),
        )
    except BankrollAdmissionError as exc:
        raise HTTPException(
            status_code=_ADMISSION_STATUS[exc.code], detail=f"{exc.code}: {exc}"
        ) from exc
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_file = str(log_dir() / f"bankroll_{report_id}_{stamp}.log")
    try:
        pid = jobs.start_job(
            job_id, BANKROLL_JOB_TYPE, _bankroll_command(report_id, job_id, jobs.db_path),
            log_file=log_file,
        )  # fmt: skip
    except Exception as exc:
        # Free the slot: an unclaimed report ends failed, so the next POST is admitted.
        message = f"launch failed: {type(exc).__name__}: {exc}"
        mgr.end_bankroll_report(report_id, "failed", message, owner_pid=None)
        logger.warning("bankroll report %d: %s", report_id, message)
        status = 409 if isinstance(exc, ValueError) else 500
        raise HTTPException(status_code=status, detail=message) from exc
    logger.info(
        "bankroll report %d started job_id=%d pid=%d run_id=%d", report_id, job_id, pid, run_id
    )
    return BankrollReportLaunchResponse(
        report_id=report_id,
        job_id=job_id,
        estimated_seconds=_estimated_seconds(mgr, run_id, body),
    )


@router.get("/runs/{run_id}/bankroll-report", response_model=BankrollReportResponse | None)
def get_bankroll_report(
    run_id: int, mgr: StateMgr, jobs: JobMgr
) -> BankrollReportResponse | None:
    """The run's latest bankroll report (None if it has none), after a reconcile.

    Freshness is evaluated in the same read snapshot; a stale report's recommendation
    is masked.
    """
    _ensure_run_exists(mgr, run_id)
    _reconcile_bankroll(mgr, jobs)
    row = mgr.get_latest_bankroll_report(
        run_id, dsl_hash=dsl_hasher(_strategy_name(mgr, run_id))
    )
    if row is None:
        return None
    return BankrollReportResponse(
        id=row["id"],
        run_id=row["run_id"],
        job_id=row["job_id"],
        status=row["status"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        request=row["request"],
        report=row["report"],
        freshness=row["freshness"],
        stale_reasons=row["stale_reasons"],
    )


def _ensure_run_exists(mgr: StateManager, run_id: int) -> None:
    try:
        run = mgr.get_backtest_run(run_id)
    except OverflowError:  # an id outside SQLite's INTEGER range matches no row
        run = None
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")


def _best_sweep_row(mgr: StateManager, run_id: int) -> dict[str, Any] | None:
    """The row of a run's best sweep: Pareto-optimal first, else every sweep.

    Ordered by sharpe DESC in the database. The screening summary and the
    chart endpoints' balance fallback must agree on which sweep row is "best"
    (R13c), so the lookup lives here once instead of at each call site.
    """
    sweeps = mgr.get_sweep_results(run_id, pareto_only=True) or mgr.get_sweep_results(run_id)
    return sweeps[0] if sweeps else None


def _get_starting_balance(mgr: StateManager, run_id: int) -> float:
    """The balance the equity/drawdown/monthly endpoints scale by (R13c).

    ONE rule, the same one the summary reports: a run's own
    ``backtest_results.starting_balance`` when it has a result row; else the
    SAME best sweep row the summary synthesizes from; else the venue default
    (1_000 — must match VenueConfig, not an arbitrary 10k: a mismatched
    fallback mis-scales every equity/drawdown/monthly %). Every recorded value
    is read through ``recorded_balance``, where a NULL column (a row written
    before the column existed) means the default — there is no second NULL
    rule and the requested ``initial_balance`` is never used.
    """
    result = mgr.get_backtest_result(run_id)
    if result is not None:
        return float(recorded_balance(result.get("starting_balance")))
    best = _best_sweep_row(mgr, run_id)
    if best is not None:
        return float(recorded_balance(best.get("starting_balance")))
    return float(DEFAULT_STARTING_BALANCE_USDT)
