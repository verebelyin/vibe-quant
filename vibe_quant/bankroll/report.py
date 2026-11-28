"""The bankroll report coordinator: one background job per report (design 2gyk1.3 §2.4).

``python -m vibe_quant.bankroll report --report-id R --job-id J --db PATH`` runs
:func:`run_report`:

1. **Claim.** ``claim_bankroll_report`` (one IMMEDIATE transaction) records this process as
   the owner; a refused claim does no engine work and fails the job row only when that
   row is open and unclaimed (a finished job or another attempt's job is left alone).
2. **Snapshot.** Under ``<scratch>/vq-bankroll-<db key>/vq-bankroll/<report_id>/`` (the
   scratch is namespaced by the state DB: report ids are per DB) it captures the market once
   (``seed/`` catalog clone + a small ``archive.db`` holding only the rows the runs read)
   and clones the seed into a NON-existing private catalog per engine.
3. **Baseline.** ``run_variant(inp, {}, mtm_probe=True)`` on the baseline clone: the
   recorded input identity first (``stale_parent`` / ``inputs_mismatch``), then the stored
   headline, exact float equality (``stale_parent`` / ``headline_mismatch``), then NT's
   force-stop flag (``engine_force_stopped``: report failed).
4. **Stress cells** in a process pool. Workers get only the snapshotted
   :class:`~vibe_quant.validation.variant.VariantInput`, the overrides, their own clone and
   the archive snapshot: they open no state DB. Before each cell is handed out the
   coordinator re-checks (one read) that the report is still running and owned by it. A
   worker process that dies (abort, OOM kill) breaks the pool and every cell in flight:
   the pool is rebuilt and each broken cell re-runs alone, so only the crashing cell's
   row fails (``BrokenProcessPool: ...``) and serial/parallel rows stay equal.
   With ``margin_sweep`` on and a recommended size, a second stage runs the margin cells
   (mode x leverage x margin limit) at that size through the same pool and fence, gives
   each cell its liquidating move and names the safest setting; no recommended size skips
   it, and an error outside the per-cell catch fails only the sweep (the size report stays).
5. **Publish** in one transaction (``publish_bankroll_report``: ownership, open job and
   parent freshness checked where the report is written); a publish that raises ends the
   report ``failed``.
6. ``finally``: the report's scratch root is removed (a SIGTERM raises ``SystemExit`` so
   this runs); roots of reports that are no longer running are reclaimed at start.

Every number in a row comes from that row's own run: nothing is scaled by ``k``. The
report is a replay on its captured market data whose parent headlines matched; the
original market series is not verified.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import logging
import math
import multiprocessing
import os
import shutil
import signal
import tempfile
import time
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from vibe_quant.bankroll.kelly import DEFAULT_TARGET_VOL, KellyStats, kelly_stats
from vibe_quant.bankroll.recommend import (
    Recommendation,
    kelly_actionable,
    recommend,
    safest_setting,
)
from vibe_quant.bankroll.snapshot import (
    MarketSnapshot,
    capture_market,
    clone_for_engine,
    db_scratch,
    reclaim_abandoned,
    report_root,
)
from vibe_quant.bankroll.stress import (
    CellPlan,
    MarginCellPlan,
    RowOutcome,
    plan_cells,
    plan_margin_cells,
    row_risk_facts,
)
from vibe_quant.bankroll_grid import build_cells
from vibe_quant.liquidation_settings import LiquidationMode, LiquidationSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence

    from vibe_quant.bankroll.liq_move import Book, Move
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.dsl.schema import StrategyDSL
    from vibe_quant.jobs.manager import BacktestJobManager
    from vibe_quant.validation.results import ValidationResult
    from vibe_quant.validation.variant import VariantInput

logger = logging.getLogger(__name__)

DEFAULT_MULTIPLES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 10.0)
DEFAULT_USER_CAP = 1.0
DEFAULT_DD_LIMIT = 0.30
REVALIDATE = "re-validate first"
REPLAY_LABEL = (
    "Replay on this report's captured market data; parent headlines matched; "
    "the original market series was not verified."
)
SLIPPAGE_TIMING: dict[str, str] = {
    "ordinary": "Modeled slippage is booked at the ledger's combined exit/valuation timestamp.",
    "overlay": (
        "Liquidation overlay: slippage is booked at the overlay's leg times and post-hit "
        "metrics are overlay results."
    ),
}
HEADLINE_FIELDS: tuple[str, ...] = (
    "total_return",
    "max_drawdown",
    "sharpe_ratio",
    "total_trades",
    "total_fees",
    "total_funding",
)
"""Stored parent headline the baseline replay must equal exactly (C21)."""
DEFAULT_SWEEP_MODES: tuple[LiquidationMode, ...] = ("isolated", "cross")
DEFAULT_SWEEP_LEVERAGES: tuple[float, ...] = (3, 5, 10)
DEFAULT_SWEEP_MARGIN_LIMITS: tuple[float, ...] = (0.5, 1.0)
SCENARIO_LABEL = (
    "Modeled scenario: the largest book the margin limit admits at the starting balance "
    "(positions at the exposure cap plus one smaller position for the margin left), entered "
    "at one moment and all moving against you at once; funding and fees ignored. Cross rows "
    "of a multi-coin strategy hold only for that simultaneous entry and are never named "
    "safest. Each row's liquidation count is from its own run."
)

RowRole = Literal["baseline", "control", "cell"]


def default_workers() -> int:
    return min(4, max(1, (os.cpu_count() or 1) // 2))


def default_scratch_base() -> Path:
    """The tmp dir; :func:`run_report` namespaces it by state DB (:func:`db_scratch`)."""
    return Path(tempfile.gettempdir())


# --- inputs ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportRequest:
    """The admitted request, with the design's defaults; ``1.0`` is always a multiple."""

    user_cap: float = DEFAULT_USER_CAP
    dd_limit: float = DEFAULT_DD_LIMIT
    target_vol: float = DEFAULT_TARGET_VOL
    multiples: tuple[float, ...] = DEFAULT_MULTIPLES
    liquidation_mode: LiquidationMode | None = None
    workers: int = field(default_factory=default_workers)
    margin_sweep: bool = False
    sweep_modes: tuple[LiquidationMode, ...] = DEFAULT_SWEEP_MODES
    sweep_leverages: tuple[float, ...] = DEFAULT_SWEEP_LEVERAGES
    sweep_margin_limits: tuple[float, ...] = DEFAULT_SWEEP_MARGIN_LIMITS

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> ReportRequest:
        multiples = raw.get("multiples")
        ks = DEFAULT_MULTIPLES if multiples is None else tuple(float(k) for k in multiples)
        workers = raw.get("workers")
        return cls(
            user_cap=float(raw.get("user_cap", DEFAULT_USER_CAP)),
            dd_limit=float(raw.get("dd_limit", DEFAULT_DD_LIMIT)),
            target_vol=float(raw.get("target_vol", DEFAULT_TARGET_VOL)),
            multiples=tuple(sorted({*ks, 1.0})),
            liquidation_mode=raw.get("liquidation_mode"),
            workers=default_workers() if workers is None else int(workers),
            # Absent (a request stored before the sweep existed) replays as v1: no sweep.
            margin_sweep=raw.get("margin_sweep") is True,
            # Absent or null reads as the default grid.
            sweep_modes=_listed(raw, "sweep_modes", DEFAULT_SWEEP_MODES),
            sweep_leverages=_listed(raw, "sweep_leverages", DEFAULT_SWEEP_LEVERAGES),
            sweep_margin_limits=_listed(raw, "sweep_margin_limits", DEFAULT_SWEEP_MARGIN_LIMITS),
        )


def _listed(raw: Mapping[str, Any], key: str, default: tuple[Any, ...]) -> tuple[Any, ...]:
    value = raw.get(key)
    return default if value is None else tuple(value)


def run_config_of(inputs: Mapping[str, Any]) -> dict[str, object]:
    """The parent's ``backtest_runs`` row as ``StateManager.get_backtest_run`` returns it."""
    row: dict[str, object] = dict(inputs["run_row"])
    for key in ("symbols", "parameters"):
        value = row.get(key)
        if isinstance(value, str):
            row[key] = json.loads(value)
    row["id"] = int(inputs["run_id"])
    return row


def variant_input_of(inputs: Mapping[str, Any]) -> VariantInput:
    """The admission snapshot as the variant's input (no DB read)."""
    from vibe_quant.validation.variant import VariantInput

    return VariantInput(
        strategy_name=str(inputs.get("strategy_name", "strategy")),
        dsl_json=str(inputs["dsl_json"]),
        run_config_json=json.dumps(run_config_of(inputs)),
        parent_identity=dict(inputs["parent_identity"]),
    )


def dsl_hasher(strategy_name: str) -> Callable[[object], str | None]:
    """``dsl_config -> dsl_sha256`` exactly as ``validation_input_identity`` records it.

    The DSL is translated (frontend format needs the strategy name) and validated first;
    an unreadable DSL hashes to ``None``.
    """

    def dsl_hash(dsl_config: object) -> str | None:
        from vibe_quant.dsl.parser import validate_strategy_dict
        from vibe_quant.dsl.translator import translate_dsl_config
        from vibe_quant.validation.runner import _canonical_json

        if not isinstance(dsl_config, dict):
            return None
        try:
            dsl = validate_strategy_dict(
                translate_dsl_config(dsl_config, strategy_name=strategy_name)
            )
        except Exception as exc:  # noqa: BLE001 -- any invalid DSL is "unreadable"
            logger.warning("bankroll: DSL of %r is unreadable (%s)", strategy_name, type(exc))
            return None
        return hashlib.sha256(_canonical_json(dsl.model_dump(mode="json")).encode()).hexdigest()

    return dsl_hash


def _validated_dsl(inputs: Mapping[str, Any]) -> StrategyDSL:
    """The validated DSL the engine runs (translated like :func:`dsl_hasher`)."""
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.dsl.translator import translate_dsl_config

    return validate_strategy_dict(
        translate_dsl_config(
            json.loads(str(inputs["dsl_json"])),
            strategy_name=str(inputs.get("strategy_name", "strategy")),
        )
    )


def _taker_symbols(inputs: Mapping[str, Any], symbols: Sequence[str]) -> tuple[str, ...]:
    """The run symbols whose 1m klines a taker-flow indicator reads (else none).

    The types come from the validated DSL: the schema upper-cases a stored
    ``taker_imbalance``.
    """
    from vibe_quant.dsl.aux_data import TAKER_TYPES

    types = {ind.type for ind in _validated_dsl(inputs).indicators.values()}
    return tuple(symbols) if TAKER_TYPES & types else ()


def _entry_sides(inputs: Mapping[str, Any]) -> tuple[int, ...]:
    """The sides the strategy can enter: +1 long, -1 short (non-empty entry conditions)."""
    entry = _validated_dsl(inputs).entry_conditions
    return tuple(side for side, conds in ((1, entry.long), (-1, entry.short)) if conds)


def _starting_balance(identity: Mapping[str, Any]) -> float | None:
    """The parent's recorded venue starting balance; ``None`` when it recorded none, or
    one that is not finite or <= 0 (R-bk-2: shown like a missing one)."""
    venue = identity.get("venue")
    if not isinstance(venue, str):
        return None
    raw = json.loads(venue).get("starting_balance_usdt")
    if raw is None:
        return None
    balance = float(raw)
    return balance if math.isfinite(balance) and balance > 0 else None


# --- one engine run (pool worker) -----------------------------------------------------


@dataclass(frozen=True)
class CellRun:
    """What one variant run produced (picklable; the worker returns this, not the result)."""

    status: Literal["ok", "failed"]
    error: str | None = None
    mismatch_keys: tuple[str, ...] | None = None
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    total_trades: int = 0
    total_fees: float = 0.0
    total_funding: float = 0.0
    total_slippage: float = 0.0
    engine_force_stopped: bool = False
    mtm_paths: str | None = None
    mtm_dd: float | None = None
    mtm_ledger: str | None = None
    bust_ts: int | None = None
    path_divergence: bool = False
    daily: tuple[float, ...] | None = None
    liquidation: dict[str, Any] | None = None

    @classmethod
    def of(cls, res: ValidationResult) -> CellRun:
        probe = res.mtm_probe
        liq = res.notes.get("liquidation")
        return cls(
            status="ok",
            total_return=res.total_return,
            max_drawdown=res.max_drawdown,
            sharpe_ratio=res.sharpe_ratio,
            sortino_ratio=res.sortino_ratio,
            total_trades=res.total_trades,
            total_fees=res.total_fees,
            total_funding=res.total_funding,
            total_slippage=res.total_slippage,
            engine_force_stopped=res.engine_force_stopped,
            mtm_paths=None if probe is None else probe.paths,
            mtm_dd=None if probe is None else probe.dd,
            mtm_ledger=None if probe is None else probe.ledger,
            bust_ts=None if probe is None else probe.bust_ts,
            path_divergence=False if probe is None else probe.path_divergence,
            daily=None if probe is None or probe.daily is None else tuple(probe.daily.values),
            liquidation=dict(liq) if isinstance(liq, dict) else None,
        )


def run_cell(
    inp: VariantInput, overrides: Mapping[str, object], catalog: Path, archive: Path
) -> CellRun:
    """One engine run on its own catalog clone and the archive snapshot (no state DB).

    A raised error becomes a failed :class:`CellRun` with ``"Type: message"``.
    """
    from vibe_quant.validation.variant import VariantInputsMismatch, run_variant

    try:
        res = run_variant(
            inp, dict(overrides), catalog_path=catalog, archive_path=archive, mtm_probe=True
        )
    except VariantInputsMismatch as exc:
        return CellRun(status="failed", error=f"{type(exc).__name__}: {exc}",
                       mismatch_keys=exc.keys)  # fmt: skip
    except Exception as exc:  # noqa: BLE001 -- a failed cell is a reported row, not a crash
        logger.warning("bankroll cell %s failed: %s", dict(overrides), type(exc).__name__)
        return CellRun(status="failed", error=f"{type(exc).__name__}: {exc}")
    return CellRun.of(res)


def _executor(workers: int) -> Executor:
    """The cell pool: one fresh ``spawn`` process per engine run (cold caches, no inherited
    DB handles; NT's force-stop flag and the aux-data configuration are process-global)."""
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1
    )


def _stop_executor(executor: Executor) -> None:
    """Cancel queued cells and stop running workers (they would write into a removed root)."""
    # Python 3.13 has no public way to stop running workers (terminate_workers() is 3.14),
    # so this reads ProcessPoolExecutor's private worker map; test_fence_stops_running_workers
    # pins the effect on a real spawn pool.
    procs = getattr(executor, "_processes", None)
    if isinstance(procs, dict):
        for proc in list(procs.values()):
            with contextlib.suppress(Exception):
                proc.terminate()
    executor.shutdown(wait=False, cancel_futures=True)


class _Pool:
    """The cell executor; a pool broken by a dead worker is replaced by a fresh one."""

    def __init__(self, workers: int) -> None:
        self.workers = max(1, workers)
        self.executor = _executor(self.workers)

    def submit(self, *args: Any) -> tuple[Future[CellRun], Executor]:
        """``run_cell(*args)`` on the current executor (rebuilt if it is already broken)."""
        try:
            return self.executor.submit(run_cell, *args), self.executor
        except BrokenProcessPool:
            self.rebuild(self.executor)
            return self.executor.submit(run_cell, *args), self.executor

    def rebuild(self, broken: Executor) -> None:
        if self.executor is broken:
            _stop_executor(broken)
            self.executor = _executor(self.workers)


# --- report assembly ------------------------------------------------------------------


class ReportStop(Exception):  # noqa: N818 -- a stop outcome, not an error type
    """The report ends before publishing: ``status`` is ``failed`` or ``stale_parent``."""

    def __init__(self, status: Literal["failed", "stale_parent"], message: str) -> None:
        super().__init__(message)
        self.status = status


class Fenced(Exception):  # noqa: N818 -- the per-cell fence refused
    """This process no longer owns the running report (cancelled, reconciled, deleted)."""


def _num(value: object) -> float | int | None:
    """JSON-safe number: non-finite floats become ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _same(stored: object, replay: object) -> bool:
    """Exact equality; NULL/NaN on both sides counts as equal (SQLite stores NaN as NULL)."""
    missing_a = stored is None or (isinstance(stored, float) and math.isnan(stored))
    missing_b = replay is None or (isinstance(replay, float) and math.isnan(replay))
    if missing_a or missing_b:
        return missing_a and missing_b
    return stored == replay


def headline_mismatch(stored: Mapping[str, Any], replay: CellRun) -> list[str]:
    """Each headline field whose replay differs from the stored value, with both values."""
    return [
        f"{name} stored={stored.get(name)!r} replay={getattr(replay, name)!r}"
        for name in HEADLINE_FIELDS
        if not _same(stored.get(name), getattr(replay, name))
    ]


def _outcome(k: float, run: CellRun | None, error: str | None) -> RowOutcome:
    if run is None:
        return RowOutcome(k=k, status="not_runnable", error=error)
    if run.status != "ok":
        return RowOutcome(k=k, status="failed", error=run.error)
    if run.engine_force_stopped:
        return RowOutcome(k=k, status="failed", engine_force_stopped=True,
                          trades=run.total_trades, error="engine_force_stopped")  # fmt: skip
    return RowOutcome(
        k=k,
        status="ok",
        trades=run.total_trades,
        total_return=run.total_return,
        mtm_paths=run.mtm_paths,
        mtm_dd=run.mtm_dd,
        bust_ts=run.bust_ts,
        path_divergence=run.path_divergence,
        liquidation=run.liquidation,
    )


def _row_json(
    plan: CellPlan,
    role: RowRole,
    run: CellRun | None,
    outcome: RowOutcome,
    reason: str | None,
    start_ns: int,
) -> dict[str, object]:
    facts = row_risk_facts(outcome, start_ns)
    out: dict[str, object] = {
        "k": plan.k,
        "role": role,
        "status": outcome.status,
        "error": outcome.error,
        "requested_risk": plan.requested_risk,
        "effective_risk": plan.effective_risk,
        "effective_cap": plan.effective_cap,
        "leverage": plan.leverage,
        "mode": plan.mode,
        "risk_held_at_limit": plan.risk_held_at_limit,
        "eligible": reason is None,
        "ineligible_reason": reason,
        "liquidation_count": facts.liquidation_count,
        "time_to_first_liquidation_s": facts.time_to_first_liquidation_s,
        "time_to_ruin_s": facts.time_to_ruin_s,
    }
    if run is not None and run.status == "ok":
        out.update(
            trades=run.total_trades,
            total_return=_num(run.total_return),
            max_drawdown=_num(run.max_drawdown),
            mtm_dd=_num(run.mtm_dd),
            mtm_paths=run.mtm_paths,
            sharpe_ratio=_num(run.sharpe_ratio),
            sortino_ratio=_num(run.sortino_ratio),
            total_fees=_num(run.total_fees),
            total_funding=_num(run.total_funding),
            total_slippage=_num(run.total_slippage),
            bust_ts=run.bust_ts,
            path_divergence=run.path_divergence,
            engine_force_stopped=run.engine_force_stopped,
        )
    return out


def _start_ns(identity: Mapping[str, Any]) -> int:
    from vibe_quant.validation.extraction import date_to_ns

    start_ns = date_to_ns(str(identity["start_date"]))
    if start_ns is None:
        raise ValueError(f"unreadable start_date {identity['start_date']!r}")
    return start_ns


@dataclass(frozen=True)
class _SizeVerdict:
    baseline: RowOutcome
    stats: KellyStats | None
    outcomes: list[RowOutcome]
    rec: Recommendation


def size_recommendation(
    request: ReportRequest, plans: Sequence[CellPlan], baseline: CellRun,
    runs: Mapping[float, CellRun],
) -> _SizeVerdict:  # fmt: skip
    """Kelly from the baseline's daily series, every size row's outcome, the recommendation."""
    base_outcome = _outcome(1.0, baseline, None)
    stats: KellyStats | None = None
    if baseline.daily is not None:
        stats = kelly_stats(baseline.daily, target_vol=request.target_vol)
    outcomes = [_outcome(plan.k, runs.get(plan.k), plan.error) for plan in plans]
    rec = recommend(
        outcomes, stats, baseline=base_outcome, user_cap=request.user_cap, dd_limit=request.dd_limit
    )
    return _SizeVerdict(base_outcome, stats, outcomes, rec)


def build_report(
    *,
    inputs: Mapping[str, Any],
    request: ReportRequest,
    mode: LiquidationMode,
    plans: Sequence[CellPlan],
    baseline: CellRun,
    runs: Mapping[float, CellRun],
    k1_is_baseline: bool,
    margin_sweep: dict[str, object] | None = None,
) -> dict[str, object]:
    """The published report: Kelly card, size-stress rows, recommendation, margin sweep
    (``None`` when not requested), disclosures."""
    start_ns = _start_ns(inputs["parent_identity"])
    verdict = size_recommendation(request, plans, baseline, runs)
    base_outcome, stats, outcomes, rec = (
        verdict.baseline, verdict.stats, verdict.outcomes, verdict.rec,
    )  # fmt: skip
    reasons = dict(rec.row_reasons)
    rows: list[dict[str, object]] = []
    for plan, outcome in zip(plans, outcomes, strict=True):
        role: RowRole = "cell"
        if plan.k == 1.0:
            role = "baseline" if k1_is_baseline else "control"
        rows.append(_row_json(plan, role, runs.get(plan.k), outcome, reasons[plan.k], start_ns))
    params = run_config_of(inputs)["parameters"]
    assert isinstance(params, dict)
    parent_mode = LiquidationSettings.from_params(params).mode
    baseline_row = _row_json(
        plan_cells(params, [1.0], parent_mode)[0], "baseline", baseline, base_outcome, None,
        start_ns,
    )  # fmt: skip
    del baseline_row["eligible"], baseline_row["ineligible_reason"]
    return {
        "v": 1,
        "label": REPLAY_LABEL,
        "parent": {
            "run_id": inputs["run_id"],
            "result_id": inputs["parent_result_id"],
            "dsl_sha256": inputs["dsl_sha256"],
            "mode": parent_mode,
        },
        "request": asdict(request),
        "mode": mode,
        "baseline": baseline_row,
        "kelly": {
            **(stats.to_dict() if stats is not None else {"status": None, "reason": None}),
            "actionable_reason": kelly_actionable(base_outcome, stats),
        },
        "rows": rows,
        "recommendation": {
            "k": rec.k,
            "ceiling": rec.ceiling,
            "reason": rec.reason,
            "binding": rec.binding,
            "text": rec.text,
        },
        "margin_sweep": margin_sweep,
        "disclosures": {
            "slippage_timing": SLIPPAGE_TIMING.get(baseline.mtm_ledger or ""),
            "k1_control": None
            if k1_is_baseline
            else f"k=1 is a control run in {mode} mode; it is not compared to the parent.",
        },
    }


# --- the margin sweep (stage 2) -------------------------------------------------------

MarginKey = tuple[str, LiquidationMode, float, float]


def _sweep_stub(status: Literal["skipped", "failed"], reason: str,
                rec: Recommendation) -> dict[str, object]:  # fmt: skip
    return {
        "status": status, "reason": reason, "recommendation_reason": rec.reason, "k": rec.k,
        "scenario": None, "cells": [], "safest": None, "safest_reason": None,
    }  # fmt: skip


def _move_json(move: Move) -> dict[str, object]:
    return {
        "status": move.status,
        "value": _num(move.value),
        "side": move.side,
        "legs": [[symbol, _num(notional)] for symbol, notional in move.legs],
        "reason": move.reason,
        "path_dependent": move.path_dependent,
    }


def _margin_sweep(
    *,
    inputs: Mapping[str, Any],
    request: ReportRequest,
    params: Mapping[str, object],
    symbols: Sequence[str],
    size_plan: CellPlan,
    size_run: CellRun,
    run_jobs: Callable[
        [Sequence[tuple[MarginKey, Mapping[str, object]]]], dict[MarginKey, CellRun]
    ],
) -> dict[str, object]:
    """Run the margin cells at ``size_plan``; each cell's row, liquidating move, the safest.

    A reused cell (its merged parameters equal the k-row's) takes the k-row's run.
    """
    # Lazy: the API imports this module and must not pull vibe_quant.liquidation
    # (test_symbol_leverage::test_importing_the_rule_never_imports_the_liquidation_package).
    from vibe_quant.bankroll import liq_move
    from vibe_quant.liquidation.tiers import load_tier_tables

    identity = inputs["parent_identity"]
    cells = build_cells(request.sweep_modes, request.sweep_leverages, request.sweep_margin_limits)
    margin = plan_margin_cells(params, size_plan, cells, symbols)
    keys: list[MarginKey] = [("margin", c.mode, c.leverage, c.max_margin_pct) for c in margin]
    runs = run_jobs([(key, c.overrides) for key, c in zip(keys, margin, strict=True)
                     if c.status == "planned" and not c.reused])  # fmt: skip
    balance = _starting_balance(identity)
    sides = _entry_sides(inputs)
    tables = load_tier_tables()
    start_ns = _start_ns(identity)
    scored: list[tuple[MarginCellPlan, RowOutcome, Move]] = []
    rows: list[tuple[CellRun | None, Book | None]] = []
    for key, cell in zip(keys, margin, strict=True):
        assert cell.mode != "off"  # build_cells admits isolated and cross only
        run = size_run if cell.reused else runs.get(key)
        book, move = liq_move.cell_move(
            mode=cell.mode, leverage=cell.leverage, max_margin_pct=cell.max_margin_pct,
            cap=size_plan.effective_cap, starting_balance=balance, symbols=symbols,
            sides=sides, tables=tables,
        )  # fmt: skip
        scored.append((cell, _outcome(size_plan.k, run, cell.error), move))
        rows.append((run, book))
    best, safest_reason, reasons = safest_setting(scored, dd_limit=request.dd_limit)
    out_cells: list[dict[str, object]] = []
    for (cell, outcome, move), (run, book), reason in zip(scored, rows, reasons, strict=True):
        row = _row_json(replace(cell.size, leverage=cell.leverage, mode=cell.mode), "cell",
                        run, outcome, reason, start_ns)  # fmt: skip
        out_cells.append({
            **row,
            "max_margin_pct": cell.max_margin_pct,
            "reused": cell.reused,
            "positions": None if book is None else book.positions,
            "remainder": None if book is None else _num(book.remainder),
            "move": _move_json(move),
        })  # fmt: skip
    safest: dict[str, object] | None = None
    if best is not None:
        cell, _outcome_best, move = scored[best]
        safest = {"mode": cell.mode, "leverage": cell.leverage,
                  "max_margin_pct": cell.max_margin_pct, "move": _move_json(move)}  # fmt: skip
    return {
        "status": "completed",
        "reason": None,
        "recommendation_reason": None,
        "k": size_plan.k,
        "scenario": {
            "starting_balance": _num(balance),
            "notional": None if balance is None else _num(size_plan.effective_cap * balance),
            "sides": [name for side, name in ((1, "long"), (-1, "short")) if side in sides],
            "symbols": list(symbols),
            "label": SCENARIO_LABEL,
        },
        "cells": out_cells,
        "safest": safest,
        "safest_reason": safest_reason,
    }


# --- the coordinator ------------------------------------------------------------------


@dataclass
class _Sources:
    catalog: Path
    archive: Path
    scratch: Path


def _baseline_checks(stored: Mapping[str, Any] | None, inputs: Mapping[str, Any],
                     baseline: CellRun) -> None:  # fmt: skip
    """C21 + C10a: identity, then headline, then force stop; raises :class:`ReportStop`."""
    if baseline.mismatch_keys is not None:
        keys = ", ".join(baseline.mismatch_keys)
        raise ReportStop("stale_parent", f"stale_parent: inputs_mismatch: {keys}: {REVALIDATE}")
    if baseline.status != "ok":
        raise ReportStop("failed", f"baseline failed: {baseline.error}")
    if stored is None:
        raise ReportStop("stale_parent", f"stale_parent: result_missing: {REVALIDATE}")
    if stored.get("id") != inputs["parent_result_id"]:
        raise ReportStop("stale_parent", f"stale_parent: result_replaced: {REVALIDATE}")
    diffs = headline_mismatch(stored, baseline)
    if diffs:
        raise ReportStop(
            "stale_parent", f"stale_parent: headline_mismatch: {'; '.join(diffs)}: {REVALIDATE}"
        )
    if baseline.engine_force_stopped:
        raise ReportStop("failed", "engine_force_stopped")


def _coordinate(
    state: StateManager, report_id: int, pid: int, root: Path, src: _Sources
) -> dict[str, object]:
    row = state.get_bankroll_report(report_id)
    if row is None:
        raise Fenced(f"report {report_id} vanished")
    inputs: Mapping[str, Any] = row["inputs"]
    request = ReportRequest.from_json(row["request"] or {})
    inp = variant_input_of(inputs)
    run_config = run_config_of(inputs)
    params = run_config["parameters"]
    assert isinstance(params, dict)
    parent_mode = LiquidationSettings.from_params(params).mode
    mode: LiquidationMode = request.liquidation_mode or (
        parent_mode if parent_mode != "off" else "cross"
    )
    plans = plan_cells(params, request.multiples, mode)
    k1_is_baseline = parent_mode == mode
    symbols = [str(s) for s in run_config["symbols"]]  # type: ignore[attr-defined]

    t0 = time.monotonic()
    snapshot = capture_market(
        src.catalog, src.archive, root, symbols=symbols,
        kline_symbols=_taker_symbols(inputs, symbols),
    )  # fmt: skip
    logger.info(
        "bankroll report %d: market snapshot in %.3f s, archive %d bytes",
        report_id, time.monotonic() - t0, snapshot.archive.stat().st_size,
    )  # fmt: skip

    pool = _Pool(request.workers)
    clones = itertools.count()
    try:
        baseline = pool.executor.submit(
            run_cell, inp, {}, clone_for_engine(snapshot, root / "baseline" / "catalog"),
            snapshot.archive,
        ).result()  # fmt: skip
        _baseline_checks(state.get_backtest_result(int(inputs["run_id"])), inputs, baseline)
        runs = _run_cells(state, report_id, pid, pool, inp, snapshot, clones,
                          [(p.k, p.overrides) for p in plans if p.status == "planned"
                           and not (k1_is_baseline and p.k == 1.0)])  # fmt: skip
        if k1_is_baseline:
            runs[1.0] = baseline
        sweep: dict[str, object] | None = None
        if request.margin_sweep:
            rec = size_recommendation(request, plans, baseline, runs).rec
            sweep = _sweep_stub("skipped", "no_recommended_size", rec)
            if rec.k is not None:
                size_plan = next(p for p in plans if p.k == rec.k)
                try:
                    sweep = _margin_sweep(
                        inputs=inputs, request=request, params=params, symbols=symbols,
                        size_plan=size_plan, size_run=runs[rec.k],
                        run_jobs=lambda jobs: _run_cells(state, report_id, pid, pool, inp,
                                                         snapshot, clones, jobs),
                    )  # fmt: skip
                except (Fenced, ReportStop):
                    raise
                except Exception as exc:  # noqa: BLE001 -- a sweep bug keeps the size report
                    logger.warning("bankroll report %d: margin sweep failed (%s)",
                                   report_id, type(exc).__name__)  # fmt: skip
                    sweep = _sweep_stub("failed", f"{type(exc).__name__}: {exc}", rec)
    except BaseException:
        _stop_executor(pool.executor)
        raise
    pool.executor.shutdown(wait=True)
    return build_report(inputs=inputs, request=request, mode=mode, plans=plans,
                        baseline=baseline, runs=runs, k1_is_baseline=k1_is_baseline,
                        margin_sweep=sweep)  # fmt: skip


def _run_cells[K: Hashable](
    state: StateManager,
    report_id: int,
    pid: int,
    pool: _Pool,
    inp: VariantInput,
    snapshot: MarketSnapshot,
    clones: itertools.count[int],
    jobs: Sequence[tuple[K, Mapping[str, object]]],
) -> dict[K, CellRun]:
    """Run ``jobs`` (key, overrides) at most ``pool.workers`` at a time; fence-check before
    handing out each. Size cells are keyed by ``k``, margin cells by ``("margin", mode,
    leverage, margin limit)``; ``clones`` numbers the engines' private catalogs across
    both stages.

    A dead worker breaks every cell in flight with it: each broken cell re-runs alone on a
    fresh pool, and one that breaks alone is that cell's failed row (C23).
    """
    runs: dict[K, CellRun] = {}
    for job, _exc in _run_batch(state, report_id, pid, pool, inp, snapshot, runs, clones,
                                jobs, pool.workers):  # fmt: skip
        for (key, _), exc in _run_batch(state, report_id, pid, pool, inp, snapshot, runs,
                                        clones, [job], 1):  # fmt: skip
            runs[key] = CellRun(status="failed", error=f"{type(exc).__name__}: {exc}")
    return runs


def _run_batch[K: Hashable](
    state: StateManager,
    report_id: int,
    pid: int,
    pool: _Pool,
    inp: VariantInput,
    snapshot: MarketSnapshot,
    runs: dict[K, CellRun],
    clones: itertools.count[int],
    jobs: Sequence[tuple[K, Mapping[str, object]]],
    limit: int,
) -> list[tuple[tuple[K, Mapping[str, object]], BrokenProcessPool]]:
    """Run ``jobs`` (``limit`` in flight) into ``runs``; returns the cells a dead worker
    broke, after replacing the broken pool."""
    broken: list[tuple[tuple[K, Mapping[str, object]], BrokenProcessPool]] = []
    pending: dict[Future[CellRun], tuple[tuple[K, Mapping[str, object]], Executor]] = {}
    queue = list(jobs)
    while queue or pending:
        while queue and len(pending) < limit:
            job = queue.pop(0)
            if not state.bankroll_report_owned(report_id, pid):
                raise Fenced(f"report {report_id} is no longer owned by pid {pid}")
            catalog = clone_for_engine(snapshot, snapshot.root / f"w{next(clones)}" / "catalog")
            fut, executor = pool.submit(inp, job[1], catalog, snapshot.archive)
            pending[fut] = (job, executor)
        done = next(iter(_wait_any(pending)))
        job, executor = pending.pop(done)
        try:
            runs[job[0]] = done.result()
        except BrokenProcessPool as exc:
            broken.append((job, exc))
            pool.rebuild(executor)
    return broken


def _wait_any(pending: Mapping[Future[CellRun], object]) -> set[Future[CellRun]]:
    from concurrent.futures import FIRST_COMPLETED, wait

    return wait(list(pending), return_when=FIRST_COMPLETED).done


def _raise_exit(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def run_report(
    report_id: int,
    job_id: int,
    db_path: Path,
    *,
    catalog: Path,
    archive: Path,
    scratch: Path,
) -> int:
    """Claim, run and publish one report; returns the process exit code (0 = completed)."""
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.jobs.manager import process_start_time, run_with_heartbeat

    pid = os.getpid()
    previous = signal.signal(signal.SIGTERM, _raise_exit)
    scratch = db_scratch(scratch, db_path)
    state = StateManager(db_path)
    jobs: BacktestJobManager | None = None
    stop_heartbeat: Callable[[], None] | None = None
    try:
        reclaim_abandoned(scratch, state.list_running_bankroll_report_ids())
        refusal = state.claim_bankroll_report(report_id, job_id, pid, process_start_time(pid))
        if refusal is not None:
            _fail_refused_job(db_path, job_id, pid, f"claim refused: {refusal}")
            return 2
        jobs, stop_heartbeat = run_with_heartbeat(job_id, db_path)
        return _owned_run(state, jobs, report_id, job_id, pid,
                          _Sources(Path(catalog), Path(archive), Path(scratch)))  # fmt: skip
    finally:
        if stop_heartbeat is not None:
            stop_heartbeat()
        if jobs is not None:
            jobs.close()
        state.close()
        signal.signal(signal.SIGTERM, previous)


def _fail_refused_job(db_path: Path, job_id: int, pid: int, error: str) -> None:
    """Fail the job row only if it is open and unclaimed (or this pid's): a refused claim
    never rewrites a finished job nor ends another attempt's."""
    from vibe_quant.db import run_lifecycle
    from vibe_quant.db.connection import get_connection
    from vibe_quant.jobs.manager import process_start_time

    identity = process_start_time(pid)
    conn = get_connection(db_path)
    try:
        claimed = identity is not None and run_lifecycle.claim_job_row(conn, job_id, pid, identity)
        conn.commit()
        if claimed:
            run_lifecycle.finish(conn, job_id, run_lifecycle.JobStatus.FAILED, error,
                                 reconcile=False, owner=pid)  # fmt: skip
    finally:
        conn.close()


def _owned_run(
    state: StateManager,
    jobs: BacktestJobManager,
    report_id: int,
    job_id: int,
    pid: int,
    src: _Sources,
) -> int:
    root = report_root(src.scratch, report_id)
    inputs = (state.get_bankroll_report(report_id) or {}).get("inputs") or {}
    hasher = dsl_hasher(str(inputs.get("strategy_name", "strategy")))
    try:
        try:
            report = _coordinate(state, report_id, pid, root, src)
            refused = state.publish_bankroll_report(report_id, pid, report, dsl_hash=hasher)
        except ReportStop as stop:
            state.end_bankroll_report(report_id, stop.status, str(stop), owner_pid=pid)
            jobs.mark_completed(job_id, error=str(stop))
            return 1
        except Fenced as fenced:
            jobs.mark_completed(job_id, error=str(fenced))
            return 1
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("bankroll report %d failed", report_id)
            state.end_bankroll_report(report_id, "failed", message, owner_pid=pid)
            jobs.mark_completed(job_id, error=message)
            return 1
        except BaseException as exc:  # SIGTERM (SystemExit) / KeyboardInterrupt
            message = f"terminated ({type(exc).__name__})"
            state.end_bankroll_report(report_id, "failed", message, owner_pid=pid)
            jobs.mark_completed(job_id, error=message)
            raise
        if refused is not None:
            jobs.mark_completed(job_id, error=refused)
            return 1
        jobs.mark_completed(job_id)
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


__all__ = [
    "DEFAULT_MULTIPLES",
    "REPLAY_LABEL",
    "SCENARIO_LABEL",
    "CellRun",
    "ReportRequest",
    "build_report",
    "default_scratch_base",
    "dsl_hasher",
    "headline_mismatch",
    "run_cell",
    "run_report",
    "variant_input_of",
]
