"""Carry CLI: ``python -m vibe_quant.portfolio run ...`` (design §4, §8-T6a).

One run is: check the paths and validate the trials ledger, load and preflight
the inputs, execute them with the NT engine or the reference simulator, compute
the metrics, register the trial (one ``completed`` line in the DSR ledger of §6,
:mod:`vibe_quant.portfolio.trials`), then write the six output files into
``--out``. Registering before publishing means a ledger failure publishes no
output, while a trial registered before a failed publication stays counted. A
run ruined (``CarryEquityTooSmall``) AFTER its first decision is a trial too:
one ``ruined`` line, then exit 2; a stop at/before the first decision, a data
error or an execution error records nothing. With ``--liquidation-mode cross``
the completed run is scanned for a cross-margin account bust before its metrics
(:mod:`vibe_quant.portfolio.liquidation`): a bust is ``CarryLiquidated``, one
``liquidated`` line, exit 2; a symbol without a tier table is refused before
any load. A ledger that is the out dir or
lies inside it (``run``), or that is ``research.json`` (``evaluate``), is refused
before any load.

Outputs go into a temp dir next to ``--out`` and are renamed into place only
after the metrics succeed, so a failed run never leaves a partial out dir. An
``--out`` that already exists is a ``ValueError`` unless ``--force``, and
``--force`` may replace it only when it is a previous carry output (a dir that
holds ``summary.json``, R53); the replace moves the old dir aside, renames the
staged dir in and deletes the old one -- a NON-ATOMIC replace (two renames,
R39). The check runs twice: before the inputs are loaded (a refused ``--out``
costs no data load and no engine run, SF2) and again at write time, so a dir
that appears at ``--out`` during the run is refused too and never replaced.

Every path resolves from the repo root (``Path(__file__).resolve().parents[2]``),
never from cwd, so the LIVE entry runs from anywhere. A relative ``--catalog`` /
``--archive`` / ``--out`` / ``--trials`` is repo-root-relative too.

A trend run (``--signal trend``, arch-c9-2) implies the daily schedule and
time-series weights, charges ``DEFAULT_SPREAD`` unless ``--spread`` is given, and
writes the evidence block of :func:`~vibe_quant.portfolio.metrics.provenance` into
``summary.json``; at ``--fill-granularity 1h`` stdout line 1 says it is research
only, not a verdict. ``--gross`` above 1.0 needs ``--allow-gross-above-one`` and
prints a warning (liquidation is not modelled). ``--universe FILE`` replaces
``--symbols`` and is recorded as ``{path, sha256}``. A default funding run writes
main's keys and columns (R3).

``evaluate`` takes the same spec flags. A trend evaluate needs ``--family`` (the frozen
trial family; an arm outside it is refused before any load), may take ``--criteria``
(the verdict; never PASS without it, ``research_only`` at 1h) and ``--signal-lag-days``
(NC1: the same spec loaded K days earlier for its closes only); it prints the
statistic, null, placebo, beta/alpha t, fee2x, DSR and verdict lines. Those three
flags are refused on funding, whose ``research.json`` is unchanged (R63).

Failures (design §1.4) are caught in exactly one tuple -- ``DataUnavailableError``
(and its ``CarryDataError``), ``CarryEquityTooSmall``, ``CarryExecutionError`` and
``ValueError`` -- and become one stderr line plus exit 2. There is no
``except Exception``: an unexpected error propagates with its traceback.

``summary.json`` and the trials lines are written with
``json.dumps(..., allow_nan=False)`` after every non-finite float (NaN, +-inf)
is mapped to ``null`` (R52), so both payloads are strict JSON; the trials
record's ``ts`` is the integer epoch-ns of the same instant as ``created_at``.

The parquet files carry the run's records verbatim: the mapping-valued decision
columns (``signals``, ``excluded``, ``weights``, ``prices``, ``target_steps`` and
``orders``, plus ``vols`` under inverse_vol weighting) are JSON-encoded strings,
one row per decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from vibe_quant.errors import DataUnavailableError
from vibe_quant.portfolio import research
from vibe_quant.portfolio.metrics import carry_metrics, provenance
from vibe_quant.portfolio.sizing import DEFAULT_SPREAD
from vibe_quant.portfolio.spec import (
    CarryEquityTooSmall,
    CarryExecutionError,
    CarrySpec,
)
from vibe_quant.portfolio.ticks import load_carry_inputs
from vibe_quant.portfolio.trials import (
    completion_line,
    dumps,
    failure_line,
    failure_outcome,
    git_sha,
    read_trials,
    record_completion,
    record_failure,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from vibe_quant.portfolio.ledger import CarryRun
    from vibe_quant.portfolio.metrics import CarryReport
    from vibe_quant.portfolio.ticks import CarryInputs

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = "data/catalog"
DEFAULT_ARCHIVE = "data/archive/raw_data.db"
DEFAULT_OUT_ROOT = "data/portfolio"
TRIALS_FILE = "trials.jsonl"

SUMMARY_FILE = "summary.json"
RESEARCH_FILE = "research.json"
PARQUET_FILES = (
    "daily.parquet",
    "weekly.parquet",
    "fills.parquet",
    "funding.parquet",
    "decisions.parquet",
)

# §4: the T4 summary dict split into the headline, the counts and the decomposition.
_HEADLINE_KEYS = (
    "weekly_sharpe",
    "weekly_sharpe_lo_se",
    "weeks",
    "daily_sharpe_365",
    "total_return",
    "max_drawdown_daily",
)
_COUNT_KEYS = ("n_fills", "n_decisions", "flat_decisions", "band_skips", "min_notional_skips")
_TOTAL_KEYS = (
    "price_pnl_total",
    "funding_paid_total",
    "fees_total",
    "turnover_total",
    "turnover_annual",
)
_FUNDING_KEYS = (
    "funding_fallback_settlements",
    "funding_estimated",
    "funding_fallback_amount",
)
_DRIFT_KEYS = ("rounding_drift_mean", "rounding_drift_max")
# arch-c9-2: the trend run's extra summary keys (a funding run keeps main's set, R3)
_TREND_HEADLINE_KEYS = ("max_gross_daily", "days_gross_above_cap")
_TREND_FUNDING_KEYS = ("funding_switch_holes_upper", "funding_fallback_charges")
DEFAULT_SPREAD_SOURCE = "DEFAULT_SPREAD (spread table not landed)"

_NS_PER_S = 10**9
_LIQUIDATION_HELP = (
    "off (default; summary liquidation 'not_modelled') or cross: after the run completes, "
    "scan 1m MARK bars for a cross-margin account bust (Binance USDT-M tiers); a bust fails "
    "the run (exit 2, CarryLiquidated) and is recorded as a 'liquidated' trial. Conservative "
    "model, not a venue replay: each leg at its adverse extreme of the same minute together; "
    "a fill-minute wick counts at the fill ts; a leg inside an allowed (<= 60 min) MARK gap "
    "is valued at its previous close; a missing mark, a symbol without a tier table or an "
    "out-of-tier mark is a CarryDataError (no trial); a tier error can mask an earlier hit "
    "in its segment; an executor abort stays itself, never a bust"
)


def _add_variant_args(parser: argparse.ArgumentParser) -> None:
    """The signal/weighting/trend/tier flags ``run`` and ``evaluate`` share (arch-c9-2)."""
    parser.add_argument(
        "--allow-gross-above-one",
        action="store_true",
        help="Allow --gross > 1.0 (liquidation is not modelled; a warning is printed)",
    )
    parser.add_argument(
        "--signal",
        default="funding",
        help="funding (default: weekly trailing-funding rank) or trend (daily close-vs-SMA "
        "ensemble, time-series weights with a portfolio-vol target)",
    )
    parser.add_argument(
        "--weighting",
        default=None,
        help="Leg weights: equal (funding default) or inverse_vol (per side, 1/sigma; may "
        "fail closed with CarryEquityTooSmall on a small book); time_series for trend",
    )
    parser.add_argument(
        "--vol-lookback-days",
        type=int,
        default=30,
        help="inverse_vol sigma lookback, or the trend book-vol lookback (days)",
    )
    parser.add_argument("--direction", default=None, help="trend: long_flat or long_short")
    parser.add_argument(
        "--trend-filter", default=None, help="trend (long_flat): regime symbol, e.g. BTCUSDT"
    )
    parser.add_argument(
        "--vol-target", type=float, default=None, help="trend: annual book vol target, (0, 2]"
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=None,
        help=f"trend: full quoted spread charged per fill as |q p| spread / 2 "
        f"(default {DEFAULT_SPREAD}; funding: 0 only)",
    )
    parser.add_argument(
        "--fill-granularity",
        default="1m",
        help="1m (catalog bars, default) or 1h (archive bars + the research instrument "
        "snapshot: research only, not a verdict)",
    )


def build_parser() -> argparse.ArgumentParser:
    """The ``vibe-quant portfolio`` parser (T8 adds an ``evaluate`` subcommand)."""
    parser = argparse.ArgumentParser(
        prog="vibe-quant portfolio",
        description="Run the funding-carry book and write its metrics",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="backtest one carry book and write its outputs")
    cohort = run.add_mutually_exclusive_group(required=True)
    cohort.add_argument("--symbols", nargs="+", help="Futures symbols to trade")
    cohort.add_argument(
        "--universe",
        default=None,
        help="File with one symbol per line (# comments); recorded as {path, sha256}",
    )
    run.add_argument("--start", required=True, help="First day, YYYY-MM-DD UTC 00:00")
    run.add_argument("--end", required=True, help="Last day, YYYY-MM-DD UTC 00:00")
    run.add_argument("--initial-balance", type=float, default=10_000.0)
    run.add_argument("--legs-divisor", type=int, default=3)
    run.add_argument("--gross", type=float, default=1.0)
    run.add_argument("--band", type=float, default=0.10)
    _add_variant_args(run)
    run.add_argument(
        "--min-quote-volume",
        type=float,
        default=None,
        help="Liquidity filter: exclude a symbol whose mean daily 1m quote volume (USDT) "
        "over the 7 prior UTC days is below this (off by default)",
    )
    run.add_argument("--liquidation-mode", default="off", help=_LIQUIDATION_HELP)
    run.add_argument(
        "--engine",
        choices=("nt", "reference"),
        default="nt",
        help="Executor: the NautilusTrader engine (default) or the no-NT reference oracle",
    )
    run.add_argument("--out", default=None, help=f"Output dir (default: {DEFAULT_OUT_ROOT}/...)")
    run.add_argument("--force", action="store_true", help="Replace an existing --out dir")
    run.add_argument("--catalog", default=DEFAULT_CATALOG)
    run.add_argument("--archive", default=DEFAULT_ARCHIVE)
    run.add_argument(
        "--trials",
        default=None,
        help=f"DSR trial ledger to append to (default: {DEFAULT_OUT_ROOT}/{TRIALS_FILE})",
    )
    evaluate = commands.add_parser(
        "evaluate", help="research robustness report over the reference (no NT, T8)"
    )
    evaluate_cohort = evaluate.add_mutually_exclusive_group(required=True)
    evaluate_cohort.add_argument("--symbols", nargs="+", help="Futures symbols to trade")
    evaluate_cohort.add_argument(
        "--universe",
        default=None,
        help="File with one symbol per line (# comments); recorded as {path, sha256}",
    )
    evaluate.add_argument("--start", required=True, help="First day, YYYY-MM-DD UTC 00:00")
    evaluate.add_argument("--end", required=True, help="Last day, YYYY-MM-DD UTC 00:00")
    evaluate.add_argument("--initial-balance", type=float, default=10_000.0)
    evaluate.add_argument("--legs-divisor", type=int, default=3)
    evaluate.add_argument("--gross", type=float, default=1.0)
    evaluate.add_argument("--band", type=float, default=0.10)
    _add_variant_args(evaluate)
    evaluate.add_argument(
        "--family",
        default=None,
        help="trend (required): frozen family JSON {n_prior, arms: [arm hashes]}; "
        "DSR N = n_prior + len(arms), theoretical variance",
    )
    evaluate.add_argument(
        "--criteria",
        default=None,
        help="trend: pre-registered criteria JSON {metric: {op, bar}}; without it the "
        "verdict is never PASS",
    )
    evaluate.add_argument(
        "--signal-lag-days",
        type=int,
        default=0,
        help="trend (NC1 calibration): lag only the signal by K days; prices and book vol "
        "stay those of the decision",
    )
    evaluate.add_argument(
        "--min-quote-volume",
        type=float,
        default=None,
        help="Liquidity filter: exclude a symbol whose mean daily 1m quote volume (USDT) "
        "over the 7 prior UTC days is below this (off by default)",
    )
    evaluate.add_argument("--liquidation-mode", default="off", help=_LIQUIDATION_HELP)
    evaluate.add_argument(
        "--draws", type=int, default=200, help="Null draws (funding: random rank; trend: rotation)"
    )
    evaluate.add_argument("--seed", type=int, default=0, help="Null seed (draw i uses seed + i)")
    evaluate.add_argument("--out", default=None, help=f"Output dir (default: {DEFAULT_OUT_ROOT}/...)")
    evaluate.add_argument("--force", action="store_true", help="Overwrite an existing research.json")
    evaluate.add_argument("--catalog", default=DEFAULT_CATALOG)
    evaluate.add_argument("--archive", default=DEFAULT_ARCHIVE)
    evaluate.add_argument(
        "--trials",
        default=None,
        help=(
            "DSR trial ledger (read; evaluate records its own spec once) "
            f"(default: {DEFAULT_OUT_ROOT}/{TRIALS_FILE})"
        ),
    )
    return parser


def _resolve(value: str | Path) -> Path:
    """A path as given, or relative to the repo root -- never to cwd."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _date(ts_ns: int) -> str:
    return datetime.fromtimestamp(ts_ns // _NS_PER_S, UTC).strftime("%Y-%m-%d")


def _default_out_dir(engine: str, spec: CarrySpec) -> Path:
    """``<out root>/<engine>-<start>-<end>-<sha8 of the spec json>`` (§4)."""
    name = f"{engine}-{_date(spec.start_ns)}-{_date(spec.end_ns)}-{spec.spec_hash()[:8]}"
    return _resolve(DEFAULT_OUT_ROOT) / name


def _engine_runner(engine: str) -> Callable[[CarryInputs], CarryRun]:
    """The executor for ``--engine``; NT is imported lazily, so the reference needs no NT."""
    if engine == "nt":
        from vibe_quant.portfolio.runner import run_carry_engine

        return run_carry_engine
    from vibe_quant.portfolio.reference import simulate_reference

    return simulate_reference


def _summary_json(
    engine: str,
    spec: CarrySpec,
    report: CarryReport,
    git_sha: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``summary.json`` payload: spec, headline, counts, decomposition, by_year.

    A trend run adds its gross drift, spread total and funding counters to those
    blocks and ``evidence`` (:func:`~vibe_quant.portfolio.metrics.provenance`) at the
    top level (R53, R54); a funding run adds only ``universe``, and only when run
    from ``--universe``.
    """
    summary = report.summary
    trend = spec.signal == "trend"
    headline = _HEADLINE_KEYS + (_TREND_HEADLINE_KEYS if trend else ())
    totals = _TOTAL_KEYS + (("spread_total",) if trend else ())
    funding = _FUNDING_KEYS + (_TREND_FUNDING_KEYS if trend else ())
    payload: dict[str, Any] = {
        "spec": spec.to_json(),
        "spec_hash": spec.spec_hash(),
        "engine": engine,
        "git_sha": git_sha,
        "liquidation": summary["liquidation"],
        "headline": {key: summary[key] for key in headline},
        "counts": {key: summary[key] for key in _COUNT_KEYS},
        "decomposition": {
            "totals": {key: summary[key] for key in totals},
            "funding": {key: summary[key] for key in funding},
            "rounding_drift": {key: summary[key] for key in _DRIFT_KEYS},
            "legs": report.legs,
        },
        "by_year": report.by_year,
    }
    if evidence is not None:
        payload.update(evidence)
    return payload


def _frames(run: CarryRun, report: CarryReport) -> dict[str, pd.DataFrame]:
    """The five parquet payloads; mapping-valued decision columns are JSON strings.

    A trend run adds the fill ``spread`` and the decision's ``book_state``,
    ``min_leg_skips`` and Decimal ``exposure_after_plan`` (as text).
    """
    trend = run.spec.signal == "trend"
    return {
        "daily.parquet": pd.DataFrame(
            [
                {"ts": ts, "equity": equity, "return": move}
                for ts, equity, move in report.daily
            ]
        ),
        "weekly.parquet": pd.DataFrame(
            [{"ts": ts, "return": move} for ts, move in report.weekly]
        ),
        "fills.parquet": pd.DataFrame(
            [
                {
                    "ts": fill.ts,
                    "symbol": fill.symbol,
                    "steps": fill.steps,
                    "qty": fill.qty,
                    "price": fill.price,
                    "fee": fill.fee,
                    "reduce_only": fill.reduce_only,
                    **({"spread": fill.spread} if trend else {}),
                }
                for fill in run.fills
            ]
        ),
        "funding.parquet": pd.DataFrame(
            [
                {
                    "ts": payment.ts,
                    "symbol": payment.symbol,
                    "amount": payment.amount,
                    "fallback": payment.fallback,
                }
                for payment in run.payments
            ]
        ),
        "decisions.parquet": pd.DataFrame(
            [
                {
                    "ts": decision.ts,
                    "equity": decision.equity,
                    "signals": json.dumps(decision.signals, sort_keys=True),
                    "excluded": json.dumps(decision.excluded, sort_keys=True),
                    "weights": json.dumps(decision.weights, sort_keys=True),
                    "prices": json.dumps(decision.prices, sort_keys=True),
                    "target_steps": json.dumps(decision.target_steps, sort_keys=True),
                    "orders": json.dumps(
                        [
                            [order.symbol, order.steps, order.reduce_only]
                            for order in decision.orders
                        ]
                    ),
                    "band_skips": decision.band_skips,
                    "min_notional_skips": decision.min_notional_skips,
                    "rounding_drift": decision.rounding_drift,
                    **(
                        {"vols": json.dumps(decision.vols, sort_keys=True)}
                        if run.spec.weighting == "inverse_vol"
                        else {}
                    ),
                    **(
                        {
                            "book_state": decision.book_state,
                            "min_leg_skips": decision.min_leg_skips,
                            "exposure_after_plan": str(decision.exposure_after_plan),
                        }
                        if trend
                        else {}
                    ),
                }
                for decision in run.decisions
            ]
        ),
    }


def _check_out_dir(out_dir: Path, force: bool) -> None:
    """Refuse an existing ``--out`` before any input load (R39, R53, SF2).

    Without ``--force`` nothing may be overwritten. With ``--force`` the dir is
    only replaceable when it is a previous carry output (holds ``summary.json``),
    so ``--force`` pointing at an unrelated dir deletes nothing.
    """
    if not out_dir.exists():
        return
    if not force:
        raise ValueError(f"{out_dir} exists; pass --force")
    if not (out_dir / SUMMARY_FILE).is_file():
        raise ValueError(f"refusing to replace {out_dir}: not a carry output dir")


def _replace_dir(staging: Path, out_dir: Path) -> None:
    """Rename the staged dir into place; an existing dir is moved aside first (non-atomic).

    If the second rename fails the old dir is renamed back into place before the
    error propagates, so a failed replace restores the previous output instead of
    destroying it (the pre-fix ``finally: rmtree(aside)`` deleted the old dir).
    """
    if not out_dir.exists():
        staging.rename(out_dir)
        return
    aside = staging.with_name(f"{staging.name}.old")
    out_dir.rename(aside)
    try:
        staging.rename(out_dir)
    except BaseException:
        aside.rename(out_dir)  # restore the old output, then re-raise
        raise
    shutil.rmtree(aside, ignore_errors=True)


def _write_outputs(
    out_dir: Path,
    payload: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    force: bool,
) -> None:
    """Write the six files into a temp dir next to ``out_dir``, then rename it in (R39).

    ``_check_out_dir`` runs again here -- the late half of the ``--out`` guard: a dir
    that appears at ``out_dir`` while the inputs load or the engine runs (another
    process, or a race with the early check) is refused with the same ``ValueError``
    before anything is staged or replaced. With ``--force`` the R53 check still
    applies: only a previous carry output may be replaced.
    """
    _check_out_dir(out_dir, force)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    try:
        (staging / SUMMARY_FILE).write_text(
            dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name, frame in frames.items():
            frame.to_parquet(staging / name, index=False)
        _replace_dir(staging, out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)  # gone already after a successful rename


def _report_line(engine: str, summary: Mapping[str, Any], out_dir: Path) -> str:
    """The one stdout line of §8-T6a, every number as ``repr()``.

    A trend run also prints its spread total and both funding counters (the causal
    grid's switch-hole upper bound and its fallback charges).
    """
    trend = (
        f"spread={summary['spread_total']!r} "
        f"switch_holes={summary['funding_switch_holes_upper']!r} "
        f"fallback_charges={summary['funding_fallback_charges']!r} "
        if "spread_total" in summary
        else ""
    )
    return (
        f"carry[{engine}]: weekly_sharpe={summary['weekly_sharpe']!r} "
        f"total_return={summary['total_return']!r} "
        f"max_dd={summary['max_drawdown_daily']!r} "
        f"fills={summary['n_fills']!r} "
        f"funding_paid={summary['funding_paid_total']!r} "
        f"fees={summary['fees_total']!r} "
        f"fallback={summary['funding_fallback_settlements']!r} "
        f"{trend}"
        f"out={out_dir}"
    )


def _trials_path(args: argparse.Namespace) -> Path:
    if args.trials is not None:
        return _resolve(args.trials)
    return _resolve(DEFAULT_OUT_ROOT) / TRIALS_FILE


_MAX_LINKS = 40


def _resolution_path(path: Path) -> list[Path]:
    """``path`` (absolute), each symlink its resolution follows, and the final target.

    Resolved left to right like ``realpath``; a symlink is listed by its own
    location (parent already resolved), so an alias kept inside an out dir shows up
    even when the target lies outside it.
    """
    seen = [Path(os.path.normpath(path))]
    resolved = Path(path.anchor)
    pending = list(path.parts[1:])
    links = 0
    while pending and links <= _MAX_LINKS:
        part = pending.pop(0)
        if part == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / part
        if not candidate.is_symlink():
            resolved = candidate
            continue
        links += 1
        seen.append(candidate)
        target = Path(os.readlink(candidate))
        parts = target.parts
        if target.is_absolute():
            resolved, parts = Path(target.anchor), parts[1:]
        pending = [*parts, *pending]
    seen.append(resolved.joinpath(*pending))
    return seen


def _check_run_collision(trials: Path, out_dir: Path) -> None:
    """Refuse a ledger that ``--force`` replacement of the out dir could delete (R14c).

    The ledger collides when its lexical path, its resolved path or any symlink
    its resolution follows is the out dir (lexical or resolved) or lies inside it.
    """
    outs = {Path(os.path.normpath(out_dir)), out_dir.resolve()}
    for hop in _resolution_path(trials):
        if any(hop == out or out in hop.parents for out in outs):
            raise ValueError(f"trial ledger {trials} lies in the output dir {out_dir}")


def _check_evaluate_collision(trials: Path, research_file: Path) -> None:
    """Refuse a ledger that is ``research.json`` itself, by path or by inode (R14c)."""
    same = trials.resolve() == research_file.resolve()
    if not same and trials.exists() and research_file.exists():
        same = os.path.samefile(trials, research_file)
    if same:
        raise ValueError(f"trial ledger {trials} is the research output {research_file}")


def _check_tiers(spec: CarrySpec) -> None:
    """Cross needs a shipped tier table for every symbol: refuse before any load (R46)."""
    if spec.liquidation_mode == "cross":
        from vibe_quant.portfolio.liquidation import require_tier_symbols

        require_tier_symbols(spec.symbols)


def _record_ruin(
    trials: Path, engine: str, inputs: CarryInputs, error: CarryEquityTooSmall, sha: str
) -> None:
    """A ruin after the first decision is a trial: one failure line (once per outcome)."""
    outcome = failure_outcome(error, inputs.decisions)
    if outcome is not None and error.decision_ns is not None:
        record_failure(trials, failure_line(engine, inputs.spec, outcome, error.decision_ns, sha))


def _universe(value: str) -> tuple[list[str], dict[str, str]]:
    """The symbols of a universe file (one per line, ``#`` comments) and its identity."""
    path = _resolve(value)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"universe file {path} unreadable: {error}") from error
    lines = (line.strip() for line in raw.decode().splitlines())
    symbols = [line for line in lines if line and not line.startswith("#")]
    return symbols, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _run_spec(args: argparse.Namespace) -> tuple[CarrySpec, str, dict[str, str] | None]:
    """The run's spec, its spread source and its universe identity (``None``: --symbols).

    The trend signal implies the daily schedule and time-series weights; its spread
    defaults to ``DEFAULT_SPREAD``. Funding keeps today's defaults (R3).
    """
    trend = args.signal == "trend"
    symbols, universe = (args.symbols, None) if args.universe is None else _universe(args.universe)
    spread = args.spread
    source = "--spread"
    if spread is None:
        spread, source = (DEFAULT_SPREAD, DEFAULT_SPREAD_SOURCE) if trend else (0.0, "none")
    weighting = args.weighting
    if weighting is None:
        weighting = "time_series" if trend else "equal"
    spec = CarrySpec.from_dates(
        symbols,
        args.start,
        args.end,
        initial_balance=args.initial_balance,
        legs_divisor=args.legs_divisor,
        gross=args.gross,
        band=args.band,
        weighting=weighting,
        vol_lookback_days=args.vol_lookback_days,
        min_quote_volume=args.min_quote_volume,
        liquidation_mode=args.liquidation_mode,
        signal=args.signal,
        schedule="daily" if trend else "weekly",
        fill_granularity=args.fill_granularity,
        direction=args.direction,
        trend_filter=args.trend_filter,
        vol_target=args.vol_target,
        spread=spread,
        allow_gross_above_one=args.allow_gross_above_one,
    )
    return spec, source, universe


def _cmd_run(args: argparse.Namespace) -> int:
    """One carry run end to end; returns the process exit code."""
    spec, spread_source, universe = _run_spec(args)
    if spec.gross > 1.0:
        print(
            f"WARNING: gross {spec.gross!r} > 1.0 and liquidation is not modelled",
            file=sys.stderr,
        )
    out_dir = _resolve(args.out) if args.out is not None else _default_out_dir(args.engine, spec)
    trials = _trials_path(args)
    # R14c/R39/R53 and ledger validation before the load, so a refusal costs nothing
    _check_run_collision(trials, out_dir)
    _check_out_dir(out_dir, args.force)
    read_trials(trials)
    _check_tiers(spec)
    inputs = load_carry_inputs(spec, _resolve(args.catalog), _resolve(args.archive))
    sha = git_sha()
    try:
        run = _engine_runner(args.engine)(inputs)
        report = carry_metrics(run, inputs)
    except CarryEquityTooSmall as error:
        _record_ruin(trials, args.engine, inputs, error, sha)
        raise
    evidence: dict[str, Any] | None = None
    if spec.signal == "trend":
        evidence = provenance(run, inputs, spread_source=spread_source, universe=universe)
    elif universe is not None:
        evidence = {"universe": universe}
    record_completion(
        trials, completion_line(args.engine, inputs.spec, report.summary, sha), history=True
    )
    _write_outputs(
        out_dir,
        _summary_json(args.engine, inputs.spec, report, sha, evidence),
        _frames(run, report),
        args.force,
    )
    if spec.fill_granularity == "1h":
        print(f"carry[{args.engine}]: fill_granularity=1h (research only, not a verdict)")
    print(_report_line(args.engine, report.summary, out_dir))
    return 0


def _research_line(result: Mapping[str, Any], out_dir: Path, elapsed: float) -> str:
    """The one stdout line of ``evaluate`` (§8-T8); includes the total wall time (R55)."""
    return (
        f"research[reference]: draws={result['draws']!r} "
        f"ruined={result['ruined_draws']!r} "
        f"p={result['p_value']!r} "
        f"p_pre_cost={result['p_value_pre_cost']!r} "
        f"dsr={result['dsr']!r} "
        f"out={out_dir} wall={elapsed:.3f}s"
    )


def _trend_research_lines(result: Mapping[str, Any], out_dir: Path, elapsed: float) -> list[str]:
    """The trend evaluate's stdout (bead AC): statistic, null p, placebo, beta/alpha t,
    fee2x and DSR lines, then the verdict; a 1h run says research only first (R62)."""
    exposure = result["exposure"]
    lines = [
        f"statistic={result['statistic']} sharpe={result['sharpe']!r} "
        f"total_return={result['total_return']!r} max_dd={result['max_drawdown_daily']!r}",
        f"null kind=rotation draws={result['draws']!r} seed={result['seed']!r} "
        f"ruined={result['ruined_draws']!r} undefined={result['undefined_draws']!r} "
        f"degenerate_segments={result['degenerate_segments']!r} p={result['p_value']!r} "
        f"null_percentile={result['null_percentile']!r} "
        f"p_pre_cost={result['p_value_pre_cost']!r} "
        f"gross own={exposure['own']['mean_gross']!r} "
        f"null={None if exposure['null_mean'] is None else exposure['null_mean']['mean_gross']!r}",
        f"placebo kind={result['placebo_kind']} sharpe={result['placebo_sharpe']!r}",
        f"beta_btc={result['beta_btc']!r} alpha_t_btc={result['alpha_t_btc']!r} "
        f"beta_ew={result['beta_ew']!r} alpha_t={result['alpha_t']!r}",
        f"fee2x sharpe={result['fee2x_sharpe']!r} capped_sharpe={result['capped_sharpe']!r} "
        f"drop_top3_weeks_sharpe={result['drop_top3_weeks_sharpe']!r}",
        f"dsr={result['dsr']!r} dsr_p={result['dsr_p_value']!r} "
        f"N={result['dsr_num_trials']!r} variance={result['dsr_variance']} "
        f"family={result['family']['path']}",
        f"verdict={result['verdict']} criteria_all_pass={result['criteria_all_pass']!r} "
        f"lag={result['signal_lag_days']!r} out={out_dir} wall={elapsed:.3f}s",
    ]
    if result["fill_granularity"] == "1h":
        lines.insert(0, "fill_granularity=1h (research only, not a verdict)")
    return [f"research[reference]: {line}" for line in lines]


def _evaluate_files(
    args: argparse.Namespace, spec: CarrySpec
) -> tuple[research.Family | None, research.Criteria | None]:
    """The frozen family and criteria files, validated before any load (R55, R62)."""
    if spec.signal != "trend":
        for flag, value in (
            ("--family", args.family),
            ("--criteria", args.criteria),
            ("--signal-lag-days", args.signal_lag_days or None),
        ):
            if value is not None:
                raise ValueError(f"{flag} needs --signal trend")
        return None, None
    if args.family is None:
        raise ValueError("signal='trend' needs a frozen family (--family)")
    family = research.load_family(_resolve(args.family))
    research.require_arm(family, spec)
    criteria = None if args.criteria is None else research.load_criteria(_resolve(args.criteria))
    research.lag_spec(spec, args.signal_lag_days)
    return family, criteria


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """The research evaluator: reference-only; records its own spec once (R38).

    A trend evaluate also needs its frozen ``--family``; ``--signal-lag-days K`` loads
    the same spec K days earlier for the closes only (NC1, C9-17).
    """
    if args.draws < 1:
        raise ValueError(f"--draws must be >= 1, got {args.draws}")
    spec, spread_source, universe = _run_spec(args)
    if spec.gross > 1.0:
        print(
            f"WARNING: gross {spec.gross!r} > 1.0 and liquidation is not modelled",
            file=sys.stderr,
        )
    family, criteria = _evaluate_files(args, spec)
    out_dir = _resolve(args.out) if args.out is not None else _default_out_dir("research", spec)
    research_file = out_dir / RESEARCH_FILE
    trials = _trials_path(args)
    _check_evaluate_collision(trials, research_file)
    if research_file.exists() and not args.force:
        raise ValueError(f"{research_file} exists; pass --force")
    read_trials(trials)
    _check_tiers(spec)
    catalog, archive = _resolve(args.catalog), _resolve(args.archive)
    inputs = load_carry_inputs(spec, catalog, archive)
    lag = args.signal_lag_days
    if lag:
        longer = load_carry_inputs(research.lag_spec(spec, lag), catalog, archive)
        inputs = replace(
            inputs,
            trend_closes=research.extend_closes(inputs.trend_closes or {}, longer.trend_closes or {}),
        )

    started = time.perf_counter()
    result = research.evaluate(
        inputs,
        draws=args.draws,
        seed=args.seed,
        trials_path=trials,
        family=family,
        criteria=criteria,
        signal_lag_days=lag,
        spread_source=spread_source,
        universe=universe,
    )
    elapsed = time.perf_counter() - started
    payload = {"spec": inputs.spec.to_json(), "spec_hash": inputs.spec.spec_hash(), **result}
    if spec.signal != "trend" and universe is not None:
        payload["universe"] = universe
    out_dir.mkdir(parents=True, exist_ok=True)
    research_file.write_text(
        dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if spec.signal == "trend":
        print("\n".join(_trend_research_lines(result, out_dir, elapsed)))
    else:
        print(_research_line(result, out_dir, elapsed))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """The CLI entry point: 0 on success, 2 on a known failure (§1.4)."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "evaluate":
            return _cmd_evaluate(args)
        return _cmd_run(args)
    except (DataUnavailableError, CarryEquityTooSmall, CarryExecutionError, ValueError) as error:
        engine = getattr(args, "engine", args.command)
        print(f"carry[{engine}]: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
