"""Carry CLI: ``python -m vibe_quant.portfolio run ...`` (design §4, §8-T6a).

One run is: load and preflight the inputs, execute them with the NT engine or
the reference simulator, compute the metrics, write the six output files into
``--out`` and append one line to the trials ledger (the DSR trial count of §6).

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

Failures (design §1.4) are caught in exactly one tuple -- ``DataUnavailableError``
(and its ``CarryDataError``), ``CarryEquityTooSmall``, ``CarryExecutionError`` and
``ValueError`` -- and become one stderr line plus exit 2. There is no
``except Exception``: an unexpected error propagates with its traceback.

``summary.json`` and the trials line are written with
``json.dumps(..., allow_nan=False)`` after every non-finite float (NaN, +-inf)
is mapped to ``null`` (R52), so both payloads are strict JSON; the trials
record's ``ts`` is the integer epoch-ns of the same instant as ``created_at``.

The parquet files carry the run's records verbatim: the mapping-valued decision
columns (``signals``, ``excluded``, ``weights``, ``prices``, ``target_steps`` and
``orders``) are JSON-encoded strings, one row per decision.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from vibe_quant.errors import DataUnavailableError
from vibe_quant.portfolio.metrics import carry_metrics
from vibe_quant.portfolio.spec import (
    CarryEquityTooSmall,
    CarryExecutionError,
    CarrySpec,
)
from vibe_quant.portfolio.ticks import load_carry_inputs

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

_NS_PER_S = 10**9
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def build_parser() -> argparse.ArgumentParser:
    """The ``vibe-quant portfolio`` parser (T8 adds an ``evaluate`` subcommand)."""
    parser = argparse.ArgumentParser(
        prog="vibe-quant portfolio",
        description="Run the funding-carry book and write its metrics",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="backtest one carry book and write its outputs")
    run.add_argument("--symbols", nargs="+", required=True, help="Futures symbols to trade")
    run.add_argument("--start", required=True, help="First day, YYYY-MM-DD UTC 00:00")
    run.add_argument("--end", required=True, help="Last day, YYYY-MM-DD UTC 00:00")
    run.add_argument("--initial-balance", type=float, default=10_000.0)
    run.add_argument("--legs-divisor", type=int, default=3)
    run.add_argument("--gross", type=float, default=1.0)
    run.add_argument("--band", type=float, default=0.10)
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
    evaluate.add_argument("--symbols", nargs="+", required=True, help="Futures symbols to trade")
    evaluate.add_argument("--start", required=True, help="First day, YYYY-MM-DD UTC 00:00")
    evaluate.add_argument("--end", required=True, help="Last day, YYYY-MM-DD UTC 00:00")
    evaluate.add_argument("--initial-balance", type=float, default=10_000.0)
    evaluate.add_argument("--legs-divisor", type=int, default=3)
    evaluate.add_argument("--gross", type=float, default=1.0)
    evaluate.add_argument("--band", type=float, default=0.10)
    evaluate.add_argument("--draws", type=int, default=200, help="Random-rank null draws")
    evaluate.add_argument("--seed", type=int, default=0, help="Null seed (draw i uses seed + i)")
    evaluate.add_argument("--out", default=None, help=f"Output dir (default: {DEFAULT_OUT_ROOT}/...)")
    evaluate.add_argument("--force", action="store_true", help="Overwrite an existing research.json")
    evaluate.add_argument("--catalog", default=DEFAULT_CATALOG)
    evaluate.add_argument("--archive", default=DEFAULT_ARCHIVE)
    evaluate.add_argument(
        "--trials",
        default=None,
        help=f"DSR trial ledger to read (default: {DEFAULT_OUT_ROOT}/{TRIALS_FILE})",
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


def _git_sha() -> str:
    """The worktree's HEAD, or ``"unknown"`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _summary_json(
    engine: str, spec: CarrySpec, report: CarryReport, git_sha: str
) -> dict[str, Any]:
    """The ``summary.json`` payload: spec, headline, counts, decomposition, by_year."""
    summary = report.summary
    return {
        "spec": spec.to_json(),
        "spec_hash": spec.spec_hash(),
        "engine": engine,
        "git_sha": git_sha,
        "liquidation": summary["liquidation"],
        "headline": {key: summary[key] for key in _HEADLINE_KEYS},
        "counts": {key: summary[key] for key in _COUNT_KEYS},
        "decomposition": {
            "totals": {key: summary[key] for key in _TOTAL_KEYS},
            "funding": {key: summary[key] for key in _FUNDING_KEYS},
            "rounding_drift": {key: summary[key] for key in _DRIFT_KEYS},
            "legs": report.legs,
        },
        "by_year": report.by_year,
    }


def _frames(run: CarryRun, report: CarryReport) -> dict[str, pd.DataFrame]:
    """The five parquet payloads; mapping-valued decision columns are JSON strings."""
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
                }
                for decision in run.decisions
            ]
        ),
    }


def _jsonable(value: Any) -> Any:
    """Non-finite floats become ``null``, so ``allow_nan=False`` never trips (R52)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dumps(payload: Any, **kw: Any) -> str:
    """Strict JSON of ``payload``: NaN/+-inf already mapped to ``null`` (R52)."""
    return json.dumps(_jsonable(payload), allow_nan=False, **kw)


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
            _dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name, frame in frames.items():
            frame.to_parquet(staging / name, index=False)
        _replace_dir(staging, out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)  # gone already after a successful rename


def _trial_line(engine: str, spec: CarrySpec, report: CarryReport, git_sha: str) -> str:
    """One DSR trial record (§4, §6): spec, engine, headline, provenance."""
    summary = report.summary
    now_ns = time.time_ns()
    created_at = (
        _EPOCH + timedelta(seconds=now_ns // _NS_PER_S, microseconds=now_ns % _NS_PER_S // 1000)
    ).isoformat()
    return _dumps(
        {
            "spec_hash": spec.spec_hash(),
            "spec": spec.to_json(),
            "engine": engine,
            "git_sha": git_sha,
            "created_at": created_at,
            "ts": now_ns,
            "weekly_sharpe": summary["weekly_sharpe"],
            "weeks": summary["weeks"],
            "total_return": summary["total_return"],
            "max_drawdown_daily": summary["max_drawdown_daily"],
        },
        sort_keys=True,
    )


def _append_trial(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _report_line(engine: str, summary: Mapping[str, Any], out_dir: Path) -> str:
    """The one stdout line of §8-T6a, every number as ``repr()``."""
    return (
        f"carry[{engine}]: weekly_sharpe={summary['weekly_sharpe']!r} "
        f"total_return={summary['total_return']!r} "
        f"max_dd={summary['max_drawdown_daily']!r} "
        f"fills={summary['n_fills']!r} "
        f"funding_paid={summary['funding_paid_total']!r} "
        f"fees={summary['fees_total']!r} "
        f"fallback={summary['funding_fallback_settlements']!r} "
        f"out={out_dir}"
    )


def _cmd_run(args: argparse.Namespace) -> int:
    """One carry run end to end; returns the process exit code."""
    spec = CarrySpec.from_dates(
        args.symbols,
        args.start,
        args.end,
        initial_balance=args.initial_balance,
        legs_divisor=args.legs_divisor,
        gross=args.gross,
        band=args.band,
    )
    out_dir = _resolve(args.out) if args.out is not None else _default_out_dir(args.engine, spec)
    _check_out_dir(out_dir, args.force)  # R39/R53 before the load, so a refusal costs nothing
    inputs = load_carry_inputs(spec, _resolve(args.catalog), _resolve(args.archive))
    run = _engine_runner(args.engine)(inputs)
    report = carry_metrics(run, inputs)
    git_sha = _git_sha()
    trials = (
        _resolve(args.trials)
        if args.trials is not None
        else _resolve(DEFAULT_OUT_ROOT) / TRIALS_FILE
    )
    _write_outputs(
        out_dir,
        _summary_json(args.engine, inputs.spec, report, git_sha),
        _frames(run, report),
        args.force,
    )
    _append_trial(trials, _trial_line(args.engine, inputs.spec, report, git_sha))
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


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """The T8 research evaluator: reference-only, never appends to the ledger (R38)."""
    if args.draws < 1:
        raise ValueError(f"--draws must be >= 1, got {args.draws}")
    spec = CarrySpec.from_dates(
        args.symbols,
        args.start,
        args.end,
        initial_balance=args.initial_balance,
        legs_divisor=args.legs_divisor,
        gross=args.gross,
        band=args.band,
    )
    out_dir = _resolve(args.out) if args.out is not None else _default_out_dir("research", spec)
    research_file = out_dir / RESEARCH_FILE
    if research_file.exists() and not args.force:
        raise ValueError(f"{research_file} exists; pass --force")
    inputs = load_carry_inputs(spec, _resolve(args.catalog), _resolve(args.archive))
    trials = (
        _resolve(args.trials)
        if args.trials is not None
        else _resolve(DEFAULT_OUT_ROOT) / TRIALS_FILE
    )
    from vibe_quant.portfolio import research

    started = time.perf_counter()
    result = research.evaluate(inputs, draws=args.draws, seed=args.seed, trials_path=trials)
    elapsed = time.perf_counter() - started
    payload = {"spec": inputs.spec.to_json(), "spec_hash": inputs.spec.spec_hash(), **result}
    out_dir.mkdir(parents=True, exist_ok=True)
    research_file.write_text(
        _dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
