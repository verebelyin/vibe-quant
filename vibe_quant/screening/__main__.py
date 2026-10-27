"""Screening pipeline CLI entrypoint.

Provides run/list/status subcommands for screening backtests.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.sizing_settings import SizingSettings

logger = logging.getLogger(__name__)


def _store_compiler_version(state: object, run_id: int, version: str) -> None:
    """Store compiler version hash in run's notes JSON."""
    import json

    from vibe_quant.db.result_notes import dump_notes, parse_notes

    try:
        conn = getattr(state, "conn", None)
        if conn is None:
            return
        row = conn.execute(
            "SELECT notes FROM backtest_results WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row and row[0]:
            notes = cast("dict[str, object]", parse_notes(row[0]))
            notes["compiler_version"] = version
            conn.execute(
                "UPDATE backtest_results SET notes = ? WHERE run_id = ?",
                (dump_notes(notes), run_id),
            )
            conn.commit()
        else:
            # No results row yet — store in parameters instead
            row2 = conn.execute(
                "SELECT parameters FROM backtest_runs WHERE id = ?", (run_id,)
            ).fetchone()
            params = json.loads(row2[0]) if row2 and row2[0] else {}
            params["compiler_version"] = version
            conn.execute(
                "UPDATE backtest_runs SET parameters = ? WHERE id = ?",
                (json.dumps(params), run_id),
            )
            conn.commit()
    except Exception:
        pass  # Non-critical metadata


def cmd_run(args: argparse.Namespace) -> int:
    """Run screening pipeline for a given run_id.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.jobs.manager import run_with_heartbeat

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    state = StateManager(db_path)
    manager, stop_heartbeat = run_with_heartbeat(args.run_id, db_path)

    def fail(message: str) -> int:
        """Early exit: record the failure so the run doesn't stay 'running' forever."""
        print(message)
        manager.mark_completed(args.run_id, error=message)
        return 1

    try:
        run_config = state.get_backtest_run(args.run_id)
        if run_config is None:
            return fail(f"Run {args.run_id} not found")

        from vibe_quant.dsl.schema import StrategyDSL
        from vibe_quant.dsl.translator import translate_dsl_config
        from vibe_quant.screening.pipeline import create_screening_pipeline

        # Check for DSL override (used by discovery replay)
        parameters = run_config.get("parameters", {}) or {}
        dsl_override = parameters.get("dsl_override") if isinstance(parameters, dict) else None
        if dsl_override and isinstance(dsl_override, dict):
            raw_config = dsl_override
        else:
            strategy_id = run_config["strategy_id"]
            if strategy_id is None or str(strategy_id) == "None":
                return fail(
                    f"Run {args.run_id} has no strategy_id "
                    f"(mode: {run_config.get('run_mode', '?')})"
                )
            strategy = state.get_strategy(int(str(strategy_id)))
            if strategy is None:
                return fail(f"Strategy {strategy_id} not found")
            raw_config = translate_dsl_config(
                strategy["dsl_config"], strategy_name=str(strategy["name"])
            )
        dsl = StrategyDSL.model_validate(raw_config)

        # Extract run parameters
        symbols = run_config["symbols"]
        start_date = run_config["start_date"]
        end_date = run_config["end_date"]
        parameters = run_config.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}

        # The run's launch settings, parsed ONE field at a time in the screening CLI's
        # order (balance, liquidation, sizing, fee), printing each field's line(s)
        # right after that field parses. A later field's failure therefore still
        # leaves the earlier lines on stdout (R27c), exactly as before. The fee
        # schedule is venue-level and screening runs have no ``backtest_results`` row
        # to hold a notes payload, so the effective settings are logged into the run
        # log (bd vibe-quant-2gyk1.4); the run-level liquidation settings come from
        # the run's own parameters (no screening CLI flag) and an invalid stored mode
        # or leverage raises here (E10) before any engine exists. The venue leverage
        # applies in EVERY mode (vibe-quant-6wq6w), so it is logged separately from
        # the mode-on liquidation notes. The account size (bead vibe-quant-zpbmo) and
        # the sizing settings (bead vibe-quant-2gyk1.2) are logged the same way;
        # create_screening_pipeline parses the same values for the venue, so an
        # invalid stored one fails here or there, either way through the except below
        # (never silently defaulted).
        from vibe_quant.launch_settings import (
            PARSE_ORDER,
            SCREENING_SKIP_KEYS,
            iter_fields,
        )

        for field, value in iter_fields(parameters, PARSE_ORDER["screening_cli"]):
            if field == "fee":
                print(f"Screening fee schedule: {cast('FeeSchedule', value).to_notes()}")
            elif field == "liquidation":
                liquidation = cast("LiquidationSettings", value)
                print(f"Screening liquidation: {liquidation.to_notes()}")
                print(f"Screening venue leverage: {liquidation.leverage:g}x")
            elif field == "balance":
                print(f"Screening starting balance: {value} USDT")
            elif field == "sizing":
                sizing = cast("SizingSettings", value)
                print(
                    f"Screening sizing: max_position_pct={sizing.max_position_pct:g} "
                    f"risk_per_trade={sizing.risk_per_trade:g}"
                )

        # Fail closed on a top-level parameter that names no compiled config field, and
        # forward the ones that do to the runner as typed fixed config overrides, like
        # validation (bead vibe-quant-nxmxd.5): the sweep is numeric-only and cannot carry
        # a string or boolean (bead vibe-quant-fahwt). The run's own launch settings and
        # the screening metadata its own rows carry are skipped (``SCREENING_SKIP_KEYS``).
        # The split is the ONE helper the launch API's 422 precheck runs too, so an
        # approved launch cannot disagree with this run: it also refuses a malformed
        # ``sweep`` and a parameter given both at top level and in the sweep. Placed AFTER
        # the launch-settings parse so a bad stored setting still fails first.
        from vibe_quant.screening.nt_runner import config_fields_for, resolve_screening_params

        sweep_params, overrides = resolve_screening_params(
            parameters,
            lambda: config_fields_for(dsl.model_dump()),
            skip_keys=SCREENING_SKIP_KEYS,
        )
        if sweep_params:
            dsl_dict = dsl.model_dump()
            dsl_dict["sweep"] = sweep_params
            dsl = StrategyDSL.model_validate(dsl_dict)

        pipeline = create_screening_pipeline(
            dsl,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters,
            strategy_overrides=overrides or None,
        )
        result = pipeline.run()
        pipeline.save_results(result, state, args.run_id)

        # Store compiler version for staleness detection
        from vibe_quant.dsl.compiler import compiler_version_hash

        _store_compiler_version(state, args.run_id, compiler_version_hash())

        # Post-promote replay drift check (no-op when not a promote run)
        from vibe_quant.screening.replay_drift import check_replay_drift

        check_replay_drift(state, args.run_id)

        state.update_backtest_run_status(args.run_id, "completed")
        manager.mark_completed(args.run_id)
        print(
            f"Screening complete: {result.total_combinations} combos, {len(result.pareto_optimal_indices)} Pareto-optimal"
        )
        return 0
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        try:
            state.update_backtest_run_status(args.run_id, "failed", error_message=error_msg)
            manager.mark_completed(args.run_id, error=error_msg)
        except Exception as write_exc:
            logger.error(
                "could not record run %s as failed (%s)",
                args.run_id,
                type(write_exc).__name__,
            )
        print(f"Screening failed: {exc}")
        return 1
    finally:
        stop_heartbeat()
        state.close()
        manager.close()


def cmd_list(args: argparse.Namespace) -> int:
    """List screening backtest runs.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.db.state_manager import StateManager

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    state = StateManager(db_path)

    try:
        runs = state.list_backtest_runs(status=args.status)
        screening_runs = [r for r in runs if r.get("run_mode") == "screening"]

        if args.limit:
            screening_runs = screening_runs[: args.limit]

        if not screening_runs:
            print("No screening runs found.")
            return 0

        print(f"{'ID':<6} {'Strategy':<12} {'Status':<12} {'Symbols':<20} {'Created'}")
        print("-" * 70)

        for run in screening_runs:
            run_id = run.get("id", "")
            strategy = run.get("strategy_id", "")
            status = run.get("status", "")
            symbols = run.get("symbols", [])
            symbols_str = ",".join(symbols[:2])
            if len(symbols) > 2:
                symbols_str += f"+{len(symbols) - 2}"
            created_raw = run.get("created_at", "")
            created = str(created_raw)[:16] if created_raw else ""

            print(f"{run_id:<6} {strategy:<12} {status:<12} {symbols_str:<20} {created}")

        return 0
    finally:
        state.close()


def cmd_status(args: argparse.Namespace) -> int:
    """Show status of a specific screening run.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).
    """
    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.jobs.manager import BacktestJobManager

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    state = StateManager(db_path)
    job_mgr = BacktestJobManager(db_path)

    try:
        run_config = state.get_backtest_run(args.run_id)
        if run_config is None:
            print(f"Run {args.run_id} not found")
            return 1

        print(f"Run ID:     {args.run_id}")
        print(f"Strategy:   {run_config.get('strategy_id', '-')}")
        print(f"Mode:       {run_config.get('run_mode', '-')}")
        print(f"Status:     {run_config.get('status', '-')}")
        print(f"Symbols:    {run_config.get('symbols', [])}")
        print(f"Timeframe:  {run_config.get('timeframe', '-')}")
        print(f"Start:      {run_config.get('start_date', '-')}")
        print(f"End:        {run_config.get('end_date', '-')}")
        print(f"Created:    {run_config.get('created_at', '-')}")

        job_info = job_mgr.get_job_info(args.run_id)
        if job_info is not None:
            print(f"\nJob PID:    {job_info.pid}")
            print(f"Job Status: {job_info.status.value}")
            print(f"Heartbeat:  {job_info.heartbeat_at or '-'}")
            print(f"Stale:      {job_info.is_stale}")
            if job_info.log_file:
                print(f"Log File:   {job_info.log_file}")

        error = run_config.get("error_message")
        if error:
            print(f"\nError:      {error}")

        return 0
    finally:
        state.close()
        job_mgr.close()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with run/list/status subcommands.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="vibe-quant screening",
        description="Screening pipeline parameter sweep",
    )
    parser.add_argument("--db", type=str, default=None, help="Database path")

    subparsers = parser.add_subparsers(
        title="commands",
        dest="subcommand",
        help="Available screening commands",
    )

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run screening pipeline")
    run_parser.add_argument("--run-id", type=int, required=True, help="Backtest run ID")
    run_parser.add_argument(
        "--strategy-id", type=int, default=None, help="Strategy ID (uses run's strategy if omitted)"
    )
    run_parser.add_argument("--symbols", type=str, nargs="+", default=None, help="Override symbols")
    run_parser.add_argument("--timeframe", type=str, default=None, help="Override timeframe")
    run_parser.add_argument("--db", type=str, default=None, help="Database path")
    run_parser.set_defaults(func=cmd_run)

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List screening runs")
    list_parser.add_argument("--limit", type=int, default=20, help="Max runs to show (default: 20)")
    list_parser.add_argument("--status", type=str, default=None, help="Filter by status")
    list_parser.add_argument("--db", type=str, default=None, help="Database path")
    list_parser.set_defaults(func=cmd_list)

    # status subcommand
    status_parser = subparsers.add_parser("status", help="Show run status")
    status_parser.add_argument("--run-id", type=int, required=True, help="Backtest run ID")
    status_parser.add_argument("--db", type=str, default=None, help="Database path")
    status_parser.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run screening CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        return 0

    func = getattr(args, "func", None)
    if callable(func):
        return int(func(args))

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
