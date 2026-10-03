"""Run a ``vibe_quant.data`` CLI command as a tracked background job.

Usage::

    python -m vibe_quant.jobs.data_job --run-id -123 [--db PATH] -- ingest --symbols BTCUSDT ...

Data jobs (ingest/update/rebuild) used to be spawned as the bare data CLI: they
never heartbeat, so after 120 s the stale-job cleanup (triggered by the
Backtest page) SIGKILLed every download, and they never recorded completion.
This wrapper heartbeats for the job's whole lifetime and records the outcome.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from vibe_quant.jobs.manager import run_with_heartbeat


def _run_data_cli(data_args: list[str]) -> int:
    from vibe_quant.data.ingest import main as data_main

    try:
        return int(data_main(data_args) or 0)
    except SystemExit as exc:  # argparse errors / --help
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the data command's exit code."""
    parser = argparse.ArgumentParser(prog="python -m vibe_quant.jobs.data_job")
    parser.add_argument("--run-id", type=int, required=True, help="background_jobs.run_id")
    parser.add_argument("--db", type=str, default=None, help="State database path")
    parser.add_argument("data_args", nargs=argparse.REMAINDER, help="-- <data CLI args>")
    args = parser.parse_args(argv)
    data_args = list(args.data_args)
    if data_args[:1] == ["--"]:
        data_args = data_args[1:]
    if not data_args:
        parser.error("missing data command after --")

    db_path = Path(args.db) if args.db else None
    manager, stop_heartbeat = run_with_heartbeat(args.run_id, db_path)
    try:
        try:
            rc = _run_data_cli(data_args)
        except Exception as exc:
            traceback.print_exc()
            manager.mark_completed(args.run_id, error=f"{type(exc).__name__}: {exc}")
            return 1
        if rc == 0:
            manager.mark_completed(args.run_id)
        else:
            manager.mark_completed(args.run_id, error=f"data command exited with code {rc}")
        return rc
    finally:
        stop_heartbeat()
        manager.close()


if __name__ == "__main__":
    sys.exit(main())
