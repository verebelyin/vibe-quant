"""Shadow tracker CLI: ``python -m vibe_quant.shadow update [--db PATH] [--job-id N]``.

Runs one update of every live shadow row on the state DB, then sends pending alerts. It
downloads nothing (``python -m vibe_quant.data update`` does, and runs the same update at
its end). Exit codes: 0 done, 2 another update of this DB is running (skipped), 1 error.
With ``--job-id`` the run heartbeats and records its outcome on that background job; a
skip completes the job with the text ``skipped: shadow update already running``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from vibe_quant.db.connection import DEFAULT_DB_PATH
from vibe_quant.db.state_manager import ShadowUpdateRunning, StateManager
from vibe_quant.shadow import notify, tracker

SKIPPED_JOB_TEXT = "skipped: shadow update already running"


def _update(db: Path) -> int:
    if not db.exists():  # a mistyped --db must not create a fresh, empty state DB
        print(f"ERROR: state DB {db} does not exist", file=sys.stderr)
        return 1
    state = StateManager(db)
    try:
        try:
            summary = tracker.update_all(
                state, today=datetime.now(UTC).date(), trigger="cli", failed_streams={}
            )
        except ShadowUpdateRunning as exc:
            print(str(exc), file=sys.stderr)
            return 2
        for sid, row_state, note in summary.rows:
            print(f"shadow {sid}: {row_state} ({note})")
        print(f"shadow update: {len(summary.rows)} row(s), {summary.runs} engine run(s)")
        notify.send_pending(state)
        return 0
    finally:
        state.close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the exit code."""
    from vibe_quant.jobs.manager import JobStatus, run_with_heartbeat

    parser = argparse.ArgumentParser(prog="python -m vibe_quant.shadow")
    sub = parser.add_subparsers(dest="command", required=True)
    update = sub.add_parser("update", help="Advance every live shadow row (no download)")
    update.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="State database")
    update.add_argument("--job-id", type=int, default=None, help="background_jobs.run_id")
    args = parser.parse_args(argv)

    if args.job_id is None:
        return _update(args.db)
    manager, stop_heartbeat = run_with_heartbeat(args.job_id, args.db)
    try:
        try:
            rc = _update(args.db)
        except Exception as exc:
            manager.mark_completed(args.job_id, error=f"{type(exc).__name__}: {exc}")
            raise
        if rc == 2:  # a skip is not a failure; the text is the job's only message column
            manager._finish(args.job_id, JobStatus.COMPLETED, SKIPPED_JOB_TEXT)
        else:
            manager.mark_completed(args.job_id)
        return rc
    finally:
        stop_heartbeat()
        manager.close()


if __name__ == "__main__":
    sys.exit(main())
