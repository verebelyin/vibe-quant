"""Validation runner CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Run validation backtest from CLI."""
    parser = argparse.ArgumentParser(description="Run validation backtest")
    parser.add_argument("--run-id", type=int, required=True, help="Backtest run ID")
    parser.add_argument("--db", type=str, default=None, help="Database path")
    args = parser.parse_args()

    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.jobs.manager import run_with_heartbeat
    from vibe_quant.validation.runner import ValidationRunner

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    job_manager = run_with_heartbeat(args.run_id, db_path)

    runner = ValidationRunner(db_path=db_path)
    try:
        result = runner.run(args.run_id)
        job_manager.mark_completed(args.run_id)
        print(f"Validation complete: Sharpe={result.sharpe_ratio:.2f}, Return={result.total_return * 100:.2f}%", file=sys.stderr)
        return 0
    except Exception as exc:
        import contextlib

        error_msg = f"{type(exc).__name__}: {exc}"
        with contextlib.suppress(Exception):
            job_manager.mark_completed(args.run_id, error=error_msg)
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        runner.close()
        job_manager.close()


if __name__ == "__main__":
    sys.exit(main())
