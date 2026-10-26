"""Discovery pipeline CLI entrypoint."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import vibe_quant.discovery.run as discovery_run
from vibe_quant.discovery.run_spec import build_parser, check_launch_leverage

logger = logging.getLogger(__name__)


def main() -> int:
    """Run discovery pipeline and persist summary metrics for the run."""
    parser = build_parser()
    args = parser.parse_args()
    check_launch_leverage(parser, args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from vibe_quant.db.connection import DEFAULT_DB_PATH
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.jobs.manager import run_with_heartbeat

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    state = StateManager(db_path)
    job_manager, stop_heartbeat = run_with_heartbeat(args.run_id, db_path)
    started_at = time.perf_counter()
    try:
        backtests = discovery_run.MockBacktests() if args.mock else discovery_run.NTBacktests()
        store = discovery_run.StateRecordStore(state, job_manager)
        outcome = discovery_run.run_discovery(
            args, backtests, store, run_id=args.run_id, started_at=started_at
        )
        return outcome.exit_code
    finally:
        stop_heartbeat()
        state.close()
        job_manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
