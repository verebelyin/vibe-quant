"""Background job management for backtests."""

from vibe_quant.jobs.manager import (
    HEARTBEAT_INTERVAL_SECONDS,
    STALE_THRESHOLD_SECONDS,
    BacktestJobManager,
    JobInfo,
    JobStatus,
    run_with_heartbeat,
)

__all__ = [
    "BacktestJobManager",
    "JobInfo",
    "JobStatus",
    "HEARTBEAT_INTERVAL_SECONDS",
    "STALE_THRESHOLD_SECONDS",
    "run_with_heartbeat",
]
