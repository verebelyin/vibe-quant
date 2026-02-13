"""Job status components for backtest launch page.

Provides:
- Active job list with auto-refresh, status indicators, kill button
- Recent runs list with status icons
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from vibe_quant.jobs.manager import JobStatus

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.jobs.manager import BacktestJobManager


@st.fragment(run_every=5)
def render_active_jobs(manager: StateManager, job_manager: BacktestJobManager) -> None:
    """Render active jobs list with status and kill button.

    Auto-refreshes every 5 seconds.
    """
    st.subheader("Active Jobs")

    # Sync job status with actual process state
    for job in job_manager.list_active_jobs():
        job_manager.sync_job_status(job.run_id)

    active_jobs = job_manager.list_active_jobs()

    # Stale jobs warning
    stale_jobs = job_manager.list_stale_jobs()
    if stale_jobs:
        st.warning(f"Found {len(stale_jobs)} stale jobs (no heartbeat >120s)")
        if st.button("Cleanup Stale Jobs"):
            cleaned = job_manager.cleanup_stale_jobs()
            st.info(f"Cleaned up {cleaned} stale jobs")
            st.rerun()

    if not active_jobs:
        st.info("No active jobs. Launch a backtest above.")
        return

    for job in active_jobs:
        run = manager.get_backtest_run(job.run_id)
        if not run:
            continue

        strategy = manager.get_strategy(run["strategy_id"])
        strategy_name = strategy["name"] if strategy else f"Strategy {run['strategy_id']}"

        with st.container():
            c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])

            with c1:
                st.write(f"**Run #{job.run_id}**")
                st.caption(f"{strategy_name} | {job.job_type}")

            with c2:
                symbols_str = ", ".join(run["symbols"][:3])
                if len(run["symbols"]) > 3:
                    symbols_str += f" +{len(run['symbols']) - 3}"
                st.write(f"Symbols: {symbols_str}")
                st.caption(f"{run['start_date']} to {run['end_date']}")

            with c3:
                status_colors = {
                    JobStatus.RUNNING: "green",
                    JobStatus.PENDING: "gray",
                    JobStatus.COMPLETED: "blue",
                    JobStatus.FAILED: "red",
                    JobStatus.KILLED: "orange",
                }
                color = status_colors.get(job.status, "gray")
                st.markdown(f":{color}[{job.status.value.upper()}]")

            with c4:
                if job.is_stale:
                    st.warning("STALE")
                elif job.heartbeat_at:
                    st.caption(f"HB: {job.heartbeat_at.strftime('%H:%M:%S')}")

            with c5:
                if st.button("Kill", key=f"kill_{job.run_id}"):
                    if job_manager.kill_job(job.run_id):
                        st.success(f"Killed job {job.run_id}")
                        st.rerun()
                    else:
                        st.error("Failed to kill job")

            st.divider()

    if st.button("Refresh Jobs"):
        st.rerun()


def _sync_recent_running_runs(
    manager: StateManager, job_manager: BacktestJobManager, limit: int = 10,
) -> list[dict[str, object]]:
    """Synchronize stale running statuses before rendering recent runs."""
    runs = manager.list_backtest_runs()[:limit]
    for run in runs:
        if run.get("status") == "running":
            job_manager.sync_job_status(run["id"])
    return manager.list_backtest_runs()[:limit]


def render_recent_runs(manager: StateManager, job_manager: BacktestJobManager) -> None:
    """Render recent backtest runs."""
    with st.expander("Recent Runs", expanded=False):
        runs = _sync_recent_running_runs(manager, job_manager, limit=10)

        if not runs:
            st.info("No backtest runs yet")
            return

        for run in runs:
            strategy_id_raw = run.get("strategy_id")
            if isinstance(strategy_id_raw, (int, str, bytes, bytearray)):
                try:
                    strategy_id = int(strategy_id_raw)
                except ValueError:
                    strategy_id = -1
            else:
                strategy_id = -1

            strategy = manager.get_strategy(strategy_id) if strategy_id >= 0 else None
            strategy_name = strategy["name"] if strategy else f"ID {strategy_id_raw}"

            status = str(run.get("status", ""))
            status_icon = {
                "pending": "clock1",
                "running": "hourglass_flowing_sand",
                "completed": "white_check_mark",
                "failed": "x",
                "killed": "octagonal_sign",
            }.get(status, "question")

            symbols_raw = run.get("symbols")
            symbols = (
                ", ".join(str(s) for s in symbols_raw)
                if isinstance(symbols_raw, list)
                else str(symbols_raw)
            )

            st.write(
                f":{status_icon}: **Run #{run['id']}** - {strategy_name} "
                f"({run['run_mode']}) - {status}"
            )
            st.caption(
                f"Symbols: {symbols} | "
                f"{run['start_date']} to {run['end_date']} | "
                f"Created: {run['created_at']}"
            )
