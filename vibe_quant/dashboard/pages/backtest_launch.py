"""Backtest Launch Tab for vibe-quant dashboard.

Provides interface for launching backtests with:
- Strategy/symbol/timeframe selectors
- Parameter sweep range config (auto-form from DSL)
- Overfitting filter toggles
- Sizing/risk module selectors
- Latency preset selector
- Run Screening/Validation buttons
- Active jobs list with status, progress, Kill button
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from vibe_quant.dashboard.utils import get_job_manager, get_state_manager
from vibe_quant.dsl.schema import VALID_TIMEFRAMES
from vibe_quant.jobs.manager import JobStatus
from vibe_quant.validation.latency import LATENCY_PRESETS, LatencyPreset

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import JsonDict

# Common crypto perpetual symbols
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
]

# Latency options: presets + custom entry
LATENCY_OPTIONS = ["None (screening mode)"] + [p.value for p in LatencyPreset] + ["custom"]


def _render_strategy_selector(manager: StateManager) -> JsonDict | None:
    """Render strategy selector with rich summary card."""
    strategies = manager.list_strategies(active_only=True)

    if not strategies:
        st.warning("No active strategies found. Create one in Strategy Management tab.")
        return None

    strategy_options = {s["name"]: s for s in strategies}
    selected_name = st.selectbox(
        "Select Strategy",
        options=list(strategy_options.keys()),
        key="strategy_select",
        help="Choose a strategy to backtest",
    )

    if selected_name:
        strategy = strategy_options[selected_name]
        dsl = strategy["dsl_config"]
        indicators = dsl.get("indicators", {})
        entry = dsl.get("entry_conditions", {})
        sweep = dsl.get("sweep", {})

        # Strategy summary card
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**{selected_name}** v{strategy.get('version', 1)}")
                st.caption(strategy.get("description") or "No description")
            with col2:
                st.metric("Timeframe", dsl.get("timeframe", "N/A"))
            with col3:
                n_ind = len(indicators)
                n_long = len(entry.get("long", []))
                n_short = len(entry.get("short", []))
                st.metric("Indicators", n_ind)
                st.caption(f"{n_long}L / {n_short}S entries")
            with col4:
                n_sweep = len(sweep)
                combos = 1
                for v in sweep.values():
                    combos *= len(v)
                st.metric("Sweep Params", n_sweep)
                if n_sweep:
                    st.caption(f"{combos:,} combinations")

        return strategy

    return None


def _render_symbol_timeframe_selector(strategy: JsonDict | None) -> tuple[list[str], str]:
    """Render symbol and timeframe selectors."""
    col1, col2 = st.columns([2, 1])

    with col1:
        symbols = st.multiselect(
            "Symbols",
            options=DEFAULT_SYMBOLS,
            default=["BTCUSDT", "ETHUSDT"],
            key="symbols_select",
            help="Select one or more perpetual futures symbols",
        )

        custom_symbol = st.text_input(
            "Add custom symbol",
            placeholder="e.g., ARBUSDT",
            key="custom_symbol",
        )
        if (
            custom_symbol
            and custom_symbol not in symbols
            and st.button("Add Symbol", key="add_symbol")
        ):
            symbols.append(custom_symbol.upper())

    with col2:
        # Use strategy's timeframe as default if available
        default_tf = "1h"
        if strategy:
            default_tf = strategy["dsl_config"].get("timeframe", "1h")

        tf_list = sorted(VALID_TIMEFRAMES)
        default_idx = tf_list.index(default_tf) if default_tf in tf_list else 0

        timeframe = st.selectbox(
            "Timeframe",
            options=tf_list,
            index=default_idx,
            key="timeframe_select",
        )

    return symbols, timeframe


def _render_date_range_selector() -> tuple[str, str]:
    """Render date range selector with presets."""
    default_end = date.today()

    # Date presets
    st.caption("**Quick presets:**")
    preset_cols = st.columns(5)
    presets = [
        ("30 days", 30),
        ("90 days", 90),
        ("6 months", 180),
        ("1 year", 365),
        ("Full history", None),
    ]
    for col, (label, days) in zip(preset_cols, presets, strict=False):
        with col:
            if st.button(label, key=f"date_preset_{label}", use_container_width=True):
                if days is not None:
                    st.session_state["start_date"] = default_end - timedelta(days=days)
                else:
                    st.session_state["start_date"] = date(2019, 1, 1)
                st.session_state["end_date"] = default_end
                st.rerun()

    col1, col2, col3 = st.columns([2, 2, 1])

    default_start = default_end - timedelta(days=365)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.get("start_date", default_start),
            min_value=date(2019, 1, 1),
            max_value=default_end,
            key="start_date",
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=st.session_state.get("end_date", default_end),
            min_value=start_date,
            max_value=date.today(),
            key="end_date",
        )

    with col3:
        duration = (end_date - start_date).days
        st.metric("Duration", f"{duration} days")

    return start_date.isoformat(), end_date.isoformat()


def _render_sweep_params_form(strategy: JsonDict | None) -> dict[str, list[int | float]]:
    """Render parameter sweep configuration form from DSL.

    Auto-generates form fields based on strategy's sweep config.
    """
    st.subheader("Parameter Sweep Configuration")

    if not strategy:
        st.info("Select a strategy to configure sweep parameters")
        return {}

    dsl = strategy["dsl_config"]
    sweep_config = dsl.get("sweep", {})

    if not sweep_config:
        st.info("No sweep parameters defined in strategy DSL. Edit strategy to add sweep ranges.")
        return {}

    sweep_values: dict[str, list[int | float]] = {}

    st.markdown("**Sweep Ranges** (from strategy DSL)")

    for param_name, default_values in sweep_config.items():
        col1, col2, col3 = st.columns([2, 3, 1])

        with col1:
            st.write(f"**{param_name}**")

        with col2:
            # Allow editing the sweep values
            default_str = ", ".join(str(v) for v in default_values)
            values_str = st.text_input(
                f"Values for {param_name}",
                value=default_str,
                key=f"sweep_{param_name}",
                label_visibility="collapsed",
                help="Comma-separated values",
            )

            try:
                # Parse values (handle both int and float)
                parsed_values = []
                for v in values_str.split(","):
                    v = v.strip()
                    if "." in v:
                        parsed_values.append(float(v))
                    else:
                        parsed_values.append(int(v))
                sweep_values[param_name] = parsed_values
            except ValueError:
                st.error(f"Invalid values for {param_name}")
                sweep_values[param_name] = list(default_values)

        with col3:
            st.caption(f"{len(sweep_values.get(param_name, default_values))} values")

    # Show total combinations
    if sweep_values:
        total = 1
        for values in sweep_values.values():
            total *= len(values)
        st.info(f"Total parameter combinations: **{total:,}**")

    return sweep_values


def _render_overfitting_filter_toggles(
    strategy: JsonDict | None,
    start_date: str,
    end_date: str,
) -> dict[str, bool]:
    """Render overfitting filter toggles with risk indicator."""
    st.subheader("Overfitting Protection")

    # Overfitting risk indicator
    if strategy:
        dsl = strategy["dsl_config"]
        sweep = dsl.get("sweep", {})
        combos = 1
        for v in sweep.values():
            combos *= len(v)
        n_conditions = (
            len(dsl.get("entry_conditions", {}).get("long", []))
            + len(dsl.get("entry_conditions", {}).get("short", []))
        )

        # Calculate risk level
        if combos > 500 or n_conditions > 6:
            risk_level = "High"
            risk_color = "red"
        elif combos > 50 or n_conditions > 3:
            risk_level = "Medium"
            risk_color = "orange"
        else:
            risk_level = "Low"
            risk_color = "green"

        col_risk, col_detail = st.columns([1, 3])
        with col_risk:
            st.markdown(f"**Overfitting Risk:** :{risk_color}[**{risk_level}**]")
        with col_detail:
            st.caption(
                f"{combos:,} parameter combinations | "
                f"{n_conditions} entry conditions | "
                f"Data: {start_date} to {end_date}"
            )
            if combos > 20:
                st.caption(
                    f"With {combos} comparisons, there is a "
                    f"~{min(99, int((1 - 0.95**combos) * 100))}% chance of finding "
                    f"a spuriously profitable combination by chance alone."
                )

    col1, col2, col3 = st.columns(3)

    with col1:
        enable_dsr = st.checkbox(
            "Deflated Sharpe Ratio (DSR)",
            value=True,
            key="filter_dsr",
            help="Tests statistical significance of Sharpe ratio "
                 "accounting for multiple comparisons. Adjusts the Sharpe "
                 "ratio downward based on how many strategies were tested.",
        )

    with col2:
        enable_wfa = st.checkbox(
            "Walk-Forward Analysis (WFA)",
            value=True,
            key="filter_wfa",
            help="Splits data into in-sample (optimization) and out-of-sample "
                 "(validation) periods. Tests if optimized parameters work on "
                 "unseen data. The gold standard for overfitting detection.",
        )

    with col3:
        enable_pkfold = st.checkbox(
            "Purged K-Fold CV",
            value=True,
            key="filter_pkfold",
            help="Cross-validation with purging to prevent data leakage "
                 "between train and test folds. More robust than simple "
                 "train/test split for time series data.",
        )

    # Educational context
    with st.expander("Why overfitting protection matters", expanded=False):
        st.markdown(
            "Research shows a **100% average performance gap** between in-sample "
            "and out-of-sample results for trading strategies. Without these filters:\n"
            "- Strategies that look profitable in backtests often fail in live trading\n"
            "- Testing many parameter combinations guarantees finding false positives\n"
            "- The more you optimize, the worse real performance typically gets\n\n"
            "**Recommendation:** Keep all three filters enabled. They work together "
            "to catch different types of overfitting."
        )

    return {
        "enable_dsr": enable_dsr,
        "enable_wfa": enable_wfa,
        "enable_purged_kfold": enable_pkfold,
    }


def _render_config_selectors(manager: StateManager) -> tuple[int | None, int | None]:
    """Render sizing and risk config selectors."""
    col1, col2 = st.columns(2)

    # Sizing config
    with col1:
        st.subheader("Position Sizing")
        sizing_configs = manager.list_sizing_configs()

        if sizing_configs:
            sizing_options = {c["name"]: c["id"] for c in sizing_configs}
            sizing_options["None (use defaults)"] = None

            selected_sizing = st.selectbox(
                "Sizing Config",
                options=list(sizing_options.keys()),
                key="sizing_config",
            )
            sizing_id = sizing_options[selected_sizing]
        else:
            st.info("No sizing configs. Create one in Settings.")
            sizing_id = None

    # Risk config
    with col2:
        st.subheader("Risk Management")
        risk_configs = manager.list_risk_configs()

        if risk_configs:
            risk_options = {c["name"]: c["id"] for c in risk_configs}
            risk_options["None (use defaults)"] = None

            selected_risk = st.selectbox(
                "Risk Config",
                options=list(risk_options.keys()),
                key="risk_config",
            )
            risk_id = risk_options[selected_risk]
        else:
            st.info("No risk configs. Create one in Settings.")
            risk_id = None

    return sizing_id, risk_id


def _render_latency_selector() -> str | None:
    """Render latency preset selector."""
    st.subheader("Latency Model")

    col1, col2 = st.columns([2, 1])

    with col1:
        selected = st.selectbox(
            "Latency Preset",
            options=LATENCY_OPTIONS,
            index=0,
            key="latency_preset",
            help="Select latency model. Use None for screening (fast), presets for validation, or custom.",
        )

    with col2:
        if selected == "custom":
            base_ms = st.number_input(
                "Base latency (ms)", min_value=0, value=50, step=1, key="custom_base_ms"
            )
            insert_ms = st.number_input(
                "Insert latency (ms)", min_value=0, value=25, step=1, key="custom_insert_ms"
            )
            st.metric("Total Insert Latency", f"{base_ms + insert_ms} ms")
            st.session_state["custom_latency"] = {
                "base_latency_nanos": base_ms * 1_000_000,
                "insert_latency_nanos": insert_ms * 1_000_000,
            }
        elif selected != "None (screening mode)":
            preset = LatencyPreset(selected)
            values = LATENCY_PRESETS[preset]
            st.metric(
                "Total Insert Latency",
                f"{values.base_ms + values.insert_ms} ms",
            )
        else:
            st.metric("Total Insert Latency", "0 ms (screening)")

    return None if selected == "None (screening mode)" else selected


def _render_run_buttons(
    manager: StateManager,
    job_manager: BacktestJobManager,
    strategy: JsonDict | None,
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    sweep_params: dict[str, list[int | float]],
    overfitting_filters: dict[str, bool],
    sizing_config_id: int | None,
    risk_config_id: int | None,
    latency_preset: str | None,
) -> None:
    """Render run buttons and handle job creation."""
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        run_screening = st.button(
            "Run Screening",
            type="primary",
            key="run_screening",
            disabled=strategy is None or not symbols,
            help="Fast parallel parameter sweep with simplified fills",
        )

    with col2:
        run_validation = st.button(
            "Run Validation",
            key="run_validation",
            disabled=strategy is None or not symbols or latency_preset is None,
            help="Full fidelity backtest with latency/slippage modeling",
        )

    # Handle run buttons
    if run_screening or run_validation:
        if not strategy:
            st.error("Please select a strategy")
            return
        if not symbols:
            st.error("Please select at least one symbol")
            return

        run_mode = "screening" if run_screening else "validation"

        # Build parameters dict
        parameters = {
            "sweep": sweep_params,
            "overfitting_filters": overfitting_filters,
        }

        # Create backtest run record
        run_id = manager.create_backtest_run(
            strategy_id=strategy["id"],
            run_mode=run_mode,
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters,
            sizing_config_id=sizing_config_id,
            risk_config_id=risk_config_id,
            latency_preset=latency_preset,
        )

        # Build command for subprocess
        db_path = st.session_state.get("db_path", str(DEFAULT_DB_PATH))
        command = [
            "python",
            "-m",
            f"vibe_quant.{'screening' if run_screening else 'validation'}",
            "--run-id",
            str(run_id),
            "--db",
            db_path,
        ]

        # Log file
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / f"backtest_{run_id}.log")

        try:
            # Start job
            pid = job_manager.start_job(
                run_id=run_id,
                job_type=run_mode,
                command=command,
                log_file=log_file,
            )

            st.success(
                f"Started {run_mode} job (run_id={run_id}, pid={pid}). "
                f"See job list below for progress."
            )
            st.rerun()

        except ValueError as e:
            st.error(f"Failed to start job: {e}")
        except Exception as e:
            st.error(f"Error starting job: {e}")
            # Update run status to failed
            manager.update_backtest_run_status(run_id, "failed", error_message=str(e))


@st.fragment(run_every=5)
def _render_active_jobs(manager: StateManager, job_manager: BacktestJobManager) -> None:
    """Render active jobs list with status and kill button.

    Auto-refreshes every 5 seconds to balance responsiveness and server load.
    """
    st.subheader("Active Jobs")

    # Sync job status with actual process state before listing.
    # This detects processes that exited without updating the DB
    # (e.g. crashes, missing status updates) and marks them accordingly.
    for job in job_manager.list_active_jobs():
        job_manager.sync_job_status(job.run_id)

    # Now get the (potentially updated) active jobs list
    active_jobs = job_manager.list_active_jobs()

    # Also check for stale jobs
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

    # Render job table
    for job in active_jobs:
        run = manager.get_backtest_run(job.run_id)
        if not run:
            continue

        strategy = manager.get_strategy(run["strategy_id"])
        strategy_name = strategy["name"] if strategy else f"Strategy {run['strategy_id']}"

        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])

            with col1:
                st.write(f"**Run #{job.run_id}**")
                st.caption(f"{strategy_name} | {job.job_type}")

            with col2:
                symbols_str = ", ".join(run["symbols"][:3])
                if len(run["symbols"]) > 3:
                    symbols_str += f" +{len(run['symbols']) - 3}"
                st.write(f"Symbols: {symbols_str}")
                st.caption(f"{run['start_date']} to {run['end_date']}")

            with col3:
                # Status indicator
                status_colors = {
                    JobStatus.RUNNING: "green",
                    JobStatus.PENDING: "gray",
                    JobStatus.COMPLETED: "blue",
                    JobStatus.FAILED: "red",
                    JobStatus.KILLED: "orange",
                }
                color = status_colors.get(job.status, "gray")
                st.markdown(f":{color}[{job.status.value.upper()}]")

            with col4:
                # Stale indicator
                if job.is_stale:
                    st.warning("STALE")
                else:
                    if job.heartbeat_at:
                        st.caption(f"HB: {job.heartbeat_at.strftime('%H:%M:%S')}")

            with col5:
                if st.button("Kill", key=f"kill_{job.run_id}"):
                    if job_manager.kill_job(job.run_id):
                        st.success(f"Killed job {job.run_id}")
                        st.rerun()
                    else:
                        st.error("Failed to kill job")

            st.divider()

    # Refresh button
    if st.button("Refresh Jobs"):
        st.rerun()


def _render_recent_runs(manager: StateManager) -> None:
    """Render recent backtest runs."""
    with st.expander("Recent Runs", expanded=False):
        runs = manager.list_backtest_runs()[:10]

        if not runs:
            st.info("No backtest runs yet")
            return

        for run in runs:
            strategy = manager.get_strategy(run["strategy_id"])
            strategy_name = strategy["name"] if strategy else f"ID {run['strategy_id']}"

            status = run["status"]
            status_icon = {
                "pending": "clock1",
                "running": "hourglass_flowing_sand",
                "completed": "white_check_mark",
                "failed": "x",
                "killed": "octagonal_sign",
            }.get(status, "question")

            st.write(
                f":{status_icon}: **Run #{run['id']}** - {strategy_name} "
                f"({run['run_mode']}) - {status}"
            )
            st.caption(
                f"Symbols: {', '.join(run['symbols'])} | "
                f"{run['start_date']} to {run['end_date']} | "
                f"Created: {run['created_at']}"
            )


def _render_preflight_summary(
    strategy: JsonDict,
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    sweep_params: dict[str, list[int | float]],
    overfitting_filters: dict[str, bool],
    latency_preset: str | None,
) -> None:
    """Render pre-flight summary before launching backtest."""
    st.divider()
    st.subheader("Pre-flight Summary")

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"**Strategy:** {strategy['name']}")
            st.caption(f"Timeframe: {timeframe}")

        with col2:
            st.markdown(f"**Symbols:** {len(symbols)}")
            sym_display = ", ".join(symbols[:4])
            if len(symbols) > 4:
                sym_display += f" +{len(symbols) - 4}"
            st.caption(sym_display)

        with col3:
            st.markdown(f"**Period:** {start_date} to {end_date}")
            mode = "Screening" if latency_preset is None else "Validation"
            st.caption(f"Mode: {mode}")

        with col4:
            total_combos = 1
            for v in sweep_params.values():
                total_combos *= len(v)
            total_combos = max(1, total_combos)
            total_runs = total_combos * len(symbols)
            st.markdown(f"**Total backtests:** {total_runs:,}")
            filters_on = sum(1 for v in overfitting_filters.values() if v)
            st.caption(f"{filters_on}/3 overfitting filters active")


def render_backtest_launch_tab() -> None:
    """Render the complete backtest launch tab."""
    st.title("Backtest Launch")

    manager = get_state_manager()
    job_manager = get_job_manager()

    # Strategy selector
    strategy = _render_strategy_selector(manager)

    st.divider()

    # Symbol/timeframe selector
    symbols, timeframe = _render_symbol_timeframe_selector(strategy)

    # Date range
    start_date, end_date = _render_date_range_selector()

    st.divider()

    # Parameter sweep config
    sweep_params = _render_sweep_params_form(strategy)

    st.divider()

    # Overfitting filters
    overfitting_filters = _render_overfitting_filter_toggles(strategy, start_date, end_date)

    st.divider()

    # Sizing and risk configs
    sizing_id, risk_id = _render_config_selectors(manager)

    st.divider()

    # Latency preset
    latency_preset = _render_latency_selector()

    # Pre-flight summary
    if strategy and symbols:
        _render_preflight_summary(strategy, symbols, timeframe, start_date, end_date,
                                   sweep_params, overfitting_filters, latency_preset)

    # Run buttons
    _render_run_buttons(
        manager=manager,
        job_manager=job_manager,
        strategy=strategy,
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        sweep_params=sweep_params,
        overfitting_filters=overfitting_filters,
        sizing_config_id=sizing_id,
        risk_config_id=risk_id,
        latency_preset=latency_preset,
    )

    st.divider()

    # Active jobs list
    _render_active_jobs(manager, job_manager)

    # Recent runs
    _render_recent_runs(manager)


# Top-level call for st.navigation API
render_backtest_launch_tab()
