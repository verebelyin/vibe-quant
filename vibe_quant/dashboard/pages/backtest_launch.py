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

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from vibe_quant.dashboard.components.backtest_config import LATENCY_OPTIONS  # noqa: F401
from vibe_quant.dashboard.components.backtest_config import (
    render_date_range_selector as _component_render_date_range_selector,
)
from vibe_quant.dashboard.components.backtest_config import (
    render_latency_selector as _component_render_latency_selector,
)
from vibe_quant.dashboard.components.backtest_config import (
    render_strategy_selector as _component_render_strategy_selector,
)
from vibe_quant.dashboard.components.backtest_config import (
    render_symbol_timeframe_selector as _component_render_symbol_timeframe_selector,
)
from vibe_quant.dashboard.components.job_status import render_active_jobs, render_recent_runs
from vibe_quant.dashboard.components.overfitting_panel import render_overfitting_panel
from vibe_quant.dashboard.components.preflight_summary import render_preflight_summary
from vibe_quant.dashboard.utils import get_job_manager, get_state_manager
from vibe_quant.db.connection import DEFAULT_DB_PATH

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import JsonDict, StateManager
    from vibe_quant.jobs.manager import BacktestJobManager


def _render_strategy_selector(manager: StateManager) -> JsonDict | None:
    """Render strategy selector with rich summary card."""
    return _component_render_strategy_selector(manager)


def _render_symbol_timeframe_selector(strategy: JsonDict | None) -> tuple[list[str], str]:
    """Render symbol and timeframe selectors."""
    return _component_render_symbol_timeframe_selector(strategy)


def _render_date_range_selector() -> tuple[str, str]:
    """Render date range selector with presets."""
    return _component_render_date_range_selector()


def _render_sweep_params_form(strategy: JsonDict | None) -> dict[str, list[int | float]]:
    """Render parameter sweep configuration form from DSL."""
    st.subheader("Parameter Sweep Configuration")

    if not strategy:
        st.info("Select a strategy to configure sweep parameters")
        return {}

    sweep_config = strategy["dsl_config"].get("sweep", {})
    if not sweep_config:
        st.info("No sweep parameters defined in strategy DSL. Edit strategy to add sweep ranges.")
        return {}

    sweep_values: dict[str, list[int | float]] = {}
    st.markdown("**Sweep Ranges** (from strategy DSL)")

    for param_name, default_values in sweep_config.items():
        c1, c2, c3 = st.columns([2, 3, 1])
        with c1:
            st.write(f"**{param_name}**")
        with c2:
            default_str = ", ".join(str(v) for v in default_values)
            values_str = st.text_input(
                f"Values for {param_name}", value=default_str,
                key=f"sweep_{param_name}", label_visibility="collapsed",
                help="Comma-separated values",
            )
            try:
                parsed = []
                for v in values_str.split(","):
                    v = v.strip()
                    parsed.append(float(v) if "." in v else int(v))
                sweep_values[param_name] = parsed
            except ValueError:
                st.error(f"Invalid values for {param_name}")
                sweep_values[param_name] = list(default_values)
        with c3:
            st.caption(f"{len(sweep_values.get(param_name, default_values))} values")

    if sweep_values:
        total = 1
        for values in sweep_values.values():
            total *= len(values)
        st.info(f"Total parameter combinations: **{total:,}**")

    return sweep_values


def _render_config_selectors(manager: StateManager) -> tuple[int | None, int | None]:
    """Render sizing and risk config selectors."""
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Position Sizing")
        sizing_configs = manager.list_sizing_configs()
        if sizing_configs:
            opts = {c["name"]: c["id"] for c in sizing_configs}
            opts["None (use defaults)"] = None
            sizing_id = opts[st.selectbox("Sizing Config", options=list(opts.keys()), key="sizing_config")]
        else:
            st.info("No sizing configs. Create one in Settings.")
            sizing_id = None

    with c2:
        st.subheader("Risk Management")
        risk_configs = manager.list_risk_configs()
        if risk_configs:
            opts = {c["name"]: c["id"] for c in risk_configs}
            opts["None (use defaults)"] = None
            risk_id = opts[st.selectbox("Risk Config", options=list(opts.keys()), key="risk_config")]
        else:
            st.info("No risk configs. Create one in Settings.")
            risk_id = None

    return sizing_id, risk_id


def _render_latency_selector() -> str | None:
    """Render latency preset selector."""
    return _component_render_latency_selector()


def _render_run_buttons(
    manager: StateManager, job_manager: BacktestJobManager,
    strategy: JsonDict | None, symbols: list[str], timeframe: str,
    start_date: str, end_date: str,
    sweep_params: dict[str, list[int | float]],
    overfitting_filters: dict[str, bool],
    sizing_config_id: int | None, risk_config_id: int | None,
    latency_preset: str | None,
) -> None:
    """Render run buttons and handle job creation."""
    st.divider()
    c1, c2, _c3 = st.columns([1, 1, 2])

    with c1:
        run_screening = st.button(
            "Run Screening", type="primary", key="run_screening",
            disabled=strategy is None or not symbols,
            help="Fast parallel parameter sweep with simplified fills",
        )
    with c2:
        run_validation = st.button(
            "Run Validation", key="run_validation",
            disabled=strategy is None or not symbols or latency_preset is None,
            help="Full fidelity backtest with latency/slippage modeling",
        )

    if not (run_screening or run_validation):
        return
    if not strategy:
        st.error("Please select a strategy")
        return
    if not symbols:
        st.error("Please select at least one symbol")
        return

    run_mode = "screening" if run_screening else "validation"
    parameters = {"sweep": sweep_params, "overfitting_filters": overfitting_filters}

    run_id = manager.create_backtest_run(
        strategy_id=strategy["id"], run_mode=run_mode,
        symbols=symbols, timeframe=timeframe,
        start_date=start_date, end_date=end_date,
        parameters=parameters,
        sizing_config_id=sizing_config_id, risk_config_id=risk_config_id,
        latency_preset=latency_preset,
    )

    db_path = st.session_state.get("db_path", str(DEFAULT_DB_PATH))
    command = [
        sys.executable, "-m", f"vibe_quant.{'screening' if run_screening else 'validation'}",
        "--run-id", str(run_id), "--db", db_path,
    ]
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    try:
        pid = job_manager.start_job(
            run_id=run_id, job_type=run_mode,
            command=command, log_file=str(log_dir / f"backtest_{run_id}.log"),
        )
        st.success(f"Started {run_mode} job (run_id={run_id}, pid={pid}). See job list below.")
        st.rerun()
    except ValueError as e:
        st.error(f"Failed to start job: {e}")
    except Exception as e:
        st.error(f"Error starting job: {e}")
        manager.update_backtest_run_status(run_id, "failed", error_message=str(e))


# ── Main Entry Point ──────────────────────────────────────────────────────


def render_backtest_launch_tab() -> None:
    """Render the complete backtest launch tab."""
    st.title("Backtest Launch")
    manager = get_state_manager()
    job_manager = get_job_manager()

    strategy = _render_strategy_selector(manager)
    st.divider()

    symbols, timeframe = _render_symbol_timeframe_selector(strategy)
    start_date, end_date = _render_date_range_selector()
    st.divider()

    sweep_params = _render_sweep_params_form(strategy)
    st.divider()

    overfitting_filters = render_overfitting_panel(strategy, start_date, end_date)
    st.divider()

    sizing_id, risk_id = _render_config_selectors(manager)
    st.divider()

    latency_preset = _render_latency_selector()

    if strategy and symbols:
        render_preflight_summary(
            strategy, symbols, timeframe, start_date, end_date,
            sweep_params, overfitting_filters, latency_preset,
        )

    _render_run_buttons(
        manager=manager, job_manager=job_manager,
        strategy=strategy, symbols=symbols, timeframe=timeframe,
        start_date=start_date, end_date=end_date,
        sweep_params=sweep_params, overfitting_filters=overfitting_filters,
        sizing_config_id=sizing_id, risk_config_id=risk_id,
        latency_preset=latency_preset,
    )

    st.divider()
    render_active_jobs(manager, job_manager)
    render_recent_runs(manager)


# Top-level call for st.navigation API
render_backtest_launch_tab()
