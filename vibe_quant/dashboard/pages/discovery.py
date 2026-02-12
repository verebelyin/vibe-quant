"""Discovery Tab for vibe-quant dashboard.

Genetic strategy discovery interface:
- Configuration: population, generations, mutation, indicators, symbols, timeframe
- Live generation-by-generation progress visualization
- Fitness evolution chart (best/mean/worst per generation)
- Results: top discovered strategies with metrics, DSL export
- Background job tracking via BacktestJobManager
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
import streamlit as st
import yaml

from vibe_quant.dashboard.utils import get_job_manager, get_state_manager
from vibe_quant.db.connection import DEFAULT_DB_PATH
from vibe_quant.discovery.genome import INDICATOR_POOL
from vibe_quant.discovery.pipeline import (
    DiscoveryConfig,
    DiscoveryResult,
    GenerationResult,
)
from vibe_quant.dsl.schema import VALID_TIMEFRAMES
from vibe_quant.jobs.manager import BacktestJobManager, JobStatus

if TYPE_CHECKING:
    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.discovery.operators import StrategyChromosome

# Symbols available for discovery
DISCOVERY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def build_fitness_chart_data(
    generations: list[GenerationResult],
) -> dict[str, list[float]]:
    """Extract fitness evolution data from generation results.

    Args:
        generations: List of GenerationResult from pipeline.

    Returns:
        Dict with keys 'generation', 'best', 'mean', 'worst' as parallel lists.
    """
    data: dict[str, list[float]] = {
        "generation": [],
        "best": [],
        "mean": [],
        "worst": [],
    }
    for g in generations:
        data["generation"].append(float(g.generation))
        data["best"].append(g.best_fitness)
        data["mean"].append(g.mean_fitness)
        data["worst"].append(g.worst_fitness)
    return data


def build_results_table(
    top_strategies: list[tuple[StrategyChromosome, FitnessResult]],
) -> list[dict[str, Any]]:
    """Format top strategies for table display.

    Args:
        top_strategies: Ranked (chromosome, fitness) pairs.

    Returns:
        List of row dicts with Rank, Sharpe, MaxDD, PF, Trades, Genes, Score.
    """
    rows: list[dict[str, Any]] = []
    for rank, (chrom, fr) in enumerate(top_strategies, 1):
        gene_count = len(chrom.entry_genes) + len(chrom.exit_genes)
        rows.append({
            "Rank": rank,
            "Sharpe": round(fr.sharpe_ratio, 4),
            "MaxDD": round(fr.max_drawdown, 4),
            "PF": round(fr.profit_factor, 4),
            "Total Trades": fr.total_trades,
            "Genes": gene_count,
            "Score": round(fr.adjusted_score, 4),
        })
    return rows


def _ops_chromosome_to_dsl(chrom: StrategyChromosome) -> dict[str, Any]:
    """Convert an operators.StrategyChromosome to a DSL-like dict.

    Builds a simplified DSL dict from the chromosome for display and export.
    """
    indicators: dict[str, Any] = {}
    entry_conditions: list[str] = []
    exit_conditions: list[str] = []

    for i, gene in enumerate(chrom.entry_genes):
        name = f"{gene.indicator_type.lower()}_entry_{i}"
        cfg: dict[str, Any] = {"type": gene.indicator_type}
        for pname, val in gene.parameters.items():
            cfg[pname] = int(val) if val == int(val) else round(val, 4)
        indicators[name] = cfg
        op = gene.condition.value
        thr = int(gene.threshold) if gene.threshold == int(gene.threshold) else round(gene.threshold, 4)
        entry_conditions.append(f"{name} {op} {thr}")

    for i, gene in enumerate(chrom.exit_genes):
        name = f"{gene.indicator_type.lower()}_exit_{i}"
        cfg = {"type": gene.indicator_type}
        for pname, val in gene.parameters.items():
            cfg[pname] = int(val) if val == int(val) else round(val, 4)
        indicators[name] = cfg
        op = gene.condition.value
        thr = int(gene.threshold) if gene.threshold == int(gene.threshold) else round(gene.threshold, 4)
        exit_conditions.append(f"{name} {op} {thr}")

    direction = chrom.direction.value

    entry_dict: dict[str, list[str]] = {}
    exit_dict: dict[str, list[str]] = {}
    if direction in ("long", "both"):
        entry_dict["long"] = entry_conditions
        exit_dict["long"] = exit_conditions
    if direction in ("short", "both"):
        entry_dict["short"] = entry_conditions
        exit_dict["short"] = exit_conditions

    return {
        "name": f"discovered_{id(chrom)}",
        "indicators": indicators,
        "entry_conditions": entry_dict,
        "exit_conditions": exit_dict,
        "stop_loss": {"type": "fixed_pct", "percent": round(chrom.stop_loss_pct, 2)},
        "take_profit": {"type": "fixed_pct", "percent": round(chrom.take_profit_pct, 2)},
    }


def chromosome_to_yaml(chrom: StrategyChromosome) -> str:
    """Convert chromosome to YAML string for display.

    Args:
        chrom: Strategy chromosome (operators.StrategyChromosome).

    Returns:
        YAML-formatted string of the DSL dict.
    """
    dsl = _ops_chromosome_to_dsl(chrom)
    result: str = yaml.dump(dsl, default_flow_style=False, sort_keys=False)
    return result


# ---------------------------------------------------------------------------
# Render sections
# ---------------------------------------------------------------------------


def _render_config_section() -> DiscoveryConfig | None:
    """Render configuration sidebar/section. Returns config or None if invalid."""
    st.subheader("Discovery Configuration")

    col1, col2 = st.columns(2)

    with col1:
        population_size = st.slider(
            "Population Size",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            key="disc_pop_size",
            help="Number of strategies per generation",
        )

        max_generations = st.slider(
            "Max Generations",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key="disc_max_gen",
            help="Maximum evolutionary generations",
        )

        mutation_rate = st.slider(
            "Mutation Rate",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            key="disc_mutation_rate",
            help="Per-gene mutation probability",
        )

    with col2:
        elite_count = st.slider(
            "Elite Count",
            min_value=1,
            max_value=10,
            value=2,
            key="disc_elite_count",
            help="Top individuals preserved unchanged each generation",
        )

        symbols = st.multiselect(
            "Symbols",
            options=DISCOVERY_SYMBOLS,
            default=["BTCUSDT"],
            key="disc_symbols",
            help="Symbols to evaluate strategies on",
        )

        tf_list = sorted(VALID_TIMEFRAMES)
        default_idx = tf_list.index("1h") if "1h" in tf_list else 0
        timeframe = st.selectbox(
            "Timeframe",
            options=tf_list,
            index=default_idx,
            key="disc_timeframe",
        )

    # Date range
    st.markdown("**Date Range**")
    date_col1, date_col2 = st.columns(2)
    default_end = date.today()
    default_start = default_end - timedelta(days=365)

    with date_col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            min_value=date(2019, 1, 1),
            max_value=default_end,
            key="disc_start_date",
        )
    with date_col2:
        end_date = st.date_input(
            "End Date",
            value=default_end,
            min_value=start_date,
            max_value=date.today(),
            key="disc_end_date",
        )

    # Indicator pool display
    with st.expander("Available Indicators", expanded=False):
        for name, ind_def in INDICATOR_POOL.items():
            params_str = ", ".join(
                f"{p}: [{lo}-{hi}]"
                for p, (lo, hi) in ind_def.param_ranges.items()
            )
            st.caption(f"**{name}** ({ind_def.dsl_type}) -- {params_str}")

    # Validate
    if not symbols:
        st.warning("Select at least one symbol")
        return None

    if elite_count >= population_size:
        st.warning("Elite count must be less than population size")
        return None

    return DiscoveryConfig(
        population_size=population_size,
        max_generations=max_generations,
        mutation_rate=mutation_rate,
        elite_count=elite_count,
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )


def _render_start_button(
    config: DiscoveryConfig | None,
    job_manager: BacktestJobManager,
) -> None:
    """Render start discovery button and handle job launch."""
    st.divider()

    start_disabled = config is None
    if st.button(
        "Start Discovery",
        type="primary",
        key="start_discovery",
        disabled=start_disabled,
        help="Launch genetic strategy discovery as background job",
    ):
        if config is None:
            st.error("Fix configuration errors before starting")
            return

        # Build subprocess command
        db_path = st.session_state.get("db_path", str(DEFAULT_DB_PATH))
        command = [
            "python",
            "-m",
            "vibe_quant.discovery",
            "--population-size",
            str(config.population_size),
            "--max-generations",
            str(config.max_generations),
            "--mutation-rate",
            str(config.mutation_rate),
            "--elite-count",
            str(config.elite_count),
            "--symbols",
            ",".join(config.symbols),
            "--timeframe",
            config.timeframe,
            "--start-date",
            config.start_date,
            "--end-date",
            config.end_date,
            "--db",
            db_path,
        ]

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "discovery_latest.log")

        # Create a real backtest_run entry for FK integrity
        try:
            sm = get_state_manager()
            # Create a placeholder strategy for discovery if needed
            disc_strategy = sm.get_strategy_by_name("__discovery__")
            if disc_strategy is None:
                strategy_id = sm.create_strategy(
                    name="__discovery__",
                    dsl_config={"type": "discovery"},
                    description="Auto-created for discovery job tracking",
                    strategy_type="discovery",
                )
            else:
                strategy_id = disc_strategy["id"]

            run_id = sm.create_backtest_run(
                strategy_id=strategy_id,
                run_mode="discovery",
                symbols=config.symbols,
                timeframe=config.timeframe,
                start_date=config.start_date,
                end_date=config.end_date,
                parameters={
                    "population_size": config.population_size,
                    "max_generations": config.max_generations,
                    "mutation_rate": config.mutation_rate,
                    "elite_count": config.elite_count,
                },
            )

            pid = job_manager.start_job(
                run_id=run_id,
                job_type="discovery",
                command=command,
                log_file=log_file,
            )
            st.success(f"Discovery started (pid={pid}). Monitor progress below.")
            st.rerun()
        except ValueError as e:
            st.error(f"Failed to start: {e}")
        except Exception as e:
            st.error(f"Error starting discovery: {e}")


def _render_progress_section(
    generations: list[GenerationResult],
    max_generations: int,
) -> None:
    """Render generation progress bar and counter."""
    st.subheader("Progress")

    current = len(generations)
    st.write(f"Generation: **{current}** / **{max_generations}**")
    st.progress(min(current / max(max_generations, 1), 1.0))


def _render_fitness_chart(generations: list[GenerationResult]) -> None:
    """Render fitness evolution line chart with best/mean/worst."""
    if not generations:
        st.info("No generation data yet")
        return

    data = build_fitness_chart_data(generations)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["generation"],
        y=data["best"],
        mode="lines",
        name="Best",
        line={"color": "green", "width": 2},
    ))
    fig.add_trace(go.Scatter(
        x=data["generation"],
        y=data["mean"],
        mode="lines",
        name="Mean",
        line={"color": "blue", "width": 2},
    ))
    fig.add_trace(go.Scatter(
        x=data["generation"],
        y=data["worst"],
        mode="lines",
        name="Worst",
        line={"color": "red", "width": 2},
    ))

    fig.update_layout(
        title="Fitness Evolution",
        xaxis_title="Generation",
        yaxis_title="Fitness Score",
        height=400,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_results_section(result: DiscoveryResult) -> None:
    """Render top discovered strategies table and details."""
    st.subheader("Discovered Strategies")

    # Convergence info
    if result.converged:
        st.info(
            f"Converged at generation {result.convergence_generation}. "
            f"Total candidates evaluated: {result.total_candidates_evaluated:,}"
        )
    else:
        st.info(
            f"Completed {len(result.generations)} generations. "
            f"Total candidates evaluated: {result.total_candidates_evaluated:,}"
        )

    # Results table
    rows = build_results_table(result.top_strategies)
    if not rows:
        st.warning("No strategies discovered")
        return

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Expandable detail per strategy
    for i, (chrom, fr) in enumerate(result.top_strategies):
        with st.expander(f"Strategy #{i + 1} (Score: {fr.adjusted_score:.4f})", expanded=False):
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sharpe", f"{fr.sharpe_ratio:.4f}")
            with col2:
                st.metric("Max DD", f"{fr.max_drawdown:.2%}")
            with col3:
                st.metric("Profit Factor", f"{fr.profit_factor:.4f}")
            with col4:
                st.metric("Trades", str(fr.total_trades))

            # DSL YAML
            st.markdown("**Strategy DSL:**")
            yaml_str = chromosome_to_yaml(chrom)
            st.code(yaml_str, language="yaml")

            # Export button
            if st.button("Export to Strategy", key=f"export_strategy_{i}"):
                st.session_state[f"exported_dsl_{i}"] = _ops_chromosome_to_dsl(chrom)
                st.success(f"Strategy #{i + 1} exported. Go to Strategy Management to save.")


@st.fragment(run_every=5)
def _render_active_discovery_jobs(job_manager: BacktestJobManager) -> None:
    """Render active discovery jobs with status and kill button. Auto-refreshes every 5s."""
    st.subheader("Active Discovery Jobs")

    active_jobs = job_manager.list_active_jobs()
    discovery_jobs = [j for j in active_jobs if j.job_type == "discovery"]

    if not discovery_jobs:
        st.info("No active discovery jobs")
        return

    for job in discovery_jobs:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.write(f"**Discovery Job** (PID: {job.pid})")
                if job.started_at:
                    st.caption(f"Started: {job.started_at.strftime('%Y-%m-%d %H:%M:%S')}")

            with col2:
                status_colors = {
                    JobStatus.RUNNING: "green",
                    JobStatus.PENDING: "gray",
                    JobStatus.COMPLETED: "blue",
                    JobStatus.FAILED: "red",
                    JobStatus.KILLED: "orange",
                }
                color = status_colors.get(job.status, "gray")
                st.markdown(f":{color}[{job.status.value.upper()}]")

                if job.is_stale:
                    st.warning("STALE")

            with col3:
                if st.button("Kill", key=f"kill_disc_{job.run_id}_{job.pid}"):
                    if job_manager.kill_job(job.run_id):
                        st.success("Discovery job killed")
                        st.rerun()
                    else:
                        st.error("Failed to kill job")

        st.divider()


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def render_discovery_tab() -> None:
    """Render the complete discovery tab."""
    st.header("Strategy Discovery")
    st.caption("Genetic algorithm-based strategy evolution")

    job_manager = get_job_manager()

    # Configuration
    config = _render_config_section()

    # Start button
    _render_start_button(config, job_manager)

    st.divider()

    # Active jobs
    _render_active_discovery_jobs(job_manager)

    # Show results if available in session state
    discovery_result: DiscoveryResult | None = st.session_state.get("discovery_result")

    if discovery_result is not None:
        st.divider()

        # Progress
        max_gen = config.max_generations if config else 100
        _render_progress_section(discovery_result.generations, max_gen)

        # Fitness chart
        _render_fitness_chart(discovery_result.generations)

        # Results
        _render_results_section(discovery_result)


# Convenience alias matching pattern from other pages
render = render_discovery_tab

# Top-level call for st.navigation API
render_discovery_tab()
