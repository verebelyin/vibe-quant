"""Regime-cross discovery campaign orchestrator.

Motivation (from the 2026-04-17 handover): before promoting a 1m short
champion we need proof that it survives an opposing-regime window. A
strategy that only worked in the 2025 Q4 bear is not tradeable. This
module drives a multi-discovery campaign:

1. **Train**: run GA discovery on each training window (e.g. pure bear,
   pure bull, mixed) using the existing ``vibe_quant.discovery`` CLI
   as a subprocess.
2. **Promote**: extract top-K chromosomes per training window from
   ``backtest_results.notes.top_strategies``.
3. **OOS**: validate each chromosome against each OOS window via the
   existing ``vibe_quant.validation`` CLI.
4. **Matrix**: aggregate results into a per-(champion, window) grid.

Campaign metadata lives in ``backtest_runs.parameters`` under the key
``campaign_id`` (a random hex string) so all related rows can be
recovered without a new table. Discovery rows use
``run_mode='regime_cross_discovery'`` and OOS validations use
``run_mode='regime_cross_oos'``.

Typical usage::

    from vibe_quant.discovery.campaign import (
        RegimeCrossConfig, plan_campaign, run_campaign, build_matrix_report,
    )
    cfg = RegimeCrossConfig.from_yaml("campaign.yaml")
    plan = plan_campaign(cfg, state)
    run_campaign(plan, state)
    print(build_matrix_report(plan.campaign_id, state).as_text())
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibe_quant.db.state_manager import StateManager

logger = logging.getLogger(__name__)


RUN_MODE_DISCOVERY = "regime_cross_discovery"
RUN_MODE_OOS = "regime_cross_oos"


@dataclass(frozen=True, slots=True)
class TrainWindow:
    """One training window for GA discovery.

    Attributes:
        label: Short human label (e.g. ``"bear_2025_q4"``).
        start: ISO date (``YYYY-MM-DD``).
        end: ISO date.
        direction: ``"long"``, ``"short"``, or ``"both"``.
    """

    label: str
    start: str
    end: str
    direction: str


@dataclass(frozen=True, slots=True)
class OOSWindow:
    """One out-of-sample window used to cross-validate survivors."""

    label: str
    start: str
    end: str


@dataclass(frozen=True, slots=True)
class RegimeCrossConfig:
    """Parsed campaign configuration.

    Loaded via ``RegimeCrossConfig.from_yaml``. Arbitrary extras (GA
    knobs) flow through into discovery CLI args.
    """

    name: str
    symbol: str
    timeframe: str
    train_windows: list[TrainWindow]
    oos_windows: list[OOSWindow]
    population_size: int = 100
    max_generations: int = 20
    top_k: int = 3
    extra_discovery_args: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path | str) -> RegimeCrossConfig:
        """Parse a YAML campaign config. Raises ValueError on missing fields."""
        import yaml  # deferred: pyyaml is already a dep but keep import local

        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            msg = f"Campaign config must be a YAML mapping, got {type(raw).__name__}"
            raise ValueError(msg)

        required = ("name", "symbol", "timeframe", "train_windows", "oos_windows")
        missing = [k for k in required if k not in raw]
        if missing:
            msg = f"Campaign config missing required fields: {missing}"
            raise ValueError(msg)

        train = [
            TrainWindow(
                label=str(w["label"]),
                start=str(w["start"]),
                end=str(w["end"]),
                direction=str(w.get("direction", "both")),
            )
            for w in raw["train_windows"]
        ]
        oos = [
            OOSWindow(label=str(w["label"]), start=str(w["start"]), end=str(w["end"]))
            for w in raw["oos_windows"]
        ]
        return cls(
            name=str(raw["name"]),
            symbol=str(raw["symbol"]),
            timeframe=str(raw["timeframe"]),
            train_windows=train,
            oos_windows=oos,
            population_size=int(raw.get("population_size", 100)),
            max_generations=int(raw.get("max_generations", 20)),
            top_k=int(raw.get("top_k", 3)),
            extra_discovery_args=list(raw.get("extra_discovery_args", [])),
        )


@dataclass(slots=True)
class CampaignPlan:
    """Materialized campaign state: which runs to execute, keyed by window."""

    campaign_id: str
    config: RegimeCrossConfig
    # window label -> discovery run_id
    discovery_runs: dict[str, int] = field(default_factory=dict)
    # (train_label, champion_idx, oos_label) -> oos run_id
    oos_runs: dict[tuple[str, int, str], int] = field(default_factory=dict)


def plan_campaign(config: RegimeCrossConfig, state: StateManager) -> CampaignPlan:
    """Create ``backtest_runs`` rows for every training discovery.

    Only training discoveries are planned upfront. OOS rows are planned
    after discovery finishes — they need the surviving chromosomes as
    strategies.
    """
    campaign_id = uuid.uuid4().hex[:12]
    plan = CampaignPlan(campaign_id=campaign_id, config=config)

    for tw in config.train_windows:
        params = {
            "campaign_id": campaign_id,
            "campaign_name": config.name,
            "window_label": tw.label,
            "direction": tw.direction,
            "population_size": config.population_size,
            "max_generations": config.max_generations,
        }
        run_id = state.create_backtest_run(
            strategy_id=None,
            run_mode=RUN_MODE_DISCOVERY,
            symbols=[config.symbol],
            timeframe=config.timeframe,
            start_date=tw.start,
            end_date=tw.end,
            parameters=params,
        )
        plan.discovery_runs[tw.label] = run_id
        logger.info(
            "planned regime-cross discovery: window=%s direction=%s run_id=%d",
            tw.label, tw.direction, run_id,
        )

    return plan


def _run_subprocess(cmd: list[str], log_path: Path | None) -> int:
    """Run ``cmd`` synchronously, optionally tee'ing output to ``log_path``."""
    logger.info("launching: %s", " ".join(cmd))
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_fp:
            proc = subprocess.run(
                cmd, stdout=log_fp, stderr=subprocess.STDOUT, check=False
            )
    else:
        proc = subprocess.run(cmd, check=False)
    return proc.returncode


def run_discovery_subprocess(
    run_id: int,
    config: RegimeCrossConfig,
    window: TrainWindow,
    python_bin: str = sys.executable,
    log_dir: Path | str | None = None,
) -> int:
    """Launch a single discovery as a subprocess. Returns exit code."""
    cmd = [
        python_bin,
        "-m",
        "vibe_quant.discovery",
        "--run-id", str(run_id),
        "--symbols", config.symbol,
        "--timeframe", config.timeframe,
        "--start-date", window.start,
        "--end-date", window.end,
        "--direction", window.direction,
        "--population-size", str(config.population_size),
        "--max-generations", str(config.max_generations),
        *config.extra_discovery_args,
    ]
    log_path = Path(log_dir) / f"regime_cross_{run_id}.log" if log_dir else None
    rc = _run_subprocess(cmd, log_path)
    logger.info("discovery run_id=%d exit=%d", run_id, rc)
    return rc


def extract_top_chromosomes(
    run_id: int, state: StateManager, top_k: int = 3
) -> list[dict[str, object]]:
    """Pull top-K chromosomes from a finished discovery's result notes.

    Discovery writes ``top_strategies`` into ``backtest_results.notes``
    (JSON). Each entry has ``dsl``, ``chromosome``, and fitness fields.
    Returns the first ``top_k`` entries as-is.
    """
    row = state.conn.execute(
        "SELECT notes FROM backtest_results WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None or not row[0]:
        return []
    try:
        notes = json.loads(row[0])
    except json.JSONDecodeError:
        logger.warning("malformed notes JSON for run_id=%d", run_id)
        return []
    tops = notes.get("top_strategies") or []
    if not isinstance(tops, list):
        return []
    return list(tops[:top_k])


def plan_oos_validations(
    plan: CampaignPlan, state: StateManager
) -> CampaignPlan:
    """For each discovery's survivors, create a strategy + OOS run per OOS window.

    Mutates ``plan.oos_runs`` in place. Safe to re-call — existing OOS
    rows matching ``(train_label, champion_idx, oos_label)`` are kept.
    """
    for tw in plan.config.train_windows:
        disc_run_id = plan.discovery_runs.get(tw.label)
        if disc_run_id is None:
            continue
        champions = extract_top_chromosomes(disc_run_id, state, plan.config.top_k)
        if not champions:
            logger.warning("no champions extracted for window=%s run_id=%d", tw.label, disc_run_id)
            continue

        for idx, champ in enumerate(champions):
            dsl = champ.get("dsl")
            if not isinstance(dsl, dict):
                continue
            # Name strategy uniquely so reruns don't collide
            strategy_name = f"rcc_{plan.campaign_id}_{tw.label}_top{idx}"
            try:
                strategy_id = state.create_strategy(
                    name=strategy_name,
                    dsl_config=dsl,
                    description=(
                        f"Regime-cross campaign {plan.config.name}: top-{idx} "
                        f"from window {tw.label}"
                    ),
                    strategy_type="regime_cross",
                )
            except Exception as exc:  # noqa: BLE001 — UNIQUE violation on rerun
                logger.info(
                    "strategy %s already exists, reusing (%s)", strategy_name, exc
                )
                existing = state.conn.execute(
                    "SELECT id FROM strategies WHERE name = ?", (strategy_name,)
                ).fetchone()
                if existing is None:
                    continue
                strategy_id = int(existing[0])

            for oos in plan.config.oos_windows:
                key = (tw.label, idx, oos.label)
                if key in plan.oos_runs:
                    continue
                params = {
                    "campaign_id": plan.campaign_id,
                    "campaign_name": plan.config.name,
                    "source_window": tw.label,
                    "champion_idx": idx,
                    "oos_window": oos.label,
                    "discovery_run_id": disc_run_id,
                }
                run_id = state.create_backtest_run(
                    strategy_id=strategy_id,
                    run_mode=RUN_MODE_OOS,
                    symbols=[plan.config.symbol],
                    timeframe=plan.config.timeframe,
                    start_date=oos.start,
                    end_date=oos.end,
                    parameters=params,
                )
                plan.oos_runs[key] = run_id
                logger.info(
                    "planned OOS: source=%s champ=%d oos=%s run_id=%d",
                    tw.label, idx, oos.label, run_id,
                )

    return plan


def run_oos_subprocess(
    run_id: int,
    python_bin: str = sys.executable,
    log_dir: Path | str | None = None,
) -> int:
    """Launch a single validation (OOS) subprocess. Returns exit code."""
    cmd = [python_bin, "-m", "vibe_quant.validation", "--run-id", str(run_id)]
    log_path = Path(log_dir) / f"regime_cross_oos_{run_id}.log" if log_dir else None
    rc = _run_subprocess(cmd, log_path)
    logger.info("oos run_id=%d exit=%d", run_id, rc)
    return rc


def run_campaign(
    plan: CampaignPlan,
    state: StateManager,
    log_dir: Path | str | None = None,
    skip_existing: bool = True,
) -> CampaignPlan:
    """Execute discoveries then OOS validations sequentially.

    When ``skip_existing`` is true, any run already marked ``completed``
    is skipped — this makes the runner resumable after crashes.
    """
    # Phase 2: discoveries
    for tw in plan.config.train_windows:
        run_id = plan.discovery_runs[tw.label]
        if skip_existing and _run_completed(run_id, state):
            logger.info("skipping completed discovery run_id=%d", run_id)
            continue
        rc = run_discovery_subprocess(run_id, plan.config, tw, log_dir=log_dir)
        if rc != 0:
            logger.warning(
                "discovery run_id=%d returned non-zero (%d); continuing", run_id, rc
            )

    # Phase 3: plan OOS once discoveries are done, then run them
    plan_oos_validations(plan, state)
    for key, run_id in plan.oos_runs.items():
        if skip_existing and _run_completed(run_id, state):
            logger.info("skipping completed oos run_id=%d (%s)", run_id, key)
            continue
        rc = run_oos_subprocess(run_id, log_dir=log_dir)
        if rc != 0:
            logger.warning("oos run_id=%d returned non-zero (%d); continuing", run_id, rc)

    return plan


def _run_completed(run_id: int, state: StateManager) -> bool:
    row = state.conn.execute(
        "SELECT status FROM backtest_runs WHERE id = ?", (run_id,)
    ).fetchone()
    return row is not None and row[0] == "completed"


@dataclass(slots=True)
class MatrixCell:
    """One (champion, oos_window) result cell."""

    source_window: str
    champion_idx: int
    oos_window: str
    sharpe: float | None = None
    total_return: float | None = None
    max_drawdown: float | None = None
    total_trades: int | None = None
    oos_run_id: int | None = None


@dataclass(slots=True)
class MatrixReport:
    """Aggregated campaign report."""

    campaign_id: str
    campaign_name: str
    cells: list[MatrixCell] = field(default_factory=list)

    def as_text(self) -> str:
        """Render as an ASCII table for terminal viewing."""
        if not self.cells:
            return f"Campaign {self.campaign_name} ({self.campaign_id}): no results yet\n"
        header = (
            f"{'source':<18}{'#':>3}  {'oos':<18}"
            f"  {'sharpe':>7}  {'return%':>8}  {'mdd%':>7}  {'trades':>7}"
        )
        rows = [f"Campaign {self.campaign_name} ({self.campaign_id})", "", header, "-" * len(header)]
        for c in self.cells:
            sharpe = f"{c.sharpe:7.2f}" if c.sharpe is not None else "  n/a  "
            ret = (
                f"{c.total_return * 100:8.2f}"
                if c.total_return is not None
                else "   n/a  "
            )
            mdd = (
                f"{c.max_drawdown * 100:7.2f}"
                if c.max_drawdown is not None
                else "  n/a  "
            )
            trades = f"{c.total_trades:7d}" if c.total_trades is not None else "   n/a "
            rows.append(
                f"{c.source_window:<18}{c.champion_idx:>3}  {c.oos_window:<18}"
                f"  {sharpe}  {ret}  {mdd}  {trades}"
            )
        return "\n".join(rows) + "\n"


def build_matrix_report(campaign_id: str, state: StateManager) -> MatrixReport:
    """Query all OOS rows for a campaign and assemble the matrix."""
    rows = state.conn.execute(
        """
        SELECT br.id, br.parameters, r.sharpe_ratio, r.total_return,
               r.max_drawdown, r.total_trades
        FROM backtest_runs br
        LEFT JOIN backtest_results r ON r.run_id = br.id
        WHERE br.run_mode = ?
          AND json_extract(br.parameters, '$.campaign_id') = ?
        ORDER BY br.id
        """,
        (RUN_MODE_OOS, campaign_id),
    ).fetchall()

    cells: list[MatrixCell] = []
    campaign_name = ""
    for row in rows:
        try:
            params = json.loads(row[1]) if row[1] else {}
        except json.JSONDecodeError:
            continue
        if not campaign_name:
            campaign_name = str(params.get("campaign_name", ""))
        cells.append(
            MatrixCell(
                source_window=str(params.get("source_window", "")),
                champion_idx=int(params.get("champion_idx", 0)),
                oos_window=str(params.get("oos_window", "")),
                sharpe=float(row[2]) if row[2] is not None else None,
                total_return=float(row[3]) if row[3] is not None else None,
                max_drawdown=float(row[4]) if row[4] is not None else None,
                total_trades=int(row[5]) if row[5] is not None else None,
                oos_run_id=int(row[0]),
            )
        )
    return MatrixReport(campaign_id=campaign_id, campaign_name=campaign_name, cells=cells)


__all__ = [
    "CampaignPlan",
    "MatrixCell",
    "MatrixReport",
    "OOSWindow",
    "RUN_MODE_DISCOVERY",
    "RUN_MODE_OOS",
    "RegimeCrossConfig",
    "TrainWindow",
    "build_matrix_report",
    "extract_top_chromosomes",
    "plan_campaign",
    "plan_oos_validations",
    "run_campaign",
    "run_discovery_subprocess",
    "run_oos_subprocess",
]
