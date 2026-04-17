#!/usr/bin/env python3
"""Regime-cross discovery campaign runner.

Executes a multi-window GA discovery campaign where each surviving
strategy is validated against opposing-regime windows as OOS. This
closes the step-1 gate from the handover — no strategy promotes to
paper/live without positive results across opposing regimes.

Subcommands:
  plan <config.yaml>         Create discovery rows; print campaign id.
  run <campaign-id>          Execute discoveries + OOS (resumable).
  report <campaign-id>       Print the cross-regime result matrix.
  run-all <config.yaml>      plan + run + report in one shot.

Example::

    .venv/bin/python scripts/regime_cross_campaign.py run-all \\
        docs/plans/regime_cross_example.yaml

Warnings:
  - Discoveries are slow (~30m per window at pop=100 gen=20 on 1h).
    run-all on a 3-train × 3-OOS config is multi-hour compute.
  - Runs sequentially. Use shell & + multiple configs for parallelism.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.campaign import (
    CampaignPlan,
    RegimeCrossConfig,
    build_matrix_report,
    plan_campaign,
    run_campaign,
)

logger = logging.getLogger("regime_cross_campaign")


def _load_plan_from_db(campaign_id: str, state: StateManager) -> CampaignPlan | None:
    """Rehydrate a CampaignPlan from persisted backtest_runs rows.

    A plan is identified by the ``campaign_id`` stored in each row's
    ``parameters`` JSON. Without a saved config we reconstruct a
    minimal one — run_campaign only needs the runs + top_k.
    """
    import json

    rows = state.conn.execute(
        "SELECT id, run_mode, parameters, symbols, timeframe, start_date, end_date "
        "FROM backtest_runs WHERE run_mode IN "
        "('regime_cross_discovery', 'regime_cross_oos') ORDER BY id"
    ).fetchall()

    discovery_runs: dict[str, int] = {}
    oos_runs: dict[tuple[str, int, str], int] = {}
    campaign_name = ""
    symbol = ""
    timeframe = ""
    top_k = 3
    pop = 100
    gens = 20
    train_windows_map: dict[str, tuple[str, str, str]] = {}
    oos_windows_map: dict[str, tuple[str, str]] = {}

    for row in rows:
        try:
            params = json.loads(row[2]) if row[2] else {}
        except json.JSONDecodeError:
            continue
        if params.get("campaign_id") != campaign_id:
            continue

        campaign_name = str(params.get("campaign_name", campaign_name))
        try:
            syms = json.loads(row[3])
            if syms:
                symbol = str(syms[0])
        except (json.JSONDecodeError, IndexError):
            pass
        timeframe = str(row[4] or timeframe)

        if row[1] == "regime_cross_discovery":
            label = str(params.get("window_label", ""))
            discovery_runs[label] = int(row[0])
            train_windows_map[label] = (
                str(row[5]), str(row[6]), str(params.get("direction", "both"))
            )
            pop = int(params.get("population_size", pop))
            gens = int(params.get("max_generations", gens))
        elif row[1] == "regime_cross_oos":
            key = (
                str(params.get("source_window", "")),
                int(params.get("champion_idx", 0)),
                str(params.get("oos_window", "")),
            )
            oos_runs[key] = int(row[0])
            oos_windows_map[key[2]] = (str(row[5]), str(row[6]))

    if not discovery_runs:
        return None

    from vibe_quant.discovery.campaign import OOSWindow, TrainWindow

    train_windows = [
        TrainWindow(label=label, start=s, end=e, direction=d)
        for label, (s, e, d) in train_windows_map.items()
    ]
    oos_windows = [
        OOSWindow(label=label, start=s, end=e)
        for label, (s, e) in oos_windows_map.items()
    ]
    config = RegimeCrossConfig(
        name=campaign_name or campaign_id,
        symbol=symbol,
        timeframe=timeframe or "1h",
        train_windows=train_windows,
        oos_windows=oos_windows,
        population_size=pop,
        max_generations=gens,
        top_k=top_k,
    )
    plan = CampaignPlan(campaign_id=campaign_id, config=config)
    plan.discovery_runs = discovery_runs
    plan.oos_runs = oos_runs
    return plan


def cmd_plan(args: argparse.Namespace) -> int:
    state = StateManager()
    try:
        config = RegimeCrossConfig.from_yaml(args.config)
        plan = plan_campaign(config, state)
    finally:
        state.close()

    print(f"campaign_id: {plan.campaign_id}")
    print(f"name:        {plan.config.name}")
    print(f"discoveries planned: {len(plan.discovery_runs)}")
    for label, run_id in plan.discovery_runs.items():
        print(f"  {label:<20} run_id={run_id}")
    print(f"\nRun:    .venv/bin/python scripts/regime_cross_campaign.py run {plan.campaign_id}")
    print(f"Report: .venv/bin/python scripts/regime_cross_campaign.py report {plan.campaign_id}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    state = StateManager()
    try:
        plan = _load_plan_from_db(args.campaign_id, state)
        if plan is None:
            print(f"error: campaign {args.campaign_id} not found", file=sys.stderr)
            return 2
        run_campaign(plan, state, log_dir=args.log_dir, skip_existing=not args.force)
    finally:
        state.close()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    state = StateManager()
    try:
        report = build_matrix_report(args.campaign_id, state)
    finally:
        state.close()
    print(report.as_text())
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    state = StateManager()
    try:
        config = RegimeCrossConfig.from_yaml(args.config)
        plan = plan_campaign(config, state)
        logger.info("campaign %s planned: %d discoveries", plan.campaign_id, len(plan.discovery_runs))
        run_campaign(plan, state, log_dir=args.log_dir, skip_existing=True)
        report = build_matrix_report(plan.campaign_id, state)
    finally:
        state.close()
    print(report.as_text())
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Create discovery rows from config")
    p_plan.add_argument("config", type=Path)
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="Execute planned runs")
    p_run.add_argument("campaign_id")
    p_run.add_argument("--log-dir", type=Path, default=Path("logs/regime_cross"))
    p_run.add_argument("--force", action="store_true", help="Re-run completed runs")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="Print cross-regime matrix")
    p_rep.add_argument("campaign_id")
    p_rep.set_defaults(func=cmd_report)

    p_all = sub.add_parser("run-all", help="Plan + run + report in one shot")
    p_all.add_argument("config", type=Path)
    p_all.add_argument("--log-dir", type=Path, default=Path("logs/regime_cross"))
    p_all.set_defaults(func=cmd_run_all)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
