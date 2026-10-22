"""Discovery run notes: the provenance writers of one discovery run.

Moved verbatim from ``vibe_quant.discovery.__main__`` (bead vibe-quant-rabmo.5,
step T9) so the CLI and ``DiscoveryRunSpec.to_notes`` share one notes writer.
Nothing here imports ``vibe_quant.discovery.__main__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_quant.discovery.neff import N_RULE_EFFECTIVE
from vibe_quant.fast_tier import fallbacks
from vibe_quant.fees import FeeSchedule
from vibe_quant.liquidation_settings import DEFAULT_LEVERAGE, LiquidationSettings

if TYPE_CHECKING:
    from vibe_quant.discovery.pipeline import DiscoveryResult
    from vibe_quant.sizing_settings import SizingSettings


def compiler_version() -> str:
    """Get compiler version hash for staleness detection."""
    try:
        from vibe_quant.dsl.compiler import compiler_version_hash

        return compiler_version_hash()
    except Exception:
        return "unknown"


def attach_ind_memo(notes: dict[str, object], *, use_mock: bool) -> None:
    """Record the indicator-memo run summary in the persisted notes.

    Called right before each save so the full-range headline backtests are
    included; mock runs never touch the memo, so they carry no key.
    """
    if use_mock:
        return
    from vibe_quant.dsl import indicator_memo

    notes["ind_memo"] = indicator_memo.run_summary()


def provenance_notes(
    result: DiscoveryResult,
    *,
    use_mock: bool,
    eval_windows_count: int,
    eval_windows: list[tuple[str, str]] | None,
    symbol_agg: str,
    split_ratio: float,
    direction: str | None,
    cross_window_months: list[int],
    cross_window_min_sharpe: float,
    wfa_oos_step_days: int,
    wfa_min_consistency: float,
    num_seeds: int,
    seed: int,
    bootstrap_min_sharpe: float,
    holdout_min_sharpe: float,
    symbols: list[str],
    fee_schedule: FeeSchedule | None = None,
    prior_run_ids: list[int] | None = None,
    prior_legacy: dict[str, int] | None = None,
    liquidation: LiquidationSettings | None = None,
    sizing: SizingSettings | None = None,
    indicator_pool: list[str] | None = None,
) -> dict[str, object]:
    """Run-level notes shared by the champion and zero-champion outcomes."""
    schedule = fee_schedule if fee_schedule is not None else FeeSchedule()
    liquidation_notes = liquidation.to_notes() if liquidation is not None else None
    notes: dict[str, object] = {
        "type": "discovery",
        "generations": len(result.generations),
        # Distinct strategies backtested == DSR trial count N (pooled across
        # seeds for multi-seed runs).
        "evaluated": result.total_candidates_evaluated,
        "converged": result.converged,
        "mock": use_mock,
        "synthetic": use_mock,
        "compiler_version": compiler_version(),
        "eval_windows": eval_windows_count if eval_windows_count > 1 else None,
        "eval_window_aggregation": "worst_of_n" if eval_windows_count > 1 else None,
        "eval_window_ranges": eval_windows if eval_windows else None,
        # "worst": each symbol scored alone, genome keeps its worst symbol.
        "symbol_agg": symbol_agg,
        "train_test_split": split_ratio if split_ratio > 0 else None,
        "train_dates": list(result.train_dates) if result.train_dates else None,
        "holdout_dates": list(result.holdout_dates) if result.holdout_dates else None,
        "holdout_gate": (
            {
                "min_trades": result.holdout_min_trades,
                "min_sharpe": holdout_min_sharpe,
                "min_return": 0.0,
            }
            if result.holdout_dates
            else None
        ),
        "direction": direction,
        "cross_window_months": cross_window_months or None,
        "cross_window_min_sharpe": cross_window_min_sharpe if cross_window_months else None,
        "wfa_oos_step_days": wfa_oos_step_days if wfa_oos_step_days > 0 else None,
        "wfa_min_consistency": wfa_min_consistency if wfa_oos_step_days > 0 else None,
        "num_seeds": num_seeds if num_seeds > 1 else None,
        # Single-seed: the GA seed. Multi-seed: base; seed i = base + i * 7919.
        "seed": seed,
        "bootstrap_min_sharpe": bootstrap_min_sharpe,
        # Venue-level fee schedule (bd vibe-quant-2gyk1.4): the same schedule
        # every screening backtest of this run used.
        "fee_schedule": schedule.to_notes(),
    }
    # Run-level liquidation settings (bead vibe-quant-2gyk1.1.9, D2): the same
    # settings every screening backtest of this run used. The key is absent when
    # the mode is off, so existing runs stay unchanged (R4).
    if liquidation_notes is not None:
        notes["liquidation"] = liquidation_notes
    # Run-level sizing settings (bead vibe-quant-2gyk1.2): the settings every
    # screening backtest of this run used, recorded so promote/replay inherit
    # them. Only the keys moved off the default are written -- a default run
    # adds no key and its notes stay byte-identical to today (S23c).
    if sizing is not None:
        sizing_params = sizing.to_params()
        if sizing_params:
            notes["sizing"] = sizing_params
    # A non-default venue leverage is recorded so promote/replay inherit it
    # (bead vibe-quant-6wq6w). Only a mode-OFF run needs the top-level key: a
    # mode-on run already carries its leverage in notes.liquidation, and an
    # explicit default (10) writes nothing (R4/R7).
    if (
        liquidation is not None
        and not liquidation.on
        and liquidation.leverage != DEFAULT_LEVERAGE
    ):
        notes["leverage"] = liquidation.leverage
    # An explicit GA pool (scalar AND MA genes) is recorded so a harness can check what
    # the run used, not what it asked for (P-2, 2026-10-19: 96+ runs ignored the pool for
    # MA genes). A default-pool run adds no key.
    if indicator_pool is not None:
        notes["indicator_pool"] = list(indicator_pool)
    if symbol_agg == "worst":
        # GA rank score over the per-symbol adjusted fitness; champions still
        # need every symbol's train return > 0 and trades >= min (hard gate).
        notes["worst_mode_score"] = "0.5*min+0.5*median"
    if result.fast_tier_confirm is not None:
        notes["evaluator"] = {
            "name": "fast",
            "nt_confirmed": result.fast_tier_confirm.confirmed,
            "nt_fallbacks": fallbacks.summary(),
        }
    if result.dsr_trials is not None:
        notes["dsr_trials"] = dsr_trials_note(result, prior_run_ids or [], prior_legacy)
    elif not use_mock:
        # B1 (bd vibe-quant-megq1): a --no-dsr run runs no gate, but it must still
        # record n_rule NEW at run_n_eff == raw, or A3 would treat it as a legacy
        # run and never charge its trials. Mock runs are not real trials at all.
        notes["dsr_trials"] = dsr_trials_note(result, prior_run_ids or [], prior_legacy)
    # Buy-and-hold gate provenance (bd vibe-quant-wcn55). ``enforced`` comes from
    # the pipeline (False only for a non-NT backtest fn); the rest is the gate's
    # fixed configuration. A run with --no-buy-hold-gate has enforced False.
    notes["buy_hold_gate"] = {
        "enforced": bool((result.buy_hold_gate or {}).get("enforced", False)),
        "benchmark": "equal_weight_daily_rebalanced_1d_close",
        "candidate_series": "mark_to_market_daily",
        "symbols": list(symbols),
        "ann_factor": 365,
        "per_symbol": symbol_agg == "worst",
    }
    return notes


def dsr_trials_note(
    result: DiscoveryResult,
    prior_run_ids: list[int],
    prior_legacy: dict[str, int] | None = None,
) -> dict[str, object]:
    """``notes.dsr_trials``: gate N and the effective-N rule that produced it.

    The gate N is ``this run's run_n_eff + prior_n`` -- the summed
    ``run_n_eff`` of earlier non-mock discovery runs with the same timeframe and
    an OVERLAPPING train window that were themselves recorded under
    ``N_RULE_EFFECTIVE`` (bd vibe-quant-megq1, decision A3). Earlier runs with no
    such rule are listed in ``prior_legacy_runs`` / ``prior_legacy_raw`` but are
    NOT charged. The pipeline records ``gate_n``/``prior_n`` when it runs the
    gate; ``prior_run_ids`` lists the contributing runs. A failed prior lookup
    fails open to the run-only N and sets ``prior_error``.

    A ``--no-dsr`` run (``require_dsr`` False) has ``result.dsr_trials`` None: it
    still records ``n_rule`` NEW with ``run_n_eff`` == the raw evaluated count and
    ``gate_n`` None (no gate ran), so the run is charged by later runs (B1).
    """
    note: dict[str, object] = dict(result.dsr_trials or {})
    if not note:
        raw = result.total_candidates_evaluated
        note["raw"] = raw
        note["run_n_eff"] = raw
        note["neff_ev_source"] = "dsr_disabled"
        note["gate_n"] = None
    legacy = prior_legacy or {}
    note["n_rule"] = N_RULE_EFFECTIVE
    note["prior_run_ids"] = list(prior_run_ids)
    if "prior_n" not in note:
        note["prior_n"] = 0
    if "gate_n" not in note:
        prior_raw = note["prior_n"]
        prior = prior_raw if isinstance(prior_raw, int) else 0
        run_side = note.get("run_n_eff")
        if not isinstance(run_side, int) or isinstance(run_side, bool):
            # Legacy/partial dsr_trials payload with no effective count: fall
            # back to the raw evaluated count (the pre-T3 run-side N).
            run_side = result.total_candidates_evaluated
        note["gate_n"] = run_side + prior
    note["prior_legacy_runs"] = int(legacy.get("runs", 0))
    note["prior_legacy_raw"] = int(legacy.get("raw", 0))
    return note
