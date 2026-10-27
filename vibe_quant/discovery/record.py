"""Discovery run notes: the writers and readers of one discovery run's notes.

The provenance writers moved verbatim from ``vibe_quant.discovery.__main__``
(bead vibe-quant-rabmo.5, step T9) so the CLI and ``DiscoveryRunSpec.to_notes``
share one notes writer. The readers moved verbatim from
``vibe_quant.api.routers.discovery`` (bead vibe-quant-rabmo.8, step T10):
``DiscoveryRecord`` is the ONE view of a run's stored notes, so promote / replay
no longer parse the column once per inherited setting. Nothing here imports
``vibe_quant.discovery.__main__``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vibe_quant.db.result_notes import notes_mapping, parse_notes
from vibe_quant.discovery.neff import N_RULE_EFFECTIVE
from vibe_quant.fast_tier import fallbacks
from vibe_quant.fees import FeeSchedule
from vibe_quant.liquidation_settings import DEFAULT_LEVERAGE, LiquidationSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from vibe_quant.bankroll_grid import BankrollCell
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.discovery.gate_chain import (
        CrossWindowResult,
        HoldoutResult,
        WFARollingResult,
    )
    from vibe_quant.discovery.operators import StrategyChromosome
    from vibe_quant.discovery.pipeline import DiscoveryResult
    from vibe_quant.sizing_settings import SizingSettings


logger = logging.getLogger(__name__)


def compiler_version() -> str:
    """Get compiler version hash for staleness detection."""
    try:
        from vibe_quant.dsl.compiler import compiler_version_hash

        return compiler_version_hash()
    except Exception:
        return "unknown"


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


# ---------------------------------------------------------------------------
# Writers of one champion entry (moved verbatim from discovery/__main__.py, T12)
# ---------------------------------------------------------------------------


def _window_metrics(hr: HoldoutResult) -> dict[str, object]:
    return {
        "sharpe": hr.sharpe_ratio,
        "max_dd": hr.max_drawdown,
        "pf": hr.profit_factor,
        "trades": hr.total_trades,
        "return_pct": hr.total_return,
    }


def cross_window_entry(cwr: CrossWindowResult) -> dict[str, object]:
    """Persisted cross-window payload: SHIFTED windows only, with offsets + dates.

    ``windows[i]`` corresponds to ``cross_window_months[i]`` (the in-sample
    window is not listed -- it never counts as a pass).
    """
    return {
        "windows_passed": cwr.windows_passed,
        "total_windows": cwr.total_windows,
        "required": cwr.required,
        "passed": cwr.passed,
        "in_sample_excluded": True,
        "windows": [
            {
                **_window_metrics(w),
                "offset_months": cwr.offsets_months[i] if i < len(cwr.offsets_months) else None,
                "dates": list(cwr.window_dates[i]) if i < len(cwr.window_dates) else None,
            }
            for i, w in enumerate(cwr.window_results)
        ],
    }


def wfa_entry(wfa: WFARollingResult) -> dict[str, object]:
    """Persisted rolling-window (train range) stability payload."""
    return {
        "scope": "train_range_rolling",
        "windows_profitable": wfa.windows_profitable,
        "windows_sharpe_positive": wfa.windows_sharpe_positive,
        "total_windows": wfa.total_windows,
        "consistency": wfa.consistency,
        "sharpe_consistency": wfa.sharpe_consistency,
        "passed": wfa.passed,
        "windows": [
            {
                "dates": wfa.window_dates[j] if j < len(wfa.window_dates) else None,
                "sharpe": w.sharpe_ratio,
                "return_pct": w.total_return,
                "trades": w.total_trades,
            }
            for j, w in enumerate(wfa.oos_windows)
        ],
    }


def holdout_entry(hr: HoldoutResult) -> dict[str, object]:
    """Persisted holdout metrics: the base five, then optional sum / liquidation keys.

    ``trades_sum``/``symbol_metrics`` (worst mode only) and ``liquidation_count``
    (mode on only) are omitted when unavailable -- never ``null`` (R2).
    """
    entry: dict[str, object] = {
        "sharpe": hr.sharpe_ratio,
        "max_dd": hr.max_drawdown,
        "pf": hr.profit_factor,
        "trades": hr.total_trades,
        "return_pct": hr.total_return,
    }
    if hr.trades_sum is not None:
        entry["trades_sum"] = hr.trades_sum
        entry["symbol_metrics"] = hr.symbol_metrics
    # Report-only liquidation count (R5): written only when on.
    if hr.liquidation_count is not None:
        entry["liquidation_count"] = hr.liquidation_count
    return entry


def champion_entry(
    chrom: StrategyChromosome,
    fitness: FitnessResult,
    *,
    timeframe: str,
    dsr: dict[str, object] | None,
    buy_hold: dict[str, object] | None,
) -> dict[str, object]:
    """Base champion entry (dsr/buy_hold/symbol scores + fee drag).

    ``fee_drag``/``fees_pct_equity_yr`` (bd vibe-quant-2gyk1.4) are reporting
    only -- they never influence selection. Full-range/holdout/cross-window/WFA
    payloads are added by the caller (they are index-dependent).
    """
    from vibe_quant.discovery.genome import chromosome_to_dsl, chromosome_to_serializable

    dsl = chromosome_to_dsl(chrom)
    dsl["timeframe"] = timeframe
    entry: dict[str, object] = {
        "dsl": dsl,
        "chromosome": chromosome_to_serializable(chrom),
        "score": fitness.adjusted_score,
        "sharpe": fitness.sharpe_ratio,
        "max_dd": fitness.max_drawdown,
        "pf": fitness.profit_factor,
        "trades": fitness.total_trades,
        "return_pct": fitness.total_return,
    }
    if dsr is not None:
        entry["dsr"] = dsr
    if buy_hold is not None:
        entry["buy_hold"] = buy_hold
    if fitness.symbol_scores is not None:
        entry["symbol_scores"] = {
            sym: round(sc, 6) for sym, sc in fitness.symbol_scores.items()
        }
    # Always persisted (JSON null when unavailable) so readers can tell
    # "no edge to attribute" from "key never written".
    entry["fee_drag"] = fitness.fee_drag
    entry["fees_pct_equity_yr"] = fitness.fees_pct_equity_yr
    # Report-only liquidation counts (bead vibe-quant-2gyk1.1.9, R1): written
    # only when the mode is on (the FitnessResult fields stay None off -> the
    # keys are absent, R4).
    if fitness.liquidation_count is not None:
        entry["liquidation_count"] = fitness.liquidation_count
    if fitness.liquidation_forfeited is not None:
        entry["liquidation_forfeited"] = fitness.liquidation_forfeited
    return entry


def no_champion_notes(
    run_notes: Mapping[str, object],
    *,
    best_raw_score: float,
    guardrail_rejections: list[dict[str, object]],
) -> dict[str, object]:
    """Zero-champion summary notes: run notes + the structured rejection outcome.

    ``reason`` names the distinct rejection stages (or the no-positive-score
    fallback). Pure: ``run_notes`` is copied, never mutated (R3).
    """
    stages = sorted({
        str(r.get("stage", "guardrails")) for r in guardrail_rejections
    })
    return {
        **run_notes,
        "outcome": "no_viable_strategies",
        "best_raw_score": best_raw_score,
        "reason": (
            "No candidate passed every gate"
            + (f" (rejected at: {', '.join(stages)})" if stages else
               " (no candidate scored above zero)")
            + ". See guardrail_rejections for per-candidate reasons. Consider: "
            "longer date range, larger population, or a different indicator pool."
        ),
        "top_strategies": [],
        "guardrail_rejections": guardrail_rejections,
    }


# ---------------------------------------------------------------------------
# Readers of one stored discovery run
# ---------------------------------------------------------------------------


class RecordUnreadable(Exception):
    """A stored notes key that cannot be read (the router maps it to HTTP 409).

    Deliberately NOT a ``ValueError`` / ``TypeError``: :meth:`DiscoveryRecord.sizing_params`
    resolves its leverage through :meth:`DiscoveryRecord.liquidation_params` inside
    ``except (ValueError, TypeError)``, so a liquidation failure must pass through it
    unchanged (R23b).
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ChampionBankrollInvalid(Exception):
    """A grid champion without a readable bankroll cell (the router maps it to HTTP 422)."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True)
class ChildParams:
    """The run parameters a promoted / replayed child inherits from its parent run.

    Attributes:
        fee: Fee-schedule run parameters (empty for the default schedule).
        fee_source: Where the schedule came from (``discovery_notes`` or the legacy tag).
        liquidation: Liquidation run parameters (empty when the parent ran off at 10).
        sizing: Sizing run parameters (empty when the parent ran at the defaults).
    """

    fee: dict[str, object]
    fee_source: str
    liquidation: dict[str, object]
    sizing: dict[str, float]

    def run_parameters(self) -> dict[str, object]:
        """The child's stored parameters: fee, then liquidation, then sizing."""
        return {**self.fee, **self.liquidation, **self.sizing}


@dataclass(frozen=True)
class DiscoveryRecord:
    """One discovery run's stored notes, read ONCE and answered from memory.

    ``payload`` is :func:`notes_mapping` of the result row's ``notes`` (``{}`` when the run has
    no result row or no readable notes object), so every reader of a promote / replay request
    shares ONE ``state.get_backtest_result`` call.
    """

    run_id: int
    payload: Mapping[str, object]

    @classmethod
    def load(cls, state: StateManager, run_id: int) -> DiscoveryRecord:
        """Load the run's notes through ONE ``state.get_backtest_result`` call."""
        result = state.get_backtest_result(run_id)
        notes = "" if result is None else result.get("notes", "")
        return cls(run_id, notes_mapping(notes))

    def top_strategies(self) -> list[dict[str, object]]:
        """Discovered strategies as stored (elements unchecked) else ``[]``."""
        strategies = self.payload.get("top_strategies")
        return strategies if isinstance(strategies, list) else []

    def seed(self) -> int | None:
        """The run's GA seed (an ``int`` that is not a ``bool``), else ``None``."""
        raw = self.payload.get("seed")
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else None

    def reason(self) -> str | None:
        """``notes.reason`` when it is a string."""
        raw = self.payload.get("reason")
        return raw if isinstance(raw, str) else None

    def guardrail_rejections(self) -> list[dict[str, object]] | None:
        """``notes.guardrail_rejections`` when it is a list."""
        raw = self.payload.get("guardrail_rejections")
        return raw if isinstance(raw, list) else None

    def holdout_dates(self) -> tuple[str, str] | None:
        """The stored holdout range as ``(start, end)``, else ``None``."""
        raw = self.payload.get("holdout_dates")
        if isinstance(raw, list) and len(raw) == 2:
            return (str(raw[0]), str(raw[1]))
        return None

    def symbol_agg(self) -> object:
        """``notes.symbol_agg`` (a CLI-launched run records it only here)."""
        return self.payload.get("symbol_agg")

    def compiler_version(self) -> object:
        """``notes.compiler_version`` of the run that produced these notes."""
        return self.payload.get("compiler_version")

    def fee_params(self) -> tuple[dict[str, object], str]:
        """Fee-schedule run parameters for a child of this run.

        The parent's ``notes.fee_schedule`` is authoritative: a promoted or replayed
        child must be judged at the schedule the champion was found under. A run
        that predates the schedule (no ``fee_schedule`` in its notes) was found at
        VIP0 without BNB, so the child gets no fee keys and is tagged
        ``legacy_default_vip0``. A recorded schedule that cannot be parsed fails
        closed (409) instead of silently reverting to VIP0.

        Returns:
            ``(fee run parameters, source tag)``; the parameters are empty for the
            default schedule so default children keep their stored parameters.
        """
        from vibe_quant.fees import DEFAULT_TIER, FeeSchedule

        raw = self.payload.get("fee_schedule")
        if raw is None:
            return {}, "legacy_default_vip0"
        try:
            if not isinstance(raw, dict):
                raise ValueError(f"expected an object, got {raw!r}")
            schedule = FeeSchedule.from_params(raw)
        except (ValueError, TypeError) as exc:
            raise RecordUnreadable(
                f"discovery run {self.run_id} has an unreadable notes.fee_schedule: {exc}"
            ) from exc
        fee_params: dict[str, object] = {}
        if schedule.bnb_discount:
            fee_params["bnb_discount"] = True
        if schedule.tier != DEFAULT_TIER:
            fee_params["fee_tier"] = schedule.tier
        return fee_params, "discovery_notes"

    def grid_cell(self, strategy_index: int | None) -> BankrollCell | None:
        """The bankroll cell a child of a GRID run must inherit (``None`` for a non-grid run).

        A grid run (``notes.bankroll_grid`` present) has no run-level setup: each champion carries
        its own ``bankroll``. A missing / null / wrong-type / invalid one fails closed with a 422
        naming the champion (bead vibe-quant-t8nrv, R56b) instead of reverting to a default.
        """
        from vibe_quant.bankroll_grid import BankrollCell, BankrollGridError

        payload = self.payload
        if "bankroll_grid" not in payload:
            return None
        top = payload.get("top_strategies")
        try:
            if (
                strategy_index is None
                or not isinstance(top, list)
                or not 0 <= strategy_index < len(top)
            ):
                raise BankrollGridError("no champion to read a bankroll from")
            entry = top[strategy_index]
            if not isinstance(entry, dict):
                raise BankrollGridError("champion entry is not an object")
            return BankrollCell.from_note(entry.get("bankroll"))
        except BankrollGridError as exc:
            raise ChampionBankrollInvalid(
                f"discovery run {self.run_id} strategy {strategy_index} "
                "has no valid bankroll setup"
            ) from exc

    def liquidation_params(self, *, strategy_index: int | None = None) -> dict[str, object]:
        """Liquidation run parameters for a child of this run.

        A grid run hands the child ITS CHAMPION's cell (``strategy_index``); see
        :meth:`grid_cell`.

        The parent's ``notes.liquidation`` (mode-on) and ``notes.leverage`` (a
        mode-off run at a non-default leverage) are authoritative: a promoted or
        replayed child must run at the same leverage the champion was found under
        (vibe-quant-6wq6w). The leverage comes from ``notes.liquidation.leverage``
        when present, else ``notes.leverage``, else the default. A run that predates
        both ran off at the default 10, so the child gets no liquidation keys and
        keeps its stored parameters unchanged. Notes that cannot be parsed, or that
        carry two disagreeing leverages, fail closed (409) instead of silently
        reverting to the default.

        Returns:
            The liquidation run parameters (``{}`` when the parent ran off at 10).
        """
        from vibe_quant.liquidation_settings import (
            DEFAULT_LEVERAGE,
            MISSING,
            LiquidationSettings,
            resolve_leverage,
        )

        cell = self.grid_cell(strategy_index)
        if cell is not None:
            return cell.liquidation.to_params()
        payload = self.payload
        raw = payload.get("liquidation")
        # An explicit null is a value (rejected below), not an absent key.
        raw_leverage = payload.get("leverage", MISSING)
        if raw is None and raw_leverage is MISSING:
            return {}
        try:
            if raw is None:
                params: dict[str, object] = {"leverage": raw_leverage}
            else:
                if not isinstance(raw, dict):
                    raise ValueError(f"expected an object, got {raw!r}")
                note_leverage = raw.get("leverage", MISSING)
                if note_leverage is MISSING and raw_leverage is MISSING:
                    note_leverage = DEFAULT_LEVERAGE
                params = {
                    "liquidation_mode": raw.get("mode"),
                    "leverage": resolve_leverage(note_leverage, raw_leverage),
                }
            settings = LiquidationSettings.from_params(params)
        except (ValueError, TypeError) as exc:
            raise RecordUnreadable(
                f"discovery run {self.run_id} has an unreadable notes.liquidation: {exc}"
            ) from exc
        return settings.to_params()

    def sizing_params(self, *, strategy_index: int | None = None) -> dict[str, float]:
        """Sizing run parameters for a child of this run (S23-S23d).

        The parent's ``notes.sizing`` is authoritative: a promoted or replayed child must
        be sized exactly as the champion was found. A run that predates the key (no
        ``sizing`` key in its notes) ran at the defaults, so the child gets no sizing keys
        (S23c). A present-but-unreadable value -- null, not an object, an invalid number, or a
        cap the child's inherited leverage cannot cover -- fails closed (409) instead of
        silently reverting to the default (S23b/S23d).

        A grid run adds its champion's margin cushion (``strategy_index``) to the run's cap
        and risk, and checks the cap at the CHAMPION's leverage.

        Returns:
            ``SizingSettings.to_params()`` of the parent (``{}`` when the parent ran at
            the defaults).
        """
        from vibe_quant.liquidation_settings import DEFAULT_LEVERAGE
        from vibe_quant.sizing_settings import SizingSettings

        payload = self.payload
        cell = self.grid_cell(strategy_index)
        if cell is None and "sizing" not in payload:
            return {}
        # present: null / non-object / invalid fails closed below (a grid run may omit it = defaults)
        raw = payload.get("sizing", {})
        try:
            if not isinstance(raw, dict):
                raise ValueError(f"expected an object, got {raw!r}")
            settings = SizingSettings.from_params(raw)
            # The child runs at the parent's inherited leverage; a cap above it would
            # be unmarginable (S13/S23d). Resolved here from the same notes the child's
            # liquidation params come from.
            if cell is not None:
                settings = cell.sizing(settings)
                leverage: object = cell.leverage
            else:
                leverage = self.liquidation_params().get("leverage", DEFAULT_LEVERAGE)
            settings.check_leverage(float(leverage))  # type: ignore[arg-type]
        except (ValueError, TypeError) as exc:
            raise RecordUnreadable(
                f"discovery run {self.run_id} has an unreadable notes.sizing: {exc}"
            ) from exc
        return settings.to_params()

    def child_params(self, strategy_index: int) -> ChildParams:
        """Everything a child inherits from this run: fee, THEN liquidation, THEN sizing."""
        fee, fee_source = self.fee_params()
        return ChildParams(
            fee=fee,
            fee_source=fee_source,
            liquidation=self.liquidation_params(strategy_index=strategy_index),
            sizing=self.sizing_params(strategy_index=strategy_index),
        )


# ---------------------------------------------------------------------------
# Prior-trial scan: the cumulative DSR trial count of earlier runs
# ---------------------------------------------------------------------------


def iso_day(value: object) -> str | None:
    """The ``YYYY-MM-DD`` of a date-ish value; None when unusable."""
    if not isinstance(value, str) or not value:
        return None
    day = value[:10]
    return day if len(day) == 10 else None


def run_train_window(
    notes: dict[str, object],
    start_date: object,
    end_date: object,
) -> tuple[str, str] | None:
    """A prior run's train window: ``notes.train_dates`` when present, else run dates.

    Returns None (skip the row) when the window is missing or garbled.
    """
    if notes.get("train_dates"):
        train_dates = notes["train_dates"]
        if not isinstance(train_dates, (list, tuple)) or len(train_dates) != 2:
            return None
        start, end = iso_day(train_dates[0]), iso_day(train_dates[1])
    else:
        start, end = iso_day(start_date), iso_day(end_date)
    if start is None or end is None:
        return None
    return start, end


def prior_discovery_trials(
    state: StateManager,
    run_id: int,
    timeframe: str,
    train_window: tuple[str, str],
) -> tuple[int | None, list[int], int, int]:
    """Cumulative prior trial count + contributing run ids (bd vibe-quant-megq1, A3).

    Walks earlier (``b.id < run_id``) non-mock discovery runs with the SAME
    timeframe whose train window overlaps ``train_window`` (counts only, no
    de-dup; symbols are not filtered) and charges this run for their EFFECTIVE
    trial counts:

    - A row whose ``notes.dsr_trials.n_rule`` is ``N_RULE_EFFECTIVE`` contributes
      its ``run_n_eff`` iff that is a plain int in ``1..evaluated``; a missing,
      null, bool, string, float, zero or over-evaluated value fails closed by
      contributing ``evaluated`` (and logs one warning naming the run).
    - Any other row is LEGACY (recorded before the switch): it is NOT charged,
      but is tallied in ``(legacy_runs, legacy_raw)`` so the note can name it.

    Returns ``(total, ids, legacy_runs, legacy_raw)``; ``(None, [], 0, 0)`` when
    the DB lookup fails or ``train_window`` is unusable -- the caller fails open
    to the run-only N and records ``prior_error``.
    """
    this_start, this_end = iso_day(train_window[0]), iso_day(train_window[1])
    if this_start is None or this_end is None:
        # Unusable current window: cannot tell which runs overlap. Fail open
        # (like a DB error) but loudly -- the caller records prior_error.
        logger.warning("Unusable train window %r; cannot count prior trials", train_window)
        return None, [], 0, 0
    try:
        rows = state.list_prior_discovery_notes(run_id, timeframe)
    except Exception:
        logger.warning("Could not count prior discovery trials", exc_info=True)
        return None, [], 0, 0
    total = 0
    ids: set[int] = set()
    legacy_runs = 0
    legacy_raw = 0
    for row_id, raw, start_date, end_date in rows:
        try:
            notes = parse_notes(raw) if raw else {}
        except (TypeError, ValueError):
            logger.warning("prior trial count: run %s %s; skipped", row_id, "garbled notes")
            continue  # garbled notes -> skip the row
        if not isinstance(notes, dict) or notes.get("mock") or not notes.get("evaluated"):
            continue
        window = run_train_window(notes, start_date, end_date)
        if window is None:
            continue  # missing/garbled train window -> skip the row
        prior_start, prior_end = window
        if not (prior_start < this_end and this_start < prior_end):
            continue  # disjoint windows -> not a prior trial
        try:
            evaluated = int(notes["evaluated"])  # type: ignore[call-overload]
        except (TypeError, ValueError):
            logger.warning(
                "prior trial count: run %s %s; skipped",
                row_id,
                "garbled evaluated count",
            )
            continue  # garbled evaluated count -> skip only this row

        dsr = notes.get("dsr_trials")
        if not isinstance(dsr, dict) or dsr.get("n_rule") != N_RULE_EFFECTIVE:
            # Legacy (pre-switch) run: tallied, never charged (decision A3).
            legacy_runs += 1
            legacy_raw += evaluated
            continue

        run_n_eff = dsr.get("run_n_eff")
        if (
            isinstance(run_n_eff, int)
            and not isinstance(run_n_eff, bool)
            and 1 <= run_n_eff <= evaluated
        ):
            total += run_n_eff
        else:
            logger.warning(
                "prior trial count: run %s has an invalid run_n_eff %r; charging evaluated %d",
                row_id,
                run_n_eff,
                evaluated,
            )
            total += evaluated
        ids.add(int(row_id))
    return total, sorted(ids), legacy_runs, legacy_raw


def build_prior_trials_fn(
    state: StateManager,
    run_id: int,
    timeframe: str,
    *,
    mock: bool,
) -> tuple[Callable[[str, str], int | None] | None, list[int], dict[str, int]]:
    """``(DiscoveryConfig.prior_trials_fn, prior_run_ids, prior_legacy)`` for the DSR N.

    Mock runs get ``(None, [], {})`` (run-only N; their evaluations are not real
    trials). Otherwise the fn runs ``prior_discovery_trials`` for the train
    window the pipeline hands it and refreshes the SHARED ``prior_run_ids`` list
    (so ``notes.dsr_trials`` can name the contributing runs) and the SHARED
    ``prior_legacy`` dict (``{"runs", "raw"}``, bd vibe-quant-megq1).
    """
    prior_run_ids: list[int] = []
    prior_legacy: dict[str, int] = {"runs": 0, "raw": 0}
    if mock:
        return None, prior_run_ids, {}

    def prior_trials(start: str, end: str) -> int | None:
        n, ids, legacy_runs, legacy_raw = prior_discovery_trials(
            state, run_id, timeframe, (start, end)
        )
        prior_run_ids[:] = ids
        prior_legacy["runs"] = legacy_runs
        prior_legacy["raw"] = legacy_raw
        return n

    return prior_trials, prior_run_ids, prior_legacy
