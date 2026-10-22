"""Champion gate vocabulary shared by the pipeline, power harness and routes.

Behaviour-preserving home (rabmo.6 T6) for the pieces that carry no gate state:
the window result types, the holdout / cross-window / WFA window builders, the
DSR input checks and the rejection record. ``vibe_quant.discovery.pipeline``
imports this module and re-exports the three result classes.

Runtime imports stay limited to config_check, genome and utils so that
``pipeline.py`` can import this module without a cycle; the config / fitness /
operator types are TYPE_CHECKING-only (the PEP 695 ``Candidate`` alias is lazy).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from vibe_quant.discovery.config_check import (
    cross_window_ranges_for,
    wfa_window_ranges_for,
)
from vibe_quant.discovery.genome import chromosome_to_dsl
from vibe_quant.discovery.guardrails import (
    GuardrailConfig,
    GuardrailResult,
    apply_guardrails,
)
from vibe_quant.discovery.operators import MAX_SCALAR_GENES, scalar_gene_count
from vibe_quant.errors import DataUnavailableError
from vibe_quant.utils import compute_day_count

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.discovery.operators import StrategyChromosome
    from vibe_quant.discovery.pipeline import DiscoveryConfig
    from vibe_quant.metrics import DailyReturns

logger = logging.getLogger(__name__)

type Candidate = tuple[StrategyChromosome, FitnessResult]

# ---------------------------------------------------------------------------
# Window results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    """Metrics of one strategy on one (holdout / shifted / rolling) window.

    Attributes:
        sharpe_ratio: Sharpe ratio.
        max_drawdown: Max drawdown.
        profit_factor: Profit factor.
        total_trades: Trade count.
        total_return: Total return.
    """

    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    total_return: float
    # symbol_agg="worst" only: portfolio-comparable trade sum and per-symbol
    # {sharpe, trades, return} (total_trades above is the MIN symbol).
    trades_sum: int | None = None
    symbol_metrics: dict[str, dict[str, float | int]] | None = None
    # Report-only liquidation count of this window (bead vibe-quant-2gyk1.1.9,
    # R5); None when the run's mode is off (the bt dict has no key).
    liquidation_count: int | None = None


@dataclass(frozen=True, slots=True)
class CrossWindowResult:
    """Cross-window validation result for a single strategy.

    Attributes:
        window_results: Per SHIFTED window metrics (the in-sample training
            window is excluded -- it never counts as a pass).
        windows_passed: Number of shifted windows that passed.
        total_windows: Number of shifted windows evaluated.
        passed: Whether ``windows_passed >= required`` shifted windows.
        window_dates: (start, end) per shifted window, parallel to results.
        offsets_months: Month offset per shifted window, parallel to results.
        required: Shifted windows that had to pass.
    """

    window_results: list[HoldoutResult]
    windows_passed: int
    total_windows: int
    passed: bool
    window_dates: list[tuple[str, str]] = field(default_factory=list)
    offsets_months: list[int] = field(default_factory=list)
    required: int = 0


@dataclass(frozen=True, slots=True)
class WFARollingResult:
    """Rolling-window stability check (over the TRAIN range) for one strategy.

    Attributes:
        oos_windows: Per-window HoldoutResult for each rolling window.
        window_dates: (start, end) for each rolling window.
        windows_profitable: Number of windows with total_return > 0.
        windows_sharpe_positive: Number of windows with sharpe > 0.
        total_windows: Total rolling windows.
        consistency: Fraction of profitable (return-based) windows — the gate.
        sharpe_consistency: Fraction of sharpe-positive windows — exposure only.
        passed: Whether consistency >= wfa_min_consistency (return-based).
    """

    oos_windows: list[HoldoutResult]
    window_dates: list[tuple[str, str]]
    windows_profitable: int
    windows_sharpe_positive: int
    total_windows: int
    consistency: float
    sharpe_consistency: float
    passed: bool


def metrics_to_holdout_result(bt: Mapping[str, float | int]) -> HoldoutResult:
    """Window metrics dict -> HoldoutResult, NaN coerced to failing values."""
    sharpe = float(bt.get("sharpe_ratio", 0.0))
    max_dd = float(bt.get("max_drawdown", 1.0))
    pf = float(bt.get("profit_factor", 0.0))
    trades = int(bt.get("total_trades", 0))
    ret = float(bt.get("total_return", 0.0))
    return HoldoutResult(
        sharpe_ratio=0.0 if math.isnan(sharpe) else sharpe,
        max_drawdown=1.0 if math.isnan(max_dd) else max_dd,
        profit_factor=0.0 if math.isnan(pf) else pf,
        total_trades=trades,
        total_return=0.0 if math.isnan(ret) else ret,
        trades_sum=int(bt["trades_sum"]) if "trades_sum" in bt else None,
        symbol_metrics=bt.get("symbol_metrics"),  # type: ignore[arg-type]
        liquidation_count=int(bt["liquidation_count"]) if "liquidation_count" in bt else None,
    )


FAILED_WINDOW = HoldoutResult(
    sharpe_ratio=0.0,
    max_drawdown=1.0,
    profit_factor=0.0,
    total_trades=0,
    total_return=0.0,
)

# ---------------------------------------------------------------------------
# Holdout / cross-window / WFA window planning
# ---------------------------------------------------------------------------


def symbol_mtm_stats(
    symbol_mtm_daily: dict[str, DailyReturns | None] | None,
) -> dict[str, dict[str, float] | None] | None:
    """Per-symbol DSR Sharpe inputs from each symbol's MARK-TO-MARKET series.

    One dict per symbol (sharpe / skewness / kurtosis / observations / mean
    via :func:`daily_sharpe_inputs`), built from ``symbol_mtm_daily`` so the worst-mode per-symbol PSR floor judges the
    SAME series the pooled test does (vibe-quant-y9ywf). A missing or empty
    symbol series maps to ``None`` so the floor fails closed; a ``None`` input
    (portfolio mode) yields ``None`` -- no per-symbol floor applies.
    """
    if symbol_mtm_daily is None:
        return None
    from vibe_quant.overfitting.dsr import daily_sharpe_inputs

    stats: dict[str, dict[str, float] | None] = {}
    for sym, series in symbol_mtm_daily.items():
        if series is None or len(series) == 0:
            stats[sym] = None
            continue
        sharpe, skewness, kurtosis = daily_sharpe_inputs(series.values)
        stats[sym] = {
            "sharpe": sharpe,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "observations": len(series),
            "mean": sum(series.values) / len(series),
        }
    return stats


def worst_symbol_gate_reasons(fitness: FitnessResult, min_trades: int) -> list[str]:
    """Hard train gate for ``symbol_agg="worst"`` champions (vibe-quant-ox73t).

    The soft GA score lets a genome with a losing symbol rank; a champion must
    still have EVERY symbol profitable (train, worst-of-windows) with at least
    ``min_trades`` trades. Empty list = pass (also for portfolio mode).
    """
    if fitness.symbol_stats is None:
        return []
    reasons: list[str] = []
    for sym, (ret, trades) in fitness.symbol_stats.items():
        if not ret > 0:
            reasons.append(f"worst-symbol train return <= 0 ({sym})")
        if trades < min_trades:
            reasons.append(f"worst-symbol train trades < {min_trades} ({sym})")
    return reasons


def holdout_reasons(hr: HoldoutResult, min_trades: int, min_sharpe: float) -> list[str]:
    """Failure reasons of the holdout gate (empty = pass)."""
    reasons: list[str] = []
    if hr.total_trades < min_trades:
        reasons.append(f"Holdout: {hr.total_trades} trades < {min_trades}")
    if not hr.total_return > 0:
        reasons.append(f"Holdout: return {hr.total_return:.2%} <= 0")
    if not hr.sharpe_ratio > min_sharpe:
        reasons.append(f"Holdout: sharpe {hr.sharpe_ratio:.2f} <= {min_sharpe:.2f}")
    return reasons


def holdout_min_trades(config: DiscoveryConfig) -> int:
    """Holdout trade floor (explicit, or half the train-gate trade rate)."""
    if config.holdout_min_trades is not None:
        return config.holdout_min_trades
    train_days = compute_day_count(config.start_date, config.end_date)
    holdout_days = compute_day_count(config.holdout_start_date, config.holdout_end_date)
    if not train_days or not holdout_days:
        return 1
    return max(1, int(config.min_trades * holdout_days / (2 * train_days)))


def cross_window_ranges(config: DiscoveryConfig) -> list[tuple[int, str, str]]:
    """Shifted cross-windows ``(offset_months, start, end)`` inside the TRAIN range.

    Window k starts ``offset_k`` months after the train start and ends at
    the train end -- shifting the END forward would run into the holdout
    (or past the data), so windows are clipped to the train range.

    Raises:
        ValueError: A window is shorter than ``_MIN_CROSS_WINDOW_DAYS`` or
            (defensively) overlaps the holdout.
    """
    return cross_window_ranges_for(
        config.cross_window_months,
        config.start_date,
        config.end_date,
        config.holdout_start_date if config.has_holdout else "",
    )


def wfa_window_ranges(
    config: DiscoveryConfig, range_start: str, range_end: str
) -> list[tuple[str, str]]:
    """Rolling ``wfa_oos_step_days`` windows tiling [range_start, range_end]."""
    return wfa_window_ranges_for(
        config.wfa_oos_step_days,
        range_start,
        range_end,
        config.holdout_start_date if config.has_holdout else "",
    )


def dsr_requires_returns(config: DiscoveryConfig, backtest_fn: object) -> bool:
    """Whether DSR tests the MARK-TO-MARKET series for this backtest fn."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    return isinstance(backtest_fn, NTBacktestFn) and config.require_dsr


# ---------------------------------------------------------------------------
# Rejection record
# ---------------------------------------------------------------------------


def rejection_record(
    chrom: StrategyChromosome,
    fitness: FitnessResult,
    stage: str,
    reasons: list[str],
    *,
    timeframe: str,
    dsr: dict[str, object] | None = None,
    buy_hold: dict[str, object] | None = None,
) -> dict[str, object]:
    """Record a rejected candidate (persisted with the run, shown in the UI)."""
    entry: dict[str, object] = {
        "uid": chrom.uid,
        "stage": stage,
        "score": round(fitness.adjusted_score, 4),
        "sharpe": round(fitness.sharpe_ratio, 3),
        "trades": fitness.total_trades,
        "reasons": list(reasons),
        # DSL kept so a near-miss can still be backtested/validated by hand.
        "dsl": {**chromosome_to_dsl(chrom), "timeframe": timeframe},
    }
    if dsr is not None:
        entry["dsr"] = dsr
    if buy_hold is not None:
        entry["buy_hold"] = buy_hold
    # Fee drag on the rejection row (bd vibe-quant-2gyk1.4): most runs end
    # with zero champions, so this is often the only place a user sees that
    # fees, not the signal, killed a high-turnover genome. Always written
    # (JSON null when unavailable).
    entry["fee_drag"] = fitness.fee_drag
    entry["fees_pct_equity_yr"] = fitness.fees_pct_equity_yr
    # Report-only liquidation counts (bead vibe-quant-2gyk1.1.9, R1): written
    # only when the mode is on, so rejection rows stay unchanged off (R4).
    if fitness.liquidation_count is not None:
        entry["liquidation_count"] = fitness.liquidation_count
    if fitness.liquidation_forfeited is not None:
        entry["liquidation_forfeited"] = fitness.liquidation_forfeited
    return entry


# ---------------------------------------------------------------------------
# Gate 1: guardrail screen and its rejection ledger
# ---------------------------------------------------------------------------

# Informational N_eff eigendecomposition (never gates) is O(m^3): cap the series
# count it runs on so a large retained pool cannot stall the run (bd yul7u.30).
_NEFF_MAX_SERIES: int = 500


@dataclass(frozen=True, slots=True)
class TrialBudget:
    """The trial counts one screen (run or pooled re-check) is judged under.

    Attributes:
        total_evaluated: Raw distinct strategies tried (this run, or the pooled
            multi-seed total).
        series_pool: Fitness rows whose retained MARK-TO-MARKET series feed the
            effective-N estimate. Read in ``sorted(str(key))`` order so the
            subsample (and therefore ``dsr_trials``) is insertion-order-free.
    """

    total_evaluated: int
    series_pool: Mapping[str, FitnessResult]


@dataclass(frozen=True, slots=True)
class GateEvidence:
    """Everything a gate needs to observe, fixed when the chain is built.

    Attributes:
        backtest_fn: The run's backtest callable (NT check for buy-hold +
            ``.symbols``).
        dsr_requires_returns: Whether DSR tests the MARK-TO-MARKET series
            (``dsr_requires_returns()``; explicit so tests can set it).
        holdout_backtest_fn: Backtest over the holdout range (final gate).
        backtest_fn_factory: ``(start, end) -> backtest_fn`` for cross-window /
            rolling windows (all inside the TRAIN range).
        buy_hold_closes_fn: Test seam loading ``{symbol: {day: close}}`` for the
            buy-and-hold bar.
    """

    backtest_fn: object
    dsr_requires_returns: bool
    holdout_backtest_fn: Callable[[StrategyChromosome], dict[str, float | int]] | None = None
    backtest_fn_factory: (
        Callable[[str, str], Callable[[StrategyChromosome], dict[str, float | int]]] | None
    ) = None
    buy_hold_closes_fn: Callable[[list[str]], dict[str, dict[int, float]]] | None = None


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """Every result of one validation ladder pass.

    Attributes:
        survivors: Candidates that passed every enabled gate, in input order.
        cross_window: Cross-window results of the passers, by uid.
        wfa: Rolling-WFA results of the passers, by uid.
        holdout: Holdout results of the final survivors, by uid.
        holdout_min_trades: Trade floor the holdout gate applied (None = no gate).
    """

    survivors: list[Candidate]
    cross_window: dict[str, CrossWindowResult]  # by uid, passers only
    wfa: dict[str, WFARollingResult]
    holdout: dict[str, HoldoutResult]
    holdout_min_trades: int | None


class GateChain:
    """One gate pass over candidates, owning its own rejection ledger.

    A chain is built per run (or per pooled re-check) and reads the config and
    evidence it was constructed with: callers never reassign a pipeline field
    afterwards. ``screen`` is Gate 1; ``validate`` runs the ladder
    (buy-hold -> cross-window -> WFA -> holdout). Both ledgers are what the run
    persists.
    """

    def __init__(self, config: DiscoveryConfig, evidence: GateEvidence) -> None:
        self.config = config
        self._evidence = evidence
        self._rejections: list[dict[str, object]] = []
        self._dsr_records: dict[str, dict[str, object]] = {}
        self._dsr_trials: dict[str, object] | None = None
        self._buy_hold_records: dict[str, dict[str, object]] = {}
        self._buy_hold_gate: dict[str, object] | None = None

    @property
    def rejections(self) -> list[dict[str, object]]:
        """Rejected candidates in the order the gates saw them."""
        return self._rejections

    @property
    def dsr_records(self) -> dict[str, dict[str, object]]:
        """DSR note per screened uid (empty when DSR is off)."""
        return self._dsr_records

    @property
    def dsr_trials(self) -> dict[str, object] | None:
        """Run-level DSR trial record (None when DSR is off)."""
        return self._dsr_trials

    @property
    def buy_hold_records(self) -> dict[str, dict[str, object]]:
        """Buy-and-hold record per checked uid (empty when the gate did not run)."""
        return self._buy_hold_records

    @property
    def buy_hold_gate(self) -> dict[str, object] | None:
        """Run-level buy-and-hold gate record (None when the gate is off)."""
        return self._buy_hold_gate

    def screen(self, candidates: list[Candidate], budget: TrialBudget) -> list[Candidate]:
        """Apply guardrails (DSR, complexity, min trades, bootstrap CI) to top strategies.

        FAILS CLOSED: returns only the candidates that pass every enabled
        guardrail -- an empty list when none do (vibe-quant-e70tl.5). Every
        rejected candidate is recorded in ``rejections`` with its reasons so
        the UI can explain an empty run.

        DSR uses theoretical variance (1/(T-1)) rather than empirical
        cross-strategy variance, which inflates SR₀ beyond what any
        strategy can achieve (see vibe-quant-fici).

        Args:
            candidates: Candidates to check.
            budget: The trial counts for this run (or pooled re-check).
        """
        guardrail_cfg = GuardrailConfig(
            min_trades=self.config.min_trades,
            max_complexity=MAX_SCALAR_GENES,
            require_dsr=self.config.require_dsr,
            dsr_rule=self.config.dsr_rule,
            require_wfa=False,  # WFA requires separate out-of-sample data
            require_purged_kfold=False,
            require_bootstrap_ci=self.config.require_bootstrap_ci,
            bootstrap_min_sharpe=self.config.bootstrap_min_sharpe,
            bootstrap_ci_level=self.config.bootstrap_ci_level,
        )

        # DSR observation count: NT Sharpe is annualized from daily returns,
        # and apply_discovery_dsr de-annualizes to daily units — so T must be
        # the day count of the window, not the bar count (vibe-quant-zmzzh).
        day_count = compute_day_count(self.config.start_date, self.config.end_date)

        # Cumulative trial count (bd vibe-quant-yul7u.28): the gate's N adds the
        # same-timeframe, overlapping-train prior runs' effective counts to this
        # run's. A missing/raising fn (or one returning None) fails open to the
        # run-only N and is recorded as prior_error.
        prior_n = 0
        prior_error = False
        prior_fn = self.config.prior_trials_fn
        if prior_fn is not None:
            try:
                prior_raw = prior_fn(self.config.start_date, self.config.end_date)
            except Exception:
                logger.warning("prior_trials_fn failed; using prior_n=0", exc_info=True)
                prior_raw = None
            if prior_raw is None:
                prior_error = True
            else:
                prior_n = max(0, int(prior_raw))

        # T3 (bd vibe-quant-megq1): the gate N is this run's EV-matched effective
        # trial count + prior_n, not the raw evaluated count. The estimator runs
        # BEFORE the loop (it reads the completed fitness cache) and only when
        # DSR is required; require_dsr False records no DSR note here (B1).
        trials: dict[str, object] | None = None
        null_spread: float | None = None
        gate_n = budget.total_evaluated + prior_n
        if self.config.require_dsr:
            trials = self.trial_counts(budget)
            gate_n = cast("int", trials["run_n_eff"]) + prior_n
            trials["dsr_rule"] = self.config.dsr_rule
            if self.config.dsr_rule == "q95":
                null_spread = cast("float | None", trials["neff_ev_q95_spread"])

        import numpy as np

        validated: list[Candidate] = []
        for chrom, fitness in candidates:
            gate_reasons = worst_symbol_gate_reasons(fitness, self.config.min_trades)
            if gate_reasons:
                self.reject(chrom, fitness, "worst_symbol_train", gate_reasons)
                continue
            num_genes = scalar_gene_count(chrom)
            num_obs = day_count if day_count else max(100, fitness.total_trades * 5)

            trade_ret = np.array(fitness.trade_returns) if fitness.trade_returns else None
            series = fitness.mtm_daily_returns
            result: GuardrailResult = apply_guardrails(
                fitness=fitness,
                num_genes=num_genes,
                config=guardrail_cfg,
                num_trials=max(1, gate_n),
                num_observations=num_obs,
                skewness=fitness.skewness,
                kurtosis=fitness.kurtosis,
                trade_returns=trade_ret,
                daily_returns=series.values if series is not None else None,
                symbol_daily=symbol_mtm_stats(fitness.symbol_mtm_daily),
                dsr_requires_returns=self._evidence.dsr_requires_returns,
                null_spread=null_spread,
                # trials_sharpe_variance intentionally omitted — use theoretical
                # 1/(T-1). Cross-strategy Sharpe dispersion from GA is NOT what
                # the paper's V[{SR_n}] measures (see vibe-quant-fici).
            )
            if result.dsr_record is not None:
                self._dsr_records[chrom.uid] = result.dsr_record
            if result.passed:
                validated.append((chrom, fitness))
                logger.info(
                    "Guardrail PASS: %s score=%.4f sharpe=%.2f trades=%d",
                    chrom.uid,
                    fitness.adjusted_score,
                    fitness.sharpe_ratio,
                    fitness.total_trades,
                )
            else:
                self.reject(
                    chrom, fitness, "guardrails", list(result.reasons), dsr=result.dsr_record
                )

        if trials is not None:
            trials["gate_n"] = gate_n
            trials["prior_n"] = None if prior_error else prior_n
            if prior_error:
                trials["prior_error"] = True
            self._dsr_trials = trials

        if not validated and candidates:
            logger.warning(
                "All %d top strategies failed guardrails — no champion is statistically "
                "significant (failing closed)",
                len(candidates),
            )
        else:
            logger.info("%d/%d top strategies passed guardrails", len(validated), len(candidates))
        return validated

    def pooled_recheck(self, candidates: list[Candidate], budget: TrialBudget) -> list[Candidate]:
        """Screen pooled champions, stamping the pooled stage and RAW pooled N.

        The pick is made across seeds, so the multiple-testing burden is the
        POOLED trial count: every rejection is re-staged as
        ``pooled_guardrails`` with ``Multi-seed pooled N=<raw>`` in front of
        each reason (the gate itself may have used a smaller effective N).
        """
        kept = self.screen(candidates, budget)
        for rejection in self._rejections:
            raw_reasons = rejection.get("reasons")
            reasons = raw_reasons if isinstance(raw_reasons, list) else []
            rejection["stage"] = "pooled_guardrails"
            rejection["reasons"] = [
                f"Multi-seed pooled N={budget.total_evaluated}: {reason}" for reason in reasons
            ]
        return kept

    def trial_counts(self, budget: TrialBudget) -> dict[str, object]:
        """Run-level DSR trial counts: the effective N (USED) + informational N_eff.

        The retained mark-to-market daily series (only train-gate-passing
        genomes keep one) are handed to the shared EV-matched estimator
        (:func:`vibe_quant.discovery.neff.effective_trials`), which returns the
        legacy informational estimates ``neff_mean_corr`` / ``neff_eig_pr``
        (recorded, never used by the gate — bd yul7u.24) plus the T1 keys
        ``neff_ev``, ``run_n_eff``, ``neff_ev_source`` and
        ``neff_fallback_reason``. ``run_n_eff`` IS the gate's run-side N since T3
        (``gate_n = run_n_eff + prior_n``).

        The series are in sorted uid order, the deterministic basis for the
        estimator's evenly-spaced subsample (bd yul7u.30). ``non_flat`` always
        stays the true count. The MC schedule constants are recorded so a
        past verdict can be reproduced.
        """
        import numpy as np

        from vibe_quant.discovery.neff import NEFF_EV_DRAWS, NEFF_EV_SEED, effective_trials

        # Sorted uid order: the deterministic basis for the subsample below.
        entries = sorted(budget.series_pool.items(), key=lambda kv: str(kv[0]))
        raw_rows = [
            np.frombuffer(fr.mtm_daily_returns.values, dtype=np.float64)  # type: ignore[union-attr]
            for _, fr in entries
            if fr.mtm_daily_returns is not None and len(fr.mtm_daily_returns) >= 2
        ]
        eff = effective_trials(raw_rows, budget.total_evaluated, max_series=_NEFF_MAX_SERIES)
        return {
            "raw": eff.raw,
            "non_flat": eff.non_flat,
            "neff_mean_corr": eff.neff_mean_corr,
            "neff_eig_pr": eff.neff_eig_pr,
            "neff_subsampled_from": eff.subsampled_from,
            "neff_ev": eff.neff_ev,
            "neff_ev_q95_spread": eff.ev_q95_spread,
            "run_n_eff": eff.run_n_eff,
            "neff_ev_source": eff.source,
            "neff_fallback_reason": eff.fallback_reason,
            "neff_ev_draws": NEFF_EV_DRAWS,
            "neff_ev_seed": NEFF_EV_SEED,
        }

    def validate(self, candidates: list[Candidate]) -> ValidationOutcome:
        """Run the validation ladder: buy-hold -> cross-window -> WFA -> holdout.

        Each gate only sees the previous gate's survivors and rejects (never
        keeps) failures. The holdout runs last and exactly once per survivor.
        """
        cfg = self.config
        survivors = list(candidates)
        cw_by_uid: dict[str, CrossWindowResult] = {}
        wfa_by_uid: dict[str, WFARollingResult] = {}
        holdout_by_uid: dict[str, HoldoutResult] = {}

        # Gate 1b: fail-closed buy-and-hold (bd vibe-quant-wcn55). Runs BEFORE
        # cross-window: every gate only rejects, so the champion set is
        # order-invariant and the holdout stays the final gate.
        if cfg.require_buy_hold:
            survivors = self.apply_buy_hold(survivors)

        # Gate 2: cross-window (shifted sub-windows of the train range)
        if cfg.cross_window_months:
            if not survivors:
                logger.warning("Cross-window requested but no strategies survived guardrails")
            elif self._evidence.backtest_fn_factory is None:
                logger.warning(
                    "Cross-window requested (months=%s) but cannot run: no backtest_fn_factory "
                    "configured — failing closed",
                    cfg.cross_window_months,
                )
                for chrom, fit in survivors:
                    self.reject(chrom, fit, "cross_window", ["Cross-window: no backtest factory"])
                survivors = []
            else:
                results, error = self.evaluate_cross_windows(survivors)
                kept: list[Candidate] = []
                if error is not None:
                    for chrom, fit in survivors:
                        self.reject(chrom, fit, "cross_window", [error])
                for (chrom, fit), cwr in zip(survivors, results, strict=False):
                    if cwr.passed:
                        cw_by_uid[chrom.uid] = cwr
                        kept.append((chrom, fit))
                    else:
                        self.reject(
                            chrom, fit, "cross_window",
                            [
                                f"Cross-window: {cwr.windows_passed}/{cwr.total_windows} "
                                f"shifted windows passed (need {cwr.required}; "
                                f"sharpe>={cfg.cross_window_min_sharpe} and return>0)"
                            ],
                        )
                survivors = kept

        # Gate 3: rolling-window stability (WFA) over the train range
        if cfg.wfa_oos_step_days > 0:
            skip_reason: str | None = None
            if not survivors:
                skip_reason = "no top strategies survived guardrails"
            elif self._evidence.backtest_fn_factory is None:
                skip_reason = "no backtest_fn_factory configured"

            if skip_reason is not None:
                logger.warning(
                    "WFA requested (wfa_oos_step_days=%d) but not run: %s",
                    cfg.wfa_oos_step_days, skip_reason,
                )
                for chrom, fit in survivors:  # fail closed
                    self.reject(chrom, fit, "wfa", [f"WFA: {skip_reason}"])
                survivors = []
            else:
                wfa_results, error = self.evaluate_wfa_rolling(
                    survivors, cfg.start_date, cfg.end_date,
                )
                kept = []
                if error is not None:
                    for chrom, fit in survivors:
                        self.reject(chrom, fit, "wfa", [error])
                else:
                    for (chrom, fit), wfa in zip(survivors, wfa_results, strict=True):
                        if wfa.passed:
                            wfa_by_uid[chrom.uid] = wfa
                            kept.append((chrom, fit))
                        else:
                            self.reject(
                                chrom, fit, "wfa",
                                [
                                    f"WFA: {wfa.windows_profitable}/{wfa.total_windows} rolling "
                                    f"windows profitable ({wfa.consistency:.0%} < "
                                    f"{cfg.wfa_min_consistency:.0%})"
                                ],
                            )
                survivors = kept

        # Gate 4: holdout -- the single final out-of-sample pass/fail
        holdout_floor: int | None = holdout_min_trades(cfg) if cfg.has_holdout else None
        if cfg.train_test_split > 0:
            if not cfg.has_holdout:
                logger.warning(
                    "train_test_split=%.2f but no holdout dates — failing closed",
                    cfg.train_test_split,
                )
                for chrom, fit in survivors:
                    self.reject(chrom, fit, "holdout", ["Holdout: no holdout dates"])
                survivors = []
            elif self._evidence.holdout_backtest_fn is None:
                logger.warning("Holdout configured but no holdout_backtest_fn — failing closed")
                for chrom, fit in survivors:
                    self.reject(chrom, fit, "holdout", ["Holdout: no holdout backtest"])
                survivors = []
            elif survivors and holdout_floor is not None:
                holdout_results = self.evaluate_holdout(survivors)
                kept = []
                for (chrom, fit), hr in zip(survivors, holdout_results, strict=True):
                    reasons = holdout_reasons(hr, holdout_floor, cfg.holdout_min_sharpe)
                    if reasons:
                        self.reject(chrom, fit, "holdout", reasons)
                    else:
                        holdout_by_uid[chrom.uid] = hr
                        kept.append((chrom, fit))
                logger.info(
                    "  Holdout gate: %d/%d passed (min_trades=%d, sharpe>%.2f, return>0)",
                    len(kept), len(survivors), holdout_floor, cfg.holdout_min_sharpe,
                )
                survivors = kept

        if not survivors:
            logger.warning(
                "No champion passed every gate — run persists 0 champions (%d rejections)",
                len(self._rejections),
            )

        return ValidationOutcome(
            survivors=survivors,
            cross_window=cw_by_uid,
            wfa=wfa_by_uid,
            holdout=holdout_by_uid,
            holdout_min_trades=holdout_floor,
        )

    def reject(
        self,
        chrom: StrategyChromosome,
        fitness: FitnessResult,
        stage: str,
        reasons: list[str],
        dsr: dict[str, object] | None = None,
        buy_hold: dict[str, object] | None = None,
    ) -> None:
        """Record a rejected candidate (persisted with the run, shown in the UI)."""
        entry = rejection_record(
            chrom,
            fitness,
            stage,
            reasons,
            timeframe=self.config.timeframe,
            dsr=dsr,
            buy_hold=buy_hold,
        )
        self._rejections.append(entry)
        logger.info("Gate FAIL [%s]: %s reasons=%s", stage, chrom.uid, reasons)

    def apply_buy_hold(self, candidates: list[Candidate]) -> list[Candidate]:
        """Gate 1b: reject candidates that do not beat buy-and-hold. Fails closed.

        A non-NT backtest fn cannot produce a market-valued daily series, so the
        gate records ``{"enforced": False, "reason": "non-NT backtest fn"}`` and
        skips. An NT fn without an injected ``buy_hold_closes_fn`` loads the
        catalog daily closes itself (once, in the parent). ANY load exception
        rejects EVERY survivor (stage ``"buy_hold"``): a champion whose benchmark
        cannot be built is never promoted.
        """
        from vibe_quant.discovery.backtest_fn import NTBacktestFn

        fn = self._evidence.backtest_fn
        if not isinstance(fn, NTBacktestFn):
            self._buy_hold_gate = {"enforced": False, "reason": "non-NT backtest fn"}
            logger.info("Buy-hold gate skipped: non-NT backtest fn")
            return candidates

        if not candidates:
            self._buy_hold_gate = {"enforced": True}
            return candidates

        symbols = list(fn.symbols)
        try:
            if self._evidence.buy_hold_closes_fn is not None:
                closes = self._evidence.buy_hold_closes_fn(symbols)
            else:
                from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH
                from vibe_quant.discovery.buyhold import load_daily_closes

                closes = load_daily_closes(symbols, DEFAULT_CATALOG_PATH)
        except Exception as exc:
            reason = f"Buy-hold: benchmark closes unavailable: {exc}"
            logger.warning("Buy-hold gate failing closed on %d candidates: %s", len(candidates), exc)
            for chrom, fit in candidates:
                self.reject(chrom, fit, "buy_hold", [reason])
            self._buy_hold_gate = {"enforced": True, "error": str(exc)}
            return []

        from vibe_quant.discovery.buyhold import buy_hold_gate

        per_symbol = self.config.symbol_agg == "worst"
        kept: list[Candidate] = []
        for chrom, fit in candidates:
            direction = chrom.direction.value
            if per_symbol:
                passed, record, reasons = buy_hold_gate(
                    fit.mtm_daily_returns,
                    direction,
                    closes,
                    symbols,
                    per_symbol=fit.symbol_mtm_daily if fit.symbol_mtm_daily is not None else {},
                )
            else:
                passed, record, reasons = buy_hold_gate(
                    fit.mtm_daily_returns, direction, closes, symbols
                )
            self._buy_hold_records[chrom.uid] = record
            if passed:
                kept.append((chrom, fit))
            else:
                self.reject(chrom, fit, "buy_hold", reasons, buy_hold=record)
        self._buy_hold_gate = {"enforced": True}
        logger.info("Buy-hold gate: %d/%d passed", len(kept), len(candidates))
        return kept

    def evaluate_holdout(self, candidates: list[Candidate]) -> list[HoldoutResult]:
        """Evaluate strategies on the holdout (out-of-sample) period.

        Returns HoldoutResult for each strategy, parallel to ``candidates``.
        A failed backtest yields failing metrics (fails the gate).
        """
        assert self._evidence.holdout_backtest_fn is not None
        holdout_fn = self._evidence.holdout_backtest_fn
        results: list[HoldoutResult] = []

        logger.info("=== HOLDOUT EVALUATION: %d strategies ===", len(candidates))

        for rank, (chrom, train_fit) in enumerate(candidates, 1):
            try:
                hr = metrics_to_holdout_result(holdout_fn(chrom))
            except DataUnavailableError:
                raise
            except Exception:
                logger.warning("Holdout eval failed for %s", chrom.uid, exc_info=True)
                hr = FAILED_WINDOW
            results.append(hr)

            # Log train vs holdout comparison
            logger.info(
                "  #%d %s: TRAIN sharpe=%.2f dd=%.1f%% ret=%.1f%% trades=%d → "
                "HOLDOUT sharpe=%.2f dd=%.1f%% ret=%.1f%% trades=%d",
                rank, chrom.uid,
                train_fit.sharpe_ratio, train_fit.max_drawdown * 100,
                train_fit.total_return * 100, train_fit.total_trades,
                hr.sharpe_ratio, hr.max_drawdown * 100,
                hr.total_return * 100, hr.total_trades,
            )

        # Summary: how much did strategies degrade on holdout?
        if results:
            train_sharpes = [f.sharpe_ratio for _, f in candidates]
            holdout_sharpes = [h.sharpe_ratio for h in results]
            avg_train = sum(train_sharpes) / len(train_sharpes)
            avg_holdout = sum(holdout_sharpes) / len(holdout_sharpes)
            degradation = (avg_train - avg_holdout) / avg_train * 100 if avg_train > 0 else 0
            logger.info(
                "  Holdout summary: avg_train_sharpe=%.2f → avg_holdout_sharpe=%.2f "
                "(%.1f%% degradation)",
                avg_train, avg_holdout, degradation,
            )

        return results

    def evaluate_cross_windows(
        self, candidates: list[Candidate]
    ) -> tuple[list[CrossWindowResult], str | None]:
        """Evaluate strategies on shifted sub-windows of the train range.

        Only the SHIFTED windows count -- the full train window is where the
        GA selected the strategy, so it is never evidence of robustness.

        Returns:
            (cross_window_results parallel to ``candidates``, error). ``error``
            is set (and results empty) when the windows can't be built.
        """
        assert self._evidence.backtest_fn_factory is not None
        cfg = self.config
        min_sharpe = cfg.cross_window_min_sharpe

        try:
            windows = cross_window_ranges(cfg)
        except ValueError as exc:
            logger.warning("%s — failing closed", exc)
            return [], str(exc)

        required = len(windows)
        if cfg.cross_window_min_pass is not None:
            required = min(cfg.cross_window_min_pass, len(windows))

        logger.info(
            "=== CROSS-WINDOW VALIDATION: %d strategies × %d shifted windows (need %d) ===",
            len(candidates), len(windows), required,
        )
        for months, ws, we in windows:
            logger.info("  Window +%dmo: %s → %s", months, ws, we)

        cross_results: list[CrossWindowResult] = []
        for rank, (chrom, _train_fit) in enumerate(candidates, 1):
            window_hrs: list[HoldoutResult] = []
            passes = 0
            for months, ws, we in windows:
                try:
                    hr = metrics_to_holdout_result(self._evidence.backtest_fn_factory(ws, we)(chrom))
                except DataUnavailableError:
                    raise
                except Exception:
                    logger.warning(
                        "Cross-window eval failed: %s window +%dmo", chrom.uid, months,
                        exc_info=True,
                    )
                    hr = FAILED_WINDOW
                window_hrs.append(hr)
                if hr.total_return > 0 and hr.sharpe_ratio >= min_sharpe:
                    passes += 1

            passed = passes >= required
            cross_results.append(
                CrossWindowResult(
                    window_results=window_hrs,
                    windows_passed=passes,
                    total_windows=len(windows),
                    passed=passed,
                    window_dates=[(ws, we) for _, ws, we in windows],
                    offsets_months=[m for m, _, _ in windows],
                    required=required,
                )
            )

            window_strs = [
                f"W+{months}mo: sharpe={hr.sharpe_ratio:.2f} ret={hr.total_return*100:.1f}% "
                f"[{'PASS' if (hr.total_return > 0 and hr.sharpe_ratio >= min_sharpe) else 'FAIL'}]"
                for (months, _, _), hr in zip(windows, window_hrs, strict=True)
            ]
            logger.info(
                "  #%d %s: %s → %d/%d shifted windows %s",
                rank, chrom.uid,
                " | ".join(window_strs),
                passes, len(windows),
                "PROMOTED" if passed else "REJECTED",
            )

        logger.info(
            "  Cross-window summary: %d/%d strategies promoted",
            sum(1 for r in cross_results if r.passed), len(candidates),
        )
        return cross_results, None

    def evaluate_wfa_rolling(
        self, candidates: list[Candidate], range_start: str, range_end: str
    ) -> tuple[list[WFARollingResult], str | None]:
        """Rolling-window stability check over the TRAIN range.

        Tiles [range_start, range_end] with ``wfa_oos_step_days`` windows and
        requires ``wfa_min_consistency`` of them to be profitable. The windows
        are in-sample for the GA, so this measures stability over time, not
        out-of-sample skill (the holdout is the out-of-sample gate).

        Returns:
            (results parallel to ``candidates``, error). ``error`` is set when
            the range is too short for a single window.
        """
        assert self._evidence.backtest_fn_factory is not None
        cfg = self.config
        step = cfg.wfa_oos_step_days
        min_consistency = cfg.wfa_min_consistency

        try:
            windows = wfa_window_ranges(cfg, range_start, range_end)
        except ValueError as exc:
            logger.warning("%s — failing closed", exc)
            return [], str(exc)
        if not windows:
            error = f"WFA: range {range_start}..{range_end} too short for {step}d windows"
            logger.warning("%s — failing closed", error)
            return [], error

        logger.info(
            "=== WFA ROLLING VALIDATION: %d strategies × %d windows (%dd each, train range) ===",
            len(candidates), len(windows), step,
        )
        for i, (ws, we) in enumerate(windows):
            logger.info("  Window %d: %s → %s", i, ws, we)

        wfa_results: list[WFARollingResult] = []
        for rank, (chrom, _train_fit) in enumerate(candidates, 1):
            oos_results: list[HoldoutResult] = []
            profitable = 0
            sharpe_positive = 0

            for ws, we in windows:
                try:
                    hr = metrics_to_holdout_result(self._evidence.backtest_fn_factory(ws, we)(chrom))
                except DataUnavailableError:
                    raise
                except Exception:
                    logger.warning("WFA eval failed: %s", chrom.uid, exc_info=True)
                    hr = FAILED_WINDOW
                oos_results.append(hr)
                if hr.total_return > 0:
                    profitable += 1
                if hr.sharpe_ratio > 0:
                    sharpe_positive += 1

            consistency = profitable / len(windows)
            sharpe_consistency = sharpe_positive / len(windows)
            passed = consistency >= min_consistency

            wfa_results.append(
                WFARollingResult(
                    oos_windows=oos_results,
                    window_dates=list(windows),
                    windows_profitable=profitable,
                    windows_sharpe_positive=sharpe_positive,
                    total_windows=len(windows),
                    consistency=consistency,
                    sharpe_consistency=sharpe_consistency,
                    passed=passed,
                )
            )

            w_strs = [
                f"W{i}: ret={hr.total_return*100:.1f}% {'OK' if hr.total_return > 0 else 'LOSS'}"
                for i, hr in enumerate(oos_results)
            ]
            logger.info(
                "  #%d %s: %s → %d/%d profitable (%.0f%%) | %d/%d sharpe+ (%.0f%%) %s",
                rank, chrom.uid, " | ".join(w_strs),
                profitable, len(windows), consistency * 100,
                sharpe_positive, len(windows), sharpe_consistency * 100,
                "PASS" if passed else "FAIL",
            )

        logger.info(
            "  WFA summary: %d/%d passed (min_consistency=%.0f%%)",
            sum(1 for r in wfa_results if r.passed), len(candidates), min_consistency * 100,
        )
        return wfa_results, None
