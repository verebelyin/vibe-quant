"""Score ONE candidate on the REAL discovery gate chain and report the chain.

The power test asks whether a strategy with a known edge can pass the gates
discovery actually applies. This module answers it for a single candidate: every
gate is the repo's OWN function --

* ``guardrails.apply_guardrails`` for min-trades, min-return, complexity and the
  bootstrap t-stat CI;
* ``guardrails.apply_discovery_dsr_returns`` for the per-N deflated Sharpe;
* ``buyhold.buy_hold_gate`` for the buy-and-hold bar;
* ``DiscoveryPipeline._holdout_reasons`` for the holdout gate;
* ``DiscoveryPipeline._evaluate_cross_windows`` / ``_evaluate_wfa_rolling`` for
  the shifted cross-windows and the rolling-window stability check.

Nothing here re-implements gate or metric math. The chains are ANDs:
``chain_recent`` is the gate set a discovery champion must clear, plus the two
leave-out readings the verdicts need -- ``chain_recent_minus_bh`` (drop the
buy-and-hold bar) and ``chain_recent_minus_dsr_bh`` (drop DSR and the bar).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from vibe_quant.discovery.buyhold import buy_hold_gate
from vibe_quant.discovery.config_check import WindowPlan, plan_windows
from vibe_quant.discovery.gate_chain import metrics_to_holdout_result, symbol_mtm_stats
from vibe_quant.discovery.guardrails import (
    GuardrailConfig,
    apply_discovery_dsr_returns,
    apply_guardrails,
)
from vibe_quant.discovery.operators import MAX_SCALAR_GENES, scalar_gene_count
from vibe_quant.discovery.pipeline import (
    DiscoveryConfig,
    DiscoveryPipeline,
)
from vibe_quant.overfitting.dsr import DeflatedSharpeRatio
from vibe_quant.utils import compute_day_count

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from vibe_quant.discovery.fitness import FitnessResult
    from vibe_quant.discovery.operators import StrategyChromosome

#: DSR trial counts reported per candidate: the runs' own effective N (7/9/11/12),
#: the single-trial N = 1, and the production gate_n (4h 22121, 1h 3901, 15m 752, 1d 167).
DSR_TRIALS: tuple[int, ...] = (1, 7, 9, 11, 12, 167, 752, 3901, 22121)

#: The two plan windows the power test scores: (timeframe, full range start/end,
#: cross-window month offsets, WFA rolling step in days).
_WINDOWS: dict[str, tuple[str, str, str, tuple[int, ...], int]] = {
    "W3": ("4h", "2023-06-01", "2026-06-01", (1, 2), 125),
    "W6": ("1d", "2021-01-01", "2026-10-15", (), 120),
}

TRAIN_TEST_SPLIT: float = 0.8
EVAL_WINDOWS: int = 2


@dataclass(frozen=True, slots=True)
class ChainSettings:
    """One plan window's discovery config plus the derived window plan."""

    name: str
    config: DiscoveryConfig
    plan: WindowPlan
    symbols: tuple[str, ...]
    dsr_trials: tuple[int, ...] = DSR_TRIALS


def settings_for(name: str) -> ChainSettings:
    """The ``DiscoveryConfig`` / window plan of plan window ``W3`` or ``W6``.

    The windows themselves come from the real :func:`~vibe_quant.discovery.
    config_check.plan_windows`, so they are byte-identical to a CLI run's.
    """
    if name not in _WINDOWS:
        msg = f"unknown plan window {name!r}; known: {sorted(_WINDOWS)}"
        raise ValueError(msg)
    timeframe, full_start, full_end, cross_months, wfa_step = _WINDOWS[name]
    plan = plan_windows(
        full_start,
        full_end,
        train_test_split=TRAIN_TEST_SPLIT,
        eval_windows=EVAL_WINDOWS,
        cross_window_months=cross_months,
        wfa_oos_step_days=wfa_step,
    )
    if plan.holdout_start is None or plan.holdout_end is None:
        msg = f"{name}: no holdout split from {full_start}..{full_end}"
        raise ValueError(msg)
    from vibe_quant.discovery.run_spec import _default_bootstrap_min_sharpe

    config = DiscoveryConfig(
        population_size=12,
        max_generations=4,
        elite_count=2,
        top_k=5,
        min_trades=50,
        symbols=["BTCUSDT"],
        timeframe=timeframe,
        start_date=plan.train_start,
        end_date=plan.train_end,
        holdout_start_date=plan.holdout_start,
        holdout_end_date=plan.holdout_end,
        train_test_split=TRAIN_TEST_SPLIT,
        eval_windows=EVAL_WINDOWS,
        cross_window_months=list(cross_months),
        wfa_oos_step_days=wfa_step,
        require_dsr=True,
        dsr_rule="emax",  # the DiscoveryConfig default is q95; research stays on the legacy rule
        require_bootstrap_ci=True,
        bootstrap_min_sharpe=_default_bootstrap_min_sharpe(timeframe),
    )
    return ChainSettings(name=name, config=config, plan=plan, symbols=("BTCUSDT",))


def dsr_verdict(
    daily_returns: Sequence[float] | None,
    num_trials: int,
    *,
    null_spread: float | None = None,
) -> tuple[bool, dict[str, object]]:
    """``(passed, dsr_record)`` of one dense daily series at ``num_trials``.

    Thin wrapper over the repo's :func:`apply_discovery_dsr_returns`: passed iff
    the real function reports no rejection reason. ``null_spread`` is the
    opt-in q95 rule's spread (design-dsr-q95); None keeps the legacy bar.
    """
    _result, record, reasons = apply_discovery_dsr_returns(
        daily_returns, num_trials, null_spread=null_spread
    )
    return (not reasons), record


def posthoc_dsr_p(record: Mapping[str, object], num_trials: int) -> float:
    """DSR p-value of a stored ``dsr_record`` re-read at ``num_trials``.

    Uses the record's own moment inputs (``sharpe_daily`` / ``observations`` /
    ``skewness_used`` / ``kurtosis_used``) and the repo's
    :class:`DeflatedSharpeRatio`, so it reproduces the gate's p at that N.
    """
    return DeflatedSharpeRatio(0.05).calculate(
        observed_sharpe=cast("float", record["sharpe_daily"]),
        num_trials=num_trials,
        num_observations=cast("int", record["observations"]),
        skewness=cast("float", record["skewness_used"]),
        kurtosis=cast("float", record["kurtosis_used"]),
    ).p_value


def _null_backtest(_chrom: StrategyChromosome) -> dict[str, float | int]:
    """Stand-in train fn: this module never runs a train backtest."""
    return {}


def _min_return_passed(fitness: FitnessResult) -> bool:
    """The repo's min-return gate, isolated from every other guardrail.

    ``GuardrailResult`` exposes the min-trades/bootstrap/DSR verdicts but not the
    min-return one, so the real ``apply_guardrails`` is re-run with the other
    gates neutralised (min_trades 0, num_genes 0 so the complexity guard cannot
    bind, bootstrap/DSR off). Its ``passed`` is then exactly
    ``fitness.total_return >= GuardrailConfig.min_return``.
    """
    return apply_guardrails(
        fitness=fitness,
        num_genes=0,
        config=GuardrailConfig(
            min_trades=0,
            require_dsr=False,
            require_wfa=False,
            require_purged_kfold=False,
            require_bootstrap_ci=False,
        ),
    ).passed


def evaluate_candidate(
    chrom: StrategyChromosome,
    fitness: FitnessResult,
    holdout_bt: Mapping[str, float | int],
    window_bt: Callable[[str, str], Callable[[StrategyChromosome], dict[str, float | int]]],
    closes: Mapping[str, Mapping[int, float]],
    settings: ChainSettings,
    *,
    null_spread: float | None = None,
) -> dict[str, object]:
    """Every gate verdict for one candidate, plus the per-N chain readings.

    ``holdout_bt`` is the candidate's holdout window metrics; ``window_bt`` the
    cross-window / WFA backtest factory (same shape as the pipeline's). Returns
    ``floor`` (the fitness floor ``adjusted_score > 0``), the per-gate
    ``marginals``, the per-N ``dsr`` verdicts, and the chains ``chain_recent``,
    ``chain_recent_minus_bh``, ``chain_recent_minus_dsr_bh`` and ``chain_full``.
    """
    cfg = settings.config
    pipeline = DiscoveryPipeline(cfg, backtest_fn=_null_backtest, backtest_fn_factory=window_bt)
    num_genes = scalar_gene_count(chrom)

    series = fitness.mtm_daily_returns
    values = list(series.values) if series is not None else None
    day_count = compute_day_count(cfg.start_date, cfg.end_date)
    num_obs = day_count if day_count else max(100, fitness.total_trades * 5)
    trade_ret = np.array(fitness.trade_returns) if fitness.trade_returns else None

    guardrails = apply_guardrails(
        fitness=fitness,
        num_genes=num_genes,
        config=GuardrailConfig(
            min_trades=cfg.min_trades,
            max_complexity=MAX_SCALAR_GENES,
            require_dsr=False,
            require_wfa=False,
            require_purged_kfold=False,
            require_bootstrap_ci=cfg.require_bootstrap_ci,
            bootstrap_min_sharpe=cfg.bootstrap_min_sharpe,
            bootstrap_ci_level=cfg.bootstrap_ci_level,
        ),
        num_trials=1,
        num_observations=num_obs,
        skewness=fitness.skewness,
        kurtosis=fitness.kurtosis,
        trade_returns=trade_ret,
        daily_returns=values,
        symbol_daily=symbol_mtm_stats(fitness.symbol_mtm_daily),
        dsr_requires_returns=True,
    )

    dsr: dict[int, bool] = {}
    dsr_records: dict[int, dict[str, object]] = {}
    for num_trials in settings.dsr_trials:
        passed, record = dsr_verdict(values, num_trials, null_spread=null_spread)
        dsr[num_trials] = passed
        dsr_records[num_trials] = record

    buy_passed, buy_record, _buy_reasons = buy_hold_gate(
        series, chrom.direction.value, closes, list(settings.symbols)
    )
    holdout = metrics_to_holdout_result(dict(holdout_bt))
    holdout_passed = not pipeline._holdout_reasons(holdout, pipeline.holdout_min_trades())

    if cfg.cross_window_months:
        cw_results, cw_error = pipeline._evaluate_cross_windows([(chrom, fitness)])
        cross_window = cw_error is None and bool(cw_results) and cw_results[0].passed
    else:
        cross_window = True
    if cfg.wfa_oos_step_days > 0:
        wfa_results, wfa_error = pipeline._evaluate_wfa_rolling(
            [(chrom, fitness)], cfg.start_date, cfg.end_date
        )
        wfa = wfa_error is None and bool(wfa_results) and wfa_results[0].passed
    else:
        wfa = True

    floor = fitness.adjusted_score > 0
    marginals: dict[str, bool] = {
        "min_trades": guardrails.min_trades_passed,
        "min_return": _min_return_passed(fitness),
        "complexity": guardrails.complexity_passed,
        "bootstrap": bool(guardrails.bootstrap_passed),
        "buy_hold": buy_passed,
        "holdout": holdout_passed,
        "cross_window": cross_window,
        "wfa": wfa,
    }

    def _recent(num_trials: int, *, buy_hold: bool = True) -> bool:
        return bool(
            floor
            and marginals["min_trades"]
            and marginals["min_return"]
            and marginals["bootstrap"]
            and dsr[num_trials]
            and (marginals["buy_hold"] if buy_hold else True)
            and marginals["holdout"]
        )

    chain_recent = {n: _recent(n) for n in settings.dsr_trials}
    return {
        "floor": floor,
        "marginals": marginals,
        "dsr": dsr,
        "dsr_records": dsr_records,
        "chain_recent": chain_recent,
        "chain_recent_minus_bh": {n: _recent(n, buy_hold=False) for n in settings.dsr_trials},
        "chain_recent_minus_dsr_bh": bool(
            floor
            and marginals["min_trades"]
            and marginals["min_return"]
            and marginals["bootstrap"]
            and marginals["holdout"]
        ),
        "chain_full": {
            n: chain_recent[n] and cross_window and wfa for n in settings.dsr_trials
        },
        "buy_hold_record": buy_record,
    }
