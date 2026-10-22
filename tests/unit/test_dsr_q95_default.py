"""T4 (vibe-quant-rf0y7): q95 is the production default, re-check uses the run's rule.

The quantile-matched DSR rule landed opt-in (``--dsr-rule q95``, design-dsr-q95).
T4 makes it the DEFAULT for discovery, and makes the promote-time overfitting
re-check of a champion apply the SAME rule the run used (it read only N before,
so a q95 champion that the legacy gate rejects would be dropped at promote).

D1 default flip; D2 the run records the rule; D3 the re-check reproduces the
discovery q95 verdict on the same MTM inputs; D4 old/emax/broken payloads fail
closed to the legacy decision byte-for-byte; D5 the re-check reject names q95.
"""

from __future__ import annotations

import json
import math
from array import array
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.gate_chain import TrialBudget
from vibe_quant.discovery.guardrails import (
    GuardrailConfig,
    apply_discovery_dsr_returns,
    apply_guardrails,
)
from vibe_quant.discovery.neff import N_RULE_EFFECTIVE
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline
from vibe_quant.discovery.run_spec import build_parser
from vibe_quant.metrics import DailyReturns
from vibe_quant.overfitting.pipeline import OverfittingPipeline
from vibe_quant.overfitting.types import FilterConfig

if TYPE_CHECKING:
    from pathlib import Path

#: ``notes.dsr_trials.n_rule`` a post-switch discovery run records.
DISC_N_RULE = N_RULE_EFFECTIVE
_FIRST_DAY = 19_000
#: Legacy verdict of the R7 inputs at N=88 (design-dsr-q95 R7).
_R7_LEGACY_P = 0.10996293959564483

#: A symmetric 875-day shape (400 +/-1, 37 +/-r, one 0) whose bias-corrected
#: kurtosis is exactly 7.7; scaling/centring leaves skew 0 and kurtosis intact.
_R7_R = 6.637640625033072
_R7_SHAPE = np.array(
    [1.0] * 400 + [-1.0] * 400 + [_R7_R] * 37 + [-_R7_R] * 37 + [0.0],
    dtype=np.float64,
)


def _r7_values() -> list[float]:
    """R7 inputs: daily SR 0.12611, T 875, g3 0, g4 7.7."""
    sigma = 0.01
    return list(_R7_SHAPE * sigma + 0.12611 * sigma * float(_R7_SHAPE.std(ddof=1)))


def _fitness(**kw: Any) -> Any:
    from vibe_quant.discovery.fitness import FitnessResult

    base: dict[str, Any] = {
        "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 200, "total_return": 0.3, "complexity_penalty": 0.0,
        "overtrade_penalty": 0.0, "sl_tp_penalty": 0.0, "raw_score": 0.5,
        "adjusted_score": 0.5, "passed_filters": True, "filter_results": {},
    }  # fmt: skip
    base.update(kw)
    return FitnessResult(**base)


def _chrom() -> Any:
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0, None)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
        direction=Direction.LONG,
    )


# ---------------------------------------------------------------------------
# Overfitting re-check fixture: a promoted champion with stored MTM inputs
# ---------------------------------------------------------------------------


def _r7_record() -> dict[str, Any]:
    return {
        "input": "pooled_mtm_daily_returns",
        "observations": 875,
        "sharpe_daily": 0.12611,
        "skewness": 0.0,
        "kurtosis": 7.7,
    }


def _discovery_db(
    tmp_path: Path, *, dsr_trials: dict[str, Any]
) -> tuple[Path, int]:
    db = tmp_path / "disc.db"
    sm = StateManager(db)
    rid = sm.create_backtest_run(
        None, "discovery", ["BTCUSDT"], "4h", "2022-01-01", "2026-03-17", {}
    )
    notes = {
        "type": "discovery", "evaluated": 88, "symbols": ["BTCUSDT"],
        "symbol_agg": "portfolio", "train_dates": ["2022-01-01", "2026-03-17"],
        "top_strategies": [{"sharpe": 2.0, "dsr": _r7_record()}],
        "dsr_trials": dsr_trials,
    }  # fmt: skip
    sm.save_backtest_result(
        rid,
        {
            "sharpe_ratio": 2.0, "total_return": 0.4, "max_drawdown": 0.1,
            "profit_factor": 1.5, "total_trades": 120, "skewness": 0.0,
            "kurtosis": 3.0, "notes": json.dumps(notes),
        },
    )  # fmt: skip
    sm.close()
    return db, rid


def _run_recheck(db: Path, rid: int) -> Any:
    p = OverfittingPipeline(db)
    try:
        res = p.run(run_id=rid, config=FilterConfig.dsr_only())
    finally:
        p.close()
    return res.candidates[0]


# ---------------------------------------------------------------------------
# D1: the default is q95; an explicit emax keeps the legacy path bit-identical
# ---------------------------------------------------------------------------


def test_default_is_q95_everywhere() -> None:
    """GuardrailConfig, DiscoveryConfig and the CLI parse no flag as q95."""
    assert GuardrailConfig().dsr_rule == "q95"
    assert DiscoveryConfig().dsr_rule == "q95"
    assert build_parser().parse_args(["--run-id", "1"]).dsr_rule == "q95"
    assert (
        build_parser().parse_args(["--run-id", "1", "--dsr-rule", "emax"]).dsr_rule
        == "emax"
    )
    # The API launch body has no DSR field: the CLI default governs.
    from vibe_quant.api.schemas.discovery import DiscoveryLaunchRequest

    assert "dsr_rule" not in DiscoveryLaunchRequest.model_fields
    assert "dsr_rule" not in DiscoveryLaunchRequest().model_dump()


def test_explicit_emax_is_the_bit_identical_legacy_path() -> None:
    """``--dsr-rule emax`` ignores a spread and reproduces the R7 legacy verdict."""
    values = _r7_values()
    cfg = GuardrailConfig(require_wfa=False, require_bootstrap_ci=False, dsr_rule="emax")
    fit = _fitness(
        mtm_daily_returns=DailyReturns(_FIRST_DAY, array("d", np.array(values).tobytes()))
    )
    with_spread = apply_guardrails(
        fit, 2, cfg, num_trials=88, daily_returns=values, null_spread=0.8206
    )
    without = apply_guardrails(fit, 2, cfg, num_trials=88, daily_returns=values)

    assert with_spread.dsr_record == without.dsr_record  # the spread is ignored
    assert with_spread.dsr_record is not None
    assert with_spread.dsr_record["p_value"] == pytest.approx(_R7_LEGACY_P, abs=1e-12)
    assert with_spread.dsr_record["null_spread"] is None
    assert with_spread.dsr_passed is False
    assert with_spread.reasons == [
        "DSR not significant: p=0.1100 >= 0.05 (pooled daily SR=2.002 ann, T=875, N=88)"
    ]


# ---------------------------------------------------------------------------
# D2: a run without the flag records notes.dsr_trials.dsr_rule == "q95"
# ---------------------------------------------------------------------------


def _gate_run(**cfg_over: Any) -> Any:
    base: dict[str, Any] = {
        "population_size": 6, "max_generations": 1, "elite_count": 1, "top_k": 5,
        "min_trades": 50, "max_workers": None, "timeframe": "4h",
        "start_date": "2022-06-01", "end_date": "2025-11-17",
        "require_bootstrap_ci": False, "require_dsr": True, "require_buy_hold": False,
    }  # fmt: skip
    base.update(cfg_over)
    pipe = DiscoveryPipeline(DiscoveryConfig(**base), backtest_fn=lambda chrom: {})
    row = np.sin(np.linspace(0.0, 20.0, 900)) + np.linspace(0.1, 1.0, 900)
    pool: dict[str, Any] = {}
    for i in range(3):
        pool[f"k{i:03d}"] = _fitness(
            mtm_daily_returns=DailyReturns(
                _FIRST_DAY, array("d", (row + i * 1e-6).tobytes())
            )
        )
    fit = _fitness(
        mtm_daily_returns=DailyReturns(_FIRST_DAY, array("d", row.tobytes()))
    )
    pipe.gates.screen([(_chrom(), fit)], TrialBudget(88, pool))
    return pipe


@pytest.mark.parametrize(("over", "expected"), [({}, "q95"), ({"dsr_rule": "emax"}, "emax")])
def test_run_records_the_rule_it_used(over: dict[str, Any], expected: str) -> None:
    pipe = _gate_run(**over)
    assert pipe.gates.dsr_trials is not None
    assert pipe.gates.dsr_trials["dsr_rule"] == expected


# ---------------------------------------------------------------------------
# D3: the re-check applies the discovery q95 decision on the same MTM inputs
# ---------------------------------------------------------------------------


def test_recheck_applies_the_q95_rule_the_run_used(tmp_path: Path) -> None:
    """R7: discovery passes at spread 0.8206; legacy rejects the same champion."""
    from vibe_quant.overfitting.dsr import daily_sharpe_inputs

    sharpe, skew, kurt = daily_sharpe_inputs(_r7_values())
    assert sharpe == pytest.approx(0.12611, abs=1e-9)
    assert skew == pytest.approx(0.0, abs=1e-12)
    assert kurt == pytest.approx(7.7, abs=1e-9)

    discovery_result, _rec, discovery_reasons = apply_discovery_dsr_returns(
        _r7_values(), 88, null_spread=0.8206
    )
    assert discovery_result is not None and discovery_reasons == []

    db, rid = _discovery_db(
        tmp_path,
        dsr_trials={
            "n_rule": DISC_N_RULE,
            "gate_n": 88,
            "dsr_rule": "q95",
            "neff_ev_q95_spread": 0.8206,
        },
    )
    cand = _run_recheck(db, rid)

    assert cand.dsr_result is not None
    # Same statistic as the discovery guardrail: the rule changes only the bar.
    assert cand.dsr_result.deflated_sharpe == pytest.approx(
        discovery_result.deflated_sharpe, abs=1e-12
    )
    assert cand.dsr_result.p_value == pytest.approx(_R7_LEGACY_P, abs=1e-12)
    assert cand.passed_dsr is True
    assert cand.dsr_fail_reason is None


# ---------------------------------------------------------------------------
# D4: old / emax / broken payloads fail closed to the legacy decision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dsr_trials",
    [
        # legacy payload: the run predates the rule (spread present but rule not)
        {"n_rule": DISC_N_RULE, "gate_n": 88, "neff_ev_q95_spread": 0.8206},
        # explicit emax
        {"n_rule": DISC_N_RULE, "gate_n": 88, "dsr_rule": "emax", "neff_ev_q95_spread": 0.8206},
        # q95 but no spread / null / non-numeric / bool -> legacy fail closed
        {"n_rule": DISC_N_RULE, "gate_n": 88, "dsr_rule": "q95"},
        {"n_rule": DISC_N_RULE, "gate_n": 88, "dsr_rule": "q95", "neff_ev_q95_spread": None},
        {"n_rule": DISC_N_RULE, "gate_n": 88, "dsr_rule": "q95", "neff_ev_q95_spread": "0.8206"},
        {"n_rule": DISC_N_RULE, "gate_n": 88, "dsr_rule": "q95", "neff_ev_q95_spread": True},
        # non-finite spread -> legacy
        {"n_rule": DISC_N_RULE, "gate_n": 88, "dsr_rule": "q95", "neff_ev_q95_spread": math.nan},
        {"n_rule": DISC_N_RULE, "gate_n": 88, "dsr_rule": "q95", "neff_ev_q95_spread": math.inf},
        # int too large for float() (OverflowError) -> legacy, not a crash
        {"n_rule": DISC_N_RULE, "gate_n": 88, "dsr_rule": "q95", "neff_ev_q95_spread": 10**400},
    ],
)
def test_recheck_without_usable_rule_is_legacy(
    tmp_path: Path, dsr_trials: dict[str, Any]
) -> None:
    db, rid = _discovery_db(tmp_path, dsr_trials=dsr_trials)
    cand = _run_recheck(db, rid)

    assert cand.dsr_result is not None
    assert cand.dsr_result.p_value == pytest.approx(_R7_LEGACY_P, abs=1e-12)
    assert cand.passed_dsr is False
    assert cand.dsr_fail_reason == "DSR not significant: p=0.1100 (T=875, N=88)"
    assert "q95" not in cand.dsr_fail_reason


# ---------------------------------------------------------------------------
# D5: the re-check reject names the q95 rule, with the same text as discovery
# ---------------------------------------------------------------------------


def test_recheck_q95_reject_uses_the_shared_text(tmp_path: Path) -> None:
    db, rid = _discovery_db(
        tmp_path,
        dsr_trials={
            "n_rule": DISC_N_RULE,
            "gate_n": 88,
            "dsr_rule": "q95",
            "neff_ev_q95_spread": 1.5275,  # R7: stat 1.2267 < spread -> reject
        },
    )
    cand = _run_recheck(db, rid)

    _result, _rec, discovery_reasons = apply_discovery_dsr_returns(
        _r7_values(), 88, null_spread=1.5275
    )
    expected = (
        "DSR q95 not significant: stat=1.2267 < spread=1.5275 "
        "(ann SR=2.002, T=875, N=88)"
    )
    assert discovery_reasons == [expected]
    assert cand.passed_dsr is False
    assert cand.dsr_fail_reason == expected
