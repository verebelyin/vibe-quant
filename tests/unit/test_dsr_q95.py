"""Quantile-matched DSR critical value, opt-in (design-dsr-q95.md, T1).

Today's DSR passes iff the PSR statistic ``(SR - SR0)/SE >= 1.6449`` (p < 0.05). The opt-in
``--dsr-rule q95`` / ``GuardrailConfig.dsr_rule="q95"`` replaces the fixed
single-trial bar with the run's own Monte-Carlo spread
``quantile(M, 0.95) - mean(M)``, ``M = row-wise max of z @ root`` over the SAME
draws as the EV-matched effective-N estimator (``neff.effective_trials``).

Every literal asserted here is the design's contract R1-R10 (arch-dsr measured
the spread table at seed 20261018, 4096 draws on exactly equicorrelated
matrices). The default remains ``emax``: flag OFF changes no number.
"""

from __future__ import annotations

import math
from array import array
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest

from vibe_quant.discovery.guardrails import (
    GuardrailConfig,
    apply_discovery_dsr_returns,
    apply_guardrails,
)
from vibe_quant.discovery.neff import EffectiveTrials, effective_trials
from vibe_quant.metrics import DailyReturns
from vibe_quant.overfitting.dsr import (
    DeflatedSharpeRatio,
    daily_sharpe_inputs,
)

LEGACY_SPREAD = 1.6448536269514722
_FIRST_DAY = 19_000


def _equicorr_rows(n: int, rho: float, days: int = 900) -> list[np.ndarray]:
    """``n`` rows whose sample correlation matrix is EXACTLY equicorrelated.

    Orthonormal columns (mean-centred QR) make the off-diagonals exactly
    ``rho`` and the diagonals exactly 1, so the MC spread is the design's
    contract value rather than a realisation-dependent estimate.
    """
    g = np.random.default_rng(0).standard_normal((days, n + 1))
    g -= g.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(g)
    common = q[:, 0]
    return [np.sqrt(rho) * common + np.sqrt(1.0 - rho) * q[:, i + 1] for i in range(n)]


def _gaussian_values(daily_sharpe: float, n: int = 875, seed: int = 11) -> list[float]:
    """Gaussian daily returns standardized to EXACTLY ``daily_sharpe`` per day."""
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    return list(z * 0.01 + daily_sharpe * 0.01)


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
# R1-R4: the spread on exactly equicorrelated matrices
# ---------------------------------------------------------------------------


def test_q95_spread_iid_88() -> None:
    """R1: iid 88 series x 900 days -> spread 0.8206, crit 3.3059."""
    eff = effective_trials(_equicorr_rows(88, 0.0), 88)
    assert eff.source == "ev_mc" and eff.fallback_reason is None
    assert eff.ev_q95_spread == pytest.approx(0.8206, abs=1e-4)
    assert eff.ev_q95_spread is not None
    crit = DeflatedSharpeRatio._expected_max_sharpe(88) + eff.ev_q95_spread
    assert crit == pytest.approx(3.3059, abs=1e-4)
    sidak = NormalDist().inv_cdf(0.95 ** (1.0 / 88.0))
    assert abs(crit - sidak) <= 0.07


@pytest.mark.parametrize(("rho", "expected"), [(0.5, 1.2613), (0.8, 1.5275)])
def test_q95_spread_equicorrelated_88(rho: float, expected: float) -> None:
    """R2/R3: equicorrelation rho 0.5 / 0.8 at n 88."""
    eff = effective_trials(_equicorr_rows(88, rho), 88)
    assert eff.ev_q95_spread == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(("n", "expected"), [(22, 0.9308), (134, 0.7893)])
def test_q95_spread_iid_n(n: int, expected: float) -> None:
    """R4: iid at n 22 and n 134."""
    eff = effective_trials(_equicorr_rows(n, 0.0), n)
    assert eff.ev_q95_spread == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# R5-R6: degenerate and fallback inputs
# ---------------------------------------------------------------------------


def _decisions(values: list[float], spread: float | None, n: int = 88) -> bool:
    _result, _record, reasons = apply_discovery_dsr_returns(
        values, n, null_spread=spread
    )
    return not reasons


def test_q95_identical_series_uses_legacy_spread() -> None:
    """R5: one distinct series -> spread == 1.6449, decision == emax."""
    row = np.sin(np.linspace(0.0, 20.0, 900)) + np.linspace(0.1, 1.0, 900)
    eff = effective_trials([row.copy() for _ in range(5)], 40)
    assert eff.n_distinct == 1
    assert eff.ev_q95_spread == LEGACY_SPREAD
    for daily_sharpe in (0.20, 0.01):
        values = _gaussian_values(daily_sharpe)
        assert _decisions(values, eff.ev_q95_spread) == _decisions(values, None)


def test_q95_fallback_is_legacy_emax() -> None:
    """R6: 1 series / short window -> None, so the legacy decision and text."""
    one = effective_trials([np.arange(900, dtype=np.float64)], 88)
    assert one.source == "fallback_raw" and one.ev_q95_spread is None
    short = effective_trials([np.arange(100.0) + i for i in range(3)], 88)
    assert short.fallback_reason == "too_short" and short.ev_q95_spread is None

    values = _gaussian_values(0.01)
    _result, record, reasons = apply_discovery_dsr_returns(values, 88)
    assert reasons and reasons[0].startswith("DSR not significant: p=")
    assert "q95" not in reasons[0]
    assert record["null_spread"] is None
    assert _decisions(values, short.ev_q95_spread) == _decisions(values, None)

    # A non-finite spread fails closed to the legacy rule too.
    for bad in (math.nan, math.inf, -math.inf):
        bad_result, bad_record, bad_reasons = apply_discovery_dsr_returns(
            values, 88, null_spread=bad
        )
        assert bad_result is not None and bad_record["null_spread"] is None
        assert bad_reasons and bad_reasons[0].startswith("DSR not significant: p=")
        assert bad_reasons == reasons


def _verdict(values: list[float], spread: float, n: int = 88) -> tuple[bool, list[str], Any]:
    _r, record, reasons = apply_discovery_dsr_returns(values, n, null_spread=spread)
    return not reasons, reasons, record


def _r7_values_at(daily_sharpe: float) -> list[float]:
    sigma = 0.01
    return list(_R7_SHAPE * sigma + daily_sharpe * sigma * float(_R7_SHAPE.std(ddof=1)))


def test_q95_f1_five_identical_series_pass_like_legacy() -> None:
    """R11/F1: 5 identical 875-day series (kurt 7.7), prior_n 87 -> gate_n 88.

    Daily SR is tuned so the legacy p-value sits just under 0.05 (0.0489 in the
    review's reproduction); q95 must accept it exactly like legacy.
    """
    lo, hi = 0.12, 0.16
    for _ in range(60):
        mid = (lo + hi) / 2
        _r, rec, reasons = apply_discovery_dsr_returns(_r7_values_at(mid), 88)
        lo, hi = (lo, mid) if not reasons else (mid, hi)
    values = _r7_values_at(hi)
    rows = [np.array(values) for _ in range(5)]
    eff = effective_trials(rows, 5)
    assert eff.n_distinct == 1 and eff.run_n_eff == 1
    assert eff.ev_q95_spread == LEGACY_SPREAD
    gate_n = eff.run_n_eff + 87
    assert gate_n == 88
    _r, rec, reasons = apply_discovery_dsr_returns(values, gate_n)
    assert reasons == [] and 0.0487 < rec["p_value"] < 0.05  # type: ignore[operator]
    assert _decisions(values, eff.ev_q95_spread, gate_n) is True
    q_result = apply_discovery_dsr_returns(values, gate_n, null_spread=eff.ev_q95_spread)
    assert q_result[1]["significant"] is True and q_result[1]["psr_stat"] == rec["psr_stat"]


@pytest.mark.parametrize("kurt_shape", [0, 1, 2])
def test_q95_legacy_spread_same_verdict_on_real_series(kurt_shape: int) -> None:
    """R11(a) through apply_discovery_dsr_returns on series of kurtosis ~3..70."""
    rng = np.random.default_rng(kurt_shape)
    for daily_sharpe in (0.02, 0.06, 0.10, 0.141, 0.2):
        for df in (100.0, 5.0, 2.5):
            z = rng.standard_t(df=df, size=875) if kurt_shape else rng.normal(size=875)
            z = (z - z.mean()) / z.std(ddof=1)
            values = list(z * 0.01 + daily_sharpe * 0.01)
            for n in (1, 40, 88, 300):
                legacy = _decisions(values, None, n)
                assert _decisions(values, LEGACY_SPREAD, n) == legacy
                if not legacy:
                    continue
                assert _decisions(values, 0.8206, n)
                assert _decisions(values, 1.5275, n)


def test_q95_boundary_equal_and_adjacent_spread() -> None:
    """stat >= spread: equal passes; one ulp above the stat rejects with full text."""
    values = _r7_values()
    _r, record, _re = apply_discovery_dsr_returns(values, 88, null_spread=0.5)
    stat = record["psr_stat"]
    assert isinstance(stat, float)
    ok, reasons, rec = _verdict(values, stat)
    assert ok and reasons == [] and rec["significant"] is True
    below = float(np.nextafter(stat, -np.inf))
    ok, reasons, _ = _verdict(values, below)
    assert ok and reasons == []
    above = float(np.nextafter(stat, np.inf))
    ok, reasons, rec = _verdict(values, above)
    assert not ok and rec["significant"] is False
    assert reasons == [
        f"DSR q95 not significant: stat={stat:.4f} < spread={above:.4f} "
        "(ann SR=2.002, T=875, N=88)"
    ]


def test_gatechain_dsr_verdict_forwards_null_spread() -> None:
    """gatechain.dsr_verdict (not stubbed) honours R7 spreads; None = legacy."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))
    import gatechain

    values = _r7_values()
    passed, record = gatechain.dsr_verdict(values, 88, null_spread=0.8206)
    assert passed and record["null_spread"] == 0.8206
    passed, record = gatechain.dsr_verdict(values, 88, null_spread=1.5275)
    assert not passed and record["null_spread"] == 1.5275
    passed, record = gatechain.dsr_verdict(values, 88)
    assert not passed and record["null_spread"] is None


# ---------------------------------------------------------------------------
# R7: the stored P-2 candidate uid 7f9ad10125f4
# ---------------------------------------------------------------------------

#: A symmetric 875-day shape (400 +/-1, 37 +/-r, one 0) whose bias-corrected
#: kurtosis is exactly 7.7; scaling/centring leaves skew 0 and kurtosis intact.
_R7_R = 6.637640625033072
_R7_SHAPE = np.array(
    [1.0] * 400 + [-1.0] * 400 + [_R7_R] * 37 + [-_R7_R] * 37 + [0.0],
    dtype=np.float64,
)


def _r7_values() -> list[float]:
    sigma = 0.01
    return list(_R7_SHAPE * sigma + 0.12611 * sigma * float(_R7_SHAPE.std(ddof=1)))


def test_q95_stored_candidate_r7() -> None:
    """R7: daily SR 0.12611, T 875, N 88, g3 0, g4 7.7 -> z 3.6796."""
    values = _r7_values()
    sharpe, skew, kurt = daily_sharpe_inputs(values)
    assert sharpe == pytest.approx(0.12611, abs=1e-9)
    assert skew == pytest.approx(0.0, abs=1e-12)
    assert kurt == pytest.approx(7.7, abs=1e-9)

    _pass_result, pass_record, pass_reasons = apply_discovery_dsr_returns(
        values, 88, null_spread=0.8206
    )
    assert pass_reasons == []
    assert pass_record is not None
    assert pass_record["psr_stat"] == pytest.approx(1.226725236387463, abs=1e-9)
    assert pass_record["crit"] == 0.8206 and pass_record["null_spread"] == 0.8206

    _fail_result, fail_record, fail_reasons = apply_discovery_dsr_returns(
        values, 88, null_spread=1.5275
    )
    assert fail_reasons and fail_reasons[0].startswith("DSR q95 not significant: stat=")
    assert fail_reasons == [
        "DSR q95 not significant: stat=1.2267 < spread=1.5275 "
        "(ann SR=2.002, T=875, N=88)"
    ]
    assert fail_record is not None and fail_record["crit"] == 1.5275

    _emax_result, emax_record, emax_reasons = apply_discovery_dsr_returns(values, 88)
    assert emax_reasons and emax_reasons[0].startswith("DSR not significant: p=")
    assert emax_record is not None
    assert emax_record["p_value"] == pytest.approx(0.1098, abs=1e-3)

    # The rule switch, end to end: same null_spread, emax ignores it.
    fit = _fitness(mtm_daily_returns=DailyReturns(_FIRST_DAY, array("d", np.array(values).tobytes())))
    emax = apply_guardrails(
        fit, 2, GuardrailConfig(require_wfa=False, require_bootstrap_ci=False, dsr_rule="emax"),
        num_trials=88, daily_returns=values, null_spread=0.8206,
    )  # fmt: skip
    q95 = apply_guardrails(
        fit, 2, GuardrailConfig(require_wfa=False, require_bootstrap_ci=False, dsr_rule="q95"),
        num_trials=88, daily_returns=values, null_spread=0.8206,
    )  # fmt: skip
    assert emax.dsr_passed is False
    assert q95.dsr_passed is True


# ---------------------------------------------------------------------------
# R8: flag OFF is bit-identical; only record-only keys are added
# ---------------------------------------------------------------------------

_DEFAULT_RECORD_KEYS = {
    "input",
    "trials",
    "significance_level",
    "day0_convention",
    "observations",
    "sharpe_daily",
    "sharpe_annualized",
    "skewness",
    "kurtosis",
    "skewness_used",
    "kurtosis_used",
    "expected_max_sharpe_daily",
    "p_value",
    "p_value_unclamped",
    "significant",
    "lag1_autocorrelation",
    "psr_stat",
    "crit",
    "null_spread",
}


def test_q95_flag_off_is_bit_identical() -> None:
    """R8: explicit rule emax -> same p-value, decision, text; only new keys."""
    values = _gaussian_values(0.05)
    _result, record, reasons = apply_discovery_dsr_returns(values, 88)
    sharpe, skew, kurt = daily_sharpe_inputs(values)
    legacy = DeflatedSharpeRatio(0.05).calculate(
        observed_sharpe=sharpe,
        num_trials=88,
        num_observations=len(values),
        skewness=min(skew, 0.0),
        kurtosis=max(kurt, 3.0),
    )
    assert record["p_value"] == legacy.p_value
    assert record["significant"] == legacy.is_significant
    assert set(record) == _DEFAULT_RECORD_KEYS
    assert record["psr_stat"] == legacy.deflated_sharpe
    assert record["crit"] == LEGACY_SPREAD
    assert record["null_spread"] is None
    assert reasons == ([] if legacy.is_significant else [
        f"DSR not significant: p={legacy.p_value:.4f} >= 0.05 "
        f"(pooled daily SR={sharpe * math.sqrt(252.0):.3f} ann, "
        f"T={len(values)}, N=88)"
    ])

    # emax ignores a passed spread end to end (the rule switch is the only gate).
    fit = _fitness(mtm_daily_returns=DailyReturns(_FIRST_DAY, array("d", np.array(values).tobytes())))
    emax_cfg = GuardrailConfig(require_wfa=False, require_bootstrap_ci=False, dsr_rule="emax")
    with_spread = apply_guardrails(
        fit, 2, emax_cfg, num_trials=88, daily_returns=values, null_spread=0.90,
    )  # fmt: skip
    without = apply_guardrails(
        fit, 2, emax_cfg, num_trials=88, daily_returns=values,
    )  # fmt: skip
    assert with_spread.dsr_record == without.dsr_record
    assert with_spread.passed == without.passed


# ---------------------------------------------------------------------------
# R9: absent rule in the API body / parser defaults to q95 (T4)
# ---------------------------------------------------------------------------


def test_q95_rule_absent_defaults_to_q95() -> None:
    """R9/T4: no body/JSON/config key -> "q95"; emax is the explicit opt-out."""
    from vibe_quant.api.schemas.discovery import DiscoveryLaunchRequest
    from vibe_quant.discovery.__main__ import build_parser
    from vibe_quant.discovery.pipeline import DiscoveryConfig

    assert DiscoveryConfig().dsr_rule == "q95"
    assert "dsr_rule" not in DiscoveryLaunchRequest.model_fields
    assert "dsr_rule" not in DiscoveryLaunchRequest().model_dump()
    assert build_parser().parse_args(["--run-id", "1"]).dsr_rule == "q95"
    assert (
        build_parser().parse_args(["--run-id", "1", "--dsr-rule", "emax"]).dsr_rule
        == "emax"
    )


# ---------------------------------------------------------------------------
# R10: prior N > 0 -- the crit uses gate_n with THIS run's spread
# ---------------------------------------------------------------------------


def _pipeline(**cfg: Any) -> Any:
    from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline

    base: dict[str, Any] = {
        "population_size": 6, "max_generations": 1, "elite_count": 1, "top_k": 5,
        "min_trades": 50, "max_workers": None, "timeframe": "4h",
        "start_date": "2022-06-01", "end_date": "2025-11-17",
        "require_bootstrap_ci": False, "require_dsr": True, "require_buy_hold": False,
        "dsr_rule": "q95", "prior_trials_fn": lambda start, end: 50,
    }  # fmt: skip
    base.update(cfg)
    return DiscoveryPipeline(DiscoveryConfig(**base), backtest_fn=lambda chrom: {})


def test_q95_prior_n_charges_gate_n_with_run_spread() -> None:
    """R10: gate_n = run_n_eff + prior_n (133), spread from this run's series."""
    rows = _equicorr_rows(88, 0.0)
    pipe = _pipeline()
    for i, row in enumerate(rows):
        pipe._fitness_cache[f"k{i:03d}"] = _fitness(
            mtm_daily_returns=DailyReturns(_FIRST_DAY, array("d", row.tobytes()))
        )
    chrom = _chrom()
    fit = _fitness(
        mtm_daily_returns=DailyReturns(_FIRST_DAY, array("d", rows[0].tobytes()))
    )
    pipe._validate_top_strategies([(chrom, fit)], 88)

    trials = pipe._dsr_trials
    assert trials is not None
    assert trials["dsr_rule"] == "q95"
    assert trials["raw"] == 88 and trials["run_n_eff"] == 83
    assert trials["prior_n"] == 50 and trials["gate_n"] == 133
    spread = trials["neff_ev_q95_spread"]
    assert spread == pytest.approx(0.8206, abs=1e-4)

    record = pipe._dsr_records[chrom.uid]
    sharpe, skew, kurt = daily_sharpe_inputs(list(rows[0]))
    stat = {
        n: DeflatedSharpeRatio().calculate(
            sharpe, n, len(rows[0]), min(skew, 0.0), max(kurt, 3.0)
        ).deflated_sharpe
        for n in (83, 133)
    }
    assert stat[133] != pytest.approx(stat[83], abs=1e-9)
    assert record["psr_stat"] == pytest.approx(stat[133])
    assert record["crit"] == pytest.approx(spread)
    assert record["null_spread"] == pytest.approx(spread)


# ---------------------------------------------------------------------------
# Harness plumbing: the per-replicate spread reaches the real chain
# ---------------------------------------------------------------------------


def _eff(spread: float | None) -> EffectiveTrials:
    return EffectiveTrials(
        raw=167,
        non_flat=1,
        n_used=1,
        n_distinct=1,
        subsampled_from=None,
        neff_mean_corr=None,
        neff_eig_pr=None,
        neff_ev=None,
        ev_q95_spread=spread,
        run_n_eff=167,
        source="fallback_raw" if spread is None else "ev_mc",
        fallback_reason="fewer_than_2_series" if spread is None else None,
    )


def _seen_spread(before: int, seen: list[float | None], expected: float | None) -> None:
    assert seen[before:] and all(v == expected for v in seen[before:])


def test_q95_neff_null_forwards_per_rep_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    """neff_null: q95 forwards THIS rep's spread; emax forwards None."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))
    import gatechain
    import layer1
    import neff_null

    queue = [_eff(1.25), _eff(None), _eff(0.99)]
    monkeypatch.setattr(neff_null, "effective_trials", lambda rows, raw, **kw: queue.pop(0))
    flat = DailyReturns(_FIRST_DAY, array("d", np.zeros(10).tobytes()))
    monkeypatch.setattr(layer1, "_null_fitness", lambda sched, chrom, tf: _fitness(mtm_daily_returns=flat))
    monkeypatch.setattr(layer1, "select_picks", lambda scored, **kw: list(scored))
    seen: list[float | None] = []

    def fake_chain(*args: Any, **kwargs: Any) -> dict[str, object]:
        seen.append(kwargs.get("null_spread"))
        settings = args[5]
        return {"chain_recent": {int(n): False for n in settings.dsr_trials}}

    monkeypatch.setattr(gatechain, "evaluate_candidate", fake_chain)
    base, other = neff_null.configs()[0], neff_null.configs()[4]
    closes = {"BTCUSDT": {0: 100.0}}
    for cfg, rule, expected in (
        (base, "q95", 1.25),
        (other, "q95", None),
        (base, "emax", None),
    ):
        before = len(seen)
        neff_null.run_config(cfg, 1, closes, candidates=20, dsr_rule=rule)
        _seen_spread(before, seen, expected)


def test_q95_layer1_selection_null_forwards_spread(monkeypatch: pytest.MonkeyPatch) -> None:
    """layer1 V2: q95 forwards the K-series spread; emax still forwards None."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))
    import gatechain
    import layer1

    queue = [_eff(0.77), _eff(None), _eff(0.99)]
    monkeypatch.setattr(layer1, "effective_trials", lambda rows, raw, **kw: queue.pop(0))
    monkeypatch.setattr(layer1, "select_picks", lambda scored, **kw: list(scored))
    flat = DailyReturns(_FIRST_DAY, array("d", np.zeros(10).tobytes()))
    monkeypatch.setattr(
        layer1, "_null_fitness", lambda sched, chrom, tf: _fitness(mtm_daily_returns=flat)
    )
    seen: list[float | None] = []

    def fake_chain(*args: Any, **kwargs: Any) -> dict[str, object]:
        seen.append(kwargs.get("null_spread"))
        settings = args[5]
        return {"chain_recent": {int(n): False for n in settings.dsr_trials}}

    monkeypatch.setattr(gatechain, "evaluate_candidate", fake_chain)
    settings = gatechain.settings_for("W3")
    closes = {"BTCUSDT": {0: 100.0}}
    for rule, expected in (("q95", 0.77), ("q95", None), ("emax", None)):
        before = len(seen)
        layer1.selection_null(
            "W3", 0.0, 1, closes, settings, candidates=3, dsr_rule=rule
        )
        _seen_spread(before, seen, expected)
