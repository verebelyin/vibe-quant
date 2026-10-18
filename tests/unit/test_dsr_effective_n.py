"""EV-matched effective-N estimator (bd vibe-quant-megq1, task T1).

Contract rows R1-R12, R28-R29 plus the review's R4b/R6b/R9b from
``data/swarm/20261017-power-paper/design-neff.md`` section 5. T1 RECORDS ONLY:
``notes.dsr_trials`` gains ``neff_ev`` / ``run_n_eff`` / ``neff_ev_source`` /
``neff_fallback_reason`` and the gate still uses today's raw+prior N.

Walsh rows are rows of ``scipy.linalg.hadamard(32)``: exactly uncorrelated,
zero-mean except row 0 (used as the flat row). Row 0 is all-ones -> std 0.
"""

from __future__ import annotations

import logging
import math
from array import array
from typing import Any

import numpy as np
import pytest
from scipy.linalg import hadamard

from vibe_quant.discovery.neff import (
    NEFF_EV_DRAWS,
    NEFF_EV_SEED,
    NEFF_MIN_DAYS,
    EffectiveTrials,
    _distinct_indices,
    effective_trials,
    expected_max_inverse,
)
from vibe_quant.metrics import DailyReturns
from vibe_quant.overfitting.dsr import DeflatedSharpeRatio

_H = hadamard(32).astype(np.float64)
# Tiling a Walsh row repeats its values, so the 32x block correlations are
# unchanged -- but the row must clear NEFF_MIN_DAYS to reach the EV path.
_TILE = math.ceil(NEFF_MIN_DAYS / _H.shape[1])
# Chief ruling 2026-10-18 (design-neff.md, R1 boundary): exactly-identical
# retained series are collapsed BEFORE the MC/EV step, so 5 identical rows yield
# ONE series -> n_m 1 analytically -> run_n_eff == 8 regardless of the seed.
# R2/R3/R5 keep their run_n_eff (their targets are unchanged by the collapse,
# since duplicate columns do not change the max).


def _rows(*arrs: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(a, dtype=np.float64) for a in arrs]


def _w(*indices: int) -> list[np.ndarray]:
    """Tiled Walsh rows (length >= NEFF_MIN_DAYS, correlations unchanged)."""
    return [np.tile(_H[i], _TILE) for i in indices]


def _series(sr_ann: float, n: int = 1264, seed: int = 1) -> DailyReturns:
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    x = z * 0.01 + sr_ann / math.sqrt(252.0) * 0.01
    return DailyReturns(19_000, array("d", x.tobytes()))


# ---------------------------------------------------------------------------
# R1-R3: the estimator's core arithmetic
# ---------------------------------------------------------------------------


def test_identical_series_count_once_scaled() -> None:
    """R1 (chief ruling): 5 identical rows collapse to n_m 1 -> run_n_eff 8."""
    eff = effective_trials(_w(1, 1, 1, 1, 1), 40)
    assert isinstance(eff, EffectiveTrials)
    assert eff.source == "ev_mc" and eff.fallback_reason is None
    # n_used stays the PRE-collapse retained count, so it scales raw/n_used.
    assert eff.n_used == 5
    assert eff.n_distinct == 1
    # one distinct series -> n_m 1 analytically, scaled by raw/n_used = 40/5
    assert eff.neff_ev == pytest.approx(8.0)
    assert eff.run_n_eff == 8
    # the legacy informational count still treats them as one
    assert eff.neff_mean_corr == pytest.approx(8.0)


@pytest.mark.parametrize("seed", [20261018, 2, 5])
def test_identical_series_seed_independent(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    """R1 (chief ruling): 5 identical rows -> run_n_eff 8 for any NEFF_EV_SEED.

    Without the collapse these seeds land above the emax(1)/emax(2) boundary
    (n_m 2 -> 16), so this test dies if the collapse is dropped.
    """
    import vibe_quant.discovery.neff as neff

    monkeypatch.setattr(neff, "NEFF_EV_SEED", seed)
    eff = neff.effective_trials(_w(1, 1, 1, 1, 1), 40)
    assert eff.n_distinct == 1
    assert eff.neff_ev == pytest.approx(8.0)
    assert eff.run_n_eff == 8


def test_duplicate_rows_collapse_to_distinct_series() -> None:
    """Chief ruling: 4 identical + 1 independent row counts as 2 retained series.

    The MC runs on the 2 distinct series (n_m 3, E[max of 2] -> emax(3)), but the
    scaling keeps the pre-collapse retained count: neff_ev = n_m * raw / 5.
    """
    eff = effective_trials(_w(1, 1, 1, 1, 2), 40)
    assert eff.source == "ev_mc" and eff.fallback_reason is None
    assert eff.n_distinct == 2
    assert eff.n_used == 5
    # 2 independent series -> n_m 3, scaled by raw/n_used = 40/5 -> 24.0
    assert eff.neff_ev == pytest.approx(24.0)
    assert eff.run_n_eff == 24


def test_uncorrelated_series_keep_raw() -> None:
    """R2: uncorrelated rows -> n_m == n_used, so run_n_eff == raw."""
    rows = _w(*range(1, 11))
    eff = effective_trials(rows, 30)
    assert eff.source == "ev_mc"
    # C = I -> the MC target is the sample mean of max(10 iid normals) for the
    # fixed seed, and the inverse maps it back to 10.
    z = np.random.default_rng(NEFF_EV_SEED).standard_normal((NEFF_EV_DRAWS, 10))
    assert float(z.max(axis=1).mean()) == pytest.approx(1.5286247323465862, rel=1e-15)
    assert eff.run_n_eff == 30
    assert eff.neff_ev == pytest.approx(30.0)
    assert eff.neff_mean_corr == pytest.approx(30.0)
    assert eff.neff_eig_pr == pytest.approx(30.0)


def test_anticorrelated_clamps_to_raw() -> None:
    """R3: perfect anti-correlation -> n_m 3 (neff 15) then clamp to raw 10."""
    row = _w(1)[0]
    eff = effective_trials(_rows(row, -row), 10)
    assert eff.source == "ev_mc"
    assert eff.neff_ev == pytest.approx(15.0)  # n_m 3 * 10 / 2
    assert eff.run_n_eff == 10  # clamped to raw
    assert eff.neff_mean_corr is None  # denom 1 + (2-1)*(-1) == 0 -> no estimate


# ---------------------------------------------------------------------------
# r2 findings: the collapse contract, near-identical rows, eigenvector signs
# ---------------------------------------------------------------------------


def test_collapse_never_chains_across_rows() -> None:
    """B1: a pairwise-close fan must not collapse transitively into one row.

    Adjacent rows of a slow 30-row fan around a Walsh pair are correlated above
    1 - 1e-12, but each row is far from row 0, so the transitive closure of the
    old union-find collapsed all 30 rows into one representative (and discarded
    canonical index 0). Greedy canonical-first collapsing keeps only the 13 rows
    that directly match an existing representative.
    """
    h = np.tile(_H, (1, _TILE))
    a, b = h[1], h[2]
    rows = _rows(*[np.cos(t) * a + np.sin(t) * b for t in np.arange(30) * 1.3e-6])
    ordered = sorted(rows, key=lambda r: r.tobytes())
    corr = np.asarray(np.nan_to_num(np.corrcoef(np.vstack(ordered))))

    reps = _distinct_indices(corr)
    assert reps[0] == 0  # canonical index 0 is always a representative
    assert len(reps) == 13

    eff = effective_trials(rows, 40)
    assert eff.source == "ev_mc" and eff.fallback_reason is None
    assert eff.n_used == 30  # pre-collapse retained count is unchanged
    assert eff.n_distinct == 13


def test_near_identical_rows_stay_distinct() -> None:
    """SF1: corr 0.999999 is a distinct series (tolerance is 1e-12, not 1e-5).

    A tolerance relaxed to 1e-5 merges the pair, giving n_m 1 and run_n_eff 20
    instead of 2 and 40.
    """
    a, b = _w(1)[0], _w(2)[0]
    near = 0.999999 * a + math.sqrt(1.0 - 0.999999**2) * b
    assert float(np.corrcoef(a, near)[0, 1]) == pytest.approx(0.999999)

    eff = effective_trials(_rows(a, near), 40)
    assert eff.source == "ev_mc" and eff.fallback_reason is None
    assert eff.n_distinct == 2
    assert eff.neff_ev == pytest.approx(40.0)
    assert eff.run_n_eff == 40


@pytest.mark.parametrize(
    ("offset", "expected"),
    [(0.0, [0]), (-1e-15, [0, 1]), (1e-15, [0])],
    ids=["exact-threshold", "just-below", "just-above"],
)
def test_collapse_tolerance_boundary(offset: float, expected: list[int]) -> None:
    """SF1: a row joins a representative iff corr >= 1 - 1e-12 exactly.

    The loose-tolerance mutant (1e-5) collapses the just-below pair.
    """
    edge = (1.0 - 1e-12) + offset
    corr = np.array([[1.0, edge], [edge, 1.0]])
    assert _distinct_indices(corr) == expected


def test_symmetric_root_ignores_eigenvector_signs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SF2: the MC root must not depend on the eigenvector signs.

    The symmetric root V diag(sqrt(w)) V.T is invariant under V -> -V; the
    forbidden basis-dependent form (V*sqrt(w)).T flips the target under the sign
    flip and gives n_m 2 instead of 3 on this fixture.
    """
    a, b = _w(1)[0], _w(2)[0]
    rows = _rows(a, a, a, a, 0.1 * a + math.sqrt(0.99) * b)
    baseline = effective_trials(rows, 40)
    assert baseline.n_distinct == 2
    assert baseline.neff_ev == pytest.approx(24.0)
    assert baseline.run_n_eff == 24

    real_eigh = np.linalg.eigh

    def sign_flipped(*args: Any, **kwargs: Any) -> Any:
        w, vec = real_eigh(*args, **kwargs)
        return w, -vec

    monkeypatch.setattr(np.linalg, "eigh", sign_flipped)
    flipped = effective_trials(rows, 40)
    assert flipped.n_distinct == baseline.n_distinct
    assert flipped.neff_ev == pytest.approx(24.0)
    assert flipped.run_n_eff == 24


# ---------------------------------------------------------------------------
# R4-R8: fallbacks fail closed at raw
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [[], _w(1)], ids=["no rows", "one non-flat row"])
def test_too_few_series_falls_back_raw(rows: list[np.ndarray]) -> None:
    """R4: 0 or 1 non-flat row -> reason fewer_than_2_series, raw N."""
    eff = effective_trials(_rows(*rows), 40)
    assert eff.run_n_eff == 40
    assert eff.neff_ev is None
    assert eff.source == "fallback_raw"
    assert eff.fallback_reason == "fewer_than_2_series"


def test_cluster_structure_not_undercounted() -> None:
    """R5: 150 identical + 17 independent rows -> 17, not the mean-corr 1.24."""
    eff = effective_trials(_w(*([1] * 150 + list(range(2, 19)))), 167)
    assert eff.source == "ev_mc"
    assert eff.n_used == 167
    assert eff.run_n_eff == 17
    assert eff.neff_ev == pytest.approx(17.0)
    assert eff.neff_mean_corr == pytest.approx(1.2385752986632321)


def test_unequal_lengths_fall_back() -> None:
    """R6: series of different lengths cannot share a correlation matrix."""
    long_ = np.tile(_H[1], 40)[:1264]
    short = np.tile(_H[2], 40)[:1263]
    eff = effective_trials(_rows(long_, short), 40)
    assert eff.source == "fallback_raw"
    assert eff.fallback_reason == "unequal_series_lengths"
    assert eff.neff_ev is None
    assert eff.run_n_eff == 40


@pytest.mark.parametrize(
    ("raised", "marker", "n_series", "expected"),
    [
        (
            np.linalg.LinAlgError("no convergence"),
            "no convergence",
            2,
            "EV-matched effective-N estimator failed (LinAlgError); "
            "falling back to raw N=2",
        ),
        (
            ValueError("secret-sentinel"),
            "secret-sentinel",
            3,
            "EV-matched effective-N estimator failed (ValueError); "
            "falling back to raw N=3",
        ),
    ],
    ids=["LinAlgError-2-series", "ValueError-3-series"],
)
def test_estimator_error_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    raised: Exception,
    marker: str,
    n_series: int,
    expected: str,
) -> None:
    """R7: an eigendecomposition failure is caught, logged, never escapes.

    The warning names the exception CLASS and n_used, but must not leak the raw
    exception text (no ``exc_info``, no exception message) on a money path.
    """
    def boom(*_a: Any, **_k: Any) -> Any:
        raise raised

    monkeypatch.setattr(np.linalg, "eigh", boom)
    with caplog.at_level(logging.WARNING):
        eff = effective_trials(_w(*range(1, n_series + 1)), 40)
    assert eff.source == "fallback_raw"
    assert eff.fallback_reason == "estimator_error"
    assert eff.neff_ev is None
    assert eff.run_n_eff == 40
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warnings] == [expected]
    assert marker not in caplog.text


def test_zero_raw_gives_one() -> None:
    """R8: raw 0 with no rows -> run_n_eff 1 (never 0)."""
    eff = effective_trials([], 0)
    assert eff.run_n_eff == 1
    assert eff.neff_ev is None
    assert eff.source == "fallback_raw"


# ---------------------------------------------------------------------------
# R9-R11: determinism, the subsample, the inverse
# ---------------------------------------------------------------------------


def test_deterministic_and_order_invariant() -> None:
    """R9: repeat and reversed input give identical results."""
    rows = _w(*range(1, 11))
    first = effective_trials(rows, 30)
    repeat = effective_trials(rows, 30)
    reversed_ = effective_trials(list(reversed(rows)), 30)
    assert first.run_n_eff == repeat.run_n_eff == reversed_.run_n_eff
    assert first.neff_ev == repeat.neff_ev == reversed_.neff_ev
    assert first.neff_eig_pr == reversed_.neff_eig_pr


def test_ev_uses_subsample() -> None:
    """R10: the EV runs on the 500-row subsample and scales by raw/n_used."""
    rng = np.random.default_rng(4)
    common = rng.normal(0.0, 0.01, 1264)
    rows = _rows(*[common + rng.normal(0.0, 0.001, 1264) for _ in range(600)])
    eff = effective_trials(rows, 600)
    assert eff.subsampled_from == 600
    assert eff.n_used == 500
    assert eff.non_flat == 600
    assert eff.source == "ev_mc"
    # neff_ev = n_m * raw / n_used, with run_n_eff the clamped ceiling
    assert eff.neff_ev == pytest.approx(2.4)
    assert eff.run_n_eff == 3


def test_expected_max_inverse_edges() -> None:
    """R11: the inverse over the emax table, plus a non-finite guard."""
    assert expected_max_inverse(-1.0) == 1
    assert expected_max_inverse(0.0) == 1
    assert expected_max_inverse(0.5197553442805939) == 2  # exactly E[max of 2]
    assert expected_max_inverse(0.52) == 3
    assert expected_max_inverse(1.5287) == 10
    with pytest.raises(ValueError):
        expected_max_inverse(float("nan"))


# ---------------------------------------------------------------------------
# R4b / R6b: review r1 additions
# ---------------------------------------------------------------------------


def test_non_finite_row_falls_back() -> None:
    """R4b: a NaN row falls back BEFORE today's std()>0 filter can hide it."""
    rows = _w(*range(1, 11))
    rows[4] = rows[4].copy()
    rows[4][3] = np.nan
    eff = effective_trials(rows, 30)
    assert eff.source == "fallback_raw"
    assert eff.fallback_reason == "non_finite"
    assert eff.neff_ev is None
    assert eff.run_n_eff == 30


def test_short_series_fall_back() -> None:
    """R6b: NEFF_MIN_DAYS floor -> 364 days falls back, 365 runs the EV."""
    short_a = np.tile(_H[1], 12)[: NEFF_MIN_DAYS - 1]
    short_b = np.tile(_H[2], 12)[: NEFF_MIN_DAYS - 1]
    short = effective_trials(_rows(short_a, short_b), 40)
    assert short.source == "fallback_raw"
    assert short.fallback_reason == "too_short"
    assert short.run_n_eff == 40

    ok_a = np.tile(_H[1], 12)[:NEFF_MIN_DAYS]
    ok_b = np.tile(_H[2], 12)[:NEFF_MIN_DAYS]
    ok = effective_trials(_rows(ok_a, ok_b), 40)
    assert ok.source == "ev_mc" and ok.fallback_reason is None
    assert ok.neff_ev is not None


# ---------------------------------------------------------------------------
# R9b: eval_windows > 1 chains to a shared grid -> equal lengths
# ---------------------------------------------------------------------------


def test_eval_windows_chained_series_use_ev_mc(monkeypatch: pytest.MonkeyPatch) -> None:
    """R9b: a 3-window chain gives equal-length series, so the EV runs."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.utils import compute_day_count, split_into_windows
    from vibe_quant.validation.extraction import date_to_ns

    day_ns = 86_400 * 1_000_000_000

    def fake_single(self: NTBacktestFn, chrom: Any, s: str, e: str, symbols: Any = None) -> Any:
        n = compute_day_count(s, e) or 0
        off = 0.001 if chrom.stop_loss_pct == 2.0 else -0.0005
        out: dict[str, Any] = {
            "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
            "total_trades": 60, "total_return": 0.1,
        }  # fmt: skip
        if self.collect_daily_returns:
            vals = [off + (i % 5) * 1e-4 for i in range(n)]
            first = (date_to_ns(s) or 0) // day_ns
            out["mtm_daily_returns"] = DailyReturns(first, array("d", vals))
        return out

    monkeypatch.setattr(NTBacktestFn, "_run_single", fake_single)
    start, end = "2024-01-01", "2025-07-01"  # chained length clears NEFF_MIN_DAYS
    fn = NTBacktestFn(
        ["X"], "4h", start, end, windows=split_into_windows(start, end, 3),
        min_trades=10, collect_daily_returns=True,
    )  # fmt: skip
    a = fn(_chrom(stop_loss_pct=2.0))["mtm_daily_returns"]
    b = fn(_chrom(stop_loss_pct=3.0))["mtm_daily_returns"]
    assert isinstance(a, DailyReturns) and isinstance(b, DailyReturns)
    assert len(a) == len(b) == (compute_day_count(start, end) or 0) - 1

    eff = effective_trials(_rows(np.asarray(a.values), np.asarray(b.values)), 50)
    assert eff.source == "ev_mc" and eff.fallback_reason is None
    assert eff.n_used == 2


# ---------------------------------------------------------------------------
# R12: T1 records, the gate is unchanged
# ---------------------------------------------------------------------------


def _chrom(**kw: Any) -> Any:
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    base: dict[str, Any] = {
        "entry_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0, None)],
        "exit_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        "stop_loss_pct": 2.0, "take_profit_pct": 4.0, "direction": Direction.LONG,
    }  # fmt: skip
    base.update(kw)
    return StrategyChromosome(**base)


def _fit(**kw: Any) -> Any:
    from vibe_quant.discovery.fitness import FitnessResult

    base: dict[str, Any] = {
        "sharpe_ratio": 0.1, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 200, "total_return": 0.3, "complexity_penalty": 0.0,
        "overtrade_penalty": 0.0, "sl_tp_penalty": 0.0, "raw_score": 0.5,
        "adjusted_score": 0.5, "passed_filters": True, "filter_results": {},
    }  # fmt: skip
    base.update(kw)
    return FitnessResult(**base)


def _pipeline(**cfg: Any) -> Any:
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline

    base: dict[str, Any] = {
        "population_size": 6, "max_generations": 1, "elite_count": 1, "top_k": 5,
        "min_trades": 50, "max_workers": None, "timeframe": "4h",
        "start_date": "2022-06-01", "end_date": "2025-11-17",
        "require_bootstrap_ci": False, "require_dsr": True,
    }  # fmt: skip
    base.update(cfg)
    fn = NTBacktestFn(["BTCUSDT"], "4h", base["start_date"], base["end_date"])
    return DiscoveryPipeline(DiscoveryConfig(**base), backtest_fn=fn)


def test_t1_records_but_does_not_gate() -> None:
    """R12: the 4 info keys are recorded; N is still raw+prior (T3 switches)."""
    pipe = _pipeline(prior_trials_fn=lambda s, e: 921)
    rng = np.random.default_rng(11)
    common = rng.normal(0.0005, 0.01, 1264)
    for i in range(6):
        x = common + rng.normal(0, 0.001, 1264)
        pipe._fitness_cache[f"k{i}"] = _fit(
            mtm_daily_returns=DailyReturns(19_000, array("d", x.tobytes()))
        )
    pipe._fitness_cache["flat"] = _fit()

    chrom = _chrom()
    pipe._validate_top_strategies([(chrom, _fit(mtm_daily_returns=_series(2.5)))], 79)

    trials = pipe._dsr_trials
    assert trials is not None
    for key in ("neff_ev", "run_n_eff", "neff_ev_source", "neff_fallback_reason"):
        assert key in trials, key
    assert trials["neff_ev_source"] == "ev_mc"
    assert trials["neff_fallback_reason"] is None
    assert isinstance(trials["run_n_eff"], int)
    assert trials["non_flat"] == 6
    # the gate is UNCHANGED at T1: raw 79 + prior 921
    assert trials["gate_n"] == 1000 and trials["prior_n"] == 921
    assert pipe._dsr_records[chrom.uid]["trials"] == 1000


# ---------------------------------------------------------------------------
# R28/R29: stored records, p-value at the effective Ns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sr_daily", "obs", "skew", "kurt", "n_pass", "p_pass", "n_reject", "p_reject"),
    [
        (0.07145514172223692, 583, 0.4411043186673014, 5.5771365757692095, 752,
         0.925788404900211, 13, 0.49184659184404467),  # run 920 e4f18102194c
        (0.05171608339198556, 1689, 5.596524297180125, 95.36507893278105, 167,
         0.7135411452541981, 27, 0.4632292151894185),  # run 950 a083a1f3236b
    ],
    ids=["920-e4f18102194c", "950-a083a1f3236b"],
)
def test_stored_record_p_values(
    sr_daily: float, obs: int, skew: float, kurt: float,
    n_pass: int, p_pass: float, n_reject: int, p_reject: float,
) -> None:
    """R28/R29: the DSR p-values at the raw and effective Ns of two stored runs."""
    dsr = DeflatedSharpeRatio()
    at_pass = dsr.calculate(sr_daily, n_pass, obs, min(skew, 0.0), max(kurt, 3.0))
    at_reject = dsr.calculate(sr_daily, n_reject, obs, min(skew, 0.0), max(kurt, 3.0))
    assert at_pass.p_value == pytest.approx(p_pass)
    assert at_reject.p_value == pytest.approx(p_reject)
