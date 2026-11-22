"""Shadow tracker evidence statistics: weekly-block e-process, Lo futility bound, verdict precedence.

Expected numbers are the design's contract rows (arch-c7-2 §4, probes contract.out / null_final.out).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibe_quant.shadow import estat


def _daily(block_values: list[float]) -> np.ndarray:
    """Daily series whose 7-day block sums equal block_values exactly (value on day 1, zeros after)."""
    x = np.zeros(estat.BLOCK_DAYS * len(block_values))
    x[:: estat.BLOCK_DAYS] = block_values
    return x


def _stats(days: int, ub95: float | None, max_dd: float = 0.0) -> estat.ForwardStats:
    return estat.ForwardStats(
        days=days,
        total_return=0.0,
        sharpe_ann=None if ub95 is None else 0.0,
        se_ann=None if ub95 is None else 0.0,
        ub95=ub95,
        max_dd=max_dd,
        blocks=days // estat.BLOCK_DAYS,
        e_value=1.0,
    )


# R1
@pytest.mark.simple_target(
    reason="design contract R1: e-process closed form (kill-zone evidence math)"
)
@pytest.mark.parametrize(
    ("n_b", "s", "expected"),
    [
        (26, 0.0, 0.5954913341754137),
        (26, 10.0, 3.88305406444306),
        (52, 12.0, 2.5578175307293063),
        (104, -5.0, 0.24941151239014153),
        (104, 20.0, 3.64562662168753),
    ],
)
def test_e_value_closed_form(n_b: int, s: float, expected: float) -> None:
    assert math.exp(estat.log_e(n_b, s)) == expected


# R2
@pytest.mark.simple_target(reason="design contract R2: E >= 100 boundary (kill-zone pass gate)")
@pytest.mark.parametrize(("n_b", "s_star"), [(52, 24.9098198051035), (104, 34.288507159948175)])
def test_pass_boundary(n_b: int, s_star: float) -> None:
    assert math.exp(estat.log_e(n_b, s_star)) >= estat.E_PASS
    assert math.exp(estat.log_e(n_b, s_star - 1e-9)) < estat.E_PASS


def test_pass_comparison_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    """E exactly 100 passes; the next float below does not (R2 at the comparison itself)."""
    x = np.full(182, 0.001)
    monkeypatch.setattr(estat, "e_value", lambda _x: 100.0)
    assert estat.first_verdict(x, start_index=181, max_dd_limit=0.5, can_pass=True) == (
        181,
        "passed",
        "evidence 100.0 >= 100",
    )
    monkeypatch.setattr(estat, "e_value", lambda _x: math.nextafter(100.0, 0.0))
    assert estat.first_verdict(x, start_index=181, max_dd_limit=0.5, can_pass=True) is None


# R3
@pytest.mark.simple_target(reason="design contract R3: min-blocks gate fails closed (kill-zone)")
def test_min_blocks() -> None:
    y26 = [0.02, 0.0] * 13
    assert estat.e_value(_daily(y26[:25])) == 1.0
    assert estat.e_value(_daily(y26[:25])[:181]) == 1.0
    assert estat.e_value(np.concatenate([_daily(y26[:25]), np.full(6, 0.3)])) == 1.0
    assert estat.e_value(_daily(y26)) == 3797.602131007042


# R4
@pytest.mark.simple_target(reason="design contract R4: HAC floor (kill-zone evidence math)")
def test_hac_floor() -> None:
    y = np.array([0.01, -0.01] * 13)
    assert estat.s_hac(y) == 0.010198039027185569
    assert estat.e_value(_daily(list(y))) == 0.5954913341754137


# R5
@pytest.mark.simple_target(reason="design contract R5: HAC inflation (kill-zone evidence math)")
def test_hac_inflates() -> None:
    y = np.array(([0.01] * 3 + [-0.005] * 3) * 5)
    assert estat.s_hac(y) == 0.010043010951179793
    assert float(np.std(y, ddof=1)) == 0.0076282144157341185
    assert estat.e_value(_daily(list(y))) == 1.8530105277969755


# R6: float noise makes np.std of 200 x 0.01 = 1.7e-18, not 0; it must still read as "no variation".
@pytest.mark.parametrize("r", [0.01, -0.01])
def test_constant_returns(r: float) -> None:
    x = np.full(200, r)
    st = estat.forward_stats(x)
    assert (st.sharpe_ann, st.se_ann, st.ub95, st.e_value, st.blocks) == (None, None, None, 1.0, 28)
    assert estat.e_value(x) == 1.0
    assert estat.s_hac(estat.block_sums(x)) == 0.0
    assert estat.first_verdict(x, start_index=0, max_dd_limit=1.0, can_pass=True) is None


# s_HAC = 0 must hold for block sums equal up to float rounding: 0.1 + 0.2 = 0.30000000000000004 vs 0.3.
def test_rounding_only_variation_is_no_evidence() -> None:
    x = np.zeros(7 * 26)
    x[0::14], x[1::14], x[7::14] = 0.1, 0.2, 0.3
    y = estat.block_sums(x)
    assert set(y.tolist()) == {0.30000000000000004, 0.3}
    assert estat.s_hac(y) == 0.0
    assert estat.e_value(x) == 1.0
    assert estat.log_e_path(y)[0, -1] == 0.0
    assert estat.first_verdict(x, start_index=0, max_dd_limit=1.0, can_pass=True) is None


@pytest.mark.simple_target(reason="s_HAC zero-floor equality edge (kill-zone evidence math)")
def test_s_hac_relative_floor_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    y = np.array([0.0, 1.0] * 13)  # max|y| = 1, so the floor is S_REL_FLOOR itself
    s = 0.5099019513592785
    monkeypatch.setattr(estat, "S_REL_FLOOR", s)
    assert estat.s_hac(y) == 0.0
    monkeypatch.setattr(estat, "S_REL_FLOOR", math.nextafter(s, 0.0))
    assert estat.s_hac(y) == s


# R7
@pytest.mark.parametrize(
    ("x", "ub", "verdict"),
    [
        (np.array([0.01, -0.01] * 1975), 0.5000059140776378, None),
        (np.array([0.0] + [0.01, -0.01] * 1975), 0.49994263420469215, "futile"),
    ],
)
def test_futility_edges(x: np.ndarray, ub: float, verdict: str | None) -> None:
    st = estat.forward_stats(x)
    assert (
        st.sharpe_ann is not None and abs(st.sharpe_ann) < 1e-15
    )  # pairwise-sum mean: 0 up to rounding
    assert (st.days, st.ub95) == (len(x), ub)
    out = estat.first_verdict(x, start_index=len(x) - 1, max_dd_limit=1.0, can_pass=True)
    assert (out[1] if out else None) == verdict
    if out:
        assert out[0] == len(x) - 1
        assert out[2] == "forward Sharpe 0.00, 95% upper bound 0.4999 < 0.5 after 3951 days"


# R7b
def test_futility_min_days_and_strict_bound() -> None:
    losing = np.array([-0.01, 0.005] * 45)  # SR far below 0
    assert estat.first_verdict(losing[:89], start_index=0, max_dd_limit=1.0, can_pass=True) is None
    day, state, reason = estat.first_verdict(
        losing, start_index=0, max_dd_limit=1.0, can_pass=True
    ) or (0, "", "")
    assert (day, state) == (89, "futile")
    assert reason.startswith("forward Sharpe -") and reason.endswith(" after 90 days")
    assert estat.futility_reason(_stats(89, 0.4), 1.0) is None
    assert estat.futility_reason(_stats(90, 0.5), 1.0) is None
    assert estat.futility_reason(_stats(90, math.nextafter(0.5, 0.0)), 1.0) is not None
    assert estat.futility_reason(_stats(90, None), 1.0) is None


# R8
@pytest.mark.parametrize(("delta", "verdict"), [(1e-9, None), (-1e-9, "futile")])
def test_futility_200(delta: float, verdict: str | None) -> None:
    z = np.array([1.0, -1.0] * 100)
    s = 0.01 * math.sqrt(200 / 199)
    target = -1.7266075181748632 + delta
    x = target * s / math.sqrt(365) + 0.01 * z
    st = estat.forward_stats(x)
    assert st.sharpe_ann is not None and abs(st.sharpe_ann - target) < 1e-12
    out = estat.first_verdict(x, start_index=199, max_dd_limit=1.0, can_pass=True)
    assert (out[1] if out else None) == verdict


# R9
def test_dd_futility() -> None:
    x = np.array([0.0, 0.0, -0.5, 0.1])
    assert estat.forward_stats(x).max_dd == 0.5
    assert estat.first_verdict(x, start_index=0, max_dd_limit=0.5, can_pass=True) is None
    assert estat.first_verdict(x, start_index=0, max_dd_limit=0.49, can_pass=True) == (
        2,
        "futile",
        "max drawdown 50.00% > limit 49.00%",
    )
    # the peak includes the starting equity 1.0: a first-day loss is a drawdown
    assert estat.first_verdict(
        np.array([-0.3]), start_index=0, max_dd_limit=0.25, can_pass=True
    ) == (
        0,
        "futile",
        "max drawdown 30.00% > limit 25.00%",
    )


def test_forward_bust() -> None:
    x = np.array([0.01, 0.02, -1.0, 0.0])
    assert estat.first_verdict(x, start_index=0, max_dd_limit=1.0, can_pass=True) == (
        2,
        "futile",
        "account bust on forward day 3",
    )


# R10
def test_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = [0.01 + 0.01 * (-1) ** j for j in range(25)] + [0.0]
    x = _daily(blocks)
    x[181] = -0.001  # last day of block 26: drawdown 0.1 %
    assert estat.e_value(x) >= estat.E_PASS
    assert estat.first_verdict(x, start_index=0, max_dd_limit=0.5, can_pass=True) == (
        181,
        "passed",
        f"evidence {estat.e_value(x):.1f} >= 100",
    )
    out = estat.first_verdict(x, start_index=0, max_dd_limit=0.0005, can_pass=True)
    assert out is not None and out[:2] == (181, "futile") and out[2].startswith("max drawdown")
    assert estat.first_verdict(x, start_index=0, max_dd_limit=0.5, can_pass=False) is None

    losing = np.array([-0.01, 0.005] * 91)  # 182 days: SR-futile from day 90
    assert estat.first_verdict(losing, start_index=181, max_dd_limit=1.0, can_pass=True) is not None
    monkeypatch.setattr(estat, "e_value", lambda _x: 1e6)
    assert estat.first_verdict(losing, start_index=181, max_dd_limit=1.0, can_pass=True) == (
        181,
        "passed",
        "evidence 1000000.0 >= 100",
    )
    out = estat.first_verdict(losing, start_index=181, max_dd_limit=1.0, can_pass=False)
    assert out is not None and out[1] == "futile"


# R11
def test_block_assembly() -> None:
    x = np.arange(1.0, 9.0) / 100
    assert estat.block_sums(x).tolist() == [float(np.sum(x[:7]))]
    assert len(estat.block_sums(x[:6])) == 0
    assert len(estat.block_sums(np.zeros(14))) == 2

    # E(26 blocks) = 61.9, E(27 blocks) = 119.8; block 27's only move lands on its first day (index 182)
    x = _daily([0.007 + 0.01 * (-1) ** j for j in range(27)])
    assert estat.e_value(x[:183]) == estat.e_value(x[:182]) < estat.E_PASS
    for start in (0, 150, 182, 188):
        assert estat.first_verdict(x, start_index=start, max_dd_limit=0.5, can_pass=True) == (
            188,
            "passed",
            "evidence 119.8 >= 100",
        )


# R12
def test_single_lucky_block() -> None:
    blocks = [0.0] * 103 + [0.05]
    y = np.array(blocks)
    assert y.sum() / estat.s_hac(y) == 10.198039027185569
    x = _daily(blocks)
    assert estat.e_value(x) == 0.8908567856826466
    assert estat.first_verdict(x, start_index=0, max_dd_limit=0.5, can_pass=True) is None


@pytest.mark.simple_target(reason="metric math: Sharpe, Lo SE, drawdown, total return (kill-zone)")
def test_forward_stats_values() -> None:
    x = np.array([0.1, -0.2, 0.05])
    st = estat.forward_stats(x)
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    mu = m / s
    assert st.days == 3 and st.blocks == 0 and st.e_value == 1.0
    assert st.total_return == 1.1 * 0.8 * 1.05 - 1.0
    assert st.max_dd == 1.0 - (1.1 * 0.8) / 1.1
    assert st.sharpe_ann == mu * math.sqrt(365)
    assert st.se_ann == math.sqrt(365.0 * (1.0 + mu * mu / 2.0) / 3)
    assert st.ub95 == st.sharpe_ann + estat.Z95 * st.se_ann


def test_non_finite_returns_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        estat.first_verdict(
            np.array([0.01, math.nan]), start_index=0, max_dd_limit=0.5, can_pass=True
        )
    with pytest.raises(ValueError, match="non-finite"):
        estat.forward_stats(np.array([math.inf]))


def _ar1(rng: np.random.Generator, paths: int, days: int, rho: float) -> np.ndarray:
    x = rng.standard_normal((paths, days))
    for d in range(1, days):
        x[:, d] = rho * x[:, d - 1] + math.sqrt(1 - rho * rho) * x[:, d]
    return x


# R13
@pytest.mark.simple_target(
    reason="design contract R13: null false-pass rate of the shipped rule (kill-zone gate)"
)
def test_null_ar05() -> None:
    rng = np.random.default_rng(7)
    x = np.concatenate([_ar1(rng, 2000, 1825, 0.5) for _ in range(2)])
    lp = estat.log_e_path(estat.block_sums(x))
    crossed = lp >= math.log(estat.E_PASS)
    rate = float(crossed.any(axis=1).mean())
    assert 0.0 < rate <= 0.01
    # the vectorised path is the scalar rule: first crossing block agrees with e_value on the prefix
    for p in np.flatnonzero(crossed.any(axis=1)):
        k = int(crossed[p].argmax())
        assert estat.e_value(x[p, : 7 * (k + 1)]) >= estat.E_PASS
        assert k == 0 or estat.e_value(x[p, : 7 * k]) < estat.E_PASS


@pytest.mark.simple_target(
    reason="the null check's vectorised rule must equal the scalar gate (kill-zone)"
)
def test_vectorised_path_matches_scalar() -> None:
    x = np.random.default_rng(3).standard_normal((3, 7 * 60)) * 0.01 + 0.002
    lp = estat.log_e_path(estat.block_sums(x))
    for p in range(3):
        for k in range(60):
            e = estat.e_value(x[p, : 7 * (k + 1)])
            if k + 1 < estat.MIN_BLOCKS:
                assert lp[p, k] == 0.0 and e == 1.0
            else:
                assert math.exp(lp[p, k]) == pytest.approx(e, rel=1e-9)
