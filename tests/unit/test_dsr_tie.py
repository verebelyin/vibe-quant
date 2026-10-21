"""DSR tie behaviour pin (bead vibe-quant-gfehd; design lane4-gates T1, rows R1-R6).

The discovery legacy rule is strict (``p_value < 0.05``, ``overfitting/dsr.py``
``calculate``) while the overfitting re-check's legacy rule is inclusive
(``1 - p >= confidence_threshold``, ``DeflatedSharpeRatio.passes_threshold``).
They therefore disagree on the float band ``[0.05, 0.0500000000000001]``,
reachable through the DSR statistic; ``p`` exactly 0.05 is not reachable
through ``_norm_sf``. These rows pin both comparators and every band edge, at
the discovery gate (``apply_discovery_dsr_returns``) and at the q95 helper.
No production code changes.
"""

from __future__ import annotations

import math

import numpy as np

from vibe_quant.discovery.guardrails import (
    apply_discovery_dsr_returns,
    q95_dsr_decision,
)
from vibe_quant.discovery.neff import LEGACY_NULL_SPREAD
from vibe_quant.overfitting.dsr import (
    DeflatedSharpeRatio,
    DSRResult,
    _norm_sf,
)


def _result_at(p_value: float) -> DSRResult:
    """A DSRResult on the p-value only (``passes_threshold`` reads nothing else)."""
    return DSRResult(
        deflated_sharpe=1.0,
        p_value=p_value,
        is_significant=p_value < 0.05,
        expected_max_sharpe=0.0,
        sharpe_variance=1.0,
        observed_sharpe=0.1,
        num_trials=1,
        num_observations=100,
    )


def _tie_series() -> list[float]:
    """A deterministic finite daily-return series (T = 250, daily SR ~ 0.98)."""
    z = np.random.default_rng(11).normal(size=250)
    return list(z * 0.01 + 0.0005)


# ---------------------------------------------------------------------------
# R1-R2: the two legacy comparators at p exactly 0.05
# ---------------------------------------------------------------------------


def test_legacy_discovery_rejects_p_exactly_005(monkeypatch) -> None:
    """R1: discovery's strict ``p < 0.05`` rejects the tie."""
    monkeypatch.setattr("vibe_quant.overfitting.dsr._norm_sf", lambda x: 0.05)
    result = DeflatedSharpeRatio().calculate(0.1, 1, 100)
    assert result.p_value == 0.05
    assert result.is_significant is False


def test_overfitting_legacy_passes_p_exactly_005(monkeypatch) -> None:
    """R2: the re-check's inclusive ``1 - p >= 0.95`` passes the same result."""
    monkeypatch.setattr("vibe_quant.overfitting.dsr._norm_sf", lambda x: 0.05)
    result = DeflatedSharpeRatio().calculate(0.1, 1, 100)
    assert result.p_value == 0.05
    assert DeflatedSharpeRatio().passes_threshold(result, 0.95) is True


# ---------------------------------------------------------------------------
# R3: the disagreement band edges
# ---------------------------------------------------------------------------


def test_disagreement_band_edges() -> None:
    """R3: discovery rejects all four band values; the re-check passes the first three."""
    dsr = DeflatedSharpeRatio()
    for p_value, passes in (
        (0.05, True),
        (0.050000000000000065, True),
        (0.0500000000000001, True),
        (0.05000000000000011, False),
    ):
        assert (p_value < 0.05) is False
        assert dsr.passes_threshold(_result_at(p_value), 0.95) is passes
    assert _norm_sf(LEGACY_NULL_SPREAD) == 0.050000000000000065


# ---------------------------------------------------------------------------
# R4: the discovery gate text and record at the tie
# ---------------------------------------------------------------------------


def test_discovery_gate_reason_at_tie(monkeypatch) -> None:
    """R4: the discovery gate rejects the tie with the legacy text, crit 1.6449."""
    monkeypatch.setattr("vibe_quant.overfitting.dsr._norm_sf", lambda x: 0.05)
    result, record, reasons = apply_discovery_dsr_returns(_tie_series(), 7)
    assert result is not None and result.p_value == 0.05
    assert reasons
    assert reasons[0].startswith("DSR not significant: p=0.0500 >= 0.05 (pooled daily SR=")
    assert record["significant"] is False
    assert record["crit"] == 1.6448536269514722
    assert record["null_spread"] is None


# ---------------------------------------------------------------------------
# R5-R6: the q95 helper at the tie and on its fallbacks
# ---------------------------------------------------------------------------


def test_q95_tie_passes() -> None:
    """R5: the q95 rule is a non-strict ``>=``: stat == spread passes."""
    assert q95_dsr_decision(1.5, False, 1.5) == (True, 1.5)


def test_q95_fallbacks() -> None:
    """R6: legacy spread, NaN stat and non-finite spread fail closed."""
    assert q95_dsr_decision(2.0, False, LEGACY_NULL_SPREAD) == (
        False,
        LEGACY_NULL_SPREAD,
    )
    assert q95_dsr_decision(math.nan, True, 1.2) == (True, None)
    assert q95_dsr_decision(2.0, False, math.inf) == (False, None)
