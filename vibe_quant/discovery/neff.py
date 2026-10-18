"""Shared EV-matched effective-N estimator (bd vibe-quant-megq1).

"EV-matched effective N" is the N whose expected maximum of N iid standard
normals equals the MONTE-CARLO expected maximum of a run's retained, correlated
trial series. Variance-based estimators (mean correlation, participation ratio)
count one big common factor as "1 trial" and are far too lenient for a best-of-N
statistic: on a Gaussian selection null they let 12-16% of pure-noise best picks
through DSR at rho 0.5-0.8, while this estimator stays <= 3.25%
(``data/swarm/20261017-power-paper/design-neff.md`` section 0). It is also robust
to clusters: 150 identical series plus 17 independent ones gives ~17 where the
mean-correlation count gives ~1.2.

The inverse maps the true Monte-Carlo E[max] through Bailey's expected-max
approximation, so it is only exact at small N: 167 iid series map to about 157,
not 167 (review N4). That is intended -- the DSR null SR0 is the true E[max]
times sigma, so the identity ``run_n_eff == raw`` holds only for small n.

Layering (review N7): this module imports only ``vibe_quant.overfitting.dsr``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from vibe_quant.overfitting.dsr import DeflatedSharpeRatio

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

# Monte-Carlo schedule: fixed so the target (and hence run_n_eff) is deterministic.
NEFF_EV_DRAWS: int = 4096
NEFF_EV_SEED: int = 20261018
# Eigendecomposition is O(m^3); cap the rows it runs on.
NEFF_MAX_SERIES: int = 500
# Below this many days every correlation is near +/-1 and n_m collapses to 1-2.
NEFF_MIN_DAYS: int = 365
# ``notes.dsr_trials.n_rule`` value that marks a run's recorded N as effective.
N_RULE_EFFECTIVE: str = "cumulative_effective_same_tf_overlapping_train"

FallbackReason = Literal[
    "fewer_than_2_series",
    "unequal_series_lengths",
    "non_finite",
    "too_short",
    "estimator_error",
]


@dataclass(frozen=True, slots=True)
class EffectiveTrials:
    """One run's recorded trial counts (design-neff.md section 2).

    ``neff_mean_corr`` / ``neff_eig_pr`` are the two legacy informational
    estimators, kept verbatim. ``n_used`` is the retained count after the
    non-flat/subsample filters and gates the scaling ``raw / n_used``;
    ``n_distinct`` is the count of exactly-identical series collapsed out of it
    (in-memory only, not in the ``notes.dsr_trials`` contract). On any fallback
    ``neff_ev`` is None and ``run_n_eff == max(1, raw)`` (today's behaviour,
    fail closed).
    """

    raw: int
    non_flat: int
    n_used: int
    # Distinct series after collapsing exactly-identical rows (chief ruling on
    # R1). In-memory only: not part of the ``notes.dsr_trials`` contract.
    n_distinct: int
    subsampled_from: int | None
    neff_mean_corr: float | None
    neff_eig_pr: float | None
    neff_ev: float | None
    run_n_eff: int
    source: Literal["ev_mc", "fallback_raw"]
    fallback_reason: FallbackReason | None


def expected_max_inverse(target: float) -> int:
    """Smallest N >= 1 whose expected max of N standard normals is >= ``target``.

    Bisection over [1, 2**31] on ``DeflatedSharpeRatio._expected_max_sharpe``.
    ``target <= 0`` -> 1. A non-finite ``target`` raises ``ValueError`` so the
    caller fails closed through ``estimator_error``.
    """
    if not math.isfinite(target):
        msg = f"target must be finite, got {target!r}"
        raise ValueError(msg)
    if target <= 0.0:
        return 1
    lo, hi = 1, 2**31
    if DeflatedSharpeRatio._expected_max_sharpe(hi) < target:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if DeflatedSharpeRatio._expected_max_sharpe(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _distinct_indices(corr: NDArray[np.float64], *, tol: float = 1e-12) -> list[int]:
    """Canonical-first representative of each correlation-identical row group.

    Walks ``corr`` in canonical order and greedily keeps a row as a new
    representative unless it matches an EXISTING representative directly at
    ``corr >= 1 - tol`` (chief ruling on R1). Matching only a retained row that
    is itself a duplicate is deliberately not enough: a transitive closure would
    merge a chain of pairwise-close rows into one series that no two of its
    members are actually close to (review B1). Because only earlier indices are
    candidates, index 0 is always a representative.
    """
    reps: list[int] = []
    for i in range(corr.shape[0]):
        if reps and bool((corr[i, reps] >= 1.0 - tol).any()):
            continue
        reps.append(i)
    return reps


def effective_trials(
    rows: Sequence[NDArray[np.float64]],
    raw: int,
    *,
    max_series: int = NEFF_MAX_SERIES,
) -> EffectiveTrials:
    """EV-matched effective trial count over a run's retained MTM series.

    ``rows`` is the retained MTM daily series in the caller's deterministic
    (sorted-uid) order; ``raw`` is the run's distinct evaluated trial count.
    Non-flat rows (std > 0) are kept, capped at ``max_series`` by an evenly
    spaced subsample, then put in canonical ``tobytes`` order before the MC so
    the result does not depend on the input order or the eigenvector basis.
    """
    raw_rows = [np.asarray(r, dtype=np.float64) for r in rows]
    # Non-finite rows must be caught BEFORE the std()>0 filter, which silently
    # drops a NaN row (NaN > 0 is False) without a trace (review B4).
    non_finite = any(not bool(np.isfinite(r).all()) for r in raw_rows)
    lengths = {len(r) for r in raw_rows}
    kept = [r for r in raw_rows if r.std() > 0]
    non_flat = len(kept)
    subsampled_from: int | None = None
    if non_flat > max_series:
        subsampled_from = non_flat
        idx = np.unique(np.linspace(0, non_flat - 1, max_series).astype(int))
        kept = [kept[i] for i in idx]
    n_used = len(kept)

    neff_mean_corr: float | None = None
    neff_eig_pr: float | None = None
    if n_used >= 2 and len(lengths) == 1:
        corr = np.nan_to_num(np.corrcoef(np.vstack(kept)))
        rho_bar = (float(corr.sum()) - float(np.trace(corr))) / (n_used * (n_used - 1))
        denom = 1.0 + (n_used - 1) * rho_bar
        if denom > 0:
            neff_mean_corr = float(raw) / denom
        ev = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
        sq = float((ev**2).sum())
        if sq > 0:
            neff_eig_pr = float(ev.sum() ** 2 / sq) * raw / n_used

    source: Literal["ev_mc", "fallback_raw"] = "fallback_raw"
    fallback_reason: FallbackReason | None = None
    neff_ev: float | None = None
    run_n_eff = max(1, raw)
    # Distinct count is only computed when the EV actually runs; any fallback
    # fails closed without collapsing.
    n_distinct = n_used
    if non_finite:
        fallback_reason = "non_finite"
    elif n_used < 2:
        fallback_reason = "fewer_than_2_series"
    elif len({len(r) for r in kept}) != 1:
        fallback_reason = "unequal_series_lengths"
    elif min(len(r) for r in kept) < NEFF_MIN_DAYS:
        fallback_reason = "too_short"
    else:
        try:
            # Symmetric square root V diag(sqrt(w)) V.T is basis-invariant, so the
            # target cannot drift with the BLAS choice of V inside degenerate
            # eigenspaces (review B2); canonical row order fixes the z pairing.
            ordered = sorted(kept, key=lambda r: r.tobytes())
            corr = np.asarray(
                np.nan_to_num(np.corrcoef(np.vstack(ordered))), dtype=np.float64
            )
            # Collapse exactly-identical series BEFORE the MC so a degenerate
            # all-identical input cannot be decided by Monte-Carlo noise: one
            # distinct series gives n_m == 1 analytically (chief ruling on R1).
            reps = _distinct_indices(corr)
            n_distinct = len(reps)
            if n_distinct == 1:
                n_m = 1
            else:
                sub = corr[np.ix_(reps, reps)]
                w, vec = np.linalg.eigh(sub)
                root = (vec * np.sqrt(np.clip(w, 0.0, None))) @ vec.T
                z = np.random.default_rng(NEFF_EV_SEED).standard_normal(
                    (NEFF_EV_DRAWS, n_distinct)
                )
                target = float((z @ root).max(axis=1).mean())
                n_m = expected_max_inverse(target)
            # Scaling keeps the PRE-collapse retained count (n_used).
            neff_ev = n_m * raw / n_used
            run_n_eff = min(max(1, math.ceil(neff_ev)), max(1, raw))
            source = "ev_mc"
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            fallback_reason = "estimator_error"

    return EffectiveTrials(
        raw=raw,
        non_flat=non_flat,
        n_used=n_used,
        n_distinct=n_distinct,
        subsampled_from=subsampled_from,
        neff_mean_corr=neff_mean_corr,
        neff_eig_pr=neff_eig_pr,
        neff_ev=neff_ev,
        run_n_eff=run_n_eff,
        source=source,
        fallback_reason=fallback_reason,
    )
