"""Evidence statistics for the shadow forward tracker (pure numpy/scipy, no I/O).

Input is the forward daily mark-to-market return series ``x_1..x_n`` (simple returns), day 1 = the first
forward day F. Two rules decide a row:

- Evidence (pass): complete 7-day block sums ``y_j``, lag-1 HAC standard deviation (lag-1 term floored at
  0), and the half-normal mixture e-process of Robbins (1970) with ``tau = 0.1 * sqrt(7)``. Below
  ``MIN_BLOCKS`` blocks, or with ``s_HAC = 0``, E = 1. ``s_HAC <= S_REL_FLOOR * max|y|`` counts as 0: block
  sums equal up to float rounding (0.1 + 0.2 vs 0.3) would otherwise give s_HAC ~ 1e-17 and E ~ 1e304.
  Evaluated only at block ends, so the verdict day does not depend on update cadence. Family F1 = ``SLOTS``
  slots, online Bonferroni with alpha_k = ``ALPHA / SLOTS`` = 0.01, so a slot passes at E >= ``E_PASS`` = 100.
- Futility (daily): Lo (2002) SE of the annualised Sharpe; futile when ``n >= 90`` and the one-sided 95 %
  upper bound < 0.5, when the max drawdown on daily closes exceeds the row's limit, or on an account bust.

Same-day precedence: futile (drawdown/bust) > passed > futile (Sharpe bound).

"No variation" (UI/tracker note) = ``ForwardStats.sharpe_ann is None`` with ``days >= 2``: every return is
identical, so Sharpe and E are undefined (E = 1.0) and the row is never Sharpe-futile. Equality is exact
(max == min), because float rounding gives a constant series a tiny non-zero SD (200 x 0.01 -> 1.7e-18).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.special import log_ndtr

from vibe_quant.metrics import SHARPE_UNIT

BLOCK_DAYS: Final = 7
TAU_B: Final = 0.1 * math.sqrt(7)
MIN_BLOCKS: Final = 26
ALPHA: Final = 0.05
SLOTS: Final = 5
E_PASS: Final = 100.0  # 1 / (ALPHA / SLOTS)
LOG_E_CAP: Final = 700.0
FUTILITY_MIN_DAYS: Final = 90
FUTILITY_SR: Final = 0.5
Z95: Final = 1.6448536269514722
DAYS_PER_YEAR: Final = 365
# s_HAC <= S_REL_FLOOR * max|y| is float rounding -> 0. log_e_path's running-sum variance leaves up to
# 1.4e-7 * max|y| on constant series (MEASURED, 20,000 constants x 261 blocks), so 1e-9 is too tight there.
S_REL_FLOOR: Final = 1e-6


@dataclass(frozen=True)
class ForwardStats:
    """Forward-window statistics; Sharpe fields are None when the series has no variation."""

    days: int
    total_return: float
    sharpe_ann: float | None
    se_ann: float | None
    ub95: float | None
    max_dd: float
    blocks: int
    e_value: float


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite forward return")
    return arr


def block_sums(x: np.ndarray) -> np.ndarray:
    """Sums of complete 7-day blocks over the last axis (days 1..7 = block 1); a partial block is dropped."""
    arr = np.asarray(x, dtype=float)
    n_b = arr.shape[-1] // BLOCK_DAYS
    return arr[..., : n_b * BLOCK_DAYS].reshape(*arr.shape[:-1], n_b, BLOCK_DAYS).sum(axis=-1)


def s_hac(y: np.ndarray) -> float:
    """Lag-1 HAC SD of block sums: (sum d^2 + 2 max(0, sum d_j d_{j-1})) / (n-1).

    0 for a series constant up to float rounding (SD <= ``S_REL_FLOOR * max|y|``).
    """
    arr = np.asarray(y, dtype=float)
    if len(arr) < 2:
        return 0.0
    d = arr - arr.mean()
    num = float((d * d).sum()) + 2.0 * max(0.0, float((d[1:] * d[:-1]).sum()))
    s = math.sqrt(num / (len(arr) - 1))
    return 0.0 if s <= S_REL_FLOOR * float(np.abs(arr).max()) else s


def log_e(n_b: int, s: float) -> float:
    """log E of the half-normal mixture (tau = TAU_B) at n_b blocks and standardised sum S."""
    v = 1.0 + n_b * TAU_B * TAU_B
    b = TAU_B * s / math.sqrt(v)
    return math.log(2.0) - 0.5 * math.log(v) + 0.5 * b * b + float(log_ndtr(b))


def e_value(x: np.ndarray) -> float:
    """Evidence E of a forward daily series (complete blocks only)."""
    y = block_sums(_finite(x))
    if len(y) < MIN_BLOCKS:
        return 1.0
    s = s_hac(y)
    if not s > 0.0:
        return 1.0
    return math.exp(min(log_e(len(y), float(y.sum()) / s), LOG_E_CAP))


def log_e_path(y: np.ndarray) -> np.ndarray:
    """log E at every block end of block sums ``y`` (paths x blocks), vectorised for null simulations.

    Same rule as ``e_value`` via running sums (agrees to float rounding, not bitwise); 0 before MIN_BLOCKS
    and where s_HAC = 0 (same ``S_REL_FLOOR`` relative floor on the running max|y|).
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    n_paths, n_blocks = y.shape
    n = np.arange(1, n_blocks + 1, dtype=float)
    cs = np.cumsum(y, axis=1)
    csq = np.cumsum(y * y, axis=1)
    m = cs / n
    nm1 = np.maximum(n - 1, 1)
    var = (csq - cs * cs / n) / nm1
    cp = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(y[:, 1:] * y[:, :-1], axis=1)], axis=1)
    lag_sum = (
        cp - m * ((cs - y[:, :1]) + (cs - y)) + (n - 1) * m * m
    )  # sum_{j>=2} (y_j - m)(y_{j-1} - m)
    var = var + 2.0 * np.maximum(lag_sum, 0.0) / nm1
    sd = np.sqrt(np.maximum(var, 0.0))
    has_var = sd > S_REL_FLOOR * np.maximum.accumulate(np.abs(y), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(has_var, cs / sd, 0.0)
    v = 1.0 + n * TAU_B * TAU_B
    b = TAU_B * s / np.sqrt(v)
    le = np.minimum(math.log(2.0) - 0.5 * np.log(v) + 0.5 * b * b + log_ndtr(b), LOG_E_CAP)
    le[:, : MIN_BLOCKS - 1] = 0.0
    return np.where(has_var, le, 0.0)


def _sharpe(x: np.ndarray) -> tuple[float | None, float | None, float | None]:
    """(annual Sharpe, Lo SE, one-sided 95 % upper bound); all None without variation."""
    n = len(x)
    if n < 2 or x.max() == x.min():
        return None, None, None
    mu = float(np.mean(x)) / float(np.std(x, ddof=1))
    sr = mu * math.sqrt(DAYS_PER_YEAR)
    se = math.sqrt(DAYS_PER_YEAR * (1.0 + mu * mu / 2.0) / n)
    return sr, se, sr + Z95 * se


def _equity(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Daily-close equity from 1.0 and its drawdown (the peak includes the starting 1.0)."""
    eq = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(np.maximum(eq, 1.0))
    return eq, 1.0 - eq / peak


def forward_stats(x: np.ndarray) -> ForwardStats:
    arr = _finite(x)
    sr, se, ub = _sharpe(arr)
    eq, dd = _equity(arr)
    return ForwardStats(
        days=len(arr),
        total_return=float(eq[-1]) - 1.0 if len(arr) else 0.0,
        sharpe_ann=sr,
        se_ann=se,
        ub95=ub,
        max_dd=float(dd.max()) if len(arr) else 0.0,
        blocks=len(arr) // BLOCK_DAYS,
        e_value=e_value(arr),
    )


def _sharpe_futility(days: int, sr: float | None, ub: float | None) -> str | None:
    if days >= FUTILITY_MIN_DAYS and sr is not None and ub is not None and ub < FUTILITY_SR:
        return (
            f"forward Sharpe {sr:.2f} ({SHARPE_UNIT}), 95% upper bound {ub:.4f} < {FUTILITY_SR} after {days} days"
        )
    return None


def futility_reason(stats: ForwardStats, max_dd_limit: float) -> str | None:
    """Drawdown or Sharpe-bound futility text (drawdown first), else None. ``ub95 == 0.5`` is not futile."""
    if stats.max_dd > max_dd_limit:
        return f"max drawdown {stats.max_dd:.2%} > limit {max_dd_limit:.2%}"
    return _sharpe_futility(stats.days, stats.sharpe_ann, stats.ub95)


def first_verdict(
    x: np.ndarray, *, start_index: int, max_dd_limit: float, can_pass: bool
) -> tuple[int, str, str] | None:
    """First verdict (0-based day index, state, reason) on days ``start_index..len(x)-1``, else None.

    Each day is decided on the prefix ``x[: i + 1]`` alone, so the verdict never depends on how the days
    were batched. Precedence: futile (bust, drawdown) > passed (block end, ``can_pass``) > futile (Sharpe).
    """
    arr = _finite(x)
    eq, dd = _equity(arr)
    for i in range(start_index, len(arr)):
        if eq[i] <= 0.0:
            return i, "futile", f"account bust on forward day {i + 1}"
        if dd[i] > max_dd_limit:
            return i, "futile", f"max drawdown {dd[i]:.2%} > limit {max_dd_limit:.2%}"
        prefix = arr[: i + 1]
        if can_pass and (i + 1) % BLOCK_DAYS == 0:
            e = e_value(prefix)
            if e >= E_PASS:
                return i, "passed", f"evidence {e:.1f} >= {E_PASS:g}"
        if i + 1 >= FUTILITY_MIN_DAYS:
            sr, _se, ub = _sharpe(prefix)
            reason = _sharpe_futility(i + 1, sr, ub)
            if reason is not None:
                return i, "futile", reason
    return None
