"""Continuous Kelly, half-Kelly and vol-target numbers on daily MTM returns.

``f* = mu / sigma^2`` on the run's own daily mark-to-market returns, read as a multiple of
the validated sizing (the series already carries that sizing). Nothing here recommends a
size: :mod:`vibe_quant.bankroll.recommend` turns these numbers into a tested row.

Every number is computed under ``np.errstate(all="raise")``: an overflow, underflow or
invalid operation, or any non-finite result, gives ``status="numeric"`` with every value
``None``, so a payload always serializes with ``json.dumps(allow_nan=False)``.

Imports no engine, validation or DB module (import isolation).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

DAY_NS = 86_400 * 1_000_000_000
"""One UTC day in nanoseconds (the grid of :func:`vibe_quant.metrics.mark_to_market_daily_returns`)."""

MIN_DAYS = 60
DEFAULT_TARGET_VOL = 0.20
DAYS_PER_YEAR = 365
"""Crypto perps trade every calendar day."""

FAT_TAIL_EXCESS_KURTOSIS = 3.0
"""Excess kurtosis strictly above this flags a fat-tailed daily series."""

KellyStatus = Literal["ok", "insufficient_data", "zero_variance", "numeric"]


def day_buckets(start_ns: int, end_ns: int) -> int:
    """UTC day buckets from ``start_ns // DAY`` to ``(end_ns - 1) // DAY`` inclusive.

    Partial first and last days count; an end exactly at midnight adds no bucket. Same
    grid as :func:`vibe_quant.metrics.mark_to_market_daily_returns`.
    """
    first = start_ns // DAY_NS
    return max(first, (end_ns - 1) // DAY_NS) - first + 1


def is_fat_tail(excess_kurtosis: float) -> bool:
    """Whether the excess kurtosis is strictly above :data:`FAT_TAIL_EXCESS_KURTOSIS`."""
    return excess_kurtosis > FAT_TAIL_EXCESS_KURTOSIS


@dataclass(frozen=True)
class KellyStats:
    """Kelly numbers of one daily series; every value is ``None`` unless ``status == "ok"``.

    ``f_full``/``f_half``/``vol_target`` are multiples of the validated sizing.
    ``vol_target = target_vol / sigma_ann`` is displayed only, never recommended.
    """

    status: KellyStatus
    n: int
    target_vol: float
    mean: float | None = None
    variance: float | None = None
    sigma_ann: float | None = None
    f_full: float | None = None
    f_half: float | None = None
    vol_target: float | None = None
    excess_kurtosis: float | None = None
    fat_tail: bool | None = None

    @property
    def reason(self) -> str | None:
        """Why the values are unavailable (``None`` when ``status == "ok"``)."""
        return None if self.status == "ok" else self.status

    def to_dict(self) -> dict[str, object]:
        """JSON-safe payload: unavailable values are ``null`` plus ``reason``."""
        return {**asdict(self), "reason": self.reason}


def kelly_stats(
    daily: Sequence[float], *, min_days: int = MIN_DAYS, target_vol: float = DEFAULT_TARGET_VOL
) -> KellyStats:
    """Kelly statistics of ``daily`` (one return per UTC bucket, see :func:`day_buckets`).

    ``mean``, sample ``variance`` (ddof=1), ``sigma_ann = sqrt(365 * variance)``,
    ``f_full = mean / variance``, ``f_half = f_full / 2``, ``vol_target = target_vol /
    sigma_ann`` and population excess kurtosis ``m4 / m2^2 - 3``.

    Fewer than ``min_days`` values -> ``insufficient_data``; all values equal ->
    ``zero_variance``; any numeric fault -> ``numeric``. A non-positive mean is reported
    as computed (``f_full <= 0``); the recommendation refuses it.
    """
    n = len(daily)
    if n < max(min_days, 2):
        return KellyStats(status="insufficient_data", n=n, target_vol=target_vol)
    x = np.asarray(daily, dtype=np.float64)
    if not bool(np.all(np.isfinite(x))):
        return KellyStats(status="numeric", n=n, target_vol=target_vol)
    if bool(np.all(x == x[0])):
        return KellyStats(status="zero_variance", n=n, target_vol=target_vol)
    try:
        with np.errstate(all="raise"):
            mean = np.mean(x)
            dev = x - mean
            sq = dev * dev
            variance = np.sum(sq) / np.float64(n - 1)
            sigma_ann = np.sqrt(np.float64(DAYS_PER_YEAR) * variance)
            f_full = mean / variance
            f_half = f_full / np.float64(2.0)
            vol_target = np.float64(target_vol) / sigma_ann
            m2 = np.mean(sq)
            m4 = np.mean(sq * sq)
            excess = m4 / (m2 * m2) - np.float64(3.0)
    except (FloatingPointError, ZeroDivisionError):
        return KellyStats(status="numeric", n=n, target_vol=target_vol)
    values = [float(v) for v in (mean, variance, sigma_ann, f_full, f_half, vol_target, excess)]
    if not all(math.isfinite(v) for v in values):
        return KellyStats(status="numeric", n=n, target_vol=target_vol)
    mean_f, var_f, sigma_f, full_f, half_f, vt_f, excess_f = values
    return KellyStats(
        status="ok",
        n=n,
        target_vol=target_vol,
        mean=mean_f,
        variance=var_f,
        sigma_ann=sigma_f,
        f_full=full_f,
        f_half=half_f,
        vol_target=vt_f,
        excess_kurtosis=excess_f,
        fat_tail=is_fat_tail(excess_f),
    )
