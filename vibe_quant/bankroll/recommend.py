"""Row eligibility and the bankroll recommendation (always a tested row).

The recommendation is the largest requested ``k`` of an ELIGIBLE tested row with
``k <= C``, ``C = min(f_half, user_cap)``; never an interpolated or rounded value. It is
null when the baseline cannot make Kelly actionable, when the mean is not positive
(``C = 0``), or when no eligible row lies at or under ``C``. Every function returns a
reason string (``None`` = passes) so the report can show why a row or Kelly was refused.

Imports no engine, validation or DB module (import isolation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vibe_quant.bankroll.stress import liquidation_count

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibe_quant.bankroll.kelly import KellyStats
    from vibe_quant.bankroll.stress import RowOutcome

BINDING_TEXT: dict[str, str] = {
    "dd_limit": "Kelly overstates: drawdown limit binds",
    "row_ineligible": "Larger tested sizes are not eligible",
    "user_cap": "Your size cap binds",
    "half_kelly": "Half-Kelly binds",
}


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _bust_or_divergence(row: RowOutcome, *, divergence: bool) -> str | None:
    liq = row.liquidation or {}
    if row.bust_ts is not None or liq.get("bust_ts") is not None:
        return "bust"
    if divergence and (row.path_divergence or bool(liq.get("path_divergence"))):
        return "path_divergence"
    return None


def row_eligibility(row: RowOutcome, *, dd_limit: float) -> str | None:
    """Why ``row`` cannot be recommended, or ``None`` when it is eligible.

    Eligible: ran ok, not force-stopped, trades > 0, finite return > 0, MTM paths ok (no
    closed-trade fallback), finite MTM DD in [0, 1] and <= ``dd_limit``, liquidation
    modeled with count 0, no bust and no path divergence.
    """
    if row.engine_force_stopped:
        return "engine_force_stopped"
    if row.status != "ok":
        return row.status
    if row.trades <= 0:
        return "no_trades"
    if not _finite(row.total_return):
        return "non_finite"
    if not row.total_return > 0:  # type: ignore[operator]  # finite above
        return "non_positive_return"
    if row.mtm_paths != "ok":
        return "mtm_unavailable"
    dd = row.mtm_dd
    if dd is None or not math.isfinite(dd):
        return "non_finite"
    if not 0.0 <= dd <= 1.0:
        return "dd_out_of_range"
    if dd > dd_limit:
        return "dd_over_limit"
    if row.liquidation is None:
        return "liquidation_not_modeled"
    count = liquidation_count(row.liquidation)
    if count is None:
        return "liquidation_not_modeled"
    if count > 0:
        return "liquidated"
    return _bust_or_divergence(row, divergence=True)


def kelly_actionable(baseline: RowOutcome, stats: KellyStats | None) -> str | None:
    """Why the baseline's Kelly numbers cannot drive a recommendation, or ``None``.

    ``stats is None`` means the baseline produced no daily series. The baseline may run
    with liquidation off; its MTM DD must still be finite and below 1 (an intraday
    wipeout whose daily closes recover is not actionable), and ``bust_ts`` is checked on
    its own. With liquidation modeled, the count must be 0 and paths must not diverge.
    """
    if baseline.engine_force_stopped:
        return "engine_force_stopped"
    if baseline.status != "ok":
        return baseline.status
    if baseline.trades <= 0:
        return "no_trades"
    if baseline.mtm_paths != "ok":
        return "mtm_unavailable"
    if stats is None:
        return "daily_unavailable"
    dd = baseline.mtm_dd
    if dd is None or not math.isfinite(dd):
        return "non_finite"
    if dd < 0.0:
        return "dd_out_of_range"
    if dd >= 1.0:
        return "intraday_wipeout"
    modeled = baseline.liquidation is not None
    blocked = _bust_or_divergence(baseline, divergence=modeled)
    if blocked is not None:
        return blocked
    if baseline.liquidation is not None:
        count = liquidation_count(baseline.liquidation)
        if count is None:
            return "liquidation_not_modeled"
        if count > 0:
            return "liquidated"
    return stats.reason


@dataclass(frozen=True)
class Recommendation:
    """The recommended tested ``k`` (or ``None`` with ``reason``) and every row's verdict.

    ``ceiling`` is ``min(f_half, user_cap)`` (0 for a non-positive mean; ``None`` when
    Kelly is not actionable). ``binding`` names what stopped a larger pick.
    """

    k: float | None
    ceiling: float | None
    reason: str | None
    binding: str | None
    text: str | None
    row_reasons: tuple[tuple[float, str | None], ...]


def recommend(
    rows: Sequence[RowOutcome],
    stats: KellyStats | None,
    *,
    baseline: RowOutcome,
    user_cap: float,
    dd_limit: float,
) -> Recommendation:
    """Recommend the largest eligible tested ``k`` at or under ``min(f_half, user_cap)``."""
    row_reasons = tuple((row.k, row_eligibility(row, dd_limit=dd_limit)) for row in rows)
    refused = kelly_actionable(baseline, stats)
    if refused is not None or stats is None or stats.f_half is None:
        return Recommendation(None, None, refused or "daily_unavailable", None, None, row_reasons)
    f_half = stats.f_half
    if not f_half > 0:
        return Recommendation(None, 0.0, "non_positive_mean", None, None, row_reasons)
    ceiling = min(f_half, user_cap)
    under = [k for k, reason in row_reasons if reason is None and k <= ceiling]
    if not under:
        return Recommendation(None, ceiling, "no_eligible_row", None, None, row_reasons)
    best = max(under)
    blockers = {reason for k, reason in row_reasons if reason is not None and best < k <= ceiling}
    if "dd_over_limit" in blockers:
        binding = "dd_limit"
    elif blockers:
        binding = "row_ineligible"
    elif user_cap < f_half:
        binding = "user_cap"
    else:
        binding = "half_kelly"
    return Recommendation(best, ceiling, None, binding, BINDING_TEXT[binding], row_reasons)
