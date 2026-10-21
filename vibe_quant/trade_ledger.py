"""Shared trade-ledger helpers for the screening and validation extractors.

The screening runner (``NTScreeningRunner._extract_metrics``) and the validation
extractor (``extract_stats`` / ``_extract_return_moments``) used to carry two copies
of the same NT stat parse, the same return moments and the same closed/open position
split. Bead vibe-quant-rabmo.2 L4 moves them here as behaviour-preserving pure moves:
no number, log line, message or error precedence changes.

The tier differences stay named in this module:

* the NT stat parse logs a WARNING in screening and a DEBUG in validation for a
  non-numeric stat, and only validation reads ``Avg Winner`` / ``Avg Loser``;
* screening returns the per-trade returns with its moments, validation leaves
  ``ValidationResult``'s defaults (``0.0`` / ``3.0``) unless it sets them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from vibe_quant.validation.results import ValidationResult

logger = logging.getLogger(__name__)

Tier = Literal["screening", "validation"]


@dataclass(frozen=True)
class NtStats:
    """The NT aggregate stats both tiers read, ``None`` when a key is absent.

    ``total_return`` is already a fraction (NT reports percent), ``max_drawdown``
    is already absolute. Screening never reads ``avg_win`` / ``avg_loss``.
    """

    total_return: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    avg_win: float | None = None
    avg_loss: float | None = None


# The key sets are the ones each tier matched before the move: validation knows the
# extra stats it stores, so screening still logs them as unmatched at DEBUG.
_SCREENING_PNL_KEYS = frozenset(
    {
        "pnl (total)",
        "pnl% (total)",
        "sharpe",
        "sortino",
        "max drawdown",
        "win rate",
        "profit factor",
    }
)
_VALIDATION_PNL_KEYS = _SCREENING_PNL_KEYS | frozenset(
    {"expectancy", "avg winner", "avg loser", "long ratio"}
)
_SCREENING_RETURNS_KEYS = frozenset(
    {"sharpe", "sortino", "max drawdown", "win rate", "profit factor"}
)
_VALIDATION_RETURNS_KEYS = _SCREENING_RETURNS_KEYS | frozenset(
    {"expectancy", "avg winner", "avg loser", "long ratio"}
)


def parse_nt_stats(bt_result: Any, *, tier: Tier) -> NtStats:
    """Parse ``stats_pnls`` then ``stats_returns`` of an NT result into :class:`NtStats`.

    ``stats_pnls`` wins: a field it populated keeps even a legitimate ``0.0``, and
    only a field it left absent falls back to ``stats_returns``. The per-tier log
    text and level are unchanged from the two copies this replaces.
    """
    validation = tier == "validation"
    known_pnl = _VALIDATION_PNL_KEYS if validation else _SCREENING_PNL_KEYS
    known_returns = _VALIDATION_RETURNS_KEYS if validation else _SCREENING_RETURNS_KEYS

    values: dict[str, float] = {}
    stats_pnls = bt_result.stats_pnls or {}
    for _currency, pnl_stats in stats_pnls.items():
        for key, value in pnl_stats.items():
            if value is None:
                continue
            key_lower = key.lower()
            try:
                fval = float(value)
            except (ValueError, TypeError):
                if validation:
                    logger.debug("Non-numeric PnL stat skipped: %s = %r", key, value)
                else:
                    logger.warning("Could not convert PnL stat %s=%r to float", key, value)
                continue
            if key_lower == "pnl% (total)":
                # NT reports as percentage; store as fraction
                values["total_return"] = fval / 100.0
            elif "sharpe" in key_lower:
                values["sharpe"] = fval
            elif "sortino" in key_lower:
                values["sortino"] = fval
            elif key_lower == "max drawdown":
                values["max_drawdown"] = abs(fval)
            elif key_lower == "win rate":
                values["win_rate"] = fval
            elif validation and key_lower == "avg winner":
                values["avg_win"] = fval
            elif validation and key_lower == "avg loser":
                values["avg_loss"] = fval
            elif not any(k in key_lower for k in known_pnl):
                logger.debug("Unmatched PnL stats key: %s = %s", key, value)

    # Frozen before the returns pass: only stats_pnls fields block the fallback, so
    # the last matching returns key keeps winning (as in the two copies replaced).
    populated = frozenset(values)
    stats_returns = bt_result.stats_returns or {}
    for key, value in stats_returns.items():
        if value is None:
            continue
        key_lower = key.lower()
        try:
            fval = float(value)
        except (ValueError, TypeError):
            if validation:
                logger.debug("Non-numeric returns stat skipped: %s = %r", key, value)
            else:
                logger.warning("Could not convert returns stat %s=%r to float", key, value)
            continue
        if "sharpe" in key_lower and "sharpe" not in populated:
            values["sharpe"] = fval
        elif "sortino" in key_lower and "sortino" not in populated:
            values["sortino"] = fval
        elif "max drawdown" in key_lower and "max_drawdown" not in populated:
            values["max_drawdown"] = abs(fval)
        elif key_lower == "win rate" and "win_rate" not in populated:
            values["win_rate"] = fval
        elif not any(k in key_lower for k in known_returns):
            logger.debug("Unmatched returns stats key: %s = %s", key, value)

    return NtStats(
        total_return=values.get("total_return"),
        sharpe=values.get("sharpe"),
        sortino=values.get("sortino"),
        max_drawdown=values.get("max_drawdown"),
        win_rate=values.get("win_rate"),
        avg_win=values.get("avg_win"),
        avg_loss=values.get("avg_loss"),
    )


def _cache_positions(engine: Any) -> list[Any]:
    """Every position incarnation in the engine cache (see ``extraction.all_positions``)."""
    cache = engine.kernel.cache
    return list(cache.positions()) + list(cache.position_snapshots())


def split_positions(engine: Any) -> tuple[list[Any], list[Any]]:
    """``(closed, still_open)`` of the engine cache; an unreadable cache is logged.

    NT netting mode removes closed positions from the main index, so both
    ``positions()`` and ``position_snapshots()`` are needed. A position that is
    neither closed nor open is in neither list, exactly as before the move.
    """
    try:
        everything = _cache_positions(engine)
        closed = [p for p in everything if p.is_closed]
        still_open = [p for p in everything if not p.is_closed and getattr(p, "is_open", False)]
    except Exception:
        logger.warning("Could not read positions from engine cache", exc_info=True)
        closed, still_open = [], []
    return closed, still_open


def closed_positions(engine: Any) -> list[Any]:
    """Closed positions of the engine cache: reads no notional (see ``closed_returns``)."""
    return [p for p in _cache_positions(engine) if p.is_closed]


def closed_returns(closed: list[Any]) -> list[float]:
    """Per-trade return of each closed position: ``realized pnl / entry notional``.

    A zero-notional position is dropped. Callers must place the ``n < 4`` cutoff
    BEFORE this call, so no notional is read when the moments are unavailable.
    """
    returns: list[float] = []
    for pos in closed:
        # Use peak_qty (not quantity which is 0 for closed positions)
        entry_val = abs(float(pos.peak_qty) * float(pos.avg_px_open))
        if entry_val > 0:
            returns.append(float(pos.realized_pnl) / entry_val)
    return returns


def moments_from_returns(returns: list[float]) -> tuple[float, float, tuple[float, ...]]:
    """Skewness, kurtosis and per-trade returns of ``returns``.

    Adjusted Fisher-Pearson G1 (skewness) and G2 (excess kurtosis) formulas — the
    same as scipy.stats.skew(bias=False) and scipy.stats.kurtosis(bias=False). Returns
    full kurtosis (G2 + 3). Falls back to (0.0, 3.0, tuple) below four values or a
    flat series.
    """
    n = len(returns)
    if n < 4:
        return 0.0, 3.0, tuple(returns)

    mean = sum(returns) / n
    diffs = [r - mean for r in returns]
    # Biased central moments (population moments)
    m2 = sum(d * d for d in diffs) / n
    if m2 == 0:
        return 0.0, 3.0, tuple(returns)

    m3 = sum(d**3 for d in diffs) / n
    m4 = sum(d**4 for d in diffs) / n

    # Adjusted Fisher-Pearson G1 skewness (bias-corrected)
    # G1 = sqrt(n(n-1)) / (n-2) * (m3 / m2^1.5)
    skewness = (math.sqrt(n * (n - 1)) / (n - 2)) * (m3 / m2**1.5)

    # Adjusted Fisher G2 excess kurtosis (bias-corrected)
    # G2 = (n+1)(n-1) / ((n-2)(n-3)) * m4/m2² - 3(n-1)² / ((n-2)(n-3))
    excess = ((n + 1) * (n - 1) * m4) / (
        (n - 2) * (n - 3) * m2**2
    ) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

    # Full kurtosis = excess + 3 (DSR expects gamma_4, not excess)
    kurtosis = excess + 3.0

    # Clamp kurtosis to valid range (>=1 per DSR validator)
    kurtosis = max(1.0, kurtosis)

    return round(skewness, 4), round(kurtosis, 4), tuple(returns)


def screening_moments(engine: Any) -> tuple[float, float, tuple[float, ...]]:
    """Screening's return moments: the ``n < 4`` cutoff precedes every notional read."""
    try:
        closed = closed_positions(engine)
        if len(closed) < 4:
            return 0.0, 3.0, ()
        return moments_from_returns(closed_returns(closed))
    except Exception:
        logger.warning("Could not compute return moments", exc_info=True)
        return 0.0, 3.0, ()


def screening_liquidation_moments(
    returns: list[float], n_closed: int
) -> tuple[float, float, tuple[float, ...]]:
    """The return moments of the liquidation-adjusted trades (same rules as the NT path)."""
    try:
        if n_closed < 4:
            return 0.0, 3.0, ()
        return moments_from_returns(returns)
    except Exception:
        logger.warning("Could not compute return moments", exc_info=True)
        return 0.0, 3.0, ()


def apply_validation_moments(result: ValidationResult, returns: list[float]) -> None:
    """Skewness/kurtosis of per-trade returns (``realized pnl / entry notional``) on result."""
    try:
        n = len(returns)
        if n < 4:
            return

        mean = sum(returns) / n
        diffs = [r - mean for r in returns]
        m2 = sum(d * d for d in diffs) / n
        if m2 == 0:
            return

        m3 = sum(d**3 for d in diffs) / n
        m4 = sum(d**4 for d in diffs) / n

        result.skewness = round(
            (math.sqrt(n * (n - 1)) / (n - 2)) * (m3 / m2**1.5), 4
        )
        excess = ((n + 1) * (n - 1) * m4) / (
            (n - 2) * (n - 3) * m2**2
        ) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        result.kurtosis = round(max(1.0, excess + 3.0), 4)
    except Exception:
        logger.warning("Could not compute return moments", exc_info=True)


def validation_engine_moments(result: ValidationResult, engine: Any) -> None:
    """Validation's engine moments: set on ``result`` only, leaving its defaults on failure."""
    try:
        closed = closed_positions(engine)
        if len(closed) < 4:
            return
        returns = closed_returns(closed)
    except Exception:
        logger.warning("Could not compute return moments", exc_info=True)
        return
    apply_validation_moments(result, returns)
