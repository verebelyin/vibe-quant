"""Shared trade-ledger helpers for the screening and validation extractors.

The screening runner (``NTScreeningRunner._extract_metrics``) and the validation
extractor (``extract_stats`` / ``_extract_return_moments``) used to carry two copies
of the same NT stat parse, the same return moments and the same closed/open position
split. Bead vibe-quant-rabmo.2 L4 moves them here as behaviour-preserving pure moves:
no number, log line, message or error precedence changes.

L5 adds the one cost/accumulate loop both tiers run (``cost_positions`` then
``accumulate``). Every per-tier difference stays an explicit :class:`TierRules` value
(``SCREENING_RULES`` / ``VALIDATION_RULES``): the fee association, the open-mark PnL
association, the funding exit, the same-day cash order and the summary order. Each is
a real inconsistency kept bit-for-bit until its own follow-up bead closes it, and each
is witnessed by a literal test in ``tests/unit/test_trade_ledger.py`` (R5-R8).

The tier differences stay named in this module:

* the NT stat parse logs a WARNING in screening and a DEBUG in validation for a
  non-numeric stat, and only validation reads ``Avg Winner`` / ``Avg Loser``;
* screening returns the per-trade returns with its moments, validation leaves
  ``ValidationResult``'s defaults (``0.0`` / ``3.0``) unless it sets them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from vibe_quant.validation.extraction import OpenPositionMark
    from vibe_quant.validation.funding import FundingAccrual, FundingCalculator
    from vibe_quant.validation.results import ValidationResult

logger = logging.getLogger(__name__)

Tier = Literal["screening", "validation"]


@dataclass(frozen=True)
class TierRules:
    """The per-tier differences of the shared cost/accumulate loop, as named values.

    Each field preserves an inconsistency the tiers really have today; fixing one is
    its own bead (epic rabmo, section 8). The literals are witnessed by
    ``tests/unit/test_trade_ledger.py::test_*_witness`` with the base numbers.
    """

    #: neuvu.3 (closed): screening sums each commission leg by absolute value,
    #: validation sums the signed commissions then takes the absolute total.
    fee_rule: Literal["sum_of_abs", "abs_of_net"]
    #: Association of the open mark's unrealized PnL and estimated exit fee.
    mark_pnl: Literal["realized+(unreal-fee)", "(realized+unreal)-fee"]
    #: 6osqd D1 is screening-only: a boundary resting exit refines the funding exit
    #: only for an execution bar strictly longer than 8h.
    funding_exit: Literal["refined_over_8h", "ts_closed"]
    #: Same-day cash sums in append order: the realized leg first, or the funding first.
    cash_order: Literal["realized_first", "funding_first"]
    #: PF / win rate / liquidated return input order: cache order, or entry-time order.
    summary_order: Literal["loop", "entry_time"]
    #: Whether the ledger reports per-trade returns (validation's skew/kurtosis input).
    #: Off for screening: its moments come from the engine and the n<4 cutoff reads no
    #: notional (r3 F11).
    moment_returns: bool


SCREENING_RULES = TierRules(
    fee_rule="sum_of_abs",
    mark_pnl="realized+(unreal-fee)",
    funding_exit="refined_over_8h",
    cash_order="realized_first",
    summary_order="loop",
    moment_returns=False,
)

VALIDATION_RULES = TierRules(
    fee_rule="abs_of_net",
    mark_pnl="(realized+unreal)-fee",
    funding_exit="ts_closed",
    cash_order="funding_first",
    summary_order="entry_time",
    moment_returns=True,
)


def position_direction(pos: Any) -> str:
    """'LONG' / 'SHORT' from a position's entry side."""
    return "LONG" if getattr(pos.entry, "name", str(pos.entry)).upper() == "BUY" else "SHORT"


def position_key(pos: Any) -> str:
    """The overlay's key of a position (``<position id>@<ts_opened>``)."""
    return f"{pos.id}@{int(pos.ts_opened)}"


# One 8h funding period in ns: the smallest execution bar whose close can host a
# settlement that validation's 1m fill never charged (vibe-quant-6osqd). A bar of
# exactly this length is NOT refined (R9b: the guard is a strict ``>``).
# Inlined (validation.funding.FUNDING_PERIOD_HOURS) so this module imports without the
# validation package, whose __init__ imports the extractor that imports this module.
_FUNDING_REFINE_MIN_BAR_NS = 8 * 3_600 * 1_000_000_000


def _funding_close_ns(
    pos: Any,
    scan_fill_ts: Mapping[str, Mapping[int, int]] | None,
    execution_bar_ns: int,
) -> int:
    """Funding exit of a CLOSED position: its closing fill's refined ts, else ``ts_closed``.

    R9/R9b: only an execution bar strictly longer than 8h is refined; anything else (and an
    empty scan) returns the NT ``ts_closed`` without importing ``vibe_quant.liquidation``.
    R6: the CLOSING fill is ``events[len - 1]``, so a partial exit refined earlier does not
    move the funding exit.
    """
    nt = int(pos.ts_closed)
    if execution_bar_ns <= _FUNDING_REFINE_MIN_BAR_NS or not scan_fill_ts:
        return nt
    refined = (scan_fill_ts.get(position_key(pos)) or {}).get(len(pos.events) - 1)
    return nt if refined is None else refined


def accrue_position_funding(
    calculator: FundingCalculator,
    pos: Any,
    exit_ns: int | None = None,
) -> FundingAccrual:
    """Funding for one position (entry notional, held until close or ``exit_ns``)."""
    entry_price = float(pos.avg_px_open)
    quantity = float(pos.peak_qty)
    close_ns = exit_ns if exit_ns is not None else (int(pos.ts_closed) or None)
    return calculator.accrue(
        instrument_id=str(pos.instrument_id),
        direction=position_direction(pos),
        entry_notional=entry_price * quantity,
        entry_ns=int(pos.ts_opened),
        exit_ns=close_ns,
    )


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


@dataclass
class PositionCosts:
    """One trade's realized numbers before they become a trade record.

    ``realized_pnl`` is NT's (commissions included, modeled slippage/funding not); a
    liquidated position carries the overlay's account loss in it instead and sets the
    ``fee_legs`` / ``gross_pnl`` overrides. ``slippage_cost`` = ``entry_slippage`` (booked
    at the entry) + ``exit_slippage`` (booked at the exit). ``exit_price`` /
    ``exit_reason`` are the trade-record projection: set for a marked position and a
    liquidated one, ``None`` for an ordinary close (the caller reads ``avg_px_close``).
    """

    pos: Any
    mark: OpenPositionMark | None
    realized_pnl: float
    exit_price: float | None
    exit_ns: int
    exit_reason: str | None
    pos_fees: float
    slippage_cost: float
    accrual: FundingAccrual | None
    fee_legs: tuple[float, float] | None = None
    gross_pnl: float | None = None
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0


@dataclass
class Ledger:
    """Every cost accumulator the two tiers share, after one pass over ``costed``.

    ``net_pnls`` is in ``rules.summary_order`` order (the PF / win-rate / liquidated-return
    input), ``closed_net`` and ``moment_returns`` in cache order, and ``cash_events`` in
    ``rules.cash_order`` order (the daily-balance Sharpe input).
    """

    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0
    funding_fallbacks: int = 0
    unrealized_at_end: float = 0.0
    cash_events: list[tuple[int, float]] = field(default_factory=list)
    modeled_cash: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    net_pnls: list[float] = field(default_factory=list)
    closed_net: list[tuple[int, float]] = field(default_factory=list)
    moment_returns: list[float] = field(default_factory=list)


def cost_positions(
    trade_inputs: Iterable[tuple[Any, OpenPositionMark | None]],
    rules: TierRules,
    *,
    slippage: Any | None = None,
    funding: FundingCalculator | None = None,
    funding_by_key: Mapping[str, FundingAccrual] | None = None,
    scan_fill_ts: Mapping[str, Mapping[int, int]] | None = None,
    execution_bar_ns: int = 0,
) -> list[PositionCosts]:
    """Each position's realized numbers and modeled costs, in input order.

    Pass 1 of both extractors. ``slippage`` is validation's post-fill model (screening
    models slippage inside the fill prices, so it passes ``None`` and every slip is 0.0);
    ``funding_by_key`` reuses a liquidation scan's accruals. The per-tier fee and funding
    exit associations are ``rules`` values.
    """
    costed: list[PositionCosts] = []
    for pos, mark in trade_inputs:
        commands = list(pos.commissions())
        pos_fees = (
            sum(abs(float(c)) for c in commands)
            if rules.fee_rule == "sum_of_abs"
            else sum(float(c) for c in commands)
        )
        if mark is None:
            realized_pnl = float(pos.realized_pnl)
            exit_ns = int(pos.ts_closed)
            exit_price = None
            exit_reason = None
            if rules.funding_exit == "refined_over_8h":
                funding_exit_ns = _funding_close_ns(pos, scan_fill_ts, execution_bar_ns)
            else:
                funding_exit_ns = exit_ns
        else:
            if rules.mark_pnl == "realized+(unreal-fee)":
                realized_pnl = float(pos.realized_pnl) + (mark.unrealized - mark.exit_fee)
            else:
                realized_pnl = float(pos.realized_pnl) + mark.unrealized - mark.exit_fee
            exit_ns = mark.ts
            exit_price = mark.price
            exit_reason = "end_of_data"
            pos_fees += mark.exit_fee
            funding_exit_ns = mark.ts
        # Post-hoc funding accrual from archived rates (positive = paid). A recomputed run
        # reuses the scan's accrual: the ledger the overlay built carries exactly that cash.
        # A liquidation scan already accrued it (to this very exit ts for closed positions
        # and marked survivors): reuse, the ledger carries exactly that funding.
        accrual = None
        if funding_by_key:
            accrual = funding_by_key.get(position_key(pos))
        if accrual is None and funding is not None:
            accrual = accrue_position_funding(funding, pos, exit_ns=funding_exit_ns)
        entry_slip = 0.0
        exit_slip = 0.0
        if slippage is not None:
            entry_slip = slippage.leg(pos, float(pos.avg_px_open))
            exit_slip = slippage.leg(
                pos,
                float(pos.avg_px_close) if exit_price is None else exit_price,
            )
        costed.append(
            PositionCosts(
                pos=pos,
                mark=mark,
                realized_pnl=realized_pnl,
                exit_price=exit_price,
                exit_ns=exit_ns,
                exit_reason=exit_reason,
                pos_fees=pos_fees,
                slippage_cost=entry_slip + exit_slip,
                accrual=accrual,
                entry_slippage=entry_slip,
                exit_slippage=exit_slip,
            )
        )
    return costed


def accumulate(
    costed: Iterable[PositionCosts],
    rules: TierRules,
    *,
    per_leg_cash: bool = False,
) -> Ledger:
    """One pass over ``costed`` into a :class:`Ledger` (pass 2's accounting).

    ``per_leg_cash`` is validation's liquidation-overlay shape: the overlay books each
    slippage leg at its own instant, so a recomputed run reports the entry leg at
    ``ts_opened`` and ``realized - exit_slippage`` at the exit instead of one lump.
    It is a per-RUN input, not a tier rule. Nothing else here is per-run: every
    remaining difference is a ``rules`` value, and each summation keeps the base order.
    """
    ledger = Ledger()
    nets: list[float] = []
    entries: list[tuple[int, float]] = []
    for item in costed:
        pos, mark = item.pos, item.mark
        instrument_id = str(pos.instrument_id)
        # Screening creates the bucket for every position; an empty one is invisible to
        # the mark-to-market ledger (which already holds every traded instrument).
        ledger.modeled_cash.setdefault(instrument_id, [])
        if mark is not None:
            ledger.unrealized_at_end += mark.unrealized - mark.exit_fee
            ledger.modeled_cash[instrument_id].append((mark.ts, -mark.exit_fee))
        if rules.fee_rule == "sum_of_abs":
            # R5: screening charges each commission leg by absolute value and the mark's
            # exit fee as its own add; validation takes one absolute total (neuvu.3).
            ledger.total_fees += sum(abs(float(c)) for c in pos.commissions())
            if mark is not None:
                ledger.total_fees += mark.exit_fee
        else:
            ledger.total_fees += abs(item.pos_fees)
        ledger.total_slippage += item.slippage_cost

        funding_fees = 0.0
        funding_events: list[tuple[int, float]] = []
        if item.accrual is not None:
            funding_fees = item.accrual.total
            ledger.funding_fallbacks += item.accrual.fallback_settlements
            funding_events = [(ts, -amount) for ts, amount in item.accrual.payments]
        ledger.total_funding += funding_fees

        net_pnl = item.realized_pnl - item.slippage_cost - funding_fees
        nets.append(net_pnl)
        if rules.summary_order == "entry_time":
            entries.append((int(pos.ts_opened), net_pnl))
        ledger.closed_net.append((item.exit_ns, net_pnl))
        if rules.cash_order == "realized_first":
            # R7: screening books the realized leg, then the funding payments.
            ledger.cash_events.append((item.exit_ns, item.realized_pnl))
            ledger.cash_events.extend(funding_events)
        else:
            # R7: validation books the funding payments first, then the realized leg.
            ledger.cash_events.extend(funding_events)
            if per_leg_cash and item.entry_slippage:
                ledger.cash_events.append((int(pos.ts_opened), -item.entry_slippage))
                ledger.cash_events.append(
                    (item.exit_ns, item.realized_pnl - item.exit_slippage)
                )
            else:
                ledger.cash_events.append(
                    (item.exit_ns, item.realized_pnl - item.slippage_cost)
                )
        ledger.modeled_cash[instrument_id].extend(funding_events)
        if item.slippage_cost:
            ledger.modeled_cash[instrument_id].append((item.exit_ns, -item.slippage_cost))

        if rules.moment_returns and mark is None:
            notional = abs(float(pos.peak_qty) * float(pos.avg_px_open))
            if notional > 0:
                ledger.moment_returns.append(item.realized_pnl / notional)

    if rules.summary_order == "entry_time":
        ledger.net_pnls = [net for _, net in sorted(entries, key=lambda pair: pair[0])]
    else:
        ledger.net_pnls = nets
    return ledger
