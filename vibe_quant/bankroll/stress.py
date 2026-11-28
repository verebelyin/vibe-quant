"""The bankroll size grid: one cell per requested multiple ``k`` of the parent's sizing.

Rule (user decision Q1(d), 2026-11-19): ``effective_cap = parent_cap * k``,
``requested_risk = parent_risk * k``, ``effective_risk = min(requested_risk, 0.50)``;
``risk_held_at_limit`` is true exactly when ``requested_risk > 0.50``. The 50% is the
configured sizing-budget limit at the signal price, before minimum quantity, fills and
costs. It is not a bound on the realized result of a trade, so no row carries one: every
row reports its own measured trades, fees, funding, MTM drawdown and liquidation facts,
and ``k`` is never a multiplier of actual exposure or drawdown.

Each cell is admitted on its own through the existing parsers (sizing, liquidation,
cap-vs-leverage); a refusal makes that cell ``not_runnable`` with the parser's text and the
other cells still run. Imports no engine, validation or DB module (import isolation).

The margin sweep (:func:`plan_margin_cells`) varies margin mode x leverage x margin limit at
one k-row's sizing; each of its cells is admitted the same way, plus the coin maximum
leverage of every traded symbol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from vibe_quant.liquidation_settings import (
    LiquidationConfigError,
    LiquidationMode,
    LiquidationSettings,
)
from vibe_quant.sizing_settings import MAX_RISK_PER_TRADE, SizingConfigError, SizingSettings
from vibe_quant.symbol_leverage import SymbolLeverageError, check_leverage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vibe_quant.bankroll_grid import BankrollCell

RISK_LIMIT: float = MAX_RISK_PER_TRADE
"""The sizing-budget limit a cell's risk is held at (the shared 50% admission limit)."""

CellStatus = Literal["planned", "not_runnable"]
RowStatus = Literal["ok", "failed", "not_runnable"]


@dataclass(frozen=True)
class CellPlan:
    """One planned grid cell. ``overrides`` (empty when not runnable) merge into a copy of
    the parent's parameters for the variant run."""

    k: float
    status: CellStatus
    requested_risk: float
    effective_risk: float
    effective_cap: float
    leverage: float
    mode: LiquidationMode
    risk_held_at_limit: bool
    overrides: dict[str, object] = field(default_factory=dict)
    error: str | None = None


def plan_cells(
    parent_params: Mapping[str, object], multiples: Sequence[float], mode: LiquidationMode
) -> list[CellPlan]:
    """Plan one cell per multiple, in the given order, at the parent's leverage.

    The parent's own settings must parse (a completed parent always does); a cell's
    refusal is recorded on that cell only. ``parent_params`` is never modified.
    """
    parent = SizingSettings.from_params(parent_params)
    leverage = LiquidationSettings.from_params(parent_params).leverage
    plans: list[CellPlan] = []
    for k in multiples:
        cap = parent.max_position_pct * k
        requested = parent.risk_per_trade * k
        effective = min(requested, RISK_LIMIT)
        overrides: dict[str, object] = {
            "max_position_pct": cap,
            "risk_per_trade": effective,
            "liquidation_mode": mode,
        }
        merged = {**parent_params, **overrides}
        error: str | None = None
        try:
            sizing = SizingSettings.from_params(merged)
            LiquidationSettings.from_params(merged)
            sizing.check_leverage(leverage)
        except (SizingConfigError, LiquidationConfigError) as exc:
            error = str(exc)
        plans.append(
            CellPlan(
                k=k,
                status="planned" if error is None else "not_runnable",
                requested_risk=requested,
                effective_risk=effective,
                effective_cap=cap,
                leverage=leverage,
                mode=mode,
                risk_held_at_limit=requested > RISK_LIMIT,
                overrides=overrides if error is None else {},
                error=error,
            )
        )
    return plans


@dataclass(frozen=True)
class MarginCellPlan:
    """One margin-sweep cell at the k-row ``size``. ``overrides`` (empty when not runnable)
    merge into a copy of the parent's parameters; ``reused`` marks the cell whose merged
    parameters equal the k-row's, which takes the k-row's run instead of a new one."""

    mode: LiquidationMode
    leverage: float
    max_margin_pct: float
    size: CellPlan
    status: CellStatus
    overrides: dict[str, object] = field(default_factory=dict)
    error: str | None = None
    reused: bool = False


def plan_margin_cells(
    parent_params: Mapping[str, object],
    size_plan: CellPlan,
    cells: Sequence[BankrollCell],
    symbols: Sequence[str],
) -> list[MarginCellPlan]:
    """Plan one cell per grid cell, in the given order, at ``size_plan``'s cap and risk.

    A cell is refused (``not_runnable``, with the parser's text) when its settings do not
    parse, its leverage cannot margin the cap, or it exceeds a symbol's Binance maximum;
    the other cells still plan. ``parent_params`` is never modified.

    Raises ``ValueError`` when ``size_plan`` is not ``planned``: it has no overrides, so
    its cells would run at the parent's sizing under the k-row's label.
    """
    if size_plan.status != "planned":
        raise ValueError(
            f"size row k={size_plan.k} is {size_plan.status}; plan margin cells "
            "only on a planned k-row"
        )
    k_row = {**parent_params, **size_plan.overrides}
    plans: list[MarginCellPlan] = []
    for cell in cells:
        overrides: dict[str, object] = {
            **size_plan.overrides,
            "liquidation_mode": cell.mode,
            "leverage": cell.leverage,
            "max_margin_pct": cell.max_margin_pct,
        }
        merged = {**parent_params, **overrides}
        error: str | None = None
        try:
            sizing = SizingSettings.from_params(merged)
            LiquidationSettings.from_params(merged)
            sizing.check_leverage(cell.leverage)
            check_leverage(symbols, cell.leverage)
        except (SizingConfigError, LiquidationConfigError, SymbolLeverageError) as exc:
            error = str(exc)
        plans.append(
            MarginCellPlan(
                mode=cell.mode,
                leverage=cell.leverage,
                max_margin_pct=cell.max_margin_pct,
                size=size_plan,
                status="planned" if error is None else "not_runnable",
                overrides=overrides if error is None else {},
                error=error,
                reused=error is None and merged == k_row,
            )
        )
    return plans


@dataclass(frozen=True)
class RowOutcome:
    """What one engine run (baseline or cell) produced, as eligibility needs it.

    ``liquidation`` is the run's ``notes["liquidation"]`` (``None``: not modeled).
    ``bust_ts``/``path_divergence`` are the MTM probe's flags; the liquidation payload
    carries its own, and both are honoured.
    """

    k: float
    status: RowStatus
    trades: int = 0
    total_return: float | None = None
    mtm_paths: str | None = None
    mtm_dd: float | None = None
    bust_ts: int | None = None
    path_divergence: bool = False
    engine_force_stopped: bool = False
    liquidation: Mapping[str, object] | None = None
    error: str | None = None


@dataclass(frozen=True)
class RiskFacts:
    """Liquidation facts of one row; all ``None`` when liquidation was not modeled."""

    liquidation_count: int | None
    time_to_first_liquidation_s: float | None
    time_to_ruin_s: float | None


def liquidation_count(liquidation: Mapping[str, object]) -> int | None:
    """The payload's hit count, or ``None`` when it is missing or not a whole number."""
    count = liquidation.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return count


def _elapsed_s(ts: object, start_ns: int) -> float | None:
    if isinstance(ts, bool) or not isinstance(ts, int):
        return None
    return (ts - start_ns) / 1e9  # integer ns until this one conversion


def row_risk_facts(row: RowOutcome, start_ns: int) -> RiskFacts:
    """Count, time to first liquidation and time to ruin from the clamped start.

    Time to ruin is the elapsed time to a CROSS account's ``bust_ts``; an isolated hit has
    a count and a first-liquidation time but is not account ruin. Mode off means
    unevaluated: every fact is ``None``.
    """
    liq = row.liquidation
    if liq is None:
        return RiskFacts(None, None, None)
    first: float | None = None
    events = liq.get("events")
    if isinstance(events, list):
        times = [e.get("ts") for e in events if isinstance(e, dict)]
        valid = [t for t in times if isinstance(t, int) and not isinstance(t, bool)]
        if valid:
            first = _elapsed_s(min(valid), start_ns)
    ruin = _elapsed_s(liq.get("bust_ts"), start_ns) if liq.get("mode") == "cross" else None
    return RiskFacts(liquidation_count(liq), first, ruin)
