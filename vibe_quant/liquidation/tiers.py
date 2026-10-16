"""Binance USDT-M maintenance-margin tiers and liquidation-price math.

Pure functions plus a versioned per-symbol tier-table snapshot; no engine
integration. Formula basis: Binance USDⓈ-M FAQ "How to calculate liquidation
price" (one-way mode):

    LP = (WB - TMM1 + UPNL1 + cum - side*q*EP) / (q*MMR - side*q)

side = +1 long / -1 short, q = position size, EP = entry price, WB = wallet
balance, TMM1 = maintenance margin of all OTHER positions, UPNL1 = unrealized
PnL of all OTHER positions, and (MMR, cum) come from the maintenance-margin
bracket whose [floor, cap) contains the notional at LP (q*LP).

A position's maintenance margin is notional * MMR(tier) - cum(tier). Bracket
continuity (Binance): cum_i == cum_{i-1} + floor_i * (MMR_i - MMR_{i-1}); the
loader enforces it and raises on violation. Brackets must also be contiguous
(floor_i == cap_{i-1} exactly); the tolerance applies to cum continuity only.

Solvency semantics. Let D(P) = WB - TMM1 + UPNL1 + side*q*(P - EP) - MM(q*P),
the margin balance minus maintenance margin at mark price P. MM is continuous
(cum continuity) and piecewise linear, so D is continuous and piecewise linear
with slope q*(side - MMR): strictly increasing for a long, strictly decreasing
for a short (0 <= MMR < 1). The position liquidates at the root D(LP) = 0.
  * long : D rises with price. LP <= 0 means D(0) >= 0, so the position can
    never be liquidated (returned as is, callers treat it as "never").
  * short: D falls with price, so LP <= 0 means D(P) < 0 at every positive
    price: insolvent throughout. ``InsolventPosition`` is raised; callers
    liquidate immediately at the current mark.
  * otherwise the LP is returned whatever its position relative to the entry
    price. This module cannot judge solvency at the current mark (funding and
    realised PnL move the balance away from entry): callers compare the LP with
    the CURRENT mark (long liquidated if mark <= LP, short if mark >= LP).
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

_RESOURCE = "binance_um_brackets.json"
_CUM_TOL_REL = 1e-6
# Relative slack (on the bracket-boundary notional) when rounding leaves a
# solved LP a few ulps outside the bracket it was solved with.
_BOUNDARY_TOL_REL = 1e-9
# Narrowest bracket a snapshot may declare (USDT). An absolute floor here, not
# an epsilon: a sub-1-USDT bracket makes the interior-slack class reachable in
# everyday notionals, and the floor keeps that class out of loaded snapshots.
MIN_BRACKET_WIDTH_USDT: float = 1.0


class TierTableError(ValueError):
    """A tier-table snapshot is malformed or internally inconsistent."""


class TierTableExceeded(TierTableError):
    """A notional lies at or above the largest bracket cap."""


class InsolventPosition(ValueError):
    """A short whose margin balance is below maintenance margin at every price.

    Its liquidation price is <= 0, so no positive price keeps it solvent.
    Callers liquidate immediately at the current mark.
    """

    def __init__(self, margin_balance: float, liquidation_price: float) -> None:
        super().__init__(
            f"insolvent short: margin balance {margin_balance!r} (wallet - TMM1 + UPNL1) "
            f"leaves liquidation price {liquidation_price!r} <= 0"
        )
        self.margin_balance = margin_balance
        self.liquidation_price = liquidation_price


class UnknownSymbol(KeyError):
    """No tier table exists for the requested symbol."""


@dataclass(frozen=True)
class Bracket:
    """One maintenance-margin bracket: [floor, cap) -> (mmr, cum, leverage)."""

    floor: float
    cap: float
    mmr: float
    cum: float
    max_leverage: int | None

    def __post_init__(self) -> None:
        for name in ("floor", "cap", "mmr", "cum"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise TierTableError(f"bracket {name} must be finite, got {value!r}")
        if self.floor < 0:
            raise TierTableError(f"bracket floor must be >= 0, got {self.floor}")
        if not 0 <= self.mmr < 1:
            raise TierTableError(f"bracket mmr must satisfy 0 <= mmr < 1, got {self.mmr}")
        if self.cum < 0:
            raise TierTableError(f"bracket cum must be >= 0, got {self.cum}")
        if self.max_leverage is not None and self.max_leverage < 1:
            raise TierTableError(f"bracket max_leverage must be >= 1, got {self.max_leverage}")

    def contains(self, notional: float) -> bool:
        return self.floor <= notional < self.cap


@dataclass(frozen=True)
class TierTable:
    symbol: str
    brackets: tuple[Bracket, ...]
    clearance_fee_rate: float | None

    def bracket_for(self, notional: float) -> Bracket:
        if not math.isfinite(notional) or notional < 0:
            raise ValueError(f"notional must be finite and non-negative, got {notional!r}")
        for bracket in self.brackets:
            if bracket.contains(notional):
                return bracket
        raise TierTableExceeded(
            f"{self.symbol}: notional {notional} >= last cap {self.brackets[-1].cap}"
        )

    def maintenance_margin(self, notional: float) -> float:
        bracket = self.bracket_for(notional)
        return notional * bracket.mmr - bracket.cum


class TierTables(dict[str, TierTable]):
    """Mapping of symbol -> TierTable carrying the snapshot metadata."""

    def __init__(self, *, version: str, fetched_at: str, source: str, note: str) -> None:
        super().__init__()
        self.version = version
        self.fetched_at = fetched_at
        self.source = source
        self.note = note

    def __getitem__(self, symbol: str) -> TierTable:
        try:
            return super().__getitem__(symbol)
        except KeyError:
            raise UnknownSymbol(symbol) from None


def load_tier_tables(path: Path | None = None) -> TierTables:
    """Load and validate a tier-table snapshot.

    ``path=None`` reads the shipped snapshot from this package. Raises
    ``TierTableError`` on malformed JSON, missing fields, unsorted/overlapping
    brackets or a cum discontinuity.
    """
    if path is None:
        text = files("vibe_quant.liquidation").joinpath(_RESOURCE).read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TierTableError(f"invalid tier-table JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise TierTableError("tier-table snapshot must be a JSON object")

    tables = TierTables(
        version=_require_str(raw, "version"),
        fetched_at=_require_str(raw, "fetched_at"),
        source=_require_str(raw, "source"),
        note=_require_str(raw, "note"),
    )
    symbols = raw.get("symbols")
    if not isinstance(symbols, dict):
        raise TierTableError("snapshot is missing the 'symbols' object")
    for symbol, entry in symbols.items():
        tables[symbol] = _parse_symbol(symbol, entry)
    if not tables:
        raise TierTableError("snapshot contains no symbols")
    return tables


def isolated_liq_price(
    side: int,
    qty: float,
    entry_price: float,
    isolated_margin: float,
    table: TierTable,
) -> float:
    """Liquidation price for an isolated-margin position.

    WB = the position's isolated margin; TMM1 = UPNL1 = 0. A long return value
    <= 0 means the position never liquidates (price cannot cross it); callers
    treat it as "never". Callers compare the LP with the current mark. Raises
    ``InsolventPosition`` for a short with LP <= 0 (see module docstring) and
    ``ValueError`` for invalid inputs.
    """
    _check_common(side, qty, entry_price)
    margin = _check_number("isolated_margin", isolated_margin, non_negative=True)
    return _resolve_liq_price(side, qty, entry_price, margin, 0.0, 0.0, table)


def cross_liq_price(
    side: int,
    qty: float,
    entry_price: float,
    wallet_balance: float,
    other_maint_margin: float,
    other_upnl: float,
    table: TierTable,
) -> float:
    """Liquidation price for a one-way cross-margin position.

    ``other_maint_margin``/``other_upnl`` describe all OTHER positions in the
    account (TMM1/UPNL1). ``wallet_balance`` may be negative: a cross wallet
    goes below 0 after funding or realised losses while the margin balance
    (wallet + uPnL) still exceeds MM, and the LP uses the signed wallet. A long
    return value <= 0 means "never liquidates". Callers compare the LP with the
    current mark. Raises ``InsolventPosition`` for a short with LP <= 0 (see
    module docstring) and ``ValueError`` for invalid inputs.
    """
    _check_common(side, qty, entry_price)
    wallet = _check_number("wallet_balance", wallet_balance, non_negative=False)
    other_mm = _check_number("other_maint_margin", other_maint_margin, non_negative=True)
    other_pnl = _check_number("other_upnl", other_upnl, non_negative=False)
    return _resolve_liq_price(side, qty, entry_price, wallet, other_mm, other_pnl, table)


def _resolve_liq_price(
    side: int,
    qty: float,
    entry_price: float,
    wallet_balance: float,
    other_maint_margin: float,
    other_upnl: float,
    table: TierTable,
) -> float:
    """Solve LP per bracket and keep the bracket consistent with its own LP.

    For each bracket we price the LP with that bracket's (MMR, cum) and accept
    it only if the notional at that LP (qty*LP) falls back inside the same
    half-open bracket. Exactly one bracket is normally consistent. When
    rounding leaves the LP notional a few ulps outside the bracket on a shared
    INTERIOR boundary, the candidates within ``_BOUNDARY_TOL_REL`` are used
    instead (cum continuity makes both neighbours price the same LP); the last
    bracket's cap gets no slack, so a notional at or beyond it raises
    ``TierTableExceeded``. A long whose first-bracket LP is <= 0 never
    liquidates (that LP is returned); the same for a short is insolvent.
    """
    candidates = [
        _liq_price_for_bracket(
            side, qty, entry_price, wallet_balance, other_maint_margin, other_upnl, bracket
        )
        for bracket in table.brackets
    ]
    if candidates[0] <= 0:
        if side == -1:
            raise InsolventPosition(wallet_balance - other_maint_margin + other_upnl, candidates[0])
        return candidates[0]
    strict = [lp for lp, b in zip(candidates, table.brackets, strict=True) if b.contains(qty * lp)]
    if len(strict) == 1:
        return strict[0]
    if len(strict) > 1:
        raise TierTableError(
            f"{table.symbol}: LP notional is consistent with multiple brackets: {strict}"
        )
    last = table.brackets[-1]
    if qty * candidates[-1] >= last.cap:
        raise TierTableExceeded(
            f"{table.symbol}: liquidation notional {qty * candidates[-1]} >= last cap {last.cap}"
        )
    last_index = len(table.brackets) - 1
    near = [
        lp
        for index, (lp, b) in enumerate(zip(candidates, table.brackets, strict=True))
        if _near_bracket(qty * lp, b, slack_cap=index != last_index) and qty * lp < last.cap
    ]
    if near:
        return near[0]
    raise TierTableError(
        f"{table.symbol}: no bracket is consistent with its own liquidation price "
        f"(side={side}, qty={qty}, entry_price={entry_price}, wallet_balance={wallet_balance}, "
        f"other_maint_margin={other_maint_margin}, other_upnl={other_upnl}); "
        f"first-bracket LP={candidates[0]}"
    )


def _near_bracket(notional: float, bracket: Bracket, *, slack_cap: bool) -> bool:
    """True if ``notional`` is within float-rounding distance of [floor, cap).

    ``slack_cap=False`` (the last bracket) keeps the cap exact.
    """
    floor_tol = _BOUNDARY_TOL_REL * max(1.0, bracket.floor)
    cap = bracket.cap + (_BOUNDARY_TOL_REL * max(1.0, bracket.cap) if slack_cap else 0.0)
    return bracket.floor - floor_tol <= notional < cap


def _liq_price_for_bracket(
    side: int,
    qty: float,
    entry_price: float,
    wallet_balance: float,
    other_maint_margin: float,
    other_upnl: float,
    bracket: Bracket,
) -> float:
    numerator = (
        wallet_balance - other_maint_margin + other_upnl + bracket.cum - side * qty * entry_price
    )
    denominator = qty * bracket.mmr - side * qty
    if denominator == 0:
        raise TierTableError(
            f"liquidation-price denominator is zero for bracket "
            f"[{bracket.floor}, {bracket.cap}) mmr={bracket.mmr}"
        )
    return numerator / denominator


def _check_common(side: int, qty: float, entry_price: float) -> None:
    if not isinstance(side, numbers.Integral) or isinstance(side, bool) or side not in (1, -1):
        raise ValueError(f"side must be +1 or -1, got {side!r}")
    _check_number("qty", qty, positive=True)
    _check_number("entry_price", entry_price, positive=True)


def _check_number(
    name: str, value: object, *, non_negative: bool = False, positive: bool = False
) -> float:
    """Validate a real, finite, non-bool number; raise ``ValueError`` otherwise."""
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and number <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    if non_negative and number < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return number


def _parse_symbol(symbol: str, entry: object) -> TierTable:
    if not isinstance(entry, dict):
        raise TierTableError(f"{symbol}: tier entry must be a JSON object")
    raw_brackets = entry.get("brackets")
    if not isinstance(raw_brackets, list) or not raw_brackets:
        raise TierTableError(f"{symbol}: 'brackets' must be a non-empty list")
    brackets = tuple(_parse_bracket(symbol, i, raw) for i, raw in enumerate(raw_brackets))
    _validate_brackets(symbol, brackets)
    fee = entry.get("clearance_fee_rate")
    if fee is None:
        clearance_fee_rate = None
    elif isinstance(fee, (int, float)) and not isinstance(fee, bool):
        clearance_fee_rate = float(fee)
        if not math.isfinite(clearance_fee_rate) or not 0 <= clearance_fee_rate <= 1:
            raise TierTableError(f"{symbol}: 'clearance_fee_rate' must be within [0, 1]")
    else:
        raise TierTableError(f"{symbol}: 'clearance_fee_rate' must be a number or null")
    return TierTable(symbol=symbol, brackets=brackets, clearance_fee_rate=clearance_fee_rate)


def _parse_bracket(symbol: str, index: int, raw: object) -> Bracket:
    if not isinstance(raw, dict):
        raise TierTableError(f"{symbol}: bracket {index} must be a JSON object")
    leverage = raw.get("max_leverage")
    if leverage is None:
        max_leverage = None
    elif isinstance(leverage, int) and not isinstance(leverage, bool):
        max_leverage = leverage
    else:
        raise TierTableError(f"{symbol}: bracket {index} 'max_leverage' must be an int or null")
    try:
        return Bracket(
            floor=_require_number(raw, "floor", symbol, index),
            cap=_require_number(raw, "cap", symbol, index),
            mmr=_require_number(raw, "mmr", symbol, index),
            cum=_require_number(raw, "cum", symbol, index),
            max_leverage=max_leverage,
        )
    except TierTableError as exc:
        if str(exc).startswith(f"{symbol}:"):
            raise
        raise TierTableError(f"{symbol}: bracket {index}: {exc}") from exc


def _validate_brackets(symbol: str, brackets: tuple[Bracket, ...]) -> None:
    first = brackets[0]
    if first.floor != 0:
        raise TierTableError(f"{symbol}: first bracket floor must be 0, got {first.floor}")
    for index, bracket in enumerate(brackets):
        if not bracket.cap > bracket.floor:
            raise TierTableError(
                f"{symbol}: bracket {index} cap {bracket.cap} <= floor {bracket.floor}"
            )
        if bracket.cap - bracket.floor < MIN_BRACKET_WIDTH_USDT:
            raise TierTableError(
                f"{symbol}: bracket {index} width {bracket.cap - bracket.floor} < "
                f"{MIN_BRACKET_WIDTH_USDT} USDT (degenerate thin tier)"
            )
    for previous, current in zip(brackets, brackets[1:], strict=False):
        if current.floor != previous.cap:
            raise TierTableError(
                f"{symbol}: brackets must be contiguous (overlap or gap): floor "
                f"{current.floor} != previous cap {previous.cap}"
            )
        expected = previous.cum + current.floor * (current.mmr - previous.mmr)
        cum_tolerance = _CUM_TOL_REL * max(1.0, abs(current.cum))
        if abs(current.cum - expected) > cum_tolerance:
            raise TierTableError(
                f"{symbol}: cum discontinuity at floor {current.floor}: cum {current.cum} "
                f"!= expected {expected} (tolerance {cum_tolerance})"
            )


def _require_number(raw: dict[object, object], key: str, symbol: str, index: int) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TierTableError(f"{symbol}: bracket {index} missing numeric '{key}'")
    return float(value)


def _require_str(raw: dict[object, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise TierTableError(f"snapshot missing non-empty string '{key}'")
    return value
