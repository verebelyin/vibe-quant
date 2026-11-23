"""The stop cap: the widest stop the GA draws with liquidation off (arch-c29-1, bead clmmc.29).

With liquidation off, a stop drawn past the isolated liquidation distance is booked as a stop
loss although the exchange would have liquidated the position closer to entry. The cap keeps
every drawn stop inside ``STOP_CAP_MARGIN`` of the worst (symbol, side) liquidation distance at
the run leverage. One module owns the rule; the CLI run, the API precheck and the notes read it.

The distance comes from :func:`vibe_quant.liquidation.tiers.isolated_liq_price` (the one
implementation of the LP formula), imported lazily: a mode-off run must not load the liquidation
package until it is capped (``test_stop_cap.py::test_import_isolation``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from vibe_quant.discovery.operators import SL_RANGE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibe_quant.liquidation_settings import LiquidationSettings

STOP_CAP_MARGIN = 0.95
"""A stop may be at most this share of the liquidation distance."""
STOP_CAP_REF_NOTIONAL_USDT = 1000.0
"""Discovery always starts at 1 000 USDT; inside the first bracket the distance ignores size."""
_REF_ENTRY = 100.0


class StopCapLaunchError(ValueError):
    """A launch whose stop range cannot be bounded honestly (CLI exit 2, API 422)."""


@dataclass(frozen=True, slots=True)
class StopCap:
    """The run's stop ceiling and the (symbol, side) liquidation distance that binds it."""

    max_pct: float
    leverage: float
    symbol: str
    side: Literal["long", "short"]
    liquidation_distance_pct: float
    tier_snapshot: str

    def to_note(self) -> dict[str, object]:
        """The ``notes["stop_loss_cap"]`` payload."""
        return {
            "max_pct": self.max_pct,
            "leverage": self.leverage,
            "symbol": self.symbol,
            "side": self.side,
            "liquidation_distance_pct": self.liquidation_distance_pct,
            "margin": STOP_CAP_MARGIN,
            "reference_notional_usdt": STOP_CAP_REF_NOTIONAL_USDT,
            "tier_snapshot": self.tier_snapshot,
        }


def stop_cap(symbols: Sequence[str], leverage: float) -> StopCap | None:
    """The cap for a liquidation-off run on ``symbols`` at ``leverage``; None = range unchanged.

    ``min(10.0, floor(0.95 x worst isolated liquidation distance) to 0.01)``: worst over the
    symbols (one genome trades them all) and over long/short (direction can flip under
    crossover). A tie goes to the first symbol in sorted order, the short side.

    Raises:
        ValueError: No symbols.
        StopCapLaunchError: A symbol has no tier table, or the cap is at or below the smallest
            stop the GA draws.
    """
    if not symbols:
        msg = "stop cap needs the run's symbols"
        raise ValueError(msg)
    from vibe_quant.liquidation.tiers import UnknownSymbol, isolated_liq_price, load_tier_tables

    tables = load_tier_tables()
    notional = STOP_CAP_REF_NOTIONAL_USDT
    worst: tuple[float, str, Literal["long", "short"]] | None = None
    for symbol in sorted(set(symbols)):
        try:
            table = tables[symbol]
        except UnknownSymbol:
            msg = (
                f"no Binance tier table for {symbol} (snapshot {tables.version}): cannot bound "
                "the stop range with liquidation off"
            )
            raise StopCapLaunchError(msg) from None
        sides: tuple[tuple[Literal["long", "short"], int], ...] = (("short", -1), ("long", 1))
        for name, side in sides:  # short first: it wins a tie
            lp = isolated_liq_price(
                side, notional / _REF_ENTRY, _REF_ENTRY, notional / leverage, table
            )
            distance = (_REF_ENTRY - lp if side == 1 else lp - _REF_ENTRY) / _REF_ENTRY * 100
            if worst is None or distance < worst[0]:
                worst = (distance, symbol, name)
    assert worst is not None  # symbols is non-empty
    distance, symbol, side_name = worst
    cap = min(SL_RANGE[1], math.floor(STOP_CAP_MARGIN * distance * 100) / 100)
    if cap <= SL_RANGE[0]:
        msg = (
            f"--leverage {leverage:g} with liquidation off liquidates a {symbol} {side_name} "
            f"{distance:.2f}% from entry; a {STOP_CAP_MARGIN:.0%} stop cap ({cap:.2f}%) leaves "
            f"no room above the smallest stop discovery draws ({SL_RANGE[0]:g}%). Turn "
            "liquidation on (--liquidation-mode isolated or cross) or lower --leverage."
        )
        raise StopCapLaunchError(msg)
    if cap >= SL_RANGE[1]:
        return None
    return StopCap(
        max_pct=cap,
        leverage=float(leverage),
        symbol=symbol,
        side=side_name,
        liquidation_distance_pct=distance,
        tier_snapshot=tables.version,
    )


def run_stop_cap(
    symbols: Sequence[str], liquidation: LiquidationSettings, *, grid: bool, port: bool
) -> StopCap | None:
    """The cap a discovery launch applies: only a plain liquidation-off chromosome run has one.

    Liquidation on books the liquidation itself; a bankroll grid always runs a liquidation mode;
    a ``--port`` run evolves the port's own params.
    """
    if liquidation.mode != "off" or grid or port:
        return None
    return stop_cap(symbols, liquidation.leverage)
