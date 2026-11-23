"""Run-level liquidation settings shared by screening, validation and discovery.

The one parse rule for ``liquidation_mode`` / ``leverage`` (bead vibe-quant-2gyk1.1.8). It
lives OUTSIDE :mod:`vibe_quant.liquidation` on purpose: the screening engine of a mode-off run must never
import the liquidation package (import isolation), and these settings are pure stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

LiquidationMode = Literal["off", "isolated", "cross"]
LIQUIDATION_MODES: tuple[LiquidationMode, ...] = ("off", "isolated", "cross")
DEFAULT_LEVERAGE = 10.0
MIN_LEVERAGE = 1.0
MAX_LEVERAGE = 125.0

MISSING: object = object()
"""Sentinel for a leverage argument a caller did not supply (:func:`resolve_leverage`)."""


class LiquidationConfigError(ValueError):
    """A run's liquidation settings are invalid (they are never silently defaulted)."""


def resolve_leverage(primary: object = MISSING, alias: object = MISSING) -> object:
    """The one rule for a ``leverage`` given under two names (``leverage`` / ``liquidation_leverage``).

    Returns the first of ``(primary, alias)`` that is not :data:`MISSING`; the :data:`MISSING`
    sentinel (not ``None``) marks an argument a caller did not supply, so an explicit ``None``
    stays distinguishable and is still rejected by :class:`LiquidationSettings`. When BOTH are
    supplied they must be equal (``3 == 3.0``), otherwise a
    :class:`LiquidationConfigError` names both.

    Shared by the discovery CLI, the API schema and the notes inheritance so they cannot drift.
    """
    if primary is MISSING:
        return alias
    if alias is MISSING:
        return primary
    if primary != alias:  # int/float compare exactly (3 == 3.0), with no float() overflow
        msg = f"conflicting leverage {primary!r} and liquidation_leverage {alias!r}"
        raise LiquidationConfigError(msg)
    return primary


@dataclass(frozen=True)
class LiquidationSettings:
    """The validated ``liquidation_mode`` + ``leverage`` of one run.

    ``mode`` is case-sensitive and ``None`` is NOT "off" (P4). ``leverage`` is validated in
    EVERY mode (L4) as a whole number in ``[1, 125]`` and stored as a float (L2); a missing key
    means :data:`DEFAULT_LEVERAGE`, an explicit ``None`` is rejected.
    """

    mode: LiquidationMode = "off"
    leverage: float = DEFAULT_LEVERAGE

    def __post_init__(self) -> None:
        if self.mode not in LIQUIDATION_MODES:
            msg = f"liquidation_mode must be one of {LIQUIDATION_MODES}, got {self.mode!r}"
            raise LiquidationConfigError(msg)
        lev = self.leverage
        # Type and range checks come BEFORE any float()/int() conversion: float(10**400)
        # raises OverflowError, and an int compares against the float bounds exactly.
        if isinstance(lev, bool) or not isinstance(lev, (int, float)):
            valid = False
        elif isinstance(lev, float):
            valid = math.isfinite(lev) and lev.is_integer() and MIN_LEVERAGE <= lev <= MAX_LEVERAGE
        else:
            valid = MIN_LEVERAGE <= lev <= MAX_LEVERAGE
        if not valid:
            msg = f"leverage must be a whole number in [1, 125], got {lev!r}"
            raise LiquidationConfigError(msg)
        object.__setattr__(self, "leverage", float(lev))

    @property
    def on(self) -> bool:
        """Whether liquidation runs at all."""
        return self.mode != "off"

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> LiquidationSettings:
        """Read ``liquidation_mode`` / ``leverage`` from a run's parameters.

        A missing mode key means off (P1/P2: old rows and clients); a missing leverage key
        means :data:`DEFAULT_LEVERAGE`. An explicit ``None`` is rejected (never "off").
        """
        return cls(
            mode=cast("LiquidationMode", params.get("liquidation_mode", "off")),
            leverage=cast("float", params.get("leverage", DEFAULT_LEVERAGE)),
        )

    def to_params(self) -> dict[str, object]:
        """The parameters to store.

        On -> ``{mode, leverage}``; off at a non-default leverage -> ``{"leverage": L}``; off
        at the default -> ``{}``, so existing default rows are unchanged.
        """
        if self.mode != "off":
            return {"liquidation_mode": self.mode, "leverage": self.leverage}
        if self.leverage != DEFAULT_LEVERAGE:
            return {"leverage": self.leverage}
        return {}

    def to_notes(self) -> dict[str, object] | None:
        """The ``notes["liquidation"]`` payload: None when off."""
        if self.mode == "off":
            return None
        return {"mode": self.mode, "leverage": self.leverage}
