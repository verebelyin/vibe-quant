"""The ONE run-sizing rule: exposure cap and risk per trade (bead vibe-quant-2gyk1.2).

Stdlib only, and deliberately outside :mod:`vibe_quant.validation` and
:mod:`vibe_quant.liquidation`: the API schemas import it on every launch request, so
reading a run's sizing settings must pull none of the engine stack (import isolation).

``max_position_pct`` is an exposure CAP, not a target: the compiled template sizes
``min(equity * risk_per_trade / stop_distance, equity * max_position_pct / price)`` and a
position is skipped below the instrument's min qty / min notional, so the actual exposure is
the smaller of the two. Margin leverage is a separate setting
(:mod:`vibe_quant.liquidation_settings`) and must be at least the cap
(:meth:`SizingSettings.check_leverage`).

Both fields are validated in EVERY mode and stored as floats: an invalid stored value must
fail the run rather than quietly become a different number downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_MAX_POSITION_PCT: float = 0.5
"""The one default exposure cap: at most 50% of equity per position."""

DEFAULT_RISK_PER_TRADE: float = 0.02
"""The one default risk budget: 2% of equity risked per trade."""

MAX_MAX_POSITION_PCT: float = 125.0
MAX_RISK_PER_TRADE: float = 0.5

SIZING_FIELDS: tuple[str, ...] = ("max_position_pct", "risk_per_trade")
"""The ``parameters`` keys a run's sizing settings live under (``parameters.<key>``)."""


class SizingConfigError(ValueError):
    """A run's sizing settings are invalid (they are never silently defaulted)."""


def _invalid_cap(value: object) -> SizingConfigError:
    return SizingConfigError(
        f"max_position_pct must be a number in (0, {MAX_MAX_POSITION_PCT:g}], got {value!r}"
    )


def _invalid_risk(value: object) -> SizingConfigError:
    return SizingConfigError(
        "risk_per_trade must be more than 0 and at most "
        f"{MAX_RISK_PER_TRADE:g} ({MAX_RISK_PER_TRADE:.0%} of equity per trade), got {value!r}"
    )


def _is_amount(value: object, upper: float) -> bool:
    """Whether ``value`` is a finite number in ``(0, upper]``.

    The type is checked BEFORE any conversion: comparing an int against the bounds is exact
    (``float(10**400)`` would raise OverflowError), and a bool — an int subclass — is not a
    number here. ``0 < value <= upper`` is also False for ``nan``, ``inf`` and ``-inf``, so
    one comparison covers the whole non-finite family.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0.0 < value <= upper


@dataclass(frozen=True)
class SizingSettings:
    """The validated ``max_position_pct`` + ``risk_per_trade`` of one run.

    A missing key means its default; an explicit ``None`` is rejected (a client that sends
    null learns about it instead of quietly getting the default). A boolean is not a number,
    and both fields are validated in every mode and stored as floats.
    """

    max_position_pct: float = DEFAULT_MAX_POSITION_PCT
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE

    def __post_init__(self) -> None:
        cap = self.max_position_pct
        if not _is_amount(cap, MAX_MAX_POSITION_PCT):
            raise _invalid_cap(cap)
        risk = self.risk_per_trade
        if not _is_amount(risk, MAX_RISK_PER_TRADE):
            raise _invalid_risk(risk)
        object.__setattr__(self, "max_position_pct", float(cap))
        object.__setattr__(self, "risk_per_trade", float(risk))

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> SizingSettings:
        """Read both sizing settings from a run's parameters.

        A missing key means its default; an explicit ``None`` is rejected. Every other key is
        ignored and the mapping is never modified. Both fields are validated, and
        ``max_position_pct`` is reported first when both are bad.
        """
        return cls(
            max_position_pct=cast(
                "float", params.get("max_position_pct", DEFAULT_MAX_POSITION_PCT)
            ),
            risk_per_trade=cast("float", params.get("risk_per_trade", DEFAULT_RISK_PER_TRADE)),
        )

    def to_params(self) -> dict[str, float]:
        """The parameters to store: only the keys a run moved off the default.

        ``SizingSettings().to_params() == {}``, so an explicit default stores byte-identically
        to a default launch.
        """
        params: dict[str, float] = {}
        if self.max_position_pct != DEFAULT_MAX_POSITION_PCT:
            params["max_position_pct"] = self.max_position_pct
        if self.risk_per_trade != DEFAULT_RISK_PER_TRADE:
            params["risk_per_trade"] = self.risk_per_trade
        return params

    def check_leverage(self, leverage: float) -> None:
        """Reject a cap the run's margin leverage cannot cover (S13).

        Checked in EVERY liquidation mode: the cap is exposure, the leverage only changes how
        much margin it needs, and a cap above the leverage would be unmarginable. Equality is
        allowed — a cap exactly at the leverage is fully marginable.
        """
        if self.max_position_pct > float(leverage):
            raise SizingConfigError(
                f"max_position_pct ({self.max_position_pct!r}) must not exceed "
                f"leverage ({float(leverage)!r})"
            )


def recorded_sizing(params: object) -> SizingSettings:
    """The one rule for reading sizing settings recorded with a run.

    A legacy row whose parameters are absent or not a mapping (``None``, a list, a string,
    a number) reads as the defaults. A mapping goes through the same rule as a launch
    request, so a corrupt recorded value is an error rather than a silent default.
    """
    if not isinstance(params, dict):
        return SizingSettings()
    return SizingSettings.from_params(params)
