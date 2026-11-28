"""Binance USDT-M perpetual fee schedule (bd vibe-quant-2gyk1.4, D3-1).

A single immutable schedule is threaded through the backtest stack. The default
(VIP0, no BNB discount) keeps NT's ``MakerTakerFeeModel`` path, so every
pre-existing number stays bit-identical; the discounts are opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

#: (maker, taker) rates per VIP tier, as fractions of notional.
BINANCE_USDTM_TIERS: dict[str, tuple[Decimal, Decimal]] = {
    "VIP0": (Decimal("0.0002"), Decimal("0.0005")),
}

#: Binance's BNB discount: paying the fee in BNB costs 90% of the tier rate.
BNB_DISCOUNT_FACTOR = Decimal("0.9")

DEFAULT_TIER = "VIP0"

#: The tier names the launch APIs accept; mirrors the keys of :data:`BINANCE_USDTM_TIERS`.
FeeTier = Literal["VIP0"]


@dataclass(frozen=True)
class FeeSchedule:
    """A Binance USDT-M fee schedule: a VIP tier plus an optional BNB discount.

    Attributes:
        tier: VIP tier name; must be a key of :data:`BINANCE_USDTM_TIERS`.
        bnb_discount: Charge the BNB-discounted rate (tier rate x 0.9).
    """

    tier: str = DEFAULT_TIER
    bnb_discount: bool = False

    def __post_init__(self) -> None:
        # A list/dict tier would raise TypeError (unhashable) at the lookup; runners and
        # the API treat a ValueError as a bad launch.
        if not isinstance(self.tier, str) or self.tier not in BINANCE_USDTM_TIERS:
            raise ValueError(
                f"unknown fee tier {self.tier!r}; known tiers: {sorted(BINANCE_USDTM_TIERS)}"
            )
        if not isinstance(self.bnb_discount, bool):
            raise ValueError(f"bnb_discount must be a bool, got {self.bnb_discount!r}")

    @property
    def maker_rate(self) -> Decimal:
        """Maker fee as a fraction of notional."""
        rate = BINANCE_USDTM_TIERS[self.tier][0]
        return rate * BNB_DISCOUNT_FACTOR if self.bnb_discount else rate

    @property
    def taker_rate(self) -> Decimal:
        """Taker fee as a fraction of notional."""
        rate = BINANCE_USDTM_TIERS[self.tier][1]
        return rate * BNB_DISCOUNT_FACTOR if self.bnb_discount else rate

    @property
    def is_default(self) -> bool:
        """True for VIP0 without the BNB discount (the NT MakerTakerFeeModel path)."""
        return self.tier == DEFAULT_TIER and not self.bnb_discount

    def to_notes(self) -> dict[str, Any]:
        """Notes payload; round-trips through :meth:`from_params`."""
        return {"fee_tier": self.tier, "bnb_discount": self.bnb_discount}

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> FeeSchedule:
        """Build a schedule from run parameters (``fee_tier``, ``bnb_discount``)."""
        return cls(
            tier=params.get("fee_tier", DEFAULT_TIER),
            bnb_discount=params.get("bnb_discount", False),
        )
