"""Ethereal DEX venue configuration for backtesting and paper trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from vibe_quant.validation.latency import LatencyValues

if TYPE_CHECKING:
    from nautilus_trader.config import LatencyModelConfig


class EtherealLatencyPreset(StrEnum):
    """Latency preset for Ethereal DEX execution simulation.

    Blockchain settlement adds extra latency vs centralized exchanges.
    """

    TESTNET = "testnet"
    MAINNET = "mainnet"


# Ethereal fee structure (0% maker, 0.03% taker)
ETHEREAL_MAKER_FEE = Decimal("0.0000")  # 0%
ETHEREAL_TAKER_FEE = Decimal("0.0003")  # 0.03%

# Funding interval: 1 hour (3600 seconds)
ETHEREAL_FUNDING_INTERVAL = 3600

# Latency presets for Ethereal DEX
# Blockchain-based settlement has higher latency than centralized exchanges
ETHEREAL_LATENCY_PRESETS: dict[EtherealLatencyPreset, LatencyValues] = {
    # Testnet: moderate latency, faster block times for testing
    EtherealLatencyPreset.TESTNET: LatencyValues(
        base_ms=500.0,
        insert_ms=250.0,
        update_ms=250.0,
        cancel_ms=250.0,
    ),
    # Mainnet: ~2 second block time for Ethereum settlement
    EtherealLatencyPreset.MAINNET: LatencyValues(
        base_ms=2000.0,
        insert_ms=1000.0,
        update_ms=1000.0,
        cancel_ms=1000.0,
    ),
}


@dataclass
class EtherealVenueConfig:
    """Configuration for Ethereal DEX venue.

    Attributes:
        name: Venue identifier, always "ETHEREAL".
        starting_balance_usde: Starting account balance in USDe.
        default_leverage: Default leverage for all instruments.
        leverages: Per-instrument max leverage overrides.
        funding_interval: Funding interval in seconds (default 1 hour).
        maker_fee: Maker fee rate as decimal (0%).
        taker_fee: Taker fee rate as decimal (0.03%).
        latency_preset: Latency preset for execution simulation.
    """

    name: str = "ETHEREAL"
    starting_balance_usde: int = 100_000
    default_leverage: Decimal = Decimal("10")
    leverages: dict[str, Decimal] = field(default_factory=dict)
    funding_interval: int = ETHEREAL_FUNDING_INTERVAL
    maker_fee: Decimal = ETHEREAL_MAKER_FEE
    taker_fee: Decimal = ETHEREAL_TAKER_FEE
    latency_preset: EtherealLatencyPreset | None = None


def get_ethereal_latency_model(
    preset: EtherealLatencyPreset | str,
) -> LatencyModelConfig:
    """Get LatencyModelConfig for an Ethereal preset.

    Args:
        preset: Latency preset name or enum value.

    Returns:
        Configured LatencyModelConfig.

    Raises:
        ValueError: If preset is not recognized.
    """
    if isinstance(preset, str):
        try:
            preset = EtherealLatencyPreset(preset)
        except ValueError:
            valid = [p.value for p in EtherealLatencyPreset]
            msg = f"Unknown Ethereal latency preset: {preset}. Valid: {valid}"
            raise ValueError(msg) from None

    values = ETHEREAL_LATENCY_PRESETS[preset]
    return values.to_config()


def get_ethereal_venue_for_backtesting(
    starting_balance_usde: int = 100_000,
    default_leverage: Decimal = Decimal("10"),
    leverages: dict[str, Decimal] | None = None,
) -> EtherealVenueConfig:
    """Get Ethereal venue config for backtesting.

    Uses mainnet latency preset for realistic simulation.

    Args:
        starting_balance_usde: Starting balance.
        default_leverage: Default leverage.
        leverages: Per-instrument max leverage overrides.

    Returns:
        EtherealVenueConfig for backtesting.
    """
    return EtherealVenueConfig(
        name="ETHEREAL",
        starting_balance_usde=starting_balance_usde,
        default_leverage=default_leverage,
        leverages=leverages or {},
        latency_preset=EtherealLatencyPreset.MAINNET,
    )


def get_ethereal_venue_for_paper_trading(
    starting_balance_usde: int = 100_000,
    default_leverage: Decimal = Decimal("10"),
    leverages: dict[str, Decimal] | None = None,
) -> EtherealVenueConfig:
    """Get Ethereal venue config for paper trading (testnet).

    Uses testnet latency preset for faster iteration.

    Args:
        starting_balance_usde: Starting balance.
        default_leverage: Default leverage.
        leverages: Per-instrument max leverage overrides.

    Returns:
        EtherealVenueConfig for paper trading.
    """
    return EtherealVenueConfig(
        name="ETHEREAL",
        starting_balance_usde=starting_balance_usde,
        default_leverage=default_leverage,
        leverages=leverages or {},
        latency_preset=EtherealLatencyPreset.TESTNET,
    )
