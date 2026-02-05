"""Latency model configuration and presets for validation backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from nautilus_trader.config import LatencyModelConfig


class LatencyPreset(StrEnum):
    """Latency preset for execution simulation."""

    COLOCATED = "colocated"
    DOMESTIC = "domestic"
    INTERNATIONAL = "international"
    RETAIL = "retail"


# Nanosecond conversion constants
_MS_TO_NS = 1_000_000


@dataclass(frozen=True)
class LatencyValues:
    """Latency values in milliseconds for readability."""

    base_ms: float
    insert_ms: float
    update_ms: float
    cancel_ms: float

    def to_config(self) -> LatencyModelConfig:
        """Convert to NautilusTrader LatencyModelConfig.

        Note: NautilusTrader adds operation latencies to base_latency internally.
        So total insert latency = base_latency_nanos + insert_latency_nanos.
        We configure operation latencies as half of total desired latency.
        """
        return LatencyModelConfig(
            base_latency_nanos=int(self.base_ms * _MS_TO_NS),
            insert_latency_nanos=int(self.insert_ms * _MS_TO_NS),
            update_latency_nanos=int(self.update_ms * _MS_TO_NS),
            cancel_latency_nanos=int(self.cancel_ms * _MS_TO_NS),
        )


# Latency presets based on typical network conditions
# Values represent one-way latency from order submission to exchange
LATENCY_PRESETS: dict[LatencyPreset, LatencyValues] = {
    # Co-located: Server in same datacenter as exchange
    LatencyPreset.COLOCATED: LatencyValues(
        base_ms=0.5,
        insert_ms=0.25,
        update_ms=0.25,
        cancel_ms=0.25,
    ),
    # Domestic: Same country/region as exchange
    LatencyPreset.DOMESTIC: LatencyValues(
        base_ms=10.0,
        insert_ms=5.0,
        update_ms=5.0,
        cancel_ms=5.0,
    ),
    # International: Cross-continent connection
    LatencyPreset.INTERNATIONAL: LatencyValues(
        base_ms=50.0,
        insert_ms=25.0,
        update_ms=25.0,
        cancel_ms=25.0,
    ),
    # Retail: Typical home internet connection
    LatencyPreset.RETAIL: LatencyValues(
        base_ms=100.0,
        insert_ms=50.0,
        update_ms=50.0,
        cancel_ms=50.0,
    ),
}


def get_latency_model(preset: LatencyPreset | str) -> LatencyModelConfig:
    """Get LatencyModelConfig for a preset.

    Args:
        preset: Latency preset name or enum value.

    Returns:
        Configured LatencyModelConfig.

    Raises:
        ValueError: If preset is not recognized.
    """
    if isinstance(preset, str):
        try:
            preset = LatencyPreset(preset)
        except ValueError:
            valid = [p.value for p in LatencyPreset]
            raise ValueError(f"Unknown latency preset: {preset}. Valid: {valid}") from None

    values = LATENCY_PRESETS[preset]
    return values.to_config()


def create_custom_latency_model(
    base_ms: float,
    insert_ms: float | None = None,
    update_ms: float | None = None,
    cancel_ms: float | None = None,
) -> LatencyModelConfig:
    """Create custom LatencyModelConfig with specified values.

    Args:
        base_ms: Base latency in milliseconds.
        insert_ms: Insert order latency (defaults to base_ms / 2).
        update_ms: Update order latency (defaults to base_ms / 2).
        cancel_ms: Cancel order latency (defaults to base_ms / 2).

    Returns:
        Configured LatencyModelConfig.
    """
    half_base = base_ms / 2
    values = LatencyValues(
        base_ms=base_ms,
        insert_ms=insert_ms if insert_ms is not None else half_base,
        update_ms=update_ms if update_ms is not None else half_base,
        cancel_ms=cancel_ms if cancel_ms is not None else half_base,
    )
    return values.to_config()
