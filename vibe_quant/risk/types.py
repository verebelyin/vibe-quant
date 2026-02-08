"""Shared risk management types.

Enums and data classes used by both strategy-level and portfolio-level
risk actors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RiskState(StrEnum):
    """Risk monitoring state for strategies and portfolio."""

    ACTIVE = "ACTIVE"  # Normal trading
    WARNING = "WARNING"  # Approaching limits
    HALTED = "HALTED"  # Trading stopped


@dataclass
class RiskEvent:
    """Structured risk event for logging.

    Attributes:
        timestamp: When event occurred (ISO format).
        event_type: Type of risk event.
        level: Strategy or portfolio level.
        strategy_id: Strategy ID if applicable.
        metric: Which metric triggered the event.
        current_value: Current value of the metric.
        threshold: Threshold that was breached.
        state: Resulting risk state.
        message: Human-readable message.
    """

    timestamp: str
    event_type: str
    level: str
    strategy_id: str | None
    metric: str
    current_value: str
    threshold: str
    state: str
    message: str

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary for JSON logging."""
        return {
            "ts": self.timestamp,
            "event": self.event_type,
            "level": self.level,
            "strategy_id": self.strategy_id,
            "metric": self.metric,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "state": self.state,
            "message": self.message,
        }
