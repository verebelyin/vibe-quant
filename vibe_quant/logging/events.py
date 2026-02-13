"""Event types and dataclasses for structured event logging.

Provides typed events for backtest tracking: signals, orders, fills, positions,
risk checks, funding, and liquidations. All events serialize to JSON for
queryable JSONL logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self


class EventType(StrEnum):
    """Backtest event types for structured logging."""

    SIGNAL = "SIGNAL"
    TIME_FILTER = "TIME_FILTER"
    ORDER = "ORDER"
    FILL = "FILL"
    POSITION_OPEN = "POSITION_OPEN"
    POSITION_CLOSE = "POSITION_CLOSE"
    RISK_CHECK = "RISK_CHECK"
    FUNDING = "FUNDING"
    LIQUIDATION = "LIQUIDATION"
    LIFECYCLE = "LIFECYCLE"


@dataclass
class Event:
    """Base event for structured logging.

    All events include strategy context for traceability. Subclasses add
    event-specific payloads in the `data` dict.

    Attributes:
        timestamp: When event occurred (UTC).
        event_type: Type of event from EventType enum.
        run_id: Unique backtest run identifier.
        strategy_name: Name of the strategy generating the event.
        data: Event-specific payload (indicator values, order details, etc.).
    """

    timestamp: datetime
    event_type: EventType
    run_id: str
    strategy_name: str
    data: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert event to dictionary for JSON serialization.

        Returns:
            Dict with timestamp as ISO string, event_type as string, and all fields.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "run_id": self.run_id,
            "strategy_name": self.strategy_name,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        """Create Event from dictionary.

        Args:
            d: Dictionary with event fields.

        Returns:
            Event instance.
        """
        timestamp_raw = d.get("timestamp")
        if isinstance(timestamp_raw, str):
            timestamp = datetime.fromisoformat(timestamp_raw)
        elif isinstance(timestamp_raw, datetime):
            timestamp = timestamp_raw
        else:
            timestamp = datetime.now(UTC)

        event_type_raw = d.get("event_type", "SIGNAL")
        event_type = EventType(str(event_type_raw))

        data_raw = d.get("data")
        data: dict[str, object] = dict(data_raw) if isinstance(data_raw, dict) else {}

        return cls(
            timestamp=timestamp,
            event_type=event_type,
            run_id=str(d.get("run_id", "")),
            strategy_name=str(d.get("strategy_name", "")),
            data=data,
        )


@dataclass
class SignalEvent(Event):
    """Signal event with indicator values and condition evaluation.

    Attributes:
        indicator: Name of indicator that triggered.
        value: Current indicator value.
        condition: Condition string that evaluated true.
        side: Signal direction ('long' or 'short').
    """

    indicator: str = ""
    value: float = 0.0
    condition: str = ""
    side: str = ""

    def __post_init__(self) -> None:
        """Populate data dict from typed fields."""
        self.event_type = EventType.SIGNAL
        self.data = {
            "indicator": self.indicator,
            "value": self.value,
            "condition": self.condition,
            "side": self.side,
        }


@dataclass
class TimeFilterEvent(Event):
    """Time filter evaluation event.

    Attributes:
        filter_name: Name of the time filter.
        passed: Whether filter allowed trading.
        reason: Why filter blocked/allowed.
    """

    filter_name: str = ""
    passed: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        """Populate data dict from typed fields."""
        self.event_type = EventType.TIME_FILTER
        self.data = {
            "filter_name": self.filter_name,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass
class OrderEvent(Event):
    """Order submission event.

    Attributes:
        order_id: Unique order identifier.
        side: Order side ('BUY' or 'SELL').
        quantity: Order quantity.
        price: Order price (limit) or None for market.
        order_type: Order type ('MARKET', 'LIMIT', etc.).
        reason: Why order was submitted.
    """

    order_id: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float | None = None
    order_type: str = "MARKET"
    reason: str = ""

    def __post_init__(self) -> None:
        """Populate data dict from typed fields."""
        self.event_type = EventType.ORDER
        self.data = {
            "order_id": self.order_id,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "order_type": self.order_type,
            "reason": self.reason,
        }


@dataclass
class FillEvent(Event):
    """Order fill event.

    Attributes:
        order_id: Order that was filled.
        fill_price: Actual fill price.
        quantity: Filled quantity.
        fees: Fees paid.
        slippage: Price slippage from requested.
    """

    order_id: str = ""
    fill_price: float = 0.0
    quantity: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0

    def __post_init__(self) -> None:
        """Populate data dict from typed fields."""
        self.event_type = EventType.FILL
        self.data = {
            "order_id": self.order_id,
            "fill_price": self.fill_price,
            "quantity": self.quantity,
            "fees": self.fees,
            "slippage": self.slippage,
        }


@dataclass
class PositionOpenEvent(Event):
    """Position opened event.

    Attributes:
        position_id: Unique position identifier.
        symbol: Instrument symbol.
        side: Position side ('LONG' or 'SHORT').
        quantity: Position size.
        entry_price: Average entry price.
        leverage: Leverage used.
    """

    position_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    entry_price: float = 0.0
    leverage: int = 1

    def __post_init__(self) -> None:
        """Populate data dict from typed fields."""
        self.event_type = EventType.POSITION_OPEN
        self.data = {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "leverage": self.leverage,
        }


@dataclass
class PositionCloseEvent(Event):
    """Position closed event.

    Attributes:
        position_id: Position that was closed.
        symbol: Instrument symbol.
        exit_price: Exit price.
        gross_pnl: PnL before fees.
        net_pnl: PnL after fees.
        exit_reason: Why position was closed.
    """

    position_id: str = ""
    symbol: str = ""
    exit_price: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""

    def __post_init__(self) -> None:
        """Populate data dict from typed fields."""
        self.event_type = EventType.POSITION_CLOSE
        self.data = {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "exit_price": self.exit_price,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "exit_reason": self.exit_reason,
        }


@dataclass
class RiskCheckEvent(Event):
    """Risk check event.

    Attributes:
        check_type: Type of risk check performed.
        metric: Metric being checked.
        current_value: Current value of metric.
        threshold: Threshold being compared against.
        action: Action taken (if any).
        passed: Whether risk check passed.
    """

    check_type: str = ""
    metric: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    action: str = ""
    passed: bool = True

    def __post_init__(self) -> None:
        """Populate data dict from typed fields."""
        self.event_type = EventType.RISK_CHECK
        self.data = {
            "check_type": self.check_type,
            "metric": self.metric,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "action": self.action,
            "passed": self.passed,
        }


@dataclass
class FundingEvent(Event):
    """Funding rate payment event.

    Attributes:
        symbol: Instrument symbol.
        funding_rate: Applied funding rate.
        funding_payment: Payment amount (negative = paid, positive = received).
        position_value: Position value at funding time.
    """

    symbol: str = ""
    funding_rate: float = 0.0
    funding_payment: float = 0.0
    position_value: float = 0.0

    def __post_init__(self) -> None:
        """Populate data dict from typed fields."""
        self.event_type = EventType.FUNDING
        self.data = {
            "symbol": self.symbol,
            "funding_rate": self.funding_rate,
            "funding_payment": self.funding_payment,
            "position_value": self.position_value,
        }


@dataclass
class LiquidationEvent(Event):
    """Liquidation event.

    Attributes:
        position_id: Liquidated position.
        symbol: Instrument symbol.
        liquidation_price: Price at liquidation.
        quantity: Quantity liquidated.
        loss: Total loss from liquidation.
    """

    position_id: str = ""
    symbol: str = ""
    liquidation_price: float = 0.0
    quantity: float = 0.0
    loss: float = 0.0

    def __post_init__(self) -> None:
        """Populate data dict from typed fields."""
        self.event_type = EventType.LIQUIDATION
        self.data = {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "liquidation_price": self.liquidation_price,
            "quantity": self.quantity,
            "loss": self.loss,
        }


def create_event(
    event_type: EventType,
    run_id: str,
    strategy_name: str,
    data: dict[str, object],
    timestamp: datetime | None = None,
) -> Event:
    """Factory function to create events.

    Args:
        event_type: Type of event to create.
        run_id: Backtest run identifier.
        strategy_name: Strategy name.
        data: Event-specific data.
        timestamp: Event timestamp (defaults to now).

    Returns:
        Event instance with populated data.
    """
    if timestamp is None:
        timestamp = datetime.now(UTC)

    return Event(
        timestamp=timestamp,
        event_type=event_type,
        run_id=run_id,
        strategy_name=strategy_name,
        data=data,
    )


__all__ = [
    "EventType",
    "Event",
    "SignalEvent",
    "TimeFilterEvent",
    "OrderEvent",
    "FillEvent",
    "PositionOpenEvent",
    "PositionCloseEvent",
    "RiskCheckEvent",
    "FundingEvent",
    "LiquidationEvent",
    "create_event",
]
