"""Structured event logging for vibe-quant backtests.

Provides JSONL event logging for backtest events with DuckDB querying support.

Usage:
    from vibe_quant.logging import EventWriter, EventType, Event

    with EventWriter(run_id="abc123") as writer:
        event = Event(
            timestamp=datetime.now(UTC),
            event_type=EventType.SIGNAL,
            run_id="abc123",
            strategy_name="rsi_mean_reversion",
            data={"indicator": "rsi", "value": 28.5},
        )
        writer.write(event)

Querying:
    from vibe_quant.logging import query_events, query_events_df

    events = query_events("abc123", event_type=EventType.SIGNAL)
    df = query_events_df("abc123")
"""

from vibe_quant.logging.events import (
    Event,
    EventType,
    FillEvent,
    FundingEvent,
    LiquidationEvent,
    OrderEvent,
    PositionCloseEvent,
    PositionOpenEvent,
    RiskCheckEvent,
    SignalEvent,
    TimeFilterEvent,
    create_event,
)
from vibe_quant.logging.query import (
    count_events_by_type,
    get_run_summary,
    list_runs,
    query_events,
    query_events_df,
)
from vibe_quant.logging.writer import EventWriter

__all__ = [
    # Event types
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
    # Writer
    "EventWriter",
    # Query
    "query_events",
    "query_events_df",
    "count_events_by_type",
    "get_run_summary",
    "list_runs",
]
