"""Paper trading module for live simulated execution.

Provides TradingNode configuration for Binance testnet with:
- WebSocket data feed
- Simulated execution against testnet
- Strategy deployment from DSL
- Position sizing and risk module attachment
- Error handling with retry logic
"""

from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    PaperTradingConfig,
    create_trading_node_config,
)
from vibe_quant.paper.errors import (
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    RetryConfig,
    classify_error,
)
from vibe_quant.paper.node import PaperTradingNode
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence, recover_state

__all__ = [
    "BinanceTestnetConfig",
    "ErrorCategory",
    "ErrorContext",
    "ErrorHandler",
    "PaperTradingConfig",
    "PaperTradingNode",
    "RetryConfig",
    "StateCheckpoint",
    "StatePersistence",
    "classify_error",
    "create_trading_node_config",
    "recover_state",
]
