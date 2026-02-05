"""Paper trading module for live simulated execution.

Provides TradingNode configuration for Binance testnet with:
- WebSocket data feed
- Simulated execution against testnet
- Strategy deployment from DSL
- Position sizing and risk module attachment
"""

from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    PaperTradingConfig,
    create_trading_node_config,
)
from vibe_quant.paper.node import PaperTradingNode

__all__ = [
    "BinanceTestnetConfig",
    "PaperTradingConfig",
    "PaperTradingNode",
    "create_trading_node_config",
]
