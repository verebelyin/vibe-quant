"""Risk management Actors for NautilusTrader.

Provides strategy-level and portfolio-level risk monitoring with circuit breaker
functionality to halt trading when risk limits are breached.

This module re-exports from :mod:`types`, :mod:`strategy_actor`, and
:mod:`portfolio_actor` for backward compatibility.
"""

from vibe_quant.risk.portfolio_actor import (
    PortfolioRiskActor,
    PortfolioRiskActorConfig,
    PortfolioRiskState,
)
from vibe_quant.risk.strategy_actor import (
    StrategyRiskActor,
    StrategyRiskActorConfig,
    StrategyRiskState,
)
from vibe_quant.risk.types import RiskEvent, RiskState

__all__ = [
    "RiskState",
    "RiskEvent",
    "StrategyRiskState",
    "StrategyRiskActorConfig",
    "StrategyRiskActor",
    "PortfolioRiskState",
    "PortfolioRiskActorConfig",
    "PortfolioRiskActor",
]
