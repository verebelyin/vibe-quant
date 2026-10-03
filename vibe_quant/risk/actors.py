"""Risk management Actors for NautilusTrader.

.. warning::
    NOT WIRED and NOT FIT FOR LIVE USE (audit 2026-10-02). Paper/live risk is
    enforced by :class:`vibe_quant.paper.guard.TradingGuard`. These actors call
    ``portfolio.account(venue=None)`` (raises), ``Decimal(str(Money))`` (raises),
    ignore unrealized PnL and rely on ``on_position_*`` hooks that NT ``Actor``
    never receives. Do not register them on a node without fixing that first.

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
