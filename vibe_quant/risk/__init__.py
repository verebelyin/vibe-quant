"""Position sizing modules and risk management circuit breakers."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Risk configuration - no NautilusTrader dependency
from vibe_quant.risk.config import (
    PortfolioRiskConfig,
    StrategyRiskConfig,
    create_default_portfolio_risk_config,
    create_default_strategy_risk_config,
)

# Position sizing - no NautilusTrader dependency
from vibe_quant.risk.sizing import (
    ATRConfig,
    ATRSizer,
    FixedFractionalConfig,
    FixedFractionalSizer,
    KellyConfig,
    KellySizer,
    PositionSizer,
    SizerConfig,
)

# Risk actors - requires NautilusTrader (lazy import to allow config-only usage)
if TYPE_CHECKING:
    from vibe_quant.risk.actors import (
        PortfolioRiskActor,
        PortfolioRiskActorConfig,
        PortfolioRiskState,
        RiskEvent,
        RiskState,
        StrategyRiskActor,
        StrategyRiskActorConfig,
        StrategyRiskState,
    )

__all__ = [
    # Position sizing
    "ATRConfig",
    "ATRSizer",
    "FixedFractionalConfig",
    "FixedFractionalSizer",
    "KellyConfig",
    "KellySizer",
    "PositionSizer",
    "SizerConfig",
    # Risk configuration
    "PortfolioRiskConfig",
    "StrategyRiskConfig",
    "create_default_portfolio_risk_config",
    "create_default_strategy_risk_config",
    # Risk actors (require NautilusTrader - use: from vibe_quant.risk.actors import ...)
    "PortfolioRiskActor",
    "PortfolioRiskActorConfig",
    "PortfolioRiskState",
    "RiskEvent",
    "RiskState",
    "StrategyRiskActor",
    "StrategyRiskActorConfig",
    "StrategyRiskState",
]


def __getattr__(name: str) -> object:
    """Lazy import for NautilusTrader-dependent classes."""
    if name in (
        "PortfolioRiskActor",
        "PortfolioRiskActorConfig",
        "PortfolioRiskState",
        "RiskEvent",
        "RiskState",
        "StrategyRiskActor",
        "StrategyRiskActorConfig",
        "StrategyRiskState",
    ):
        from vibe_quant.risk import actors

        return getattr(actors, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
