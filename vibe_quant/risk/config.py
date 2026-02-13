"""Risk configuration dataclasses for strategy and portfolio level risk management."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class StrategyRiskConfig:
    """Configuration for strategy-level risk management.

    Monitors per-strategy equity, drawdown, and daily loss limits.
    When limits are breached, the strategy is halted.

    Attributes:
        max_drawdown_pct: Maximum drawdown percentage before halting strategy.
            Measured from high water mark. E.g., 0.15 = halt at 15% drawdown.
        max_daily_loss_pct: Maximum daily loss percentage before halting for the day.
            Resets at UTC midnight. E.g., 0.02 = halt at 2% daily loss.
        max_consecutive_losses: Maximum consecutive losing trades before halting.
            E.g., 10 = halt after 10 consecutive losses.
        max_position_count: Maximum number of open positions allowed.
            E.g., 5 = no new positions when 5 are already open.
        drawdown_scale_pct: Drawdown threshold where position scaling begins.
            At this DD level, new positions are scaled to 50% size.
            Linear interpolation from 100% (0 DD) to 50% (scale threshold)
            to 0% (halt threshold). E.g., 0.10 = start scaling at 10% DD.
            None disables drawdown scaling.
        cooldown_after_halt_hours: Hours to wait before resuming after halt.
            0 means manual resume only (no auto-resume).
    """

    max_drawdown_pct: Decimal = Decimal("0.15")
    max_daily_loss_pct: Decimal = Decimal("0.02")
    max_consecutive_losses: int = 10
    max_position_count: int = 5
    drawdown_scale_pct: Decimal | None = Decimal("0.10")
    cooldown_after_halt_hours: int = 24

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_drawdown_pct <= Decimal("0"):
            raise ValueError("max_drawdown_pct must be positive")
        if self.max_drawdown_pct > Decimal("1"):
            raise ValueError("max_drawdown_pct must be <= 1 (100%)")
        if self.max_daily_loss_pct <= Decimal("0"):
            raise ValueError("max_daily_loss_pct must be positive")
        if self.max_daily_loss_pct > Decimal("1"):
            raise ValueError("max_daily_loss_pct must be <= 1 (100%)")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be >= 1")
        if self.max_position_count < 1:
            raise ValueError("max_position_count must be >= 1")
        if self.drawdown_scale_pct is not None:
            if self.drawdown_scale_pct <= Decimal("0"):
                raise ValueError("drawdown_scale_pct must be positive")
            if self.drawdown_scale_pct >= self.max_drawdown_pct:
                raise ValueError("drawdown_scale_pct must be < max_drawdown_pct")
        if self.cooldown_after_halt_hours < 0:
            raise ValueError("cooldown_after_halt_hours must be >= 0")


@dataclass(frozen=True)
class PortfolioRiskConfig:
    """Configuration for portfolio-level risk management.

    Monitors total equity and exposure across all strategies.
    When limits are breached, all trading is halted.

    Attributes:
        max_portfolio_drawdown_pct: Maximum portfolio drawdown before halting all.
            E.g., 0.20 = halt at 20% portfolio drawdown.
        max_total_exposure_pct: Maximum total exposure as percentage of equity.
            Sum of all position notional values / equity.
            E.g., 0.50 = max 50% of equity in positions.
        max_single_instrument_pct: Maximum concentration in single instrument.
            E.g., 0.30 = max 30% of equity in any one instrument.
        max_portfolio_heat_pct: Maximum portfolio heat (total risk across all
            open positions). Heat = sum of (position_size * distance_to_stop / equity).
            E.g., 0.06 = max 6% of equity at risk from all stops combined.
            None disables portfolio heat checking.
    """

    max_portfolio_drawdown_pct: Decimal = Decimal("0.20")
    max_total_exposure_pct: Decimal = Decimal("0.50")
    max_single_instrument_pct: Decimal = Decimal("0.30")
    max_portfolio_heat_pct: Decimal | None = Decimal("0.06")

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_portfolio_drawdown_pct <= Decimal("0"):
            raise ValueError("max_portfolio_drawdown_pct must be positive")
        if self.max_portfolio_drawdown_pct > Decimal("1"):
            raise ValueError("max_portfolio_drawdown_pct must be <= 1 (100%)")
        if self.max_total_exposure_pct <= Decimal("0"):
            raise ValueError("max_total_exposure_pct must be positive")
        if self.max_single_instrument_pct <= Decimal("0"):
            raise ValueError("max_single_instrument_pct must be positive")
        if self.max_single_instrument_pct > Decimal("1"):
            raise ValueError("max_single_instrument_pct must be <= 1 (100%)")
        if self.max_portfolio_heat_pct is not None:
            if self.max_portfolio_heat_pct <= Decimal("0"):
                raise ValueError("max_portfolio_heat_pct must be positive")
            if self.max_portfolio_heat_pct > Decimal("1"):
                raise ValueError("max_portfolio_heat_pct must be <= 1 (100%)")


def create_default_strategy_risk_config() -> StrategyRiskConfig:
    """Create default strategy risk configuration.

    Returns:
        Default StrategyRiskConfig matching SPEC.md recommendations.
    """
    return StrategyRiskConfig(
        max_drawdown_pct=Decimal("0.15"),
        max_daily_loss_pct=Decimal("0.02"),
        max_consecutive_losses=10,
        max_position_count=5,
        drawdown_scale_pct=Decimal("0.10"),
        cooldown_after_halt_hours=24,
    )


def create_default_portfolio_risk_config() -> PortfolioRiskConfig:
    """Create default portfolio risk configuration.

    Returns:
        Default PortfolioRiskConfig matching SPEC.md recommendations.
    """
    return PortfolioRiskConfig(
        max_portfolio_drawdown_pct=Decimal("0.20"),
        max_total_exposure_pct=Decimal("0.50"),
        max_single_instrument_pct=Decimal("0.30"),
        max_portfolio_heat_pct=Decimal("0.06"),
    )


__all__ = [
    "StrategyRiskConfig",
    "PortfolioRiskConfig",
    "create_default_strategy_risk_config",
    "create_default_portfolio_risk_config",
]
