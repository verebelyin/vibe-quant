"""Risk management Actors for NautilusTrader.

Provides strategy-level and portfolio-level risk monitoring with circuit breaker
functionality to halt trading when risk limits are breached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig

from vibe_quant.risk.config import PortfolioRiskConfig, StrategyRiskConfig

if TYPE_CHECKING:
    from nautilus_trader.model.events import OrderFilled, PositionChanged, PositionClosed
    from nautilus_trader.model.identifiers import StrategyId


logger = logging.getLogger(__name__)


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


@dataclass
class StrategyRiskState:
    """Tracks risk state for a single strategy.

    Attributes:
        high_water_mark: Maximum equity reached.
        current_drawdown_pct: Current drawdown from HWM.
        daily_pnl: PnL since start of day.
        daily_start_equity: Equity at start of day.
        consecutive_losses: Count of consecutive losing trades.
        current_date: Date for daily reset tracking.
        state: Current risk state.
        position_count: Number of open positions.
    """

    high_water_mark: Decimal = Decimal("0")
    current_drawdown_pct: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    daily_start_equity: Decimal = Decimal("0")
    consecutive_losses: int = 0
    current_date: date | None = None
    state: RiskState = RiskState.ACTIVE
    position_count: int = 0


class StrategyRiskActorConfig(ActorConfig, frozen=True):
    """Configuration for StrategyRiskActor.

    Attributes:
        strategy_id: ID of strategy to monitor.
        max_drawdown_pct: Maximum drawdown before halt.
        max_daily_loss_pct: Maximum daily loss before halt.
        max_consecutive_losses: Maximum consecutive losses before halt.
        max_position_count: Maximum open positions allowed.
    """

    strategy_id: str
    max_drawdown_pct: Decimal = Decimal("0.15")
    max_daily_loss_pct: Decimal = Decimal("0.02")
    max_consecutive_losses: int = 10
    max_position_count: int = 5


class StrategyRiskActor(Actor):  # type: ignore[misc]
    """Monitors and enforces strategy-level risk limits.

    Tracks high water mark, drawdown, daily loss, and consecutive losses.
    Halts strategy trading when limits are breached by canceling orders
    and logging risk events.

    Example usage:
        config = StrategyRiskActorConfig(
            component_id=ComponentId("RISK-001"),
            strategy_id="RSI_MEAN_REV",
            max_drawdown_pct=Decimal("0.15"),
        )
        risk_actor = StrategyRiskActor(config)
    """

    def __init__(self, config: StrategyRiskActorConfig) -> None:
        """Initialize StrategyRiskActor.

        Args:
            config: Actor configuration.
        """
        super().__init__(config)
        self._config = config
        self._risk_config = StrategyRiskConfig(
            max_drawdown_pct=config.max_drawdown_pct,
            max_daily_loss_pct=config.max_daily_loss_pct,
            max_consecutive_losses=config.max_consecutive_losses,
            max_position_count=config.max_position_count,
        )
        self._state = StrategyRiskState()
        self._strategy_id_str = config.strategy_id

    @property
    def risk_state(self) -> RiskState:
        """Get current risk state."""
        return self._state.state

    @property
    def current_drawdown_pct(self) -> Decimal:
        """Get current drawdown percentage."""
        return self._state.current_drawdown_pct

    @property
    def consecutive_losses(self) -> int:
        """Get current consecutive loss count."""
        return self._state.consecutive_losses

    @property
    def daily_pnl(self) -> Decimal:
        """Get current daily PnL."""
        return self._state.daily_pnl

    def on_start(self) -> None:
        """Initialize actor on start."""
        # Subscribe to order fills for our strategy
        self.subscribe_order_book_deltas(instrument_id=None)

        # Initialize daily tracking
        self._reset_daily_state()

        self._log_info(f"StrategyRiskActor started for {self._strategy_id_str}")

    def on_order_filled(self, event: OrderFilled) -> None:
        """Handle order fill events.

        Updates risk state based on filled orders. Checks PnL for position
        closes and tracks consecutive losses.

        Args:
            event: The order filled event.
        """
        # Only process fills for our strategy
        if str(event.strategy_id) != self._strategy_id_str:
            return

        # Check for daily reset
        self._check_daily_reset()

        # Update state and check risk
        self._update_after_fill(event)
        self._check_risk()

    def on_position_changed(self, event: PositionChanged) -> None:
        """Handle position changed events.

        Args:
            event: The position changed event.
        """
        if str(event.strategy_id) != self._strategy_id_str:
            return

        self._update_position_count()
        self._check_risk()

    def on_position_closed(self, event: PositionClosed) -> None:
        """Handle position closed events.

        Tracks PnL from closed positions for consecutive loss counting.

        Args:
            event: The position closed event.
        """
        if str(event.strategy_id) != self._strategy_id_str:
            return

        # Extract realized PnL from the event
        realized_pnl = Decimal(str(event.realized_pnl))

        # Update consecutive loss tracking
        if realized_pnl < Decimal("0"):
            self._state.consecutive_losses += 1
        else:
            self._state.consecutive_losses = 0

        # Update daily PnL
        self._state.daily_pnl += realized_pnl

        # Update HWM and drawdown
        self._update_hwm_and_drawdown()

        # Update position count
        self._update_position_count()

        self._check_risk()

    def check_risk(self) -> RiskState:
        """Check current risk state against limits.

        Returns:
            Current RiskState after evaluation.
        """
        return self._check_risk()

    def _check_risk(self) -> RiskState:
        """Internal risk check implementation.

        Returns:
            Current RiskState after evaluation.
        """
        if self._state.state == RiskState.HALTED:
            return RiskState.HALTED

        # Check drawdown limit
        if self._state.current_drawdown_pct >= self._risk_config.max_drawdown_pct:
            self._halt_strategy("drawdown", self._state.current_drawdown_pct)
            return RiskState.HALTED

        # Check daily loss limit
        if self._state.daily_start_equity > Decimal("0"):
            daily_loss_pct = abs(self._state.daily_pnl) / self._state.daily_start_equity
            if self._state.daily_pnl < Decimal("0") and daily_loss_pct >= self._risk_config.max_daily_loss_pct:
                self._halt_strategy("daily_loss", daily_loss_pct)
                return RiskState.HALTED

        # Check consecutive losses
        if self._state.consecutive_losses >= self._risk_config.max_consecutive_losses:
            self._halt_strategy(
                "consecutive_losses", Decimal(str(self._state.consecutive_losses))
            )
            return RiskState.HALTED

        # Check position count (warning only - prevents new positions)
        if self._state.position_count >= self._risk_config.max_position_count:
            self._state.state = RiskState.WARNING
            return RiskState.WARNING

        self._state.state = RiskState.ACTIVE
        return RiskState.ACTIVE

    def _halt_strategy(self, metric: str, value: Decimal) -> None:
        """Halt strategy trading due to risk breach.

        Args:
            metric: Which metric triggered the halt.
            value: Current value of the breached metric.
        """
        self._state.state = RiskState.HALTED

        # Cancel all open orders for this strategy
        self._cancel_all_strategy_orders()

        # Get threshold for the metric
        thresholds = {
            "drawdown": self._risk_config.max_drawdown_pct,
            "daily_loss": self._risk_config.max_daily_loss_pct,
            "consecutive_losses": Decimal(str(self._risk_config.max_consecutive_losses)),
        }
        threshold = thresholds.get(metric, Decimal("0"))

        # Log risk event
        event = RiskEvent(
            timestamp=datetime.now(UTC).isoformat(),
            event_type="CIRCUIT_BREAKER",
            level="strategy",
            strategy_id=self._strategy_id_str,
            metric=metric,
            current_value=str(value),
            threshold=str(threshold),
            state="HALTED",
            message=f"Strategy {self._strategy_id_str} halted: {metric}={value} >= {threshold}",
        )

        logger.warning(
            "Risk circuit breaker triggered",
            extra={"risk_event": event.to_dict()},
        )
        self._log_warning(event.message)

    def _cancel_all_strategy_orders(self) -> None:
        """Cancel all open orders for the monitored strategy."""
        # In NautilusTrader, we use the cache to find open orders
        # and then cancel them through the trading commands
        try:
            open_orders = self.cache.orders_open(strategy_id=self._get_strategy_id())
            for order in open_orders:
                self.cancel_order(order)
                self._log_info(f"Canceled order {order.client_order_id}")
        except Exception as e:
            self._log_error(f"Error canceling orders: {e}")

    def _get_strategy_id(self) -> StrategyId:
        """Get StrategyId object for the monitored strategy."""
        from nautilus_trader.model.identifiers import StrategyId

        return StrategyId(self._strategy_id_str)

    def _update_after_fill(self, event: OrderFilled) -> None:
        """Update state after an order fill.

        Args:
            event: The order filled event.
        """
        # Update HWM and drawdown based on current equity
        self._update_hwm_and_drawdown()

    def _update_hwm_and_drawdown(self) -> None:
        """Update high water mark and current drawdown."""
        equity = self._get_current_equity()
        if equity <= Decimal("0"):
            return

        # Update HWM
        if equity > self._state.high_water_mark:
            self._state.high_water_mark = equity

        # Calculate drawdown
        if self._state.high_water_mark > Decimal("0"):
            self._state.current_drawdown_pct = (
                self._state.high_water_mark - equity
            ) / self._state.high_water_mark

    def _update_position_count(self) -> None:
        """Update count of open positions for the strategy."""
        try:
            positions = self.cache.positions_open(strategy_id=self._get_strategy_id())
            self._state.position_count = len(positions)
        except Exception:
            self._state.position_count = 0

    def _get_current_equity(self) -> Decimal:
        """Get current equity from portfolio.

        Returns:
            Current equity as Decimal.
        """
        try:
            # Access portfolio through self.portfolio
            account = self.portfolio.account(venue=None)
            if account is not None:
                balance = account.balance_total()
                if balance is not None:
                    return Decimal(str(balance.as_double()))
        except Exception as e:
            self._log_error(f"Error getting equity: {e}")
        return Decimal("0")

    def _reset_daily_state(self) -> None:
        """Reset daily tracking state."""
        today = datetime.now(UTC).date()
        self._state.current_date = today
        self._state.daily_pnl = Decimal("0")
        self._state.daily_start_equity = self._get_current_equity()

    def _check_daily_reset(self) -> None:
        """Check if daily state should be reset (new day)."""
        today = datetime.now(UTC).date()
        if self._state.current_date != today:
            self._reset_daily_state()

    def _log_info(self, msg: str) -> None:
        """Log info message."""
        logger.info(msg, extra={"strategy_id": self._strategy_id_str})

    def _log_warning(self, msg: str) -> None:
        """Log warning message."""
        logger.warning(msg, extra={"strategy_id": self._strategy_id_str})

    def _log_error(self, msg: str) -> None:
        """Log error message."""
        logger.error(msg, extra={"strategy_id": self._strategy_id_str})


class PortfolioRiskActorConfig(ActorConfig, frozen=True):
    """Configuration for PortfolioRiskActor.

    Attributes:
        max_portfolio_drawdown_pct: Maximum portfolio drawdown before halt.
        max_total_exposure_pct: Maximum total exposure as pct of equity.
        max_single_instrument_pct: Maximum concentration in single instrument.
    """

    max_portfolio_drawdown_pct: Decimal = Decimal("0.20")
    max_total_exposure_pct: Decimal = Decimal("0.50")
    max_single_instrument_pct: Decimal = Decimal("0.30")


@dataclass
class PortfolioRiskState:
    """Tracks portfolio-level risk state.

    Attributes:
        high_water_mark: Maximum portfolio equity.
        current_drawdown_pct: Current drawdown from HWM.
        total_exposure_pct: Current total exposure as pct of equity.
        instrument_exposures: Per-instrument exposure as pct of equity.
        state: Current risk state.
    """

    high_water_mark: Decimal = Decimal("0")
    current_drawdown_pct: Decimal = Decimal("0")
    total_exposure_pct: Decimal = Decimal("0")
    instrument_exposures: dict[str, Decimal] = field(default_factory=dict)
    state: RiskState = RiskState.ACTIVE


class PortfolioRiskActor(Actor):  # type: ignore[misc]
    """Monitors and enforces portfolio-level risk limits.

    Tracks total equity, exposure, and instrument concentration.
    Halts all trading when limits are breached.

    Example usage:
        config = PortfolioRiskActorConfig(
            component_id=ComponentId("PORTFOLIO-RISK"),
            max_portfolio_drawdown_pct=Decimal("0.20"),
        )
        portfolio_risk = PortfolioRiskActor(config)
    """

    def __init__(self, config: PortfolioRiskActorConfig) -> None:
        """Initialize PortfolioRiskActor.

        Args:
            config: Actor configuration.
        """
        super().__init__(config)
        self._config = config
        self._risk_config = PortfolioRiskConfig(
            max_portfolio_drawdown_pct=config.max_portfolio_drawdown_pct,
            max_total_exposure_pct=config.max_total_exposure_pct,
            max_single_instrument_pct=config.max_single_instrument_pct,
        )
        self._state = PortfolioRiskState()
        self._halted_strategies: set[str] = set()

    @property
    def risk_state(self) -> RiskState:
        """Get current portfolio risk state."""
        return self._state.state

    @property
    def current_drawdown_pct(self) -> Decimal:
        """Get current portfolio drawdown percentage."""
        return self._state.current_drawdown_pct

    @property
    def total_exposure_pct(self) -> Decimal:
        """Get current total exposure percentage."""
        return self._state.total_exposure_pct

    def on_start(self) -> None:
        """Initialize actor on start."""
        # Initialize HWM with current equity
        equity = self._get_portfolio_equity()
        if equity > Decimal("0"):
            self._state.high_water_mark = equity

        self._log_info("PortfolioRiskActor started")

    def on_order_filled(self, event: OrderFilled) -> None:
        """Handle order fill events.

        Updates portfolio risk state after any fill.

        Args:
            event: The order filled event.
        """
        self._update_state()
        self._check_risk()

    def on_position_changed(self, event: PositionChanged) -> None:
        """Handle position changed events.

        Args:
            event: The position changed event.
        """
        self._update_state()
        self._check_risk()

    def on_position_closed(self, event: PositionClosed) -> None:
        """Handle position closed events.

        Args:
            event: The position closed event.
        """
        self._update_state()
        self._check_risk()

    def check_portfolio_risk(self) -> RiskState:
        """Check current portfolio risk state.

        Returns:
            Current RiskState after evaluation.
        """
        self._update_state()
        return self._check_risk()

    def _check_risk(self) -> RiskState:
        """Internal risk check implementation.

        Returns:
            Current RiskState after evaluation.
        """
        if self._state.state == RiskState.HALTED:
            return RiskState.HALTED

        # Check portfolio drawdown
        if self._state.current_drawdown_pct >= self._risk_config.max_portfolio_drawdown_pct:
            self._halt_portfolio("portfolio_drawdown", self._state.current_drawdown_pct)
            return RiskState.HALTED

        # Check total exposure
        if self._state.total_exposure_pct > self._risk_config.max_total_exposure_pct:
            self._halt_portfolio("total_exposure", self._state.total_exposure_pct)
            return RiskState.HALTED

        # Check single instrument concentration
        for instrument_id, exposure in self._state.instrument_exposures.items():
            if exposure > self._risk_config.max_single_instrument_pct:
                self._halt_portfolio(
                    f"instrument_concentration:{instrument_id}", exposure
                )
                return RiskState.HALTED

        self._state.state = RiskState.ACTIVE
        return RiskState.ACTIVE

    def _halt_portfolio(self, metric: str, value: Decimal) -> None:
        """Halt all portfolio trading due to risk breach.

        Args:
            metric: Which metric triggered the halt.
            value: Current value of the breached metric.
        """
        self._state.state = RiskState.HALTED

        # Cancel all open orders across entire portfolio
        self._cancel_all_orders()

        # Determine threshold based on metric
        if metric == "portfolio_drawdown":
            threshold = self._risk_config.max_portfolio_drawdown_pct
        elif metric == "total_exposure":
            threshold = self._risk_config.max_total_exposure_pct
        else:
            threshold = self._risk_config.max_single_instrument_pct

        # Log risk event
        event = RiskEvent(
            timestamp=datetime.now(UTC).isoformat(),
            event_type="CIRCUIT_BREAKER",
            level="portfolio",
            strategy_id=None,
            metric=metric,
            current_value=str(value),
            threshold=str(threshold),
            state="HALTED",
            message=f"Portfolio halted: {metric}={value} >= {threshold}",
        )

        logger.warning(
            "Portfolio risk circuit breaker triggered",
            extra={"risk_event": event.to_dict()},
        )
        self._log_warning(event.message)

    def _cancel_all_orders(self) -> None:
        """Cancel all open orders across the portfolio."""
        try:
            open_orders = self.cache.orders_open()
            for order in open_orders:
                self.cancel_order(order)
                self._log_info(f"Canceled order {order.client_order_id}")
        except Exception as e:
            self._log_error(f"Error canceling orders: {e}")

    def _update_state(self) -> None:
        """Update portfolio risk state."""
        equity = self._get_portfolio_equity()
        if equity <= Decimal("0"):
            return

        # Update HWM
        if equity > self._state.high_water_mark:
            self._state.high_water_mark = equity

        # Calculate drawdown
        if self._state.high_water_mark > Decimal("0"):
            self._state.current_drawdown_pct = (
                self._state.high_water_mark - equity
            ) / self._state.high_water_mark

        # Calculate exposures
        self._update_exposures(equity)

    def _update_exposures(self, equity: Decimal) -> None:
        """Update exposure calculations.

        Args:
            equity: Current portfolio equity.
        """
        if equity <= Decimal("0"):
            self._state.total_exposure_pct = Decimal("0")
            self._state.instrument_exposures = {}
            return

        try:
            # Get all open positions
            positions = self.cache.positions_open()

            total_notional = Decimal("0")
            instrument_notionals: dict[str, Decimal] = {}

            for position in positions:
                # Calculate notional value (quantity * avg price)
                notional = abs(
                    Decimal(str(position.quantity.as_double()))
                    * Decimal(str(position.avg_px_open))
                )
                total_notional += notional

                # Track per-instrument
                instrument_id = str(position.instrument_id)
                if instrument_id not in instrument_notionals:
                    instrument_notionals[instrument_id] = Decimal("0")
                instrument_notionals[instrument_id] += notional

            # Calculate percentages
            self._state.total_exposure_pct = total_notional / equity
            self._state.instrument_exposures = {
                k: v / equity for k, v in instrument_notionals.items()
            }

        except Exception as e:
            self._log_error(f"Error calculating exposures: {e}")

    def _get_portfolio_equity(self) -> Decimal:
        """Get total portfolio equity.

        Returns:
            Portfolio equity as Decimal.
        """
        try:
            account = self.portfolio.account(venue=None)
            if account is not None:
                balance = account.balance_total()
                if balance is not None:
                    return Decimal(str(balance.as_double()))
        except Exception as e:
            self._log_error(f"Error getting portfolio equity: {e}")
        return Decimal("0")

    def _log_info(self, msg: str) -> None:
        """Log info message."""
        logger.info(msg, extra={"level": "portfolio"})

    def _log_warning(self, msg: str) -> None:
        """Log warning message."""
        logger.warning(msg, extra={"level": "portfolio"})

    def _log_error(self, msg: str) -> None:
        """Log error message."""
        logger.error(msg, extra={"level": "portfolio"})


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
