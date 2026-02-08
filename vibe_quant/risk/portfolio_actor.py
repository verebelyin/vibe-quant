"""Portfolio-level risk monitoring Actor for NautilusTrader.

Tracks total equity, exposure, and instrument concentration.
Halts all trading when limits are breached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig

from vibe_quant.risk.config import PortfolioRiskConfig
from vibe_quant.risk.types import RiskEvent, RiskState

if TYPE_CHECKING:
    from nautilus_trader.model.events import OrderFilled, PositionChanged, PositionClosed


logger = logging.getLogger(__name__)


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
        """Initialize PortfolioRiskActor."""
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
        equity = self._get_portfolio_equity()
        if equity > Decimal("0"):
            self._state.high_water_mark = equity
        self._log_info("PortfolioRiskActor started")

    def on_order_filled(self, event: OrderFilled) -> None:
        """Handle order fill events."""
        self._update_state()
        self._check_risk()

    def on_position_changed(self, event: PositionChanged) -> None:
        """Handle position changed events."""
        self._update_state()
        self._check_risk()

    def on_position_closed(self, event: PositionClosed) -> None:
        """Handle position closed events."""
        self._update_state()
        self._check_risk()

    def check_portfolio_risk(self) -> RiskState:
        """Check current portfolio risk state."""
        self._update_state()
        return self._check_risk()

    def _check_risk(self) -> RiskState:
        """Internal risk check implementation."""
        if self._state.state == RiskState.HALTED:
            return RiskState.HALTED

        if self._state.current_drawdown_pct >= self._risk_config.max_portfolio_drawdown_pct:
            self._halt_portfolio("portfolio_drawdown", self._state.current_drawdown_pct)
            return RiskState.HALTED

        if self._state.total_exposure_pct > self._risk_config.max_total_exposure_pct:
            self._halt_portfolio("total_exposure", self._state.total_exposure_pct)
            return RiskState.HALTED

        for instrument_id, exposure in self._state.instrument_exposures.items():
            if exposure > self._risk_config.max_single_instrument_pct:
                self._halt_portfolio(
                    f"instrument_concentration:{instrument_id}", exposure
                )
                return RiskState.HALTED

        self._state.state = RiskState.ACTIVE
        return RiskState.ACTIVE

    def _halt_portfolio(self, metric: str, value: Decimal) -> None:
        """Halt all portfolio trading due to risk breach."""
        self._state.state = RiskState.HALTED
        self._cancel_all_orders()

        if metric == "portfolio_drawdown":
            threshold = self._risk_config.max_portfolio_drawdown_pct
        elif metric == "total_exposure":
            threshold = self._risk_config.max_total_exposure_pct
        else:
            threshold = self._risk_config.max_single_instrument_pct

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

        if equity > self._state.high_water_mark:
            self._state.high_water_mark = equity

        if self._state.high_water_mark > Decimal("0"):
            self._state.current_drawdown_pct = (
                self._state.high_water_mark - equity
            ) / self._state.high_water_mark

        self._update_exposures(equity)

    def _update_exposures(self, equity: Decimal) -> None:
        """Update exposure calculations."""
        if equity <= Decimal("0"):
            self._state.total_exposure_pct = Decimal("0")
            self._state.instrument_exposures = {}
            return

        try:
            positions = self.cache.positions_open()

            total_notional = Decimal("0")
            instrument_notionals: dict[str, Decimal] = {}

            for position in positions:
                notional = abs(
                    Decimal(str(position.quantity.as_double()))
                    * Decimal(str(position.avg_px_open))
                )
                total_notional += notional

                instrument_id = str(position.instrument_id)
                if instrument_id not in instrument_notionals:
                    instrument_notionals[instrument_id] = Decimal("0")
                instrument_notionals[instrument_id] += notional

            self._state.total_exposure_pct = total_notional / equity
            self._state.instrument_exposures = {
                k: v / equity for k, v in instrument_notionals.items()
            }

        except Exception as e:
            self._log_error(f"Error calculating exposures: {e}")

    def _get_portfolio_equity(self) -> Decimal:
        """Get total portfolio equity."""
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
