"""Strategy-level risk monitoring Actor for NautilusTrader.

Tracks high water mark, drawdown, daily loss, and consecutive losses.
Halts strategy trading when limits are breached by canceling orders
and logging risk events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig

from vibe_quant.risk.config import StrategyRiskConfig
from vibe_quant.risk.types import RiskEvent, RiskState

if TYPE_CHECKING:
    from nautilus_trader.model.events import OrderFilled, PositionChanged, PositionClosed
    from nautilus_trader.model.identifiers import StrategyId


logger = logging.getLogger(__name__)


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
        position_scale_factor: Current position size scaling factor (0.0-1.0).
            Reduced by drawdown scaling. 1.0 = full size.
        halted_at: Datetime when strategy was halted (for cooldown tracking).
    """

    high_water_mark: Decimal = Decimal("0")
    current_drawdown_pct: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    daily_start_equity: Decimal = Decimal("0")
    consecutive_losses: int = 0
    current_date: date | None = None
    state: RiskState = RiskState.ACTIVE
    position_count: int = 0
    position_scale_factor: Decimal = Decimal("1")
    halted_at: datetime | None = None


class StrategyRiskActorConfig(ActorConfig, frozen=True):
    """Configuration for StrategyRiskActor.

    Attributes:
        strategy_id: ID of strategy to monitor.
        max_drawdown_pct: Maximum drawdown before halt.
        max_daily_loss_pct: Maximum daily loss before halt.
        max_consecutive_losses: Maximum consecutive losses before halt.
        max_position_count: Maximum open positions allowed.
        drawdown_scale_pct: DD threshold where position scaling begins (None=disabled).
        cooldown_after_halt_hours: Hours to wait before auto-resuming after halt (0=manual only).
    """

    strategy_id: str
    max_drawdown_pct: Decimal = Decimal("0.15")
    max_daily_loss_pct: Decimal = Decimal("0.02")
    max_consecutive_losses: int = 10
    max_position_count: int = 5
    drawdown_scale_pct: Decimal | None = Decimal("0.10")
    cooldown_after_halt_hours: int = 24


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
            drawdown_scale_pct=config.drawdown_scale_pct,
            cooldown_after_halt_hours=config.cooldown_after_halt_hours,
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

    @property
    def position_scale_factor(self) -> Decimal:
        """Get current position size scaling factor (0.0-1.0)."""
        return self._state.position_scale_factor

    def on_start(self) -> None:
        """Initialize actor on start."""
        self._reset_daily_state()
        self._log_info(f"StrategyRiskActor started for {self._strategy_id_str}")

    def on_order_filled(self, event: OrderFilled) -> None:
        """Handle order fill events."""
        if str(event.strategy_id) != self._strategy_id_str:
            return
        self._check_daily_reset()
        self._update_after_fill(event)
        self._check_risk()

    def on_position_changed(self, event: PositionChanged) -> None:
        """Handle position changed events."""
        if str(event.strategy_id) != self._strategy_id_str:
            return
        self._update_position_count()
        self._check_risk()

    def on_position_closed(self, event: PositionClosed) -> None:
        """Handle position closed events."""
        if str(event.strategy_id) != self._strategy_id_str:
            return

        realized_pnl = Decimal(str(event.realized_pnl))
        if realized_pnl < Decimal("0"):
            self._state.consecutive_losses += 1
        else:
            self._state.consecutive_losses = 0

        self._state.daily_pnl += realized_pnl
        self._update_hwm_and_drawdown()
        self._update_position_count()
        self._check_risk()

    def check_risk(self) -> RiskState:
        """Check current risk state against limits."""
        return self._check_risk()

    def _check_risk(self) -> RiskState:
        """Internal risk check implementation.

        Includes cooldown auto-resume: if halted and cooldown period has
        elapsed, transitions back to ACTIVE.
        Also computes drawdown-based position scaling factor.
        """
        if self._state.state == RiskState.HALTED:
            if self._check_cooldown_expired():
                self._state.state = RiskState.ACTIVE
                self._state.halted_at = None
                self._log_info(
                    f"Cooldown expired for {self._strategy_id_str}, resuming"
                )
            else:
                return RiskState.HALTED

        if self._state.current_drawdown_pct >= self._risk_config.max_drawdown_pct:
            self._halt_strategy("drawdown", self._state.current_drawdown_pct)
            return RiskState.HALTED

        if self._state.daily_start_equity <= Decimal("0"):
            # Re-fetch equity — on startup the exchange may not have connected yet
            self._state.daily_start_equity = self._get_current_equity()
            if self._state.daily_start_equity <= Decimal("0"):
                # Still zero — skip daily loss check rather than halt on missing data
                return self._state.state

        daily_loss_pct = abs(self._state.daily_pnl) / self._state.daily_start_equity
        if self._state.daily_pnl < Decimal("0") and daily_loss_pct >= self._risk_config.max_daily_loss_pct:
            self._halt_strategy("daily_loss", daily_loss_pct)
            return RiskState.HALTED

        if self._state.consecutive_losses >= self._risk_config.max_consecutive_losses:
            self._halt_strategy(
                "consecutive_losses", Decimal(str(self._state.consecutive_losses))
            )
            return RiskState.HALTED

        # Drawdown-based position scaling
        self._update_position_scale_factor()

        if self._state.position_count >= self._risk_config.max_position_count:
            self._halt_strategy(
                "position_count", Decimal(str(self._state.position_count))
            )
            return RiskState.HALTED

        self._state.state = RiskState.ACTIVE
        return RiskState.ACTIVE

    def _halt_strategy(self, metric: str, value: Decimal) -> None:
        """Halt strategy trading due to risk breach."""
        self._state.state = RiskState.HALTED
        self._state.halted_at = datetime.now(UTC)
        self._state.position_scale_factor = Decimal("0")
        self._cancel_all_strategy_orders()

        thresholds = {
            "drawdown": self._risk_config.max_drawdown_pct,
            "daily_loss": self._risk_config.max_daily_loss_pct,
            "consecutive_losses": Decimal(str(self._risk_config.max_consecutive_losses)),
        }
        threshold = thresholds.get(metric, Decimal("0"))

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
        """Update state after an order fill."""
        self._update_hwm_and_drawdown()

    def _update_hwm_and_drawdown(self) -> None:
        """Update high water mark and current drawdown."""
        equity = self._get_current_equity()
        if equity <= Decimal("0"):
            return

        if equity > self._state.high_water_mark:
            self._state.high_water_mark = equity

        if self._state.high_water_mark > Decimal("0"):
            self._state.current_drawdown_pct = (
                self._state.high_water_mark - equity
            ) / self._state.high_water_mark

    def _update_position_count(self) -> None:
        """Update count of open positions for the strategy."""
        try:
            positions = self.cache.positions_open(strategy_id=self._get_strategy_id())
            self._state.position_count = len(positions)
        except Exception as exc:
            logger.warning("Failed to update position count: %s", exc)
            self._state.position_count = 0

    def _get_current_equity(self) -> Decimal:
        """Get current equity from portfolio."""
        try:
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

    def _update_position_scale_factor(self) -> None:
        """Update position size scaling based on drawdown.

        When drawdown exceeds drawdown_scale_pct, linearly reduce the
        position scale factor from 1.0 (at 0 DD) through 0.5 (at scale
        threshold) down to 0.0 (at halt threshold). If drawdown scaling
        is disabled (None), factor stays at 1.0.
        """
        scale_pct = self._risk_config.drawdown_scale_pct
        if scale_pct is None:
            self._state.position_scale_factor = Decimal("1")
            return

        dd = self._state.current_drawdown_pct
        if dd <= Decimal("0"):
            self._state.position_scale_factor = Decimal("1")
        elif dd < scale_pct:
            # Linear from 1.0 at 0 DD to 0.5 at scale_pct
            ratio = dd / scale_pct
            self._state.position_scale_factor = Decimal("1") - (ratio * Decimal("0.5"))
        elif dd < self._risk_config.max_drawdown_pct:
            # Linear from 0.5 at scale_pct to 0.0 at halt_pct
            remaining = self._risk_config.max_drawdown_pct - scale_pct
            if remaining > Decimal("0"):
                ratio = (dd - scale_pct) / remaining
                self._state.position_scale_factor = Decimal("0.5") * (Decimal("1") - ratio)
            else:
                self._state.position_scale_factor = Decimal("0")
        else:
            self._state.position_scale_factor = Decimal("0")

    def _check_cooldown_expired(self) -> bool:
        """Check if cooldown period has elapsed since halt.

        Returns:
            True if cooldown has expired and strategy can resume.
            False if still in cooldown or cooldown is disabled (0 hours).
        """
        cooldown_hours = self._risk_config.cooldown_after_halt_hours
        if cooldown_hours <= 0:
            return False  # Manual resume only

        if self._state.halted_at is None:
            return False

        halted_at = self._state.halted_at
        if halted_at.tzinfo is None:
            halted_at = halted_at.replace(tzinfo=UTC)

        elapsed = datetime.now(UTC) - halted_at
        from datetime import timedelta
        return elapsed >= timedelta(hours=cooldown_hours)

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
