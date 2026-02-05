"""Paper trading node implementation.

Provides PaperTradingNode class that wraps NautilusTrader TradingNode
for paper trading with Binance testnet.
"""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.logging.events import EventType, create_event
from vibe_quant.logging.writer import EventWriter
from vibe_quant.paper.config import (
    ConfigurationError,
    PaperTradingConfig,
    create_trading_node_config,
)
from vibe_quant.paper.errors import ErrorContext, ErrorHandler

if TYPE_CHECKING:
    from vibe_quant.dsl.schema import StrategyDSL


class NodeState(StrEnum):
    """State of the paper trading node."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    HALTED = "halted"
    STOPPED = "stopped"
    ERROR = "error"


class HaltReason(StrEnum):
    """Reason for halting the trading node."""

    MAX_DRAWDOWN = "max_drawdown"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_CONSECUTIVE_LOSSES = "max_consecutive_losses"
    MANUAL = "manual"
    ERROR = "error"
    SIGNAL = "signal"


@dataclass
class NodeStatus:
    """Current status of the paper trading node.

    Attributes:
        state: Current node state.
        started_at: When node was started.
        updated_at: Last status update.
        halt_reason: Reason if halted.
        error_message: Error message if in error state.
        positions: Number of open positions.
        daily_pnl: Daily PnL.
        total_pnl: Total PnL since start.
        trades_today: Number of trades today.
        consecutive_losses: Current consecutive loss count.
    """

    state: NodeState = NodeState.INITIALIZING
    started_at: datetime | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    halt_reason: HaltReason | None = None
    error_message: str | None = None
    positions: int = 0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "state": self.state.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat(),
            "halt_reason": self.halt_reason.value if self.halt_reason else None,
            "error_message": self.error_message,
            "positions": self.positions,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
        }


class PaperTradingNode:
    """Paper trading node for live simulated execution.

    Wraps NautilusTrader TradingNode for paper trading with Binance testnet.
    Provides:
    - Strategy deployment from DSL
    - Position sizing and risk module attachment
    - State persistence for crash recovery
    - Signal handling for graceful shutdown

    Example:
        config = PaperTradingConfig.create(
            trader_id="PAPER-001",
            symbols=["BTCUSDT", "ETHUSDT"],
            strategy_id=42,
        )
        node = PaperTradingNode(config)
        await node.start()  # Runs until signal
    """

    def __init__(self, config: PaperTradingConfig) -> None:
        """Initialize paper trading node.

        Args:
            config: Paper trading configuration.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        errors = config.validate()
        if errors:
            raise ConfigurationError(f"Invalid configuration: {'; '.join(errors)}")

        self._config = config
        self._state_manager = StateManager(config.db_path)
        self._compiler = StrategyCompiler()
        self._status = NodeStatus()
        self._event_writer: EventWriter | None = None
        self._trading_node: Any = None  # NautilusTrader TradingNode
        self._shutdown_event = asyncio.Event()
        self._strategy: StrategyDSL | None = None
        self._error_handler = ErrorHandler(
            on_halt=self._on_error_halt,
            on_alert=self._on_error_alert,
        )
        self._previous_state: NodeState | None = None  # For resume from halt

    @property
    def config(self) -> PaperTradingConfig:
        """Get node configuration."""
        return self._config

    @property
    def status(self) -> NodeStatus:
        """Get current node status."""
        return self._status

    @property
    def error_handler(self) -> ErrorHandler:
        """Get error handler."""
        return self._error_handler

    def _on_error_halt(self, reason: str, message: str) -> None:
        """Callback when error handler triggers halt.

        Args:
            reason: Halt reason identifier.
            message: Human-readable message.
        """
        # Store previous state for potential resume
        if self._status.state in (NodeState.RUNNING, NodeState.PAUSED):
            self._previous_state = self._status.state

        self.halt(HaltReason.ERROR, message)

    def _on_error_alert(self, alert_type: str, context: ErrorContext) -> None:
        """Callback when error handler triggers alert.

        Args:
            alert_type: Type of alert.
            context: Error context with details.
        """
        self._write_event(
            EventType.RISK_CHECK,
            {
                "action": "alert",
                "alert_type": alert_type,
                "error_category": context.category.value,
                "error_type": type(context.error).__name__,
                "error_message": str(context.error),
                "operation": context.operation,
                "symbol": context.symbol,
                "retry_count": context.retry_count,
            },
        )

    def _load_strategy(self) -> StrategyDSL:
        """Load and validate strategy from database.

        Returns:
            Validated StrategyDSL model.

        Raises:
            ConfigurationError: If strategy not found or invalid.
        """
        strategy_id = self._config.strategy_id
        if strategy_id is None:
            raise ConfigurationError("strategy_id is required")

        strategy_data = self._state_manager.get_strategy(strategy_id)
        if strategy_data is None:
            raise ConfigurationError(f"Strategy {strategy_id} not found")

        dsl_config = strategy_data["dsl_config"]

        # Validate DSL
        try:
            dsl = validate_strategy_dict(dsl_config)
        except Exception as e:
            raise ConfigurationError(f"Invalid strategy DSL: {e}") from e

        return dsl

    def _compile_strategy(self, dsl: StrategyDSL) -> Any:
        """Compile strategy DSL to NautilusTrader Strategy.

        Args:
            dsl: Validated strategy DSL.

        Returns:
            Compiled NautilusTrader Strategy.

        Raises:
            ConfigurationError: If compilation fails.
        """
        try:
            return self._compiler.compile(dsl)
        except Exception as e:
            raise ConfigurationError(f"Strategy compilation failed: {e}") from e

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        def signal_handler(sig: int) -> None:
            self._status.state = NodeState.STOPPED
            self._status.halt_reason = HaltReason.SIGNAL
            self._shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler, sig)

    def _write_event(self, event_type: EventType, data: dict[str, Any]) -> None:
        """Write event to log.

        Args:
            event_type: Type of event.
            data: Event data.
        """
        if self._event_writer is not None:
            strategy_name = self._strategy.name if self._strategy else "unknown"
            event = create_event(
                event_type=event_type,
                run_id=self._config.trader_id,
                strategy_name=strategy_name,
                data=data,  # type: ignore[arg-type]
            )
            self._event_writer.write(event)

    async def _initialize(self) -> None:
        """Initialize node components.

        Raises:
            ConfigurationError: If initialization fails.
        """
        self._status.state = NodeState.INITIALIZING
        self._status.updated_at = datetime.now(UTC)

        # Load and compile strategy
        self._strategy = self._load_strategy()
        compiled_strategy = self._compile_strategy(self._strategy)

        # Create event writer
        self._config.logs_path.mkdir(parents=True, exist_ok=True)
        self._event_writer = EventWriter(
            run_id=self._config.trader_id,
            base_path=self._config.logs_path,
        )

        # Log start event
        self._write_event(
            EventType.SIGNAL,
            {
                "action": "start",
                "trader_id": self._config.trader_id,
                "strategy_id": self._config.strategy_id,
                "symbols": self._config.symbols,
                "testnet": self._config.binance.testnet,
            },
        )

        # Create trading node config
        node_config = create_trading_node_config(self._config)

        # Note: Actual TradingNode instantiation requires nautilus_trader
        # This is a placeholder for the actual implementation
        self._trading_node = {
            "config": node_config,
            "strategy": compiled_strategy,
            "status": "initialized",
        }

    async def _run_loop(self) -> None:
        """Main event loop for paper trading.

        This is a placeholder for the actual NautilusTrader event loop.
        In production, this would delegate to TradingNode.run().
        """
        self._status.state = NodeState.RUNNING
        self._status.started_at = datetime.now(UTC)
        self._status.updated_at = datetime.now(UTC)

        self._write_event(
            EventType.SIGNAL,
            {
                "action": "state_change",
                "from_state": NodeState.INITIALIZING.value,
                "to_state": NodeState.RUNNING.value,
            },
        )

        # Wait for shutdown signal
        await self._shutdown_event.wait()

    async def _shutdown(self) -> None:
        """Graceful shutdown of node components."""
        self._status.state = NodeState.STOPPED
        self._status.updated_at = datetime.now(UTC)

        self._write_event(
            EventType.SIGNAL,
            {
                "action": "stop",
                "trader_id": self._config.trader_id,
                "reason": self._status.halt_reason.value if self._status.halt_reason else "normal",
                "total_pnl": self._status.total_pnl,
            },
        )

        # Close event writer
        if self._event_writer is not None:
            self._event_writer.close()
            self._event_writer = None

        # Close state manager
        self._state_manager.close()

    async def start(self) -> None:
        """Start the paper trading node.

        Runs until interrupted by signal (SIGINT/SIGTERM) or halt condition.
        """
        try:
            self._setup_signal_handlers()
            await self._initialize()
            await self._run_loop()
        except Exception as e:
            self._status.state = NodeState.ERROR
            self._status.error_message = str(e)
            self._write_event(EventType.RISK_CHECK, {"action": "error", "error": str(e)})
            raise
        finally:
            await self._shutdown()

    def halt(self, reason: HaltReason, message: str | None = None) -> None:
        """Halt the trading node.

        Args:
            reason: Reason for halting.
            message: Optional message.
        """
        self._status.state = NodeState.HALTED
        self._status.halt_reason = reason
        self._status.error_message = message
        self._status.updated_at = datetime.now(UTC)

        self._write_event(
            EventType.RISK_CHECK,
            {
                "action": "halt",
                "reason": reason.value,
                "message": message,
            },
        )

        self._shutdown_event.set()

    def pause(self) -> None:
        """Pause trading (close no new positions)."""
        if self._status.state == NodeState.RUNNING:
            self._status.state = NodeState.PAUSED
            self._status.updated_at = datetime.now(UTC)

            self._write_event(
                EventType.SIGNAL,
                {
                    "action": "state_change",
                    "from_state": NodeState.RUNNING.value,
                    "to_state": NodeState.PAUSED.value,
                },
            )

    def resume(self) -> None:
        """Resume trading from paused state."""
        if self._status.state == NodeState.PAUSED:
            self._status.state = NodeState.RUNNING
            self._status.updated_at = datetime.now(UTC)

            self._write_event(
                EventType.SIGNAL,
                {
                    "action": "state_change",
                    "from_state": NodeState.PAUSED.value,
                    "to_state": NodeState.RUNNING.value,
                },
            )

    def resume_from_halt(self) -> bool:
        """Resume trading from halted state.

        Can only resume if halted due to error and previous state was recoverable.
        Returns True if resumed, False if cannot resume.

        Returns:
            True if successfully resumed, False otherwise.
        """
        if self._status.state != NodeState.HALTED:
            return False

        if self._status.halt_reason != HaltReason.ERROR:
            # Only error halts can be resumed (drawdown/loss halts need manual review)
            return False

        # Determine target state (previous state or RUNNING)
        target_state = self._previous_state or NodeState.RUNNING
        if target_state == NodeState.HALTED:
            target_state = NodeState.RUNNING

        prev_state = self._status.state
        self._status.state = target_state
        self._status.halt_reason = None
        self._status.error_message = None
        self._status.updated_at = datetime.now(UTC)
        self._previous_state = None

        self._write_event(
            EventType.SIGNAL,
            {
                "action": "state_change",
                "from_state": prev_state.value,
                "to_state": target_state.value,
                "resumed": True,
            },
        )

        return True

    def handle_error(
        self,
        error: BaseException,
        operation: str = "",
        symbol: str = "",
    ) -> ErrorContext:
        """Handle error through the error handler.

        Args:
            error: Exception that occurred.
            operation: Operation being performed.
            symbol: Symbol involved (if applicable).

        Returns:
            ErrorContext with classification and handling result.
        """
        return self._error_handler.handle_error(
            error=error,
            operation=operation,
            symbol=symbol,
        )


async def run_paper_trading(config: PaperTradingConfig) -> None:
    """Run paper trading with the given configuration.

    Convenience function that creates and starts a PaperTradingNode.

    Args:
        config: Paper trading configuration.
    """
    node = PaperTradingNode(config)
    await node.start()
