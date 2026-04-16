"""Paper trading node implementation.

Provides PaperTradingNode class that wraps NautilusTrader TradingNode
for paper trading with Binance testnet.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, cast

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
from vibe_quant.paper.persistence import StateCheckpoint, StatePersistence

if TYPE_CHECKING:
    from types import ModuleType

    from vibe_quant.alerts.telegram import TelegramBot
    from vibe_quant.dsl.schema import StrategyDSL

logger = logging.getLogger(__name__)


class _TradingNodeLifecycle(Protocol):
    """Minimal lifecycle contract for live trading node integration."""

    def run(self, raise_exception: bool = False) -> None:
        """Start and run the node (blocking — used only by legacy tests)."""

    async def run_async(self) -> None:
        """Start and run the node on the current event loop."""

    def stop(self) -> None:
        """Stop the node gracefully."""

    async def stop_async(self) -> None:
        """Stop the node gracefully on the current event loop."""

    def dispose(self) -> None:
        """Dispose resources."""


class _StrategyRuntime(Protocol):
    """Minimal strategy API needed for halt-time cleanup."""

    id: object

    def cancel_order(self, order: object) -> None:
        """Cancel a single open order."""


class _TraderRuntime(Protocol):
    """Minimal trader API needed for halt-time cleanup."""

    def strategies(self) -> list[_StrategyRuntime]:
        """Return loaded runtime strategies."""


class _CacheRuntime(Protocol):
    """Minimal cache API needed for halt-time cleanup."""

    def orders_open(self, strategy_id: object | None = None) -> list[object]:
        """Return open orders filtered by strategy."""


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

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Convert to dictionary for storage/serialization."""
        return {
            "state": self.state.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat(),
            "halt_reason": self.halt_reason.value if self.halt_reason else None,
            "error_message": self.error_message,
            "positions": self.positions,
            "daily_pnl": round(self.daily_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
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
        self._trading_node: _TradingNodeLifecycle | None = None
        self._persistence: StatePersistence | None = None
        self._shutdown_event = asyncio.Event()
        self._strategy: StrategyDSL | None = None
        self._compiled_module: ModuleType | None = None
        self._compiled_strategies: list[object] = []
        self._error_handler = ErrorHandler(
            on_halt=self._on_error_halt,
            on_alert=self._on_error_alert,
        )
        self._previous_state: NodeState | None = None  # For resume from halt

        # Optional Telegram alerts (if env vars configured)
        self._telegram: TelegramBot | None = None
        try:
            from vibe_quant.alerts.telegram import TelegramBot as _TBot

            self._telegram = _TBot.from_env()
        except Exception:
            pass  # Telegram not configured, alerts disabled

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

        # Send Telegram alert for halt events
        self._send_telegram_alert(
            "circuit_breaker",
            f"HALT: {reason}\n{message}",
        )

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

        # Send Telegram alert
        self._send_telegram_alert(
            "error",
            f"{alert_type}: {context.error}\nOp: {context.operation}",
        )

    def _send_telegram_alert(self, kind: str, message: str) -> None:
        """Send alert via Telegram if configured. Fire-and-forget."""
        if self._telegram is None:
            return
        try:
            loop = asyncio.get_running_loop()
            if kind == "circuit_breaker":
                loop.create_task(self._telegram.send_circuit_breaker(message))
            else:
                loop.create_task(self._telegram.send_error(message))
        except RuntimeError:
            pass  # No running event loop
        except Exception:
            pass  # Never let telegram failure affect trading

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

    def _compile_strategy(self, dsl: StrategyDSL) -> ModuleType:
        """Compile DSL to a live-loadable Python module.

        Returns the module containing the generated Strategy/Config classes;
        instance construction happens in :meth:`_instantiate_strategies` so
        each symbol gets its own strategy registered with the TradingNode.
        """
        try:
            return self._compiler.compile_to_module(dsl)
        except Exception as e:
            raise ConfigurationError(f"Strategy compilation failed: {e}") from e

    def _instantiate_strategies(
        self, module: ModuleType, dsl: StrategyDSL
    ) -> list[object]:
        """Build one Strategy instance per configured symbol.

        Nautilus requires each Strategy in a TradingNode to have a unique
        ``order_id_tag``; we derive it from the symbol index so two symbols
        running the same generated class don't collide.
        """
        class_name = "".join(word.capitalize() for word in dsl.name.split("_"))
        strategy_cls = getattr(module, f"{class_name}Strategy", None)
        config_cls = getattr(module, f"{class_name}Config", None)
        if strategy_cls is None or config_cls is None:
            raise ConfigurationError(
                f"Compiled module missing {class_name}Strategy/{class_name}Config"
            )

        strategies: list[object] = []
        for idx, symbol in enumerate(self._config.symbols):
            instrument_id = f"{symbol}-PERP.BINANCE"
            cfg = config_cls(instrument_id=instrument_id, order_id_tag=f"{idx:03d}")
            strategies.append(strategy_cls(config=cfg))
        return strategies

    def _create_live_trading_node(self) -> _TradingNodeLifecycle:
        """Create and build a NautilusTrader TradingNode instance.

        Raises:
            ConfigurationError: If NautilusTrader or Binance adapter
                dependencies are not installed or incompatible.
        """
        try:
            from nautilus_trader.adapters.binance import config as binance_config
            from nautilus_trader.adapters.binance.config import (
                BinanceDataClientConfig,
                BinanceExecClientConfig,
            )
            from nautilus_trader.adapters.binance.factories import (
                BinanceLiveDataClientFactory,
                BinanceLiveExecClientFactory,
            )
            from nautilus_trader.config import InstrumentProviderConfig
            from nautilus_trader.live.config import TradingNodeConfig
            from nautilus_trader.live.node import TradingNode
            from nautilus_trader.model.identifiers import InstrumentId
        except ImportError as e:
            raise ConfigurationError(
                f"NautilusTrader live trading dependencies not installed: {e}. "
                "Install nautilus_trader with Binance adapter support."
            ) from e

        account_type_cls = getattr(binance_config, "BinanceAccountType", None)
        if account_type_cls is None:
            raise ConfigurationError(
                "BinanceAccountType is unavailable in NautilusTrader config module"
            )

        try:
            account_type = account_type_cls(self._config.binance.account_type)
        except ValueError as e:
            raise ConfigurationError(
                f"Unsupported Binance account_type '{self._config.binance.account_type}'",
            ) from e

        # Without `instrument_provider` the Binance adapter logs
        # "No loading configured" and the ExecEngine never reaches a
        # connected state. Scope loading to the configured symbols so startup
        # stays fast on USDT_FUTURES (~400 instruments otherwise).
        load_ids = frozenset(
            InstrumentId.from_str(f"{symbol}-PERP.BINANCE")
            for symbol in self._config.symbols
        )
        instrument_provider = InstrumentProviderConfig(load_ids=load_ids)

        node_config = TradingNodeConfig(
            trader_id=self._config.trader_id,
            data_clients={
                "BINANCE": BinanceDataClientConfig(
                    api_key=self._config.binance.api_key,
                    api_secret=self._config.binance.api_secret,
                    account_type=account_type,
                    testnet=self._config.binance.testnet,
                    instrument_provider=instrument_provider,
                ),
            },
            exec_clients={
                "BINANCE": BinanceExecClientConfig(
                    api_key=self._config.binance.api_key,
                    api_secret=self._config.binance.api_secret,
                    account_type=account_type,
                    testnet=self._config.binance.testnet,
                    instrument_provider=instrument_provider,
                ),
            },
        )
        node = TradingNode(node_config)
        # Strategies MUST be registered on the trader before ``node.build()``
        # or the live engine starts with zero strategies and exits early.
        for strategy in self._compiled_strategies:
            node.trader.add_strategy(strategy)
        node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
        node.add_exec_client_factory("BINANCE", BinanceLiveExecClientFactory)
        node.build()
        return node

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for halt/resume/close-all.

        SIGINT/SIGTERM are deliberately NOT registered here — Nautilus
        installs its own graceful-shutdown handlers on the running loop
        when ``run_async`` starts, and a double handler causes the
        "Event loop stopped before Future completed" crash during teardown.

        SIGUSR1 → halt/pause trading
        SIGUSR2 → resume trading
        SIGWINCH → close all positions (non-destructive keep-running)
        """
        loop = asyncio.get_running_loop()

        def halt_handler(_sig: int) -> None:
            if self._status.state == NodeState.RUNNING:
                self.pause()
            elif self._status.state == NodeState.PAUSED:
                self.halt(HaltReason.MANUAL, "Manual halt via SIGUSR1")

        def resume_handler(_sig: int) -> None:
            if self._status.state == NodeState.PAUSED:
                self.resume()
            elif self._status.state == NodeState.HALTED:
                self.resume_from_halt()

        def close_all_handler(_sig: int) -> None:
            if self._status.state == NodeState.RUNNING:
                self._close_all_positions()

        loop.add_signal_handler(signal.SIGUSR1, halt_handler, signal.SIGUSR1)
        loop.add_signal_handler(signal.SIGUSR2, resume_handler, signal.SIGUSR2)
        loop.add_signal_handler(signal.SIGWINCH, close_all_handler, signal.SIGWINCH)

    def _capture_checkpoint(self) -> StateCheckpoint:
        """Capture current node state for persistence.

        Reads open positions/orders and per-account balances off the
        TradingNode cache so a crash-restart can reconcile against the
        live venue. Before ``node.build()`` and after teardown the cache
        is unavailable — in those cases we emit an empty snapshot rather
        than propagate an exception that would kill the periodic
        checkpoint loop.
        """
        positions: dict[str, object] = {}
        orders: dict[str, object] = {}
        balance: dict[str, object] = {}

        cache = getattr(self._trading_node, "cache", None)
        if cache is not None:
            with contextlib.suppress(Exception):
                for pos in cache.positions_open():
                    positions[pos.id.value] = pos.to_dict()
            with contextlib.suppress(Exception):
                for order in cache.orders_open():
                    orders[order.client_order_id.value] = order.to_dict()
            with contextlib.suppress(Exception):
                for account in cache.accounts():
                    balance[account.id.value] = self._serialize_account_balance(account)

        return StateCheckpoint(
            trader_id=self._config.trader_id,
            positions=positions,
            orders=orders,
            balance=balance,
            node_status=self._status.to_dict(),
        )

    @staticmethod
    def _serialize_account_balance(account: object) -> dict[str, dict[str, str]]:
        """Serialize AccountBalance data to a JSON-friendly shape.

        Returns ``{"total": {CCY: amount_str}, "free": {...}, "locked": {...}}``.
        Decimal amounts are stringified so ``json.dumps`` in
        :meth:`StatePersistence.save_checkpoint` doesn't choke.
        """

        def _as_currency_amounts(balances: object) -> dict[str, str]:
            out: dict[str, str] = {}
            if balances is None:
                return out
            # ``balances_total/free/locked`` return dict[Currency, Money].
            items = balances.items() if hasattr(balances, "items") else []
            for currency, money in items:
                code = getattr(currency, "code", str(currency))
                amount = money.as_decimal() if hasattr(money, "as_decimal") else money
                out[code] = str(amount)
            return out

        return {
            "total": _as_currency_amounts(getattr(account, "balances_total", lambda: None)()),
            "free": _as_currency_amounts(getattr(account, "balances_free", lambda: None)()),
            "locked": _as_currency_amounts(getattr(account, "balances_locked", lambda: None)()),
        }

    def _write_event(self, event_type: EventType, data: dict[str, object]) -> None:
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

    def _cancel_open_orders_for_halt(self) -> int:
        """Best-effort cancellation of all open orders before halt shutdown."""
        if self._trading_node is None:
            return 0

        trader = getattr(self._trading_node, "trader", None)
        cache = getattr(self._trading_node, "cache", None)
        if trader is None or cache is None:
            return 0

        try:
            strategies = list(cast("_TraderRuntime", trader).strategies())
        except Exception:
            return 0

        cancelled_orders = 0
        cache_runtime = cast("_CacheRuntime", cache)
        for strategy in strategies:
            strategy_id = getattr(strategy, "id", None)
            if strategy_id is None:
                continue

            try:
                open_orders = cache_runtime.orders_open(strategy_id=strategy_id)
            except Exception:
                continue

            for order in open_orders:
                try:
                    strategy.cancel_order(order)
                    cancelled_orders += 1
                except Exception as exc:
                    self._write_event(
                        EventType.RISK_CHECK,
                        {
                            "action": "halt_cancel_order_failed",
                            "strategy_id": str(strategy_id),
                            "error": str(exc),
                        },
                    )

        return cancelled_orders

    def _close_all_positions(self) -> None:
        """Close all open positions via strategy close_all_positions calls."""
        if self._trading_node is None:
            return
        trader = getattr(self._trading_node, "trader", None)
        if trader is None:
            return
        try:
            for strategy in cast("_TraderRuntime", trader).strategies():
                close_fn = getattr(strategy, "close_all_positions", None)
                if close_fn is not None:
                    close_fn()
        except Exception as exc:
            self._write_event(
                EventType.RISK_CHECK,
                {"action": "close_all_positions_failed", "error": str(exc)},
            )
        self._write_event(
            EventType.SIGNAL,
            {"action": "close_all_positions", "trader_id": self._config.trader_id},
        )

    async def _initialize(self) -> None:
        """Initialize node components.

        Raises:
            ConfigurationError: If initialization fails.
        """
        self._status.state = NodeState.INITIALIZING
        self._status.updated_at = datetime.now(UTC)

        # Load, compile, and instantiate strategy (one per symbol)
        self._strategy = self._load_strategy()
        self._compiled_module = self._compile_strategy(self._strategy)
        self._compiled_strategies = self._instantiate_strategies(
            self._compiled_module, self._strategy
        )

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
        create_trading_node_config(
            self._config
        )  # Keep compatibility with existing config path/tests.
        self._trading_node = self._create_live_trading_node()

        # Start persistence lifecycle and save initial checkpoint.
        self._persistence = StatePersistence(
            db_path=self._config.db_path,
            trader_id=self._config.trader_id,
            checkpoint_interval=self._config.state_persistence_interval,
        )
        self._persistence.save_checkpoint(self._capture_checkpoint())

    async def _run_loop(self) -> None:
        """Main event loop for paper trading.

        Delegates execution lifecycle to NautilusTrader TradingNode.
        """
        if self._trading_node is None:
            raise RuntimeError("Trading node is not initialized")

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

        if self._persistence is not None:
            await self._persistence.start_periodic_checkpointing(self._capture_checkpoint)

        # Run Nautilus on our event loop (run_async) instead of in a worker
        # thread — wrapping the blocking ``run()`` in ``asyncio.to_thread``
        # produces "Event loop stopped before Future completed" on shutdown
        # because Nautilus spawns its own loop inside the thread.
        run_task = asyncio.create_task(self._trading_node.run_async())
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())

        done, pending = await asyncio.wait(
            {run_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        run_exc: BaseException | None = None
        try:
            if shutdown_task in done and not run_task.done():
                await self._trading_node.stop_async()
                await run_task
            elif run_task in done:
                self._shutdown_event.set()
                run_exc = run_task.exception()
        finally:
            # Always clean up pending tasks to prevent leaks
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        if run_exc is not None:
            raise run_exc

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

        if self._persistence is not None:
            with contextlib.suppress(Exception):
                self._persistence.save_checkpoint(self._capture_checkpoint())
            with contextlib.suppress(Exception):
                await self._persistence.stop_periodic_checkpointing()
            with contextlib.suppress(Exception):
                self._persistence.close()
            self._persistence = None

        # Stop trading node. ``TradingNode.dispose()`` stops the running
        # event loop (it was designed for ``run()`` which owns its own loop),
        # so we intentionally skip it — the outer ``asyncio.run`` in the CLI
        # tears the loop down cleanly on exit.
        if self._trading_node is not None:
            with contextlib.suppress(Exception):
                await self._trading_node.stop_async()
            self._trading_node = None

        # Flush and close event writer
        if self._event_writer is not None:
            with contextlib.suppress(Exception):
                self._event_writer.flush()
            self._event_writer.close()
            self._event_writer = None

        # Close Telegram client
        if self._telegram is not None:
            with contextlib.suppress(Exception):
                await self._telegram.close()

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

        Notes:
            Halt is recoverable by design and does not trigger process shutdown.
        """
        self._status.state = NodeState.HALTED
        self._status.halt_reason = reason
        self._status.error_message = message
        self._status.updated_at = datetime.now(UTC)
        cancelled_open_orders = self._cancel_open_orders_for_halt()

        self._write_event(
            EventType.RISK_CHECK,
            {
                "action": "halt",
                "reason": reason.value,
                "message": message,
                "cancelled_open_orders": cancelled_open_orders,
            },
        )

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
