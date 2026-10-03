"""Paper trading node implementation.

Provides PaperTradingNode class that wraps NautilusTrader TradingNode
for paper trading with Binance testnet (live only with explicit opt-in).

Safety model (see :mod:`vibe_quant.paper.guard`):

* a :class:`~vibe_quant.paper.guard.TradingGuard` actor gates every order and
  enforces max drawdown / daily loss / consecutive losses / position count;
* halt and kill flatten positions with reduce-only market orders, cancel orders
  and stop strategies; pause only blocks new entries (SL/TP stay live);
* operator commands arrive through the ``paper_commands`` table
  (:class:`~vibe_quant.paper.persistence.PaperCommandQueue`), not POSIX signals,
  and the system kill switch in the state DB is polled continuously;
* the node trades the exact strategy parameters + leverage of the validation run
  it was started from and logs any difference.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.logging.events import Event, EventType, create_event
from vibe_quant.logging.writer import EventWriter
from vibe_quant.paper.config import (
    DEFAULT_VALIDATION_LEVERAGE,
    ConfigurationError,
    PaperTradingConfig,
)
from vibe_quant.paper.errors import ErrorContext, ErrorHandler
from vibe_quant.paper.guard import (
    CloseAllResult,
    GuardListener,
    GuardState,
    HaltReason,
    RiskState,
    TradingGuard,
    TradingGuardConfig,
)
from vibe_quant.paper.persistence import (
    PaperCommand,
    PaperCommandQueue,
    StateCheckpoint,
    StatePersistence,
)

if TYPE_CHECKING:
    from types import ModuleType

    from vibe_quant.alerts.telegram import TelegramBot
    from vibe_quant.dsl.schema import StrategyDSL

logger = logging.getLogger(__name__)

# ValidationRunner._build_strategy_params skips these meta keys.
_RUN_META_KEYS = frozenset({"sweep", "overfitting_filters"})
# Backtest-only simulation knobs that must never reach a live strategy.
_BACKTEST_ONLY_PARAMS = frozenset({"execution_delay_probability", "execution_delay_seed"})


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


class NodeState(StrEnum):
    """State of the paper trading node."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    HALTED = "halted"
    STOPPED = "stopped"
    ERROR = "error"


_GUARD_TO_NODE_STATE = {
    GuardState.RUNNING: NodeState.RUNNING,
    GuardState.PAUSED: NodeState.PAUSED,
    GuardState.HALTED: NodeState.HALTED,
}


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


@dataclass
class ResolvedRunConfig:
    """What the node actually trades, plus the validated reference it came from."""

    symbols: list[str]
    strategy_params: dict[str, object]
    leverage: int
    margin_type: str
    validated: dict[str, object] | None = None
    diff: list[dict[str, object]] = field(default_factory=list)


class OrderRejectedError(Exception):
    """Venue rejected an order (routed through ErrorHandler for halt + alert)."""


class OrderDeniedError(Exception):
    """NT RiskEngine denied an order."""


def _ns_to_dt(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC)


def validated_run_strategy_params(run: dict[str, Any]) -> dict[str, object]:
    """Strategy overrides exactly as ``ValidationRunner`` builds them for this run."""
    from vibe_quant.validation.runner import ValidationRunner

    # _build_strategy_params only reads run_config; reuse it so paper can never
    # drift from what validation applied. (No ValidationRunner state needed.)
    runner = ValidationRunner.__new__(ValidationRunner)
    return runner._build_strategy_params(run)


def validated_run_leverage(run: dict[str, Any]) -> float:
    """Venue leverage used by ``ValidationRunner._create_venue_config`` for this run."""
    params = run.get("parameters")
    raw = params.get("leverage") if isinstance(params, dict) else None
    if not isinstance(raw, bool) and isinstance(raw, (int, float)) and raw > 0:
        return float(raw)
    return float(DEFAULT_VALIDATION_LEVERAGE)


class _GuardBridge(GuardListener):
    """Forwards guard callbacks to the node."""

    def __init__(self, node: PaperTradingNode) -> None:
        self._node = node

    def on_state_change(
        self,
        state: GuardState,
        reason: HaltReason | None,
        message: str,
        metrics: dict[str, object],
    ) -> None:
        self._node._on_guard_state_change(state, reason, message, metrics)

    def on_order_rejected(self, event: Any) -> None:
        self._node._on_order_rejected(event)

    def on_order_denied(self, event: Any, *, by_gate: bool) -> None:
        self._node._on_order_denied(event, by_gate=by_gate)

    def on_position_opened(self, event: Any) -> None:
        self._node._on_position_opened(event)

    def on_position_closed(self, event: Any) -> None:
        self._node._on_position_closed(event)

    def on_warning(self, message: str) -> None:
        self._node._on_guard_warning(message)


class PaperTradingNode:
    """Paper trading node for live simulated execution.

    Wraps NautilusTrader TradingNode for paper trading with Binance testnet.
    Provides:
    - Strategy deployment from DSL with the validated run's exact parameters
    - Risk enforcement + halt/pause/close-all via TradingGuard
    - State persistence for crash recovery (risk state restored on restart)
    - Command-queue control, kill-switch polling, Telegram alerts

    Example:
        config = PaperTradingConfig(trader_id="PAPER-001", binance=..., validation_run_id=870,
                                    strategy_id=240)
        node = PaperTradingNode(config)
        await node.start()  # Runs until SIGINT/SIGTERM
    """

    def __init__(self, config: PaperTradingConfig) -> None:
        """Initialize paper trading node.

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
        self._commands: PaperCommandQueue | None = None
        self._guard: TradingGuard | None = None
        self._restored_risk: RiskState | None = None
        self._resolved: ResolvedRunConfig | None = None
        self._shutdown_event = asyncio.Event()
        self._strategy: StrategyDSL | None = None
        self._compiled_module: ModuleType | None = None
        self._compiled_strategies: list[object] = []
        self._error_handler = ErrorHandler(
            on_halt=self._on_error_halt,
            on_alert=self._on_error_alert,
        )
        self._alert_tasks: set[asyncio.Task[bool]] = set()
        self._control_task: asyncio.Task[None] | None = None
        self._connected: bool | None = None
        self._kill_reason_seen: str | None = None

        # Optional Telegram alerts (if env vars configured)
        self._telegram: TelegramBot | None = None
        try:
            from vibe_quant.alerts.telegram import TelegramBot as _TBot

            self._telegram = _TBot.from_env()
        except Exception:
            logger.info("Telegram alerts disabled (TELEGRAM_BOT_TOKEN/CHAT_ID not set)")

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

    @property
    def guard(self) -> TradingGuard | None:
        """The trading guard (None before the trading node is built)."""
        return self._guard

    @property
    def resolved_config(self) -> ResolvedRunConfig | None:
        """Effective symbols/params/leverage (after ``_initialize``)."""
        return self._resolved

    # ------------------------------------------------------------- alerts

    def _on_error_halt(self, reason: str, message: str) -> None:
        """ErrorHandler decided an error is fatal for trading."""
        self.halt(HaltReason.ERROR, f"{reason}: {message}")

    def _on_error_alert(self, alert_type: str, context: ErrorContext) -> None:
        """ErrorHandler wants the operator to know about an error."""
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
                "passed": False,
            },
        )
        self._send_telegram_alert(
            "error",
            f"{alert_type}: {context.error}\nOp: {context.operation}"
            + (f"\nSymbol: {context.symbol}" if context.symbol else ""),
        )

    def _send_telegram_alert(self, kind: str, message: str, *, critical: bool = False) -> None:
        """Send alert via Telegram if configured. Fire-and-forget, never raises.

        ``critical`` alerts (halts, kill switch, risk breaches) bypass the
        per-type rate limit so they are never silently dropped.
        """
        if self._telegram is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("telegram alert not sent (no running event loop): %s", kind)
            return
        tg = self._telegram
        if kind == "circuit_breaker":
            coro = tg.send_circuit_breaker(message, bypass_rate_limit=critical)
        else:
            coro = tg.send_error(message, bypass_rate_limit=critical)
        task = loop.create_task(coro)
        self._alert_tasks.add(task)
        task.add_done_callback(self._alert_done)

    def _alert_done(self, task: asyncio.Task[bool]) -> None:
        self._alert_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            # TelegramBot already redacts its own failures; never feed alert
            # failures back into ErrorHandler (that would alert about alerts).
            logger.warning("telegram alert task failed: %s", type(exc).__name__)

    # ------------------------------------------------------ strategy setup

    def _load_strategy(self) -> StrategyDSL:
        """Load and validate strategy from database.

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

        try:
            dsl = validate_strategy_dict(dsl_config)
        except Exception as e:
            raise ConfigurationError(f"Invalid strategy DSL: {e}") from e

        return dsl

    def _compile_strategy(self, dsl: StrategyDSL) -> ModuleType:
        """Compile DSL to a live-loadable Python module."""
        try:
            return self._compiler.compile_to_module(dsl)
        except Exception as e:
            raise ConfigurationError(f"Strategy compilation failed: {e}") from e

    @staticmethod
    def _class_names(dsl: StrategyDSL) -> tuple[str, str]:
        class_name = "".join(word.capitalize() for word in dsl.name.split("_"))
        return f"{class_name}Strategy", f"{class_name}Config"

    def _resolve_run_config(self, module: ModuleType, dsl: StrategyDSL) -> ResolvedRunConfig:
        """Derive symbols/params/leverage; compare against the validated run.

        Raises:
            ConfigurationError: If the validation run is missing/incompatible.
        """
        from nautilus_trader.trading.config import StrategyConfig

        cfg = self._config
        _, config_cls_name = self._class_names(dsl)
        config_cls = getattr(module, config_cls_name, None)
        # Only fields the generated config declares itself; NT base fields
        # (strategy_id, order_id_tag, manage_stop, ...) are node-controlled.
        base_fields = set(StrategyConfig.__struct_fields__)
        fields: tuple[str, ...] = tuple(
            f for f in getattr(config_cls, "__struct_fields__", ()) if f not in base_fields
        )

        validated: dict[str, object] | None = None
        base_params: dict[str, object] = {}
        symbols = list(cfg.symbols)
        leverage_f = float(DEFAULT_VALIDATION_LEVERAGE)

        if cfg.validation_run_id is not None:
            run = self._state_manager.get_backtest_run(cfg.validation_run_id)
            if run is None:
                raise ConfigurationError(f"Validation run {cfg.validation_run_id} not found")
            if run.get("run_mode") != "validation":
                raise ConfigurationError(
                    f"Run {cfg.validation_run_id} is a {run.get('run_mode')!r} run, "
                    "not a validation run"
                )
            if run.get("status") != "completed":
                raise ConfigurationError(
                    f"Validation run {cfg.validation_run_id} is {run.get('status')!r}, "
                    "not completed"
                )
            if run.get("strategy_id") != cfg.strategy_id:
                raise ConfigurationError(
                    f"Validation run {cfg.validation_run_id} is for strategy "
                    f"{run.get('strategy_id')}, not {cfg.strategy_id}"
                )
            run_symbols = [str(s) for s in run.get("symbols") or []]
            if not symbols:
                symbols = run_symbols
            raw = validated_run_strategy_params(run)
            base_params = {
                k: v for k, v in raw.items() if k in fields and k not in _BACKTEST_ONLY_PARAMS
            }
            dropped = sorted(set(raw) - set(base_params))
            if dropped:
                logger.info(
                    "validation run %d params not applied to the live strategy "
                    "(venue-level or backtest-only): %s",
                    cfg.validation_run_id,
                    dropped,
                )
            leverage_f = validated_run_leverage(run)
            validated = {
                "validation_run_id": cfg.validation_run_id,
                "symbols": run_symbols,
                "strategy_params": dict(base_params),
                "leverage": leverage_f,
                "margin_type": "CROSSED",
            }

        if not symbols:
            raise ConfigurationError("No symbols to trade")

        params = dict(base_params)
        if cfg.sizing.risk_per_trade is not None:
            params["risk_per_trade"] = float(cfg.sizing.risk_per_trade)
        if cfg.sizing.max_position_pct is not None:
            params["max_position_pct"] = float(cfg.sizing.max_position_pct)
        unknown = sorted(k for k in params if fields and k not in fields)
        if unknown:
            raise ConfigurationError(f"Strategy config has no fields {unknown}")

        if cfg.leverage is not None:
            leverage_f = float(cfg.leverage)
        if leverage_f != int(leverage_f):
            raise ConfigurationError(
                f"Binance futures leverage must be an integer, validation used {leverage_f}"
            )
        leverage = int(leverage_f)
        if leverage > cfg.sizing.max_leverage:
            raise ConfigurationError(
                f"leverage {leverage} exceeds max_leverage {cfg.sizing.max_leverage}"
            )

        resolved = ResolvedRunConfig(
            symbols=symbols,
            strategy_params=params,
            leverage=leverage,
            margin_type=cfg.margin_type,
            validated=validated,
        )
        if validated is not None:
            resolved.diff = self._config_diff(validated, resolved)
        return resolved

    @staticmethod
    def _config_diff(
        validated: dict[str, object], resolved: ResolvedRunConfig
    ) -> list[dict[str, object]]:
        diff: list[dict[str, object]] = []
        v_params = validated.get("strategy_params")
        v_params = v_params if isinstance(v_params, dict) else {}
        for key in sorted(set(v_params) | set(resolved.strategy_params)):
            v, p = v_params.get(key), resolved.strategy_params.get(key)
            if v != p:
                diff.append({"field": f"strategy_params.{key}", "validated": v, "paper": p})
        v_lev = validated.get("leverage")
        if v_lev != float(resolved.leverage):
            diff.append({"field": "leverage", "validated": v_lev, "paper": resolved.leverage})
        if validated.get("margin_type") != resolved.margin_type:
            diff.append(
                {
                    "field": "margin_type",
                    "validated": validated.get("margin_type"),
                    "paper": resolved.margin_type,
                }
            )
        if validated.get("symbols") != resolved.symbols:
            diff.append(
                {"field": "symbols", "validated": validated.get("symbols"), "paper": resolved.symbols}
            )
        return diff

    def _instantiate_strategies(self, module: ModuleType, dsl: StrategyDSL) -> list[object]:
        """Build one Strategy instance per symbol with the resolved parameters.

        Configs are decoded with ``StrategyConfig.parse`` (msgspec) exactly like
        ``ImportableStrategyConfig`` does in validation, so parameter types match.
        Each strategy claims its instrument's external orders/positions so a
        restart adopts the venue position instead of opening a second one.
        """
        strategy_cls_name, config_cls_name = self._class_names(dsl)
        strategy_cls = getattr(module, strategy_cls_name, None)
        config_cls = getattr(module, config_cls_name, None)
        if strategy_cls is None or config_cls is None:
            raise ConfigurationError(
                f"Compiled module missing {strategy_cls_name}/{config_cls_name}"
            )
        resolved = self._resolved
        if resolved is None:
            resolved = self._resolve_run_config(module, dsl)
            self._resolved = resolved

        strategies: list[object] = []
        for idx, symbol in enumerate(resolved.symbols):
            instrument_id = f"{symbol}-PERP.BINANCE"
            raw_cfg = {
                **resolved.strategy_params,
                "instrument_id": instrument_id,
                "order_id_tag": f"{idx:03d}",
                "external_order_claims": [instrument_id],
            }
            try:
                cfg = config_cls.parse(json.dumps(raw_cfg))
            except Exception as e:
                raise ConfigurationError(f"Invalid strategy config {raw_cfg}: {e}") from e
            strategies.append(strategy_cls(config=cfg))
        return strategies

    def _create_live_trading_node(self) -> _TradingNodeLifecycle:
        """Create and build a NautilusTrader TradingNode instance (+ guard actor).

        Raises:
            ConfigurationError: If NautilusTrader or Binance adapter
                dependencies are not installed or incompatible.
        """
        try:
            from nautilus_trader.adapters.binance import config as binance_config
            from nautilus_trader.adapters.binance.common.enums import BinanceEnvironment
            from nautilus_trader.adapters.binance.common.symbol import BinanceSymbol
            from nautilus_trader.adapters.binance.config import (
                BinanceDataClientConfig,
                BinanceExecClientConfig,
            )
            from nautilus_trader.adapters.binance.factories import (
                BinanceLiveDataClientFactory,
                BinanceLiveExecClientFactory,
            )
            from nautilus_trader.adapters.binance.futures.enums import (
                BinanceFuturesMarginType,
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

        resolved = self._resolved
        if resolved is None:
            raise ConfigurationError("run config not resolved before building the node")

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
        # connected state. Scope loading to the configured symbols.
        load_ids = frozenset(
            InstrumentId.from_str(f"{symbol}-PERP.BINANCE") for symbol in resolved.symbols
        )
        instrument_provider = InstrumentProviderConfig(load_ids=load_ids)

        environment = (
            BinanceEnvironment.TESTNET if self._config.binance.testnet else BinanceEnvironment.LIVE
        )
        margin_type = (
            BinanceFuturesMarginType.CROSS
            if resolved.margin_type == "CROSSED"
            else BinanceFuturesMarginType.ISOLATED
        )

        node_config = TradingNodeConfig(
            trader_id=self._config.trader_id,
            data_clients={
                "BINANCE": BinanceDataClientConfig(
                    api_key=self._config.binance.api_key,
                    api_secret=self._config.binance.api_secret,
                    account_type=account_type,
                    environment=environment,
                    instrument_provider=instrument_provider,
                ),
            },
            exec_clients={
                "BINANCE": BinanceExecClientConfig(
                    api_key=self._config.binance.api_key,
                    api_secret=self._config.binance.api_secret,
                    account_type=account_type,
                    environment=environment,
                    instrument_provider=instrument_provider,
                    futures_leverages={
                        BinanceSymbol(s): resolved.leverage for s in resolved.symbols
                    },
                    futures_margin_types={BinanceSymbol(s): margin_type for s in resolved.symbols},
                ),
            },
        )
        node = TradingNode(node_config)
        self._guard = self._build_guard(node.kernel.risk_engine)
        # Actors start before strategies, so the gate is installed before any
        # strategy can submit an order.
        node.trader.add_actor(self._guard)
        # Strategies MUST be registered on the trader before ``node.build()``
        # or the live engine starts with zero strategies and exits early.
        for strategy in self._compiled_strategies:
            node.trader.add_strategy(strategy)
        node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
        node.add_exec_client_factory("BINANCE", BinanceLiveExecClientFactory)
        node.build()
        return node

    def _build_guard(self, risk_engine: Any) -> TradingGuard:
        risk = self._config.risk
        return TradingGuard(
            TradingGuardConfig(
                max_drawdown_pct=risk.max_drawdown_pct,
                max_daily_loss_pct=risk.max_daily_loss_pct,
                max_consecutive_losses=risk.max_consecutive_losses,
                max_position_count=risk.max_position_count,
            ),
            strategies=self._compiled_strategies,
            risk_engine=risk_engine,
            listener=_GuardBridge(self),
            risk_state=self._restored_risk,
        )

    # ------------------------------------------------------------ control

    def _setup_signal_handlers(self) -> None:
        """POSIX fallback controls for CLI use (the API uses the command queue).

        SIGINT/SIGTERM are deliberately NOT registered here — Nautilus
        installs its own graceful-shutdown handlers on the running loop.

        SIGUSR1 → halt (flatten, cancel, stop strategies) — one signal, full halt
        SIGUSR2 → resume (refused while the kill switch is engaged)
        """
        loop = asyncio.get_running_loop()

        def halt_handler(_sig: int) -> None:
            self.halt(HaltReason.SIGNAL, "halt via SIGUSR1")

        def resume_handler(_sig: int) -> None:
            ok, msg = self.resume()
            if not ok:
                logger.warning("resume via SIGUSR2 refused: %s", msg)

        loop.add_signal_handler(signal.SIGUSR1, halt_handler, signal.SIGUSR1)
        loop.add_signal_handler(signal.SIGUSR2, resume_handler, signal.SIGUSR2)

    def _kill_switch(self) -> tuple[bool, str | None]:
        try:
            sys_state = self._state_manager.get_system_state()
        except Exception:
            logger.exception("could not read system kill switch")
            return False, None
        return bool(sys_state.get("kill_switch")), sys_state.get("reason")

    async def _control_loop(self) -> None:
        """Poll operator commands, the kill switch and connectivity."""
        interval = self._config.control_poll_interval
        while not self._shutdown_event.is_set():
            try:
                await self.process_commands()
                self._check_kill_switch()
                self._check_connectivity()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("paper control loop iteration failed")
            await asyncio.sleep(interval)

    def _check_kill_switch(self) -> None:
        engaged, reason = self._kill_switch()
        if not engaged:
            self._kill_reason_seen = None
            return
        guard = self._guard
        already = (
            guard is not None
            and guard.guard_state == GuardState.HALTED
            and guard.halt_reason == HaltReason.KILL_SWITCH
        )
        if not already:
            self.halt(HaltReason.KILL_SWITCH, f"system kill switch: {reason or 'no reason'}")
        self._kill_reason_seen = reason

    def _check_connectivity(self) -> None:
        kernel = getattr(self._trading_node, "kernel", None)
        if kernel is None:
            return
        try:
            connected = bool(
                kernel.data_engine.check_connected() and kernel.exec_engine.check_connected()
            )
        except Exception:
            logger.exception("connectivity check failed")
            return
        if self._connected is None:
            if connected:
                self._connected = True  # first time fully connected
            return
        if self._connected and not connected:
            self._connected = False
            self.handle_error(
                ConnectionError("venue connection lost (data or execution client disconnected)"),
                operation="connectivity",
            )
        elif not self._connected and connected:
            self._connected = True
            self._error_handler.reset_retry_count("connectivity", "")
            self._write_event(EventType.SIGNAL, {"action": "reconnected"})
            self._send_telegram_alert("error", "venue connection restored")

    async def process_commands(self) -> None:
        """Execute queued operator commands for this trader."""
        if self._commands is None:
            return
        for cmd in self._commands.claim_pending(self._config.trader_id):
            try:
                ok, result, error = await self._execute_command(cmd)
            except Exception as exc:
                logger.exception("paper command %s failed", cmd.command)
                self._commands.fail(cmd.id, f"{type(exc).__name__}: {exc}")
                continue
            if ok:
                self._commands.complete(cmd.id, result)
            else:
                self._commands.fail(cmd.id, error or "failed", result)

    async def _execute_command(self, cmd: PaperCommand) -> tuple[bool, dict[str, Any], str | None]:
        payload = cmd.payload or {}
        message = str(payload.get("message") or f"{cmd.command} via API")
        if cmd.command == "halt":
            self.halt(HaltReason.MANUAL, message)
            return True, self._state_payload(), None
        if cmd.command == "kill":
            self.halt(HaltReason.KILL_SWITCH, message)
            return True, self._state_payload(), None
        if cmd.command == "pause":
            self.pause()
            return True, self._state_payload(), None
        if cmd.command == "resume":
            ok, msg = self.resume()
            return ok, {**self._state_payload(), "message": msg}, None if ok else msg
        if cmd.command == "close_all":
            wait = float(payload.get("wait_secs", 10.0))
            result = self.close_all_positions()
            still_open = await self._wait_flat(wait)
            out: dict[str, Any] = {**result.to_dict(), "still_open": still_open}
            errors = list(result.errors)
            if still_open:
                errors.append(f"positions still open after {wait:.0f}s: {still_open}")
            if errors:
                return False, out, "; ".join(errors)
            return True, out, None
        return False, {}, f"unknown command {cmd.command!r}"

    def _state_payload(self) -> dict[str, Any]:
        return {
            "state": self._status.state.value,
            "halt_reason": self._status.halt_reason.value if self._status.halt_reason else None,
            "message": self._status.error_message,
        }

    def _open_position_ids(self) -> list[str]:
        cache = getattr(self._trading_node, "cache", None)
        if cache is None:
            return []
        return [str(p.id) for p in cache.positions_open()]

    async def _wait_flat(self, timeout: float) -> list[str]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            still = self._open_position_ids()
            if not still or loop.time() >= deadline:
                return still
            await asyncio.sleep(0.25)

    # ------------------------------------------------- guard callbacks

    def _on_guard_state_change(
        self,
        state: GuardState,
        reason: HaltReason | None,
        message: str,
        metrics: dict[str, object],
    ) -> None:
        prev = self._status.state
        self._status.state = _GUARD_TO_NODE_STATE[state]
        self._status.halt_reason = reason if state == GuardState.HALTED else None
        self._status.error_message = message
        self._status.updated_at = datetime.now(UTC)
        if state == GuardState.HALTED and reason is not None:
            self._write_event(
                EventType.RISK_CHECK,
                {
                    "action": "halt",
                    "check_type": reason.value,
                    "reason": reason.value,
                    "message": message,
                    "passed": False,
                    **metrics,
                },
            )
            self._send_telegram_alert(
                "circuit_breaker", f"HALT ({reason.value}): {message}", critical=True
            )
        else:
            self._write_event(
                EventType.SIGNAL,
                {
                    "action": "state_change",
                    "from_state": prev.value,
                    "to_state": self._status.state.value,
                    "message": message,
                },
            )
            if prev == NodeState.HALTED and state == GuardState.RUNNING:
                self._send_telegram_alert(
                    "circuit_breaker", f"RESUMED: {message}", critical=True
                )
        self._save_checkpoint_now()

    def _on_order_rejected(self, event: Any) -> None:
        self._write_event(
            EventType.ORDER,
            {
                "action": "rejected",
                "order_id": str(event.client_order_id),
                "symbol": str(event.instrument_id),
                "strategy_id": str(event.strategy_id),
                "reason": str(event.reason),
            },
            timestamp=_ns_to_dt(event.ts_event),
        )
        self.handle_error(
            OrderRejectedError(f"order rejected: {event.reason}"),
            operation=f"submit_order {event.client_order_id}",
            symbol=str(event.instrument_id),
        )

    def _on_order_denied(self, event: Any, *, by_gate: bool) -> None:
        self._write_event(
            EventType.ORDER,
            {
                "action": "denied",
                "by_guard": by_gate,
                "order_id": str(event.client_order_id),
                "symbol": str(event.instrument_id),
                "strategy_id": str(event.strategy_id),
                "reason": str(event.reason),
            },
            timestamp=_ns_to_dt(event.ts_init),
        )
        if not by_gate:
            self._send_telegram_alert(
                "error", f"Order denied by risk engine: {event.reason}\n{event.instrument_id}"
            )

    def _on_position_opened(self, event: Any) -> None:
        self._write_event(
            EventType.POSITION_OPEN,
            {
                "position_id": str(event.position_id),
                "symbol": str(event.instrument_id),
                "side": event.side.name,
                "quantity": float(event.quantity),
                "entry_price": float(event.avg_px_open),
                "leverage": self._resolved.leverage if self._resolved else 1,
            },
            timestamp=_ns_to_dt(event.ts_event),
        )

    def _on_position_closed(self, event: Any) -> None:
        net = float(event.realized_pnl) if event.realized_pnl is not None else 0.0
        commissions = 0.0
        exit_reason = "unknown"
        cache = self._guard.cache if self._guard is not None else None
        if cache is not None:
            pos = cache.position(event.position_id)
            if pos is not None:
                commissions = sum(
                    float(c) for c in pos.commissions() if c.currency == event.realized_pnl.currency
                )
            closing = cache.order(event.closing_order_id) if event.closing_order_id else None
            exit_reason = self._exit_reason(closing)
        self._write_event(
            EventType.POSITION_CLOSE,
            {
                "position_id": str(event.position_id),
                "symbol": str(event.instrument_id),
                "exit_price": float(event.avg_px_close),
                "gross_pnl": net + commissions,
                "net_pnl": net,
                "exit_reason": exit_reason,
            },
            timestamp=_ns_to_dt(event.ts_event),
        )

    @staticmethod
    def _exit_reason(order: Any) -> str:
        if order is None:
            return "unknown"
        tags = order.tags or []
        if "MARKET_EXIT" in tags:
            return "halt_or_close_all"
        type_name = order.order_type.name
        if order.is_reduce_only and type_name in ("STOP_MARKET", "STOP_LIMIT"):
            return "stop_loss"
        if order.is_reduce_only and type_name == "LIMIT":
            return "take_profit"
        return "signal"

    def _on_guard_warning(self, message: str) -> None:
        self._write_event(EventType.RISK_CHECK, {"action": "warning", "message": message})
        self._send_telegram_alert("error", f"WARNING: {message}")

    # ------------------------------------------------------ checkpoints

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
        portfolio = getattr(self._trading_node, "portfolio", None)
        if cache is not None:
            with contextlib.suppress(Exception):
                for pos in cache.positions_open():
                    data = pos.to_dict()
                    data["unrealized_pnl"] = self._unrealized_pnl(portfolio, pos)
                    data["leverage"] = self._resolved.leverage if self._resolved else None
                    positions[pos.id.value] = data
            with contextlib.suppress(Exception):
                for order in cache.orders_open():
                    orders[order.client_order_id.value] = order.to_dict()
            with contextlib.suppress(Exception):
                for account in cache.accounts():
                    balance[account.id.value] = self._serialize_account_balance(account)

        node_status: dict[str, object] = dict(self._status.to_dict())
        node_status["trader_id"] = self._config.trader_id
        node_status["testnet"] = self._config.binance.testnet
        if self._guard is not None:
            guard_status = self._guard.status()
            node_status["risk"] = guard_status["risk"]
            node_status["guard"] = guard_status
        if self._resolved is not None:
            node_status["config_diff"] = self._resolved.diff
            node_status["validation_run_id"] = self._config.validation_run_id
            node_status["leverage"] = self._resolved.leverage

        return StateCheckpoint(
            trader_id=self._config.trader_id,
            positions=positions,
            orders=orders,
            balance=balance,
            node_status=node_status,
        )

    @staticmethod
    def _unrealized_pnl(portfolio: Any, pos: Any) -> float | None:
        if portfolio is None:
            return None
        try:
            pnl = portfolio.unrealized_pnl(pos.instrument_id)
        except Exception:
            return None
        return float(pnl) if pnl is not None else None

    @staticmethod
    def _serialize_account_balance(account: object) -> dict[str, dict[str, str]]:
        """Serialize AccountBalance data to a JSON-friendly shape.

        Returns ``{"total": {CCY: amount_str}, "free": {...}, "locked": {...}}``.
        """

        def _as_currency_amounts(balances: object) -> dict[str, str]:
            out: dict[str, str] = {}
            if balances is None:
                return out
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

    def _save_checkpoint_now(self) -> None:
        if self._persistence is None:
            return
        try:
            self._persistence.save_checkpoint(self._capture_checkpoint())
        except Exception:
            logger.exception("immediate checkpoint failed")

    def _restore_from_checkpoint(self) -> None:
        """Load the latest checkpoint for this trader_id (restart / /restore).

        Restores the risk bookkeeping (high water mark, UTC-day baseline,
        consecutive losses) so a restart cannot reset the drawdown/daily-loss
        limits. Positions and orders come from NautilusTrader's venue
        reconciliation (claimed via ``external_order_claims``); the checkpoint
        copy is only logged for comparison.
        """
        if self._persistence is None:
            return
        prior = self._persistence.load_latest_checkpoint(self._config.trader_id)
        if prior is None:
            return
        risk_raw = prior.node_status.get("risk") if isinstance(prior.node_status, dict) else None
        self._restored_risk = RiskState.from_dict(risk_raw if isinstance(risk_raw, dict) else None)
        self._write_event(
            EventType.SIGNAL,
            {
                "action": "restore",
                "checkpoint_time": prior.timestamp.isoformat(),
                "prior_state": prior.node_status.get("state"),
                "prior_halt_reason": prior.node_status.get("halt_reason"),
                "prior_positions": sorted(prior.positions),
                "prior_orders": sorted(prior.orders),
                "restored_risk": self._restored_risk.to_dict(),
            },
        )
        logger.info(
            "restored risk state for %s from checkpoint %s: %s",
            self._config.trader_id,
            prior.timestamp.isoformat(),
            self._restored_risk.to_dict(),
        )

    # -------------------------------------------------------------- events

    def _write_event(
        self,
        event_type: EventType,
        data: dict[str, object],
        timestamp: datetime | None = None,
    ) -> None:
        """Write an event to the paper event log, preserving the full payload.

        Position events use the typed classes (their fields are the payload);
        everything else is written as a base ``Event`` because the typed
        SIGNAL/RISK_CHECK classes drop unknown keys (and default ``passed``
        to True, which would record a halt as a passed risk check).
        """
        if self._event_writer is None:
            return
        strategy_name = self._strategy.name if self._strategy else "unknown"
        ts = timestamp or datetime.now(UTC)
        event: Event
        if event_type in (EventType.POSITION_OPEN, EventType.POSITION_CLOSE):
            event = create_event(
                event_type=event_type,
                run_id=self._config.trader_id,
                strategy_name=strategy_name,
                data=data,
                timestamp=ts,
            )
        else:
            event = Event(
                timestamp=ts,
                event_type=event_type,
                run_id=self._config.trader_id,
                strategy_name=strategy_name,
                data=data,
            )
        self._event_writer.write(event)

    # ------------------------------------------------------------ lifecycle

    def _close_all_positions(self) -> CloseAllResult:
        """Backward-compatible alias for :meth:`close_all_positions`."""
        return self.close_all_positions()

    def close_all_positions(self) -> CloseAllResult:
        """Flatten every position (reduce-only) and cancel orders; keep running."""
        if self._guard is None:
            result = CloseAllResult(errors=["trading node not running"])
        else:
            result = self._guard.close_all()
        self._write_event(
            EventType.SIGNAL,
            {"action": "close_all_positions", **result.to_dict()},
        )
        if result.errors:
            self._send_telegram_alert(
                "error", "Close-all problems: " + "; ".join(result.errors), critical=True
            )
        return result

    async def _initialize(self) -> None:
        """Initialize node components.

        Raises:
            ConfigurationError: If initialization fails.
        """
        self._status.state = NodeState.INITIALIZING
        self._status.updated_at = datetime.now(UTC)

        engaged, reason = self._kill_switch()
        if engaged:
            raise ConfigurationError(
                f"System kill switch engaged ({reason or 'no reason'}); refusing to start"
            )

        # Create event writer
        self._config.logs_path.mkdir(parents=True, exist_ok=True)
        self._event_writer = EventWriter(
            run_id=self._config.trader_id,
            base_path=self._config.logs_path,
        )

        # Load, compile, resolve params, instantiate strategy (one per symbol)
        self._strategy = self._load_strategy()
        self._compiled_module = self._compile_strategy(self._strategy)
        self._resolved = self._resolve_run_config(self._compiled_module, self._strategy)
        self._compiled_strategies = self._instantiate_strategies(
            self._compiled_module, self._strategy
        )

        resolved = self._resolved
        self._write_event(
            EventType.SIGNAL,
            {
                "action": "start",
                "trader_id": self._config.trader_id,
                "strategy_id": self._config.strategy_id,
                "validation_run_id": self._config.validation_run_id,
                "symbols": resolved.symbols,
                "testnet": self._config.binance.testnet,
                "leverage": resolved.leverage,
                "margin_type": resolved.margin_type,
                "strategy_params": resolved.strategy_params,
                "config_diff": resolved.diff,
            },
        )
        if resolved.validated is None:
            logger.warning(
                "paper %s not started from a validation run: trading compiled DSL "
                "defaults, nothing to compare against",
                self._config.trader_id,
            )
        elif resolved.diff:
            logger.warning(
                "paper %s differs from validation run %s: %s",
                self._config.trader_id,
                self._config.validation_run_id,
                resolved.diff,
            )
        else:
            logger.info(
                "paper %s config matches validation run %s exactly (diff empty)",
                self._config.trader_id,
                self._config.validation_run_id,
            )

        # Persistence first: restore risk state before the guard is built.
        self._persistence = StatePersistence(
            db_path=self._config.db_path,
            trader_id=self._config.trader_id,
            checkpoint_interval=self._config.state_persistence_interval,
        )
        self._restore_from_checkpoint()
        self._commands = PaperCommandQueue(self._config.db_path)

        self._trading_node = self._create_live_trading_node()
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

        self._control_task = asyncio.create_task(self._control_loop())

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
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await self._stop_control_loop()

        if run_exc is not None:
            raise run_exc

    async def _stop_control_loop(self) -> None:
        task = self._control_task
        self._control_task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _shutdown(self) -> None:
        """Graceful shutdown of node components."""
        if self._status.state != NodeState.ERROR:
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

        await self._stop_control_loop()

        if self._persistence is not None:
            with contextlib.suppress(Exception):
                self._persistence.save_checkpoint(self._capture_checkpoint())
            with contextlib.suppress(Exception):
                await self._persistence.stop_periodic_checkpointing()
            with contextlib.suppress(Exception):
                self._persistence.close()
            self._persistence = None

        if self._commands is not None:
            self._commands.close()
            self._commands = None

        # Stop trading node. ``TradingNode.dispose()`` stops the running
        # event loop (it was designed for ``run()`` which owns its own loop),
        # so we intentionally skip it.
        if self._trading_node is not None:
            with contextlib.suppress(Exception):
                await self._trading_node.stop_async()
            self._trading_node = None

        if self._event_writer is not None:
            with contextlib.suppress(Exception):
                self._event_writer.flush()
            self._event_writer.close()
            self._event_writer = None

        # Let in-flight alerts (e.g. the halt that caused shutdown) go out.
        if self._alert_tasks:
            with contextlib.suppress(Exception):
                await asyncio.wait(set(self._alert_tasks), timeout=10)
        if self._telegram is not None:
            with contextlib.suppress(Exception):
                await self._telegram.close()

        self._state_manager.close()

    async def start(self) -> None:
        """Start the paper trading node.

        Runs until interrupted by signal (SIGINT/SIGTERM).
        """
        try:
            self._setup_signal_handlers()
            await self._initialize()
            await self._run_loop()
        except Exception as e:
            self._status.state = NodeState.ERROR
            self._status.error_message = str(e)
            self._write_event(
                EventType.RISK_CHECK, {"action": "error", "error": str(e), "passed": False}
            )
            self._send_telegram_alert("error", f"Paper node failed: {e}", critical=True)
            raise
        finally:
            await self._shutdown()

    def halt(self, reason: HaltReason, message: str | None = None) -> None:
        """Halt trading: block entries, flatten (reduce-only), cancel orders, stop strategies.

        A daily-loss halt keeps strategies running (flat, entries blocked) and
        lifts at the next UTC day. Halts are idempotent; a halt only replaces a
        less severe one.
        """
        msg = message or reason.value
        if self._guard is not None:
            self._guard.halt(reason, msg)
            return
        # Before the trading node exists there is nothing to flatten; record it.
        self._status.state = NodeState.HALTED
        self._status.halt_reason = reason
        self._status.error_message = msg
        self._status.updated_at = datetime.now(UTC)
        self._write_event(
            EventType.RISK_CHECK,
            {"action": "halt", "reason": reason.value, "message": msg, "passed": False},
        )
        self._send_telegram_alert("circuit_breaker", f"HALT ({reason.value}): {msg}", critical=True)

    def pause(self) -> None:
        """Block new entries; keep protective SL/TP orders and exit logic running."""
        if self._guard is not None:
            self._guard.pause()
            return
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

    def resume(self) -> tuple[bool, str]:
        """Resume from pause / operator-resumable halt. Never while killed."""
        engaged, _ = self._kill_switch()
        if self._guard is not None:
            return self._guard.resume(kill_switch_engaged=engaged)
        if engaged:
            return False, "system kill switch is engaged; unlock it before resuming"
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
            return True, "resumed from pause"
        if self._status.state == NodeState.RUNNING:
            return True, "already running"
        return False, "trading node not running"

    def resume_from_halt(self) -> bool:
        """Resume trading from halted state (operator-resumable halts only)."""
        if self._status.state != NodeState.HALTED:
            return False
        ok, _ = self.resume()
        return ok

    def handle_error(
        self,
        error: BaseException,
        operation: str = "",
        symbol: str = "",
    ) -> ErrorContext:
        """Handle error through the error handler (halt and/or alert)."""
        return self._error_handler.handle_error(
            error=error,
            operation=operation,
            symbol=symbol,
        )


async def run_paper_trading(config: PaperTradingConfig) -> None:
    """Run paper trading with the given configuration."""
    node = PaperTradingNode(config)
    await node.start()


__all__ = [
    "HaltReason",
    "NodeState",
    "NodeStatus",
    "PaperTradingNode",
    "ResolvedRunConfig",
    "run_paper_trading",
    "validated_run_leverage",
    "validated_run_strategy_params",
]
