"""Validation runner for full-fidelity backtesting.

Loads strategy from SQLite, compiles to NautilusTrader Strategy,
runs backtest with realistic execution simulation, and stores results.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import CompilerError, StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.logging.events import EventType, create_event
from vibe_quant.logging.writer import EventWriter
from vibe_quant.validation.latency import LatencyPreset
from vibe_quant.validation.venue import (
    VenueConfig,
    create_venue_config_for_validation,
)

if TYPE_CHECKING:
    from vibe_quant.dsl.schema import StrategyDSL


class ValidationRunnerError(Exception):
    """Error during validation run."""

    pass


@dataclass
class TradeRecord:
    """Individual trade record for storage.

    Attributes:
        symbol: Instrument symbol.
        direction: 'LONG' or 'SHORT'.
        leverage: Leverage used.
        entry_time: Entry timestamp ISO format.
        exit_time: Exit timestamp ISO format.
        entry_price: Entry price.
        exit_price: Exit price.
        quantity: Trade quantity.
        entry_fee: Entry transaction fee.
        exit_fee: Exit transaction fee.
        funding_fees: Total funding paid/received.
        slippage_cost: Slippage cost.
        gross_pnl: PnL before fees.
        net_pnl: PnL after fees.
        roi_percent: Return on investment percent.
        exit_reason: Why position was closed.
    """

    symbol: str
    direction: str
    leverage: int
    entry_time: str
    exit_time: str | None
    entry_price: float
    exit_price: float | None
    quantity: float
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    funding_fees: float = 0.0
    slippage_cost: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    roi_percent: float = 0.0
    exit_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for database storage."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "leverage": self.leverage,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "funding_fees": self.funding_fees,
            "slippage_cost": self.slippage_cost,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "roi_percent": self.roi_percent,
            "exit_reason": self.exit_reason,
        }


@dataclass
class ValidationResult:
    """Results from a validation backtest run.

    Attributes:
        run_id: Backtest run ID.
        strategy_name: Name of strategy.
        total_return: Total return percentage.
        sharpe_ratio: Sharpe ratio.
        sortino_ratio: Sortino ratio.
        max_drawdown: Maximum drawdown percentage.
        total_trades: Number of trades.
        winning_trades: Number of winning trades.
        losing_trades: Number of losing trades.
        win_rate: Win rate percentage.
        profit_factor: Profit factor.
        total_fees: Total fees paid.
        total_funding: Total funding paid/received.
        total_slippage: Total slippage cost.
        execution_time_seconds: Time to run backtest.
        trades: List of individual trades.
    """

    run_id: int
    strategy_name: str
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_fees: float = 0.0
    total_funding: float = 0.0
    total_slippage: float = 0.0
    execution_time_seconds: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)

    def to_metrics_dict(self) -> dict[str, object]:
        """Convert to metrics dictionary for database storage."""
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "total_slippage": self.total_slippage,
            "execution_time_seconds": self.execution_time_seconds,
        }


class ValidationRunner:
    """Runner for validation backtests with full-fidelity execution.

    Loads strategy from database, compiles to NautilusTrader Strategy,
    configures venue with latency/slippage, runs backtest, stores results.

    Example:
        runner = ValidationRunner(db_path=Path("data/state/vibe_quant.db"))
        result = runner.run(run_id=42)
        print(f"Sharpe: {result.sharpe_ratio}")
    """

    def __init__(
        self,
        db_path: Path | None = None,
        logs_path: Path | str = "logs/events",
    ) -> None:
        """Initialize ValidationRunner.

        Args:
            db_path: Path to state database. Uses default if None.
            logs_path: Path for event log files.
        """
        self._state = StateManager(db_path)
        self._logs_path = Path(logs_path)
        self._compiler = StrategyCompiler()

    def close(self) -> None:
        """Close database connection."""
        self._state.close()

    def run(
        self,
        run_id: int,
        latency_preset: LatencyPreset | str | None = None,
    ) -> ValidationResult:
        """Run validation backtest for a given run_id.

        Args:
            run_id: Backtest run ID from database.
            latency_preset: Override latency preset from database.

        Returns:
            ValidationResult with metrics and trades.

        Raises:
            ValidationRunnerError: If run fails.
        """
        start_time = time.monotonic()

        # Load run config from database
        run_config = self._load_run_config(run_id)
        strategy_id_raw = run_config["strategy_id"]
        if not isinstance(strategy_id_raw, int):
            strategy_id_raw = int(str(strategy_id_raw))
        strategy_id: int = strategy_id_raw

        # Load strategy DSL
        strategy_data = self._state.get_strategy(strategy_id)
        if strategy_data is None:
            msg = f"Strategy {strategy_id} not found"
            raise ValidationRunnerError(msg)

        strategy_name = str(strategy_data["name"])
        dsl_config = strategy_data["dsl_config"]

        # Validate and compile strategy
        dsl = self._validate_dsl(dsl_config)
        self._compile_strategy(dsl)

        # Determine latency preset
        effective_latency = self._resolve_latency(run_config, latency_preset)

        # Configure venue
        venue_config = self._create_venue_config(run_config, effective_latency)

        # Update run status to running
        self._state.update_backtest_run_status(run_id, "running")

        # Create event writer
        with EventWriter(run_id=str(run_id), base_path=self._logs_path) as writer:
            # Write start event
            self._write_start_event(writer, run_id, strategy_name, venue_config)

            # Run the backtest (mocked for now)
            result = self._run_backtest_mock(
                run_id=run_id,
                strategy_name=strategy_name,
                dsl=dsl,
                venue_config=venue_config,
                run_config=run_config,
                writer=writer,
            )

            # Write completion event
            self._write_completion_event(writer, run_id, strategy_name, result)

        # Record execution time
        result.execution_time_seconds = time.monotonic() - start_time

        # Store results in database
        self._store_results(run_id, result)

        # Update run status to completed
        self._state.update_backtest_run_status(run_id, "completed")

        return result

    def _load_run_config(self, run_id: int) -> dict[str, object]:
        """Load backtest run configuration from database.

        Args:
            run_id: Run ID to load.

        Returns:
            Run configuration dict.

        Raises:
            ValidationRunnerError: If run not found.
        """
        run_config = self._state.get_backtest_run(run_id)
        if run_config is None:
            msg = f"Backtest run {run_id} not found"
            raise ValidationRunnerError(msg)

        # Validate run mode
        if run_config.get("run_mode") != "validation":
            msg = f"Run {run_id} is not a validation run (mode: {run_config.get('run_mode')})"
            raise ValidationRunnerError(msg)

        return run_config

    def _validate_dsl(self, dsl_config: dict[str, object]) -> StrategyDSL:
        """Validate DSL configuration.

        Args:
            dsl_config: DSL config dict from database.

        Returns:
            Validated StrategyDSL.

        Raises:
            ValidationRunnerError: If validation fails.
        """
        try:
            return validate_strategy_dict(dsl_config)
        except Exception as e:
            msg = f"Strategy DSL validation failed: {e}"
            raise ValidationRunnerError(msg) from e

    def _compile_strategy(self, dsl: StrategyDSL) -> str:
        """Compile strategy DSL to NautilusTrader code.

        Args:
            dsl: Validated strategy DSL.

        Returns:
            Generated Python source code.

        Raises:
            ValidationRunnerError: If compilation fails.
        """
        try:
            return self._compiler.compile(dsl)
        except CompilerError as e:
            msg = f"Strategy compilation failed: {e}"
            raise ValidationRunnerError(msg) from e

    def _resolve_latency(
        self,
        run_config: dict[str, object],
        override: LatencyPreset | str | None,
    ) -> LatencyPreset | str | None:
        """Resolve effective latency preset.

        Args:
            run_config: Run configuration from database.
            override: Override latency preset.

        Returns:
            Effective latency preset.
        """
        if override is not None:
            return override

        db_latency = run_config.get("latency_preset")
        if db_latency:
            return str(db_latency)

        return LatencyPreset.RETAIL  # Default

    def _create_venue_config(
        self,
        run_config: dict[str, object],
        latency_preset: LatencyPreset | str | None,
    ) -> VenueConfig:
        """Create venue configuration for validation.

        Args:
            run_config: Run configuration.
            latency_preset: Latency preset to use.

        Returns:
            Configured VenueConfig.
        """
        return create_venue_config_for_validation(
            starting_balance_usdt=100_000,  # Could be configurable
            latency_preset=latency_preset or LatencyPreset.RETAIL,
        )

    def _run_backtest_mock(
        self,
        run_id: int,
        strategy_name: str,
        dsl: StrategyDSL,
        venue_config: VenueConfig,
        run_config: dict[str, object],
        writer: EventWriter,
    ) -> ValidationResult:
        """Run backtest (mocked implementation).

        This is a placeholder that returns mock results. Will be replaced
        with actual NautilusTrader BacktestEngine integration.

        Args:
            run_id: Run ID.
            strategy_name: Strategy name.
            dsl: Compiled strategy DSL.
            venue_config: Venue configuration.
            run_config: Run configuration.
            writer: Event writer for logging.

        Returns:
            Mock validation result.
        """
        # Create mock result with reasonable values
        result = ValidationResult(
            run_id=run_id,
            strategy_name=strategy_name,
            total_return=5.2,  # Mock: 5.2% return
            sharpe_ratio=1.35,  # Mock: decent Sharpe
            sortino_ratio=1.85,  # Mock: better downside-adjusted
            max_drawdown=8.5,  # Mock: 8.5% max DD
            total_trades=42,  # Mock: 42 trades
            winning_trades=25,
            losing_trades=17,
            win_rate=59.5,  # 25/42
            profit_factor=1.45,  # Mock
            total_fees=125.50,  # Mock fees
            total_funding=45.20,  # Mock funding
            total_slippage=32.10,  # Mock slippage
        )

        # Generate some mock trades
        symbols_raw = run_config.get("symbols", ["BTCUSDT-PERP"])
        if isinstance(symbols_raw, str):
            symbols: list[str] = json.loads(symbols_raw)
        elif isinstance(symbols_raw, list):
            symbols = [str(s) for s in symbols_raw]
        else:
            symbols = ["BTCUSDT-PERP"]

        for i in range(min(5, result.total_trades)):
            is_winner = i < 3  # First 3 are winners
            symbol = symbols[i % len(symbols)] if symbols else "BTCUSDT-PERP"

            trade = TradeRecord(
                symbol=str(symbol),
                direction="LONG" if i % 2 == 0 else "SHORT",
                leverage=10,
                entry_time=f"2025-01-{10 + i:02d}T10:00:00Z",
                exit_time=f"2025-01-{10 + i:02d}T14:00:00Z",
                entry_price=42000.0 + i * 100,
                exit_price=42100.0 + i * 100 if is_winner else 41900.0 + i * 100,
                quantity=0.05,
                entry_fee=2.10,
                exit_fee=2.10,
                funding_fees=1.50 if i % 3 == 0 else -0.75,
                slippage_cost=0.85,
                gross_pnl=100.0 if is_winner else -100.0,
                net_pnl=95.0 if is_winner else -105.0,
                roi_percent=2.5 if is_winner else -2.6,
                exit_reason="take_profit" if is_winner else "stop_loss",
            )
            result.trades.append(trade)

            # Log trade events
            self._write_trade_events(writer, run_id, strategy_name, trade)

        return result

    def _write_start_event(
        self,
        writer: EventWriter,
        run_id: int,
        strategy_name: str,
        venue_config: VenueConfig,
    ) -> None:
        """Write backtest start event."""
        event = create_event(
            event_type=EventType.SIGNAL,  # Using SIGNAL for start marker
            run_id=str(run_id),
            strategy_name=strategy_name,
            data={
                "event": "BACKTEST_START",
                "venue": venue_config.name,
                "latency_preset": str(venue_config.latency_preset)
                if venue_config.latency_preset
                else None,
                "starting_balance": venue_config.starting_balance_usdt,
            },
        )
        writer.write(event)

    def _write_completion_event(
        self,
        writer: EventWriter,
        run_id: int,
        strategy_name: str,
        result: ValidationResult,
    ) -> None:
        """Write backtest completion event."""
        event = create_event(
            event_type=EventType.SIGNAL,  # Using SIGNAL for end marker
            run_id=str(run_id),
            strategy_name=strategy_name,
            data={
                "event": "BACKTEST_COMPLETE",
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "total_trades": result.total_trades,
            },
        )
        writer.write(event)

    def _write_trade_events(
        self,
        writer: EventWriter,
        run_id: int,
        strategy_name: str,
        trade: TradeRecord,
    ) -> None:
        """Write events for a trade (entry and exit)."""
        # Position open event
        open_event = create_event(
            event_type=EventType.POSITION_OPEN,
            run_id=str(run_id),
            strategy_name=strategy_name,
            data={
                "symbol": trade.symbol,
                "side": trade.direction,
                "entry_price": trade.entry_price,
                "quantity": trade.quantity,
                "leverage": trade.leverage,
            },
        )
        writer.write(open_event)

        # Position close event
        if trade.exit_time:
            close_event = create_event(
                event_type=EventType.POSITION_CLOSE,
                run_id=str(run_id),
                strategy_name=strategy_name,
                data={
                    "symbol": trade.symbol,
                    "exit_price": trade.exit_price,
                    "gross_pnl": trade.gross_pnl,
                    "net_pnl": trade.net_pnl,
                    "exit_reason": trade.exit_reason,
                },
            )
            writer.write(close_event)

    def _store_results(self, run_id: int, result: ValidationResult) -> None:
        """Store validation results in database.

        Args:
            run_id: Run ID.
            result: Validation result to store.
        """
        # Save backtest results
        self._state.save_backtest_result(run_id, result.to_metrics_dict())

        # Save individual trades
        trade_dicts = [t.to_dict() for t in result.trades]
        self._state.save_trades_batch(run_id, trade_dicts)


def list_validation_runs(
    db_path: Path | None = None,
    limit: int = 20,
) -> list[dict[str, object]]:
    """List recent validation runs.

    Args:
        db_path: Path to state database.
        limit: Maximum runs to return.

    Returns:
        List of validation run dicts with status and key metrics.
    """
    state = StateManager(db_path)
    try:
        # Get validation runs
        runs = state.list_backtest_runs(status=None)

        # Filter to validation mode and limit
        validation_runs = [r for r in runs if r.get("run_mode") == "validation"][:limit]

        # Enrich with results if available
        enriched = []
        for run in validation_runs:
            run_id = run["id"]
            result = state.get_backtest_result(int(run_id))

            enriched_run = {
                "run_id": run_id,
                "strategy_id": run.get("strategy_id"),
                "status": run.get("status"),
                "latency_preset": run.get("latency_preset"),
                "created_at": run.get("created_at"),
                "completed_at": run.get("completed_at"),
            }

            if result:
                enriched_run["sharpe_ratio"] = result.get("sharpe_ratio")
                enriched_run["total_return"] = result.get("total_return")
                enriched_run["max_drawdown"] = result.get("max_drawdown")
                enriched_run["total_trades"] = result.get("total_trades")

            enriched.append(enriched_run)

        return enriched
    finally:
        state.close()


__all__ = [
    "ValidationRunner",
    "ValidationRunnerError",
    "ValidationResult",
    "TradeRecord",
    "list_validation_runs",
]
