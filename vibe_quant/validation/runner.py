"""Validation runner for full-fidelity backtesting.

Loads strategy from SQLite, compiles to NautilusTrader Strategy,
runs backtest with realistic execution simulation, and stores results.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import CompilerError, StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.logging.events import EventType, create_event
from vibe_quant.logging.writer import EventWriter
from vibe_quant.validation.fill_model import SlippageEstimator
from vibe_quant.validation.latency import LatencyPreset
from vibe_quant.validation.venue import (
    VenueConfig,
    create_backtest_venue_config,
    create_venue_config_for_validation,
)

if TYPE_CHECKING:
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.backtest.results import BacktestResult

    from vibe_quant.dsl.schema import StrategyDSL

logger = logging.getLogger(__name__)


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
    cagr: float = 0.0
    calmar_ratio: float = 0.0
    volatility_annual: float = 0.0
    max_drawdown_duration_days: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration_hours: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
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
            "cagr": self.cagr,
            "calmar_ratio": self.calmar_ratio,
            "volatility_annual": self.volatility_annual,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_duration_hours": self.avg_trade_duration_hours,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
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

        try:
            # Create event writer
            with EventWriter(run_id=str(run_id), base_path=self._logs_path) as writer:
                self._write_start_event(writer, run_id, strategy_name, venue_config)

                result = self._run_backtest(
                    run_id=run_id,
                    strategy_name=strategy_name,
                    dsl=dsl,
                    venue_config=venue_config,
                    run_config=run_config,
                    writer=writer,
                )

                self._write_completion_event(writer, run_id, strategy_name, result)

            result.execution_time_seconds = time.monotonic() - start_time
            self._store_results(run_id, result)
            self._state.update_backtest_run_status(run_id, "completed")
            return result
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            try:
                self._state.update_backtest_run_status(
                    run_id, "failed", error_message=error_msg
                )
            except Exception:
                logger.exception("Failed to update run %d status to failed", run_id)
            raise ValidationRunnerError(error_msg) from exc

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

    def _run_backtest(
        self,
        run_id: int,
        strategy_name: str,
        dsl: StrategyDSL,
        venue_config: VenueConfig,
        run_config: dict[str, object],
        writer: EventWriter,
    ) -> ValidationResult:
        """Run NautilusTrader backtest with full-fidelity execution.

        Compiles the strategy DSL to a NautilusTrader Strategy, loads market
        data from the ParquetDataCatalog, configures the venue with latency
        and slippage models, and runs a BacktestNode.

        Args:
            run_id: Run ID.
            strategy_name: Strategy name.
            dsl: Validated strategy DSL.
            venue_config: Venue configuration.
            run_config: Run configuration from database.
            writer: Event writer for logging.

        Returns:
            ValidationResult with real metrics and trades.

        Raises:
            ValidationRunnerError: If backtest setup or execution fails.
        """
        from nautilus_trader.backtest.node import BacktestNode
        from nautilus_trader.config import (
            BacktestDataConfig,
            BacktestEngineConfig,
            BacktestRunConfig,
            ImportableStrategyConfig,
        )

        from vibe_quant.data.catalog import (
            DEFAULT_CATALOG_PATH,
            INSTRUMENT_CONFIGS,
            INTERVAL_TO_AGGREGATION,
            CatalogManager,
            create_instrument,
        )

        # Parse symbols from run config
        symbols = self._parse_symbols(run_config)

        # Parse date range
        start_date = str(run_config.get("start_date", "2024-01-01"))
        end_date = str(run_config.get("end_date", "2024-12-31"))

        # Collect all timeframes needed by the strategy
        all_timeframes = {dsl.timeframe}
        all_timeframes.update(dsl.additional_timeframes)
        for ind_config in dsl.indicators.values():
            if ind_config.timeframe:
                all_timeframes.add(ind_config.timeframe)

        # Ensure instruments exist in catalog
        catalog_path = DEFAULT_CATALOG_PATH
        catalog_mgr = CatalogManager(catalog_path)
        for symbol in symbols:
            if symbol in INSTRUMENT_CONFIGS:
                instrument = create_instrument(symbol)
                catalog_mgr.write_instrument(instrument)

        # Compile strategy to an importable module (registers in sys.modules)
        module = self._compiler.compile_to_module(dsl)
        class_name = "".join(word.capitalize() for word in dsl.name.split("_"))
        module_path = f"vibe_quant.dsl.generated.{dsl.name}"

        # Verify generated classes exist in the module
        strategy_cls_name = f"{class_name}Strategy"
        config_cls_name = f"{class_name}Config"
        if not hasattr(module, strategy_cls_name) or not hasattr(module, config_cls_name):
            msg = (
                f"Compiled module missing expected classes: "
                f"{strategy_cls_name}, {config_cls_name}"
            )
            raise ValidationRunnerError(msg)

        # Build strategy config dict: instrument_id + any override parameters
        strategy_params = self._build_strategy_params(run_config)

        # Build strategy configs (one per symbol)
        strategy_configs: list[ImportableStrategyConfig] = []
        for symbol in symbols:
            instrument_id = f"{symbol}-PERP.BINANCE"
            config_dict = {"instrument_id": instrument_id, **strategy_params}
            strategy_configs.append(
                ImportableStrategyConfig(
                    strategy_path=f"{module_path}:{strategy_cls_name}",
                    config_path=f"{module_path}:{config_cls_name}",
                    config=config_dict,
                )
            )

        # Build data configs (one per symbol per timeframe)
        data_configs: list[BacktestDataConfig] = []
        for symbol in symbols:
            instrument_id = f"{symbol}-PERP.BINANCE"
            for tf in sorted(all_timeframes):
                if tf not in INTERVAL_TO_AGGREGATION:
                    logger.warning("Unknown timeframe %s, skipping", tf)
                    continue
                step, agg = INTERVAL_TO_AGGREGATION[tf]
                data_configs.append(
                    BacktestDataConfig(
                        catalog_path=str(catalog_path.resolve()),
                        data_cls="nautilus_trader.model.data:Bar",
                        instrument_id=instrument_id,
                        bar_spec=f"{step}-{agg.name}-LAST",
                        start_time=start_date,
                        end_time=end_date,
                    )
                )

        if not data_configs:
            msg = "No valid data configurations could be built"
            raise ValidationRunnerError(msg)

        # Convert our VenueConfig to NautilusTrader BacktestVenueConfig
        bt_venue_config = create_backtest_venue_config(venue_config)

        # Create engine config
        engine_config = BacktestEngineConfig(
            strategies=strategy_configs,
            run_analysis=True,
        )

        # Create run config -- dispose_on_completion=False so we can
        # access engine.trader for positions report after run completes.
        bt_run_config = BacktestRunConfig(
            engine=engine_config,
            venues=[bt_venue_config],
            data=data_configs,
            start=start_date,
            end=end_date,
            dispose_on_completion=False,
        )

        logger.info(
            "Starting NautilusTrader backtest for run %d: "
            "%d symbols, %d timeframes, %s to %s",
            run_id,
            len(symbols),
            len(all_timeframes),
            start_date,
            end_date,
        )

        # Execute the backtest
        node = BacktestNode(configs=[bt_run_config])
        try:
            # Build engines, then register portfolio statistics before running
            node.build()
            self._register_statistics(node)
            node.run()

            engine = node.get_engine(bt_run_config.id)
            bt_result = engine.get_result()

            # Extract metrics and trades from the engine
            result = self._extract_results(
                run_id=run_id,
                strategy_name=strategy_name,
                bt_result=bt_result,
                engine=engine,
                venue_config=venue_config,
            )

            # Log trade events
            for trade in result.trades:
                self._write_trade_events(writer, run_id, strategy_name, trade)

            return result
        finally:
            node.dispose()

    def _register_statistics(self, node: object) -> None:
        """Register portfolio statistics on the engine's analyzer.

        NautilusTrader's PortfolioAnalyzer starts with no registered
        statistics.  We register the standard set so that BacktestResult
        stats_pnls / stats_returns are populated.

        Args:
            node: BacktestNode (after build, before run).
        """
        from nautilus_trader.core.nautilus_pyo3 import (
            AvgLoser,
            AvgWinner,
            Expectancy,
            LongRatio,
            MaxDrawdown,
            ProfitFactor,
            SharpeRatio,
            SortinoRatio,
            WinRate,
        )

        stats = [
            SharpeRatio(),
            SortinoRatio(),
            MaxDrawdown(),
            WinRate(),
            ProfitFactor(),
            Expectancy(),
            LongRatio(),
            AvgWinner(),
            AvgLoser(),
        ]

        for engine in node.get_engines():
            analyzer = engine.kernel.portfolio.analyzer
            for stat in stats:
                analyzer.register_statistic(stat)

    def _build_strategy_params(self, run_config: dict[str, object]) -> dict[str, object]:
        """Extract strategy parameter overrides from run config.

        If the run was created to validate specific sweep parameters,
        those are stored in run_config['parameters'] and should be
        forwarded to the compiled strategy's config.

        Args:
            run_config: Run configuration dict from database.

        Returns:
            Dict of parameter overrides to merge into ImportableStrategyConfig.
        """
        params: dict[str, object] = {}
        raw_params = run_config.get("parameters")
        if not isinstance(raw_params, dict):
            return params

        # Direct parameter overrides (e.g., from validated screening results)
        for key, value in raw_params.items():
            # Skip meta-keys that aren't strategy parameters
            if key in ("sweep", "overfitting_filters"):
                continue
            params[key] = value

        return params

    def _parse_symbols(self, run_config: dict[str, object]) -> list[str]:
        """Parse symbol list from run configuration.

        Args:
            run_config: Run configuration dict from database.

        Returns:
            List of symbol strings (e.g., ['BTCUSDT', 'ETHUSDT']).
        """
        symbols_raw = run_config.get("symbols", ["BTCUSDT"])
        if isinstance(symbols_raw, str):
            return json.loads(symbols_raw)
        elif isinstance(symbols_raw, list):
            return [str(s) for s in symbols_raw]
        return ["BTCUSDT"]

    def _extract_results(
        self,
        run_id: int,
        strategy_name: str,
        bt_result: BacktestResult,
        engine: BacktestEngine,
        venue_config: VenueConfig,
    ) -> ValidationResult:
        """Extract ValidationResult from NautilusTrader backtest output.

        Parses BacktestResult statistics and the engine's position/fill
        reports into our ValidationResult and TradeRecord format.

        Args:
            run_id: Run ID.
            strategy_name: Strategy name.
            bt_result: NautilusTrader BacktestResult.
            engine: BacktestEngine after run for report generation.
            venue_config: Venue config for leverage info.

        Returns:
            Populated ValidationResult.
        """
        result = ValidationResult(
            run_id=run_id,
            strategy_name=strategy_name,
        )

        if bt_result is None:
            return result

        result.execution_time_seconds = bt_result.elapsed_time
        result.total_trades = bt_result.total_positions

        # Parse aggregate statistics from BacktestResult
        self._extract_stats(result, bt_result)

        # Extract individual trades from engine position reports
        self._extract_trades(result, engine, venue_config)

        return result

    def _extract_stats(
        self,
        result: ValidationResult,
        bt_result: BacktestResult,
    ) -> None:
        """Extract aggregate statistics from BacktestResult into ValidationResult.

        NT's PortfolioAnalyzer populates stats_pnls with PnL and any
        registered statistics keyed by their ``name`` attribute, and
        stats_returns with the same registered statistics.

        Known key names from NT 1.222 (Rust statistics):
            stats_pnls:  "PnL (total)", "PnL% (total)", "Sharpe Ratio (252 days)",
                         "Sortino Ratio (252 days)", "Max Drawdown", "Win Rate",
                         "Profit Factor", "Expectancy", "Avg Winner", "Avg Loser"
            stats_returns: same statistic names

        Args:
            result: ValidationResult to populate (mutated in place).
            bt_result: NautilusTrader BacktestResult.
        """
        stats_returns = bt_result.stats_returns or {}
        stats_pnls = bt_result.stats_pnls or {}

        _known_pnl_keys = {"pnl (total)", "pnl% (total)", "sharpe", "sortino",
                           "max drawdown", "win rate", "profit factor",
                           "expectancy", "avg winner", "avg loser", "long ratio"}

        # Extract from PnL stats first (keyed by currency, e.g. "USDT")
        for _currency, pnl_stats in stats_pnls.items():
            for key, value in pnl_stats.items():
                if value is None:
                    continue
                key_lower = key.lower()
                fval = float(value)
                if key_lower == "pnl% (total)":
                    result.total_return = fval
                elif "sharpe" in key_lower:
                    result.sharpe_ratio = fval
                elif "sortino" in key_lower:
                    result.sortino_ratio = fval
                elif key_lower == "max drawdown":
                    result.max_drawdown = abs(fval)
                elif key_lower == "win rate":
                    result.win_rate = fval
                elif key_lower == "profit factor":
                    result.profit_factor = fval
                elif key_lower == "avg winner":
                    result.avg_win = fval
                elif key_lower == "avg loser":
                    result.avg_loss = fval
                elif not any(k in key_lower for k in _known_pnl_keys):
                    logger.debug("Unmatched PnL stats key: %s = %s", key, value)

        # Fill from returns stats only if not already set by PnL stats
        _known_returns_keys = {"sharpe", "sortino", "max drawdown", "win rate",
                               "profit factor", "expectancy", "avg winner",
                               "avg loser", "long ratio"}
        for key, value in stats_returns.items():
            if value is None:
                continue
            key_lower = key.lower()
            fval = float(value)
            if "sharpe" in key_lower and result.sharpe_ratio == 0.0:
                result.sharpe_ratio = fval
            elif "sortino" in key_lower and result.sortino_ratio == 0.0:
                result.sortino_ratio = fval
            elif "max drawdown" in key_lower and result.max_drawdown == 0.0:
                result.max_drawdown = abs(fval)
            elif key_lower == "win rate" and result.win_rate == 0.0:
                result.win_rate = fval
            elif key_lower == "profit factor" and result.profit_factor == 0.0:
                result.profit_factor = fval
            elif not any(k in key_lower for k in _known_returns_keys):
                logger.debug("Unmatched returns stats key: %s = %s", key, value)

    def _extract_trades(
        self,
        result: ValidationResult,
        engine: BacktestEngine,
        venue_config: VenueConfig,
    ) -> None:
        """Extract individual trade records from the engine's closed positions.

        Uses the Position objects from the engine cache directly, since the
        positions report DataFrame column names can vary across NT versions.

        Args:
            result: ValidationResult to populate trades on (mutated in place).
            engine: BacktestEngine after run.
            venue_config: Venue config for default leverage.
        """
        try:
            positions = engine.kernel.cache.positions()
        except Exception:
            logger.warning("Could not read positions from engine cache", exc_info=True)
            return

        if not positions:
            return

        default_leverage = int(venue_config.default_leverage)
        winning = 0
        losing = 0
        total_fees = 0.0
        total_slippage = 0.0

        # Slippage estimator using SPEC formula for post-fill cost analytics
        fill_cfg = venue_config.fill_config
        impact_k = getattr(fill_cfg, "impact_coefficient", 0.1) if fill_cfg else 0.1
        slippage_estimator = SlippageEstimator(impact_coefficient=impact_k)

        # Estimate avg volume and volatility from bar data in engine cache
        avg_bar_volume, bar_volatility = self._estimate_market_stats(engine)

        for pos in positions:
            if not pos.is_closed:
                continue

            realized_pnl = float(pos.realized_pnl)
            entry_price = float(pos.avg_px_open)
            exit_price = float(pos.avg_px_close)
            quantity = float(pos.quantity)

            # commissions is a list of Money objects
            pos_fees = sum(float(c) for c in pos.commissions)
            total_fees += abs(pos_fees)

            if realized_pnl > 0:
                winning += 1
            elif realized_pnl < 0:
                losing += 1

            # Estimate slippage cost using SPEC formula
            slippage_cost = slippage_estimator.estimate_cost(
                entry_price=entry_price,
                order_size=quantity,
                avg_volume=avg_bar_volume,
                volatility=bar_volatility,
                spread=0.0001,  # ~1bp spread for major pairs
            )
            total_slippage += slippage_cost

            # Compute ROI: PnL / notional value at entry
            notional = entry_price * quantity if entry_price and quantity else 1.0
            roi_pct = (realized_pnl / notional) * 100.0 if notional else 0.0

            # Format timestamps (ns -> ISO via pandas/datetime)
            entry_time = str(pos.ts_opened)
            exit_time = str(pos.ts_closed) if pos.ts_closed else None

            # entry is the opening OrderSide (BUY = LONG, SELL = SHORT)
            direction = "LONG" if str(pos.entry).upper() == "BUY" else "SHORT"

            instrument_id = str(pos.instrument_id)

            trade = TradeRecord(
                symbol=instrument_id,
                direction=direction,
                leverage=default_leverage,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                entry_fee=abs(pos_fees) / 2.0,
                exit_fee=abs(pos_fees) / 2.0,
                slippage_cost=slippage_cost,
                gross_pnl=realized_pnl + abs(pos_fees),
                net_pnl=realized_pnl,
                roi_percent=roi_pct,
                exit_reason="signal",
            )
            result.trades.append(trade)

        result.total_trades = len(result.trades)
        result.winning_trades = winning
        result.losing_trades = losing
        result.total_fees = total_fees
        result.total_slippage = total_slippage
        if result.total_trades > 0:
            result.win_rate = winning / result.total_trades

        # Compute SPEC-required extended metrics from trades
        self._compute_extended_metrics(result)

    @staticmethod
    def _estimate_market_stats(engine: BacktestEngine) -> tuple[float, float]:
        """Estimate average bar volume and daily volatility from engine cache.

        Reads bars from the engine cache to compute realistic slippage
        parameters instead of using hardcoded values.

        Args:
            engine: BacktestEngine after run.

        Returns:
            Tuple of (avg_bar_volume, daily_volatility). Falls back to
            conservative defaults (1000.0, 0.02) if data is unavailable.
        """
        default_volume = 1000.0
        default_volatility = 0.02

        try:
            bars = engine.kernel.cache.bars()
            if not bars:
                return default_volume, default_volatility

            volumes: list[float] = []
            closes: list[float] = []
            for bar in bars:
                vol = float(bar.volume)
                if vol > 0:
                    volumes.append(vol)
                close = float(bar.close)
                if close > 0:
                    closes.append(close)

            avg_volume = sum(volumes) / len(volumes) if volumes else default_volume

            # Compute daily volatility from log returns
            volatility = default_volatility
            if len(closes) >= 2:
                import math

                log_returns: list[float] = []
                for i in range(1, len(closes)):
                    if closes[i - 1] > 0:
                        log_returns.append(math.log(closes[i] / closes[i - 1]))
                if len(log_returns) >= 2:
                    mean_r = sum(log_returns) / len(log_returns)
                    var = sum((r - mean_r) ** 2 for r in log_returns) / (
                        len(log_returns) - 1
                    )
                    volatility = math.sqrt(var) if var > 0 else default_volatility

            return avg_volume, volatility
        except Exception:
            logger.debug(
                "Could not estimate market stats from engine cache, using defaults",
                exc_info=True,
            )
            return default_volume, default_volatility

    def _compute_extended_metrics(self, result: ValidationResult) -> None:
        """Compute SPEC-required extended metrics from trades.

        Populates: largest_win/loss, avg_win/loss, max_consecutive_wins/losses,
        avg_trade_duration_hours, cagr, volatility_annual, calmar_ratio.

        Args:
            result: ValidationResult to populate (mutated in place).
        """
        if not result.trades:
            return

        wins: list[float] = []
        losses: list[float] = []
        durations_hours: list[float] = []

        # Consecutive streak tracking
        max_con_wins = 0
        max_con_losses = 0
        cur_wins = 0
        cur_losses = 0

        for trade in result.trades:
            pnl = trade.net_pnl
            if pnl > 0:
                wins.append(pnl)
                cur_wins += 1
                max_con_wins = max(max_con_wins, cur_wins)
                cur_losses = 0
            elif pnl < 0:
                losses.append(pnl)
                cur_losses += 1
                max_con_losses = max(max_con_losses, cur_losses)
                cur_wins = 0
            else:
                cur_wins = 0
                cur_losses = 0

            # Trade duration
            if trade.entry_time and trade.exit_time:
                try:
                    from datetime import datetime

                    entry_dt = datetime.fromisoformat(trade.entry_time.replace("Z", "+00:00"))
                    exit_dt = datetime.fromisoformat(trade.exit_time.replace("Z", "+00:00"))
                    duration_h = (exit_dt - entry_dt).total_seconds() / 3600.0
                    if duration_h >= 0:
                        durations_hours.append(duration_h)
                except (ValueError, TypeError):
                    pass

        result.max_consecutive_wins = max_con_wins
        result.max_consecutive_losses = max_con_losses

        if wins:
            result.largest_win = max(wins)
            result.avg_win = sum(wins) / len(wins)
        if losses:
            result.largest_loss = min(losses)
            result.avg_loss = sum(losses) / len(losses)

        # Average trade duration
        if durations_hours:
            result.avg_trade_duration_hours = sum(durations_hours) / len(durations_hours)

        # CAGR: (1 + total_return)^(365/days) - 1
        if result.total_return != 0.0 and result.trades:
            try:
                from datetime import datetime

                first_entry = datetime.fromisoformat(
                    result.trades[0].entry_time.replace("Z", "+00:00")
                )
                last_exit_str = result.trades[-1].exit_time or result.trades[-1].entry_time
                last_exit = datetime.fromisoformat(last_exit_str.replace("Z", "+00:00"))
                days = max((last_exit - first_entry).total_seconds() / 86400.0, 1.0)
                total_return_frac = result.total_return / 100.0 if abs(result.total_return) > 2.0 else result.total_return
                if total_return_frac > -1.0:
                    import math
                    result.cagr = ((1.0 + total_return_frac) ** (365.0 / days)) - 1.0
            except (ValueError, TypeError):
                pass

        # Annualized volatility from per-trade returns
        if len(result.trades) >= 2:
            trade_returns: list[float] = [t.roi_percent / 100.0 for t in result.trades if t.roi_percent != 0.0]
            if len(trade_returns) >= 2:
                import math
                mean_r = sum(trade_returns) / len(trade_returns)
                var = sum((r - mean_r) ** 2 for r in trade_returns) / (len(trade_returns) - 1)
                # Annualize: assume ~252 trading days, estimate trades per day
                if durations_hours:
                    avg_dur_days = max(sum(durations_hours) / len(durations_hours) / 24.0, 0.01)
                    trades_per_year = 365.0 / avg_dur_days
                else:
                    trades_per_year = 252.0
                result.volatility_annual = math.sqrt(var * trades_per_year) if var > 0 else 0.0

        # Calmar ratio: CAGR / max_drawdown
        if result.max_drawdown > 0 and result.cagr != 0:
            result.calmar_ratio = result.cagr / result.max_drawdown

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
