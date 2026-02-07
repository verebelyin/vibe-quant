"""DSL-to-NautilusTrader Strategy compiler.

Compiles parsed StrategyDSL into NautilusTrader Strategy subclass Python source code.
Generates on_start() with multi-TF subscriptions, indicator registration,
and on_bar() with time filter evaluation, condition checking, and order submission.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from vibe_quant.dsl.conditions import Condition, Operator, parse_condition
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    from types import ModuleType

    from vibe_quant.dsl.schema import (
        IndicatorConfig,
        SessionConfig,
        StrategyDSL,
        TimeFilterConfig,
    )
    pass


class CompilerError(Exception):
    """Error raised during DSL compilation."""

    pass


@dataclass(frozen=True, slots=True)
class IndicatorInfo:
    """Info about an indicator needed for code generation.

    Attributes:
        name: DSL indicator name (e.g., "rsi", "ema_fast")
        config: IndicatorConfig from DSL
        spec: IndicatorSpec from registry
        timeframe: Effective timeframe (uses strategy primary if not specified)
        bar_type_var: Name of bar_type variable (e.g., "self.bar_type_5m")
        indicator_var: Name of indicator variable (e.g., "self.rsi")
    """

    name: str
    config: IndicatorConfig
    spec: IndicatorSpec
    timeframe: str
    bar_type_var: str
    indicator_var: str


class StrategyCompiler:
    """Compiles StrategyDSL to NautilusTrader Strategy Python source code.

    Example:
        compiler = StrategyCompiler()
        source_code = compiler.compile(dsl)
        module = compiler.compile_to_module(dsl)
    """

    def __init__(self) -> None:
        """Initialize the compiler."""
        pass

    def compile(self, dsl: StrategyDSL) -> str:
        """Compile DSL to Python source code string.

        Args:
            dsl: Parsed and validated StrategyDSL

        Returns:
            Python source code for the Strategy class and Config

        Raises:
            CompilerError: If compilation fails
        """
        indicator_names = list(dsl.indicators.keys())

        # Gather indicator info
        indicators = self._gather_indicator_info(dsl)

        # Gather all timeframes
        timeframes = self._get_all_timeframes(dsl)

        # Generate parts
        imports = self._generate_imports(dsl, indicators)
        config_class = self._generate_config_class(dsl)
        strategy_class = self._generate_strategy_class(
            dsl, indicators, timeframes, indicator_names
        )

        # Combine
        source = "\n".join([imports, "", config_class, "", strategy_class])
        return source

    def compile_to_module(self, dsl: StrategyDSL) -> ModuleType:
        """Compile DSL to a loadable Python module.

        Args:
            dsl: Parsed and validated StrategyDSL

        Returns:
            Loaded module containing the Strategy and Config classes

        Raises:
            CompilerError: If compilation fails
        """
        source = self.compile(dsl)
        module_name = f"vibe_quant.dsl.generated.{dsl.name}"

        # Create module
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        if spec is None:
            msg = f"Failed to create module spec for {module_name}"
            raise CompilerError(msg)

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # Execute code in module namespace
        try:
            exec(source, module.__dict__)  # noqa: S102
        except Exception as e:
            # Clean up failed module
            sys.modules.pop(module_name, None)
            msg = f"Failed to execute compiled strategy: {e}"
            raise CompilerError(msg) from e

        return module

    def _gather_indicator_info(self, dsl: StrategyDSL) -> list[IndicatorInfo]:
        """Gather info about all indicators in the DSL.

        Args:
            dsl: Parsed DSL

        Returns:
            List of IndicatorInfo for each indicator
        """
        indicators: list[IndicatorInfo] = []

        for name, config in dsl.indicators.items():
            spec = indicator_registry.get(config.type)
            if spec is None:
                msg = f"Unknown indicator type '{config.type}' for indicator '{name}'"
                raise CompilerError(msg)

            # Determine effective timeframe
            timeframe = config.timeframe or dsl.timeframe

            # Generate variable names
            bar_type_var = f"self.bar_type_{timeframe}"
            indicator_var = f"self.ind_{name}"

            indicators.append(
                IndicatorInfo(
                    name=name,
                    config=config,
                    spec=spec,
                    timeframe=timeframe,
                    bar_type_var=bar_type_var,
                    indicator_var=indicator_var,
                )
            )

        return indicators

    def _get_all_timeframes(self, dsl: StrategyDSL) -> set[str]:
        """Get all timeframes used in the strategy.

        Args:
            dsl: Parsed DSL

        Returns:
            Set of timeframe strings
        """
        timeframes = {dsl.timeframe}
        timeframes.update(dsl.additional_timeframes)

        # Also check indicator-specific timeframes
        for config in dsl.indicators.values():
            if config.timeframe:
                timeframes.add(config.timeframe)

        return timeframes

    def _generate_imports(
        self, dsl: StrategyDSL, indicators: list[IndicatorInfo]
    ) -> str:
        """Generate import statements.

        Args:
            dsl: Parsed DSL
            indicators: List of indicator info

        Returns:
            Import statements as string
        """
        imports = [
            '"""Auto-generated NautilusTrader Strategy from DSL.',
            "",
            f"Strategy: {dsl.name}",
            f"Generated: {datetime.now().isoformat()}",
            '"""',
            "",
            "from __future__ import annotations",
            "",
            "from datetime import time as dt_time",
            "from typing import TYPE_CHECKING",
            "import zoneinfo",
            "",
            "from nautilus_trader.core.uuid import UUID4",
            "from nautilus_trader.model.data import Bar, BarType",
            "from nautilus_trader.model.enums import OrderSide, TimeInForce",
            "from nautilus_trader.model.identifiers import InstrumentId",
            "from nautilus_trader.model.instruments import Instrument",
            "from nautilus_trader.model.objects import Price, Quantity",
            "from nautilus_trader.model.events import OrderFilled, PositionChanged, PositionOpened, PositionClosed",
            "from nautilus_trader.model.orders import LimitOrder, MarketOrder, StopMarketOrder",
            "from nautilus_trader.trading.strategy import Strategy, StrategyConfig",
        ]

        # Collect unique indicator classes
        nt_classes: dict[str, str] = {}  # class_name -> module_path
        for info in indicators:
            if info.spec.nt_class is not None:
                class_name = info.spec.nt_class.__name__
                module_path = info.spec.nt_class.__module__
                nt_classes[class_name] = module_path

        # Add indicator imports
        if nt_classes:
            imports.append("")
            imports.append("# Indicator imports")
            for class_name, module_path in sorted(nt_classes.items()):
                imports.append(f"from {module_path} import {class_name}")

        imports.append("")
        imports.append("if TYPE_CHECKING:")
        imports.append("    pass")

        return "\n".join(imports)

    def _generate_config_class(self, dsl: StrategyDSL) -> str:
        """Generate the Strategy config dataclass.

        Args:
            dsl: Parsed DSL

        Returns:
            Config class source code
        """
        class_name = _to_class_name(dsl.name)
        config_name = f"{class_name}Config"

        # Note: Don't use @dataclass - StrategyConfig handles this
        lines = [
            f"class {config_name}(StrategyConfig):",
            f'    """Configuration for {class_name} strategy.',
            "",
            f"    Generated from DSL: {dsl.name}",
            '    """',
            "",
            "    # Instrument configuration",
            '    instrument_id: str = ""  # Must be set at runtime',
            "",
        ]

        # Add risk/sizing parameter (overridable via strategy config or sweep params)
        lines.append("    # Position sizing (override via ImportableStrategyConfig.config)")
        lines.append("    risk_per_trade: float = 0.02  # 2% risk per trade")
        lines.append("")

        # Add indicator parameters
        lines.append("    # Indicator parameters")
        for name, config in dsl.indicators.items():
            if config.period is not None:
                lines.append(f"    {name}_period: int = {config.period}")
            if config.fast_period is not None:
                lines.append(f"    {name}_fast_period: int = {config.fast_period}")
            if config.slow_period is not None:
                lines.append(f"    {name}_slow_period: int = {config.slow_period}")
            if config.signal_period is not None:
                lines.append(f"    {name}_signal_period: int = {config.signal_period}")
            if config.std_dev is not None:
                lines.append(f"    {name}_std_dev: float = {config.std_dev}")
            if config.atr_multiplier is not None:
                lines.append(
                    f"    {name}_atr_multiplier: float = {config.atr_multiplier}"
                )

        # Add stop loss parameters
        lines.append("")
        lines.append("    # Stop loss parameters")
        lines.append(f'    stop_loss_type: str = "{dsl.stop_loss.type}"')
        if dsl.stop_loss.percent is not None:
            lines.append(f"    stop_loss_percent: float = {dsl.stop_loss.percent}")
        if dsl.stop_loss.atr_multiplier is not None:
            lines.append(
                f"    stop_loss_atr_multiplier: float = {dsl.stop_loss.atr_multiplier}"
            )
        if dsl.stop_loss.indicator is not None:
            lines.append(f'    stop_loss_indicator: str = "{dsl.stop_loss.indicator}"')

        # Add take profit parameters
        lines.append("")
        lines.append("    # Take profit parameters")
        lines.append(f'    take_profit_type: str = "{dsl.take_profit.type}"')
        if dsl.take_profit.percent is not None:
            lines.append(f"    take_profit_percent: float = {dsl.take_profit.percent}")
        if dsl.take_profit.atr_multiplier is not None:
            lines.append(
                f"    take_profit_atr_multiplier: float = {dsl.take_profit.atr_multiplier}"
            )
        if dsl.take_profit.risk_reward_ratio is not None:
            lines.append(
                f"    take_profit_risk_reward: float = {dsl.take_profit.risk_reward_ratio}"
            )
        if dsl.take_profit.indicator is not None:
            lines.append(
                f'    take_profit_indicator: str = "{dsl.take_profit.indicator}"'
            )

        # Add custom thresholds (extracted from conditions)
        lines.append("")
        lines.append("    # Condition thresholds (can be overridden)")
        seen_thresholds: dict[str, float | int] = {}
        for cond_str in (
            dsl.entry_conditions.long
            + dsl.entry_conditions.short
            + dsl.exit_conditions.long
            + dsl.exit_conditions.short
        ):
            cond = parse_condition(cond_str, list(dsl.indicators.keys()))
            if not cond.right.is_indicator and not cond.right.is_price and isinstance(cond.right.value, (int, float)):
                value_str = str(cond.right.value).replace(".", "_").replace("-", "neg_")
                # Try short name first (backward compatible)
                short_name = f"{cond.left.value}_{value_str}_threshold"
                if short_name not in seen_thresholds:
                    seen_thresholds[short_name] = cond.right.value
                elif seen_thresholds[short_name] != cond.right.value:
                    # Collision: same indicator, same value but different operator
                    # Use disambiguated name
                    op_name = cond.operator.name.lower()
                    long_name = f"{cond.left.value}_{op_name}_{value_str}_threshold"
                    if long_name not in seen_thresholds:
                        seen_thresholds[long_name] = cond.right.value

        for param_name, default_val in seen_thresholds.items():
            lines.append(f"    {param_name}: float = {default_val}")

        return "\n".join(lines)

    def _generate_strategy_class(
        self,
        dsl: StrategyDSL,
        indicators: list[IndicatorInfo],
        timeframes: set[str],
        indicator_names: list[str],
    ) -> str:
        """Generate the Strategy class.

        Args:
            dsl: Parsed DSL
            indicators: List of indicator info
            timeframes: All timeframes used
            indicator_names: List of indicator names for condition parsing

        Returns:
            Strategy class source code
        """
        class_name = _to_class_name(dsl.name)
        config_name = f"{class_name}Config"

        lines = [
            f"class {class_name}Strategy(Strategy):",
            f'    """NautilusTrader Strategy: {dsl.name}.',
            "",
            f"    {dsl.description or 'Auto-generated from DSL.'}",
            '    """',
            "",
            f"    def __init__(self, config: {config_name}) -> None:",
            f'        """Initialize {class_name}Strategy."""',
            "        super().__init__(config)",
            "        self.config = config",
            "",
            "        # Position tracking",
            "        self._position_open = False",
            "        self._position_side: OrderSide | None = None",
            "",
            "        # Previous indicator values for crossover detection",
            "        self._prev_values: dict[str, float] = {}",
            "",
        ]

        # Add on_start method
        on_start = self._generate_on_start(dsl, indicators, timeframes)
        lines.append(textwrap.indent(on_start, "    "))
        lines.append("")

        # Add on_bar method
        on_bar = self._generate_on_bar(dsl, indicators, indicator_names)
        lines.append(textwrap.indent(on_bar, "    "))
        lines.append("")

        # Add on_event method for position tracking
        on_event = self._generate_on_event()
        lines.append(textwrap.indent(on_event, "    "))
        lines.append("")

        # Add on_stop method for cleanup
        on_stop = self._generate_on_stop()
        lines.append(textwrap.indent(on_stop, "    "))
        lines.append("")

        # Add helper methods
        helpers = self._generate_helper_methods(dsl, indicators, indicator_names)
        lines.append(textwrap.indent(helpers, "    "))

        return "\n".join(lines)

    def _generate_on_event(self) -> str:
        """Generate on_event() method for event-based position tracking.

        SL/TP orders are submitted here on PositionOpened, not in the entry
        methods, to ensure the entry order has actually filled first.  This
        avoids a race condition where reduce_only SL/TP orders are rejected
        because no position exists yet.  The actual fill price from the
        position's avg_px_open is used instead of bar.close estimate.

        Returns:
            on_event method source code
        """
        lines = [
            "def on_event(self, event) -> None:",
            '    """Handle strategy events for position tracking and SL/TP submission."""',
            "    if isinstance(event, PositionOpened):",
            "        if event.instrument_id == self.instrument_id:",
            "            self._sync_position_state()",
            "            # Submit SL/TP using actual fill price from opened position",
            "            pos = self.cache.position(event.position_id)",
            "            if pos is not None:",
            "                entry_price = float(pos.avg_px_open)",
            "                self._submit_sl_tp_orders(entry_price, pos.entry, pos.quantity)",
            "    elif isinstance(event, PositionClosed):",
            "        if event.instrument_id == self.instrument_id:",
            "            self._position_open = False",
            "            self._position_side = None",
            "            self.cancel_all_orders(self.instrument_id)",
            "    elif isinstance(event, OrderFilled):",
            "        if event.instrument_id == self.instrument_id:",
            "            self._sync_position_state()",
        ]
        return "\n".join(lines)

    def _generate_on_stop(self) -> str:
        """Generate on_stop() method for clean shutdown.

        Returns:
            on_stop method source code
        """
        lines = [
            "def on_stop(self) -> None:",
            '    """Strategy shutdown: cancel orders and close positions."""',
            "    self.cancel_all_orders(self.instrument_id)",
            "    self.close_all_positions(self.instrument_id)",
        ]
        return "\n".join(lines)

    def _generate_on_start(
        self,
        dsl: StrategyDSL,
        indicators: list[IndicatorInfo],
        timeframes: set[str],
    ) -> str:
        """Generate on_start() method.

        Args:
            dsl: Parsed DSL
            indicators: List of indicator info
            timeframes: All timeframes used

        Returns:
            on_start method source code
        """
        lines = [
            "def on_start(self) -> None:",
            '    """Strategy startup: subscribe to bars and register indicators."""',
            "    # Resolve instrument",
            "    self.instrument_id = InstrumentId.from_str(self.config.instrument_id)",
            "    self.instrument = self.cache.instrument(self.instrument_id)",
            "    if self.instrument is None:",
            '        self.log.error(f"Instrument not found: {self.config.instrument_id}")',
            "        return",
            "",
            "    # Define bar types for all timeframes",
        ]

        # Generate bar type definitions
        tf_to_spec = {
            "1m": "1-MINUTE",
            "5m": "5-MINUTE",
            "15m": "15-MINUTE",
            "1h": "1-HOUR",
            "4h": "4-HOUR",
        }

        for tf in sorted(timeframes):
            spec = tf_to_spec.get(tf, "5-MINUTE")
            lines.append(f'    self.bar_type_{tf} = BarType.from_str(')
            lines.append(
                f'        f"{{self.instrument_id}}-{spec}-LAST-EXTERNAL"'
            )
            lines.append("    )")

        # Primary bar type
        lines.append("")
        lines.append(f"    self.primary_bar_type = self.bar_type_{dsl.timeframe}")
        lines.append("")

        # Subscribe to bars
        lines.append("    # Subscribe to bars for all timeframes")
        for tf in sorted(timeframes):
            lines.append(f"    self.subscribe_bars(self.bar_type_{tf})")

        lines.append("")
        lines.append("    # Initialize and register indicators")

        # Create indicators
        for info in indicators:
            lines.extend(self._generate_indicator_init(info))

        return "\n".join(lines)

    def _generate_indicator_init(self, info: IndicatorInfo) -> list[str]:
        """Generate indicator initialization code.

        Args:
            info: IndicatorInfo for the indicator

        Returns:
            Lines of code for indicator initialization
        """
        lines: list[str] = []
        config = info.config
        spec = info.spec

        if spec.nt_class is None:
            # TODO: Handle pandas-ta fallback
            lines.append(
                f"    # WARNING: {info.name} ({config.type}) has no NT class, using placeholder"
            )
            lines.append(f"    {info.indicator_var} = None")
            return lines

        class_name = spec.nt_class.__name__

        # Build constructor arguments
        args: list[str] = []

        # Map DSL params to NT params based on indicator type
        if config.type in {"RSI", "EMA", "SMA", "WMA", "DEMA", "TEMA", "ATR", "CCI", "ROC", "MFI"}:
            args.append(f"period=self.config.{info.name}_period")
        elif config.type == "MACD":
            args.append(f"fast_period=self.config.{info.name}_fast_period")
            args.append(f"slow_period=self.config.{info.name}_slow_period")
            # Note: NT MACD might have different param name
            args.append(f"signal_period=self.config.{info.name}_signal_period")
        elif config.type == "BBANDS":
            args.append(f"period=self.config.{info.name}_period")
            args.append(f"k=self.config.{info.name}_std_dev")
        elif config.type == "STOCH":
            args.append(f"period_k=self.config.{info.name}_period")
            args.append("period_d=3")  # Default D period
        elif config.type == "KC":
            args.append(f"period=self.config.{info.name}_period")
            args.append(f"k=self.config.{info.name}_atr_multiplier")
        elif config.type == "DONCHIAN":
            args.append(f"period=self.config.{info.name}_period")
        elif config.type in {"OBV", "VWAP"}:
            # No parameters
            pass

        args_str = ", ".join(args)
        lines.append(f"    {info.indicator_var} = {class_name}({args_str})")
        lines.append(
            f"    self.register_indicator_for_bars({info.bar_type_var}, {info.indicator_var})"
        )

        return lines

    def _generate_on_bar(
        self,
        dsl: StrategyDSL,
        indicators: list[IndicatorInfo],
        indicator_names: list[str],
    ) -> str:
        """Generate on_bar() method.

        Args:
            dsl: Parsed DSL
            indicators: List of indicator info
            indicator_names: List of indicator names

        Returns:
            on_bar method source code
        """
        lines = [
            "def on_bar(self, bar: Bar) -> None:",
            '    """Handle bar updates."""',
            "    # Only process primary timeframe bars",
            "    if bar.bar_type != self.primary_bar_type:",
            "        return",
            "",
            "    # Check if indicators are ready",
            "    if not self._indicators_ready():",
            "        return",
            "",
        ]

        # Time filters
        if dsl.time_filters.allowed_sessions or dsl.time_filters.blocked_days:
            lines.append("    # Check time filters")
            lines.append("    if not self._check_time_filters(bar.ts_event):")
            lines.append("        return")
            lines.append("")

        # Funding avoidance
        if dsl.time_filters.avoid_around_funding.enabled:
            lines.append("    # Check funding avoidance")
            lines.append("    if self._is_near_funding_time(bar.ts_event):")
            lines.append("        return")
            lines.append("")

        # Entry conditions
        lines.append("    # Evaluate entry conditions")
        lines.append("    if not self._position_open:")
        if dsl.entry_conditions.long:
            lines.append("        if self._check_long_entry(bar):")
            lines.append("            self._submit_long_entry(bar)")
        if dsl.entry_conditions.short:
            lines.append("        elif self._check_short_entry(bar):")
            lines.append("            self._submit_short_entry(bar)")

        # Exit conditions
        if dsl.exit_conditions.long or dsl.exit_conditions.short:
            lines.append("")
            lines.append("    # Evaluate exit conditions")
            lines.append("    if self._position_open:")
            if dsl.exit_conditions.long:
                lines.append("        if self._position_side == OrderSide.BUY:")
                lines.append("            if self._check_long_exit(bar):")
                lines.append("                self._submit_exit(bar)")
            if dsl.exit_conditions.short:
                if dsl.exit_conditions.long:
                    lines.append("        elif self._position_side == OrderSide.SELL:")
                else:
                    lines.append("        if self._position_side == OrderSide.SELL:")
                lines.append("            if self._check_short_exit(bar):")
                lines.append("                self._submit_exit(bar)")

        # Update previous values for crossover detection
        lines.append("")
        lines.append("    # Update previous values for crossover detection")
        lines.append("    self._update_prev_values()")

        return "\n".join(lines)

    def _generate_helper_methods(
        self,
        dsl: StrategyDSL,
        indicators: list[IndicatorInfo],
        indicator_names: list[str],
    ) -> str:
        """Generate helper methods for the strategy.

        Args:
            dsl: Parsed DSL
            indicators: List of indicator info
            indicator_names: List of indicator names

        Returns:
            Helper methods source code
        """
        lines: list[str] = []

        # _indicators_ready
        lines.extend([
            "def _indicators_ready(self) -> bool:",
            '    """Check if all indicators have enough data."""',
        ])
        for info in indicators:
            if info.spec.nt_class is not None:
                lines.append(f"    if not {info.indicator_var}.initialized:")
                lines.append("        return False")
        lines.append("    return True")
        lines.append("")

        # _get_indicator_value
        lines.extend(self._generate_get_indicator_value(indicators))
        lines.append("")

        # _update_prev_values
        lines.extend(self._generate_update_prev_values(indicators))
        lines.append("")

        # Time filter method
        lines.extend(self._generate_time_filter_method(dsl.time_filters))
        lines.append("")

        # Funding avoidance method
        if dsl.time_filters.avoid_around_funding.enabled:
            lines.extend(
                self._generate_funding_avoidance_method(dsl.time_filters.avoid_around_funding)
            )
            lines.append("")

        # Condition check methods
        if dsl.entry_conditions.long:
            lines.extend(
                self._generate_condition_check_method(
                    "_check_long_entry",
                    dsl.entry_conditions.long,
                    indicator_names,
                )
            )
            lines.append("")

        if dsl.entry_conditions.short:
            lines.extend(
                self._generate_condition_check_method(
                    "_check_short_entry",
                    dsl.entry_conditions.short,
                    indicator_names,
                )
            )
            lines.append("")

        if dsl.exit_conditions.long:
            lines.extend(
                self._generate_condition_check_method(
                    "_check_long_exit",
                    dsl.exit_conditions.long,
                    indicator_names,
                )
            )
            lines.append("")

        if dsl.exit_conditions.short:
            lines.extend(
                self._generate_condition_check_method(
                    "_check_short_exit",
                    dsl.exit_conditions.short,
                    indicator_names,
                )
            )
            lines.append("")

        # Order submission methods
        lines.extend(self._generate_order_methods())

        return "\n".join(lines)

    def _generate_get_indicator_value(
        self, indicators: list[IndicatorInfo]
    ) -> list[str]:
        """Generate _get_indicator_value method.

        Args:
            indicators: List of indicator info

        Returns:
            Method source code as lines
        """
        lines = [
            "def _get_indicator_value(self, name: str) -> float:",
            '    """Get current value of an indicator by name."""',
        ]

        for info in indicators:
            lines.append(f'    if name == "{info.name}":')
            if info.spec.nt_class is not None:
                lines.append(f"        return float({info.indicator_var}.value)")
            else:
                lines.append("        return 0.0  # Placeholder")

        lines.append('    raise ValueError(f"Unknown indicator: {name}")')
        return lines

    def _generate_update_prev_values(self, indicators: list[IndicatorInfo]) -> list[str]:
        """Generate _update_prev_values method.

        Args:
            indicators: List of indicator info

        Returns:
            Method source code as lines
        """
        lines = [
            "def _update_prev_values(self) -> None:",
            '    """Store current indicator values for crossover detection."""',
        ]

        for info in indicators:
            if info.spec.nt_class is not None:
                lines.append(
                    f'    self._prev_values["{info.name}"] = float({info.indicator_var}.value)'
                )

        if not any(info.spec.nt_class is not None for info in indicators):
            lines.append("    pass")

        return lines

    def _generate_time_filter_method(self, time_filters: TimeFilterConfig) -> list[str]:
        """Generate _check_time_filters method.

        Args:
            time_filters: Time filter configuration

        Returns:
            Method source code as lines
        """
        lines = [
            "def _check_time_filters(self, ts_ns: int) -> bool:",
            '    """Check if current time passes time filters."""',
            "    from datetime import datetime, timezone",
            "",
            "    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)",
            "",
        ]

        # Blocked days check
        if time_filters.blocked_days:
            day_map = {
                "Monday": 0,
                "Tuesday": 1,
                "Wednesday": 2,
                "Thursday": 3,
                "Friday": 4,
                "Saturday": 5,
                "Sunday": 6,
            }
            blocked_nums = [day_map[d] for d in time_filters.blocked_days]
            lines.append(f"    blocked_days = {blocked_nums}")
            lines.append("    if dt.weekday() in blocked_days:")
            lines.append("        return False")
            lines.append("")

        # Session check
        if time_filters.allowed_sessions:
            lines.append("    # Check allowed sessions")
            lines.append("    in_session = False")
            for i, session in enumerate(time_filters.allowed_sessions):
                lines.extend(self._generate_session_check(session, i))
            lines.append("    if not in_session:")
            lines.append("        return False")
            lines.append("")

        lines.append("    return True")
        return lines

    def _generate_session_check(self, session: SessionConfig, index: int) -> list[str]:
        """Generate code to check a single session.

        Args:
            session: Session configuration
            index: Session index for variable naming

        Returns:
            Code lines for session check
        """
        lines = []
        start_h, start_m = session.start.split(":")
        end_h, end_m = session.end.split(":")

        lines.append(f"    # Session {index + 1}: {session.start}-{session.end} {session.timezone}")
        if session.timezone != "UTC":
            lines.append(f'    tz_{index} = zoneinfo.ZoneInfo("{session.timezone}")')
            lines.append(f"    local_dt_{index} = dt.astimezone(tz_{index})")
            lines.append(f"    local_time_{index} = local_dt_{index}.time()")
        else:
            lines.append(f"    local_time_{index} = dt.time()")

        lines.append(f"    session_start_{index} = dt_time({int(start_h)}, {int(start_m)})")
        lines.append(f"    session_end_{index} = dt_time({int(end_h)}, {int(end_m)})")
        lines.append(f"    if session_start_{index} <= session_end_{index}:")
        lines.append(
            f"        if session_start_{index} <= local_time_{index} <= session_end_{index}:"
        )
        lines.append("            in_session = True")
        lines.append("    else:")
        lines.append("        # Overnight session (start > end)")
        lines.append(
            f"        if local_time_{index} >= session_start_{index} or local_time_{index} <= session_end_{index}:"
        )
        lines.append("            in_session = True")

        return lines

    def _generate_funding_avoidance_method(
        self, funding_config: object
    ) -> list[str]:
        """Generate _is_near_funding_time method.

        Args:
            funding_config: FundingAvoidanceConfig

        Returns:
            Method source code as lines
        """
        from vibe_quant.dsl.schema import FundingAvoidanceConfig

        cfg = funding_config if isinstance(funding_config, FundingAvoidanceConfig) else None
        minutes_before = cfg.minutes_before if cfg else 5
        minutes_after = cfg.minutes_after if cfg else 5

        lines = [
            "def _is_near_funding_time(self, ts_ns: int) -> bool:",
            '    """Check if near funding settlement (Binance: 00:00, 08:00, 16:00 UTC)."""',
            "    from datetime import datetime, timezone",
            "",
            "    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)",
            "    hour = dt.hour",
            "    minute = dt.minute",
            "",
            "    # Binance funding times: 00:00, 08:00, 16:00 UTC",
            "    funding_hours = [0, 8, 16]",
            "",
            "    for fh in funding_hours:",
            f"        minutes_before = {minutes_before}",
            f"        minutes_after = {minutes_after}",
            "",
            "        # Check if within window before funding",
            "        if hour == fh and minute < minutes_after:",
            "            return True",
            "        # Check if within window before funding (previous hour)",
            "        prev_hour = (fh - 1) % 24",
            "        if hour == prev_hour and minute >= (60 - minutes_before):",
            "            return True",
            "",
            "    return False",
        ]
        return lines

    def _generate_condition_check_method(
        self,
        method_name: str,
        conditions: list[str],
        indicator_names: list[str],
    ) -> list[str]:
        """Generate a condition check method.

        Args:
            method_name: Name of the method to generate
            conditions: List of condition strings
            indicator_names: Valid indicator names

        Returns:
            Method source code as lines
        """
        lines = [
            f"def {method_name}(self, bar: Bar) -> bool:",
            f'    """Check {method_name.replace("_", " ")} conditions."""',
        ]

        for i, cond_str in enumerate(conditions):
            cond = parse_condition(cond_str, indicator_names)
            code = self._generate_condition_code(cond, i)
            lines.append(f"    # Condition: {cond_str}")
            lines.append(f"    {code}")
            lines.append(f"    if not cond_{i}:")
            lines.append("        return False")
            lines.append("")

        lines.append("    return True")
        return lines

    def _generate_condition_code(self, cond: Condition, index: int) -> str:
        """Generate Python code for a single condition.

        Args:
            cond: Parsed Condition object
            index: Condition index for variable naming

        Returns:
            Python code for the condition check
        """
        left = self._operand_to_code(cond.left)
        right = self._operand_to_code(cond.right)

        if cond.operator == Operator.GT:
            return f"cond_{index} = {left} > {right}"
        elif cond.operator == Operator.LT:
            return f"cond_{index} = {left} < {right}"
        elif cond.operator == Operator.GTE:
            return f"cond_{index} = {left} >= {right}"
        elif cond.operator == Operator.LTE:
            return f"cond_{index} = {left} <= {right}"
        elif cond.operator == Operator.CROSSES_ABOVE:
            prev_left = self._operand_to_prev_code(cond.left)
            prev_right = self._operand_to_prev_code(cond.right)
            return (
                f"cond_{index} = ({left} > {right}) and ({prev_left} <= {prev_right})"
            )
        elif cond.operator == Operator.CROSSES_BELOW:
            prev_left = self._operand_to_prev_code(cond.left)
            prev_right = self._operand_to_prev_code(cond.right)
            return (
                f"cond_{index} = ({left} < {right}) and ({prev_left} >= {prev_right})"
            )
        elif cond.operator == Operator.BETWEEN:
            right2 = self._operand_to_code(cond.right2) if cond.right2 else "0"
            return f"cond_{index} = {right} <= {left} <= {right2}"
        else:
            return f"cond_{index} = False  # Unknown operator"

    def _operand_to_code(self, operand: object) -> str:
        """Convert an Operand to Python code.

        Args:
            operand: Operand object

        Returns:
            Python code string
        """
        from vibe_quant.dsl.conditions import Operand

        if not isinstance(operand, Operand):
            return "0.0"

        if operand.is_price:
            # Price references need bar data
            return f"float(bar.{operand.value}.as_double())" if operand.value != "volume" else "float(bar.volume.as_double())"
        elif operand.is_indicator:
            return f'self._get_indicator_value("{operand.value}")'
        else:
            # Literal value
            return str(operand.value)

    def _operand_to_prev_code(self, operand: object) -> str:
        """Convert an Operand to Python code for previous value.

        Args:
            operand: Operand object

        Returns:
            Python code string for previous value
        """
        from vibe_quant.dsl.conditions import Operand

        if not isinstance(operand, Operand):
            return "0.0"

        if operand.is_indicator:
            return f'self._prev_values.get("{operand.value}", 0.0)'
        else:
            # Literals and prices don't have previous values
            return self._operand_to_code(operand)

    def _generate_order_methods(self) -> list[str]:
        """Generate order submission methods with SL/TP and event-based tracking.

        Returns:
            Order method source code as lines
        """
        lines = [
            # _submit_long_entry
            "def _submit_long_entry(self, bar: Bar) -> None:",
            '    """Submit a long entry order with SL/TP."""',
            "    if self._position_open:",
            "        return",
            "",
            "    qty = self._calculate_position_size(bar, is_long=True)",
            "    order = self.order_factory.market(",
            "        instrument_id=self.instrument_id,",
            "        order_side=OrderSide.BUY,",
            "        quantity=qty,",
            "        time_in_force=TimeInForce.IOC,",
            "    )",
            "    self.submit_order(order)",
            "    # SL/TP orders are submitted from on_event(PositionOpened) after fill",
            "",
            # _submit_short_entry
            "def _submit_short_entry(self, bar: Bar) -> None:",
            '    """Submit a short entry order with SL/TP."""',
            "    if self._position_open:",
            "        return",
            "",
            "    qty = self._calculate_position_size(bar, is_long=False)",
            "    order = self.order_factory.market(",
            "        instrument_id=self.instrument_id,",
            "        order_side=OrderSide.SELL,",
            "        quantity=qty,",
            "        time_in_force=TimeInForce.IOC,",
            "    )",
            "    self.submit_order(order)",
            "    # SL/TP orders are submitted from on_event(PositionOpened) after fill",
            "",
            # _submit_exit
            "def _submit_exit(self, bar: Bar) -> None:",
            '    """Submit an exit order using actual position quantity from cache."""',
            "    if not self._position_open:",
            "        return",
            "",
            "    # Cancel existing SL/TP orders",
            "    self.cancel_all_orders(self.instrument_id)",
            "",
            "    # Determine exit side and get actual position quantity from cache",
            "    exit_side = OrderSide.SELL if self._position_side == OrderSide.BUY else OrderSide.BUY",
            "    quantity = self._calculate_position_size(bar)",
            "    positions = self.cache.positions_open(venue=self.instrument_id.venue)",
            "    for pos in positions:",
            "        if pos.instrument_id == self.instrument_id and pos.is_open:",
            "            quantity = pos.quantity",
            "            break",
            "",
            "    order = self.order_factory.market(",
            "        instrument_id=self.instrument_id,",
            "        order_side=exit_side,",
            "        quantity=quantity,",
            "        time_in_force=TimeInForce.IOC,",
            "    )",
            "    self.submit_order(order)",
            "",
            # _submit_sl_tp_orders
            "def _submit_sl_tp_orders(self, entry_price: float, side: OrderSide, qty: Quantity) -> None:",
            '    """Submit stop-loss and take-profit orders after entry."""',
            "    is_long = side == OrderSide.BUY",
            "",
            "    # Calculate and submit stop-loss order",
            "    sl_price = self._calculate_sl_price(entry_price, is_long)",
            "    if sl_price is not None:",
            "        sl_side = OrderSide.SELL if is_long else OrderSide.BUY",
            "        sl_order = self.order_factory.stop_market(",
            "            instrument_id=self.instrument_id,",
            "            order_side=sl_side,",
            "            quantity=qty,",
            "            trigger_price=self.instrument.make_price(sl_price),",
            "            time_in_force=TimeInForce.GTC,",
            "            reduce_only=True,",
            "        )",
            "        self.submit_order(sl_order)",
            "",
            "    # Calculate and submit take-profit order",
            "    tp_price = self._calculate_tp_price(entry_price, is_long)",
            "    if tp_price is not None:",
            "        tp_side = OrderSide.SELL if is_long else OrderSide.BUY",
            "        tp_order = self.order_factory.limit(",
            "            instrument_id=self.instrument_id,",
            "            order_side=tp_side,",
            "            quantity=qty,",
            "            price=self.instrument.make_price(tp_price),",
            "            time_in_force=TimeInForce.GTC,",
            "            reduce_only=True,",
            "        )",
            "        self.submit_order(tp_order)",
            "",
            # _calculate_sl_price
            "def _calculate_sl_price(self, entry_price: float, is_long: bool) -> float | None:",
            '    """Calculate stop-loss price based on config type."""',
            "    sl_type = self.config.stop_loss_type",
            '    if sl_type == "fixed_pct":',
            "        pct = self.config.stop_loss_percent",
            "        if is_long:",
            "            return entry_price * (1 - pct / 100)",
            "        else:",
            "            return entry_price * (1 + pct / 100)",
            '    elif sl_type in ("atr_fixed", "atr_trailing"):',
            "        atr_value = self._get_indicator_value(self.config.stop_loss_indicator)",
            "        multiplier = self.config.stop_loss_atr_multiplier",
            "        if is_long:",
            "            return entry_price - atr_value * multiplier",
            "        else:",
            "            return entry_price + atr_value * multiplier",
            "    return None",
            "",
            # _calculate_tp_price
            "def _calculate_tp_price(self, entry_price: float, is_long: bool) -> float | None:",
            '    """Calculate take-profit price based on config type."""',
            "    tp_type = self.config.take_profit_type",
            '    if tp_type == "fixed_pct":',
            "        pct = self.config.take_profit_percent",
            "        if is_long:",
            "            return entry_price * (1 + pct / 100)",
            "        else:",
            "            return entry_price * (1 - pct / 100)",
            '    elif tp_type == "atr_fixed":',
            "        atr_value = self._get_indicator_value(self.config.take_profit_indicator)",
            "        multiplier = self.config.take_profit_atr_multiplier",
            "        if is_long:",
            "            return entry_price + atr_value * multiplier",
            "        else:",
            "            return entry_price - atr_value * multiplier",
            '    elif tp_type == "risk_reward":',
            "        sl_price = self._calculate_sl_price(entry_price, is_long)",
            "        if sl_price is not None:",
            "            sl_distance = abs(entry_price - sl_price)",
            "            ratio = self.config.take_profit_risk_reward",
            "            if is_long:",
            "                return entry_price + sl_distance * ratio",
            "            else:",
            "                return entry_price - sl_distance * ratio",
            "    return None",
            "",
            # _calculate_position_size
            "def _calculate_position_size(self, bar: Bar, is_long: bool = True) -> Quantity:",
            '    """Calculate position size based on risk config."""',
            "    # Get account equity from cache",
            "    account = self.cache.account_for_venue(self.instrument_id.venue)",
            "    if account is None:",
            "        return self.instrument.make_qty(1.0)",
            "",
            "    equity = float(account.balance_total(account.currencies()[0]))",
            "    if equity <= 0:",
            "        return self.instrument.make_qty(1.0)",
            "",
            "    price = float(bar.close)",
            "    if price <= 0:",
            "        return self.instrument.make_qty(1.0)",
            "",
            "    # Fixed fractional sizing: risk_per_trade % of equity",
            "    risk_pct = getattr(self.config, 'risk_per_trade', 0.02)",
            "    risk_amount = equity * risk_pct",
            "",
            "    # Use stop loss distance if available, otherwise use 2% of price",
            "    sl_price = self._calculate_sl_price(price, is_long)",
            "    if sl_price is not None and sl_price > 0:",
            "        stop_distance = abs(price - sl_price)",
            "    else:",
            "        stop_distance = price * 0.02  # 2% default stop distance",
            "",
            "    if stop_distance <= 0:",
            "        stop_distance = price * 0.02",
            "",
            "    # Size = risk_amount / stop_distance",
            "    raw_size = risk_amount / stop_distance",
            "",
            "    # Apply max position limit (50% of equity at entry price)",
            "    max_size = (equity * 0.5) / price",
            "    final_size = min(raw_size, max_size)",
            "",
            "    # Ensure minimum quantity: 1% of equity at current price",
            "    min_size = (equity * 0.01) / price if price > 0 else 0.001",
            "    if final_size <= 0:",
            "        final_size = min_size",
            "",
            "    # Clamp to instrument minimums",
            "    final_size = max(final_size, float(self.instrument.min_quantity))",
            "    return self.instrument.make_qty(final_size)",
            "",
            # _sync_position_state
            "def _sync_position_state(self) -> None:",
            '    """Sync position tracking state from cache."""',
            "    positions = self.cache.positions_open(venue=self.instrument_id.venue)",
            "    for pos in positions:",
            "        if pos.instrument_id == self.instrument_id and pos.is_open:",
            "            self._position_open = True",
            "            self._position_side = pos.entry",
            "            return",
            "    self._position_open = False",
            "    self._position_side = None",
        ]
        return lines


def _to_class_name(snake_case: str) -> str:
    """Convert snake_case to PascalCase.

    Args:
        snake_case: String in snake_case format

    Returns:
        String in PascalCase format
    """
    return "".join(word.capitalize() for word in snake_case.split("_"))
