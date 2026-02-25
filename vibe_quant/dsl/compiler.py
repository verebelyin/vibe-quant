"""DSL-to-NautilusTrader Strategy compiler.

Compiles parsed StrategyDSL into NautilusTrader Strategy subclass Python source code.
Generates on_start() with multi-TF subscriptions, indicator registration,
and on_bar() with time filter evaluation, condition checking, and order submission.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from vibe_quant.dsl.conditions import Condition, Operator, parse_condition
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.templates import ON_EVENT_LINES, ON_STOP_LINES, ORDER_METHODS_LINES

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from types import ModuleType

    from vibe_quant.dsl.schema import (
        IndicatorConfig,
        SessionConfig,
        StrategyDSL,
        TimeFilterConfig,
    )

    pass


_ALLOWED_IMPORT_PREFIXES: tuple[str, ...] = (
    "__future__",
    "datetime",
    "typing",
    "zoneinfo",
    "nautilus_trader",
    "pandas",
    "pandas_ta_classic",
    "vibe_quant",
)

_BLOCKED_CALL_NAMES: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "open",
        "input",
    }
)

_BLOCKED_ATTR_CALLS: frozenset[tuple[str, str]] = frozenset(
    {
        ("os", "system"),
        ("os", "popen"),
        ("subprocess", "run"),
        ("subprocess", "Popen"),
    }
)


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
        # Maps (indicator_name, literal_value) → config threshold field name
        # Built during compile() for use in condition code generation
        self._threshold_map: dict[tuple[str, float], str] = {}

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

        # Force MACD to pandas-ta when signal/histogram outputs are used in conditions
        # (NT MACD only exposes .value = MACD line, not signal or histogram)
        all_condition_text = " ".join(
            dsl.entry_conditions.long
            + dsl.entry_conditions.short
            + dsl.exit_conditions.long
            + dsl.exit_conditions.short
        )
        for info in indicators:
            if (
                info.config.type == "MACD"
                and info.spec.nt_class is not None
                and info.spec.pandas_ta_func is not None
                and (
                    f"{info.name}.signal" in all_condition_text
                    or f"{info.name}.histogram" in all_condition_text
                    or f"{info.name}_signal" in all_condition_text
                    or f"{info.name}_histogram" in all_condition_text
                )
            ):
                from dataclasses import replace as _dc_replace

                # IndicatorInfo is frozen; rebuild with replaced spec
                new_spec = _dc_replace(info.spec, nt_class=None)
                indicators[indicators.index(info)] = _dc_replace(info, spec=new_spec)
                logger.info(
                    "MACD '%s' forced to pandas-ta: signal/histogram outputs referenced",
                    info.name,
                )

        # Add sub-output names for multi-output indicators (e.g., bbands_upper)
        for info in indicators:
            if info.spec.output_names != ("value",):
                for output_name in info.spec.output_names:
                    indicator_names.append(f"{info.name}_{output_name}")

        # Gather all timeframes
        timeframes = self._get_all_timeframes(dsl)

        # Generate parts
        imports = self._generate_imports(dsl, indicators)
        config_class = self._generate_config_class(dsl, indicator_names)
        strategy_class = self._generate_strategy_class(dsl, indicators, timeframes, indicator_names)

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
        self._validate_generated_source(source)
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

    def _validate_generated_source(self, source: str) -> None:
        """Validate generated source before dynamic execution."""
        try:
            tree = ast.parse(source, filename="<dsl-generated>", mode="exec")
        except SyntaxError as e:
            msg = f"Generated source failed AST parse: {e}"
            raise CompilerError(msg) from e

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith(_ALLOWED_IMPORT_PREFIXES):
                        msg = f"Disallowed import in generated source: {alias.name}"
                        raise CompilerError(msg)
            elif isinstance(node, ast.ImportFrom):
                if node.level != 0:
                    msg = "Relative imports are not allowed in generated source"
                    raise CompilerError(msg)
                module_name = node.module or ""
                if not module_name.startswith(_ALLOWED_IMPORT_PREFIXES):
                    msg = f"Disallowed import in generated source: {module_name}"
                    raise CompilerError(msg)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALL_NAMES:
                    msg = f"Unsafe call in generated source: {node.func.id}"
                    raise CompilerError(msg)
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    root_name = node.func.value.id
                    attr_name = node.func.attr
                    if (root_name, attr_name) in _BLOCKED_ATTR_CALLS:
                        msg = f"Unsafe call in generated source: {root_name}.{attr_name}"
                        raise CompilerError(msg)

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

    def _generate_imports(self, dsl: StrategyDSL, indicators: list[IndicatorInfo]) -> str:
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
            "from nautilus_trader.model.enums import OrderSide, PositionSide, TimeInForce",
            "from nautilus_trader.model.identifiers import InstrumentId",
            "from nautilus_trader.model.instruments import Instrument",
            "from nautilus_trader.model.objects import Price, Quantity",
            "from nautilus_trader.model.events import OrderFilled, PositionOpened, PositionClosed",
            "from nautilus_trader.model.orders import LimitOrder, MarketOrder, StopMarketOrder",
            "from nautilus_trader.trading.strategy import Strategy, StrategyConfig",
        ]

        # Collect unique indicator classes
        nt_classes: dict[str, str] = {}  # class_name -> module_path
        has_pta = False
        for info in indicators:
            if info.spec.nt_class is not None:
                class_name = info.spec.nt_class.__name__
                module_path = info.spec.nt_class.__module__
                nt_classes[class_name] = module_path
            elif info.spec.pandas_ta_func is not None:
                has_pta = True

        # Add indicator imports
        if nt_classes:
            imports.append("")
            imports.append("# Indicator imports")
            for class_name, module_path in sorted(nt_classes.items()):
                imports.append(f"from {module_path} import {class_name}")

        # Add pandas-ta imports for fallback indicators
        if has_pta:
            imports.append("")
            imports.append("# pandas-ta-classic fallback for indicators without NT class")
            imports.append("import pandas as pd")
            imports.append("import pandas_ta_classic as ta")

        imports.append("")
        imports.append("if TYPE_CHECKING:")
        imports.append("    pass")

        return "\n".join(imports)

    def _generate_config_class(
        self, dsl: StrategyDSL, indicator_names: list[str] | None = None
    ) -> str:
        """Generate the Strategy config dataclass.

        Args:
            dsl: Parsed DSL
            indicator_names: Expanded indicator names (includes sub-outputs)

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
        lines.append(
            "    max_position_pct: float = 0.5  # Max position as fraction of equity (0.5=50%, 2.0=2x leverage)"
        )
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
            if config.d_period is not None:
                lines.append(f"    {name}_d_period: int = {config.d_period}")
            if config.atr_multiplier is not None:
                lines.append(f"    {name}_atr_multiplier: float = {config.atr_multiplier}")

        # Add stop loss parameters
        lines.append("")
        lines.append("    # Stop loss parameters")
        lines.append(f'    stop_loss_type: str = "{dsl.stop_loss.type}"')
        if dsl.stop_loss.percent is not None:
            lines.append(f"    stop_loss_percent: float = {dsl.stop_loss.percent}")
        if dsl.stop_loss.atr_multiplier is not None:
            lines.append(f"    stop_loss_atr_multiplier: float = {dsl.stop_loss.atr_multiplier}")
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
            lines.append(f'    take_profit_indicator: str = "{dsl.take_profit.indicator}"')

        # Add custom thresholds (extracted from conditions)
        lines.append("")
        lines.append("    # Condition thresholds (can be overridden)")
        seen_thresholds: dict[str, float | int] = {}
        self._threshold_map = {}
        threshold_counter = 0
        for cond_str in (
            dsl.entry_conditions.long
            + dsl.entry_conditions.short
            + dsl.exit_conditions.long
            + dsl.exit_conditions.short
        ):
            cond = parse_condition(cond_str, indicator_names or list(dsl.indicators.keys()))
            if (
                not cond.right.is_indicator
                and not cond.right.is_price
                and isinstance(cond.right.value, (int, float))
            ):
                left_name = str(cond.left.value) if cond.left.is_indicator else str(cond.left.value)
                value_str = str(cond.right.value).replace(".", "_").replace("-", "neg_")
                short_name = f"{left_name}_{value_str}_threshold"
                if short_name not in seen_thresholds:
                    seen_thresholds[short_name] = cond.right.value
                    self._threshold_map[(left_name, float(cond.right.value))] = short_name
                elif seen_thresholds[short_name] != cond.right.value:
                    # Disambiguate with counter to avoid collisions when 3+
                    # conditions use the same indicator with different values
                    threshold_counter += 1
                    unique_name = f"{left_name}_{value_str}_{threshold_counter}_threshold"
                    if unique_name not in seen_thresholds:
                        seen_thresholds[unique_name] = cond.right.value
                    self._threshold_map[(left_name, float(cond.right.value))] = unique_name

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
            "",
            "        # Position tracking",
            "        self._position_open = False",
            "        self._position_side: OrderSide | None = None",
            "",
            "        # Previous indicator values for crossover detection",
            "        self._prev_values: dict[str, float] = {}",
            "",
        ]

        # Add pandas-ta bar buffer if any indicators need it
        has_pta = any(
            i.spec.nt_class is None and i.spec.pandas_ta_func is not None for i in indicators
        )
        if has_pta:
            lines.extend(
                [
                    "        # Bar data buffer for pandas-ta indicators",
                    "        self._pta_close: list[float] = []",
                    "        self._pta_high: list[float] = []",
                    "        self._pta_low: list[float] = []",
                    "        self._pta_open: list[float] = []",
                    "        self._pta_volume: list[float] = []",
                    "        self._pta_values: dict[str, float] = {}",
                    "",
                ]
            )

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
        return "\n".join(ON_EVENT_LINES)

    def _generate_on_stop(self) -> str:
        """Generate on_stop() method for clean shutdown.

        Returns:
            on_stop method source code
        """
        return "\n".join(ON_STOP_LINES)

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
            lines.append(f"    self.bar_type_{tf} = BarType.from_str(")
            lines.append(f'        f"{{self.instrument_id}}-{spec}-LAST-EXTERNAL"')
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

        if spec.nt_class is None and spec.pandas_ta_func is not None:
            # pandas-ta fallback: store config for _update_pta_indicators()
            lines.append(
                f"    # {info.name} ({config.type}): pandas-ta fallback via ta.{spec.pandas_ta_func}"
            )
            return lines

        assert spec.nt_class is not None  # guarded by caller check
        class_name = spec.nt_class.__name__

        # Build constructor arguments
        args: list[str] = []

        # Map DSL params to NT params based on indicator type
        if config.type in {"RSI", "EMA", "SMA", "WMA", "DEMA", "TEMA", "ATR", "CCI", "ROC", "MFI"}:
            args.append(f"period=self.config.{info.name}_period")
        elif config.type == "MACD":
            args.append(f"fast_period=self.config.{info.name}_fast_period")
            args.append(f"slow_period=self.config.{info.name}_slow_period")
            # NT MACD does not accept signal_period
        elif config.type == "BBANDS":
            args.append(f"period=self.config.{info.name}_period")
            args.append(f"k=self.config.{info.name}_std_dev")
        elif config.type == "STOCH":
            args.append(f"period_k=self.config.{info.name}_period")
            args.append(f"period_d=self.config.{info.name}_d_period")
        elif config.type == "KC":
            args.append(f"period=self.config.{info.name}_period")
            args.append(f"k_multiplier=self.config.{info.name}_atr_multiplier")
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
        has_pta = any(
            i.spec.nt_class is None and i.spec.pandas_ta_func is not None for i in indicators
        )

        lines = [
            "def on_bar(self, bar: Bar) -> None:",
            '    """Handle bar updates."""',
            "    # Only process primary timeframe bars",
            "    if bar.bar_type != self.primary_bar_type:",
            "        return",
            "",
        ]

        # Feed bar data to pandas-ta buffer before indicators_ready check
        if has_pta:
            lines.extend(
                [
                    "    # Feed bar data to pandas-ta buffer",
                    "    self._pta_close.append(float(bar.close))",
                    "    self._pta_high.append(float(bar.high))",
                    "    self._pta_low.append(float(bar.low))",
                    "    self._pta_open.append(float(bar.open))",
                    "    self._pta_volume.append(float(bar.volume))",
                    "    self._update_pta_indicators()",
                    "",
                ]
            )

        lines.extend(
            [
                "    # Check if indicators are ready",
                "    if not self._indicators_ready():",
                "        return",
                "",
            ]
        )

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
            if dsl.entry_conditions.long:
                lines.append("        elif self._check_short_entry(bar):")
            else:
                lines.append("        if self._check_short_entry(bar):")
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

        # Trailing stop update
        if dsl.stop_loss.type == "atr_trailing":
            lines.append("")
            lines.append("    # Update trailing stop loss")
            lines.append("    self._update_trailing_stop(bar)")

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
        lines.extend(
            [
                "def _indicators_ready(self) -> bool:",
                '    """Check if all indicators have enough data."""',
            ]
        )
        for info in indicators:
            if info.spec.nt_class is not None:
                lines.append(f"    if not {info.indicator_var}.initialized:")
                lines.append("        return False")
            elif info.spec.pandas_ta_func is not None:
                # Check pandas-ta indicator has computed a value
                lines.append(f'    if "{info.name}" not in self._pta_values:')
                lines.append("        return False")
                # Also check multi-output sub-names
                if info.spec.output_names != ("value",):
                    for output_name in info.spec.output_names:
                        lines.append(f'    if "{info.name}_{output_name}" not in self._pta_values:')
                        lines.append("        return False")
        lines.append("    return True")
        lines.append("")

        # _get_indicator_value
        lines.extend(self._generate_get_indicator_value(indicators))
        lines.append("")

        # _update_prev_values
        lines.extend(self._generate_update_prev_values(indicators))
        lines.append("")

        # _update_pta_indicators (if any pandas-ta indicators exist)
        pta_indicators = [
            i for i in indicators if i.spec.nt_class is None and i.spec.pandas_ta_func is not None
        ]
        if pta_indicators:
            lines.extend(self._generate_update_pta_indicators(pta_indicators))
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

    def _generate_get_indicator_value(self, indicators: list[IndicatorInfo]) -> list[str]:
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
            if info.spec.nt_class is None and info.spec.pandas_ta_func is not None:
                # pandas-ta fallback: read from _pta_values buffer
                lines.append(f'    if name == "{info.name}":')
                lines.append(f'        return self._pta_values.get("{info.name}", 0.0)')
                # Multi-output sub-names
                if info.spec.output_names != ("value",):
                    for output_name in info.spec.output_names:
                        lines.append(f'    if name == "{info.name}_{output_name}":')
                        lines.append(
                            f'        return self._pta_values.get("{info.name}_{output_name}", 0.0)'
                        )
                continue

            nt_attr = self._get_default_output_attr(info)
            lines.append(f'    if name == "{info.name}":')
            lines.append(f"        return float({info.indicator_var}.{nt_attr})")

            # Register sub-names for each output of multi-output indicators
            if info.spec.output_names != ("value",):
                for output_name in info.spec.output_names:
                    attr = self._output_to_nt_attr(info.config.type, output_name)
                    if info.config.type == "MACD" and output_name in ("signal", "histogram"):
                        logger.warning(
                            "MACD %s output '%s' not available in NT — "
                            "returns MACD line (.value) instead. "
                            "Use pandas-ta fallback for signal/histogram.",
                            info.name,
                            output_name,
                        )
                    lines.append(f'    if name == "{info.name}_{output_name}":')
                    lines.append(f"        return float({info.indicator_var}.{attr})")

        lines.append('    raise ValueError(f"Unknown indicator: {name}")')
        return lines

    @staticmethod
    def _output_to_nt_attr(indicator_type: str, output_name: str) -> str:
        """Map DSL output name to NautilusTrader attribute name."""
        # MACD: NT only exposes .value (the MACD line = fast_ema - slow_ema).
        # Signal line and histogram are NOT available in NT's MACD class.
        if indicator_type == "MACD":
            if output_name == "macd":
                return "value"
            # signal/histogram not available — map to value to prevent crash,
            # but log warning at codegen time (caller should prefer pandas-ta fallback)
            return "value"
        # STOCH outputs use value_ prefix in NT
        if indicator_type == "STOCH":
            return f"value_{output_name}"  # k -> value_k, d -> value_d
        # BBANDS/KC/DONCHIAN: output name matches NT attr directly
        return output_name  # upper -> upper, middle -> middle, lower -> lower

    @staticmethod
    def _get_default_output_attr(info: IndicatorInfo) -> str:
        """Get the default NT attribute for an indicator's primary output."""
        if info.spec.output_names == ("value",):
            return "value"
        # Channel indicators: default to middle band
        if info.config.type in {"BBANDS", "KC", "DONCHIAN"}:
            return "middle"
        # Stochastics: default to K line
        if info.config.type == "STOCH":
            return "value_k"
        # MACD: only has .value in NT
        if info.config.type == "MACD":
            return "value"
        return "value"

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

        has_any = False
        for info in indicators:
            if info.spec.nt_class is not None:
                has_any = True
                nt_attr = self._get_default_output_attr(info)
                lines.append(
                    f'    self._prev_values["{info.name}"] = float({info.indicator_var}.{nt_attr})'
                )
                # Also store sub-outputs for multi-output indicators
                if info.spec.output_names != ("value",):
                    for output_name in info.spec.output_names:
                        attr = self._output_to_nt_attr(info.config.type, output_name)
                        lines.append(
                            f'    self._prev_values["{info.name}_{output_name}"] = float({info.indicator_var}.{attr})'
                        )
            elif info.spec.pandas_ta_func is not None:
                has_any = True
                lines.append(
                    f'    self._prev_values["{info.name}"] = self._pta_values.get("{info.name}", 0.0)'
                )
                if info.spec.output_names != ("value",):
                    for output_name in info.spec.output_names:
                        lines.append(
                            f'    self._prev_values["{info.name}_{output_name}"] = self._pta_values.get("{info.name}_{output_name}", 0.0)'
                        )

        if not has_any:
            lines.append("    pass")

        return lines

    def _generate_update_pta_indicators(self, pta_indicators: list[IndicatorInfo]) -> list[str]:
        """Generate _update_pta_indicators method for pandas-ta fallback.

        Computes pandas-ta indicators from accumulated bar data buffer.
        Called on each bar before _indicators_ready check.

        Args:
            pta_indicators: Indicators with nt_class=None and pandas_ta_func set

        Returns:
            Method source code as lines
        """
        lines = [
            "def _update_pta_indicators(self) -> None:",
            '    """Compute pandas-ta indicators from bar buffer."""',
        ]

        for info in pta_indicators:
            spec = info.spec
            config = info.config
            func_name = spec.pandas_ta_func

            # Determine minimum lookback from indicator params
            lookback = self._get_pta_lookback(info)

            lines.append(f"    # {info.name} ({config.type}) via ta.{func_name}")
            lines.append(f"    if len(self._pta_close) >= {lookback}:")

            if config.type == "ICHIMOKU":
                # Ichimoku returns (ichimoku_df, span_df) tuple
                tenkan = config.period or spec.default_params.get("tenkan", 9)
                kijun = spec.default_params.get("kijun", 26)
                senkou = spec.default_params.get("senkou", 52)
                lines.extend(
                    [
                        "        _high = pd.Series(self._pta_high)",
                        "        _low = pd.Series(self._pta_low)",
                        "        _close = pd.Series(self._pta_close)",
                        f"        _ichi = ta.ichimoku(_high, _low, _close, tenkan={tenkan}, kijun={kijun}, senkou={senkou})",
                        "        if _ichi is not None and isinstance(_ichi, tuple) and len(_ichi) >= 1:",
                        "            _df = _ichi[0]",
                        "            if _df is not None and len(_df) > 0:",
                        "                _last = _df.iloc[-1]",
                        "                if not pd.isna(_last.iloc[0]):",
                        f'                    self._pta_values["{info.name}_conversion"] = float(_last.iloc[0])',
                        f'                    self._pta_values["{info.name}_base"] = float(_last.iloc[1])',
                        f'                    self._pta_values["{info.name}"] = float(_last.iloc[0])',
                        "            # Span values are in _ichi[1] (span DataFrame)",
                        "            if len(_ichi) >= 2 and _ichi[1] is not None and len(_ichi[1]) > 0:",
                        "                _span_last = _ichi[1].iloc[-1]",
                        "                if len(_span_last) >= 2 and not pd.isna(_span_last.iloc[0]):",
                        f'                    self._pta_values["{info.name}_span_a"] = float(_span_last.iloc[0])',
                        f'                    self._pta_values["{info.name}_span_b"] = float(_span_last.iloc[1])',
                    ]
                )
            elif config.type == "VOLSMA":
                # SMA applied to volume
                period = config.period or spec.default_params.get("period", 20)
                lines.extend(
                    [
                        "        _vol = pd.Series(self._pta_volume)",
                        f"        _result = ta.sma(_vol, length={period})",
                        "        if _result is not None and len(_result) > 0 and not pd.isna(_result.iloc[-1]):",
                        f'            self._pta_values["{info.name}"] = float(_result.iloc[-1])',
                    ]
                )
            elif config.type == "WILLR":
                # Williams %R needs high, low, close
                period = config.period or spec.default_params.get("period", 14)
                lines.extend(
                    [
                        "        _high = pd.Series(self._pta_high)",
                        "        _low = pd.Series(self._pta_low)",
                        "        _close = pd.Series(self._pta_close)",
                        f"        _result = ta.willr(_high, _low, _close, length={period})",
                        "        if _result is not None and len(_result) > 0 and not pd.isna(_result.iloc[-1]):",
                        f'            self._pta_values["{info.name}"] = float(_result.iloc[-1])',
                    ]
                )
            else:
                # Generic single-output indicator (TEMA, etc.) on close series
                period = config.period or spec.default_params.get("period", 14)
                source = config.source or "close"
                source_var = (
                    f"self._pta_{source}"
                    if source in ("close", "high", "low", "open", "volume")
                    else "self._pta_close"
                )
                lines.extend(
                    [
                        f"        _series = pd.Series({source_var})",
                        f"        _result = ta.{func_name}(_series, length={period})",
                        "        if _result is not None and len(_result) > 0 and not pd.isna(_result.iloc[-1]):",
                        f'            self._pta_values["{info.name}"] = float(_result.iloc[-1])',
                    ]
                )

        return lines

    @staticmethod
    def _get_pta_lookback(info: IndicatorInfo) -> int:
        """Get minimum lookback bars needed for a pandas-ta indicator."""
        config = info.config
        defaults = info.spec.default_params

        def _int_param(key: str, fallback: int) -> int:
            val = defaults.get(key, fallback)
            return val if isinstance(val, int) else fallback

        if config.type == "ICHIMOKU":
            return max(
                _int_param("tenkan", 9),
                _int_param("kijun", 26),
                _int_param("senkou", 52),
            )
        if config.type == "TEMA":
            # TEMA needs ~3x period for convergence
            period = config.period or _int_param("period", 14)
            return period * 3
        period = config.period or _int_param("period", 14)
        return period

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

    def _generate_funding_avoidance_method(self, funding_config: object) -> list[str]:
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
        # Strip leading 'check' to avoid 'Check check ...' stutter in docstring
        readable = method_name.replace("_", " ").strip()
        if readable.startswith("check "):
            readable = readable[len("check ") :]
        lines = [
            f"def {method_name}(self, bar: Bar) -> bool:",
            f'    """Check {readable} conditions."""',
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
        right = self._operand_to_threshold_code(cond.left, cond.right)

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
            prev_guard = self._crossover_prev_guard(cond)
            return f"cond_{index} = ({prev_guard}) and ({left} > {right}) and ({prev_left} <= {prev_right})"
        elif cond.operator == Operator.CROSSES_BELOW:
            prev_left = self._operand_to_prev_code(cond.left)
            prev_right = self._operand_to_prev_code(cond.right)
            prev_guard = self._crossover_prev_guard(cond)
            return f"cond_{index} = ({prev_guard}) and ({left} < {right}) and ({prev_left} >= {prev_right})"
        elif cond.operator == Operator.BETWEEN:
            right2 = self._operand_to_threshold_code(cond.left, cond.right2) if cond.right2 else "0"
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
            return (
                f"float(bar.{operand.value}.as_double())"
                if operand.value != "volume"
                else "float(bar.volume.as_double())"
            )
        elif operand.is_indicator:
            return f'self._get_indicator_value("{operand.value}")'
        else:
            # Literal value
            return str(operand.value)

    def _operand_to_threshold_code(self, left_operand: object, right_operand: object) -> str:
        """Convert right operand to code, using config threshold if available.

        For numeric literals compared against indicators, uses self.config.{threshold}
        so threshold values are sweepable via parameter grid.
        """
        from vibe_quant.dsl.conditions import Operand

        if (
            isinstance(left_operand, Operand)
            and isinstance(right_operand, Operand)
            and not right_operand.is_indicator
            and not right_operand.is_price
            and isinstance(right_operand.value, (int, float))
        ):
            left_name = str(left_operand.value)
            key = (left_name, float(right_operand.value))
            threshold_name = self._threshold_map.get(key)
            if threshold_name:
                return f"self.config.{threshold_name}"
        return self._operand_to_code(right_operand)

    @staticmethod
    def _crossover_prev_guard(cond: Condition) -> str:
        """Generate guard expression ensuring prev values exist for crossover.

        Returns 'True' if no indicators need guarding, otherwise an 'in' check
        so the first bar after warmup doesn't fire a false crossover.
        """
        from vibe_quant.dsl.conditions import Operand

        checks: list[str] = []
        for operand in (cond.left, cond.right):
            if isinstance(operand, Operand) and operand.is_indicator:
                checks.append(f'"{operand.value}" in self._prev_values')
        return " and ".join(checks) if checks else "True"

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
        return list(ORDER_METHODS_LINES)


def _to_class_name(snake_case: str) -> str:
    """Convert snake_case to PascalCase.

    Args:
        snake_case: String in snake_case format

    Returns:
        String in PascalCase format
    """
    return "".join(word.capitalize() for word in snake_case.split("_"))
