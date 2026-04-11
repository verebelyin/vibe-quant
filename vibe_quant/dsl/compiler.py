"""DSL-to-NautilusTrader Strategy compiler.

Compiles parsed StrategyDSL into NautilusTrader Strategy subclass Python source code.
Generates on_start() with multi-TF subscriptions, indicator registration,
and on_bar() with time filter evaluation, condition checking, and order submission.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import logging
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from vibe_quant.dsl.conditions import Condition, Operator, parse_condition
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.templates import (
    ON_EVENT_LINES,
    ON_RESET_LINES,
    ON_STOP_LINES,
    ORDER_METHODS_LINES,
)

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
    "random",
    "typing",
    "warnings",
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


def compiler_version_hash() -> str:
    """Return short SHA-256 hash of compiler + templates source.

    Use to detect when compiled strategies may be stale (e.g., after
    fixing the pos.entry→pos.side bug, old discovery results become
    unreliable). Stored in discovery notes and screening results.
    """
    h = hashlib.sha256()
    dsl_dir = Path(__file__).parent
    for name in ("compiler.py", "templates.py", "conditions.py", "indicators.py"):
        src = dsl_dir / name
        if src.exists():
            h.update(src.read_bytes())
    return h.hexdigest()[:12]


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

        # Generalized sub-output coverage check (formerly the MACD force-pta
        # block). If a condition references a sub-value (e.g. ``macd.signal``
        # or ``macd_histogram``) that is NOT in ``spec.nt_output_attrs`` AND
        # NOT in ``spec.computed_outputs``, force the ``compute_fn`` path for
        # that indicator by rebuilding its info with ``nt_class=None``.
        # Applies uniformly to any multi-output indicator where NT exposes
        # only a subset of the outputs the DSL knows about.
        all_condition_text = " ".join(
            dsl.entry_conditions.long
            + dsl.entry_conditions.short
            + dsl.exit_conditions.long
            + dsl.exit_conditions.short
        )
        for i in range(len(indicators)):
            info = indicators[i]
            if info.spec.nt_class is None or info.spec.compute_fn is None:
                continue
            missing: list[str] = []
            for output_name in info.spec.output_names:
                if output_name == "value":
                    continue
                if output_name in info.spec.nt_output_attrs:
                    continue
                if output_name in info.spec.computed_outputs:
                    continue
                if (
                    f"{info.name}.{output_name}" in all_condition_text
                    or f"{info.name}_{output_name}" in all_condition_text
                ):
                    missing.append(output_name)
            if missing:
                from dataclasses import replace as _dc_replace

                new_spec = _dc_replace(info.spec, nt_class=None)
                indicators[i] = _dc_replace(info, spec=new_spec)
                logger.info(
                    "Indicator '%s' (%s) forced to compute_fn: sub-outputs %s "
                    "not in nt_output_attrs/computed_outputs",
                    info.name,
                    info.config.type,
                    missing,
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
            "import random",
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
        compute_fn_imports: dict[str, set[str]] = {}  # module_path -> {fn_name}
        has_pta = False
        for info in indicators:
            if info.spec.nt_class is not None:
                class_name = info.spec.nt_class.__name__
                module_path = info.spec.nt_class.__module__
                nt_classes[class_name] = module_path
            elif info.spec.compute_fn is not None:
                has_pta = True
                fn = info.spec.compute_fn
                compute_fn_imports.setdefault(fn.__module__, set()).add(fn.__name__)

        # Collect derived-output helpers (percent_b, bandwidth, position, ...)
        # from specs whose computed_outputs dict is non-empty AND which are on
        # the NT path (compute_fn path returns the derived values directly).
        derived_helpers: set[str] = set()
        for info in indicators:
            if info.spec.nt_class is not None and info.spec.computed_outputs:
                for helper_name in info.spec.computed_outputs.values():
                    derived_helpers.add(helper_name)

        # Add indicator imports
        if nt_classes:
            imports.append("")
            imports.append("# Indicator imports")
            for class_name, module_path in sorted(nt_classes.items()):
                imports.append(f"from {module_path} import {class_name}")

        # Add compute_fn + pandas imports for compute_fn-path indicators
        if has_pta:
            imports.append("")
            imports.append("# compute_fn imports for indicators without NT class")
            imports.append("import warnings")
            imports.append("warnings.filterwarnings('ignore', category=FutureWarning)")
            imports.append("import pandas as pd")
            for module_path in sorted(compute_fn_imports):
                names = ", ".join(sorted(compute_fn_imports[module_path]))
                imports.append(f"from {module_path} import {names}")

        # Add derived-output helper imports
        if derived_helpers:
            imports.append("")
            imports.append("# Derived-output helpers (percent_b, bandwidth, position, ...)")
            names = ", ".join(sorted(derived_helpers))
            imports.append(f"from vibe_quant.dsl.derived import {names}")

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
        lines.append(
            "    execution_delay_probability: float = 0.0  # Validation-only one-bar delay"
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

        # Add per-direction stop loss parameters (if present)
        for direction in ("long", "short"):
            sl_cfg = getattr(dsl, f"stop_loss_{direction}", None)
            if sl_cfg is not None:
                lines.append("")
                lines.append(f"    # Stop loss ({direction}) parameters")
                lines.append(f'    stop_loss_{direction}_type: str = "{sl_cfg.type}"')
                if sl_cfg.percent is not None:
                    lines.append(f"    stop_loss_{direction}_percent: float = {sl_cfg.percent}")
                if sl_cfg.atr_multiplier is not None:
                    lines.append(f"    stop_loss_{direction}_atr_multiplier: float = {sl_cfg.atr_multiplier}")
                if sl_cfg.indicator is not None:
                    lines.append(f'    stop_loss_{direction}_indicator: str = "{sl_cfg.indicator}"')

        # Add per-direction take profit parameters (if present)
        for direction in ("long", "short"):
            tp_cfg = getattr(dsl, f"take_profit_{direction}", None)
            if tp_cfg is not None:
                lines.append("")
                lines.append(f"    # Take profit ({direction}) parameters")
                lines.append(f'    take_profit_{direction}_type: str = "{tp_cfg.type}"')
                if tp_cfg.percent is not None:
                    lines.append(f"    take_profit_{direction}_percent: float = {tp_cfg.percent}")
                if tp_cfg.atr_multiplier is not None:
                    lines.append(f"    take_profit_{direction}_atr_multiplier: float = {tp_cfg.atr_multiplier}")
                if tp_cfg.risk_reward_ratio is not None:
                    lines.append(f"    take_profit_{direction}_risk_reward: float = {tp_cfg.risk_reward_ratio}")
                if tp_cfg.indicator is not None:
                    lines.append(f'    take_profit_{direction}_indicator: str = "{tp_cfg.indicator}"')

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
            "        self._pending_validation_action: str | None = None",
            "",
            "        # Previous indicator values for crossover detection",
            "        self._prev_values: dict[str, float] = {}",
            "",
            "        # Last close price for computed indicator outputs (percent_b, position)",
            "        self._last_close: float = 0.0",
            "",
        ]

        # Add pandas-ta bar buffer if any indicators need it
        has_pta = any(
            i.spec.nt_class is None and i.spec.compute_fn is not None for i in indicators
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

        # Add on_reset method to suppress NT warning
        on_reset = self._generate_on_reset()
        lines.append(textwrap.indent(on_reset, "    "))
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

    def _generate_on_reset(self) -> str:
        """Generate on_reset() to suppress NT warning and reset state.

        Returns:
            on_reset method source code
        """
        return "\n".join(ON_RESET_LINES)

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

        Thin dispatcher: delegates per-indicator kwarg mapping to
        ``spec.nt_codegen_kwargs`` so new indicators (including plugins)
        don't need compiler edits. Indicators without an NT class emit a
        marker comment and skip registration — their values come from
        ``_update_pta_indicators`` instead.
        """
        lines: list[str] = []
        spec = info.spec

        if spec.nt_class is None:
            label = spec.pandas_ta_func or (
                spec.compute_fn.__name__ if spec.compute_fn is not None else "?"
            )
            lines.append(
                f"    # {info.name} ({info.config.type}): pandas-ta fallback via ta.{label}"
            )
            return lines

        class_name = spec.nt_class.__name__
        args = [
            f"{nt_kwarg}=self.config.{info.name}_{dsl_field}"
            for nt_kwarg, dsl_field in spec.nt_codegen_kwargs
        ]
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
            i.spec.nt_class is None and i.spec.compute_fn is not None for i in indicators
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
                "    # Track last close for computed indicator outputs",
                "    self._last_close = float(bar.close)",
                "",
                "    # Check if indicators are ready",
                "    if not self._indicators_ready():",
                "        return",
                "",
                "    # Execute any validation-only delayed action before new signals",
                "    if self._dispatch_pending_validation_action(bar):",
                "        self._update_prev_values()",
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
            lines.append("            if not self._maybe_delay_validation_action('long_entry'):")
            lines.append("                self._submit_long_entry(bar)")
        if dsl.entry_conditions.short:
            if dsl.entry_conditions.long:
                lines.append("        elif self._check_short_entry(bar):")
            else:
                lines.append("        if self._check_short_entry(bar):")
            lines.append("            if not self._maybe_delay_validation_action('short_entry'):")
            lines.append("                self._submit_short_entry(bar)")

        # Exit conditions
        if dsl.exit_conditions.long or dsl.exit_conditions.short:
            lines.append("")
            lines.append("    # Evaluate exit conditions")
            lines.append("    if self._position_open:")
            if dsl.exit_conditions.long:
                lines.append("        if self._position_side == OrderSide.BUY:")
                lines.append("            if self._check_long_exit(bar):")
                lines.append("                if not self._maybe_delay_validation_action('exit'):")
                lines.append("                    self._submit_exit(bar)")
            if dsl.exit_conditions.short:
                if dsl.exit_conditions.long:
                    lines.append("        elif self._position_side == OrderSide.SELL:")
                else:
                    lines.append("        if self._position_side == OrderSide.SELL:")
                lines.append("            if self._check_short_exit(bar):")
                lines.append("                if not self._maybe_delay_validation_action('exit'):")
                lines.append("                    self._submit_exit(bar)")

        # Trailing stop update
        has_trailing = (
            dsl.stop_loss.type == "atr_trailing"
            or (dsl.stop_loss_long is not None and dsl.stop_loss_long.type == "atr_trailing")
            or (dsl.stop_loss_short is not None and dsl.stop_loss_short.type == "atr_trailing")
        )
        if has_trailing:
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
            spec = info.spec
            if spec.nt_class is not None:
                lines.append(f"    if not {info.indicator_var}.initialized:")
                lines.append("        return False")
            elif spec.compute_fn is not None:
                # Check compute_fn indicator has computed a value
                lines.append(f'    if "{info.name}" not in self._pta_values:')
                lines.append("        return False")
                # Multi-output sub-names (computed_outputs are derived at read
                # time, so they never land in _pta_values and should be skipped
                # from the readiness check).
                if spec.output_names != ("value",):
                    for output_name in spec.output_names:
                        if output_name in spec.computed_outputs:
                            continue
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

        # _update_pta_indicators (if any compute_fn indicators exist)
        pta_indicators = [
            i for i in indicators if i.spec.nt_class is None and i.spec.compute_fn is not None
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

    @staticmethod
    def _effective_primary(spec: IndicatorSpec) -> str:
        """Return the output name that resolves when the indicator is
        referenced without a sub-value (e.g. ``bbands`` vs ``bbands.upper``).

        Defaults to ``spec.primary_output`` if set (BBANDS/KC/DONCHIAN pin
        this to ``"middle"``), else the first entry in ``output_names``,
        else the plain ``"value"`` sentinel.
        """
        if spec.primary_output:
            return spec.primary_output
        if spec.output_names:
            return spec.output_names[0]
        return "value"

    def _generate_get_indicator_value(self, indicators: list[IndicatorInfo]) -> list[str]:
        """Generate the ``_get_indicator_value`` lookup.

        Spec-driven: reads from ``_pta_values`` for compute_fn-path
        indicators, from ``spec.nt_output_attrs`` for NT-path indicators,
        and from ``spec.computed_outputs`` (-> derived helpers) for
        outputs that are derived at read time from the raw bands.
        """
        lines = [
            "def _get_indicator_value(self, name: str) -> float:",
            '    """Get current value of an indicator by name."""',
        ]

        for info in indicators:
            spec = info.spec
            if spec.nt_class is None:
                # compute_fn path: read from _pta_values buffer
                lines.append(f'    if name == "{info.name}":')
                lines.append(f'        return self._pta_values.get("{info.name}", 0.0)')
                if spec.output_names != ("value",):
                    for output_name in spec.output_names:
                        # Derived outputs are computed at read time by the
                        # compute_fn already (see compute_bbands), so they
                        # live in _pta_values under the sub-key too.
                        lines.append(f'    if name == "{info.name}_{output_name}":')
                        lines.append(
                            f'        return self._pta_values.get("{info.name}_{output_name}", 0.0)'
                        )
                continue

            # NT path
            primary = self._effective_primary(spec)
            primary_attr = spec.nt_output_attrs.get(primary, "value")
            lines.append(f'    if name == "{info.name}":')
            lines.append(f"        _v = {info.indicator_var}.{primary_attr}")
            lines.append("        return float(_v) if _v is not None else 0.0")

            if spec.output_names != ("value",):
                for output_name in spec.output_names:
                    key = f"{info.name}_{output_name}"
                    if output_name in spec.computed_outputs:
                        helper = spec.computed_outputs[output_name]
                        lines.append(f'    if name == "{key}":')
                        lines.append(
                            f"        return {helper}({info.indicator_var}, self._last_close)"
                        )
                    elif output_name in spec.nt_output_attrs:
                        attr = spec.nt_output_attrs[output_name]
                        lines.append(f'    if name == "{key}":')
                        lines.append(f"        _v = {info.indicator_var}.{attr}")
                        lines.append("        return float(_v) if _v is not None else 0.0")
                    # else: sub-value is not covered by NT and no derived
                    # helper — the compile-time sub-value fallback would
                    # have already forced this indicator to the compute_fn
                    # path, so this branch is unreachable for well-formed
                    # specs.

        lines.append('    raise ValueError(f"Unknown indicator: {name}")')
        return lines

    def _generate_update_prev_values(self, indicators: list[IndicatorInfo]) -> list[str]:
        """Generate the ``_update_prev_values`` helper.

        Mirrors ``_generate_get_indicator_value`` but writes into
        ``self._prev_values`` for crossover detection on the next bar.
        """
        lines = [
            "def _update_prev_values(self) -> None:",
            '    """Store current indicator values for crossover detection."""',
        ]

        has_any = False
        for info in indicators:
            spec = info.spec
            if spec.nt_class is not None:
                has_any = True
                primary = self._effective_primary(spec)
                primary_attr = spec.nt_output_attrs.get(primary, "value")
                lines.append(
                    f'    self._prev_values["{info.name}"] = float({info.indicator_var}.{primary_attr})'
                )
                if spec.output_names != ("value",):
                    for output_name in spec.output_names:
                        key = f"{info.name}_{output_name}"
                        if output_name in spec.computed_outputs:
                            # Derived outputs need the same runtime helper path
                            # used by _get_indicator_value to stay consistent.
                            lines.append(
                                f'    self._prev_values["{key}"] = self._get_indicator_value("{key}")'
                            )
                        elif output_name in spec.nt_output_attrs:
                            attr = spec.nt_output_attrs[output_name]
                            lines.append(
                                f'    self._prev_values["{key}"] = float({info.indicator_var}.{attr})'
                            )
            elif spec.compute_fn is not None:
                has_any = True
                lines.append(
                    f'    self._prev_values["{info.name}"] = self._pta_values.get("{info.name}", 0.0)'
                )
                if spec.output_names != ("value",):
                    for output_name in spec.output_names:
                        lines.append(
                            f'    self._prev_values["{info.name}_{output_name}"] = self._pta_values.get("{info.name}_{output_name}", 0.0)'
                        )

        if not has_any:
            lines.append("    pass")

        return lines

    def _generate_update_pta_indicators(self, pta_indicators: list[IndicatorInfo]) -> list[str]:
        """Generate ``_update_pta_indicators`` for compute_fn-path indicators.

        Emits a generic dispatcher that builds a single OHLCV DataFrame per
        bar, then for each compute_fn-path indicator imports the spec's
        ``compute_fn`` by name and calls it with the merged params. Results
        are unpacked into ``self._pta_values`` either as a single scalar
        (single-output) or namespaced by sub-output key (multi-output).

        This replaces the ~200-line per-type elif chain from the P4
        refactor; new plugins with a ``compute_fn`` slot in without
        touching the compiler.
        """
        lines = [
            "def _update_pta_indicators(self) -> None:",
            '    """Compute compute_fn-path indicators from bar buffer."""',
            "    _df = pd.DataFrame({",
            '        "open": self._pta_open,',
            '        "high": self._pta_high,',
            '        "low": self._pta_low,',
            '        "close": self._pta_close,',
            '        "volume": self._pta_volume,',
            "    })",
        ]

        for info in pta_indicators:
            spec = info.spec
            if spec.compute_fn is None:
                continue
            fn_name = spec.compute_fn.__name__
            lookback = self._get_pta_lookback(info)
            params_literal = self._compile_pta_params_literal(info)
            primary = self._effective_primary(spec)

            lines.append(f"    # {info.name} ({info.config.type}) via {fn_name} — lookback {lookback}")
            lines.append(f"    if len(self._pta_close) >= {lookback}:")
            lines.append(f"        _res = {fn_name}(_df, {params_literal})")

            if len(spec.output_names) > 1:
                lines.append("        if isinstance(_res, dict):")
                lines.append(f'            _primary = _res.get("{primary}")')
                lines.append("            if _primary is not None and len(_primary) > 0:")
                lines.append("                _v = _primary.iloc[-1]")
                lines.append("                if not pd.isna(_v):")
                lines.append(f'                    self._pta_values["{info.name}"] = float(_v)')
                lines.append("            for _k, _s in _res.items():")
                lines.append("                if _s is not None and len(_s) > 0:")
                lines.append("                    _v = _s.iloc[-1]")
                lines.append("                    if not pd.isna(_v):")
                lines.append(
                    f'                        self._pta_values["{info.name}_" + _k] = float(_v)'
                )
            else:
                lines.append("        if _res is not None and len(_res) > 0:")
                lines.append("            _v = _res.iloc[-1]")
                lines.append("            if not pd.isna(_v):")
                lines.append(f'                self._pta_values["{info.name}"] = float(_v)')

        return lines

    @staticmethod
    def _compile_pta_params_literal(info: IndicatorInfo) -> str:
        """Build a Python-literal dict string for the ``compute_fn`` call.

        Starts from ``spec.default_params`` and overlays any
        IndicatorConfig fields that were explicitly set in the DSL, so
        user overrides flow through even though the compute_fn call is
        emitted with hardcoded values (matching pre-refactor behavior).
        """
        merged: dict[str, object] = dict(info.spec.default_params)
        for dsl_field in (
            "period",
            "fast_period",
            "slow_period",
            "signal_period",
            "d_period",
            "std_dev",
            "atr_multiplier",
        ):
            val = getattr(info.config, dsl_field, None)
            if val is not None:
                merged[dsl_field] = val
        pairs = ", ".join(f'"{k}": {v!r}' for k, v in merged.items())
        return "{" + pairs + "}"

    @staticmethod
    def _get_pta_lookback(info: IndicatorInfo) -> int:
        """Minimum lookback bars needed before a compute_fn is valid.

        Dispatches to ``spec.pta_lookback_fn`` when set (TEMA, MACD,
        ICHIMOKU have custom formulas). Default: read ``period`` from the
        effective param dict — matches pre-refactor behavior for every
        single-period pandas-ta indicator.
        """
        spec = info.spec
        merged: dict[str, object] = dict(spec.default_params)
        for dsl_field in (
            "period",
            "fast_period",
            "slow_period",
            "signal_period",
            "d_period",
            "std_dev",
            "atr_multiplier",
        ):
            val = getattr(info.config, dsl_field, None)
            if val is not None:
                merged[dsl_field] = val
        if spec.pta_lookback_fn is not None:
            return int(spec.pta_lookback_fn(merged))
        period = merged.get("period")
        if isinstance(period, (int, float)):
            return int(period)
        return 14

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
