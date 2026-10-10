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
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from vibe_quant.dsl.conditions import Condition, Operator, parse_condition
from vibe_quant.dsl.indicators import (
    IndicatorSpec,
    indicator_registry,
    pta_buffer_cap,
    pta_lookback,
)
from vibe_quant.dsl.templates import (
    ON_EVENT_LINES,
    ON_RESET_LINES,
    ON_START_OUTBOX_LINES,
    ON_START_RECOVERY_LINES,
    ON_STOP_LINES,
    ON_TRADE_TICK_LINES,
    ORDER_METHODS_LINES,
    PTA_FEED_LINES,
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


_GENERATED_TS_LINE = re.compile(r"^Generated: .*$", re.MULTILINE)


def generated_module_name(dsl_name: str, source: str) -> str:
    """Content-addressed ``sys.modules`` name for a compiled strategy.

    Keyed by the generated source (minus the ``Generated:`` timestamp line),
    so two DSLs sharing a name but differing in any compiled detail (GA
    elite vs. mutant with the same ``genome_{uid}``) can never overwrite
    each other's module, while recompiling an identical DSL is idempotent.
    """
    digest = hashlib.sha256(_GENERATED_TS_LINE.sub("", source).encode()).hexdigest()[:16]
    return f"vibe_quant.dsl.generated.{dsl_name}_{digest}"


def compiler_version_hash() -> str:
    """Return short SHA-256 hash of compiler + templates source.

    Use to detect when compiled strategies may be stale (e.g., after
    fixing the pos.entry→pos.side bug, old discovery results become
    unreliable). Stored in discovery notes and screening results.
    """
    h = hashlib.sha256()
    dsl_dir = Path(__file__).parent
    # Everything that shapes generated code or indicator values: compute_fn
    # bodies, derived helpers, schema normalization and drop-in plugins too.
    sources = [
        dsl_dir / name
        for name in (
            "compiler.py",
            "templates.py",
            "conditions.py",
            "indicators.py",
            "compute_builtins.py",
            "aux_data.py",
            "derived.py",
            "schema.py",
            "prefix_memo.py",
            "pta_buffer.py",
        )
    ]
    sources += sorted((dsl_dir / "plugins").glob("*.py"))
    for src in sources:
        if src.exists():
            h.update(src.name.encode())
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
        config_class = self._generate_config_class(dsl, indicator_names, indicators)
        strategy_class = self._generate_strategy_class(dsl, indicators, timeframes, indicator_names)

        # Combine
        source = "\n".join([imports, "", config_class, "", strategy_class])
        return source

    def compile_to_module(self, dsl: StrategyDSL) -> ModuleType:
        """Compile DSL to a loadable Python module.

        The module is registered in ``sys.modules`` under a content-addressed
        name (see :func:`generated_module_name`); callers must use the
        returned ``module.__name__`` for ``ImportableStrategyConfig`` paths,
        never re-derive it from ``dsl.name``.

        Args:
            dsl: Parsed and validated StrategyDSL

        Returns:
            Loaded module containing the Strategy and Config classes

        Raises:
            CompilerError: If compilation fails
        """
        source = self.compile(dsl)
        self._validate_generated_source(source)
        module_name = generated_module_name(dsl.name, source)

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
            "from nautilus_trader.model.data import Bar, BarType, TradeTick",
            "from nautilus_trader.model.enums import OrderSide, OrderType, PositionSide, TimeInForce",
            "from nautilus_trader.model.identifiers import InstrumentId",
            "from nautilus_trader.model.instruments import Instrument",
            "from nautilus_trader.model.objects import Price, Quantity",
            "from nautilus_trader.model.events import OrderFilled, PositionChanged, PositionClosed, PositionOpened",
            "from nautilus_trader.model.orders import LimitOrder, MarketOrder, StopMarketOrder",
            "from nautilus_trader.trading.strategy import Strategy, StrategyConfig",
        ]

        # Single pass: collect NT classes, compute_fn imports, and derived helpers.
        nt_classes: dict[str, str] = {}  # class_name -> module_path
        compute_fn_imports: dict[str, set[str]] = {}  # module_path -> {fn_name}
        derived_helpers: set[str] = set()
        has_pta = False
        for info in indicators:
            if info.spec.nt_class is not None:
                class_name = info.spec.nt_class.__name__
                module_path = info.spec.nt_class.__module__
                nt_classes[class_name] = module_path
                # NT-path indicators with computed_outputs need derived helpers.
                for helper_name in info.spec.computed_outputs.values():
                    derived_helpers.add(helper_name)
                if info.spec.primary_helper:
                    derived_helpers.add(info.spec.primary_helper)
                for _kwarg, helper_name, _field in info.spec.nt_codegen_helper_kwargs:
                    derived_helpers.add(helper_name)
            elif info.spec.compute_fn is not None:
                has_pta = True
                fn = info.spec.compute_fn
                compute_fn_imports.setdefault(fn.__module__, set()).add(fn.__name__)

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
            imports.append("from vibe_quant.dsl.indicators import pta_buffer_cap, pta_lookback")
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
        self,
        dsl: StrategyDSL,
        indicator_names: list[str] | None = None,
        indicators: list[IndicatorInfo] | None = None,
    ) -> str:
        """Generate the Strategy config dataclass.

        Args:
            dsl: Parsed DSL
            indicator_names: Expanded indicator names (includes sub-outputs)
            indicators: Indicator infos (after compute_fn forcing); every
                compute_fn-path param gets a config field so sweeps apply.

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
            "    execution_delay_probability: float = 0.0  "
            "# One-bar exec delay: screening/discovery=1.0 (always), sub-5m validation=0.3"
        )
        lines.append(
            "    execution_delay_seed: int = 42  "
            "# Seed for probabilistic delay draws (reproducible replays)"
        )
        lines.append(
            '    command_release: str = ""  '
            '# Command outbox release: "" off (live/paper/1m), "trade_tick", "bar"'
        )
        lines.append(
            '    command_release_bar_type: str = ""  '
            '# Own detail bar type releasing the outbox when command_release="bar"'
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

        # compute_fn-path params not covered above (spec defaults such as
        # WILLR period / ICHIMOKU tenkan, plugin extras such as alpha) also
        # become config fields: the generated code reads every compute_fn
        # param from self.config so sweep/WFA overrides take effect.
        for info in indicators or []:
            if not self._is_pta(info):
                continue
            for _key, field_name, type_name, default, native in self._pta_param_fields(info):
                if not native:
                    lines.append(f"    {field_name}: {type_name} = {default!r}")

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
            "        self._trailing_best_sl: float | None = None",
            "        # Set when on_start adopts an unprotected position before indicators are ready",
            "        self._rearm_protection = False",
            "        # Command outbox: (queue ts, fn, args) awaiting this instrument's next datum",
            "        self._outbox: list = []",
            "        self._command_release_bar_type: BarType | None = None",
            "        # Seeded RNG for the probabilistic execution-delay path so",
            "        # identical validation replays are byte-reproducible.",
            "        self._delay_rng = random.Random(",
            "            getattr(config, 'execution_delay_seed', 42)",
            "        )",
            "",
            "        # Previous indicator values for crossover detection",
            "        self._prev_values: dict[str, float] = {}",
            "",
            "        # Last close price for computed indicator outputs (percent_b, position)",
            "        self._last_close: float = 0.0",
            "",
        ]

        # compute_fn (pandas) path: per-timeframe bar buffers + params read
        # from config (vibe-quant-e70tl.8).
        pta_infos = [i for i in indicators if self._is_pta(i)]
        if pta_infos:
            lines.extend(self._generate_pta_init(pta_infos))

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

        # Add on_trade_tick method (command outbox release)
        lines.append(textwrap.indent("\n".join(ON_TRADE_TICK_LINES), "    "))
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
            "1d": "1-DAY",
        }

        for tf in sorted(timeframes):
            spec = tf_to_spec.get(tf)
            if spec is None:
                # Silently defaulting to 5-MINUTE would subscribe the wrong
                # bars; fail loudly if VALID_TIMEFRAMES ever outgrows this map.
                msg = f"No bar spec mapping for timeframe '{tf}'"
                raise CompilerError(msg)
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
        lines.extend(f"    {line}" if line else "" for line in ON_START_OUTBOX_LINES)

        lines.append("")
        lines.append("    # Initialize and register indicators")

        # Create indicators
        for info in indicators:
            lines.extend(self._generate_indicator_init(info))

        lines.append("")
        lines.extend(f"    {line}" if line else "" for line in ON_START_RECOVERY_LINES)

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
            label = (
                spec.compute_fn.__name__ if spec.compute_fn is not None
                else spec.pandas_ta_func or "?"
            )
            lines.append(
                f"    # {info.name} ({info.config.type}): compute_fn path via {label}"
            )
            return lines

        class_name = spec.nt_class.__name__
        args = [
            f"{nt_kwarg}=self.config.{info.name}_{dsl_field}"
            for nt_kwarg, dsl_field in spec.nt_codegen_kwargs
        ]
        args.extend(
            f"{nt_kwarg}={helper}(self.config.{info.name}_{dsl_field})"
            for nt_kwarg, helper, dsl_field in spec.nt_codegen_helper_kwargs
        )
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
        pta_tfs = sorted({i.timeframe for i in indicators if self._is_pta(i)})

        lines = [
            "def on_bar(self, bar: Bar) -> None:",
            '    """Handle bar updates."""',
        ]

        # Feed compute_fn bar buffers BEFORE the primary-timeframe filter so an
        # indicator declared on another timeframe (e.g. a 4h ADX on a 1h
        # strategy) is computed from that timeframe's bars -- the same update
        # semantics NT applies to register_indicator_for_bars indicators.
        if pta_tfs:
            lines.append("    # Feed compute_fn indicator buffers (per timeframe)")
            for idx, tf in enumerate(pta_tfs):
                kw = "if" if idx == 0 else "elif"
                lines.append(f"    {kw} bar.bar_type == self.bar_type_{tf}:")
                lines.append(f'        self._feed_pta_buffer("{tf}", bar)')
            lines.append("")

        # Outbox release AFTER the buffer feed: an early return ahead of it
        # would starve 1m compute_fn indicators on coarser strategies.
        lines.extend(
            [
                "    # Command outbox: this instrument's own detail bar releases queued commands",
                "    if self._command_release_bar_type is not None and bar.bar_type == self._command_release_bar_type:",
                "        self._flush_outbox(bar.ts_init)",
                "",
                "    # Only process primary timeframe bars",
                "    if bar.bar_type != self.primary_bar_type:",
                "        return",
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
                "    # Restart recovery: protect a position adopted in on_start",
                "    if self._rearm_protection:",
                "        self._ensure_protection()",
                "",
                "    # Execute any validation-only delayed action before new signals",
                "    if self._dispatch_pending_validation_action(bar):",
                "        self._update_prev_values(bar)",
                "        return",
                "",
            ]
        )

        # Time filters / funding avoidance gate NEW ENTRIES only: exits,
        # trailing-stop updates and prev values always run. Evaluated at the
        # bar's close time (ts_init rounded to the bar boundary): ts_event is
        # the bar OPEN in backtests but the close for live Binance bars.
        has_session_filter = bool(
            dsl.time_filters.allowed_sessions or dsl.time_filters.blocked_days
        )
        has_funding_filter = dsl.time_filters.avoid_around_funding.enabled
        entry_gate = ""
        if has_session_filter or has_funding_filter:
            checks: list[str] = []
            if has_session_filter:
                checks.append("self._check_time_filters(_close_ns)")
            if has_funding_filter:
                checks.append("not self._is_near_funding_time(_close_ns)")
            lines.append("    # Time filters gate entries only (evaluated at bar close time)")
            lines.append("    _close_ns = self._bar_close_ns(bar)")
            lines.append(f"    _entries_allowed = {' and '.join(checks)}")
            lines.append("")
            entry_gate = " and _entries_allowed"

        # Entry conditions
        lines.append("    # Evaluate entry conditions")
        lines.append(f"    if not self._position_open{entry_gate}:")
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
        lines.append("    self._update_prev_values(bar)")

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
        lines.extend(
            self._generate_update_prev_values(
                indicators, self._crossover_price_refs(dsl, indicator_names)
            )
        )
        lines.append("")

        # _feed_pta_buffer + _update_pta_indicators (compute_fn indicators)
        pta_indicators = [i for i in indicators if self._is_pta(i)]
        if pta_indicators:
            lines.extend(self._feed_lines(pta_indicators))
            lines.append("")
            lines.extend(self._generate_update_pta_indicators(pta_indicators))
            lines.append("")

        # Time filter method
        lines.extend(self._generate_time_filter_method(dsl.time_filters))
        lines.append("")

        tf = dsl.time_filters
        if tf.allowed_sessions or tf.blocked_days or tf.avoid_around_funding.enabled:
            step_ns = self._TIMEFRAME_MINUTES[dsl.timeframe] * 60_000_000_000
            lines.extend(
                [
                    "def _bar_close_ns(self, bar: Bar) -> int:",
                    '    """Bar close time: ts_init rounded to the nearest bar boundary.',
                    "",
                    "    Backtest bars carry ts_init = exchange close_time (hh:59:59.999), live",
                    "    bars the receive time (just after the boundary); both round to the",
                    "    same boundary. ts_event would be the OPEN time in backtests.",
                    '    """',
                    f"    _step = {step_ns}",
                    "    return ((bar.ts_init + _step // 2) // _step) * _step",
                    "",
                ]
            )

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

    _TIMEFRAME_MINUTES: ClassVar[dict[str, int]] = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }

    def _primary_helper_call(self, info: IndicatorInfo) -> str:
        """``helper(ind, last_close, bar_minutes)`` for ``spec.primary_helper``."""
        minutes = self._TIMEFRAME_MINUTES.get(info.timeframe)
        if minutes is None:
            msg = f"No bar length for timeframe '{info.timeframe}' (indicator '{info.name}')"
            raise CompilerError(msg)
        return f"{info.spec.primary_helper}({info.indicator_var}, self._last_close, {minutes})"

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
            primary_scale = spec.nt_output_scale.get(primary, 1.0)
            lines.append(f'    if name == "{info.name}":')
            if spec.primary_helper:
                lines.append(f"        return {self._primary_helper_call(info)}")
            else:
                lines.append(f"        _v = {info.indicator_var}.{primary_attr}")
                if primary_scale != 1.0:
                    lines.append(
                        f"        return float(_v) * {primary_scale} if _v is not None else 0.0"
                    )
                else:
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
                        scale = spec.nt_output_scale.get(output_name, 1.0)
                        lines.append(f'    if name == "{key}":')
                        lines.append(f"        _v = {info.indicator_var}.{attr}")
                        if scale != 1.0:
                            lines.append(
                                f"        return float(_v) * {scale} if _v is not None else 0.0"
                            )
                        else:
                            lines.append("        return float(_v) if _v is not None else 0.0")
                    # else: sub-value is not covered by NT and no derived
                    # helper — the compile-time sub-value fallback would
                    # have already forced this indicator to the compute_fn
                    # path, so this branch is unreachable for well-formed
                    # specs.

        lines.append('    raise ValueError(f"Unknown indicator: {name}")')
        return lines

    @staticmethod
    def _prev_price_key(price: str) -> str:
        """``_prev_values`` key for a price operand. ``@`` cannot appear in an
        indicator name, so price keys never collide with indicator keys."""
        return f"@{price}"

    @staticmethod
    def _crossover_price_refs(dsl: StrategyDSL, indicator_names: list[str]) -> list[str]:
        """Price operands (close/open/high/low/volume) used in any crossover.

        Their previous-bar values must be tracked: without them a
        ``close crosses_above ema`` check degenerates to
        ``close > ema and close <= prev_ema`` (vibe-quant-e70tl.3).
        """
        refs: set[str] = set()
        for cond_str in (
            dsl.entry_conditions.long
            + dsl.entry_conditions.short
            + dsl.exit_conditions.long
            + dsl.exit_conditions.short
        ):
            cond = parse_condition(cond_str, indicator_names)
            if cond.operator not in (Operator.CROSSES_ABOVE, Operator.CROSSES_BELOW):
                continue
            for operand in (cond.left, cond.right):
                if operand.is_price:
                    refs.add(str(operand.value))
        return sorted(refs)

    def _generate_update_prev_values(
        self, indicators: list[IndicatorInfo], price_refs: list[str] | None = None
    ) -> list[str]:
        """Generate the ``_update_prev_values`` helper.

        Mirrors ``_generate_get_indicator_value`` but writes into
        ``self._prev_values`` for crossover detection on the next bar.
        Price operands used in crossovers are stored under ``@<price>``.
        """
        lines = [
            "def _update_prev_values(self, bar: Bar) -> None:",
            '    """Store current indicator/price values for crossover detection."""',
        ]

        has_any = False
        for price in price_refs or []:
            has_any = True
            lines.append(
                f'    self._prev_values["{self._prev_price_key(price)}"] = '
                f"float(bar.{price}.as_double())"
            )
        for info in indicators:
            spec = info.spec
            if spec.nt_class is not None:
                has_any = True
                primary = self._effective_primary(spec)
                primary_attr = spec.nt_output_attrs.get(primary, "value")
                primary_scale = spec.nt_output_scale.get(primary, 1.0)
                _scale_suffix = f" * {primary_scale}" if primary_scale != 1.0 else ""
                if spec.primary_helper:
                    lines.append(
                        f'    self._prev_values["{info.name}"] = '
                        f"{self._primary_helper_call(info)}"
                    )
                else:
                    lines.append(
                        f'    self._prev_values["{info.name}"] = '
                        f"float({info.indicator_var}.{primary_attr}){_scale_suffix}"
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
                            scale = spec.nt_output_scale.get(output_name, 1.0)
                            _suf = f" * {scale}" if scale != 1.0 else ""
                            lines.append(
                                f'    self._prev_values["{key}"] = '
                                f"float({info.indicator_var}.{attr}){_suf}"
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

    @staticmethod
    def _is_pta(info: IndicatorInfo) -> bool:
        """True when the indicator runs on the compute_fn (pandas) path."""
        return info.spec.nt_class is None and info.spec.compute_fn is not None

    @staticmethod
    def _has_context(infos: list[IndicatorInfo]) -> bool:
        """True when any compute_fn indicator needs aux context (FUNDING, ...)."""
        return any(i.spec.needs_context for i in infos)

    def _feed_lines(self, pta_infos: list[IndicatorInfo]) -> list[str]:
        """``_feed_pta_buffer`` source; context strategies also buffer bar close times."""
        if not self._has_context(pta_infos):
            return list(PTA_FEED_LINES)
        out: list[str] = []
        for line in PTA_FEED_LINES:
            out.append(line)
            if line.startswith("    _buf[\"volume\"]"):
                out.extend(
                    [
                        "    # Bar close time (ts_init rounded to the timeframe boundary) for",
                        "    # needs_context indicators (aux data is looked up as-of close).",
                        "    _step = self._pta_step_ns[tf]",
                        '    _buf["close_ns"].append(((bar.ts_init + _step // 2) // _step) * _step)',
                    ]
                )
        return out

    def _generate_pta_init(self, pta_infos: list[IndicatorInfo]) -> list[str]:
        """``__init__`` state for compute_fn-path indicators.

        * one OHLCV buffer per timeframe used by a compute_fn indicator, so a
          4h indicator on a 1h strategy is fed 4h bars;
        * ``_pta_params``: every compute_fn param read from ``config`` (sweep /
          WFA overrides of e.g. ``adx_period`` must change the computation);
        * warmup gate and rolling-buffer cap derived from those params at
          runtime via the same helpers the compiler uses.
        """
        tfs = sorted({i.timeframe for i in pta_infos})
        ctx = self._has_context(pta_infos)
        lines = [
            "        # compute_fn (pandas) indicators: one OHLCV bar buffer per timeframe",
            "        self._pta_bufs: dict[str, dict[str, list[float]]] = {"
            if not ctx
            else "        self._pta_bufs: dict[str, dict[str, list]] = {",
        ]
        for tf in tfs:
            extra = ', "close_ns": []' if ctx else ""
            lines.append(
                f'            "{tf}": {{"open": [], "high": [], "low": [], "close": [], "volume": []{extra}}},'
            )
        lines.append("        }")
        if ctx:
            lines.append("        self._pta_step_ns: dict[str, int] = {")
            for tf in tfs:
                minutes = self._TIMEFRAME_MINUTES.get(tf)
                if minutes is None:
                    msg = f"No bar length for timeframe '{tf}' (context indicator buffer)"
                    raise CompilerError(msg)
                lines.append(f'            "{tf}": {minutes * 60_000_000_000},')
            lines.append("        }")
        lines.append("        self._pta_values: dict[str, float] = {}")
        lines.append("        # compute_fn params come from config so sweep/WFA overrides apply")
        lines.append("        self._pta_params: dict[str, dict[str, object]] = {")
        for info in pta_infos:
            pairs = ", ".join(
                f'"{key}": config.{field}' for key, field, *_ in self._pta_param_fields(info)
            )
            lines.append(f'            "{info.name}": {{{pairs}}},')
        lines.append("        }")
        lines.append("        # Bars buffered before each compute_fn is first called")
        lines.append("        self._pta_lookback: dict[str, int] = {")
        for info in pta_infos:
            lines.append(
                f'            "{info.name}": pta_lookback("{info.spec.name}", '
                f'self._pta_params["{info.name}"]),'
            )
        lines.append("        }")
        lines.append(
            "        # Rolling-buffer cap per timeframe (0 = unbounded: cumulative indicator)"
        )
        lines.append("        self._pta_buffer_cap: dict[str, int] = {")
        for tf in tfs:
            tf_infos = [i for i in pta_infos if i.timeframe == tf]
            lbs = ", ".join(f'self._pta_lookback["{i.name}"]' for i in tf_infos)
            full = any(i.spec.requires_full_history for i in tf_infos)
            lines.append(f'            "{tf}": pta_buffer_cap([{lbs}], full_history={full}),')
        lines.append("        }")
        lines.append("")
        return lines

    def _generate_update_pta_indicators(self, pta_indicators: list[IndicatorInfo]) -> list[str]:
        """Generate ``_update_pta_indicators(tf)`` for compute_fn-path indicators.

        Called by ``_feed_pta_buffer`` after a bar of timeframe ``tf`` lands in
        its buffer. Builds one OHLCV DataFrame per call, then calls each of the
        timeframe's ``compute_fn`` with its config-resolved params. Results are
        unpacked into ``self._pta_values`` either as a single scalar
        (single-output) or namespaced by sub-output key (multi-output). A NaN
        result (warmup) is never stored, so the indicator stays not-ready.
        """
        lines = [
            "def _update_pta_indicators(self, tf: str) -> None:",
            '    """Compute the compute_fn indicators of timeframe ``tf`` from its buffer."""',
            "    _buf = self._pta_bufs[tf]",
            '    _n = len(_buf["close"])',
            "    _df = None",
        ]

        tfs = sorted({i.timeframe for i in pta_indicators})
        has_ctx = self._has_context(pta_indicators)
        for idx, tf in enumerate(tfs):
            kw = "if" if idx == 0 else "elif"
            lines.append(f'    {kw} tf == "{tf}":')
            for info in (i for i in pta_indicators if i.timeframe == tf):
                spec = info.spec
                if spec.compute_fn is None:
                    continue
                fn_name = spec.compute_fn.__name__
                primary = self._effective_primary(spec)
                ind = "        "
                lines.append(
                    f"{ind}# {info.name} ({info.config.type}) via {fn_name}"
                    f" — default lookback {self._get_pta_lookback(info)}"
                )
                lines.append(f'{ind}if _n >= self._pta_lookback["{info.name}"]:')
                lines.append(f"{ind}    if _df is None:")
                lines.append(
                    f'{ind}        _df = pd.DataFrame({{"open": _buf["open"], "high": _buf["high"], '
                    '"low": _buf["low"], "close": _buf["close"], "volume": _buf["volume"]})'
                )
                if has_ctx:
                    lines.append(f'{ind}        _df.attrs["bar_close_ns"] = _buf["close_ns"]')
                    lines.append(f'{ind}        _df.attrs["symbol"] = self.config.instrument_id')
                lines.append(f'{ind}    _res = {fn_name}(_df, self._pta_params["{info.name}"])')

                if len(spec.output_names) > 1:
                    lines.append(f"{ind}    if isinstance(_res, dict):")
                    lines.append(f'{ind}        _primary = _res.get("{primary}")')
                    lines.append(f"{ind}        if _primary is not None and len(_primary) > 0:")
                    lines.append(f"{ind}            _v = _primary.iloc[-1]")
                    lines.append(f"{ind}            if not pd.isna(_v):")
                    lines.append(f'{ind}                self._pta_values["{info.name}"] = float(_v)')
                    lines.append(f"{ind}        for _k, _s in _res.items():")
                    lines.append(f"{ind}            if _s is not None and len(_s) > 0:")
                    lines.append(f"{ind}                _v = _s.iloc[-1]")
                    lines.append(f"{ind}                if not pd.isna(_v):")
                    lines.append(
                        f'{ind}                    self._pta_values["{info.name}_" + _k] = float(_v)'
                    )
                else:
                    lines.append(f"{ind}    if _res is not None and len(_res) > 0:")
                    lines.append(f"{ind}        _v = _res.iloc[-1]")
                    if spec.needs_context:
                        # NaN is a real answer for aux data (hole / before coverage):
                        # keeping the last valid value would trade on stale context.
                        lines.append(f'{ind}        self._pta_values["{info.name}"] = float(_v)')
                    else:
                        lines.append(f"{ind}        if not pd.isna(_v):")
                        lines.append(f'{ind}            self._pta_values["{info.name}"] = float(_v)')

        return lines

    _DSL_CONFIG_FIELDS: tuple[str, ...] = (
        "period",
        "fast_period",
        "slow_period",
        "signal_period",
        "d_period",
        "std_dev",
        "atr_multiplier",
    )

    # Spec-default param names that alias a DSL-native field (STOCH declares
    # ``period_k``/``period_d`` while the DSL field is ``period``/``d_period``).
    # When the DSL field is set the alias is dropped so exactly one config
    # field drives the value.
    _PARAM_ALIASES: ClassVar[dict[str, str]] = {
        "period_k": "period",
        "k_period": "period",
        "period_d": "d_period",
    }

    @staticmethod
    def _merge_effective_params(info: IndicatorInfo) -> dict[str, object]:
        """Overlay DSL IndicatorConfig overrides on top of spec defaults.

        Single source for the compute_fn param set: config fields, the
        runtime ``_pta_params`` dict and the compile-time lookback all derive
        from it.

        Both schema-native fields (``period``, ``fast_period``, ...) and
        plugin-declared extras (ADAPTIVE_RSI's ``alpha``, etc.) are folded
        in. Extras are pulled from pydantic's ``model_extra`` so plugin
        params flow through without schema edits.
        """
        merged: dict[str, object] = dict(info.spec.default_params)
        for dsl_field in StrategyCompiler._DSL_CONFIG_FIELDS:
            val = getattr(info.config, dsl_field, None)
            if val is not None:
                merged[dsl_field] = val
                for alias, target in StrategyCompiler._PARAM_ALIASES.items():
                    if target == dsl_field:
                        merged.pop(alias, None)
        extras = getattr(info.config, "model_extra", None) or {}
        for key, val in extras.items():
            if val is not None and key in info.spec.param_schema:
                merged[key] = val
        return merged

    @staticmethod
    def _pta_param_fields(
        info: IndicatorInfo,
    ) -> list[tuple[str, str, str, object, bool]]:
        """Config fields backing a compute_fn indicator's params.

        Returns ``(param_key, config_field, type_name, default, native)``
        tuples. ``native`` marks DSL-native fields (``period`` etc.) that the
        indicator-parameters block of the config class already emits.
        """
        out: list[tuple[str, str, str, object, bool]] = []
        for key, val in StrategyCompiler._merge_effective_params(info).items():
            field_name = f"{info.name}_{key}"
            native = (
                key in StrategyCompiler._DSL_CONFIG_FIELDS
                and getattr(info.config, key, None) is not None
            )
            declared = info.spec.param_schema.get(key)
            if declared in (int, float, bool, str):
                type_name = declared.__name__
            elif isinstance(val, (bool, int, float, str)):
                type_name = type(val).__name__
            else:
                msg = (
                    f"Indicator '{info.name}' ({info.config.type}) param '{key}' has "
                    f"unsupported value type {type(val).__name__}"
                )
                raise CompilerError(msg)
            out.append((key, field_name, type_name, val, native))
        return out

    @classmethod
    def _compute_pta_buffer_cap(cls, pta_infos: list[IndicatorInfo]) -> int:
        """Default-param bar-buffer cap for the compute_fn path (0 = unbounded).

        The generated strategy recomputes this at runtime from its config via
        :func:`vibe_quant.dsl.indicators.pta_buffer_cap`.
        """
        return pta_buffer_cap(
            [cls._get_pta_lookback(i) for i in pta_infos],
            full_history=any(i.spec.requires_full_history for i in pta_infos),
        )

    @staticmethod
    def _get_pta_lookback(info: IndicatorInfo) -> int:
        """Default-param lookback (see :func:`vibe_quant.dsl.indicators.pta_lookback`)."""
        return pta_lookback(info.spec, StrategyCompiler._merge_effective_params(info))

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
            prev_right = self._crossover_prev_right_code(cond, right)
            prev_guard = self._crossover_prev_guard(cond)
            return f"cond_{index} = ({prev_guard}) and ({left} > {right}) and ({prev_left} <= {prev_right})"
        elif cond.operator == Operator.CROSSES_BELOW:
            prev_left = self._operand_to_prev_code(cond.left)
            prev_right = self._crossover_prev_right_code(cond, right)
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

    def _crossover_prev_right_code(self, cond: Condition, right_code: str) -> str:
        """Right-side code for the previous-bar half of a crossover check.

        Numeric literals have no previous value, but they may be lifted into
        a sweepable config threshold. The prev-side comparison must use that
        SAME config value — otherwise overriding the threshold via sweep
        params tests `x > config_threshold and prev_x <= <original literal>`,
        which is a different (and wrong) condition.
        """
        from vibe_quant.dsl.conditions import Operand

        if (
            isinstance(cond.right, Operand)
            and not cond.right.is_indicator
            and not cond.right.is_price
        ):
            return right_code
        return self._operand_to_prev_code(cond.right)

    @staticmethod
    def _crossover_prev_guard(cond: Condition) -> str:
        """Generate guard expression ensuring prev values exist for crossover.

        Returns 'True' if no operand needs guarding, otherwise an 'in' check
        (indicators AND prices) so the first bar after warmup never fires.
        """
        from vibe_quant.dsl.conditions import Operand

        checks: list[str] = []
        for operand in (cond.left, cond.right):
            if isinstance(operand, Operand) and operand.is_indicator:
                checks.append(f'"{operand.value}" in self._prev_values')
            elif isinstance(operand, Operand) and operand.is_price:
                key = StrategyCompiler._prev_price_key(str(operand.value))
                checks.append(f'"{key}" in self._prev_values')
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
        if operand.is_price:
            key = self._prev_price_key(str(operand.value))
            return f'self._prev_values.get("{key}", 0.0)'
        # Literals have no previous value
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
