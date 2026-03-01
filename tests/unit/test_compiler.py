"""Tests for DSL-to-NautilusTrader compiler."""

from __future__ import annotations

import pytest

from vibe_quant.dsl import (
    StrategyCompiler,
    parse_strategy_string,
)
from vibe_quant.dsl.compiler import CompilerError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_strategy_yaml() -> str:
    """Minimal valid strategy YAML."""
    return """
name: test_minimal
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""


@pytest.fixture
def multi_tf_strategy_yaml() -> str:
    """Multi-timeframe strategy YAML."""
    return """
name: multi_tf_strategy
description: "Multi-timeframe RSI with EMA trend filter"
version: 1
timeframe: 5m
additional_timeframes:
  - 1h
  - 4h
indicators:
  rsi:
    type: RSI
    period: 14
  rsi_1h:
    type: RSI
    period: 14
    timeframe: 1h
  ema_trend:
    type: EMA
    period: 50
    timeframe: 4h
entry_conditions:
  long:
    - rsi < 30
    - rsi_1h > 40
    - close > ema_trend
  short:
    - rsi > 70
    - rsi_1h < 60
    - close < ema_trend
exit_conditions:
  long:
    - rsi > 50
  short:
    - rsi < 50
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""


@pytest.fixture
def time_filters_strategy_yaml() -> str:
    """Strategy with time filters."""
    return """
name: time_filtered_strategy
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
time_filters:
  allowed_sessions:
    - start: "08:00"
      end: "20:00"
      timezone: UTC
  blocked_days:
    - Saturday
    - Sunday
  avoid_around_funding:
    enabled: true
    minutes_before: 5
    minutes_after: 5
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""


@pytest.fixture
def crossover_strategy_yaml() -> str:
    """Strategy with crossover conditions."""
    return """
name: crossover_strategy
timeframe: 5m
indicators:
  ema_fast:
    type: EMA
    period: 9
  ema_slow:
    type: EMA
    period: 21
entry_conditions:
  long:
    - ema_fast crosses_above ema_slow
  short:
    - ema_fast crosses_below ema_slow
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""


@pytest.fixture
def between_strategy_yaml() -> str:
    """Strategy with BETWEEN condition."""
    return """
name: between_strategy
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi between 30 50
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""


@pytest.fixture
def compiler() -> StrategyCompiler:
    """Create a StrategyCompiler instance."""
    return StrategyCompiler()


# =============================================================================
# Compile to Source Tests
# =============================================================================


class TestCompileMinimalStrategy:
    """Tests for compiling minimal strategy."""

    def test_compile_produces_valid_python(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Compiled code should be valid Python."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        # Should be valid Python (no syntax errors)
        compile(source, "<generated>", "exec")

    def test_compile_generates_config_class(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Compiled code should contain config class."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        assert "class TestMinimalConfig(StrategyConfig):" in source
        assert "rsi_period: int = 14" in source

    def test_compile_generates_strategy_class(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Compiled code should contain strategy class."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        assert "class TestMinimalStrategy(Strategy):" in source
        assert "def on_start(self)" in source
        assert "def on_bar(self, bar: Bar)" in source

    @pytest.mark.skip(reason="Requires indicator registry NT class loading fix")
    def test_compile_generates_indicator_import(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Compiled code should import indicator class."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        assert "RelativeStrengthIndex" in source

    def test_compile_generates_bar_type_subscription(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Compiled code should subscribe to bars."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        assert "self.subscribe_bars(self.bar_type_5m)" in source
        assert "5-MINUTE" in source


class TestCompileMultiTFStrategy:
    """Tests for compiling multi-timeframe strategy."""

    def test_compile_multi_tf_produces_valid_python(
        self, compiler: StrategyCompiler, multi_tf_strategy_yaml: str
    ) -> None:
        """Multi-TF strategy should compile to valid Python."""
        dsl = parse_strategy_string(multi_tf_strategy_yaml)
        source = compiler.compile(dsl)

        compile(source, "<generated>", "exec")

    def test_compile_multi_tf_subscribes_all_timeframes(
        self, compiler: StrategyCompiler, multi_tf_strategy_yaml: str
    ) -> None:
        """Multi-TF strategy should subscribe to all timeframes."""
        dsl = parse_strategy_string(multi_tf_strategy_yaml)
        source = compiler.compile(dsl)

        assert "self.subscribe_bars(self.bar_type_5m)" in source
        assert "self.subscribe_bars(self.bar_type_1h)" in source
        assert "self.subscribe_bars(self.bar_type_4h)" in source

    @pytest.mark.skip(reason="Requires indicator registry NT class loading fix")
    def test_compile_multi_tf_registers_indicators_on_correct_bars(
        self, compiler: StrategyCompiler, multi_tf_strategy_yaml: str
    ) -> None:
        """Indicators should register on their respective timeframe bars."""
        dsl = parse_strategy_string(multi_tf_strategy_yaml)
        source = compiler.compile(dsl)

        # RSI on 5m
        assert "self.register_indicator_for_bars(self.bar_type_5m, self.ind_rsi)" in source
        # RSI on 1h
        assert "self.register_indicator_for_bars(self.bar_type_1h, self.ind_rsi_1h)" in source
        # EMA on 4h
        assert "self.register_indicator_for_bars(self.bar_type_4h, self.ind_ema_trend)" in source

    def test_compile_multi_tf_has_all_entry_conditions(
        self, compiler: StrategyCompiler, multi_tf_strategy_yaml: str
    ) -> None:
        """Multi-TF strategy should have all entry conditions."""
        dsl = parse_strategy_string(multi_tf_strategy_yaml)
        source = compiler.compile(dsl)

        assert "_check_long_entry" in source
        assert "_check_short_entry" in source

    def test_compile_multi_tf_has_exit_conditions(
        self, compiler: StrategyCompiler, multi_tf_strategy_yaml: str
    ) -> None:
        """Multi-TF strategy should have exit conditions."""
        dsl = parse_strategy_string(multi_tf_strategy_yaml)
        source = compiler.compile(dsl)

        assert "_check_long_exit" in source
        assert "_check_short_exit" in source


class TestCompileWithTimeFilters:
    """Tests for compiling strategy with time filters."""

    def test_compile_time_filters_produces_valid_python(
        self, compiler: StrategyCompiler, time_filters_strategy_yaml: str
    ) -> None:
        """Strategy with time filters should compile to valid Python."""
        dsl = parse_strategy_string(time_filters_strategy_yaml)
        source = compiler.compile(dsl)

        compile(source, "<generated>", "exec")

    def test_compile_time_filters_generates_session_check(
        self, compiler: StrategyCompiler, time_filters_strategy_yaml: str
    ) -> None:
        """Strategy should check session times."""
        dsl = parse_strategy_string(time_filters_strategy_yaml)
        source = compiler.compile(dsl)

        assert "_check_time_filters" in source
        assert "08:00" in source or "dt_time(8, 0)" in source
        assert "20:00" in source or "dt_time(20, 0)" in source

    def test_compile_time_filters_generates_blocked_days_check(
        self, compiler: StrategyCompiler, time_filters_strategy_yaml: str
    ) -> None:
        """Strategy should check blocked days."""
        dsl = parse_strategy_string(time_filters_strategy_yaml)
        source = compiler.compile(dsl)

        assert "blocked_days" in source
        # Saturday=5, Sunday=6
        assert "5" in source
        assert "6" in source

    def test_compile_time_filters_generates_funding_avoidance(
        self, compiler: StrategyCompiler, time_filters_strategy_yaml: str
    ) -> None:
        """Strategy should check funding avoidance."""
        dsl = parse_strategy_string(time_filters_strategy_yaml)
        source = compiler.compile(dsl)

        assert "_is_near_funding_time" in source
        assert "funding_hours" in source


class TestConditionCodeGeneration:
    """Tests for condition code generation."""

    def test_compile_crossover_conditions(
        self, compiler: StrategyCompiler, crossover_strategy_yaml: str
    ) -> None:
        """Crossover conditions should generate correct code."""
        dsl = parse_strategy_string(crossover_strategy_yaml)
        source = compiler.compile(dsl)

        compile(source, "<generated>", "exec")
        # Should check current and previous values
        assert "_prev_values" in source
        assert "_update_prev_values" in source

    def test_compile_between_conditions(
        self, compiler: StrategyCompiler, between_strategy_yaml: str
    ) -> None:
        """BETWEEN conditions should generate correct code."""
        dsl = parse_strategy_string(between_strategy_yaml)
        source = compiler.compile(dsl)

        compile(source, "<generated>", "exec")
        # Should have range check
        assert "30" in source
        assert "50" in source

    def test_compile_comparison_operators(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Comparison operators should generate correct code."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        # RSI < 30 condition
        assert "<" in source or "30" in source


class TestCompileToModule:
    """Tests for compile_to_module."""

    def test_compile_to_module_returns_module(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """compile_to_module should return a module."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        module = compiler.compile_to_module(dsl)

        from types import ModuleType

        assert isinstance(module, ModuleType)

    def test_compile_to_module_has_config_class(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Module should have config class."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        module = compiler.compile_to_module(dsl)

        assert hasattr(module, "TestMinimalConfig")

    def test_compile_to_module_has_strategy_class(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Module should have strategy class."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        module = compiler.compile_to_module(dsl)

        assert hasattr(module, "TestMinimalStrategy")

    @pytest.mark.skip(reason="StrategyConfig uses msgspec, not stdlib dataclass")
    def test_compile_to_module_config_is_dataclass(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Config class should be a dataclass."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        module = compiler.compile_to_module(dsl)

        from dataclasses import is_dataclass

        assert is_dataclass(module.TestMinimalConfig)

    def test_compile_to_module_strategy_has_required_methods(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Strategy class should have required methods."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        module = compiler.compile_to_module(dsl)

        strategy_cls = module.TestMinimalStrategy
        assert hasattr(strategy_cls, "on_start")
        assert hasattr(strategy_cls, "on_bar")

    def test_compile_to_module_rejects_unsafe_calls(
        self,
        compiler: StrategyCompiler,
        minimal_strategy_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AST validation should block unsafe runtime calls in generated code."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        monkeypatch.setattr(
            compiler,
            "compile",
            lambda _dsl: "def _bad():\n    exec('print(1)')\n",
        )

        with pytest.raises(CompilerError, match="Unsafe call"):
            compiler.compile_to_module(dsl)

    def test_compile_to_module_rejects_disallowed_imports(
        self,
        compiler: StrategyCompiler,
        minimal_strategy_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AST validation should block non-whitelisted imports."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        monkeypatch.setattr(
            compiler,
            "compile",
            lambda _dsl: "import subprocess\nx = 1\n",
        )

        with pytest.raises(CompilerError, match="Disallowed import"):
            compiler.compile_to_module(dsl)


class TestConfigClassGeneration:
    """Tests for config class generation."""

    def test_config_has_instrument_id(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Config should have instrument_id parameter."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        assert "instrument_id" in source

    def test_config_has_indicator_params(
        self, compiler: StrategyCompiler, multi_tf_strategy_yaml: str
    ) -> None:
        """Config should have all indicator parameters."""
        dsl = parse_strategy_string(multi_tf_strategy_yaml)
        source = compiler.compile(dsl)

        assert "rsi_period" in source
        assert "rsi_1h_period" in source
        assert "ema_trend_period" in source

    def test_config_has_stop_loss_params(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Config should have stop loss parameters."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        assert "stop_loss_type" in source
        assert "stop_loss_percent" in source

    def test_config_has_take_profit_params(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Config should have take profit parameters."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        assert "take_profit_type" in source
        assert "take_profit_percent" in source


class TestHelperMethods:
    """Tests for helper method generation."""

    @pytest.mark.skip(reason="Requires indicator registry NT class loading fix")
    def test_generates_indicators_ready_method(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Should generate _indicators_ready method."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        assert "_indicators_ready" in source
        assert ".initialized" in source

    def test_generates_get_indicator_value_method(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Should generate _get_indicator_value method."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        assert "_get_indicator_value" in source

    def test_generates_order_methods(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Should generate order submission methods."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        assert "_submit_long_entry" in source
        assert "_submit_exit" in source
        assert "OrderSide.BUY" in source


class TestEdgeCases:
    """Tests for edge cases."""

    def test_compile_with_macd_indicator(self, compiler: StrategyCompiler) -> None:
        """MACD indicator should compile correctly."""
        yaml_content = """
name: macd_strategy
timeframe: 5m
indicators:
  macd:
    type: MACD
    fast_period: 12
    slow_period: 26
    signal_period: 9
entry_conditions:
  long:
    - macd > 0
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        dsl = parse_strategy_string(yaml_content)
        source = compiler.compile(dsl)

        compile(source, "<generated>", "exec")
        assert "fast_period" in source
        assert "slow_period" in source
        assert "signal_period" in source

    def test_macd_signal_forces_pandas_ta_fallback(
        self, compiler: StrategyCompiler
    ) -> None:
        """MACD with .signal/.histogram conditions must use pandas-ta, not NT.

        NT MACD only exposes .value (MACD line). When signal/histogram
        are referenced, compiler must switch to pandas-ta and generate
        proper multi-output extraction (macd, signal, histogram).
        """
        yaml_content = """
name: macd_signal_strategy
timeframe: 5m
indicators:
  macd:
    type: MACD
    fast_period: 12
    slow_period: 26
    signal_period: 9
entry_conditions:
  long:
    - macd.signal > 0
  short:
    - macd.histogram < 0
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        dsl = parse_strategy_string(yaml_content)
        source = compiler.compile(dsl)

        compile(source, "<generated>", "exec")
        # Must use pandas-ta (ta.macd), not NT MACD class
        assert "ta.macd(" in source
        # Must extract all 3 sub-outputs
        assert '"macd_signal"' in source
        assert '"macd_histogram"' in source
        assert '"macd_macd"' in source
        # Must NOT instantiate NT MACD
        assert "MovingAverageConvergenceDivergence(" not in source

    def test_compile_with_bbands_indicator(self, compiler: StrategyCompiler) -> None:
        """Bollinger Bands indicator should compile correctly."""
        yaml_content = """
name: bbands_strategy
timeframe: 5m
indicators:
  bb:
    type: BBANDS
    period: 20
    std_dev: 2.0
entry_conditions:
  long:
    - bb < 0
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        dsl = parse_strategy_string(yaml_content)
        source = compiler.compile(dsl)

        compile(source, "<generated>", "exec")
        assert "bb_period" in source
        assert "bb_std_dev" in source

    def test_compile_with_atr_stop_loss(self, compiler: StrategyCompiler) -> None:
        """ATR-based stop loss should compile correctly."""
        yaml_content = """
name: atr_sl_strategy
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
  atr:
    type: ATR
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: atr_trailing
  atr_multiplier: 2.0
  indicator: atr
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        dsl = parse_strategy_string(yaml_content)
        source = compiler.compile(dsl)

        compile(source, "<generated>", "exec")
        assert "stop_loss_atr_multiplier" in source
        assert "stop_loss_indicator" in source

    def test_compile_with_price_comparison(self, compiler: StrategyCompiler) -> None:
        """Price comparison conditions should compile correctly."""
        yaml_content = """
name: price_compare_strategy
timeframe: 5m
indicators:
  ema:
    type: EMA
    period: 20
entry_conditions:
  long:
    - close > ema
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        dsl = parse_strategy_string(yaml_content)
        source = compiler.compile(dsl)

        compile(source, "<generated>", "exec")
        # Should reference bar.close
        assert "bar.close" in source or "bar." in source

    def test_position_side_not_entry_in_on_event(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Regression: on_event must use pos.side, not pos.entry.

        pos.entry returns OrderSide (BUY=1/SELL=2), but PositionSide
        (LONG=2/SHORT=3) has overlapping numeric values. Cross-enum
        comparison pos.entry == PositionSide.LONG is True for SHORT
        positions (both == 2), causing SL/TP with wrong side.
        """
        dsl = parse_strategy_string(minimal_strategy_yaml)
        source = compiler.compile(dsl)

        assert "pos.side == PositionSide.LONG" in source
        assert "pos.entry == PositionSide.LONG" not in source
        assert "pos.entry == PositionSide.SHORT" not in source

    def test_condition_helpers_receive_bar_parameter(self, compiler: StrategyCompiler) -> None:
        """Condition check helpers must accept bar param for price operands."""
        yaml_content = """
name: bar_scope_strategy
timeframe: 5m
indicators:
  ema:
    type: EMA
    period: 20
entry_conditions:
  long:
    - close > ema
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        dsl = parse_strategy_string(yaml_content)
        source = compiler.compile(dsl)

        # Helper definition must include bar parameter
        assert "def _check_long_entry(self, bar" in source
        # Call site must pass bar
        assert "_check_long_entry(bar)" in source
        # Must NOT have parameterless call
        assert "_check_long_entry()" not in source


# =============================================================================
# Integration Tests
# =============================================================================


class TestCompiledModuleImports:
    """Tests that verify compiled modules can import."""

    def test_compiled_module_in_sys_modules(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Compiled module should be registered in sys.modules."""
        import sys

        dsl = parse_strategy_string(minimal_strategy_yaml)
        compiler.compile_to_module(dsl)

        assert "vibe_quant.dsl.generated.test_minimal" in sys.modules

    def test_compiled_module_can_instantiate_config(
        self, compiler: StrategyCompiler, minimal_strategy_yaml: str
    ) -> None:
        """Should be able to instantiate config from compiled module."""
        dsl = parse_strategy_string(minimal_strategy_yaml)
        module = compiler.compile_to_module(dsl)

        # Create config instance
        config = module.TestMinimalConfig(instrument_id="BTCUSDT.BINANCE")
        assert config.instrument_id == "BTCUSDT.BINANCE"
        assert config.rsi_period == 14


class TestPerDirectionSLTPCompiler:
    def test_per_direction_sl_emits_config_attrs(self) -> None:
        """Compiler should emit stop_loss_long_* attrs when per-direction SL is set."""
        from vibe_quant.dsl.compiler import StrategyCompiler
        from vibe_quant.dsl.schema import StrategyDSL

        dsl = StrategyDSL(
            name="test_per_dir",
            timeframe="5m",
            indicators={"rsi": {"type": "RSI", "period": 14}},
            entry_conditions={"long": ["rsi > 50"]},
            stop_loss={"type": "fixed_pct", "percent": 2.0},
            take_profit={"type": "fixed_pct", "percent": 4.0},
            stop_loss_long={"type": "fixed_pct", "percent": 1.09},
            stop_loss_short={"type": "fixed_pct", "percent": 8.29},
        )
        compiler = StrategyCompiler()
        source = compiler.compile(dsl)
        assert "stop_loss_long_type" in source
        assert "stop_loss_long_percent" in source
        assert "1.09" in source
        assert "stop_loss_short_type" in source
        assert "8.29" in source

    def test_per_direction_tp_emits_config_attrs(self) -> None:
        from vibe_quant.dsl.compiler import StrategyCompiler
        from vibe_quant.dsl.schema import StrategyDSL

        dsl = StrategyDSL(
            name="test_per_dir_tp",
            timeframe="5m",
            indicators={"rsi": {"type": "RSI", "period": 14}},
            entry_conditions={"long": ["rsi > 50"]},
            stop_loss={"type": "fixed_pct", "percent": 2.0},
            take_profit={"type": "fixed_pct", "percent": 4.0},
            take_profit_long={"type": "fixed_pct", "percent": 17.13},
            take_profit_short={"type": "fixed_pct", "percent": 13.06},
        )
        compiler = StrategyCompiler()
        source = compiler.compile(dsl)
        assert "take_profit_long_type" in source
        assert "17.13" in source
        assert "take_profit_short_type" in source
        assert "13.06" in source

    def test_no_per_direction_no_extra_attrs(self) -> None:
        """Without per-direction, should not emit *_long_* or *_short_* attrs."""
        from vibe_quant.dsl.compiler import StrategyCompiler
        from vibe_quant.dsl.schema import StrategyDSL

        dsl = StrategyDSL(
            name="test_unified",
            timeframe="5m",
            indicators={"rsi": {"type": "RSI", "period": 14}},
            entry_conditions={"long": ["rsi > 50"]},
            stop_loss={"type": "fixed_pct", "percent": 2.0},
            take_profit={"type": "fixed_pct", "percent": 4.0},
        )
        compiler = StrategyCompiler()
        source = compiler.compile(dsl)
        assert "stop_loss_long_type" not in source
        assert "stop_loss_short_type" not in source
        assert "take_profit_long_type" not in source
        assert "take_profit_short_type" not in source
