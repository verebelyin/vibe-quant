"""Tests for DSL/compiler bug fixes (vibe-quant-nhy7, s36n, npzf, aqas, o5k3)."""

from __future__ import annotations

import pytest

from vibe_quant.dsl import StrategyCompiler, parse_strategy_string
from vibe_quant.dsl.indicators import indicator_registry
from vibe_quant.dsl.schema import VALID_INDICATOR_TYPES


# =============================================================================
# Bug 1: vibe-quant-nhy7 -- TEMA nt_class=None
# =============================================================================


class TestTemaFallback:
    """TEMA should use pandas-ta fallback, not generate None placeholder."""

    def test_tema_has_pandas_ta_func(self) -> None:
        """TEMA spec must have pandas_ta_func set."""
        spec = indicator_registry.get("TEMA")
        assert spec is not None
        assert spec.pandas_ta_func == "tema"

    def test_tema_nt_class_is_none(self) -> None:
        """TEMA nt_class is None (NT lacks TripleExponentialMovingAverage)."""
        spec = indicator_registry.get("TEMA")
        assert spec is not None
        assert spec.nt_class is None

    def test_tema_strategy_compiles_without_none_indicator(self) -> None:
        """Strategy with TEMA should compile; generated code should not assign None."""
        yaml_content = """
name: tema_test
timeframe: 5m
indicators:
  tema:
    type: TEMA
    period: 14
entry_conditions:
  long:
    - tema > 0
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        dsl = parse_strategy_string(yaml_content)
        compiler = StrategyCompiler()
        source = compiler.compile(dsl)
        # Should be valid Python
        compile(source, "<generated>", "exec")
        # Should NOT contain assignment to None that would crash at runtime
        # (the placeholder comment is acceptable, but using .initialized on None is not)
        assert "self.ind_tema.initialized" not in source


# =============================================================================
# Bug 2: vibe-quant-s36n -- PositionChanged imported but unused
# =============================================================================


class TestPositionChangedImport:
    """Generated code should not import unused PositionChanged."""

    def test_no_position_changed_import(self) -> None:
        """Compiled source must not import PositionChanged."""
        yaml_content = """
name: import_test
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
        dsl = parse_strategy_string(yaml_content)
        compiler = StrategyCompiler()
        source = compiler.compile(dsl)
        assert "PositionChanged" not in source

    def test_still_imports_position_opened_closed(self) -> None:
        """PositionOpened and PositionClosed should still be imported."""
        yaml_content = """
name: import_test2
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
        dsl = parse_strategy_string(yaml_content)
        compiler = StrategyCompiler()
        source = compiler.compile(dsl)
        assert "PositionOpened" in source
        assert "PositionClosed" in source
        assert "OrderFilled" in source


# =============================================================================
# Bug 3: vibe-quant-npzf -- Missing ICHIMOKU and VOLSMA
# =============================================================================


class TestIchimokuVolsma:
    """ICHIMOKU and VOLSMA must exist in schema and registry."""

    def test_ichimoku_in_valid_types(self) -> None:
        """ICHIMOKU in VALID_INDICATOR_TYPES."""
        assert "ICHIMOKU" in VALID_INDICATOR_TYPES

    def test_volsma_in_valid_types(self) -> None:
        """VOLSMA in VALID_INDICATOR_TYPES."""
        assert "VOLSMA" in VALID_INDICATOR_TYPES

    def test_ichimoku_in_registry(self) -> None:
        """ICHIMOKU registered with pandas-ta fallback."""
        spec = indicator_registry.get("ICHIMOKU")
        assert spec is not None
        assert spec.pandas_ta_func == "ichimoku"
        assert spec.nt_class is None
        assert len(spec.output_names) > 1

    def test_volsma_in_registry(self) -> None:
        """VOLSMA registered with pandas-ta fallback."""
        spec = indicator_registry.get("VOLSMA")
        assert spec is not None
        assert spec.pandas_ta_func == "sma"
        assert spec.nt_class is None
        assert spec.default_params == {"period": 20}

    def test_schema_and_registry_in_sync(self) -> None:
        """All VALID_INDICATOR_TYPES are in registry."""
        registered = set(indicator_registry.list_indicators())
        missing = set(VALID_INDICATOR_TYPES) - registered
        assert not missing, f"Schema types not in registry: {missing}"

    def test_volsma_strategy_parses(self) -> None:
        """Strategy using VOLSMA parses without error."""
        yaml_content = """
name: volsma_test
timeframe: 5m
indicators:
  volsma:
    type: VOLSMA
    period: 20
entry_conditions:
  long:
    - volsma > 0
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        strategy = parse_strategy_string(yaml_content)
        assert "volsma" in strategy.indicators
        assert strategy.indicators["volsma"].period == 20


# =============================================================================
# Bug 4: vibe-quant-aqas -- Docstring stutter
# =============================================================================


class TestDocstringStutter:
    """Generated docstrings should not say 'Check check ...'."""

    def test_no_check_check_in_docstring(self) -> None:
        """Docstring should say 'Check long entry' not 'Check check long entry'."""
        yaml_content = """
name: docstring_test
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
  short:
    - rsi > 70
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
        dsl = parse_strategy_string(yaml_content)
        compiler = StrategyCompiler()
        source = compiler.compile(dsl)
        # Should not have doubled "check check"
        assert "Check  check" not in source
        assert "Check check" not in source
        # Should have proper docstrings
        assert "Check long entry conditions" in source
        assert "Check short entry conditions" in source


# =============================================================================
# Bug 5: vibe-quant-o5k3 -- Interval parsing no 'd' suffix
# =============================================================================


class TestIntervalParsing:
    """_interval_to_minutes should handle m, h, d suffixes."""

    def test_minutes_suffix(self) -> None:
        """'5m' -> 5 minutes."""
        from vibe_quant.data.ingest import _interval_to_minutes

        assert _interval_to_minutes("5m") == 5
        assert _interval_to_minutes("15m") == 15
        assert _interval_to_minutes("1m") == 1

    def test_hours_suffix(self) -> None:
        """'1h' -> 60, '4h' -> 240."""
        from vibe_quant.data.ingest import _interval_to_minutes

        assert _interval_to_minutes("1h") == 60
        assert _interval_to_minutes("4h") == 240

    def test_days_suffix(self) -> None:
        """'1d' -> 1440."""
        from vibe_quant.data.ingest import _interval_to_minutes

        assert _interval_to_minutes("1d") == 1440
        assert _interval_to_minutes("7d") == 7 * 1440

    def test_invalid_suffix_raises(self) -> None:
        """Unrecognized suffix raises ValueError."""
        from vibe_quant.data.ingest import _interval_to_minutes

        with pytest.raises(ValueError, match="Unrecognized interval format"):
            _interval_to_minutes("1w")

    def test_case_insensitive(self) -> None:
        """Should handle uppercase suffixes."""
        from vibe_quant.data.ingest import _interval_to_minutes

        assert _interval_to_minutes("1D") == 1440
        assert _interval_to_minutes("4H") == 240
        assert _interval_to_minutes("5M") == 5
