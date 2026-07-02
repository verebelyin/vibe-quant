"""Tests for calculation-correctness fixes in the DSL compiler/templates.

Covers beads:
- vibe-quant-mzl94: time-filter/funding early returns must update prev values
- vibe-quant-6x71o: crossover prev-side must use the config threshold
- vibe-quant-njy9l: trailing stop state reset between positions
- vibe-quant-u1i0p: bounded pandas-ta bar buffer (O(n^2) -> O(n*w))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl import StrategyCompiler, parse_strategy_string
from vibe_quant.dsl.compute_builtins import (
    compute_bbands,
    compute_macd,
    compute_rsi,
    compute_stoch,
)


@pytest.fixture
def compiler() -> StrategyCompiler:
    return StrategyCompiler()


# =============================================================================
# vibe-quant-mzl94: prev values updated on time-filter/funding early return
# =============================================================================


class TestTimeFilterPrevValues:
    YAML = """
name: tf_prev_test
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
time_filters:
  allowed_sessions:
    - start: "08:00"
      end: "20:00"
      timezone: UTC
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

    def test_blocked_bars_still_update_prev_values(self, compiler: StrategyCompiler) -> None:
        """Each early return after indicators are ready must refresh prev values."""
        dsl = parse_strategy_string(self.YAML)
        source = compiler.compile(dsl)
        compile(source, "<generated>", "exec")

        lines = [line.strip() for line in source.splitlines()]

        def next_two(after: str) -> tuple[str, str]:
            idx = lines.index(after)
            return lines[idx + 1], lines[idx + 2]

        # Time-filter early return updates prev values first
        first, second = next_two("if not self._check_time_filters(bar.ts_event):")
        assert first == "self._update_prev_values()"
        assert second == "return"

        # Funding-avoidance early return updates prev values first
        first, second = next_two("if self._is_near_funding_time(bar.ts_event):")
        assert first == "self._update_prev_values()"
        assert second == "return"


# =============================================================================
# vibe-quant-6x71o: crossover prev-side uses config threshold, not literal
# =============================================================================


class TestCrossoverThresholdConsistency:
    YAML = """
name: cross_threshold_test
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi crosses_above 50
exit_conditions:
  long:
    - rsi crosses_below 70
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""

    def test_prev_side_uses_config_threshold(self, compiler: StrategyCompiler) -> None:
        """Both sides of the crossover must reference the same config value."""
        dsl = parse_strategy_string(self.YAML)
        source = compiler.compile(dsl)
        compile(source, "<generated>", "exec")

        # Threshold params exist in config
        assert "rsi_50_0_threshold: float = 50.0" in source
        assert "rsi_70_0_threshold: float = 70.0" in source

        # crosses_above 50: current side > threshold AND prev side <= threshold
        assert "> self.config.rsi_50_0_threshold" in source
        assert "<= self.config.rsi_50_0_threshold" in source
        # crosses_below 70: current side < threshold AND prev side >= threshold
        assert "< self.config.rsi_70_0_threshold" in source
        assert ">= self.config.rsi_70_0_threshold" in source

        # No bare literal remains on the prev side of the cross checks
        assert "<= 50" not in source
        assert ">= 70" not in source

    def test_indicator_vs_indicator_cross_unchanged(self, compiler: StrategyCompiler) -> None:
        """Indicator-vs-indicator crossovers still use prev values on both sides."""
        yaml_content = """
name: cross_ind_test
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
        assert 'self._prev_values.get("ema_fast", 0.0)' in source
        assert 'self._prev_values.get("ema_slow", 0.0)' in source


# =============================================================================
# vibe-quant-njy9l: trailing stop state reset
# =============================================================================


class TestTrailingStopReset:
    YAML = """
name: trailing_reset_test
timeframe: 5m
indicators:
  atr:
    type: ATR
    period: 14
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: atr_trailing
  indicator: atr
  atr_multiplier: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""

    def test_trailing_state_initialized_and_reset(self, compiler: StrategyCompiler) -> None:
        dsl = parse_strategy_string(self.YAML)
        source = compiler.compile(dsl)
        compile(source, "<generated>", "exec")

        # Initialized in __init__
        assert "self._trailing_best_sl: float | None = None" in source
        # Reset when the position closes and on engine reset — a stale level
        # from a previous position must not gate or poison the next trail.
        assert source.count("self._trailing_best_sl = None") >= 2

    def test_position_closed_resets_trailing(self) -> None:
        from vibe_quant.dsl.templates import ON_EVENT_LINES, ON_RESET_LINES

        event_src = "\n".join(ON_EVENT_LINES)
        closed_block = event_src.split("PositionClosed")[1]
        assert "self._trailing_best_sl = None" in closed_block
        assert "self._trailing_best_sl = None" in "\n".join(ON_RESET_LINES)


# =============================================================================
# vibe-quant-u1i0p: bounded pandas-ta buffer
# =============================================================================


class TestPtaBufferCap:
    TEMA_YAML = """
name: buffer_cap_test
timeframe: 5m
indicators:
  tema:
    type: TEMA
    period: 20
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

    def test_windowed_indicator_gets_capped_buffer(self, compiler: StrategyCompiler) -> None:
        dsl = parse_strategy_string(self.TEMA_YAML)
        source = compiler.compile(dsl)
        compile(source, "<generated>", "exec")

        assert "self._pta_buffer_cap: int =" in source
        # TEMA lookback is 3*period = 60 -> cap = max(400, 600) = 600
        assert "self._pta_buffer_cap: int = 600" in source
        # Trim logic present in on_bar
        assert "del self._pta_close[:_trim]" in source
        assert "del self._pta_volume[:_trim]" in source

    def test_cumulative_indicator_disables_cap(self, compiler: StrategyCompiler) -> None:
        """OBV forced onto the compute_fn path must keep full history."""
        from vibe_quant.dsl.indicators import indicator_registry

        obv_spec = indicator_registry.get("OBV")
        assert obv_spec is not None
        assert obv_spec.requires_full_history is True
        vwap_spec = indicator_registry.get("VWAP")
        assert vwap_spec is not None
        assert vwap_spec.requires_full_history is True

    def test_min_cap_floor(self, compiler: StrategyCompiler) -> None:
        """Small lookbacks still get the minimum warmup floor."""
        yaml_content = """
name: buffer_floor_test
timeframe: 5m
indicators:
  willr:
    type: WILLR
    period: 14
entry_conditions:
  long:
    - willr < -80
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
        dsl = parse_strategy_string(yaml_content)
        source = compiler.compile(dsl)
        assert "self._pta_buffer_cap: int = 400" in source


class TestCappedBufferNumericEquivalence:
    """Capped-window indicator values must match full-history values after warmup.

    Validates the 10x-lookback convergence assumption behind the buffer cap.
    """

    @staticmethod
    def _make_ohlcv(n: int = 3000, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        volume = rng.uniform(100, 1000, n)
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
        )

    @pytest.mark.parametrize(
        ("fn", "params", "lookback"),
        [
            (compute_rsi, {"period": 14}, 14),
            (compute_macd, {"fast_period": 12, "slow_period": 26, "signal_period": 9}, 35),
            (compute_stoch, {"period_k": 14, "period_d": 3}, 17),
            (compute_bbands, {"period": 20, "std_dev": 2.0}, 20),
        ],
    )
    def test_last_value_matches_within_tolerance(
        self,
        fn: object,
        params: dict[str, object],
        lookback: int,
    ) -> None:
        df = self._make_ohlcv()
        cap = max(400, 10 * lookback)
        capped = df.tail(cap).reset_index(drop=True)

        full_res = fn(df, params)  # type: ignore[operator]
        capped_res = fn(capped, params)  # type: ignore[operator]

        if isinstance(full_res, dict):
            for key, series in full_res.items():
                full_v = float(series.iloc[-1])
                capped_v = float(capped_res[key].iloc[-1])
                assert capped_v == pytest.approx(full_v, rel=1e-6, abs=1e-9), key
        else:
            assert float(capped_res.iloc[-1]) == pytest.approx(
                float(full_res.iloc[-1]), rel=1e-6, abs=1e-9
            )
