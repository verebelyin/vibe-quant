"""Tests for per-direction SL/TP in DSL schema."""
from __future__ import annotations

import pytest
from vibe_quant.dsl.schema import StrategyDSL


def _base_dsl(**overrides) -> dict:
    """Minimal valid DSL dict."""
    d = {
        "name": "test_strategy",
        "timeframe": "5m",
        "indicators": {"rsi": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi > 50"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }
    d.update(overrides)
    return d


class TestPerDirectionSLTP:
    def test_backward_compatible_no_per_direction(self) -> None:
        """Existing strategies without per-direction fields still work."""
        strategy = StrategyDSL(**_base_dsl())
        assert strategy.stop_loss.percent == 2.0
        assert strategy.stop_loss_long is None
        assert strategy.stop_loss_short is None
        assert strategy.take_profit_long is None
        assert strategy.take_profit_short is None

    def test_per_direction_stop_loss(self) -> None:
        strategy = StrategyDSL(**_base_dsl(
            stop_loss_long={"type": "fixed_pct", "percent": 1.09},
            stop_loss_short={"type": "fixed_pct", "percent": 8.29},
        ))
        assert strategy.stop_loss_long.percent == 1.09
        assert strategy.stop_loss_short.percent == 8.29
        assert strategy.stop_loss.percent == 2.0  # base still there

    def test_per_direction_take_profit(self) -> None:
        strategy = StrategyDSL(**_base_dsl(
            take_profit_long={"type": "fixed_pct", "percent": 17.13},
            take_profit_short={"type": "fixed_pct", "percent": 13.06},
        ))
        assert strategy.take_profit_long.percent == 17.13
        assert strategy.take_profit_short.percent == 13.06

    def test_per_direction_atr_validates_indicator(self) -> None:
        """Per-direction ATR SL must reference existing indicator."""
        with pytest.raises(Exception):
            StrategyDSL(**_base_dsl(
                stop_loss_long={"type": "atr_fixed", "atr_multiplier": 1.5, "indicator": "nonexistent"},
            ))

    def test_per_direction_atr_valid_indicator(self) -> None:
        strategy = StrategyDSL(**_base_dsl(
            indicators={"rsi": {"type": "RSI"}, "atr_main": {"type": "ATR"}},
            stop_loss_long={"type": "atr_fixed", "atr_multiplier": 1.5, "indicator": "atr_main"},
        ))
        assert strategy.stop_loss_long.atr_multiplier == 1.5

    def test_only_long_override(self) -> None:
        """Can override just one direction."""
        strategy = StrategyDSL(**_base_dsl(
            stop_loss_long={"type": "fixed_pct", "percent": 1.0},
        ))
        assert strategy.stop_loss_long.percent == 1.0
        assert strategy.stop_loss_short is None
