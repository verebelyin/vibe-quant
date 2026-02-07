"""Tests for dashboard strategy management tab."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from vibe_quant.db.state_manager import StateManager


class TestStrategyManagementModule:
    """Test module import and basic functions."""

    def test_module_import(self) -> None:
        """Test that strategy management module can be imported."""
        from vibe_quant.dashboard.pages import strategy_management

        assert hasattr(strategy_management, "render_strategy_management_tab")
        assert hasattr(strategy_management, "_validate_dsl")
        assert hasattr(strategy_management, "_get_default_dsl_yaml")

    def test_validate_dsl_valid(self) -> None:
        """Test DSL validation with valid YAML."""
        from vibe_quant.dashboard.pages.strategy_management import _validate_dsl

        valid_yaml = """
name: test_strategy
description: Test strategy
version: 1
timeframe: 1h
indicators:
  rsi_14:
    type: RSI
    period: 14
  atr_14:
    type: ATR
    period: 14
entry_conditions:
  long:
    - "rsi_14 < 30"
stop_loss:
  type: atr_fixed
  atr_multiplier: 2.0
  indicator: atr_14
take_profit:
  type: risk_reward
  risk_reward_ratio: 2.0
"""
        model, error = _validate_dsl(valid_yaml)
        assert error is None
        assert model is not None
        assert model.name == "test_strategy"
        assert "rsi_14" in model.indicators

    def test_validate_dsl_invalid_yaml(self) -> None:
        """Test DSL validation with invalid YAML syntax."""
        from vibe_quant.dashboard.pages.strategy_management import _validate_dsl

        invalid_yaml = """
name: test
  invalid: indentation
"""
        model, error = _validate_dsl(invalid_yaml)
        assert model is None
        assert error is not None
        assert "YAML" in error or "parse" in error.lower()

    def test_validate_dsl_missing_required(self) -> None:
        """Test DSL validation with missing required fields."""
        from vibe_quant.dashboard.pages.strategy_management import _validate_dsl

        incomplete_yaml = """
name: test_strategy
timeframe: 1h
indicators:
  rsi_14:
    type: RSI
"""
        model, error = _validate_dsl(incomplete_yaml)
        assert model is None
        assert error is not None
        # Should complain about missing entry_conditions or stop_loss/take_profit

    def test_validate_dsl_invalid_indicator_type(self) -> None:
        """Test DSL validation with invalid indicator type."""
        from vibe_quant.dashboard.pages.strategy_management import _validate_dsl

        invalid_indicator_yaml = """
name: test_strategy
timeframe: 1h
indicators:
  invalid_ind:
    type: INVALID_TYPE
    period: 14
entry_conditions:
  long:
    - "invalid_ind < 30"
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 4.0
"""
        model, error = _validate_dsl(invalid_indicator_yaml)
        assert model is None
        assert error is not None
        assert "INVALID_TYPE" in error or "indicator" in error.lower()

    def test_default_dsl_yaml_valid(self) -> None:
        """Test that default DSL YAML template is valid."""
        from vibe_quant.dashboard.pages.strategy_management import (
            _get_default_dsl_yaml,
            _validate_dsl,
        )

        default_yaml = _get_default_dsl_yaml()
        model, error = _validate_dsl(default_yaml)
        assert error is None, f"Default YAML validation failed: {error}"
        assert model is not None
        assert model.name == "my_strategy"

    def test_validate_dsl_non_dict(self) -> None:
        """Test DSL validation with non-dict YAML."""
        from vibe_quant.dashboard.pages.strategy_management import _validate_dsl

        non_dict_yaml = """
- item1
- item2
"""
        model, error = _validate_dsl(non_dict_yaml)
        assert model is None
        assert error is not None
        assert "mapping" in error.lower()


class TestStrategyManagerIntegration:
    """Integration tests with StateManager."""

    @pytest.fixture
    def temp_db(self) -> Path:
        """Create temp database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            return Path(f.name)

    @pytest.fixture
    def manager(self, temp_db: Path) -> Generator[StateManager]:
        """Create StateManager with temp database."""
        mgr = StateManager(temp_db)
        yield mgr
        mgr.close()

    def test_create_and_list_strategy(self, manager: StateManager) -> None:
        """Test creating and listing strategies via StateManager."""
        dsl_dict = {
            "name": "test_strategy",
            "description": "Test",
            "version": 1,
            "timeframe": "1h",
            "additional_timeframes": [],
            "indicators": {
                "rsi_14": {"type": "RSI", "period": 14, "source": "close"},
                "atr_14": {"type": "ATR", "period": 14, "source": "close"},
            },
            "entry_conditions": {"long": ["rsi_14 < 30"], "short": []},
            "exit_conditions": {"long": [], "short": []},
            "time_filters": {
                "allowed_sessions": [],
                "blocked_days": [],
                "avoid_around_funding": {"enabled": False, "minutes_before": 5, "minutes_after": 5},
            },
            "stop_loss": {"type": "atr_fixed", "atr_multiplier": 2.0, "indicator": "atr_14"},
            "take_profit": {"type": "risk_reward", "risk_reward_ratio": 2.0},
            "position_management": {"scale_in": {"enabled": False}, "partial_exit": {"enabled": False}},
            "sweep": {},
        }

        strategy_id = manager.create_strategy(
            name="test_strategy",
            dsl_config=dsl_dict,
            description="Test description",
        )

        assert strategy_id > 0

        strategies = manager.list_strategies()
        assert len(strategies) == 1
        assert strategies[0]["name"] == "test_strategy"

    def test_update_strategy(self, manager: StateManager) -> None:
        """Test updating strategy."""
        dsl_dict = {
            "name": "test_strategy",
            "description": "Test",
            "version": 1,
            "timeframe": "1h",
            "additional_timeframes": [],
            "indicators": {
                "rsi_14": {"type": "RSI", "period": 14, "source": "close"},
                "atr_14": {"type": "ATR", "period": 14, "source": "close"},
            },
            "entry_conditions": {"long": ["rsi_14 < 30"], "short": []},
            "exit_conditions": {"long": [], "short": []},
            "time_filters": {
                "allowed_sessions": [],
                "blocked_days": [],
                "avoid_around_funding": {"enabled": False, "minutes_before": 5, "minutes_after": 5},
            },
            "stop_loss": {"type": "atr_fixed", "atr_multiplier": 2.0, "indicator": "atr_14"},
            "take_profit": {"type": "risk_reward", "risk_reward_ratio": 2.0},
            "position_management": {"scale_in": {"enabled": False}, "partial_exit": {"enabled": False}},
            "sweep": {},
        }

        strategy_id = manager.create_strategy(
            name="test_strategy",
            dsl_config=dsl_dict,
        )

        # Update description
        manager.update_strategy(strategy_id, description="Updated description")

        strategy = manager.get_strategy(strategy_id)
        assert strategy is not None
        assert strategy["description"] == "Updated description"

    def test_deactivate_strategy(self, manager: StateManager) -> None:
        """Test deactivating (soft delete) strategy."""
        dsl_dict = {
            "name": "test_strategy",
            "description": "Test",
            "version": 1,
            "timeframe": "1h",
            "additional_timeframes": [],
            "indicators": {
                "rsi_14": {"type": "RSI", "period": 14, "source": "close"},
                "atr_14": {"type": "ATR", "period": 14, "source": "close"},
            },
            "entry_conditions": {"long": ["rsi_14 < 30"], "short": []},
            "exit_conditions": {"long": [], "short": []},
            "time_filters": {
                "allowed_sessions": [],
                "blocked_days": [],
                "avoid_around_funding": {"enabled": False, "minutes_before": 5, "minutes_after": 5},
            },
            "stop_loss": {"type": "atr_fixed", "atr_multiplier": 2.0, "indicator": "atr_14"},
            "take_profit": {"type": "risk_reward", "risk_reward_ratio": 2.0},
            "position_management": {"scale_in": {"enabled": False}, "partial_exit": {"enabled": False}},
            "sweep": {},
        }

        strategy_id = manager.create_strategy(
            name="test_strategy",
            dsl_config=dsl_dict,
        )

        # Deactivate
        manager.update_strategy(strategy_id, is_active=False)

        # Should not show in active-only list
        active_strategies = manager.list_strategies(active_only=True)
        assert len(active_strategies) == 0

        # Should show in all list
        all_strategies = manager.list_strategies(active_only=False)
        assert len(all_strategies) == 1
        assert not all_strategies[0]["is_active"]


class TestDashboardApp:
    """Tests for main dashboard app."""

    def test_app_module_import(self) -> None:
        """Test that app module can be imported."""
        from vibe_quant.dashboard import app

        assert hasattr(app, "main")

    def test_pages_init_import(self) -> None:
        """Test that pages package can be imported."""
        from vibe_quant.dashboard import pages

        assert pages is not None
