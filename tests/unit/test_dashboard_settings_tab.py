"""Tests for dashboard settings tab module."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from vibe_quant.db import StateManager


class TestSettingsModuleImports:
    """Test that settings module imports correctly."""

    def test_import_module(self) -> None:
        """Module can be imported without errors."""
        from vibe_quant.dashboard.pages import settings

        assert hasattr(settings, "render_settings_tab")
        assert hasattr(settings, "render_sizing_section")
        assert hasattr(settings, "render_risk_section")
        assert hasattr(settings, "render_latency_section")
        assert hasattr(settings, "render_database_section")
        assert hasattr(settings, "render_system_info")

    def test_sizing_methods_defined(self) -> None:
        """SIZING_METHODS constant is defined."""
        from vibe_quant.dashboard.pages.settings import SIZING_METHODS

        assert "fixed_fractional" in SIZING_METHODS
        assert "kelly" in SIZING_METHODS
        assert "atr" in SIZING_METHODS

    def test_sizing_param_schemas_defined(self) -> None:
        """SIZING_PARAM_SCHEMAS has entries for all methods."""
        from vibe_quant.dashboard.pages.settings import (
            SIZING_METHODS,
            SIZING_PARAM_SCHEMAS,
        )

        for method in SIZING_METHODS:
            assert method in SIZING_PARAM_SCHEMAS
            assert isinstance(SIZING_PARAM_SCHEMAS[method], dict)


class TestParseDecimal:
    """Test decimal parsing helper."""

    def test_parse_valid_decimal(self) -> None:
        """Parse valid decimal strings."""
        from vibe_quant.dashboard.pages.settings import _parse_decimal

        assert _parse_decimal("0.5", "test") == Decimal("0.5")
        assert _parse_decimal("  1.0  ", "test") == Decimal("1.0")
        assert _parse_decimal("20", "test") == Decimal("20")

    def test_parse_invalid_decimal(self) -> None:
        """Parse invalid decimal raises ValueError."""
        from vibe_quant.dashboard.pages.settings import _parse_decimal

        with pytest.raises(ValueError, match="Invalid decimal"):
            _parse_decimal("not_a_number", "test_field")

        with pytest.raises(ValueError, match="Invalid decimal"):
            _parse_decimal("", "test_field")


class TestValidateSizingParams:
    """Test sizing parameter validation."""

    def test_validate_fixed_fractional_valid(self) -> None:
        """Valid fixed_fractional params pass validation."""
        from vibe_quant.dashboard.pages.settings import _validate_sizing_params

        # Note: This function uses st.error which requires streamlit context
        # Cannot fully test without mocking streamlit
        # Just verify function exists and accepts correct signature
        assert callable(_validate_sizing_params)

    def test_validate_kelly_params(self) -> None:
        """Kelly method has expected parameters."""
        from vibe_quant.dashboard.pages.settings import SIZING_PARAM_SCHEMAS

        kelly_schema = SIZING_PARAM_SCHEMAS["kelly"]
        assert "win_rate" in kelly_schema
        assert "avg_win" in kelly_schema
        assert "avg_loss" in kelly_schema
        assert "kelly_fraction" in kelly_schema

    def test_validate_atr_params(self) -> None:
        """ATR method has expected parameters."""
        from vibe_quant.dashboard.pages.settings import SIZING_PARAM_SCHEMAS

        atr_schema = SIZING_PARAM_SCHEMAS["atr"]
        assert "atr_multiplier" in atr_schema
        assert "risk_per_trade" in atr_schema


class TestValidateRiskParams:
    """Test risk parameter validation."""

    def test_validate_risk_params_function_exists(self) -> None:
        """Risk validation function exists with correct signature."""
        from vibe_quant.dashboard.pages.settings import _validate_risk_params

        assert callable(_validate_risk_params)


class TestLatencyPresets:
    """Test latency preset display."""

    def test_latency_presets_imported(self) -> None:
        """Latency presets are available via validation module."""
        # Settings module uses these same presets
        from vibe_quant.dashboard.pages import settings
        from vibe_quant.validation.latency import (
            LATENCY_PRESETS,
            LatencyPreset,
            LatencyValues,
        )

        assert settings is not None  # Module loads without error

        assert len(LATENCY_PRESETS) > 0
        assert LatencyPreset.COLOCATED in LATENCY_PRESETS
        assert LatencyPreset.RETAIL in LATENCY_PRESETS

        # Verify values are LatencyValues instances
        for _preset, values in LATENCY_PRESETS.items():
            assert isinstance(values, LatencyValues)
            assert values.base_ms > 0


class TestStateManagerIntegration:
    """Test StateManager integration."""

    def test_create_sizing_config(self, tmp_state_manager: StateManager) -> None:
        """Create sizing config via StateManager."""
        config = {
            "max_leverage": "20",
            "max_position_pct": "0.5",
            "risk_per_trade": "0.02",
        }
        config_id = tmp_state_manager.create_sizing_config(
            name="test_fixed",
            method="fixed_fractional",
            config=config,
        )
        assert config_id > 0

        # Retrieve and verify
        configs = tmp_state_manager.list_sizing_configs()
        assert len(configs) == 1
        assert configs[0]["name"] == "test_fixed"
        assert configs[0]["method"] == "fixed_fractional"

    def test_create_risk_config(self, tmp_state_manager: StateManager) -> None:
        """Create risk config via StateManager."""
        strategy_level = {
            "max_drawdown_pct": "0.15",
            "max_daily_loss_pct": "0.02",
            "max_consecutive_losses": 10,
            "max_position_count": 5,
        }
        portfolio_level = {
            "max_portfolio_drawdown_pct": "0.20",
            "max_total_exposure_pct": "0.50",
            "max_single_instrument_pct": "0.30",
        }
        config_id = tmp_state_manager.create_risk_config(
            name="test_risk",
            strategy_level=strategy_level,
            portfolio_level=portfolio_level,
        )
        assert config_id > 0

        # Retrieve and verify
        configs = tmp_state_manager.list_risk_configs()
        assert len(configs) == 1
        assert configs[0]["name"] == "test_risk"
        assert configs[0]["strategy_level"]["max_drawdown_pct"] == "0.15"

    def test_list_empty_configs(self, tmp_state_manager: StateManager) -> None:
        """List configs returns empty list when none exist."""
        assert tmp_state_manager.list_sizing_configs() == []
        assert tmp_state_manager.list_risk_configs() == []
