"""Tests for validation venue configuration."""

from decimal import Decimal

import pytest
from nautilus_trader.config import (
    BacktestVenueConfig,
    LatencyModelConfig,
)

from vibe_quant.validation import (
    BINANCE_MAKER_FEE,
    BINANCE_TAKER_FEE,
    LATENCY_PRESETS,
    LatencyPreset,
    LatencyValues,
    ScreeningFillModelConfig,
    VenueConfig,
    VolumeSlippageFillModel,
    VolumeSlippageFillModelConfig,
    create_backtest_venue_config,
    create_custom_latency_model,
    create_screening_fill_model,
    create_validation_fill_model,
    create_venue_config_for_screening,
    create_venue_config_for_validation,
    get_latency_model,
)


class TestLatencyPresets:
    """Tests for latency presets and configuration."""

    def test_latency_preset_enum_values(self) -> None:
        """All preset enum values are defined."""
        assert LatencyPreset.COLOCATED.value == "colocated"
        assert LatencyPreset.DOMESTIC.value == "domestic"
        assert LatencyPreset.INTERNATIONAL.value == "international"
        assert LatencyPreset.RETAIL.value == "retail"

    def test_all_presets_have_values(self) -> None:
        """All presets are in LATENCY_PRESETS dict."""
        for preset in LatencyPreset:
            assert preset in LATENCY_PRESETS
            values = LATENCY_PRESETS[preset]
            assert isinstance(values, LatencyValues)

    def test_get_latency_model_by_enum(self) -> None:
        """Get latency model using enum."""
        config = get_latency_model(LatencyPreset.COLOCATED)
        assert isinstance(config, LatencyModelConfig)
        # Co-located: 0.5ms base
        assert config.base_latency_nanos == 500_000

    def test_get_latency_model_by_string(self) -> None:
        """Get latency model using string."""
        config = get_latency_model("retail")
        assert isinstance(config, LatencyModelConfig)
        # Retail: 100ms base
        assert config.base_latency_nanos == 100_000_000

    def test_get_latency_model_invalid_preset(self) -> None:
        """Invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown latency preset"):
            get_latency_model("invalid")

    def test_latency_values_to_config(self) -> None:
        """LatencyValues correctly converts to LatencyModelConfig."""
        values = LatencyValues(
            base_ms=10.0,
            insert_ms=5.0,
            update_ms=5.0,
            cancel_ms=5.0,
        )
        config = values.to_config()

        assert config.base_latency_nanos == 10_000_000
        assert config.insert_latency_nanos == 5_000_000
        assert config.update_latency_nanos == 5_000_000
        assert config.cancel_latency_nanos == 5_000_000

    def test_create_custom_latency_model(self) -> None:
        """Create custom latency model with specified values."""
        config = create_custom_latency_model(
            base_ms=50.0,
            insert_ms=25.0,
            update_ms=25.0,
            cancel_ms=25.0,
        )
        assert config.base_latency_nanos == 50_000_000
        assert config.insert_latency_nanos == 25_000_000

    def test_create_custom_latency_model_defaults(self) -> None:
        """Custom latency model defaults operation latencies to half base."""
        config = create_custom_latency_model(base_ms=100.0)
        assert config.base_latency_nanos == 100_000_000
        # Defaults to half of base
        assert config.insert_latency_nanos == 50_000_000
        assert config.update_latency_nanos == 50_000_000
        assert config.cancel_latency_nanos == 50_000_000

    def test_preset_latency_ordering(self) -> None:
        """Presets are ordered by increasing latency."""
        colocated = LATENCY_PRESETS[LatencyPreset.COLOCATED]
        domestic = LATENCY_PRESETS[LatencyPreset.DOMESTIC]
        international = LATENCY_PRESETS[LatencyPreset.INTERNATIONAL]
        retail = LATENCY_PRESETS[LatencyPreset.RETAIL]

        assert colocated.base_ms < domestic.base_ms
        assert domestic.base_ms < international.base_ms
        assert international.base_ms < retail.base_ms


class TestFillModels:
    """Tests for fill model configuration."""

    def test_screening_fill_model_config_defaults(self) -> None:
        """ScreeningFillModelConfig has sensible defaults."""
        config = ScreeningFillModelConfig()
        assert config.prob_fill_on_limit == 0.8
        assert config.prob_slippage == 0.5

    def test_volume_slippage_fill_model_config_defaults(self) -> None:
        """VolumeSlippageFillModelConfig has sensible defaults."""
        config = VolumeSlippageFillModelConfig()
        assert config.impact_coefficient == 0.1
        assert config.prob_fill_on_limit == 0.8
        assert config.prob_slippage == 1.0

    def test_create_screening_fill_model(self) -> None:
        """Create screening fill model."""
        model = create_screening_fill_model()
        assert model.prob_fill_on_limit == 0.8
        assert model.prob_slippage == 0.5

    def test_create_screening_fill_model_with_config(self) -> None:
        """Create screening fill model with custom config."""
        config = ScreeningFillModelConfig(
            prob_fill_on_limit=0.9,
            prob_slippage=0.3,
        )
        model = create_screening_fill_model(config)
        assert model.prob_fill_on_limit == 0.9
        assert model.prob_slippage == 0.3

    def test_create_validation_fill_model(self) -> None:
        """Create validation fill model."""
        model = create_validation_fill_model()
        assert isinstance(model, VolumeSlippageFillModel)
        assert model.impact_coefficient == 0.1

    def test_create_validation_fill_model_with_config(self) -> None:
        """Create validation fill model with custom config."""
        config = VolumeSlippageFillModelConfig(
            impact_coefficient=0.2,
            prob_fill_on_limit=0.7,
            prob_slippage=0.9,
        )
        model = create_validation_fill_model(config)
        assert model.impact_coefficient == 0.2
        assert model.prob_fill_on_limit == 0.7
        assert model.prob_slippage == 0.9

    def test_volume_slippage_calculate_factor(self) -> None:
        """VolumeSlippageFillModel calculates slippage factor correctly."""
        model = VolumeSlippageFillModel(impact_coefficient=0.1)

        # Small order: 1% of volume -> sqrt(0.01) * 0.1 = 0.01
        factor = model.calculate_slippage_factor(order_size=100, avg_volume=10000)
        assert abs(factor - 0.01) < 1e-9

        # Larger order: 25% of volume -> sqrt(0.25) * 0.1 = 0.05
        factor = model.calculate_slippage_factor(order_size=2500, avg_volume=10000)
        assert abs(factor - 0.05) < 1e-9

    def test_volume_slippage_zero_volume(self) -> None:
        """VolumeSlippageFillModel handles zero volume gracefully."""
        model = VolumeSlippageFillModel()
        factor = model.calculate_slippage_factor(order_size=100, avg_volume=0)
        assert factor == 0.0


class TestVenueConfig:
    """Tests for venue configuration."""

    def test_venue_config_defaults(self) -> None:
        """VenueConfig has sensible defaults."""
        config = VenueConfig()
        assert config.name == "BINANCE"
        assert config.starting_balance_usdt == 100_000
        assert config.default_leverage == Decimal("10")
        assert config.leverages == {}
        assert config.latency_preset is None
        assert not config.use_volume_slippage
        assert config.maker_fee == BINANCE_MAKER_FEE
        assert config.taker_fee == BINANCE_TAKER_FEE

    def test_binance_fee_constants(self) -> None:
        """Binance fee constants are correct."""
        assert BINANCE_MAKER_FEE == Decimal("0.0002")  # 0.02%
        assert BINANCE_TAKER_FEE == Decimal("0.0004")  # 0.04%

    def test_create_venue_config_for_screening(self) -> None:
        """Create screening venue config."""
        config = create_venue_config_for_screening()
        assert config.name == "BINANCE"
        assert config.latency_preset is None
        assert not config.use_volume_slippage
        assert isinstance(config.fill_config, ScreeningFillModelConfig)

    def test_create_venue_config_for_screening_custom(self) -> None:
        """Create screening venue config with custom values."""
        config = create_venue_config_for_screening(
            starting_balance_usdt=50_000,
            default_leverage=Decimal("5"),
            leverages={"BTCUSDT-PERP.BINANCE": Decimal("20")},
        )
        assert config.starting_balance_usdt == 50_000
        assert config.default_leverage == Decimal("5")
        assert "BTCUSDT-PERP.BINANCE" in config.leverages

    def test_create_venue_config_for_validation(self) -> None:
        """Create validation venue config."""
        config = create_venue_config_for_validation()
        assert config.name == "BINANCE"
        assert config.latency_preset == LatencyPreset.RETAIL
        assert config.use_volume_slippage
        assert isinstance(config.fill_config, VolumeSlippageFillModelConfig)

    def test_create_venue_config_for_validation_custom(self) -> None:
        """Create validation venue config with custom values."""
        config = create_venue_config_for_validation(
            starting_balance_usdt=200_000,
            latency_preset=LatencyPreset.COLOCATED,
            impact_coefficient=0.05,
        )
        assert config.starting_balance_usdt == 200_000
        assert config.latency_preset == LatencyPreset.COLOCATED
        assert config.fill_config is not None
        assert isinstance(config.fill_config, VolumeSlippageFillModelConfig)
        assert config.fill_config.impact_coefficient == 0.05


class TestBacktestVenueConfig:
    """Tests for NautilusTrader BacktestVenueConfig creation."""

    def test_create_backtest_venue_config_screening(self) -> None:
        """Create BacktestVenueConfig for screening."""
        config = create_venue_config_for_screening()
        backtest_config = create_backtest_venue_config(config)

        assert isinstance(backtest_config, BacktestVenueConfig)
        assert backtest_config.name == "BINANCE"
        assert backtest_config.oms_type == "NETTING"
        assert backtest_config.account_type == "MARGIN"
        assert backtest_config.starting_balances == ["100000 USDT"]
        assert backtest_config.default_leverage == 10.0
        assert backtest_config.latency_model is None  # No latency for screening
        assert backtest_config.fill_model is not None
        assert backtest_config.fee_model is not None

    def test_create_backtest_venue_config_validation(self) -> None:
        """Create BacktestVenueConfig for validation."""
        config = create_venue_config_for_validation(
            latency_preset=LatencyPreset.DOMESTIC,
        )
        backtest_config = create_backtest_venue_config(config)

        assert isinstance(backtest_config, BacktestVenueConfig)
        assert backtest_config.latency_model is not None  # Has latency
        assert backtest_config.fill_model is not None

    def test_create_backtest_venue_config_with_leverages(self) -> None:
        """Create BacktestVenueConfig with per-instrument leverages."""
        config = VenueConfig(
            leverages={
                "BTCUSDT-PERP.BINANCE": Decimal("20"),
                "ETHUSDT-PERP.BINANCE": Decimal("15"),
            },
        )
        backtest_config = create_backtest_venue_config(config)

        assert backtest_config.leverages is not None
        assert backtest_config.leverages["BTCUSDT-PERP.BINANCE"] == 20.0
        assert backtest_config.leverages["ETHUSDT-PERP.BINANCE"] == 15.0

    def test_create_backtest_venue_config_custom_latency(self) -> None:
        """Create BacktestVenueConfig with custom latency config."""
        custom_latency = create_custom_latency_model(base_ms=75.0)
        config = VenueConfig(
            latency_config=custom_latency,
        )
        backtest_config = create_backtest_venue_config(config)

        assert backtest_config.latency_model is not None
