"""Tests for Ethereal venue configuration."""

from decimal import Decimal

import pytest
from nautilus_trader.config import LatencyModelConfig

from vibe_quant.ethereal import (
    ETHEREAL_FUNDING_INTERVAL,
    ETHEREAL_LATENCY_PRESETS,
    ETHEREAL_MAKER_FEE,
    ETHEREAL_TAKER_FEE,
    EtherealLatencyPreset,
    EtherealVenueConfig,
    get_ethereal_latency_model,
    get_ethereal_venue_for_backtesting,
    get_ethereal_venue_for_paper_trading,
)


class TestEtherealFeeStructure:
    """Tests for Ethereal fee structure."""

    def test_maker_fee_is_zero(self) -> None:
        """Maker fee is 0%."""
        assert Decimal("0.0000") == ETHEREAL_MAKER_FEE

    def test_taker_fee_is_003_percent(self) -> None:
        """Taker fee is 0.03%."""
        assert Decimal("0.0003") == ETHEREAL_TAKER_FEE

    def test_venue_config_has_correct_fees(self) -> None:
        """VenueConfig uses correct fee structure."""
        config = EtherealVenueConfig()
        assert config.maker_fee == Decimal("0.0000")
        assert config.taker_fee == Decimal("0.0003")


class TestEtherealFundingInterval:
    """Tests for Ethereal funding interval."""

    def test_funding_interval_is_one_hour(self) -> None:
        """Funding interval is 1 hour (3600 seconds)."""
        assert ETHEREAL_FUNDING_INTERVAL == 3600

    def test_venue_config_has_correct_funding_interval(self) -> None:
        """VenueConfig uses correct funding interval."""
        config = EtherealVenueConfig()
        assert config.funding_interval == 3600


class TestEtherealLatencyPresets:
    """Tests for Ethereal latency presets."""

    def test_latency_preset_enum_values(self) -> None:
        """All preset enum values are defined."""
        assert EtherealLatencyPreset.TESTNET.value == "testnet"
        assert EtherealLatencyPreset.MAINNET.value == "mainnet"

    def test_all_presets_have_values(self) -> None:
        """All presets are in ETHEREAL_LATENCY_PRESETS dict."""
        for preset in EtherealLatencyPreset:
            assert preset in ETHEREAL_LATENCY_PRESETS

    def test_testnet_latency_is_moderate(self) -> None:
        """Testnet has moderate latency (~500ms)."""
        values = ETHEREAL_LATENCY_PRESETS[EtherealLatencyPreset.TESTNET]
        assert values.base_ms == 500.0

    def test_mainnet_latency_accounts_for_block_time(self) -> None:
        """Mainnet latency accounts for ~2 second block time."""
        values = ETHEREAL_LATENCY_PRESETS[EtherealLatencyPreset.MAINNET]
        assert values.base_ms == 2000.0

    def test_mainnet_latency_higher_than_testnet(self) -> None:
        """Mainnet latency is higher than testnet."""
        testnet = ETHEREAL_LATENCY_PRESETS[EtherealLatencyPreset.TESTNET]
        mainnet = ETHEREAL_LATENCY_PRESETS[EtherealLatencyPreset.MAINNET]
        assert mainnet.base_ms > testnet.base_ms

    def test_get_latency_model_by_enum(self) -> None:
        """Get latency model using enum."""
        config = get_ethereal_latency_model(EtherealLatencyPreset.MAINNET)
        assert isinstance(config, LatencyModelConfig)
        # 2000ms base in nanoseconds
        assert config.base_latency_nanos == 2_000_000_000

    def test_get_latency_model_by_string(self) -> None:
        """Get latency model using string."""
        config = get_ethereal_latency_model("testnet")
        assert isinstance(config, LatencyModelConfig)
        # 500ms base in nanoseconds
        assert config.base_latency_nanos == 500_000_000

    def test_get_latency_model_invalid_preset(self) -> None:
        """Invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown Ethereal latency preset"):
            get_ethereal_latency_model("invalid")


class TestEtherealVenueConfig:
    """Tests for Ethereal venue configuration."""

    def test_venue_config_defaults(self) -> None:
        """VenueConfig has sensible defaults."""
        config = EtherealVenueConfig()
        assert config.name == "ETHEREAL"
        assert config.starting_balance_usdt == 100_000
        assert config.default_leverage == Decimal("10")
        assert config.leverages == {}
        assert config.funding_interval == 3600
        assert config.maker_fee == ETHEREAL_MAKER_FEE
        assert config.taker_fee == ETHEREAL_TAKER_FEE
        assert config.latency_preset is None

    def test_venue_config_custom_leverage(self) -> None:
        """VenueConfig accepts custom leverage settings."""
        config = EtherealVenueConfig(
            default_leverage=Decimal("5"),
            leverages={"BTCUSDT-PERP.ETHEREAL": Decimal("20")},
        )
        assert config.default_leverage == Decimal("5")
        assert config.leverages["BTCUSDT-PERP.ETHEREAL"] == Decimal("20")


class TestEtherealVenueForBacktesting:
    """Tests for backtesting venue configuration."""

    def test_backtesting_venue_uses_mainnet_latency(self) -> None:
        """Backtesting venue uses mainnet latency for realism."""
        config = get_ethereal_venue_for_backtesting()
        assert config.latency_preset == EtherealLatencyPreset.MAINNET

    def test_backtesting_venue_has_correct_name(self) -> None:
        """Backtesting venue has correct name."""
        config = get_ethereal_venue_for_backtesting()
        assert config.name == "ETHEREAL"

    def test_backtesting_venue_custom_params(self) -> None:
        """Backtesting venue accepts custom parameters."""
        config = get_ethereal_venue_for_backtesting(
            starting_balance_usdt=50_000,
            default_leverage=Decimal("5"),
            leverages={"ETHUSDT-PERP.ETHEREAL": Decimal("15")},
        )
        assert config.starting_balance_usdt == 50_000
        assert config.default_leverage == Decimal("5")
        assert "ETHUSDT-PERP.ETHEREAL" in config.leverages


class TestEtherealVenueForPaperTrading:
    """Tests for paper trading venue configuration."""

    def test_paper_trading_venue_uses_testnet_latency(self) -> None:
        """Paper trading venue uses testnet latency for faster iteration."""
        config = get_ethereal_venue_for_paper_trading()
        assert config.latency_preset == EtherealLatencyPreset.TESTNET

    def test_paper_trading_venue_has_correct_name(self) -> None:
        """Paper trading venue has correct name."""
        config = get_ethereal_venue_for_paper_trading()
        assert config.name == "ETHEREAL"

    def test_paper_trading_venue_custom_params(self) -> None:
        """Paper trading venue accepts custom parameters."""
        config = get_ethereal_venue_for_paper_trading(
            starting_balance_usdt=25_000,
            default_leverage=Decimal("3"),
        )
        assert config.starting_balance_usdt == 25_000
        assert config.default_leverage == Decimal("3")
