"""Unit tests for position sizing modules."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from vibe_quant.risk.sizing import (
    ATRConfig,
    ATRSizer,
    FixedFractionalConfig,
    FixedFractionalSizer,
    KellyConfig,
    KellySizer,
    SizerConfig,
)

# --- Mock NautilusTrader types ---


class MockQuantity:
    """Mock NautilusTrader Quantity for testing without NT dependency."""

    def __init__(self, value: float, precision: int) -> None:
        self.value = value
        self.precision = precision

    @classmethod
    def zero(cls, precision: int) -> "MockQuantity":
        return cls(0.0, precision)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockQuantity):
            return NotImplemented
        return self.value == other.value and self.precision == other.precision

    def __repr__(self) -> str:
        return f"MockQuantity({self.value}, precision={self.precision})"


def make_mock_instrument(size_precision: int = 3) -> MagicMock:
    """Create a mock CryptoPerpetual instrument."""
    instrument = MagicMock()
    instrument.size_precision = size_precision
    return instrument


@pytest.fixture(autouse=True)
def mock_nautilus_quantity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock NautilusTrader Quantity import."""
    mock_module = MagicMock()
    mock_module.Quantity = MockQuantity
    monkeypatch.setattr(
        "vibe_quant.risk.sizing.Quantity",
        MockQuantity,
        raising=False,
    )
    # Patch the import inside calculate_size
    import sys

    mock_nt = MagicMock()
    mock_nt.model = MagicMock()
    mock_nt.model.objects = MagicMock()
    mock_nt.model.objects.Quantity = MockQuantity
    sys.modules["nautilus_trader"] = mock_nt
    sys.modules["nautilus_trader.model"] = mock_nt.model
    sys.modules["nautilus_trader.model.objects"] = mock_nt.model.objects


# --- SizerConfig Tests ---


class TestSizerConfig:
    """Tests for base SizerConfig."""

    def test_valid_config(self) -> None:
        """Valid config creates successfully."""
        config = SizerConfig(max_leverage=Decimal("20"), max_position_pct=Decimal("0.5"))
        assert config.max_leverage == Decimal("20")
        assert config.max_position_pct == Decimal("0.5")

    def test_zero_leverage_raises(self) -> None:
        """Zero leverage raises ValueError."""
        with pytest.raises(ValueError, match="max_leverage must be positive"):
            SizerConfig(max_leverage=Decimal("0"), max_position_pct=Decimal("0.5"))

    def test_negative_leverage_raises(self) -> None:
        """Negative leverage raises ValueError."""
        with pytest.raises(ValueError, match="max_leverage must be positive"):
            SizerConfig(max_leverage=Decimal("-5"), max_position_pct=Decimal("0.5"))

    def test_zero_position_pct_raises(self) -> None:
        """Zero max_position_pct raises ValueError."""
        with pytest.raises(ValueError, match="max_position_pct must be in"):
            SizerConfig(max_leverage=Decimal("20"), max_position_pct=Decimal("0"))

    def test_over_one_position_pct_raises(self) -> None:
        """max_position_pct > 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_position_pct must be in"):
            SizerConfig(max_leverage=Decimal("20"), max_position_pct=Decimal("1.5"))


# --- FixedFractionalConfig Tests ---


class TestFixedFractionalConfig:
    """Tests for FixedFractionalConfig."""

    def test_valid_config(self) -> None:
        """Valid config creates successfully."""
        config = FixedFractionalConfig(
            max_leverage=Decimal("20"),
            max_position_pct=Decimal("0.5"),
            risk_per_trade=Decimal("0.02"),
        )
        assert config.risk_per_trade == Decimal("0.02")

    def test_zero_risk_raises(self) -> None:
        """Zero risk_per_trade raises ValueError."""
        with pytest.raises(ValueError, match="risk_per_trade must be in"):
            FixedFractionalConfig(
                max_leverage=Decimal("20"),
                max_position_pct=Decimal("0.5"),
                risk_per_trade=Decimal("0"),
            )

    def test_negative_risk_raises(self) -> None:
        """Negative risk_per_trade raises ValueError."""
        with pytest.raises(ValueError, match="risk_per_trade must be in"):
            FixedFractionalConfig(
                max_leverage=Decimal("20"),
                max_position_pct=Decimal("0.5"),
                risk_per_trade=Decimal("-0.02"),
            )


# --- KellyConfig Tests ---


class TestKellyConfig:
    """Tests for KellyConfig."""

    def test_valid_config(self) -> None:
        """Valid config creates successfully."""
        config = KellyConfig(
            max_leverage=Decimal("20"),
            max_position_pct=Decimal("0.5"),
            win_rate=Decimal("0.55"),
            avg_win=Decimal("1.5"),
            avg_loss=Decimal("1.0"),
        )
        assert config.kelly_fraction == Decimal("0.5")

    def test_zero_win_rate_raises(self) -> None:
        """Zero win_rate raises ValueError."""
        with pytest.raises(ValueError, match="win_rate must be in"):
            KellyConfig(
                max_leverage=Decimal("20"),
                max_position_pct=Decimal("0.5"),
                win_rate=Decimal("0"),
                avg_win=Decimal("1.5"),
                avg_loss=Decimal("1.0"),
            )

    def test_one_win_rate_raises(self) -> None:
        """win_rate of 1 raises ValueError."""
        with pytest.raises(ValueError, match="win_rate must be in"):
            KellyConfig(
                max_leverage=Decimal("20"),
                max_position_pct=Decimal("0.5"),
                win_rate=Decimal("1"),
                avg_win=Decimal("1.5"),
                avg_loss=Decimal("1.0"),
            )

    def test_zero_avg_win_raises(self) -> None:
        """Zero avg_win raises ValueError."""
        with pytest.raises(ValueError, match="avg_win must be positive"):
            KellyConfig(
                max_leverage=Decimal("20"),
                max_position_pct=Decimal("0.5"),
                win_rate=Decimal("0.55"),
                avg_win=Decimal("0"),
                avg_loss=Decimal("1.0"),
            )

    def test_zero_avg_loss_raises(self) -> None:
        """Zero avg_loss raises ValueError."""
        with pytest.raises(ValueError, match="avg_loss must be positive"):
            KellyConfig(
                max_leverage=Decimal("20"),
                max_position_pct=Decimal("0.5"),
                win_rate=Decimal("0.55"),
                avg_win=Decimal("1.5"),
                avg_loss=Decimal("0"),
            )


# --- ATRConfig Tests ---


class TestATRConfig:
    """Tests for ATRConfig."""

    def test_valid_config(self) -> None:
        """Valid config creates successfully."""
        config = ATRConfig(
            max_leverage=Decimal("20"),
            max_position_pct=Decimal("0.5"),
            risk_per_trade=Decimal("0.02"),
            atr_multiplier=Decimal("2.0"),
        )
        assert config.atr_multiplier == Decimal("2.0")

    def test_zero_atr_multiplier_raises(self) -> None:
        """Zero atr_multiplier raises ValueError."""
        with pytest.raises(ValueError, match="atr_multiplier must be positive"):
            ATRConfig(
                max_leverage=Decimal("20"),
                max_position_pct=Decimal("0.5"),
                risk_per_trade=Decimal("0.02"),
                atr_multiplier=Decimal("0"),
            )


# --- FixedFractionalSizer Tests ---


class TestFixedFractionalSizer:
    """Tests for FixedFractionalSizer."""

    @pytest.fixture
    def sizer(self) -> FixedFractionalSizer:
        """Create a standard FixedFractionalSizer for tests."""
        config = FixedFractionalConfig(
            max_leverage=Decimal("20"),
            max_position_pct=Decimal("0.5"),
            risk_per_trade=Decimal("0.02"),
        )
        return FixedFractionalSizer(config)

    def test_basic_calculation(self, sizer: FixedFractionalSizer) -> None:
        """Test basic position size calculation."""
        instrument = make_mock_instrument(size_precision=3)
        equity = Decimal("10000")
        entry = Decimal("50000")
        stop = Decimal("49000")  # $1000 stop distance

        # Risk = 10000 * 0.02 = $200
        # Size = 200 / 1000 = 0.2 BTC
        # But max_position_pct = 0.5, max_position = 10000 * 0.5 / 50000 = 0.1
        # So capped at 0.1
        result = sizer.calculate_size(equity, instrument, entry, stop_price=stop)
        assert result.value == 0.1

    def test_stop_above_entry_short(self, sizer: FixedFractionalSizer) -> None:
        """Test calculation with stop above entry (short position)."""
        instrument = make_mock_instrument(size_precision=3)
        equity = Decimal("10000")
        entry = Decimal("50000")
        stop = Decimal("51000")  # Stop above for short

        # Same limit applies
        result = sizer.calculate_size(equity, instrument, entry, stop_price=stop)
        assert result.value == 0.1

    def test_no_stop_raises(self, sizer: FixedFractionalSizer) -> None:
        """Test that missing stop_price raises ValueError."""
        instrument = make_mock_instrument()
        with pytest.raises(ValueError, match="stop_price required"):
            sizer.calculate_size(Decimal("10000"), instrument, Decimal("50000"))

    def test_zero_stop_raises(self, sizer: FixedFractionalSizer) -> None:
        """Test that zero stop_price raises ValueError."""
        instrument = make_mock_instrument()
        with pytest.raises(ValueError, match="stop_price must be positive"):
            sizer.calculate_size(
                Decimal("10000"), instrument, Decimal("50000"), stop_price=Decimal("0")
            )

    def test_zero_entry_raises(self, sizer: FixedFractionalSizer) -> None:
        """Test that zero entry_price raises ValueError."""
        instrument = make_mock_instrument()
        with pytest.raises(ValueError, match="entry_price must be positive"):
            sizer.calculate_size(
                Decimal("10000"), instrument, Decimal("0"), stop_price=Decimal("49000")
            )

    def test_zero_equity_returns_zero(self, sizer: FixedFractionalSizer) -> None:
        """Test that zero equity returns zero quantity."""
        instrument = make_mock_instrument()
        result = sizer.calculate_size(
            Decimal("0"), instrument, Decimal("50000"), stop_price=Decimal("49000")
        )
        assert result.value == 0.0

    def test_same_entry_stop_returns_zero(self, sizer: FixedFractionalSizer) -> None:
        """Test that entry == stop returns zero (infinite leverage)."""
        instrument = make_mock_instrument()
        result = sizer.calculate_size(
            Decimal("10000"), instrument, Decimal("50000"), stop_price=Decimal("50000")
        )
        assert result.value == 0.0

    def test_leverage_limit_applied(self) -> None:
        """Test that max leverage limits position size."""
        config = FixedFractionalConfig(
            max_leverage=Decimal("5"),  # Low leverage limit
            max_position_pct=Decimal("1"),
            risk_per_trade=Decimal("0.5"),  # Very high risk
        )
        sizer = FixedFractionalSizer(config)
        instrument = make_mock_instrument(size_precision=3)

        equity = Decimal("10000")
        entry = Decimal("50000")
        stop = Decimal("49500")  # Tight stop = large position

        result = sizer.calculate_size(equity, instrument, entry, stop_price=stop)

        # Max notional = 10000 * 5 = 50000
        # Max size = 50000 / 50000 = 1.0 BTC
        assert result.value <= 1.0

    def test_position_pct_limit_applied(self) -> None:
        """Test that max position percentage limits size."""
        config = FixedFractionalConfig(
            max_leverage=Decimal("100"),  # High leverage
            max_position_pct=Decimal("0.1"),  # 10% position limit
            risk_per_trade=Decimal("0.5"),  # High risk
        )
        sizer = FixedFractionalSizer(config)
        instrument = make_mock_instrument(size_precision=3)

        equity = Decimal("10000")
        entry = Decimal("50000")
        stop = Decimal("49500")

        result = sizer.calculate_size(equity, instrument, entry, stop_price=stop)

        # Max position = 10000 * 0.1 / 50000 = 0.02 BTC
        assert result.value == 0.02


# --- KellySizer Tests ---


class TestKellySizer:
    """Tests for KellySizer."""

    @pytest.fixture
    def sizer(self) -> KellySizer:
        """Create a standard KellySizer for tests."""
        config = KellyConfig(
            max_leverage=Decimal("20"),
            max_position_pct=Decimal("0.5"),
            win_rate=Decimal("0.55"),
            avg_win=Decimal("1.5"),
            avg_loss=Decimal("1.0"),
        )
        return KellySizer(config)

    def test_kelly_f_calculation(self, sizer: KellySizer) -> None:
        """Test Kelly fraction calculation."""
        # f* = W - (1-W)/R = 0.55 - 0.45/1.5 = 0.55 - 0.3 = 0.25
        # Half-Kelly = 0.25 * 0.5 = 0.125
        assert sizer.kelly_f == Decimal("0.125")

    def test_kelly_f_negative_edge_returns_zero(self) -> None:
        """Test that negative edge returns zero Kelly fraction."""
        config = KellyConfig(
            max_leverage=Decimal("20"),
            max_position_pct=Decimal("0.5"),
            win_rate=Decimal("0.3"),  # Low win rate
            avg_win=Decimal("1.0"),
            avg_loss=Decimal("1.0"),  # 1:1 R
            kelly_fraction=Decimal("1.0"),
        )
        sizer = KellySizer(config)
        # f* = 0.3 - 0.7/1 = -0.4, clipped to 0
        assert sizer.kelly_f == Decimal("0")

    def test_basic_calculation(self, sizer: KellySizer) -> None:
        """Test basic Kelly position size calculation."""
        instrument = make_mock_instrument(size_precision=3)
        equity = Decimal("10000")
        entry = Decimal("50000")

        # Kelly f = 0.125
        # Position value = 10000 * 0.125 = 1250
        # Size = 1250 / 50000 = 0.025
        result = sizer.calculate_size(equity, instrument, entry)
        assert result.value == 0.025

    def test_zero_entry_raises(self, sizer: KellySizer) -> None:
        """Test that zero entry_price raises ValueError."""
        instrument = make_mock_instrument()
        with pytest.raises(ValueError, match="entry_price must be positive"):
            sizer.calculate_size(Decimal("10000"), instrument, Decimal("0"))

    def test_zero_equity_returns_zero(self, sizer: KellySizer) -> None:
        """Test that zero equity returns zero quantity."""
        instrument = make_mock_instrument()
        result = sizer.calculate_size(Decimal("0"), instrument, Decimal("50000"))
        assert result.value == 0.0

    def test_negative_edge_returns_zero(self) -> None:
        """Test that sizer with negative edge returns zero size."""
        config = KellyConfig(
            max_leverage=Decimal("20"),
            max_position_pct=Decimal("0.5"),
            win_rate=Decimal("0.3"),  # Losing system
            avg_win=Decimal("1.0"),
            avg_loss=Decimal("1.0"),
        )
        sizer = KellySizer(config)
        instrument = make_mock_instrument()

        result = sizer.calculate_size(Decimal("10000"), instrument, Decimal("50000"))
        assert result.value == 0.0

    def test_full_kelly_vs_half_kelly(self) -> None:
        """Test that full Kelly is 2x half Kelly."""
        half_config = KellyConfig(
            max_leverage=Decimal("100"),
            max_position_pct=Decimal("1"),
            win_rate=Decimal("0.6"),
            avg_win=Decimal("2.0"),
            avg_loss=Decimal("1.0"),
            kelly_fraction=Decimal("0.5"),
        )
        full_config = KellyConfig(
            max_leverage=Decimal("100"),
            max_position_pct=Decimal("1"),
            win_rate=Decimal("0.6"),
            avg_win=Decimal("2.0"),
            avg_loss=Decimal("1.0"),
            kelly_fraction=Decimal("1.0"),
        )
        half_sizer = KellySizer(half_config)
        full_sizer = KellySizer(full_config)

        assert full_sizer.kelly_f == half_sizer.kelly_f * 2


# --- ATRSizer Tests ---


class TestATRSizer:
    """Tests for ATRSizer."""

    @pytest.fixture
    def sizer(self) -> ATRSizer:
        """Create a standard ATRSizer for tests."""
        config = ATRConfig(
            max_leverage=Decimal("20"),
            max_position_pct=Decimal("0.5"),
            risk_per_trade=Decimal("0.02"),
            atr_multiplier=Decimal("2.0"),
        )
        return ATRSizer(config)

    def test_basic_calculation(self, sizer: ATRSizer) -> None:
        """Test basic ATR position size calculation."""
        instrument = make_mock_instrument(size_precision=3)
        equity = Decimal("10000")
        entry = Decimal("50000")
        atr = Decimal("1000")

        # Stop distance = 1000 * 2 = 2000
        # Risk = 10000 * 0.02 = 200
        # Size = 200 / 2000 = 0.1
        result = sizer.calculate_size(equity, instrument, entry, atr=atr)
        assert result.value == 0.1

    def test_no_atr_raises(self, sizer: ATRSizer) -> None:
        """Test that missing atr raises ValueError."""
        instrument = make_mock_instrument()
        with pytest.raises(ValueError, match="atr required"):
            sizer.calculate_size(Decimal("10000"), instrument, Decimal("50000"))

    def test_zero_atr_returns_zero(self, sizer: ATRSizer) -> None:
        """Test that zero ATR returns zero quantity."""
        instrument = make_mock_instrument()
        result = sizer.calculate_size(
            Decimal("10000"), instrument, Decimal("50000"), atr=Decimal("0")
        )
        assert result.value == 0.0

    def test_zero_entry_raises(self, sizer: ATRSizer) -> None:
        """Test that zero entry_price raises ValueError."""
        instrument = make_mock_instrument()
        with pytest.raises(ValueError, match="entry_price must be positive"):
            sizer.calculate_size(Decimal("10000"), instrument, Decimal("0"), atr=Decimal("1000"))

    def test_zero_equity_returns_zero(self, sizer: ATRSizer) -> None:
        """Test that zero equity returns zero quantity."""
        instrument = make_mock_instrument()
        result = sizer.calculate_size(
            Decimal("0"), instrument, Decimal("50000"), atr=Decimal("1000")
        )
        assert result.value == 0.0

    def test_high_atr_reduces_size(self) -> None:
        """Test that higher ATR results in smaller position."""
        config = ATRConfig(
            max_leverage=Decimal("20"),
            max_position_pct=Decimal("0.5"),
            risk_per_trade=Decimal("0.02"),
            atr_multiplier=Decimal("2.0"),
        )
        sizer = ATRSizer(config)
        instrument = make_mock_instrument(size_precision=4)

        equity = Decimal("10000")
        entry = Decimal("50000")

        low_atr = sizer.calculate_size(equity, instrument, entry, atr=Decimal("500"))
        high_atr = sizer.calculate_size(equity, instrument, entry, atr=Decimal("2000"))

        assert high_atr.value < low_atr.value

    def test_leverage_limit_applied(self) -> None:
        """Test that max leverage limits position size."""
        config = ATRConfig(
            max_leverage=Decimal("2"),  # Very low leverage
            max_position_pct=Decimal("1"),
            risk_per_trade=Decimal("0.5"),  # High risk
            atr_multiplier=Decimal("0.1"),  # Tight stop
        )
        sizer = ATRSizer(config)
        instrument = make_mock_instrument(size_precision=3)

        equity = Decimal("10000")
        entry = Decimal("50000")
        atr = Decimal("100")

        result = sizer.calculate_size(equity, instrument, entry, atr=atr)

        # Raw: 5000 / 10 = 500 BTC (absurd)
        # Max notional = 10000 * 2 = 20000
        # Max size = 20000 / 50000 = 0.4 BTC
        assert result.value <= 0.4


# --- Precision Tests ---


class TestPrecision:
    """Test size precision handling."""

    def test_precision_rounding(self) -> None:
        """Test that sizes are rounded to instrument precision."""
        config = FixedFractionalConfig(
            max_leverage=Decimal("100"),
            max_position_pct=Decimal("1"),
            risk_per_trade=Decimal("0.02"),
        )
        sizer = FixedFractionalSizer(config)

        # Use precision 2
        instrument = make_mock_instrument(size_precision=2)

        equity = Decimal("10000")
        entry = Decimal("50000")
        stop = Decimal("49000")

        result = sizer.calculate_size(equity, instrument, entry, stop_price=stop)
        assert result.precision == 2

    def test_zero_precision(self) -> None:
        """Test integer-only position sizes."""
        config = KellyConfig(
            max_leverage=Decimal("100"),
            max_position_pct=Decimal("1"),
            win_rate=Decimal("0.6"),
            avg_win=Decimal("2.0"),
            avg_loss=Decimal("1.0"),
        )
        sizer = KellySizer(config)
        instrument = make_mock_instrument(size_precision=0)

        equity = Decimal("1000000")  # Large equity
        entry = Decimal("100")  # Low price

        result = sizer.calculate_size(equity, instrument, entry)
        assert result.precision == 0
        assert float(result.value).is_integer()
