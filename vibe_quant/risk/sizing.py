"""Position sizing modules for NautilusTrader strategies.

Provides pluggable position sizers that calculate trade size based on:
- Fixed Fractional: constant % risk per trade
- Kelly Criterion: optimal growth with configurable kelly_fraction
- ATR-based: volatility-adaptive sizing using ATR for stop distance
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nautilus_trader.model.instruments import CryptoPerpetual
    from nautilus_trader.model.objects import Quantity


@dataclass(frozen=True)
class SizerConfig:
    """Base configuration for all position sizers.

    Attributes:
        max_leverage: Maximum allowed leverage (e.g., 20 for 20x).
        max_position_pct: Maximum position size as fraction of equity (0.5 = 50%).
    """

    max_leverage: Decimal
    max_position_pct: Decimal

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_leverage <= Decimal(0):
            raise ValueError("max_leverage must be positive")
        if not (Decimal(0) < self.max_position_pct <= Decimal(1)):
            raise ValueError("max_position_pct must be in (0, 1]")


@dataclass(frozen=True)
class FixedFractionalConfig(SizerConfig):
    """Configuration for FixedFractionalSizer.

    Attributes:
        risk_per_trade: Fraction of equity to risk per trade (0.02 = 2%).
    """

    risk_per_trade: Decimal

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (Decimal(0) < self.risk_per_trade <= Decimal(1)):
            raise ValueError("risk_per_trade must be in (0, 1]")


@dataclass(frozen=True)
class KellyConfig(SizerConfig):
    """Configuration for KellySizer.

    Attributes:
        win_rate: Historical win rate (0.6 = 60% wins).
        avg_win: Average winning trade size (absolute value).
        avg_loss: Average losing trade size (absolute value).
        kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly, default).
    """

    win_rate: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    kelly_fraction: Decimal = Decimal("0.5")

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (Decimal(0) < self.win_rate < Decimal(1)):
            raise ValueError("win_rate must be in (0, 1)")
        if self.avg_win <= Decimal(0):
            raise ValueError("avg_win must be positive")
        if self.avg_loss <= Decimal(0):
            raise ValueError("avg_loss must be positive")
        if not (Decimal(0) < self.kelly_fraction <= Decimal(1)):
            raise ValueError("kelly_fraction must be in (0, 1]")


@dataclass(frozen=True)
class ATRConfig(SizerConfig):
    """Configuration for ATRSizer.

    Attributes:
        risk_per_trade: Fraction of equity to risk per trade (0.02 = 2%).
        atr_multiplier: Multiplier for ATR to calculate stop distance.
    """

    risk_per_trade: Decimal
    atr_multiplier: Decimal

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (Decimal(0) < self.risk_per_trade <= Decimal(1)):
            raise ValueError("risk_per_trade must be in (0, 1]")
        if self.atr_multiplier <= Decimal(0):
            raise ValueError("atr_multiplier must be positive")


class PositionSizer(ABC):
    """Abstract base class for position sizing algorithms.

    All sizers calculate position size in base currency units and enforce:
    - Maximum leverage limits
    - Maximum position size as percentage of equity
    """

    def __init__(self, config: SizerConfig) -> None:
        """Initialize the position sizer.

        Args:
            config: Sizer configuration with limits.
        """
        self._config = config

    @property
    def config(self) -> SizerConfig:
        """Return the sizer configuration."""
        return self._config

    @abstractmethod
    def calculate_size(
        self,
        account_equity: Decimal,
        instrument: "CryptoPerpetual",
        entry_price: Decimal,
        stop_price: Decimal | None = None,
        atr: Decimal | None = None,
    ) -> "Quantity":
        """Calculate position size for a trade.

        Args:
            account_equity: Current account equity in quote currency.
            instrument: The trading instrument (provides size precision, multipliers).
            entry_price: Expected entry price.
            stop_price: Stop-loss price (required for fixed fractional).
            atr: Current ATR value (required for ATR sizer).

        Returns:
            Position size as NautilusTrader Quantity.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        ...

    def _apply_limits(
        self,
        raw_size: Decimal,
        account_equity: Decimal,
        entry_price: Decimal,
        size_precision: int,
    ) -> Decimal:
        """Apply max leverage and max position limits to raw size.

        Args:
            raw_size: Calculated position size before limits.
            account_equity: Current account equity.
            entry_price: Entry price for notional calculation.
            size_precision: Decimal precision for rounding.

        Returns:
            Adjusted position size respecting all limits.
        """
        if raw_size <= Decimal(0):
            return Decimal(0)

        # Max position based on % of equity
        max_position_value = account_equity * self._config.max_position_pct
        max_size_by_pct = max_position_value / entry_price

        # Max position based on leverage
        max_notional = account_equity * self._config.max_leverage
        max_size_by_leverage = max_notional / entry_price

        # Take minimum of all limits
        final_size = min(raw_size, max_size_by_pct, max_size_by_leverage)

        # Round to instrument precision
        if final_size <= Decimal(0):
            return Decimal(0)

        # Round down to avoid exceeding limits
        quantize_str = "1." + "0" * size_precision if size_precision > 0 else "1"
        return final_size.quantize(Decimal(quantize_str), rounding=ROUND_DOWN)


class FixedFractionalSizer(PositionSizer):
    """Fixed fractional position sizing.

    Calculates position size to risk a fixed percentage of equity per trade.
    Formula: size = (equity * risk_per_trade) / |entry - stop|

    Requires stop_price for risk calculation.
    """

    def __init__(self, config: FixedFractionalConfig) -> None:
        """Initialize FixedFractionalSizer.

        Args:
            config: Configuration with risk_per_trade and limits.
        """
        super().__init__(config)
        self._risk_per_trade = config.risk_per_trade

    def calculate_size(
        self,
        account_equity: Decimal,
        instrument: "CryptoPerpetual",
        entry_price: Decimal,
        stop_price: Decimal | None = None,
        atr: Decimal | None = None,
    ) -> "Quantity":
        """Calculate position size based on fixed fractional risk.

        Args:
            account_equity: Current account equity in quote currency.
            instrument: The trading instrument.
            entry_price: Expected entry price.
            stop_price: Stop-loss price (REQUIRED).
            atr: Ignored for this sizer.

        Returns:
            Position size as NautilusTrader Quantity.

        Raises:
            ValueError: If stop_price not provided or prices invalid.
        """
        from nautilus_trader.model.objects import Quantity

        # Validate inputs
        if account_equity <= Decimal(0):
            return Quantity.zero(precision=instrument.size_precision)

        if entry_price <= Decimal(0):
            raise ValueError("entry_price must be positive")

        if stop_price is None:
            raise ValueError("stop_price required for FixedFractionalSizer")

        if stop_price <= Decimal(0):
            raise ValueError("stop_price must be positive")

        # Calculate stop distance
        stop_distance = abs(entry_price - stop_price)
        if stop_distance == Decimal(0):
            return Quantity.zero(precision=instrument.size_precision)

        # Risk amount in quote currency
        risk_amount = account_equity * self._risk_per_trade

        # Position value that would lose risk_amount at stop
        # If price moves stop_distance, we lose stop_distance per unit
        # So size = risk_amount / stop_distance
        raw_size = risk_amount / stop_distance

        # Apply limits
        final_size = self._apply_limits(
            raw_size, account_equity, entry_price, instrument.size_precision
        )

        # NautilusTrader Quantity requires float, not Decimal
        return Quantity(float(final_size), precision=instrument.size_precision)


class KellySizer(PositionSizer):
    """Kelly Criterion position sizing.

    Calculates position size for optimal geometric growth.
    Formula: f* = (win_rate - (1-win_rate)/(avg_win/avg_loss)) * kelly_fraction
             size = f* * equity / entry_price

    Uses historical win_rate, avg_win, avg_loss from config.
    """

    def __init__(self, config: KellyConfig) -> None:
        """Initialize KellySizer.

        Args:
            config: Configuration with Kelly parameters and limits.
        """
        super().__init__(config)
        self._win_rate = config.win_rate
        self._avg_win = config.avg_win
        self._avg_loss = config.avg_loss
        self._kelly_fraction = config.kelly_fraction

    @property
    def kelly_f(self) -> Decimal:
        """Calculate the Kelly fraction (f*).

        Formula: f* = W - (1-W)/R where W=win_rate, R=avg_win/avg_loss
        Then multiply by kelly_fraction for fractional Kelly.

        Returns:
            Kelly fraction, clipped to [0, 1].
        """
        w = self._win_rate
        r = self._avg_win / self._avg_loss  # Win/Loss ratio

        # Kelly formula: f* = W - (1-W)/R
        f_star = w - (Decimal(1) - w) / r

        # Apply kelly_fraction (e.g., half-Kelly)
        f_adjusted = f_star * self._kelly_fraction

        # Clip to valid range; warn if negative (unfavorable edge)
        if f_adjusted < 0:
            logger.warning(
                "Kelly f*=%.4f is negative (unfavorable edge), clipping to 0",
                float(f_adjusted),
            )
        return max(Decimal(0), min(f_adjusted, Decimal(1)))

    def calculate_size(
        self,
        account_equity: Decimal,
        instrument: "CryptoPerpetual",
        entry_price: Decimal,
        stop_price: Decimal | None = None,
        atr: Decimal | None = None,
    ) -> "Quantity":
        """Calculate position size based on Kelly Criterion.

        Args:
            account_equity: Current account equity in quote currency.
            instrument: The trading instrument.
            entry_price: Expected entry price.
            stop_price: Ignored for this sizer.
            atr: Ignored for this sizer.

        Returns:
            Position size as NautilusTrader Quantity.

        Raises:
            ValueError: If entry_price invalid.
        """
        from nautilus_trader.model.objects import Quantity

        # Validate inputs
        if account_equity <= Decimal(0):
            return Quantity.zero(precision=instrument.size_precision)

        if entry_price <= Decimal(0):
            raise ValueError("entry_price must be positive")

        # Calculate Kelly fraction
        f = self.kelly_f
        if f <= Decimal(0):
            return Quantity.zero(precision=instrument.size_precision)

        # Position value = f * equity
        position_value = f * account_equity

        # Convert to base currency units
        raw_size = position_value / entry_price

        # Apply limits
        final_size = self._apply_limits(
            raw_size, account_equity, entry_price, instrument.size_precision
        )

        # NautilusTrader Quantity requires float, not Decimal
        return Quantity(float(final_size), precision=instrument.size_precision)


class ATRSizer(PositionSizer):
    """ATR-based position sizing.

    Calculates position size using ATR as a volatility-adaptive stop distance.
    Formula: stop_distance = atr * atr_multiplier
             size = (equity * risk_per_trade) / stop_distance

    Requires ATR value for calculation.
    """

    def __init__(self, config: ATRConfig) -> None:
        """Initialize ATRSizer.

        Args:
            config: Configuration with ATR parameters and limits.
        """
        super().__init__(config)
        self._risk_per_trade = config.risk_per_trade
        self._atr_multiplier = config.atr_multiplier

    def calculate_size(
        self,
        account_equity: Decimal,
        instrument: "CryptoPerpetual",
        entry_price: Decimal,
        stop_price: Decimal | None = None,
        atr: Decimal | None = None,
    ) -> "Quantity":
        """Calculate position size based on ATR volatility.

        Args:
            account_equity: Current account equity in quote currency.
            instrument: The trading instrument.
            entry_price: Expected entry price.
            stop_price: Ignored for this sizer.
            atr: Current ATR value (REQUIRED).

        Returns:
            Position size as NautilusTrader Quantity.

        Raises:
            ValueError: If atr not provided or prices invalid.
        """
        from nautilus_trader.model.objects import Quantity

        # Validate inputs
        if account_equity <= Decimal(0):
            return Quantity.zero(precision=instrument.size_precision)

        if entry_price <= Decimal(0):
            raise ValueError("entry_price must be positive")

        if atr is None:
            raise ValueError("atr required for ATRSizer")

        if atr <= Decimal(0):
            return Quantity.zero(precision=instrument.size_precision)

        # Stop distance = ATR * multiplier
        stop_distance = atr * self._atr_multiplier

        if stop_distance <= Decimal(0):
            return Quantity.zero(precision=instrument.size_precision)

        # Risk amount in quote currency
        risk_amount = account_equity * self._risk_per_trade

        # Size = risk_amount / stop_distance
        raw_size = risk_amount / stop_distance

        # Apply limits
        final_size = self._apply_limits(
            raw_size, account_equity, entry_price, instrument.size_precision
        )

        # NautilusTrader Quantity requires float, not Decimal
        return Quantity(float(final_size), precision=instrument.size_precision)
