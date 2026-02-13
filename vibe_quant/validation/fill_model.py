"""Custom fill models and slippage estimation for validation backtesting.

Provides:
- VolumeSlippageFillModel: FillModel subclass that passes prob_slippage to NT
- SlippageEstimator: Standalone SPEC-formula slippage calculator for post-fill analytics
- ScreeningFillModelConfig / create_screening_fill_model: Simple fill model for screening
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from nautilus_trader.backtest.models import FillModel
from nautilus_trader.common.config import NautilusConfig


class VolumeSlippageFillModelConfig(NautilusConfig, frozen=True):
    """Configuration for VolumeSlippageFillModel.

    Must inherit from NautilusConfig so NT's resolve_config_path accepts it
    when used via ImportableFillModelConfig.

    Attributes:
        impact_coefficient: Market impact coefficient k in slippage formula.
            Higher values = more slippage. Default 0.1.
        prob_fill_on_limit: Probability of limit order fill when price touches.
            Default 0.8.
        prob_slippage: Probability that market orders experience slippage.
            Default 1.0 (always apply slippage in validation).
    """

    impact_coefficient: float = 0.1
    prob_fill_on_limit: float = 0.8
    prob_slippage: float = 1.0


class VolumeSlippageFillModel(FillModel):  # type: ignore[misc]
    """Fill model for validation backtesting with volume-based slippage estimation.

    NautilusTrader's matching engine uses FillModel.is_slipped() to decide
    whether a fill gets 1-tick slippage. The slippage *amount* is fixed at
    1 tick internally and cannot be overridden via subclassing.

    This class:
    1. Passes prob_slippage to the base FillModel (controlling slippage probability)
    2. Stores the impact_coefficient for use by SlippageEstimator

    For realistic slippage *cost* estimation per the SPEC formula, use
    SlippageEstimator separately in post-fill analytics.
    """

    def __init__(
        self,
        config: VolumeSlippageFillModelConfig | None = None,
        *,
        impact_coefficient: float = 0.1,
        prob_fill_on_limit: float = 0.8,
        prob_slippage: float = 1.0,
    ) -> None:
        """Initialize VolumeSlippageFillModel.

        Args:
            config: Configuration object. If provided, other args are ignored.
            impact_coefficient: Market impact coefficient k.
            prob_fill_on_limit: Probability of limit fill.
            prob_slippage: Probability of slippage.
        """
        if config is not None:
            impact_coefficient = config.impact_coefficient
            prob_fill_on_limit = config.prob_fill_on_limit
            prob_slippage = config.prob_slippage

        super().__init__(
            prob_fill_on_limit=prob_fill_on_limit,
            prob_slippage=prob_slippage,
        )

        self._impact_coefficient = impact_coefficient

    @property
    def impact_coefficient(self) -> float:
        """Get market impact coefficient."""
        return self._impact_coefficient


class SlippageEstimator:
    """Standalone slippage estimator using SPEC square-root market impact formula.

    Formula (SPEC Section 7):
        slippage = spread/2 + k * volatility * sqrt(order_size / avg_volume)

    This is used by ValidationRunner post-fill to compute realistic slippage
    costs for each trade. It is NOT integrated into NT's matching engine
    (which only supports 1-tick slippage).

    Example:
        estimator = SlippageEstimator(impact_coefficient=0.1)
        slippage = estimator.calculate(
            order_size=1.0, avg_volume=1000.0,
            volatility=0.02, spread=0.0001,
        )
    """

    def __init__(self, impact_coefficient: float = 0.1) -> None:
        """Initialize SlippageEstimator.

        Args:
            impact_coefficient: Market impact coefficient k.
        """
        self._k = impact_coefficient

    @property
    def impact_coefficient(self) -> float:
        """Get market impact coefficient."""
        return self._k

    def calculate(
        self,
        order_size: float,
        avg_volume: float,
        volatility: float = 0.0,
        spread: float = 0.0,
    ) -> float:
        """Calculate slippage factor using square-root market impact.

        Formula (from SPEC):
            slippage = spread/2 + k * volatility * sqrt(order_size / avg_volume)

        Args:
            order_size: Order quantity.
            avg_volume: Average bar volume.
            volatility: Current volatility estimate (e.g. realized vol or ATR
                as a fraction of price).
            spread: Current bid-ask spread as a fraction of price.

        Returns:
            Slippage factor as a decimal (e.g., 0.001 for 0.1% slippage).
        """
        half_spread = spread * 0.5

        # Fast path: no volume data or no volatility -> spread-only slippage
        if avg_volume <= 0 or volatility == 0.0:
            return half_spread

        # SPEC formula: spread/2 + k * volatility * sqrt(order_size / avg_volume)
        market_impact = (
            self._k * volatility * math.sqrt(abs(order_size) / avg_volume)
        )
        return half_spread + market_impact

    def estimate_cost(
        self,
        entry_price: float,
        order_size: float,
        avg_volume: float,
        volatility: float = 0.0,
        spread: float = 0.0,
    ) -> float:
        """Calculate estimated slippage cost in quote currency.

        Args:
            entry_price: Trade entry price.
            order_size: Order quantity.
            avg_volume: Average bar volume.
            volatility: Current volatility estimate.
            spread: Current bid-ask spread as fraction of price.

        Returns:
            Estimated slippage cost in quote currency.
        """
        factor = self.calculate(order_size, avg_volume, volatility, spread)
        return factor * entry_price * abs(order_size)


@dataclass(frozen=True)
class ScreeningFillModelConfig:
    """Configuration for simple screening fill model.

    Attributes:
        prob_fill_on_limit: Probability of limit order fill.
        prob_slippage: Probability of slippage on market orders.
    """

    prob_fill_on_limit: float = 0.8
    prob_slippage: float = 0.5


def create_screening_fill_model(
    config: ScreeningFillModelConfig | None = None,
) -> FillModel:
    """Create simple FillModel for screening mode.

    Screening uses probabilistic fills without volume-based impact.
    This is faster but less realistic than validation mode.

    Args:
        config: Configuration for the model.

    Returns:
        Simple FillModel suitable for screening.
    """
    if config is None:
        config = ScreeningFillModelConfig()

    return FillModel(
        prob_fill_on_limit=config.prob_fill_on_limit,
        prob_slippage=config.prob_slippage,
    )


def create_validation_fill_model(
    config: VolumeSlippageFillModelConfig | None = None,
) -> VolumeSlippageFillModel:
    """Create volume-based FillModel for validation mode.

    Validation uses realistic fill probability. For slippage cost estimation,
    use SlippageEstimator separately.

    Args:
        config: Configuration for the model.

    Returns:
        VolumeSlippageFillModel for validation execution simulation.
    """
    if config is None:
        config = VolumeSlippageFillModelConfig()

    return VolumeSlippageFillModel(config=config)
