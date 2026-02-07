"""Custom fill models for validation backtesting."""

from __future__ import annotations

import math
from dataclasses import dataclass

from nautilus_trader.backtest.models import FillModel


@dataclass(frozen=True)
class VolumeSlippageFillModelConfig:
    """Configuration for VolumeSlippageFillModel.

    Attributes:
        impact_coefficient: Market impact coefficient k in slippage formula.
            Higher values = more slippage. Default 0.1.
        prob_fill_on_limit: Probability of limit order fill when price touches.
            Default 0.8.
        prob_slippage: Probability that market orders experience slippage.
            Default 1.0 (always apply slippage formula).
    """

    impact_coefficient: float = 0.1
    prob_fill_on_limit: float = 0.8
    prob_slippage: float = 1.0


class VolumeSlippageFillModel(FillModel):  # type: ignore[misc]
    """Fill model with volume-based slippage using square-root market impact.

    Slippage formula (SPEC):
        slippage = spread/2 + k * volatility * sqrt(order_size / avg_volume)

    This models the market impact of larger orders: as order size increases
    relative to average volume, slippage increases at a diminishing rate
    (square-root relationship).

    For small orders (< 1% of avg volume), slippage is minimal.
    For large orders (> 10% of avg volume), slippage becomes significant.

    Note on NautilusTrader integration:
        NautilusTrader's ``FillModel`` uses internal C-level fill logic.  The
        ``prob_slippage`` parameter controls the *probability* that slippage
        occurs on a market order, but the actual slippage *amount* is
        determined internally by NT and cannot be overridden by subclassing
        alone.  Our custom ``calculate_slippage_factor()`` method is therefore
        **not** called automatically by NT's matching engine.  It is exposed
        for use by the ``ValidationRunner`` to adjust prices post-fill or for
        reporting/analytics purposes.
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

    def calculate_slippage_factor(
        self,
        order_size: float,
        avg_volume: float,
        volatility: float = 0.0,
        spread: float = 0.0,
    ) -> float:
        """Calculate slippage factor using square-root market impact.

        Formula (from SPEC):
            slippage = spread/2 + k * volatility * sqrt(order_size / avg_volume)

        This method is **not** called by NautilusTrader's internal matching
        engine (see class docstring).  It is available for the
        ``ValidationRunner`` to adjust fill prices post-execution or for
        analytics/reporting.

        Optimized with early exits for common zero-volatility cases
        and combined multiplication to reduce operations.

        Args:
            order_size: Order quantity.
            avg_volume: Average bar volume.
            volatility: Current volatility estimate (e.g. realized vol or ATR
                as a fraction of price).  Defaults to 0.0.
            spread: Current bid-ask spread as a fraction of price.
                Defaults to 0.0.

        Returns:
            Slippage factor as a decimal (e.g., 0.001 for 0.1% slippage).
        """
        half_spread = spread * 0.5

        # Fast path: no volume data or no volatility → spread-only slippage
        if avg_volume <= 0 or volatility == 0.0:
            return half_spread

        # SPEC formula: spread/2 + k * volatility * sqrt(order_size / avg_volume)
        # Combine k * volatility first, then multiply by sqrt
        market_impact = (
            self._impact_coefficient * volatility * math.sqrt(order_size / avg_volume)
        )
        return half_spread + market_impact


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

    Validation uses realistic volume-based slippage modeling.

    Args:
        config: Configuration for the model.

    Returns:
        VolumeSlippageFillModel for realistic execution simulation.
    """
    if config is None:
        config = VolumeSlippageFillModelConfig()

    return VolumeSlippageFillModel(config=config)
