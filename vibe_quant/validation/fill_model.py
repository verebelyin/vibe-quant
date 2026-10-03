"""Custom fill models and slippage estimation for validation backtesting.

Provides:
- VolumeSlippageFillModel: FillModel subclass that passes prob_slippage to NT
- SlippageEstimator: Standalone SPEC-formula slippage calculator for post-fill analytics
- ScreeningFillModelConfig / create_screening_fill_model: Simple fill model for screening
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from nautilus_trader.backtest.models import FillModel
from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.model.enums import OrderType

# Orders that become marketable when triggered. NT's own matching logic fills
# them at the trigger price when the bar moves through it, or at the open on a
# gap -- the fill model must NOT replace that with a synthetic book.
_TRIGGERED_MARKET_TYPES = frozenset(
    {OrderType.STOP_MARKET, OrderType.TRAILING_STOP_MARKET, OrderType.MARKET_IF_TOUCHED}
)


class VolumeSlippageFillModelConfig(NautilusConfig, frozen=True):
    """Configuration for VolumeSlippageFillModel.

    Must inherit from NautilusConfig so NT's resolve_config_path accepts it
    when used via ImportableFillModelConfig.

    Attributes:
        impact_coefficient: Market impact coefficient k in slippage formula.
            Higher values = more slippage. Default 0.1.
        prob_fill_on_limit: Probability of limit order fill when price touches.
            Default 0.8.
        prob_best_price_fill: Probability aggressive orders fill at the current
            bar's best price. Lower values degrade to an adverse synthetic book.
        max_adverse_ticks: Maximum ticks of adverse movement when the fill is
            degraded away from the best price.
        prob_slippage: Probability that market orders experience slippage.
            Default 0.0 in validation to avoid double-counting with
            post-fill SPEC slippage estimation.
        random_seed: Seed for ALL probabilistic draws (engine slippage,
            synthetic-book degradation). Unseeded draws made identical
            validation replays drift run-to-run — the same bug class fixed
            for screening in bd vibe-quant-1gvyc. None = non-deterministic.
        stop_slippage_ticks: Adverse ticks applied to every triggered
            stop-market fill (SPEC: "stop price + 1-tick slippage"). NT's
            matching engine can only slip by exactly one tick, so 0 or 1.
    """

    impact_coefficient: float = 0.1
    prob_fill_on_limit: float = 0.8
    prob_best_price_fill: float = 1.0
    max_adverse_ticks: int = 1
    prob_slippage: float = 0.0
    random_seed: int | None = 42
    stop_slippage_ticks: int = 1


class VolumeSlippageFillModel(FillModel):  # type: ignore[misc]
    """Fill model for validation backtesting with volume-based slippage estimation.

    NautilusTrader's matching engine uses FillModel.is_slipped() to decide
    whether a fill gets 1-tick slippage. The slippage *amount* is fixed at
    1 tick internally and cannot be overridden via subclassing.

    In validation mode we disable engine probabilistic slippage by default
    (`prob_slippage=0.0`) and compute SPEC slippage post-fill via
    :class:`SlippageEstimator`. This avoids maintaining two competing
    slippage models for the same trade results.

    This class:
    1. Stores `impact_coefficient` for use by SlippageEstimator.
    2. Keeps explicit fill-probability controls for limit orders.
    3. Optionally simulates an adverse L2 order book for MARKET orders only,
       which gives sub-5m validation a realistic degradation path without
       forcing a catastrophic full-bar LatencyModel delay.
    4. Leaves resting limits and triggered stops to NT's own matching logic:
       limits fill at their limit price (never better), triggered stops at
       the trigger price (or the open on a gap) plus ``stop_slippage_ticks``
       adverse ticks. Returning a synthetic book for them made NT fill at
       the bar extreme (bd vibe-quant-e70tl.2).

    For realistic slippage *cost* estimation per the SPEC formula, use
    SlippageEstimator separately in post-fill analytics.
    """

    def __init__(
        self,
        config: VolumeSlippageFillModelConfig | None = None,
        *,
        impact_coefficient: float = 0.1,
        prob_fill_on_limit: float = 0.8,
        prob_best_price_fill: float = 1.0,
        max_adverse_ticks: int = 1,
        prob_slippage: float = 0.0,
        random_seed: int | None = 42,
        stop_slippage_ticks: int = 1,
    ) -> None:
        """Initialize VolumeSlippageFillModel.

        Args:
            config: Configuration object. If provided, other args are ignored.
            impact_coefficient: Market impact coefficient k.
            prob_fill_on_limit: Probability of limit fill.
            prob_best_price_fill: Probability market orders fill at best.
            max_adverse_ticks: Maximum adverse ticks when degrading a fill.
            prob_slippage: Probability of engine slippage on market orders.
            random_seed: Seed for probabilistic draws (None = non-deterministic).
            stop_slippage_ticks: Adverse ticks (0 or 1) on triggered stops.
        """
        if config is not None:
            impact_coefficient = config.impact_coefficient
            prob_fill_on_limit = config.prob_fill_on_limit
            prob_best_price_fill = config.prob_best_price_fill
            max_adverse_ticks = config.max_adverse_ticks
            prob_slippage = config.prob_slippage
            random_seed = config.random_seed
            stop_slippage_ticks = config.stop_slippage_ticks

        if not 0.0 <= prob_best_price_fill <= 1.0:
            raise ValueError(
                f"prob_best_price_fill must be between 0 and 1, got {prob_best_price_fill}"
            )
        if max_adverse_ticks < 1:
            raise ValueError(f"max_adverse_ticks must be >= 1, got {max_adverse_ticks}")
        if stop_slippage_ticks not in (0, 1):
            # NT's matching engine slips a fill by exactly one tick when
            # is_slipped() is True; larger values cannot be expressed.
            raise ValueError(f"stop_slippage_ticks must be 0 or 1, got {stop_slippage_ticks}")

        super().__init__(
            prob_fill_on_limit=prob_fill_on_limit,
            prob_slippage=prob_slippage,
            random_seed=random_seed,
        )

        self._impact_coefficient = impact_coefficient
        self._prob_best_price_fill = prob_best_price_fill
        self._max_adverse_ticks = max_adverse_ticks
        self._prob_slippage = prob_slippage
        self._stop_slippage_ticks = stop_slippage_ticks
        # Own RNG for is_slipped/synthetic-book draws: module-level random
        # is shared process state and unseeded -> non-reproducible replays.
        self._rng = random.Random(random_seed)
        # Order type of the fill in progress. NT calls
        # get_orderbook_for_fill_simulation() for an order immediately before
        # apply_fills() consults is_slipped() for that same order's fills, so
        # is_slipped() can treat market / stop / limit fills differently.
        self._fill_order_type: OrderType = OrderType.MARKET

    @property
    def impact_coefficient(self) -> float:
        """Get market impact coefficient."""
        return self._impact_coefficient

    @property
    def prob_best_price_fill(self) -> float:
        """Get best-price fill probability for aggressive orders."""
        return self._prob_best_price_fill

    @property
    def max_adverse_ticks(self) -> int:
        """Get the maximum adverse ticks applied when degrading fills."""
        return self._max_adverse_ticks

    @property
    def stop_slippage_ticks(self) -> int:
        """Adverse ticks applied to triggered stop-market fills."""
        return self._stop_slippage_ticks

    def is_slipped(self) -> bool:
        """Decide whether NT moves the current fill one tick against the order.

        - Triggered stop-market fills: slipped iff ``stop_slippage_ticks`` is 1
          (SPEC "stop price + 1-tick slippage"); deterministic.
        - Limit fills: never (a resting limit fills at its price).
        - Market fills: probabilistic per ``prob_slippage`` (default 0.0 to
          avoid double-counting with post-fill SPEC slippage estimation).
        """
        order_type = self._fill_order_type
        if order_type in _TRIGGERED_MARKET_TYPES:
            return self._stop_slippage_ticks > 0
        if order_type != OrderType.MARKET:
            return False
        if self._prob_slippage <= 0.0:
            return False
        if self._prob_slippage >= 1.0:
            return True
        return self._rng.random() < self._prob_slippage

    def get_orderbook_for_fill_simulation(
        self,
        instrument: Any,
        order: Any,
        best_bid: Any,
        best_ask: Any,
    ) -> Any:
        """Return a synthetic L2 book for market-order degradation, else None.

        Only plain MARKET orders get the synthetic book. For every other order
        type NT's default logic must run: ``None`` makes the matching engine
        clamp maker fills to the limit price and fill triggered stops at the
        trigger price (or at the open on a gap). A synthetic book here made NT
        fill them at the current bar extreme (bd vibe-quant-e70tl.2).
        """
        from nautilus_trader.core.rust.model import BookType, OrderSide
        from nautilus_trader.model.book import OrderBook
        from nautilus_trader.model.data import BookOrder
        from nautilus_trader.model.objects import Quantity

        self._fill_order_type = order.order_type
        if order.order_type != OrderType.MARKET:
            return None

        book = OrderBook(
            instrument_id=instrument.id,
            book_type=BookType.L2_MBP,
        )
        unlimited_size = Quantity(1_000_000, instrument.size_precision)
        bid_price = best_bid
        ask_price = best_ask

        if self._rng.random() > self._prob_best_price_fill:
            adverse_ticks = (
                1
                if self._max_adverse_ticks == 1
                else self._rng.randint(1, self._max_adverse_ticks)
            )
            bid_price = self._shift_price(best_bid, instrument.price_increment, -adverse_ticks)
            ask_price = self._shift_price(best_ask, instrument.price_increment, adverse_ticks)

        book.add(
            BookOrder(
                side=OrderSide.BUY,
                price=bid_price,
                size=unlimited_size,
                order_id=1,
            ),
            0,
            0,
        )
        book.add(
            BookOrder(
                side=OrderSide.SELL,
                price=ask_price,
                size=unlimited_size,
                order_id=2,
            ),
            0,
            0,
        )

        return book

    def _shift_price(self, price: Any, tick: Any, steps: int) -> Any:
        """Move a price up or down by N instrument ticks."""
        shifted_value = price.as_double() + tick.as_double() * steps
        return type(price)(shifted_value, precision=price.precision)


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
        market_impact = self._k * volatility * math.sqrt(abs(order_size) / avg_volume)
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
        random_seed: Seed for the fill model RNG. NT's FillModel is unseeded by
            default, so with ``prob_slippage > 0`` identical screening runs draw
            different slippage realizations and metrics drift in the low-order
            digits run-to-run (the secondary cause in bd vibe-quant-1gvyc). Set a
            fixed seed to make screening/discovery fitness byte-reproducible.
    """

    prob_fill_on_limit: float = 0.8
    prob_slippage: float = 0.5
    random_seed: int | None = None


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
        random_seed=config.random_seed,
    )


def create_validation_fill_model(
    config: VolumeSlippageFillModelConfig | None = None,
) -> VolumeSlippageFillModel:
    """Create volume-based FillModel for validation mode.

    Validation uses realistic fill probability. For slippage cost estimation,
    use SlippageEstimator separately with `prob_slippage=0.0` to avoid
    double-counting against engine tick slippage.

    Args:
        config: Configuration for the model.

    Returns:
        VolumeSlippageFillModel for validation execution simulation.
    """
    if config is None:
        config = VolumeSlippageFillModelConfig()

    return VolumeSlippageFillModel(config=config)
