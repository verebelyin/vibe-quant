"""NautilusTrader indicators with the TradingView / pandas-ta-classic definition.

NT's defaults differ from the standard definition (RSI averages with an EMA of
alpha 2/(n+1), ATR is an SMA of the true range, ROC looks back n-1 bars, Stochastics %D
is a ratio of sums, Bollinger bands use hlc3, Keltner uses an hlc3 EMA and an SMA band).
These classes subclass or compose NT's own Rust-backed classes (no NT source is copied)
so the compiled strategy, the fast tier and the chart all compute the same numbers.

NT seeds its averages from the first value, TradingView from a simple average of the
first n values: the difference fades out (Wilder RSI(14): ~1e-7 after 210 bars), so
parity is checked after a warm-up, never on the first bars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nautilus_trader.indicators import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    ExponentialMovingAverage,
    MovingAverageType,
    RateOfChange,
    RelativeStrengthIndex,
    SimpleMovingAverage,
)
from nautilus_trader.indicators.base import Indicator

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar


class WilderRelativeStrengthIndex(RelativeStrengthIndex):  # type: ignore[misc]
    """RSI with Wilder smoothing (alpha 1/n), the TradingView / ta.rsi definition."""

    def __init__(self, period: int) -> None:
        super().__init__(period, ma_type=MovingAverageType.WILDER)


class WilderAverageTrueRange(AverageTrueRange):  # type: ignore[misc]
    """ATR with Wilder smoothing (RMA of the true range), the ta.atr definition."""

    def __init__(self, period: int) -> None:
        super().__init__(period, ma_type=MovingAverageType.WILDER)


class NBarRateOfChange(RateOfChange):  # type: ignore[misc]
    """ROC over n bars: ``close / close[n] - 1`` (NT's ROC(n) looks back n-1 bars)."""

    def __init__(self, period: int) -> None:
        super().__init__(period + 1)


class SmaDStochastics(Indicator):  # type: ignore[misc]
    """Fast stochastic %K (period_k) with %D = SMA(period_d) of %K, like ta.stoch(smooth_k=1).

    A flat window (high == low) gives %K 0.0, as ta's ``non_zero_range`` does. %D starts
    averaging at the first defined %K and ``initialized`` means %D is defined.
    """

    def __init__(self, period_k: int, period_d: int) -> None:
        super().__init__(params=[period_k, period_d])
        self.period_k = period_k
        self.period_d = period_d
        self._range = DonchianChannel(period_k)
        self._d = SimpleMovingAverage(period_d)
        self.value_k = 0.0
        self.value_d = 0.0

    def handle_bar(self, bar: Bar) -> None:
        self.update_raw(bar.high.as_double(), bar.low.as_double(), bar.close.as_double())

    def update_raw(self, high: float, low: float, close: float) -> None:
        self._range.update_raw(high, low)
        if not self._range.initialized:
            self._set_has_inputs(True)
            return
        hh = self._range.upper
        ll = self._range.lower
        self.value_k = 0.0 if hh == ll else 100.0 * (close - ll) / (hh - ll)
        self._d.update_raw(self.value_k)
        self.value_d = self._d.value
        self._set_has_inputs(True)
        if self._d.initialized:
            self._set_initialized(True)

    def _reset(self) -> None:
        self._range.reset()
        self._d.reset()
        self.value_k = 0.0
        self.value_d = 0.0


class CloseBollingerBands(BollingerBands):  # type: ignore[misc]
    """Bollinger bands of the close (NT feeds hlc3), the ta.bbands definition."""

    def handle_bar(self, bar: Bar) -> None:
        close = bar.close.as_double()
        self.update_raw(close, close, close)


class CloseKeltnerChannel(Indicator):  # type: ignore[misc]
    """Keltner channel of the close: EMA(close) +- k * EMA(true range), the ta.kc definition."""

    def __init__(self, period: int, k_multiplier: float) -> None:
        super().__init__(params=[period, k_multiplier])
        self.period = period
        self.k_multiplier = k_multiplier
        self._ma = ExponentialMovingAverage(period)
        self._atr = AverageTrueRange(period, MovingAverageType.EXPONENTIAL)
        self.upper = 0.0
        self.middle = 0.0
        self.lower = 0.0

    def handle_bar(self, bar: Bar) -> None:
        self.update_raw(bar.high.as_double(), bar.low.as_double(), bar.close.as_double())

    def update_raw(self, high: float, low: float, close: float) -> None:
        self._ma.update_raw(close)
        self._atr.update_raw(high, low, close)
        band = self._atr.value * self.k_multiplier
        self.middle = self._ma.value
        self.upper = self.middle + band
        self.lower = self.middle - band
        self._set_has_inputs(True)
        if self._ma.initialized and self._atr.initialized:
            self._set_initialized(True)

    def _reset(self) -> None:
        self._ma.reset()
        self._atr.reset()
        self.upper = 0.0
        self.middle = 0.0
        self.lower = 0.0
