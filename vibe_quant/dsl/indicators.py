"""Indicator registry with NautilusTrader mapping and pandas-ta-classic fallback.

Provides a pluggable indicator system for the strategy DSL. Each indicator
specifies its NautilusTrader class (if available) and pandas-ta-classic
function name for fallback.

Example usage:
    from vibe_quant.dsl.indicators import indicator_registry

    # Get indicator spec
    rsi_spec = indicator_registry.get("RSI")

    # Create NT indicator instance
    rsi = indicator_registry.create_nt_indicator("RSI", {"period": 14}, bar_type)

    # List all indicators
    names = indicator_registry.list_indicators()

    # Register custom indicator
    @indicator_registry.register("MY_INDICATOR")
    def my_indicator_spec() -> IndicatorSpec:
        return IndicatorSpec(
            name="MY_INDICATOR",
            nt_class=None,
            pandas_ta_func="myind",
            default_params={"period": 14},
            param_schema={"period": int},
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from nautilus_trader.indicators.base import Indicator
    from nautilus_trader.model.data import BarType


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    """Specification for a technical indicator.

    Attributes:
        name: Canonical indicator name (uppercase)
        nt_class: NautilusTrader indicator class (None if not available)
        pandas_ta_func: pandas-ta-classic function name (None if not available)
        default_params: Default parameter values
        param_schema: Parameter name to type mapping for validation
        output_names: Names of output values (e.g., ["value"] or ["upper", "middle", "lower"])
    """

    name: str
    nt_class: type | None
    pandas_ta_func: str | None
    default_params: dict[str, object]
    param_schema: dict[str, type]
    output_names: tuple[str, ...] = field(default=("value",))

    def __post_init__(self) -> None:
        """Validate indicator spec."""
        if self.nt_class is None and self.pandas_ta_func is None:
            msg = f"Indicator '{self.name}' must have nt_class or pandas_ta_func"
            raise ValueError(msg)


class IndicatorRegistry:
    """Registry for technical indicators.

    Provides lookup, creation, and registration of indicator specifications.
    Thread-safe for read operations after initial registration.
    """

    def __init__(self) -> None:
        self._indicators: dict[str, IndicatorSpec] = {}
        self._nt_available: bool | None = None

    def register(
        self, name: str
    ) -> Callable[[Callable[[], IndicatorSpec]], Callable[[], IndicatorSpec]]:
        """Decorator to register an indicator spec factory.

        Args:
            name: Canonical indicator name (uppercase)

        Returns:
            Decorator function

        Example:
            @indicator_registry.register("RSI")
            def rsi_spec() -> IndicatorSpec:
                return IndicatorSpec(...)
        """
        upper_name = name.upper()

        def decorator(func: Callable[[], IndicatorSpec]) -> Callable[[], IndicatorSpec]:
            spec = func()
            if spec.name != upper_name:
                msg = f"Registered name '{upper_name}' != spec.name '{spec.name}'"
                raise ValueError(msg)
            self._indicators[upper_name] = spec
            return func

        return decorator

    def register_spec(self, spec: IndicatorSpec) -> None:
        """Directly register an indicator spec.

        Args:
            spec: Indicator specification to register
        """
        self._indicators[spec.name.upper()] = spec

    def get(self, name: str) -> IndicatorSpec | None:
        """Get indicator spec by name.

        Args:
            name: Indicator name (case-insensitive)

        Returns:
            IndicatorSpec if found, None otherwise
        """
        return self._indicators.get(name.upper())

    def list_indicators(self) -> list[str]:
        """List all registered indicator names.

        Returns:
            Sorted list of indicator names
        """
        return sorted(self._indicators.keys())

    def is_nt_available(self) -> bool:
        """Check if NautilusTrader is available.

        Returns:
            True if nautilus_trader is importable
        """
        if self._nt_available is None:
            try:
                import nautilus_trader  # noqa: F401

                self._nt_available = True
            except ImportError:
                self._nt_available = False
        return self._nt_available

    def create_nt_indicator(
        self,
        name: str,
        params: dict[str, object] | None = None,
        bar_type: BarType | None = None,
    ) -> Indicator:
        """Create a NautilusTrader indicator instance.

        Args:
            name: Indicator name (case-insensitive)
            params: Parameter overrides (merged with defaults)
            bar_type: Optional BarType for indicators that require it

        Returns:
            Configured indicator instance

        Raises:
            ValueError: If indicator not found or NT class unavailable
            ImportError: If NautilusTrader not installed
        """
        spec = self.get(name)
        if spec is None:
            available = ", ".join(self.list_indicators())
            msg = f"Unknown indicator '{name}'. Available: {available}"
            raise ValueError(msg)

        if spec.nt_class is None:
            msg = f"Indicator '{name}' has no NautilusTrader class (use pandas-ta)"
            raise ValueError(msg)

        if not self.is_nt_available():
            msg = "NautilusTrader not installed"
            raise ImportError(msg)

        # Merge params with defaults
        merged_params = dict(spec.default_params)
        if params:
            merged_params.update(params)

        # Build kwargs for NT indicator
        kwargs = self._build_nt_kwargs(spec, merged_params, bar_type)
        return spec.nt_class(**kwargs)

    def _build_nt_kwargs(
        self,
        spec: IndicatorSpec,
        params: dict[str, object],
        bar_type: BarType | None,
    ) -> dict[str, object]:
        """Build kwargs for NT indicator constructor.

        Maps DSL param names to NT constructor args.
        """
        name = spec.name
        kwargs: dict[str, object] = {}

        # Common mapping from DSL names to NT names
        if name in {"RSI", "EMA", "SMA", "WMA", "DEMA", "TEMA", "ATR", "CCI", "ROC", "MFI"}:
            if "period" in params:
                kwargs["period"] = params["period"]
        elif name == "MACD":
            kwargs["fast_period"] = params.get("fast_period", 12)
            kwargs["slow_period"] = params.get("slow_period", 26)
            kwargs["signal_period"] = params.get("signal_period", 9)
        elif name == "BBANDS":
            kwargs["period"] = params.get("period", 20)
            kwargs["k"] = params.get("std_dev", 2.0)
        elif name == "STOCH":
            kwargs["period_k"] = params.get("period_k", params.get("period", 14))
            kwargs["period_d"] = params.get("period_d", 3)
        elif name == "KC":
            kwargs["period"] = params.get("period", 20)
            kwargs["k"] = params.get("atr_multiplier", 2.0)
        elif name == "DONCHIAN":
            kwargs["period"] = params.get("period", 20)

        # Add bar_type if provided (some indicators need it)
        if bar_type is not None:
            kwargs["bar_type"] = bar_type

        return kwargs

    def has_nt_class(self, name: str) -> bool:
        """Check if indicator has a NautilusTrader implementation.

        Args:
            name: Indicator name

        Returns:
            True if NT class is available for this indicator
        """
        spec = self.get(name)
        return spec is not None and spec.nt_class is not None

    def has_pandas_ta(self, name: str) -> bool:
        """Check if indicator has a pandas-ta-classic implementation.

        Args:
            name: Indicator name

        Returns:
            True if pandas-ta function is available for this indicator
        """
        spec = self.get(name)
        return spec is not None and spec.pandas_ta_func is not None


# Singleton registry instance
indicator_registry = IndicatorRegistry()


# -----------------------------------------------------------------------------
# Helper to lazily load NT indicator classes
# -----------------------------------------------------------------------------


def _get_nt_class(module_path: str, class_name: str) -> type | None:
    """Lazily import NT indicator class.

    Returns None if NautilusTrader not installed.
    """
    try:
        import importlib

        module = importlib.import_module(module_path)
        cls: type = getattr(module, class_name)
        return cls
    except (ImportError, AttributeError):
        return None


# -----------------------------------------------------------------------------
# Register built-in indicators
# -----------------------------------------------------------------------------

# Trend indicators


@indicator_registry.register("RSI")
def _rsi_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="RSI",
        nt_class=_get_nt_class("nautilus_trader.indicators", "RelativeStrengthIndex"),
        pandas_ta_func="rsi",
        default_params={"period": 14},
        param_schema={"period": int},
    )


@indicator_registry.register("EMA")
def _ema_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="EMA",
        nt_class=_get_nt_class("nautilus_trader.indicators", "ExponentialMovingAverage"),
        pandas_ta_func="ema",
        default_params={"period": 14},
        param_schema={"period": int},
    )


@indicator_registry.register("SMA")
def _sma_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="SMA",
        nt_class=_get_nt_class("nautilus_trader.indicators", "SimpleMovingAverage"),
        pandas_ta_func="sma",
        default_params={"period": 14},
        param_schema={"period": int},
    )


@indicator_registry.register("WMA")
def _wma_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="WMA",
        nt_class=_get_nt_class("nautilus_trader.indicators", "WeightedMovingAverage"),
        pandas_ta_func="wma",
        default_params={"period": 14},
        param_schema={"period": int},
    )


@indicator_registry.register("DEMA")
def _dema_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="DEMA",
        nt_class=_get_nt_class("nautilus_trader.indicators", "DoubleExponentialMovingAverage"),
        pandas_ta_func="dema",
        default_params={"period": 14},
        param_schema={"period": int},
    )


@indicator_registry.register("TEMA")
def _tema_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="TEMA",
        nt_class=_get_nt_class("nautilus_trader.indicators", "TripleExponentialMovingAverage"),
        pandas_ta_func="tema",
        default_params={"period": 14},
        param_schema={"period": int},
    )


# Momentum indicators


@indicator_registry.register("MACD")
def _macd_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="MACD",
        nt_class=_get_nt_class("nautilus_trader.indicators", "MovingAverageConvergenceDivergence"),
        pandas_ta_func="macd",
        default_params={"fast_period": 12, "slow_period": 26, "signal_period": 9},
        param_schema={"fast_period": int, "slow_period": int, "signal_period": int},
        output_names=("macd", "signal", "histogram"),
    )


@indicator_registry.register("STOCH")
def _stoch_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="STOCH",
        nt_class=_get_nt_class("nautilus_trader.indicators", "Stochastics"),
        pandas_ta_func="stoch",
        default_params={"period_k": 14, "period_d": 3},
        param_schema={"period_k": int, "period_d": int},
        output_names=("k", "d"),
    )


@indicator_registry.register("CCI")
def _cci_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="CCI",
        nt_class=_get_nt_class("nautilus_trader.indicators", "CommodityChannelIndex"),
        pandas_ta_func="cci",
        default_params={"period": 20},
        param_schema={"period": int},
    )


@indicator_registry.register("WILLR")
def _willr_spec() -> IndicatorSpec:
    # NT doesn't have Williams %R built-in
    return IndicatorSpec(
        name="WILLR",
        nt_class=None,
        pandas_ta_func="willr",
        default_params={"period": 14},
        param_schema={"period": int},
    )


@indicator_registry.register("ROC")
def _roc_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="ROC",
        nt_class=_get_nt_class("nautilus_trader.indicators", "RateOfChange"),
        pandas_ta_func="roc",
        default_params={"period": 10},
        param_schema={"period": int},
    )


# Volatility indicators


@indicator_registry.register("ATR")
def _atr_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="ATR",
        nt_class=_get_nt_class("nautilus_trader.indicators", "AverageTrueRange"),
        pandas_ta_func="atr",
        default_params={"period": 14},
        param_schema={"period": int},
    )


@indicator_registry.register("BBANDS")
def _bbands_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="BBANDS",
        nt_class=_get_nt_class("nautilus_trader.indicators", "BollingerBands"),
        pandas_ta_func="bbands",
        default_params={"period": 20, "std_dev": 2.0},
        param_schema={"period": int, "std_dev": float},
        output_names=("upper", "middle", "lower"),
    )


@indicator_registry.register("KC")
def _kc_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="KC",
        nt_class=_get_nt_class("nautilus_trader.indicators", "KeltnerChannel"),
        pandas_ta_func="kc",
        default_params={"period": 20, "atr_multiplier": 2.0},
        param_schema={"period": int, "atr_multiplier": float},
        output_names=("upper", "middle", "lower"),
    )


@indicator_registry.register("DONCHIAN")
def _donchian_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="DONCHIAN",
        nt_class=_get_nt_class("nautilus_trader.indicators", "DonchianChannel"),
        pandas_ta_func="donchian",
        default_params={"period": 20},
        param_schema={"period": int},
        output_names=("upper", "middle", "lower"),
    )


# Volume indicators


@indicator_registry.register("OBV")
def _obv_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="OBV",
        nt_class=_get_nt_class("nautilus_trader.indicators", "OnBalanceVolume"),
        pandas_ta_func="obv",
        default_params={},
        param_schema={},
    )


@indicator_registry.register("VWAP")
def _vwap_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="VWAP",
        nt_class=_get_nt_class("nautilus_trader.indicators", "VolumeWeightedAveragePrice"),
        pandas_ta_func="vwap",
        default_params={},
        param_schema={},
    )


@indicator_registry.register("MFI")
def _mfi_spec() -> IndicatorSpec:
    # NT 1.222 doesn't have MoneyFlowIndex as a top-level indicator
    return IndicatorSpec(
        name="MFI",
        nt_class=_get_nt_class("nautilus_trader.indicators", "MoneyFlowIndex"),
        pandas_ta_func="mfi",
        default_params={"period": 14},
        param_schema={"period": int},
    )
