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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from nautilus_trader.indicators.base import Indicator
    from nautilus_trader.model.data import BarType


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    """Specification for a technical indicator.

    An IndicatorSpec is the single source of truth for a technical indicator. It
    can be constructed with any combination of three execution paths (at least
    one is required):

    1. ``nt_class`` — NautilusTrader Rust-native indicator class (fastest path)
    2. ``compute_fn`` — pure Python callback taking a DataFrame + params
    3. ``pandas_ta_func`` — legacy pandas-ta-classic function name (kept for
       backward compatibility while the compiler is migrated to callbacks)

    The compiler picks the first available path, preferring NT when the spec
    declares it and all referenced outputs live in ``nt_output_attrs``. Missing
    sub-values automatically force the ``compute_fn`` fallback.

    Attributes:
        name: Canonical indicator name (uppercase)
        nt_class: NautilusTrader indicator class (None if not available)
        pandas_ta_func: pandas-ta-classic function name (None if not available)
        default_params: Default parameter values
        param_schema: Parameter name to type mapping for validation
        output_names: Names of output values (e.g., ("value",) or
            ("upper", "middle", "lower"))
        nt_kwargs_fn: Callable mapping merged params → kwargs dict passed to
            ``nt_class(**kwargs)``. If None, defaults to
            ``{"period": params.get("period", 14)}``.
        compute_fn: Pure Python indicator computation from a rolling bar buffer.
            Signature: ``(df, params) -> pd.Series | dict[str, pd.Series]``
            where ``df`` has columns ``[open, high, low, close, volume]``.
            Single-output indicators return a Series; multi-output return a
            dict keyed by output name.
        nt_output_attrs: Maps ``output_name`` → NT indicator instance attribute
            name. Defaults to ``{"value": "value"}``. Outputs not present in
            this map force the ``compute_fn`` fallback path at compile time
            (generalizes the MACD signal/histogram quirk).
        computed_outputs: Output names whose values are derived at runtime from
            other outputs. Value is the name of a helper function in
            ``vibe_quant/dsl/derived.py``. e.g. BBANDS:
            ``{"percent_b": "compute_percent_b"}``.
        pta_lookback_fn: Callable returning the minimum number of bars required
            before ``compute_fn`` output is valid. If None, the compiler falls
            back to ``max(int params) * 2``.
        requires_high_low: Indicator needs high/low price series (ATR, STOCH,
            CCI, ADX, WILLR, KC, DONCHIAN, etc.).
        requires_volume: Indicator needs volume series (MFI, OBV, VWAP,
            VOLSMA).
        display_name: Human-readable name for UI (falls back to ``name``).
        description: One-line description for UI tooltips.
        category: UI category — one of ``"Trend"``, ``"Momentum"``,
            ``"Volatility"``, ``"Volume"``, or ``"Custom"``.
        popular: Flag used by the UI to highlight commonly-used indicators.
        param_ranges: Bounds the GA uses when mutating parameters, e.g.
            ``{"period": (5, 50)}``.
        threshold_range: Expected output range for threshold mutation in the
            GA, e.g. ``(25.0, 75.0)`` for RSI. ``None`` excludes the indicator
            from GA discovery (for price-relative indicators like EMA/SMA).
    """

    name: str
    nt_class: type | None
    pandas_ta_func: str | None
    default_params: dict[str, object]
    param_schema: dict[str, type]
    output_names: tuple[str, ...] = field(default=("value",))

    # Callback-based dispatch (optional; populated during Phase 3 migration).
    nt_kwargs_fn: Callable[[dict[str, object]], dict[str, object]] | None = None
    compute_fn: (
        Callable[[pd.DataFrame, dict[str, object]], Any]
        | None
    ) = None
    nt_output_attrs: dict[str, str] = field(
        default_factory=lambda: {"value": "value"}
    )
    computed_outputs: dict[str, str] = field(default_factory=dict)
    pta_lookback_fn: Callable[[dict[str, object]], int] | None = None

    # Price-source requirements.
    requires_high_low: bool = False
    requires_volume: bool = False

    # UI / catalog metadata (absorbed from indicator_metadata.py in Phase 7).
    display_name: str = ""
    description: str = ""
    category: str = "Custom"
    popular: bool = False

    # GA enrollment metadata (absorbed from discovery/genome.py in Phase 6).
    param_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    threshold_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate indicator spec.

        A spec must be runnable via at least one execution path: ``nt_class``,
        ``compute_fn``, or ``pandas_ta_func``.
        """
        if (
            self.nt_class is None
            and self.compute_fn is None
            and self.pandas_ta_func is None
        ):
            msg = (
                f"Indicator '{self.name}' must have nt_class, compute_fn, "
                "or pandas_ta_func"
            )
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

    def all_specs(self) -> list[IndicatorSpec]:
        """Return every registered spec in name-sorted order.

        Used by dynamic consumers (GA pool builder, frontend catalog API,
        plugin discovery sanity checks) that need the full spec payload
        rather than just names.

        Returns:
            List of IndicatorSpec sorted by name.
        """
        return [self._indicators[name] for name in sorted(self._indicators)]

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
        if name in {"RSI", "EMA", "SMA", "WMA", "DEMA", "TEMA", "ATR", "CCI", "ROC", "MFI", "ADX"}:
            if "period" in params:
                kwargs["period"] = params["period"]
        elif name == "MACD":
            kwargs["fast_period"] = params.get("fast_period", 12)
            kwargs["slow_period"] = params.get("slow_period", 26)
            # NT MovingAverageConvergenceDivergence does not accept signal_period
        elif name == "BBANDS":
            kwargs["period"] = params.get("period", 20)
            kwargs["k"] = params.get("std_dev", 2.0)
        elif name == "STOCH":
            kwargs["period_k"] = params.get("period_k", params.get("period", 14))
            kwargs["period_d"] = params.get("period_d", 3)
        elif name == "KC":
            kwargs["period"] = params.get("period", 20)
            kwargs["k_multiplier"] = params.get("atr_multiplier", 2.0)
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
# Built-in compute_fn imports (populated on specs below; compiler ignores them
# until Phase 4). Keeping the imports here (rather than lazily) means every
# compute_fn is eagerly resolved at registration time so a typo surfaces at
# startup instead of during a backtest.
# -----------------------------------------------------------------------------

from vibe_quant.dsl.compute_builtins import (  # noqa: E402
    compute_adx,
    compute_atr,
    compute_bbands,
    compute_cci,
    compute_dema,
    compute_donchian,
    compute_ema,
    compute_ichimoku,
    compute_kc,
    compute_macd,
    compute_mfi,
    compute_obv,
    compute_roc,
    compute_rsi,
    compute_sma,
    compute_stoch,
    compute_tema,
    compute_volsma,
    compute_vwap,
    compute_willr,
    compute_wma,
)

# -----------------------------------------------------------------------------
# Per-indicator nt_kwargs_fn helpers. Each one takes merged params and returns
# the kwargs dict passed to the NT indicator constructor. Mirrors the existing
# hardcoded elif chain in ``IndicatorRegistry._build_nt_kwargs`` — the compiler
# still uses that elif chain until Phase 4; these helpers let the new callback
# path light up without touching the compiler.
# -----------------------------------------------------------------------------


def _int_from(params: dict[str, object], key: str, default: int) -> int:
    """Coerce a dict entry to int with a fallback.

    ``IndicatorSpec`` stores params as ``dict[str, object]`` because the
    compiler receives both literal ints and pydantic-validated floats.
    Callback helpers need a uniform way to pull ints out without sprinkling
    ``# type: ignore`` comments everywhere.
    """
    val = params.get(key, default)
    if isinstance(val, (int, float)):
        return int(val)
    return default


def _period_kwargs(params: dict[str, object]) -> dict[str, object]:
    return {"period": params["period"]} if "period" in params else {}


def _macd_kwargs(params: dict[str, object]) -> dict[str, object]:
    return {
        "fast_period": params.get("fast_period", 12),
        "slow_period": params.get("slow_period", 26),
    }


def _bbands_kwargs(params: dict[str, object]) -> dict[str, object]:
    return {
        "period": params.get("period", 20),
        "k": params.get("std_dev", 2.0),
    }


def _stoch_kwargs(params: dict[str, object]) -> dict[str, object]:
    return {
        "period_k": params.get("period_k", params.get("period", 14)),
        "period_d": params.get("period_d", 3),
    }


def _kc_kwargs(params: dict[str, object]) -> dict[str, object]:
    return {
        "period": params.get("period", 20),
        "k_multiplier": params.get("atr_multiplier", 2.0),
    }


def _donchian_kwargs(params: dict[str, object]) -> dict[str, object]:
    return {"period": params.get("period", 20)}


# -----------------------------------------------------------------------------
# Register built-in indicators
#
# Every built-in spec populates the new callback/UI/GA fields even though the
# compiler still reads only the pre-P1 fields (nt_class, pandas_ta_func,
# default_params, output_names). Phase 4 will flip the compiler to read the
# new fields; Phase 3 just puts the data in place so the migration is a pure
# code-deletion step.
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
        nt_kwargs_fn=_period_kwargs,
        compute_fn=compute_rsi,
        display_name="Relative Strength Index",
        description=(
            "Oscillator (0-100) measuring speed and magnitude of price changes. "
            "Classic overbought/oversold indicator."
        ),
        category="Momentum",
        popular=True,
        param_ranges={"period": (5.0, 50.0)},
        threshold_range=(25.0, 75.0),
    )


@indicator_registry.register("EMA")
def _ema_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="EMA",
        nt_class=_get_nt_class("nautilus_trader.indicators", "ExponentialMovingAverage"),
        pandas_ta_func="ema",
        default_params={"period": 14},
        param_schema={"period": int},
        nt_kwargs_fn=_period_kwargs,
        compute_fn=compute_ema,
        display_name="Exponential Moving Average",
        description=(
            "Weighted moving average giving more weight to recent prices. "
            "Reacts faster than SMA."
        ),
        category="Trend",
        popular=True,
        # Price-relative indicator → intentionally excluded from GA
        # (threshold_range=None keeps it out of the GA pool).
    )


@indicator_registry.register("SMA")
def _sma_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="SMA",
        nt_class=_get_nt_class("nautilus_trader.indicators", "SimpleMovingAverage"),
        pandas_ta_func="sma",
        default_params={"period": 14},
        param_schema={"period": int},
        nt_kwargs_fn=_period_kwargs,
        compute_fn=compute_sma,
        display_name="Simple Moving Average",
        description="Equal-weighted average of last N closing prices. Smooth but lagging.",
        category="Trend",
    )


@indicator_registry.register("WMA")
def _wma_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="WMA",
        nt_class=_get_nt_class("nautilus_trader.indicators", "WeightedMovingAverage"),
        pandas_ta_func="wma",
        default_params={"period": 14},
        param_schema={"period": int},
        nt_kwargs_fn=_period_kwargs,
        compute_fn=compute_wma,
        display_name="Weighted Moving Average",
        description="Linearly-weighted moving average. Middle ground between SMA and EMA.",
        category="Trend",
    )


@indicator_registry.register("DEMA")
def _dema_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="DEMA",
        nt_class=_get_nt_class("nautilus_trader.indicators", "DoubleExponentialMovingAverage"),
        pandas_ta_func="dema",
        default_params={"period": 14},
        param_schema={"period": int},
        nt_kwargs_fn=_period_kwargs,
        compute_fn=compute_dema,
        display_name="Double EMA",
        description="Double-smoothed EMA that reduces lag while maintaining smoothness.",
        category="Trend",
    )


@indicator_registry.register("TEMA")
def _tema_spec() -> IndicatorSpec:
    # NT does not have TripleExponentialMovingAverage; compute_fn is the
    # only path for runtime evaluation.
    return IndicatorSpec(
        name="TEMA",
        nt_class=None,
        pandas_ta_func="tema",
        default_params={"period": 14},
        param_schema={"period": int},
        compute_fn=compute_tema,
        pta_lookback_fn=lambda p: _int_from(p, "period", 14) * 3,
        display_name="Triple EMA",
        description="Triple-smoothed EMA with even less lag than DEMA.",
        category="Trend",
    )


# Momentum indicators


@indicator_registry.register("MACD")
def _macd_spec() -> IndicatorSpec:
    # Always uses pandas-ta: NT's MovingAverageConvergenceDivergence only
    # exposes the MACD line (.value), not signal or histogram. Strategies
    # using signal crossover silently compute wrong conditions with NT class.
    return IndicatorSpec(
        name="MACD",
        nt_class=None,
        pandas_ta_func="macd",
        default_params={"fast_period": 12, "slow_period": 26, "signal_period": 9},
        param_schema={"fast_period": int, "slow_period": int, "signal_period": int},
        output_names=("macd", "signal", "histogram"),
        compute_fn=compute_macd,
        nt_kwargs_fn=_macd_kwargs,
        nt_output_attrs={"value": "value"},  # Only macd line on NT; sub-values force compute_fn.
        pta_lookback_fn=lambda p: _int_from(p, "slow_period", 26) + _int_from(p, "signal_period", 9),
        display_name="MACD (Moving Average Convergence Divergence)",
        description=(
            "Trend-following momentum indicator showing the relationship between "
            "two EMAs. Signal line crossovers generate entries."
        ),
        category="Momentum",
        popular=True,
        param_ranges={
            "fast_period": (8.0, 21.0),
            "slow_period": (21.0, 50.0),
            "signal_period": (5.0, 13.0),
        },
        threshold_range=(-0.05, 0.05),
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
        compute_fn=compute_stoch,
        nt_kwargs_fn=_stoch_kwargs,
        nt_output_attrs={"k": "value_k", "d": "value_d"},
        requires_high_low=True,
        display_name="Stochastic Oscillator",
        description=(
            "Compares closing price to the range over N periods. "
            "Values 0-100 with 20/80 OB/OS levels."
        ),
        category="Momentum",
        # Note: genome.py uses k_period/d_period keys for the GA param sweep;
        # the DSL-level field names are period_k/period_d. Populate both
        # conventions so build_indicator_pool() can find either spelling.
        param_ranges={"period_k": (5.0, 21.0), "period_d": (3.0, 9.0)},
        threshold_range=(20.0, 80.0),
    )


@indicator_registry.register("CCI")
def _cci_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="CCI",
        nt_class=_get_nt_class("nautilus_trader.indicators", "CommodityChannelIndex"),
        pandas_ta_func="cci",
        default_params={"period": 20},
        param_schema={"period": int},
        nt_kwargs_fn=_period_kwargs,
        compute_fn=compute_cci,
        requires_high_low=True,
        display_name="Commodity Channel Index",
        description=(
            "Measures price deviation from statistical mean. "
            "Values typically between -200 and +200."
        ),
        category="Momentum",
        param_ranges={"period": (10.0, 50.0)},
        threshold_range=(-200.0, 200.0),
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
        compute_fn=compute_willr,
        requires_high_low=True,
        display_name="Williams %R",
        description="Momentum oscillator (-100 to 0). Similar to Stochastic but inverted scale.",
        category="Momentum",
        param_ranges={"period": (5.0, 30.0)},
        threshold_range=(-80.0, -20.0),
    )


@indicator_registry.register("ROC")
def _roc_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="ROC",
        nt_class=_get_nt_class("nautilus_trader.indicators", "RateOfChange"),
        pandas_ta_func="roc",
        default_params={"period": 10},
        param_schema={"period": int},
        nt_kwargs_fn=_period_kwargs,
        compute_fn=compute_roc,
        display_name="Rate of Change",
        description="Percentage change between current price and N periods ago.",
        category="Momentum",
        param_ranges={"period": (5.0, 30.0)},
        threshold_range=(-5.0, 5.0),
    )


@indicator_registry.register("ADX")
def _adx_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="ADX",
        nt_class=_get_nt_class("nautilus_trader.indicators", "DirectionalMovement"),
        pandas_ta_func="adx",
        default_params={"period": 14},
        param_schema={"period": int},
        nt_kwargs_fn=_period_kwargs,
        compute_fn=compute_adx,
        requires_high_low=True,
        display_name="Average Directional Index",
        description="Measures trend strength. ADX > 25 = trending, ADX < 20 = ranging.",
        category="Momentum",
        param_ranges={"period": (7.0, 30.0)},
        threshold_range=(15.0, 60.0),
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
        nt_kwargs_fn=_period_kwargs,
        compute_fn=compute_atr,
        requires_high_low=True,
        display_name="Average True Range",
        description=(
            "Measures market volatility by averaging the true range over N periods. "
            "Essential for position sizing and stop placement."
        ),
        category="Volatility",
        popular=True,
        param_ranges={"period": (5.0, 30.0)},
        threshold_range=(0.001, 0.15),
    )


@indicator_registry.register("BBANDS")
def _bbands_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="BBANDS",
        nt_class=_get_nt_class("nautilus_trader.indicators", "BollingerBands"),
        pandas_ta_func="bbands",
        default_params={"period": 20, "std_dev": 2.0},
        param_schema={"period": int, "std_dev": float},
        output_names=("upper", "middle", "lower", "percent_b", "bandwidth"),
        compute_fn=compute_bbands,
        nt_kwargs_fn=_bbands_kwargs,
        nt_output_attrs={"upper": "upper", "middle": "middle", "lower": "lower"},
        computed_outputs={
            "percent_b": "compute_percent_b",
            "bandwidth": "compute_bandwidth",
        },
        display_name="Bollinger Bands",
        description=(
            "Upper/lower bands at N standard deviations from SMA. "
            "Bands expand in high volatility, contract in low volatility."
        ),
        category="Volatility",
        popular=True,
        param_ranges={"period": (5.0, 50.0), "std_dev": (1.0, 3.0)},
        threshold_range=(0.0, 1.0),
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
        compute_fn=compute_kc,
        nt_kwargs_fn=_kc_kwargs,
        nt_output_attrs={"upper": "upper", "middle": "middle", "lower": "lower"},
        computed_outputs={"bandwidth": "compute_bandwidth"},
        requires_high_low=True,
        display_name="Keltner Channel",
        description="ATR-based envelope around EMA. More stable than Bollinger Bands.",
        category="Volatility",
        # KC is not in the legacy INDICATOR_POOL → stays out of GA
        # (threshold_range=None keeps build_indicator_pool from enrolling it).
    )


@indicator_registry.register("DONCHIAN")
def _donchian_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="DONCHIAN",
        nt_class=_get_nt_class("nautilus_trader.indicators", "DonchianChannel"),
        pandas_ta_func="donchian",
        default_params={"period": 20},
        param_schema={"period": int},
        output_names=("upper", "middle", "lower", "position"),
        compute_fn=compute_donchian,
        nt_kwargs_fn=_donchian_kwargs,
        nt_output_attrs={"upper": "upper", "middle": "middle", "lower": "lower"},
        computed_outputs={"position": "compute_position"},
        requires_high_low=True,
        display_name="Donchian Channel",
        description=(
            "Highest high and lowest low over N periods. "
            "Classic breakout indicator (turtle trading)."
        ),
        category="Volatility",
        param_ranges={"period": (5.0, 50.0)},
        threshold_range=(0.0, 1.0),
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
        compute_fn=compute_obv,
        requires_volume=True,
        display_name="On-Balance Volume",
        description=(
            "Cumulative volume indicator: adds volume on up days, subtracts on down days. "
            "Leading indicator."
        ),
        category="Volume",
    )


@indicator_registry.register("VWAP")
def _vwap_spec() -> IndicatorSpec:
    return IndicatorSpec(
        name="VWAP",
        nt_class=_get_nt_class("nautilus_trader.indicators", "VolumeWeightedAveragePrice"),
        pandas_ta_func="vwap",
        default_params={},
        param_schema={},
        compute_fn=compute_vwap,
        requires_high_low=True,
        requires_volume=True,
        display_name="Volume-Weighted Average Price",
        description=(
            "Average price weighted by volume. Resets daily. Institutional benchmark."
        ),
        category="Volume",
        popular=True,
    )


@indicator_registry.register("MFI")
def _mfi_spec() -> IndicatorSpec:
    # NT 1.222 doesn't expose MoneyFlowIndex as a top-level indicator.
    return IndicatorSpec(
        name="MFI",
        nt_class=_get_nt_class("nautilus_trader.indicators", "MoneyFlowIndex"),
        pandas_ta_func="mfi",
        default_params={"period": 14},
        param_schema={"period": int},
        nt_kwargs_fn=_period_kwargs,
        compute_fn=compute_mfi,
        requires_high_low=True,
        requires_volume=True,
        display_name="Money Flow Index",
        description=(
            "Volume-weighted RSI. Oscillator 0-100 incorporating both price and volume."
        ),
        category="Volume",
        param_ranges={"period": (5.0, 30.0)},
        threshold_range=(20.0, 80.0),
    )


# Trend indicators (SPEC Section 5 additions)


@indicator_registry.register("ICHIMOKU")
def _ichimoku_spec() -> IndicatorSpec:
    # NT doesn't have Ichimoku built-in; compute_fn is the only path.
    return IndicatorSpec(
        name="ICHIMOKU",
        nt_class=None,
        pandas_ta_func="ichimoku",
        default_params={"tenkan": 9, "kijun": 26, "senkou": 52},
        param_schema={"tenkan": int, "kijun": int, "senkou": int},
        output_names=("conversion", "base", "span_a", "span_b"),
        compute_fn=compute_ichimoku,
        requires_high_low=True,
        pta_lookback_fn=lambda p: max(
            _int_from(p, "tenkan", 9),
            _int_from(p, "kijun", 26),
            _int_from(p, "senkou", 52),
        ),
        display_name="Ichimoku Cloud",
        description=(
            "Multi-line trend system: conversion/base lines plus a forward-projected "
            "cloud of support/resistance."
        ),
        category="Trend",
    )


# Volume indicators (SPEC Section 5 additions)


@indicator_registry.register("VOLSMA")
def _volsma_spec() -> IndicatorSpec:
    # Volume SMA: SMA applied to the volume column. No dedicated NT class.
    return IndicatorSpec(
        name="VOLSMA",
        nt_class=None,
        pandas_ta_func="sma",  # Applied to volume column
        default_params={"period": 20},
        param_schema={"period": int},
        compute_fn=compute_volsma,
        requires_volume=True,
        display_name="Volume SMA",
        description="Simple moving average of volume — baseline for volume-anomaly filters.",
        category="Volume",
    )
