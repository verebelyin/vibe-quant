"""Tests for indicator registry."""

from __future__ import annotations

import pytest

from vibe_quant.dsl.indicators import IndicatorRegistry, IndicatorSpec, indicator_registry


class TestIndicatorSpec:
    """Tests for IndicatorSpec dataclass."""

    def test_valid_spec_with_nt_class(self) -> None:
        """Spec with only nt_class is valid."""
        spec = IndicatorSpec(
            name="TEST",
            nt_class=object,  # Dummy class
            pandas_ta_func=None,
            default_params={"period": 14},
            param_schema={"period": int},
        )
        assert spec.name == "TEST"
        assert spec.nt_class is not None
        assert spec.pandas_ta_func is None

    def test_valid_spec_with_pandas_ta(self) -> None:
        """Spec with only pandas_ta_func is valid."""
        spec = IndicatorSpec(
            name="TEST",
            nt_class=None,
            pandas_ta_func="test",
            default_params={"period": 14},
            param_schema={"period": int},
        )
        assert spec.name == "TEST"
        assert spec.nt_class is None
        assert spec.pandas_ta_func == "test"

    def test_valid_spec_with_both(self) -> None:
        """Spec with both nt_class and pandas_ta_func is valid."""
        spec = IndicatorSpec(
            name="TEST",
            nt_class=object,
            pandas_ta_func="test",
            default_params={},
            param_schema={},
        )
        assert spec.nt_class is not None
        assert spec.pandas_ta_func is not None

    def test_invalid_spec_neither(self) -> None:
        """Spec with none of nt_class / compute_fn / pandas_ta_func raises."""
        with pytest.raises(
            ValueError,
            match="must have nt_class, compute_fn, or pandas_ta_func",
        ):
            IndicatorSpec(
                name="TEST",
                nt_class=None,
                pandas_ta_func=None,
                default_params={},
                param_schema={},
            )

    def test_default_output_names(self) -> None:
        """Default output_names is ('value',)."""
        spec = IndicatorSpec(
            name="TEST",
            nt_class=object,
            pandas_ta_func=None,
            default_params={},
            param_schema={},
        )
        assert spec.output_names == ("value",)

    def test_custom_output_names(self) -> None:
        """Custom output_names can be set."""
        spec = IndicatorSpec(
            name="TEST",
            nt_class=object,
            pandas_ta_func=None,
            default_params={},
            param_schema={},
            output_names=("upper", "lower"),
        )
        assert spec.output_names == ("upper", "lower")


class TestIndicatorRegistry:
    """Tests for IndicatorRegistry class."""

    def test_get_unknown_returns_none(self) -> None:
        """get() returns None for unknown indicator."""
        registry = IndicatorRegistry()
        assert registry.get("UNKNOWN") is None

    def test_register_decorator(self) -> None:
        """@register decorator adds indicator to registry."""
        registry = IndicatorRegistry()

        @registry.register("CUSTOM")
        def _custom_spec() -> IndicatorSpec:
            return IndicatorSpec(
                name="CUSTOM",
                nt_class=None,
                pandas_ta_func="custom",
                default_params={"period": 10},
                param_schema={"period": int},
            )

        spec = registry.get("CUSTOM")
        assert spec is not None
        assert spec.name == "CUSTOM"
        assert spec.default_params == {"period": 10}

    def test_register_case_insensitive(self) -> None:
        """get() is case-insensitive."""
        registry = IndicatorRegistry()
        registry.register_spec(
            IndicatorSpec(
                name="MYIND",
                nt_class=None,
                pandas_ta_func="myind",
                default_params={},
                param_schema={},
            )
        )
        assert registry.get("myind") is not None
        assert registry.get("MYIND") is not None
        assert registry.get("MyInd") is not None

    def test_register_name_mismatch_raises(self) -> None:
        """Register with mismatched name raises ValueError."""
        registry = IndicatorRegistry()

        with pytest.raises(ValueError, match="Registered name .* != spec.name"):

            @registry.register("FOO")
            def _bar_spec() -> IndicatorSpec:
                return IndicatorSpec(
                    name="BAR",
                    nt_class=None,
                    pandas_ta_func="bar",
                    default_params={},
                    param_schema={},
                )

    def test_register_spec_direct(self) -> None:
        """register_spec() adds indicator directly."""
        registry = IndicatorRegistry()
        spec = IndicatorSpec(
            name="DIRECT",
            nt_class=None,
            pandas_ta_func="direct",
            default_params={},
            param_schema={},
        )
        registry.register_spec(spec)
        assert registry.get("DIRECT") is spec

    def test_list_indicators_sorted(self) -> None:
        """list_indicators() returns sorted list."""
        registry = IndicatorRegistry()
        registry.register_spec(
            IndicatorSpec(
                name="ZZZ", nt_class=None, pandas_ta_func="z", default_params={}, param_schema={}
            )
        )
        registry.register_spec(
            IndicatorSpec(
                name="AAA", nt_class=None, pandas_ta_func="a", default_params={}, param_schema={}
            )
        )
        registry.register_spec(
            IndicatorSpec(
                name="MMM", nt_class=None, pandas_ta_func="m", default_params={}, param_schema={}
            )
        )
        assert registry.list_indicators() == ["AAA", "MMM", "ZZZ"]

    def test_has_nt_class(self) -> None:
        """has_nt_class() returns correct value."""
        registry = IndicatorRegistry()
        registry.register_spec(
            IndicatorSpec(
                name="WITHNT",
                nt_class=object,
                pandas_ta_func=None,
                default_params={},
                param_schema={},
            )
        )
        registry.register_spec(
            IndicatorSpec(
                name="WITHOUTNT",
                nt_class=None,
                pandas_ta_func="test",
                default_params={},
                param_schema={},
            )
        )
        assert registry.has_nt_class("WITHNT") is True
        assert registry.has_nt_class("WITHOUTNT") is False
        assert registry.has_nt_class("UNKNOWN") is False

    def test_has_pandas_ta(self) -> None:
        """has_pandas_ta() returns correct value."""
        registry = IndicatorRegistry()
        registry.register_spec(
            IndicatorSpec(
                name="WITHPTA",
                nt_class=None,
                pandas_ta_func="test",
                default_params={},
                param_schema={},
            )
        )
        registry.register_spec(
            IndicatorSpec(
                name="WITHOUTPTA",
                nt_class=object,
                pandas_ta_func=None,
                default_params={},
                param_schema={},
            )
        )
        assert registry.has_pandas_ta("WITHPTA") is True
        assert registry.has_pandas_ta("WITHOUTPTA") is False
        assert registry.has_pandas_ta("UNKNOWN") is False


class TestBuiltinIndicators:
    """Tests for built-in indicator registrations."""

    def test_rsi_registered(self) -> None:
        """RSI indicator is registered with correct defaults."""
        spec = indicator_registry.get("RSI")
        assert spec is not None
        assert spec.name == "RSI"
        assert spec.pandas_ta_func == "rsi"
        assert spec.default_params == {"period": 14}
        assert spec.param_schema == {"period": int}

    def test_ema_registered(self) -> None:
        """EMA indicator is registered."""
        spec = indicator_registry.get("EMA")
        assert spec is not None
        assert spec.name == "EMA"
        assert spec.pandas_ta_func == "ema"

    def test_sma_registered(self) -> None:
        """SMA indicator is registered."""
        spec = indicator_registry.get("SMA")
        assert spec is not None
        assert spec.pandas_ta_func == "sma"

    def test_macd_registered(self) -> None:
        """MACD indicator is registered with correct defaults."""
        spec = indicator_registry.get("MACD")
        assert spec is not None
        assert spec.name == "MACD"
        assert spec.default_params == {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        assert spec.output_names == ("macd", "signal", "histogram")

    def test_bbands_registered(self) -> None:
        """Bollinger Bands is registered with correct defaults."""
        spec = indicator_registry.get("BBANDS")
        assert spec is not None
        assert spec.default_params == {"period": 20, "std_dev": 2.0}
        assert spec.output_names == ("upper", "middle", "lower", "percent_b", "bandwidth")

    def test_atr_registered(self) -> None:
        """ATR indicator is registered."""
        spec = indicator_registry.get("ATR")
        assert spec is not None
        assert spec.pandas_ta_func == "atr"

    def test_stoch_registered(self) -> None:
        """Stochastic is registered with k/d outputs."""
        spec = indicator_registry.get("STOCH")
        assert spec is not None
        assert spec.output_names == ("k", "d")
        assert spec.default_params == {"period_k": 14, "period_d": 3}

    def test_obv_registered(self) -> None:
        """OBV indicator is registered."""
        spec = indicator_registry.get("OBV")
        assert spec is not None
        assert spec.pandas_ta_func == "obv"

    def test_vwap_registered(self) -> None:
        """VWAP indicator is registered."""
        spec = indicator_registry.get("VWAP")
        assert spec is not None
        assert spec.pandas_ta_func == "vwap"

    def test_cci_registered(self) -> None:
        """CCI indicator is registered."""
        spec = indicator_registry.get("CCI")
        assert spec is not None
        assert spec.default_params == {"period": 20}

    def test_willr_has_no_nt_class(self) -> None:
        """WILLR has no NT class (pandas-ta only)."""
        spec = indicator_registry.get("WILLR")
        assert spec is not None
        assert spec.nt_class is None
        assert spec.pandas_ta_func == "willr"

    def test_all_mvp_indicators_registered(self) -> None:
        """All MVP indicators from SPEC.md are registered."""
        mvp_indicators = {
            # Trend
            "RSI",
            "EMA",
            "SMA",
            "WMA",
            "DEMA",
            "TEMA",
            # Momentum
            "MACD",
            "STOCH",
            "CCI",
            "ROC",
            "WILLR",
            # Volatility
            "ATR",
            "BBANDS",
            "KC",
            "DONCHIAN",
            # Volume
            "OBV",
            "VWAP",
            "MFI",
        }
        registered = set(indicator_registry.list_indicators())
        missing = mvp_indicators - registered
        assert not missing, f"Missing indicators: {missing}"

    def test_kc_has_atr_multiplier(self) -> None:
        """Keltner Channel has atr_multiplier param."""
        spec = indicator_registry.get("KC")
        assert spec is not None
        assert "atr_multiplier" in spec.param_schema
        assert spec.output_names == ("upper", "middle", "lower")

    def test_donchian_registered(self) -> None:
        """Donchian Channel is registered."""
        spec = indicator_registry.get("DONCHIAN")
        assert spec is not None
        assert spec.output_names == ("upper", "middle", "lower", "position")

    def test_roc_registered(self) -> None:
        """Rate of Change is registered."""
        spec = indicator_registry.get("ROC")
        assert spec is not None
        assert spec.default_params == {"period": 10}

    def test_mfi_registered(self) -> None:
        """Money Flow Index is registered."""
        spec = indicator_registry.get("MFI")
        assert spec is not None
        assert spec.pandas_ta_func == "mfi"


class TestNTIndicatorCreation:
    """Tests for create_nt_indicator (without NautilusTrader installed)."""

    def test_create_unknown_raises(self) -> None:
        """create_nt_indicator raises for unknown indicator."""
        registry = IndicatorRegistry()
        with pytest.raises(ValueError, match="Unknown indicator"):
            registry.create_nt_indicator("UNKNOWN", {})

    def test_create_no_nt_class_raises(self) -> None:
        """create_nt_indicator raises for indicator without NT class."""
        registry = IndicatorRegistry()
        registry.register_spec(
            IndicatorSpec(
                name="PANDAS_ONLY",
                nt_class=None,
                pandas_ta_func="test",
                default_params={},
                param_schema={},
            )
        )
        with pytest.raises(ValueError, match="has no NautilusTrader class"):
            registry.create_nt_indicator("PANDAS_ONLY", {})

    def test_create_without_nt_installed_raises(self) -> None:
        """create_nt_indicator raises if NT not installed."""
        # Use singleton which has NT class as None (lazy load failed)
        spec = indicator_registry.get("RSI")
        # RSI has nt_class=None if NT not installed (lazy load returned None)
        if spec and spec.nt_class is None:
            with pytest.raises(ValueError, match="has no NautilusTrader class"):
                indicator_registry.create_nt_indicator("RSI", {"period": 14})


class TestIndicatorRegistryIntegration:
    """Integration tests for indicator registry with DSL schema."""

    def test_core_indicators_registered(self) -> None:
        """The SPEC.md MVP indicator set must all live in the registry.

        Post-P5 the schema is backed by the registry (no more static
        ``VALID_INDICATOR_TYPES`` frozenset), so this test just pins the
        MVP set so shrinking the registry never silently drops a core name.
        """
        required = {
            # Trend
            "EMA", "SMA", "WMA", "DEMA", "TEMA", "ICHIMOKU",
            # Momentum
            "RSI", "MACD", "STOCH", "CCI", "WILLR", "ROC", "ADX",
            # Volatility
            "ATR", "BBANDS", "KC", "DONCHIAN",
            # Volume
            "OBV", "VWAP", "MFI", "VOLSMA",
        }
        registered = set(indicator_registry.list_indicators())
        missing = required - registered
        assert not missing, f"Core MVP indicators not in registry: {missing}"


class TestExtendedIndicatorSpecFields:
    """Coverage for the callback-dispatch fields added by P1."""

    def test_spec_accepts_compute_fn_without_nt_class(self) -> None:
        """A spec with only compute_fn (no nt_class, no pandas_ta_func) is valid."""
        # compute_fn is only type-checked; the runtime only checks the callable
        # is not None, so a plain stub callable suffices (avoids importing
        # pandas inside the test which triggers a numpy double-import under
        # --cov in some environments).
        def _noop_compute(
            df: object, params: dict[str, object]
        ) -> object:
            return df

        spec = IndicatorSpec(
            name="COMPUTE_ONLY",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 14},
            param_schema={"period": int},
            compute_fn=_noop_compute,  # type: ignore[arg-type]
        )
        assert spec.nt_class is None
        assert spec.pandas_ta_func is None
        assert spec.compute_fn is _noop_compute

    def test_spec_rejects_when_all_execution_paths_none(self) -> None:
        """__post_init__ rejects a spec with none of the three paths."""
        with pytest.raises(
            ValueError,
            match="must have nt_class, compute_fn, or pandas_ta_func",
        ):
            IndicatorSpec(
                name="NO_PATH",
                nt_class=None,
                pandas_ta_func=None,
                default_params={},
                param_schema={},
            )

    def test_all_specs_returns_registered_set(self) -> None:
        """all_specs() returns every registered spec in name-sorted order."""
        registry = IndicatorRegistry()
        registry.register_spec(
            IndicatorSpec(
                name="ZZZ",
                nt_class=None,
                pandas_ta_func="z",
                default_params={},
                param_schema={},
            )
        )
        registry.register_spec(
            IndicatorSpec(
                name="AAA",
                nt_class=None,
                pandas_ta_func="a",
                default_params={},
                param_schema={},
            )
        )
        specs = registry.all_specs()
        assert [s.name for s in specs] == ["AAA", "ZZZ"]
        assert all(isinstance(s, IndicatorSpec) for s in specs)

    def test_all_specs_on_singleton_matches_list_indicators(self) -> None:
        """The singleton registry's all_specs() and list_indicators() agree."""
        names_from_specs = [s.name for s in indicator_registry.all_specs()]
        names_from_list = indicator_registry.list_indicators()
        assert names_from_specs == names_from_list

    def test_extended_metadata_fields_persisted(self) -> None:
        """Extended UI/GA fields round-trip through IndicatorSpec unchanged."""
        spec = IndicatorSpec(
            name="EXT",
            nt_class=None,
            pandas_ta_func="ext",
            default_params={"period": 14},
            param_schema={"period": int},
            display_name="Extended",
            description="Sample extended metadata",
            category="Momentum",
            popular=True,
            param_ranges={"period": (5.0, 50.0)},
            threshold_range=(20.0, 80.0),
            requires_high_low=True,
            requires_volume=False,
            nt_output_attrs={"value": "value", "k": "k"},
            computed_outputs={"percent_b": "compute_percent_b"},
        )
        assert spec.display_name == "Extended"
        assert spec.description == "Sample extended metadata"
        assert spec.category == "Momentum"
        assert spec.popular is True
        assert spec.param_ranges == {"period": (5.0, 50.0)}
        assert spec.threshold_range == (20.0, 80.0)
        assert spec.requires_high_low is True
        assert spec.requires_volume is False
        assert spec.nt_output_attrs == {"value": "value", "k": "k"}
        assert spec.computed_outputs == {"percent_b": "compute_percent_b"}

    def test_default_extended_metadata_fields(self) -> None:
        """Extended fields have sensible defaults on a minimal spec."""
        spec = IndicatorSpec(
            name="MIN",
            nt_class=None,
            pandas_ta_func="min",
            default_params={},
            param_schema={},
        )
        assert spec.nt_kwargs_fn is None
        assert spec.compute_fn is None
        assert spec.nt_output_attrs == {"value": "value"}
        assert spec.computed_outputs == {}
        assert spec.pta_lookback_fn is None
        assert spec.requires_high_low is False
        assert spec.requires_volume is False
        assert spec.display_name == ""
        assert spec.description == ""
        assert spec.category == "Custom"
        assert spec.popular is False
        assert spec.param_ranges == {}
        assert spec.threshold_range is None

    def test_default_nt_output_attrs_are_independent_instances(self) -> None:
        """Default factory for nt_output_attrs does not share state between specs."""
        a = IndicatorSpec(
            name="A", nt_class=None, pandas_ta_func="a", default_params={}, param_schema={}
        )
        b = IndicatorSpec(
            name="B", nt_class=None, pandas_ta_func="b", default_params={}, param_schema={}
        )
        # frozen=True blocks attribute rebinding, not mutation of a mutable
        # dict stored on the spec — so in-place .__setitem__ works and this
        # test asserts each spec got its own dict from the default_factory.
        a.nt_output_attrs["extra"] = "x"
        assert "extra" not in b.nt_output_attrs


class TestBuiltinSpecsBackwardCompat:
    """Built-in specs keep their pre-P1 shape (RSI, MACD, STOCH, BBANDS)."""

    def test_rsi_backward_compat(self) -> None:
        spec = indicator_registry.get("RSI")
        assert spec is not None
        assert spec.name == "RSI"
        assert spec.pandas_ta_func == "rsi"
        assert spec.default_params == {"period": 14}
        assert spec.param_schema == {"period": int}
        assert spec.output_names == ("value",)
        # New fields default cleanly
        assert spec.nt_output_attrs == {"value": "value"}
        # Populated by P3 migration — matches legacy INDICATOR_POOL range.
        assert spec.threshold_range == (25.0, 75.0)
        assert spec.param_ranges == {"period": (5.0, 50.0)}
        assert spec.category == "Momentum"
        assert spec.compute_fn is not None

    def test_macd_backward_compat(self) -> None:
        spec = indicator_registry.get("MACD")
        assert spec is not None
        assert spec.default_params == {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
        }
        assert spec.output_names == ("macd", "signal", "histogram")
        assert spec.pandas_ta_func == "macd"
        assert spec.nt_class is None  # intentional — MACD always uses pandas-ta

    def test_stoch_backward_compat(self) -> None:
        spec = indicator_registry.get("STOCH")
        assert spec is not None
        assert spec.output_names == ("k", "d")
        assert spec.default_params == {"period_k": 14, "period_d": 3}

    def test_bbands_backward_compat(self) -> None:
        spec = indicator_registry.get("BBANDS")
        assert spec is not None
        assert spec.default_params == {"period": 20, "std_dev": 2.0}
        assert spec.output_names == (
            "upper",
            "middle",
            "lower",
            "percent_b",
            "bandwidth",
        )
