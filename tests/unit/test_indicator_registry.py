"""Tests for indicator registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl.indicators import (
    IndicatorRegistry,
    IndicatorSpec,
    indicator_registry,
    invoke_compute_fn,
)


class TestIndicatorSpec:
    """Tests for IndicatorSpec dataclass."""

    def test_valid_spec_accepts_each_execution_path(self) -> None:
        """A spec is valid with any one of nt_class / pandas_ta_func / compute_fn."""
        nt_only = IndicatorSpec(
            name="TEST",
            nt_class=object,
            pandas_ta_func=None,
            default_params={"period": 14},
            param_schema={"period": int},
        )
        assert nt_only.nt_class is not None
        assert nt_only.pandas_ta_func is None
        pta_only = IndicatorSpec(
            name="TEST",
            nt_class=None,
            pandas_ta_func="test",
            default_params={"period": 14},
            param_schema={"period": int},
        )
        assert pta_only.nt_class is None
        assert pta_only.pandas_ta_func == "test"
        both = IndicatorSpec(
            name="TEST",
            nt_class=object,
            pandas_ta_func="test",
            default_params={},
            param_schema={},
        )
        assert both.nt_class is not None
        assert both.pandas_ta_func is not None

    def test_asof_context_requires_single_output_context_compute_fn(self) -> None:
        """asof_context needs needs_context, a compute_fn and exactly one output."""
        compute = lambda df, params: df["close"]  # noqa: E731 — a callable is all the check needs
        common: dict[str, object] = {
            "name": "TEST",
            "nt_class": None,
            "pandas_ta_func": None,
            "default_params": {"period": 2},
            "param_schema": {"period": int},
            "compute_fn": compute,
        }
        cases: list[tuple[str, dict[str, object]]] = [
            ("needs_context", {"needs_context": False}),
            ("compute_fn", {"needs_context": True, "compute_fn": None, "nt_class": object}),
            (
                "output_names",
                {"needs_context": True, "output_names": ("value", "other")},
            ),
        ]
        for marker, overrides in cases:
            with pytest.raises(ValueError, match=f"TEST.*{marker}"):
                IndicatorSpec(**{**common, **overrides, "asof_context": True})  # type: ignore[arg-type]

        exact = IndicatorSpec(**{**common, "needs_context": True}, asof_context=True)  # type: ignore[arg-type]
        assert exact.asof_context is True

    def test_registered_asof_context_specs(self) -> None:
        """Every needs_context compute_fn spec opts into the whole-frame replay."""
        flagged = ("TAKER_IMBALANCE", "FUNDING", "FUNDING_Z", "BTC_TREND", "OWN_TREND", "BTC_ROC")
        for name in flagged:
            spec = indicator_registry.get(name)
            assert spec is not None, name
            assert spec.asof_context is True, name
            assert spec.needs_context is True, name
            assert spec.output_names == ("value",), name
            assert spec.compute_fn is not None, name
        unflagged_context = {
            name
            for name in indicator_registry.list_indicators()
            if (spec := indicator_registry.get(name)) is not None and spec.needs_context
        }
        assert unflagged_context == set(flagged)

    def test_unknown_nt_output_attr_raises(self) -> None:
        """nt_output_attrs keys must be declared outputs when nt_class is set."""
        with pytest.raises(
            ValueError, match="nt_output_attrs keys not in output_names"
        ):
            IndicatorSpec(
                name="TEST",
                nt_class=object,
                pandas_ta_func=None,
                default_params={"period": 14},
                param_schema={"period": int},
                output_names=("value",),
                nt_output_attrs={"bogus_output": "value"},
            )

    def test_primary_output_not_in_output_names_raises(self) -> None:
        """primary_output, when set, must be a declared output name."""
        with pytest.raises(ValueError, match="primary_output"):
            IndicatorSpec(
                name="TEST",
                nt_class=None,
                pandas_ta_func="test",
                default_params={},
                param_schema={},
                output_names=("upper", "middle", "lower"),
                primary_output="missing",
            )

    def test_computed_outputs_unknown_helper_raises(self) -> None:
        """computed_outputs values must resolve to a helper in derived.py."""
        with pytest.raises(
            ValueError, match="computed_outputs helpers not found"
        ):
            IndicatorSpec(
                name="TEST",
                nt_class=None,
                pandas_ta_func="test",
                default_params={},
                param_schema={},
                output_names=("upper", "middle", "lower", "fake_output"),
                computed_outputs={"fake_output": "compute_nonexistent_helper"},
            )

    def test_param_ranges_typo_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A typo in param_ranges keys logs a warning (not a raise)."""
        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger="vibe_quant.dsl.indicators"):
            IndicatorSpec(
                name="TEST",
                nt_class=None,
                pandas_ta_func="test",
                default_params={"period": 14},
                param_schema={"period": int},
                param_ranges={"perriod": (5.0, 50.0)},  # typo
            )
        assert any(
            "perriod" in record.getMessage() for record in caplog.records
        ), "Expected warning about param_ranges typo"

    def test_unregister_removes_spec_case_insensitively(self) -> None:
        """unregister() drops the spec and returns True; repeat returns False."""
        registry = IndicatorRegistry()
        spec = IndicatorSpec(
            name="TEMP_UNREG",
            nt_class=None,
            pandas_ta_func="tmp",
            default_params={},
            param_schema={},
        )
        registry.register_spec(spec)
        assert registry.get("TEMP_UNREG") is not None
        assert registry.unregister("TEMP_UNREG") is True
        assert registry.get("TEMP_UNREG") is None
        assert registry.unregister("TEMP_UNREG") is False

        registry.register_spec(
            IndicatorSpec(
                name="TEMP_CASE",
                nt_class=None,
                pandas_ta_func="tmp",
                default_params={},
                param_schema={},
            )
        )
        assert registry.unregister("temp_case") is True

    def test_valid_computed_output_helper_passes(self) -> None:
        """Referencing an existing derived helper must not raise."""
        # compute_percent_b lives in vibe_quant.dsl.derived — real builtin.
        spec = IndicatorSpec(
            name="TEST",
            nt_class=None,
            pandas_ta_func="test",
            default_params={},
            param_schema={},
            output_names=("upper", "middle", "lower", "percent_b"),
            computed_outputs={"percent_b": "compute_percent_b"},
        )
        assert spec.computed_outputs == {"percent_b": "compute_percent_b"}

    def test_custom_output_names(self) -> None:
        """Custom output_names can be set."""
        spec = IndicatorSpec(
            name="TEST",
            nt_class=object,
            pandas_ta_func=None,
            default_params={},
            param_schema={},
            output_names=("upper", "lower"),
            # Must declare nt_output_attrs for each output when nt_class
            # is set — the __post_init__ validator enforces this.
            nt_output_attrs={"upper": "upper", "lower": "lower"},
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

    def test_register_spec_case_insensitive(self) -> None:
        """register_spec() stores the instance itself; get() is case-insensitive."""
        registry = IndicatorRegistry()
        spec = IndicatorSpec(
            name="MYIND",
            nt_class=None,
            pandas_ta_func="myind",
            default_params={},
            param_schema={},
        )
        registry.register_spec(spec)
        assert registry.get("myind") is not None
        assert registry.get("MYIND") is spec
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


_BUILTIN_SPEC_PINS: dict[str, dict[str, object]] = {
    "RSI": {
        "pandas_ta_func": "rsi",
        "default_params": {"period": 14},
        "param_schema": {"period": int},
        "output_names": ("value",),
        "nt_output_attrs": {"value": "value"},
        "threshold_range": (25.0, 75.0),
        "param_ranges": {"period": (5.0, 50.0)},
        "category": "Momentum",
    },
    "MACD": {
        "pandas_ta_func": "macd",
        "nt_class": None,  # intentional — MACD always uses pandas-ta
        "default_params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "output_names": ("macd", "signal", "histogram"),
    },
    "STOCH": {
        "output_names": ("k", "d"),
        "default_params": {"period_k": 14, "period_d": 3},
    },
    "BBANDS": {
        "default_params": {"period": 20, "std_dev": 2.0},
        "output_names": ("upper", "middle", "lower", "percent_b", "bandwidth"),
    },
    "EMA": {"pandas_ta_func": "ema"},
    "SMA": {"pandas_ta_func": "sma"},
    "ATR": {"pandas_ta_func": "atr"},
    "OBV": {"pandas_ta_func": "obv"},
    "VWAP": {"pandas_ta_func": "vwap"},
    "CCI": {"default_params": {"period": 20}},
    "ROC": {"default_params": {"period": 10}},
    "MFI": {"pandas_ta_func": "mfi"},
    "WILLR": {"nt_class": None, "pandas_ta_func": "willr"},  # pandas-ta only
    "KC": {"output_names": ("upper", "middle", "lower")},
    "DONCHIAN": {
        "output_names": (
            "upper",
            "middle",
            "lower",
            "position",
            "upper_prev",
            "lower_prev",
        ),
    },
}


class TestBuiltinIndicators:
    """Tests for built-in indicator registrations."""

    def test_builtin_spec_data(self) -> None:
        """Every built-in's execution path, params and outputs are pinned.

        One table, not one test per indicator: these are static registry data, and a
        silent change here runs the wrong indicator or the wrong period.
        """
        for name, expected in _BUILTIN_SPEC_PINS.items():
            spec = indicator_registry.get(name)
            assert spec is not None, name
            for field, want in expected.items():
                assert getattr(spec, field) == want, f"{name}.{field}"
        kc = indicator_registry.get("KC")
        rsi = indicator_registry.get("RSI")
        assert kc is not None and "atr_multiplier" in kc.param_schema
        assert rsi is not None and rsi.compute_fn is not None

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
        assert spec.output_names == ("value",)
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
    """Registry-wide invariants on the built-in specs."""

    def test_every_computed_output_is_a_declared_output(self) -> None:
        """A computed_outputs key outside output_names is unreferenceable dead config."""
        dead = {
            spec.name: sorted(set(spec.computed_outputs) - set(spec.output_names))
            for spec in indicator_registry.all_specs()
            if set(spec.computed_outputs) - set(spec.output_names)
        }
        assert dead == {}


def _ohlcv_frame(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": np.ones(n),
            "high": np.ones(n),
            "low": np.ones(n),
            "close": np.ones(n),
            "volume": np.ones(n),
        },
        index=idx,
    )


class TestInvokeComputeFn:
    """Tests for the compute_fn output-contract adapter."""

    def test_single_output_valid_series(self) -> None:
        spec = IndicatorSpec(
            name="TEST_SINGLE",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 3},
            param_schema={"period": int},
            compute_fn=lambda df, _p: df["close"] * 2,
        )
        df = _ohlcv_frame(5)
        result = invoke_compute_fn(spec, df, {"period": 3})
        assert isinstance(result, pd.Series)
        assert len(result) == 5
        assert (result == 2.0).all()

    def test_single_output_none_normalized_to_nan_series(self) -> None:
        spec = IndicatorSpec(
            name="TEST_NONE",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 3},
            param_schema={"period": int},
            compute_fn=lambda _df, _p: None,
        )
        df = _ohlcv_frame(4)
        result = invoke_compute_fn(spec, df, {})
        assert isinstance(result, pd.Series)
        assert len(result) == 4
        assert result.isna().all()
        assert result.index.equals(df.index)

    def test_single_output_wrong_length_raises(self) -> None:
        spec = IndicatorSpec(
            name="TEST_SHORT",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 3},
            param_schema={"period": int},
            compute_fn=lambda _df, _p: pd.Series([1.0, 2.0]),
        )
        df = _ohlcv_frame(5)
        with pytest.raises(ValueError, match="length 2, expected 5"):
            invoke_compute_fn(spec, df, {})

    def test_multi_output_valid_dict(self) -> None:
        def fn(df: pd.DataFrame, _p: dict[str, object]) -> dict[str, pd.Series]:
            return {"a": df["close"], "b": df["close"] * 2}

        spec = IndicatorSpec(
            name="TEST_MULTI",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 3},
            param_schema={"period": int},
            output_names=("a", "b"),
            compute_fn=fn,
        )
        df = _ohlcv_frame(4)
        result = invoke_compute_fn(spec, df, {})
        assert isinstance(result, dict)
        assert set(result) == {"a", "b"}
        assert (result["a"] == 1.0).all()
        assert (result["b"] == 2.0).all()

    def test_multi_output_none_normalized_to_nan_dict(self) -> None:
        spec = IndicatorSpec(
            name="TEST_MULTI_NONE",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 3},
            param_schema={"period": int},
            output_names=("a", "b"),
            compute_fn=lambda _df, _p: None,
        )
        df = _ohlcv_frame(3)
        result = invoke_compute_fn(spec, df, {})
        assert isinstance(result, dict)
        assert set(result) == {"a", "b"}
        for key in ("a", "b"):
            assert result[key].isna().all()
            assert len(result[key]) == 3

    def test_multi_output_missing_key_raises(self) -> None:
        def fn(df: pd.DataFrame, _p: dict[str, object]) -> dict[str, pd.Series]:
            return {"a": df["close"]}

        spec = IndicatorSpec(
            name="TEST_MULTI_MISSING",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 3},
            param_schema={"period": int},
            output_names=("a", "b"),
            compute_fn=fn,
        )
        df = _ohlcv_frame(3)
        with pytest.raises(ValueError, match="missing declared output keys"):
            invoke_compute_fn(spec, df, {})

    def test_multi_output_nested_none_normalized(self) -> None:
        """A dict whose value is None gets normalized to a NaN series."""

        def fn(df: pd.DataFrame, _p: dict[str, object]) -> dict[str, object]:
            return {"a": df["close"], "b": None}

        spec = IndicatorSpec(
            name="TEST_MULTI_PARTIAL_NONE",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 3},
            param_schema={"period": int},
            output_names=("a", "b"),
            compute_fn=fn,
        )
        df = _ohlcv_frame(4)
        result = invoke_compute_fn(spec, df, {})
        assert isinstance(result, dict)
        assert (result["a"] == 1.0).all()
        assert result["b"].isna().all()

    def test_multi_output_wrong_length_raises(self) -> None:
        def fn(_df: pd.DataFrame, _p: dict[str, object]) -> dict[str, pd.Series]:
            return {"a": pd.Series([1.0]), "b": pd.Series([1.0])}

        spec = IndicatorSpec(
            name="TEST_MULTI_SHORT",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 3},
            param_schema={"period": int},
            output_names=("a", "b"),
            compute_fn=fn,
        )
        df = _ohlcv_frame(5)
        with pytest.raises(ValueError, match="has length 1, expected 5"):
            invoke_compute_fn(spec, df, {})

    def test_single_output_dict_return_raises(self) -> None:
        """Single-output spec whose compute_fn returns a dict is a bug."""

        def fn(df: pd.DataFrame, _p: dict[str, object]) -> dict[str, pd.Series]:
            return {"value": df["close"]}

        spec = IndicatorSpec(
            name="TEST_SINGLE_DICT",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 3},
            param_schema={"period": int},
            compute_fn=fn,
        )
        df = _ohlcv_frame(3)
        with pytest.raises(ValueError, match="returned a dict"):
            invoke_compute_fn(spec, df, {})

    def test_multi_output_bare_series_raises(self) -> None:
        """Multi-output spec whose compute_fn returns a bare Series is a bug."""
        spec = IndicatorSpec(
            name="TEST_MULTI_BARE",
            nt_class=None,
            pandas_ta_func=None,
            default_params={"period": 3},
            param_schema={"period": int},
            output_names=("a", "b"),
            compute_fn=lambda df, _p: df["close"],
        )
        df = _ohlcv_frame(3)
        with pytest.raises(ValueError, match="returned a bare Series"):
            invoke_compute_fn(spec, df, {})

    def test_raises_on_spec_without_compute_fn(self) -> None:
        spec = IndicatorSpec(
            name="TEST_NO_FN",
            nt_class=object,
            pandas_ta_func=None,
            default_params={"period": 3},
            param_schema={"period": int},
        )
        df = _ohlcv_frame(3)
        with pytest.raises(ValueError, match="has no compute_fn"):
            invoke_compute_fn(spec, df, {})
