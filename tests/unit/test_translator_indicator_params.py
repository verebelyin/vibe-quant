"""Frontend DslConfig -> StrategyDSL translator keeps every indicator param the user edited.

Params the registry spec declares (ICHIMOKU tenkan/kijun/senkou, ADAPTIVE_RSI alpha,
SQUEEZE length, ...) used to be dropped, so the backtest ran spec defaults instead of the
edited strategy. Params nobody knows must fail loudly instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.plugins.example_adaptive_rsi import compute_adaptive_rsi
from vibe_quant.dsl.translator import _build_ref_map, translate_dsl_config

if TYPE_CHECKING:
    from collections.abc import Iterator


def _cfg(
    indicators: list[dict[str, Any]], entry: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    return {
        "general": {"strategy_type": "momentum", "symbols": ["BTCUSDT"], "timeframe": "1h"},
        "indicators": indicators,
        "conditions": {
            "entry": entry or [{"left": "price", "operator": ">", "right": "100"}],
            "exit": [],
        },
        "risk": {
            "stop_loss": {"type": "fixed_pct", "value": 2},
            "take_profit": {"type": "fixed_pct", "value": 4},
        },
        "time": {},
    }


def test_ichimoku_params_reach_canonical_config_and_compiled_strategy() -> None:
    cfg = _cfg([{"type": "ICHIMOKU", "params": {"tenkan": 7, "kijun": 30, "senkou": 60}}])
    out = translate_dsl_config(cfg, strategy_name="t_ichi")
    assert out["indicators"]["ichimoku"] == {
        "type": "ICHIMOKU",
        "tenkan": 7,
        "kijun": 30,
        "senkou": 60,
    }
    dsl = validate_strategy_dict(out)
    assert dsl.indicators["ichimoku"].model_extra == {"tenkan": 7, "kijun": 30, "senkou": 60}
    code = StrategyCompiler().compile(dsl)
    assert "ichimoku_tenkan: int = 7" in code
    assert "ichimoku_kijun: int = 30" in code
    assert "ichimoku_senkou: int = 60" in code


def test_plugin_alpha_reaches_canonical_config_typed_as_float() -> None:
    # The editor sends numbers; an integral alpha still has to satisfy the float param_schema
    cfg = _cfg([{"type": "ADAPTIVE_RSI", "params": {"period": 21, "alpha": 0.8}}])
    out = translate_dsl_config(cfg, strategy_name="t_arsi")
    assert out["indicators"]["adaptive_rsi_21"] == {
        "type": "ADAPTIVE_RSI",
        "period": 21,
        "alpha": 0.8,
    }
    dsl = validate_strategy_dict(out)
    assert dsl.indicators["adaptive_rsi_21"].model_extra == {"alpha": 0.8}
    assert "adaptive_rsi_21_alpha: float = 0.8" in StrategyCompiler().compile(dsl)

    whole = translate_dsl_config(
        _cfg([{"type": "ADAPTIVE_RSI", "params": {"period": 21, "alpha": 1}}]), "t_arsi_int"
    )
    alpha = whole["indicators"]["adaptive_rsi_21"]["alpha"]
    assert alpha == 1.0
    assert isinstance(alpha, float)


def test_squeeze_length_and_multipliers_are_kept() -> None:
    cfg = _cfg(
        [{"type": "SQUEEZE_RATIO", "params": {"length": 30, "bb_mult": 2.5, "kc_mult": 1.2}}]
    )
    out = translate_dsl_config(cfg, strategy_name="t_squeeze")
    (name,) = out["indicators"]
    assert out["indicators"][name] == {
        "type": "SQUEEZE_RATIO",
        "length": 30,
        "bb_mult": 2.5,
        "kc_mult": 1.2,
    }
    assert validate_strategy_dict(out).indicators[name].model_extra == {
        "length": 30,
        "bb_mult": 2.5,
        "kc_mult": 1.2,
    }


def test_refs_to_indicators_with_extra_params_still_resolve() -> None:
    cfg = _cfg(
        [{"type": "ADAPTIVE_RSI", "params": {"period": 21, "alpha": 0.8}}],
        entry=[{"left": "ADAPTIVE_RSI(21,0.8)", "operator": "<", "right": "30"}],
    )
    out = translate_dsl_config(cfg, strategy_name="t_refs")
    assert out["entry_conditions"]["long"] == ["adaptive_rsi_21 < 30"]


@pytest.mark.parametrize(
    ("ind_type", "params", "bad"),
    [
        ("RSI", {"period": 14, "bogus": 3}, "bogus"),
        ("ICHIMOKU", {"tenkan": 9, "kijun": 26, "senkou_b": 52}, "senkou_b"),
        ("ADAPTIVE_RSI", {"period": 14, "alphaa": 0.5}, "alphaa"),
    ],
)
def test_unknown_indicator_param_raises_naming_indicator_and_param(
    ind_type: str, params: dict[str, Any], bad: str
) -> None:
    with pytest.raises(ValueError, match=rf"(?s)(?=.*{ind_type})(?=.*'{bad}')"):
        translate_dsl_config(_cfg([{"type": ind_type, "params": params}]), strategy_name="t_bad")


def test_fractional_value_for_integer_param_raises() -> None:
    # int(9.5) would silently run tenkan=9 while the user configured 9.5
    cfg = _cfg([{"type": "ICHIMOKU", "params": {"tenkan": 9.5, "kijun": 26, "senkou": 52}}])
    with pytest.raises(ValueError, match=r"ICHIMOKU.*'tenkan'"):
        translate_dsl_config(cfg, strategy_name="t_frac")


def test_integral_float_for_integer_param_is_accepted() -> None:
    # JSON round-trips can turn 9 into 9.0
    cfg = _cfg([{"type": "ICHIMOKU", "params": {"tenkan": 9.0, "kijun": 26, "senkou": 52}}])
    out = translate_dsl_config(cfg, strategy_name="t_integral")
    tenkan = out["indicators"]["ichimoku"]["tenkan"]
    assert tenkan == 9
    assert isinstance(tenkan, int)


_PLUGIN_NAME = "TEST_NATIVE_ALIAS"
_PLUGIN_STR_NAME = "TEST_NATIVE_ALIAS_STR"


@pytest.fixture
def alias_plugin() -> Iterator[IndicatorSpec]:
    """Plugin whose params (``fast``/``signal``/``mode``) collide with legacy editor spellings."""
    spec = IndicatorSpec(
        name=_PLUGIN_NAME,
        nt_class=None,
        pandas_ta_func=None,
        default_params={"fast": 10, "signal": False, "mode": "sma"},
        param_schema={"fast": int, "signal": bool, "mode": str},
        compute_fn=compute_adaptive_rsi,
        display_name="Native Alias Plugin",
        description="Synthetic plugin — do not ship.",
        category="Custom",
    )
    indicator_registry.register_spec(spec)
    try:
        yield spec
    finally:
        indicator_registry.unregister(_PLUGIN_NAME)


@pytest.fixture
def alias_str_plugin() -> Iterator[IndicatorSpec]:
    """Plugin whose ``fast`` param is a string (not a numeric period)."""
    spec = IndicatorSpec(
        name=_PLUGIN_STR_NAME,
        nt_class=None,
        pandas_ta_func=None,
        default_params={"fast": "sma"},
        param_schema={"fast": str},
        compute_fn=compute_adaptive_rsi,
        display_name="Native Alias String Plugin",
        description="Synthetic plugin — do not ship.",
        category="Custom",
    )
    indicator_registry.register_spec(spec)
    try:
        yield spec
    finally:
        indicator_registry.unregister(_PLUGIN_STR_NAME)


@pytest.mark.parametrize(
    "params",
    [
        {"fast": 18},
        {"signal": True},
        {"fast": 18, "signal": True, "mode": "ema"},
    ],
    ids=["int", "bool", "all"],
)
def test_plugin_native_params_win_over_legacy_synonyms(
    alias_plugin: IndicatorSpec, params: dict[str, Any]
) -> None:
    """A registry-native ``fast``/``signal`` is the plugin's own param, not an editor alias.

    Folding them onto ``fast_period``/``signal_period`` (or coercing them numerically)
    silently ran the plugin default plus an unrelated canonical field.
    """
    name = _PLUGIN_NAME.lower()
    expected = {"type": _PLUGIN_NAME, **params}
    out = translate_dsl_config(_cfg([{"type": _PLUGIN_NAME, "params": params}]), "t_native")
    assert out["indicators"] == {name: expected}
    dsl = validate_strategy_dict(out)
    assert dsl.indicators[name].model_extra == params
    # The canonical form translates to the same values the UI form produced.
    canonical = {**out, "indicators": {name: expected}}
    assert translate_dsl_config(canonical, strategy_name="t_native")["indicators"] == {
        name: expected
    }


def test_plugin_native_string_param_survives_ui_translation(
    alias_str_plugin: IndicatorSpec,
) -> None:
    """A string alias is passed through, not coerced to a number (which used to raise)."""
    name = _PLUGIN_STR_NAME.lower()
    expected = {"type": _PLUGIN_STR_NAME, "fast": "ema"}
    out = translate_dsl_config(
        _cfg([{"type": _PLUGIN_STR_NAME, "params": {"fast": "ema"}}]), "t_native_str"
    )
    assert out["indicators"] == {name: expected}
    assert validate_strategy_dict(out).indicators[name].model_extra == {"fast": "ema"}


@pytest.mark.parametrize(
    ("ind_type", "params", "bad"),
    [
        ("EMA", {"period": 12.5}, "period"),
        ("MACD", {"fast_period": 12.5, "slow_period": 26, "signal_period": 9}, "fast_period"),
        ("STOCH", {"period": 14, "d_period": 3.5}, "d_period"),
    ],
)
def test_fractional_legacy_int_param_raises_naming_indicator_and_param(
    ind_type: str, params: dict[str, Any], bad: str
) -> None:
    # int(12.5) would silently backtest EMA(12) / MACD fast 12 / STOCH %D 3, not the edited strategy
    cfg = _cfg([{"type": ind_type, "params": params}])
    with pytest.raises(ValueError, match=rf"(?s)(?=.*{ind_type})(?=.*'{bad}')"):
        translate_dsl_config(cfg, strategy_name="t_legacy_frac")


def test_integral_float_legacy_period_is_accepted_as_int() -> None:
    # JSON round-trips can turn 12 into 12.0; the user still configured period 12
    cfg = _cfg([{"type": "EMA", "params": {"period": 12.0}}])
    out = translate_dsl_config(cfg, strategy_name="t_legacy_int")
    period = out["indicators"]["ema_12"]["period"]
    assert period == 12
    assert isinstance(period, int)


@pytest.mark.parametrize(
    ("ind_type", "params", "expected"),
    [
        ("EMA", {"period": "14"}, {"period": 14}),
        (
            "MACD",
            {"fast_period": "12", "slow_period": "26", "signal_period": "9"},
            {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        ),
        (
            "MACD",
            {"fast": "12", "slow": "26", "signal": "9"},
            {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        ),
        ("STOCH", {"k_period": "14", "d_period": "3"}, {"period": 14, "d_period": 3}),
        ("EMA", {"period": "12.0"}, {"period": 12}),
    ],
)
def test_numeric_string_legacy_int_param_is_accepted_as_int(
    ind_type: str, params: dict[str, Any], expected: dict[str, int]
) -> None:
    # The YAML tab and LLM-extracted DSL spell a legacy period "14"; int() used to accept it
    cfg = _cfg([{"type": ind_type, "params": params}])
    (config,) = translate_dsl_config(cfg, strategy_name="t_str_int")["indicators"].values()
    for field, value in expected.items():
        assert config[field] == value
        assert isinstance(config[field], int)


@pytest.mark.parametrize(
    ("ind_type", "params", "bad"),
    [
        ("EMA", {"period": "12.5"}, "period"),
        (
            "MACD",
            {"fast_period": "12.5", "slow_period": "26", "signal_period": "9"},
            "fast_period",
        ),
        ("MACD", {"fast": "12.5", "slow": "26", "signal": "9"}, "fast_period"),
        ("STOCH", {"k_period": "14.5", "d_period": "3"}, "period"),
    ],
)
def test_fractional_numeric_string_legacy_int_param_raises_as_fractional(
    ind_type: str, params: dict[str, Any], bad: str
) -> None:
    # A numeric string is parsed, so "12.5" fails the whole-number rule (not "must be a number"):
    # int() used to silently run 12 while the file said 12.5
    cfg = _cfg([{"type": ind_type, "params": params}])
    with pytest.raises(ValueError, match=rf"(?s)(?=.*{ind_type})(?=.*'{bad}')(?=.*whole number)"):
        translate_dsl_config(cfg, strategy_name="t_str_frac")


@pytest.mark.parametrize(
    ("ind_type", "params", "bad"),
    [
        ("MACD", {"fast": True, "slow": 26, "signal": 9}, "fast_period"),
        ("MACD", {"fast": "abc", "slow": 26, "signal": 9}, "fast_period"),
    ],
)
def test_non_numeric_legacy_int_param_raises(
    ind_type: str, params: dict[str, Any], bad: str
) -> None:
    # True was silently period 1 and "abc" a TypeError/ValueError later; both now fail loudly
    cfg = _cfg([{"type": ind_type, "params": params}])
    with pytest.raises(
        ValueError, match=rf"(?s)(?=.*{ind_type})(?=.*'{bad}')(?=.*must be a number)"
    ):
        translate_dsl_config(cfg, strategy_name="t_non_numeric")


def test_null_legacy_int_param_is_dropped_not_rejected() -> None:
    # T7-R4 (supersedes vibe-quant-yubhb): an explicit null param means "not set", so
    # {k_period: None} no longer raises "must be a number"; it translates like the key absent.
    with_null = translate_dsl_config(
        _cfg([{"type": "STOCH", "params": {"k_period": None, "d_period": 3}}]),
        strategy_name="t_null",
    )
    without = translate_dsl_config(
        _cfg([{"type": "STOCH", "params": {"d_period": 3}}]), strategy_name="t_null"
    )
    assert with_null == without
    assert with_null["indicators"]["stoch"] == {"type": "STOCH", "d_period": 3}


@pytest.mark.parametrize(
    ("ind_type", "params", "ref", "expected_name"),
    [
        # MACD without signal: the ref's default signal 9 only comes from the legacy MACD(f,s,sig)
        ("MACD", {"fast_period": 12, "slow_period": 26}, "MACD(12,26,9)", "macd"),
        ("MACD", {"fast_period": "12", "slow_period": "26"}, "MACD(12,26,9)", "macd"),
        ("MACD", {"fast": "12.0", "slow": "26", "signal": "9"}, "MACD(12,26,9)", "macd"),
        # STOCH with d_period first: the TYPE(v1,v2) join reads STOCH(3,14), only STOCH(k,d) is 14,3
        ("STOCH", {"d_period": 3, "k_period": 14}, "STOCH(14,3)", "stoch"),
        ("STOCH", {"d_period": "3", "k_period": "14"}, "STOCH(14,3)", "stoch"),
        ("STOCH", {"period_d": "3.0", "period_k": "14"}, "STOCH(14,3)", "stoch"),
    ],
)
def test_legacy_macd_stoch_operand_ref_resolves_from_canonical_periods(
    ind_type: str, params: dict[str, Any], ref: str, expected_name: str
) -> None:
    cfg = _cfg(
        [{"type": ind_type, "params": params}],
        entry=[{"left": ref, "operator": ">", "right": "0"}],
    )
    out = translate_dsl_config(cfg, strategy_name="t_legacy_ref")
    assert out["entry_conditions"]["long"] == [f"{expected_name} > 0"]


def test_ref_map_rejects_non_numeric_canonical_period_instead_of_guessing_zero() -> None:
    # a 0 sentinel would build a bogus MACD(0,26,9) ref and hide the bad param
    with pytest.raises(ValueError, match=r"'abc'.*must be a number"):
        _build_ref_map(
            [{"type": "MACD", "params": {"fast_period": "abc", "slow_period": 26}}], ["macd"]
        )


def test_bool_spelling_conflicts_with_numeric_spelling_of_same_param() -> None:
    # float(True) == 1.0 used to make {fast_period: 1, fast: True} agree and fold True into period 1
    cfg = _cfg([{"type": "MACD", "params": {"fast_period": 1, "fast": True, "slow_period": 26}}])
    with pytest.raises(ValueError, match=r"Conflicting MACD params for 'fast_period'"):
        translate_dsl_config(cfg, strategy_name="t_bool_conflict")


def test_non_numeric_spelling_conflicts_with_numeric_spelling_of_same_param() -> None:
    cfg = _cfg([{"type": "MACD", "params": {"fast_period": 12, "fast": "abc", "slow_period": 26}}])
    with pytest.raises(ValueError, match=r"Conflicting MACD params for 'fast_period'"):
        translate_dsl_config(cfg, strategy_name="t_text_conflict")


def test_astronomically_large_period_raises_must_be_a_number() -> None:
    # float(10**400) is an OverflowError, which used to escape as a raw traceback
    cfg = _cfg([{"type": "EMA", "params": {"period": 10**400}}])
    with pytest.raises(ValueError, match=r"EMA param 'period'.*must be a number"):
        translate_dsl_config(cfg, strategy_name="t_huge")


@pytest.mark.parametrize("text", ["1_000", "0x10", "0b11", "1 000", "--1", "1e", "+"])
def test_non_decimal_string_period_raises_must_be_a_number(text: str) -> None:
    # Python float("1_000") is 1000.0 but JS Number("1_000") is NaN: only plain decimals count
    cfg = _cfg([{"type": "EMA", "params": {"period": text}}])
    with pytest.raises(ValueError, match=r"EMA param 'period'.*must be a number"):
        translate_dsl_config(cfg, strategy_name="t_non_decimal")


@pytest.mark.parametrize(("text", "period"), [(" 14 ", 14), ("1e1", 10), ("+7", 7), ("5.", 5)])
def test_plain_decimal_string_period_is_accepted(text: str, period: int) -> None:
    cfg = _cfg([{"type": "EMA", "params": {"period": text}}])
    (config,) = translate_dsl_config(cfg, strategy_name="t_decimal")["indicators"].values()
    assert config["period"] == period


@pytest.mark.parametrize("period", ["14", " 14 "])
def test_numeric_string_period_names_indicator_with_its_period(period: str) -> None:
    # the generated name carries the period, so a string "14" must name ema_14, not ema:
    # a string period used to be ignored by the name builder and refs pointed at the wrong indicator
    cfg = _cfg([{"type": "EMA", "params": {"period": period}}])
    out = translate_dsl_config(cfg, strategy_name="t_str_name")
    assert list(out["indicators"]) == ["ema_14"]


@pytest.mark.parametrize(
    "params",
    [
        {"fast_period": "12", "fast": 12, "slow_period": 26},
        {"fast_period": 12, "fast": 12.0},
    ],
    ids=["string-and-int", "int-and-float"],
)
def test_numeric_string_and_number_spellings_of_one_param_agree(
    params: dict[str, Any],
) -> None:
    # "12" and 12 are the same period: folding the spellings must not look like a conflict
    cfg = _cfg([{"type": "MACD", "params": params}])
    (config,) = translate_dsl_config(cfg, strategy_name="t_synonym")["indicators"].values()
    assert config["fast_period"] == 12
