"""Frontend DslConfig -> StrategyDSL translator keeps every indicator param the user edited.

Params the registry spec declares (ICHIMOKU tenkan/kijun/senkou, ADAPTIVE_RSI alpha,
SQUEEZE length, ...) used to be dropped, so the backtest ran spec defaults instead of the
edited strategy. Params nobody knows must fail loudly instead.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.parser import DSLValidationError, validate_strategy_dict
from vibe_quant.dsl.plugins.example_adaptive_rsi import compute_adaptive_rsi
from vibe_quant.dsl.schema import check_strategy_indicator_params
from vibe_quant.dsl.translator import _build_ref_map, translate_dsl_config
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path


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
    # int(9.5) would silently run tenkan=9 while the user configured 9.5 (regression: vibe-quant-yubhb)
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
    ("params", "message"),
    [
        (
            {"fast": 18, "fast_period": 12},
            "TEST_NATIVE_ALIAS sets both its native param 'fast'=18 and the legacy alias "
            "'fast_period'=12; 'fast' is a native TEST_NATIVE_ALIAS param, not an alias of "
            "'fast_period': remove one",
        ),
        (
            {"signal": True, "signal_period": 9},
            "TEST_NATIVE_ALIAS sets both its native param 'signal'=True and the legacy alias "
            "'signal_period'=9; 'signal' is a native TEST_NATIVE_ALIAS param, not an alias of "
            "'signal_period': remove one",
        ),
    ],
)
def test_native_param_and_legacy_alias_together_raise(
    alias_plugin: IndicatorSpec, params: dict[str, Any], message: str
) -> None:
    """A plugin's native ``fast``/``signal`` is not the legacy ``fast_period``/``signal_period``.

    Setting both would hand the plugin its own value while the canonical field carried the legacy
    one -- two different periods on one indicator, with no way to tell which the user meant.

    The whole message is pinned (the UI mirrors it verbatim), not just the names in it.
    """
    cfg = _cfg([{"type": _PLUGIN_NAME, "params": params}])
    with pytest.raises(ValueError) as exc:
        translate_dsl_config(cfg, strategy_name="t_native_conflict")
    assert str(exc.value) == message


def test_native_param_alone_and_legacy_alias_alone_still_translate(
    alias_plugin: IndicatorSpec,
) -> None:
    name = _PLUGIN_NAME.lower()
    native_only = translate_dsl_config(
        _cfg([{"type": _PLUGIN_NAME, "params": {"fast": 18}}]), strategy_name="t_native_only"
    )
    assert native_only["indicators"] == {name: {"type": _PLUGIN_NAME, "fast": 18}}
    legacy_only = translate_dsl_config(
        _cfg([{"type": _PLUGIN_NAME, "params": {"fast_period": 12}}]),
        strategy_name="t_legacy_only",
    )
    assert legacy_only["indicators"] == {name: {"type": _PLUGIN_NAME, "fast_period": 12}}


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
    # int(12.5) would silently backtest EMA(12) / MACD fast 12 / STOCH %D 3, not the edited strategy (regression: vibe-quant-yubhb)
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


# ---------------------------------------------------------------------------
# Plugin float params must be finite (vibe-quant-d42m5 finding 5; rows R13-R16)
# ---------------------------------------------------------------------------

_HUGE = 10**400  # an int no float can hold


def _arsi_ui(alpha: object) -> dict[str, Any]:
    entry = [{"left": "adaptive_rsi_14", "operator": "<", "right": "30"}]
    return _cfg([{"type": "ADAPTIVE_RSI", "params": {"period": 14, "alpha": alpha}}], entry)


def _arsi_canonical(alpha: object) -> dict[str, Any]:
    return {
        "name": "t_finite",
        "timeframe": "1h",
        "indicators": {"arsi": {"type": "ADAPTIVE_RSI", "period": 14, "alpha": alpha}},
        "entry_conditions": {"long": ["arsi < 30"]},
        "exit_conditions": {"long": ["arsi > 70"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }


_NON_FINITE = [float("nan"), float("inf"), float("-inf"), _HUGE]
_NON_FINITE_IDS = ["nan", "inf", "-inf", "10**400"]


@pytest.fixture()
async def api(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, StateManager]]:
    """App on a temp DB (lifespan deps wired by hand, as the api tests do)."""
    app = create_app()
    state_mgr = StateManager(db_path=tmp_path / "test.db")
    _ = state_mgr.conn  # force schema init
    job_mgr = BacktestJobManager(tmp_path / "test.db")  # never the real default DB
    ws_mgr = ConnectionManager()
    await ws_mgr.start()
    app.state.state_manager = state_mgr
    app.state.job_manager = job_mgr
    app.state.catalog_manager = CatalogManager()
    app.state.ws_manager = ws_mgr
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, state_mgr
    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


def _json_body(payload: dict[str, Any]) -> dict[str, Any]:
    """Raw JSON (``NaN``/``Infinity`` tokens allowed: httpx's json= refuses them)."""
    return {
        "content": json.dumps(payload),
        "headers": {"content-type": "application/json"},
    }


@pytest.mark.parametrize("alpha", _NON_FINITE, ids=_NON_FINITE_IDS)
def test_plugin_float_params_must_be_finite(alpha: object) -> None:
    """R13/R14: both parse paths reject a non-finite or overflowing plugin float with a
    validation error naming the indicator and the parameter (never an OverflowError)."""
    with pytest.raises(ValueError, match=r"ADAPTIVE_RSI param 'alpha'.* must be finite"):
        translate_dsl_config(_arsi_ui(alpha), strategy_name="t_finite")
    with pytest.raises(ValueError, match=r"ADAPTIVE_RSI.* param 'alpha' must be finite"):
        check_strategy_indicator_params(_arsi_ui(alpha))
    with pytest.raises(ValueError, match=r"ADAPTIVE_RSI.* param 'alpha' must be finite"):
        check_strategy_indicator_params(_arsi_canonical(alpha))
    with pytest.raises(DSLValidationError) as excinfo:
        validate_strategy_dict(_arsi_canonical(alpha))
    assert "param 'alpha' must be finite" in " ".join(excinfo.value.details)


@pytest.mark.parametrize("alpha", _NON_FINITE, ids=_NON_FINITE_IDS)
@pytest.mark.parametrize("fmt", ["editor", "canonical"])
async def test_saving_a_non_finite_plugin_float_is_422(
    api: tuple[AsyncClient, StateManager], alpha: object, fmt: str
) -> None:
    client, _mgr = api
    dsl = _arsi_ui(alpha) if fmt == "editor" else _arsi_canonical(alpha)
    r = await client.post(
        "/api/strategies", **_json_body({"name": f"t_finite_{fmt}", "dsl_config": dsl})
    )
    assert r.status_code == 422, r.text
    assert "param 'alpha'" in r.text and "must be finite" in r.text and "ADAPTIVE_RSI" in r.text
    ok = await client.post(
        "/api/strategies", json={"name": f"t_ok_{fmt}", "dsl_config": _arsi_ui(0.8)}
    )
    assert ok.status_code == 201, ok.text
    sid = ok.json()["id"]
    u = await client.put(f"/api/strategies/{sid}", **_json_body({"dsl_config": dsl}))
    assert u.status_code == 422, u.text
    assert "must be finite" in u.text


@pytest.mark.parametrize("fmt", ["editor", "canonical"])
async def test_stored_non_finite_plugin_float_validates_false(
    api: tuple[AsyncClient, StateManager], fmt: str
) -> None:
    """R13: a NaN that reached storage before this check gets valid=false, not a 500."""
    client, mgr = api
    dsl = _arsi_ui(float("nan")) if fmt == "editor" else _arsi_canonical(float("nan"))
    sid = mgr.create_strategy(name=f"t_stored_{fmt}", dsl_config=dsl, description="")
    v = await client.post(f"/api/strategies/{sid}/validate")
    assert v.status_code == 200, v.text
    body = v.json()
    assert body["valid"] is False
    assert any("param 'alpha'" in e and "must be finite" in e for e in body["errors"]), body


@pytest.mark.parametrize(("alpha", "effective"), [(0.8, 0.8), (1, 1.0)])
async def test_finite_plugin_float_saves_as_sent_and_runs_as_float(
    api: tuple[AsyncClient, StateManager], alpha: object, effective: float
) -> None:
    """R15: storage keeps the request JSON as sent (``1`` stays ``1``); the executable DSL and
    the compiled strategy param hold the float."""
    client, _mgr = api
    r = await client.post(
        "/api/strategies", json={"name": f"t_alpha_{alpha}", "dsl_config": _arsi_ui(alpha)}
    )
    assert r.status_code == 201, r.text
    stored = r.json()["dsl_config"]["indicators"][0]["params"]["alpha"]
    assert stored == alpha and type(stored) is type(alpha)
    out = translate_dsl_config(_arsi_ui(alpha), strategy_name="t_alpha")
    got = out["indicators"]["adaptive_rsi_14"]["alpha"]
    assert got == effective and isinstance(got, float)
    code = StrategyCompiler().compile(validate_strategy_dict(out))
    assert f"adaptive_rsi_14_alpha: float = {effective!r}" in code
    canonical = validate_strategy_dict(_arsi_canonical(alpha))
    assert canonical.indicators["arsi"].model_extra == {"alpha": effective}


@pytest.mark.parametrize("alpha", ["0.5", "NaN", "Infinity", True])
def test_string_and_bool_plugin_floats_rejected_as_before(alpha: object) -> None:
    """R16 (main's behaviour pinned): strings and bools are type errors on both paths."""
    with pytest.raises(ValueError, match=r"ADAPTIVE_RSI param 'alpha'=.* must be a number"):
        translate_dsl_config(_arsi_ui(alpha), strategy_name="t_type")
    with pytest.raises(DSLValidationError) as excinfo:
        validate_strategy_dict(_arsi_canonical(alpha))
    assert f"param 'alpha' expected float, got {type(alpha).__name__}" in " ".join(
        excinfo.value.details
    )


async def test_null_plugin_float_is_not_set_in_the_editor_or_canonically(
    api: tuple[AsyncClient, StateManager],
) -> None:
    """Editor null is "not set" (saved, the executable DSL runs the plugin default); a
    canonical explicit null is omitted too (ruling R4f, b9uet, supersedes d42m5 R16)."""
    client, _mgr = api
    r = await client.post("/api/strategies", json={"name": "t_null", "dsl_config": _arsi_ui(None)})
    assert r.status_code == 201, r.text
    out = translate_dsl_config(_arsi_ui(None), strategy_name="t_null")
    assert "alpha" not in out["indicators"]["adaptive_rsi_14"]
    assert "adaptive_rsi_14_alpha: float = 0.7" in StrategyCompiler().compile(
        validate_strategy_dict(out)
    )
    assert "arsi_alpha: float = 0.7" in StrategyCompiler().compile(
        validate_strategy_dict(_arsi_canonical(None))
    )


# ---------------------------------------------------------------------------
# Plugin int params must fit a float (vibe-quant-d42m5 SF2): 10**400 is a 422, never a 500
# ---------------------------------------------------------------------------

_FLOAT_MAX_INT = int(sys.float_info.max)  # the largest int a float holds: accepted


def _ichimoku_ui(tenkan: object) -> dict[str, Any]:
    return _cfg([{"type": "ICHIMOKU", "params": {"tenkan": tenkan, "kijun": 26, "senkou": 52}}])


def _ichimoku_canonical(tenkan: object) -> dict[str, Any]:
    return {
        "name": "t_int_range",
        "timeframe": "1h",
        "indicators": {"ichi": {"type": "ICHIMOKU", "tenkan": tenkan, "kijun": 26, "senkou": 52}},
        "entry_conditions": {"long": ["close > ichi"]},
        "exit_conditions": {"long": ["close < ichi"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }


@pytest.mark.parametrize("tenkan", [_HUGE, 2**1024], ids=["10**400", "2**1024"])
def test_plugin_int_param_beyond_float_range_is_rejected(tenkan: int) -> None:
    with pytest.raises(
        ValueError, match=r"^ICHIMOKU param 'tenkan' must be a whole number within range$"
    ):
        translate_dsl_config(_ichimoku_ui(tenkan), strategy_name="t_int_range")
    with pytest.raises(
        ValueError,
        match=r"^Indicator 'ICHIMOKU' param 'tenkan' must be a whole number within range$",
    ):
        check_strategy_indicator_params(_ichimoku_canonical(tenkan))
    with pytest.raises(DSLValidationError) as excinfo:
        validate_strategy_dict(_ichimoku_canonical(tenkan))
    assert "param 'tenkan' must be a whole number within range" in " ".join(excinfo.value.details)


def test_plugin_int_param_at_float_range_edge_is_accepted() -> None:
    out = translate_dsl_config(_ichimoku_ui(_FLOAT_MAX_INT), strategy_name="t_int_edge")
    assert out["indicators"]["ichimoku"]["tenkan"] == _FLOAT_MAX_INT
    canonical = validate_strategy_dict(_ichimoku_canonical(_FLOAT_MAX_INT))
    assert canonical.indicators["ichi"].model_extra == {
        "tenkan": _FLOAT_MAX_INT,
        "kijun": 26,
        "senkou": 52,
    }


@pytest.mark.parametrize("fmt", ["editor", "canonical"])
async def test_saving_an_overflowing_plugin_int_is_422(
    api: tuple[AsyncClient, StateManager], fmt: str
) -> None:
    client, _mgr = api
    dsl = _ichimoku_ui(_HUGE) if fmt == "editor" else _ichimoku_canonical(_HUGE)
    r = await client.post(
        "/api/strategies", **_json_body({"name": f"t_int_{fmt}", "dsl_config": dsl})
    )
    assert r.status_code == 422, r.text
    assert "ICHIMOKU" in r.text and "param 'tenkan' must be a whole number within range" in r.text
