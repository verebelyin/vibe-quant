"""vibe-quant-e70tl.23: frontend DslConfig -> StrategyDSL translator must not
silently drop exit conditions, indicator timeframe overrides, OR logic or a
percent trailing stop."""

from __future__ import annotations

from typing import Any

import pytest

from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.translator import translate_dsl_config


def _cfg(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "general": {"strategy_type": "momentum", "symbols": ["BTCUSDT"], "timeframe": "1h"},
        "indicators": [{"type": "RSI", "params": {"period": 14}}],
        "conditions": {
            "entry": [{"left": "RSI(14)", "operator": "<", "right": "30"}],
            "exit": [{"left": "RSI(14)", "operator": ">", "right": "70", "logic": None}],
        },
        "risk": {
            "stop_loss": {"type": "fixed_pct", "value": 2},
            "take_profit": {"type": "fixed_pct", "value": 4},
            "position_sizing": {"type": "percent_equity", "value": 10},
        },
        "time": {},
    }
    cfg.update(overrides)
    return cfg


def test_generic_exit_kept() -> None:
    out = translate_dsl_config(_cfg(), strategy_name="t_exit")
    assert out["entry_conditions"]["long"] == ["rsi_14 < 30"]
    assert out["exit_conditions"] == {"long": ["rsi_14 > 70"], "short": []}
    validate_strategy_dict(out)


def test_directional_exits_kept() -> None:
    conds = {
        "long_entry": [{"left": "RSI(14)", "operator": "<", "right": "30"}],
        "short_entry": [{"left": "RSI(14)", "operator": ">", "right": "70"}],
        "long_exit": [{"left": "RSI(14)", "operator": ">", "right": "55"}],
        "short_exit": [{"left": "RSI(14)", "operator": "<", "right": "45"}],
    }
    out = translate_dsl_config(_cfg(conditions=conds), strategy_name="t_dir")
    assert out["exit_conditions"] == {"long": ["rsi_14 > 55"], "short": ["rsi_14 < 45"]}
    validate_strategy_dict(out)


def test_and_logic_kept_and_or_logic_rejected() -> None:
    """Too many conditions are joined with ``or``: `and` survives, `or` must not be AND-ed."""
    conds = {
        "entry": [
            {"left": "RSI(14)", "operator": "<", "right": "30"},
            {"left": "RSI(14)", "operator": ">", "right": "10", "logic": "and"},
        ]
    }
    out = translate_dsl_config(_cfg(conditions=conds), strategy_name="t_and")
    assert out["entry_conditions"]["long"] == ["rsi_14 < 30", "rsi_14 > 10"]

    conds = {
        "entry": [
            {"left": "RSI(14)", "operator": "<", "right": "30"},
            {"left": "RSI(14)", "operator": ">", "right": "80", "logic": "or"},
        ]
    }
    with pytest.raises(ValueError, match="'or' logic"):
        translate_dsl_config(_cfg(conditions=conds), strategy_name="t_or")


def test_timeframe_override_kept_and_declared() -> None:
    inds = [
        {"type": "RSI", "params": {"period": 14}},
        {"type": "ADX", "params": {"period": 14}, "timeframe_override": "4h"},
    ]
    out = translate_dsl_config(_cfg(indicators=inds), strategy_name="t_tf")
    assert out["indicators"]["adx_14"]["timeframe"] == "4h"
    assert out["additional_timeframes"] == ["4h"]
    dsl = validate_strategy_dict(out)
    assert dsl.indicators["adx_14"].timeframe == "4h"


def test_trailing_stop_pct_rejected() -> None:
    risk = _cfg()["risk"] | {"trailing_stop_pct": 1.5}
    with pytest.raises(ValueError, match="trailing_stop_pct"):
        translate_dsl_config(_cfg(risk=risk), strategy_name="t_trail")


def test_stoch_k_d_periods_mapped() -> None:
    inds = [{"type": "STOCH", "params": {"k_period": 9, "d_period": 5}}]
    conds = {"entry": [{"left": "STOCH(9,5)", "operator": "<", "right": "20"}]}
    out = translate_dsl_config(_cfg(indicators=inds, conditions=conds), strategy_name="t_stoch")
    cfg = next(iter(out["indicators"].values()))
    assert (cfg["period"], cfg["d_period"]) == (9, 5)


# --- ATR stop / take-profit need an `indicator` reference (vibe-quant-yj57u) ---------------


@pytest.mark.parametrize("sl_type", ["atr_fixed", "atr_trailing"])
def test_atr_stop_reuses_ui_atr_indicator(sl_type: str) -> None:
    inds = [
        {"type": "RSI", "params": {"period": 14}},
        {"type": "ATR", "params": {"period": 20}},
    ]
    risk = _cfg()["risk"] | {"stop_loss": {"type": sl_type, "value": 2.5}}
    out = translate_dsl_config(_cfg(indicators=inds, risk=risk), strategy_name="t_atr_reuse")
    assert out["stop_loss"] == {"type": sl_type, "atr_multiplier": 2.5, "indicator": "atr_20"}
    assert list(out["indicators"]) == ["rsi_14", "atr_20"]  # nothing added
    dsl = validate_strategy_dict(out)
    assert dsl.stop_loss.indicator == "atr_20"
    StrategyCompiler().compile_to_module(dsl)


@pytest.mark.parametrize("sl_type", ["atr_fixed", "atr_trailing"])
def test_atr_stop_without_ui_atr_adds_default_indicator(sl_type: str) -> None:
    risk = _cfg()["risk"] | {"stop_loss": {"type": sl_type, "value": 1.5}}
    out = translate_dsl_config(_cfg(risk=risk), strategy_name="t_atr_add")
    assert out["indicators"]["atr_14"] == {"type": "ATR", "period": 14}
    assert out["stop_loss"] == {"type": sl_type, "atr_multiplier": 1.5, "indicator": "atr_14"}
    dsl = validate_strategy_dict(out)
    assert dsl.indicators["atr_14"].type == "ATR"
    StrategyCompiler().compile_to_module(dsl)


def test_atr_take_profit_gets_indicator_and_shares_it_with_stop() -> None:
    risk = _cfg()["risk"] | {
        "stop_loss": {"type": "atr_trailing", "value": 2},
        "take_profit": {"type": "atr_fixed", "value": 3},
    }
    out = translate_dsl_config(_cfg(risk=risk), strategy_name="t_atr_tp")
    assert [n for n, c in out["indicators"].items() if c["type"] == "ATR"] == ["atr_14"]
    assert out["take_profit"]["indicator"] == "atr_14"
    StrategyCompiler().compile_to_module(validate_strategy_dict(out))


def test_fixed_pct_stop_adds_no_atr_indicator() -> None:
    out = translate_dsl_config(_cfg(), strategy_name="t_nofix")
    assert list(out["indicators"]) == ["rsi_14"]
    assert "indicator" not in out["stop_loss"]


def test_two_atr_indicators_stop_and_take_profit_bind_to_the_first() -> None:
    # ATR(20) on 1d comes first, ATR(7) second: SL and TP must both use atr_20, not the last ATR
    inds = [
        {"type": "RSI", "params": {"period": 14}},
        {"type": "ATR", "params": {"period": 20}, "timeframe_override": "1d"},
        {"type": "ATR", "params": {"period": 7}},
    ]
    risk = _cfg()["risk"] | {
        "stop_loss": {"type": "atr_trailing", "value": 2},
        "take_profit": {"type": "atr_fixed", "value": 3},
    }
    out = translate_dsl_config(
        _cfg(indicators=inds, risk=risk, general={"timeframe": "4h", "symbols": ["BTCUSDT"]}),
        strategy_name="t_two_atr",
    )
    assert out["stop_loss"]["indicator"] == "atr_20"
    assert out["take_profit"]["indicator"] == "atr_20"
    assert out["indicators"]["atr_20"] == {"type": "ATR", "period": 20, "timeframe": "1d"}
    assert out["indicators"]["atr_7"] == {"type": "ATR", "period": 7}
    assert list(out["indicators"]) == ["rsi_14", "atr_20", "atr_7"]  # nothing added
    assert out["additional_timeframes"] == ["1d"]
    dsl = validate_strategy_dict(out)
    assert dsl.stop_loss.indicator == dsl.take_profit.indicator == "atr_20"


# Editor indicator spellings that reach a translator branch the catalog rows cannot: the BB->BBANDS
# alias rename, MACD's legacy fast/slow/signal fold and STOCH's k_period fold. (A row with the
# registry spelling walks the same branch as the catalog table in test_strategy_editor_operands.)
_EDITOR_ALIAS_INDICATORS: list[tuple[str, dict[str, Any]]] = [
    ("BB", {"period": 20, "std_dev": 2}),
    ("MACD", {"fast": 12, "slow": 26, "signal": 9}),
    ("STOCH", {"k_period": 14, "d_period": 3}),
]


@pytest.mark.parametrize(
    ("ind_type", "params"),
    _EDITOR_ALIAS_INDICATORS,
    ids=[f"{t}-{'-'.join(p) or 'noparams'}" for t, p in _EDITOR_ALIAS_INDICATORS],
)
def test_every_editor_alias_indicator_translates_validates_and_compiles(
    ind_type: str, params: dict[str, Any]
) -> None:
    cfg = _cfg(
        indicators=[{"type": ind_type, "params": params}],
        conditions={"entry": [{"left": "price", "operator": ">", "right": ind_type}]},
    )
    out = translate_dsl_config(cfg, strategy_name="t_editor_types")
    dsl = validate_strategy_dict(out)
    StrategyCompiler().compile_to_module(dsl)


def test_bb_maps_to_bbands_keeping_name_and_refs() -> None:
    # The UI sweep builder mirrors the UI-format name `<raw type lowercased>_<period>`,
    # so only the `type` changes, never the generated name.
    cfg = _cfg(
        indicators=[{"type": "BB", "params": {"period": 20, "std_dev": 2.5}}],
        conditions={
            "entry": [
                {"left": "price", "operator": "<", "right": "BB(20)"},
                {"left": "price", "operator": ">", "right": "BBANDS(20)"},
            ]
        },
    )
    out = translate_dsl_config(cfg, strategy_name="t_bb")
    assert out["indicators"] == {"bb_20": {"type": "BBANDS", "period": 20, "std_dev": 2.5}}
    assert out["entry_conditions"]["long"] == ["close < bb_20", "close > bb_20"]
    dsl = validate_strategy_dict(out)
    assert dsl.indicators["bb_20"].type == "BBANDS"


def test_registry_param_spellings_not_dropped() -> None:
    cfg = _cfg(
        indicators=[
            {"type": "MACD", "params": {"fast_period": 8, "slow_period": 30, "signal_period": 5}},
            {"type": "STOCH", "params": {"period_k": 10, "period_d": 4}},
        ],
        conditions={"entry": [{"left": "price", "operator": ">", "right": "MACD"}]},
    )
    ind = translate_dsl_config(cfg, strategy_name="t_params")["indicators"]
    assert ind["macd"] == {
        "type": "MACD", "fast_period": 8, "slow_period": 30, "signal_period": 5,
    }
    assert ind["stoch"] == {"type": "STOCH", "period": 10, "d_period": 4}


def _two_bollingers(first: str, second: str, left: str, right: str) -> dict[str, Any]:
    return _cfg(
        indicators=[
            {"type": first, "params": {"period": 20, "std_dev": 2}, "timeframe_override": "1d"},
            {"type": second, "params": {"period": 20, "std_dev": 2}},
        ],
        conditions={
            "entry": [
                {"left": "price", "operator": ">", "right": left},
                {"left": "price", "operator": "<", "right": right},
            ]
        },
    )


@pytest.mark.parametrize("native_first", [True, False])
def test_native_bbands_ref_wins_over_bb_alias(native_first: bool) -> None:
    # BBANDS(20)@1d and BB(20)@1h: each ref must pick the indicator it names, whatever the order
    first, second = ("BBANDS", "BB") if native_first else ("BB", "BBANDS")
    out = translate_dsl_config(
        _two_bollingers(first, second, "BBANDS(20)", "BB(20)"), strategy_name="t_bb_clash"
    )
    assert set(out["indicators"]) == {"bbands_20", "bb_20"}
    assert out["entry_conditions"]["long"] == ["close > bbands_20", "close < bb_20"]
    assert out["indicators"]["bbands_20"]["type"] == out["indicators"]["bb_20"]["type"] == "BBANDS"
    # the native one is the daily indicator iff it was configured first (timeframe tags the pair)
    tf = {n: c.get("timeframe") for n, c in out["indicators"].items()}
    assert tf == ({"bbands_20": "1d", "bb_20": None} if native_first else {"bb_20": "1d", "bbands_20": None})


def test_bb_alias_still_answers_to_bbands_when_no_native_bbands() -> None:
    cfg = _cfg(
        indicators=[{"type": "BB", "params": {"period": 20, "std_dev": 2}}],
        conditions={"entry": [{"left": "price", "operator": ">", "right": "BBANDS(20,2)"}]},
    )
    assert translate_dsl_config(cfg, strategy_name="t")["entry_conditions"]["long"] == [
        "close > bb_20"
    ]


@pytest.mark.parametrize(
    ("ind_type", "params", "operand", "expected"),
    [
        ("BB", {"period": 20, "std_dev": 2}, "BB(20,2)", "bb_20"),
        ("BBANDS", {"period": 20, "std_dev": 2.0}, "BBANDS(20,2)", "bbands_20"),
        ("MACD", {"fast_period": 8, "slow_period": 30, "signal_period": 5}, "MACD(8,30,5)", "macd"),
        ("STOCH", {"period_k": 10, "period_d": 4}, "STOCH(10,4)", "stoch"),
        ("STOCH", {"period_k": 10, "period_d": 4}, "STOCH(10, 4)", "stoch"),
    ],
)
def test_editor_operands_with_all_params_resolve(
    ind_type: str, params: dict[str, Any], operand: str, expected: str
) -> None:
    cfg = _cfg(
        indicators=[{"type": ind_type, "params": params}],
        conditions={"entry": [{"left": "price", "operator": ">", "right": operand}]},
    )
    out = translate_dsl_config(cfg, strategy_name="t_operand")
    assert out["entry_conditions"]["long"] == [f"close > {expected}"]
    validate_strategy_dict(out)


@pytest.mark.parametrize(
    ("ind_type", "params"),
    [
        ("STOCH", {"k_period": 10, "d_period": 4, "period_k": 14, "period_d": 3}),
        ("STOCH", {"period": 10, "k_period": 14}),
        ("STOCH", {"d_period": 4, "period_d": 3}),
        ("MACD", {"fast": 12, "fast_period": 8, "slow": 26, "signal": 9}),
        ("MACD", {"fast": 12, "slow": 26, "slow_period": 30}),
        ("MACD", {"fast": 12, "slow": 26, "signal": 9, "signal_period": 5}),
    ],
)
def test_contradictory_param_synonyms_rejected(ind_type: str, params: dict[str, Any]) -> None:
    cfg = _cfg(indicators=[{"type": ind_type, "params": params}])
    with pytest.raises(ValueError, match="Conflicting"):
        translate_dsl_config(cfg, strategy_name="t_conflict")


def test_agreeing_param_synonyms_accepted() -> None:
    cfg = _cfg(
        indicators=[
            {"type": "MACD", "params": {"fast": 8, "fast_period": 8, "slow": 30, "signal": 5}},
            {"type": "STOCH", "params": {"k_period": 10, "period_k": 10, "d_period": 4}},
        ]
    )
    ind = translate_dsl_config(cfg, strategy_name="t_agree")["indicators"]
    assert ind["macd"] == {"type": "MACD", "fast_period": 8, "slow_period": 30, "signal_period": 5}
    assert ind["stoch"] == {"type": "STOCH", "period": 10, "d_period": 4}


def _ambig_cfg(indicators: list[dict[str, Any]], operand: str) -> dict[str, Any]:
    return _cfg(
        indicators=indicators,
        conditions={"entry": [{"left": "price", "operator": ">", "right": operand}]},
    )


def test_same_operand_from_two_bollingers_is_rejected_not_last_wins() -> None:
    # BBANDS(20,2)@1d and @1h print the identical editor operand: picking one is a silent guess
    cfg = _two_bollingers("BBANDS", "BBANDS", "bbands_20", "BBANDS(20,2)")
    with pytest.raises(ValueError, match=r"BBANDS\(20,2\).*bbands_20.*bbands_20_2"):
        translate_dsl_config(cfg, strategy_name="t_ambig")
    # the canonical names stay usable
    cfg = _two_bollingers("BBANDS", "BBANDS", "bbands_20", "bbands_20_2")
    out = translate_dsl_config(cfg, strategy_name="t_ambig")
    assert out["entry_conditions"]["long"] == ["close > bbands_20", "close < bbands_20_2"]


def test_macd_configs_printing_the_same_operand_are_rejected() -> None:
    # {fast:8,slow:30,signal:5} and {signal:8,slow:30,fast:5} both emit MACD(8,30,5)
    cfg = _ambig_cfg(
        [
            {"type": "MACD", "params": {"fast_period": 8, "slow_period": 30, "signal_period": 5}},
            {"type": "MACD", "params": {"signal_period": 8, "slow_period": 30, "fast_period": 5}},
        ],
        "MACD(8,30,5)",
    )
    with pytest.raises(ValueError, match=r"MACD\(8,30,5\).*macd.*macd_2"):
        translate_dsl_config(cfg, strategy_name="t_ambig_macd")


def test_unambiguous_refs_survive_next_to_an_ambiguous_one() -> None:
    cfg = _cfg(
        indicators=[
            {"type": "BBANDS", "params": {"period": 20, "std_dev": 2}, "timeframe_override": "1d"},
            {"type": "BBANDS", "params": {"period": 20, "std_dev": 2}},
            {"type": "BBANDS", "params": {"period": 20, "std_dev": 3}},
        ],
        conditions={"entry": [{"left": "price", "operator": ">", "right": "BBANDS(20,3)"}]},
    )
    out = translate_dsl_config(cfg, strategy_name="t_mixed")
    assert out["entry_conditions"]["long"] == ["close > bbands_20_3"]


@pytest.mark.parametrize(
    ("alpha", "operand"),
    [
        (0.000001, "ADAPTIVE_RSI(14,0.000001)"),  # JavaScript String(1e-6)
        (0.0000001, "ADAPTIVE_RSI(14,1e-7)"),  # JavaScript String(1e-7) vs Python 1e-07
        (0.5, "ADAPTIVE_RSI(14, 0.50)"),
    ],
)
def test_numeric_operand_args_compare_numerically(alpha: float, operand: str) -> None:
    cfg = _ambig_cfg([{"type": "ADAPTIVE_RSI", "params": {"period": 14, "alpha": alpha}}], operand)
    out = translate_dsl_config(cfg, strategy_name="t_sci")
    assert out["entry_conditions"]["long"] == ["close > adaptive_rsi_14"]
    validate_strategy_dict(out)


def test_numeric_operand_args_do_not_merge_different_values() -> None:
    cfg = _ambig_cfg([{"type": "ADAPTIVE_RSI", "params": {"period": 14, "alpha": 0.000001}}],
                     "ADAPTIVE_RSI(14,0.00001)")
    out = translate_dsl_config(cfg, strategy_name="t_sci2")
    # different value: stays unresolved text, never silently mapped to the configured indicator
    assert out["entry_conditions"]["long"] == ["close > ADAPTIVE_RSI(14,0.00001)"]
