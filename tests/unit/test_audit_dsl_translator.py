"""vibe-quant-e70tl.23: frontend DslConfig -> StrategyDSL translator must not
silently drop exit conditions, indicator timeframe overrides, OR logic or a
percent trailing stop."""

from __future__ import annotations

from typing import Any

import pytest

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


def test_or_logic_rejected() -> None:
    conds = {
        "entry": [
            {"left": "RSI(14)", "operator": "<", "right": "30"},
            {"left": "RSI(14)", "operator": ">", "right": "80", "logic": "or"},
        ]
    }
    with pytest.raises(ValueError, match="'or' logic"):
        translate_dsl_config(_cfg(conditions=conds), strategy_name="t_or")


def test_and_logic_accepted() -> None:
    conds = {
        "entry": [
            {"left": "RSI(14)", "operator": "<", "right": "30"},
            {"left": "RSI(14)", "operator": ">", "right": "10", "logic": "and"},
        ]
    }
    out = translate_dsl_config(_cfg(conditions=conds), strategy_name="t_and")
    assert out["entry_conditions"]["long"] == ["rsi_14 < 30", "rsi_14 > 10"]


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
