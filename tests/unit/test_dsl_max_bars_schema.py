"""T1 (bead vibe-quant-z1b4e): StrategyDSL.max_bars_in_trade and the editor
``risk.max_bars_in_trade`` translator mapping (design R11/R11b + section 7)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from vibe_quant.dsl.parser import strategy_to_dict
from vibe_quant.dsl.schema import StrategyDSL
from vibe_quant.dsl.translator import translate_dsl_config


def _base_dsl(**overrides: Any) -> dict[str, Any]:
    dsl: dict[str, Any] = {
        "name": "test_strategy",
        "timeframe": "5m",
        "indicators": {"rsi": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi > 50"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }
    dsl.update(overrides)
    return dsl


def _editor_cfg(**risk_overrides: Any) -> dict[str, Any]:
    risk: dict[str, Any] = {
        "stop_loss": {"type": "fixed_pct", "value": 2},
        "take_profit": {"type": "fixed_pct", "value": 4},
        "position_sizing": {"type": "percent_equity", "value": 10},
    }
    risk.update(risk_overrides)
    return {
        "general": {"strategy_type": "momentum", "symbols": ["BTCUSDT"], "timeframe": "1h"},
        "indicators": [{"type": "RSI", "params": {"period": 14}}],
        "conditions": {
            "entry": [{"left": "RSI(14)", "operator": "<", "right": "30"}],
            "exit": [{"left": "RSI(14)", "operator": ">", "right": "70", "logic": None}],
        },
        "risk": risk,
        "time": {},
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (12, 12),
        (12.0, 12),
        ("12", 12),
        ("12.0", 12),
        (100000, 100000),
    ],
)
def test_input_classes_ok(value: Any, expected: int) -> None:
    strategy = StrategyDSL(**_base_dsl(max_bars_in_trade=value))
    assert strategy.max_bars_in_trade == expected


@pytest.mark.parametrize(
    ("value", "err_type"),
    [
        # Error TYPE pinned per row: before the field existed every row failed on
        # extra="forbid" with a different type, so loc alone proved nothing.
        pytest.param(True, "value_error", id="true"),
        pytest.param(False, "value_error", id="false"),
        pytest.param(0, "greater_than_equal", id="zero"),
        pytest.param(-1, "greater_than_equal", id="negative"),
        pytest.param(100001, "less_than_equal", id="above_max"),
        pytest.param(10**30, "less_than_equal", id="huge_int"),
        pytest.param(10**400, "less_than_equal", id="huge_int_400"),
        pytest.param("9" * 500, "less_than_equal", id="numeric_string_overflow"),
        pytest.param(12.5, "int_from_float", id="fraction"),
        pytest.param(float("inf"), "finite_number", id="inf"),
        pytest.param(float("nan"), "finite_number", id="nan"),
        pytest.param("abc", "int_parsing", id="non_numeric_string"),
        pytest.param([], "int_type", id="list"),
    ],
)
def test_input_classes_invalid(value: Any, err_type: str) -> None:
    with pytest.raises(ValidationError) as exc:
        StrategyDSL(**_base_dsl(max_bars_in_trade=value))
    errs = [e for e in exc.value.errors() if "max_bars_in_trade" in e["loc"]]
    assert [e["type"] for e in errs] == [err_type]


def test_absent_is_none_and_omitted() -> None:
    absent = StrategyDSL(**_base_dsl())
    assert absent.max_bars_in_trade is None
    assert "max_bars_in_trade" not in strategy_to_dict(absent)

    explicit_null = StrategyDSL(**_base_dsl(max_bars_in_trade=None))
    assert explicit_null.max_bars_in_trade is None
    assert "max_bars_in_trade" not in strategy_to_dict(explicit_null)

    present = StrategyDSL(**_base_dsl(max_bars_in_trade=12))
    assert strategy_to_dict(present)["max_bars_in_trade"] == 12


def test_translator_maps_risk_max_bars() -> None:
    out = translate_dsl_config(_editor_cfg(max_bars_in_trade=12), strategy_name="t_nbar")
    assert out["max_bars_in_trade"] == 12
    assert StrategyDSL(**out).max_bars_in_trade == 12

    numeric_string = translate_dsl_config(
        _editor_cfg(max_bars_in_trade="12"), strategy_name="t_nbar_str"
    )
    assert numeric_string["max_bars_in_trade"] == "12"
    assert StrategyDSL(**numeric_string).max_bars_in_trade == 12

    absent = translate_dsl_config(_editor_cfg(), strategy_name="t_nbar_absent")
    assert "max_bars_in_trade" not in absent

    null = translate_dsl_config(_editor_cfg(max_bars_in_trade=None), strategy_name="t_nbar_null")
    assert "max_bars_in_trade" not in null


def test_translator_passes_bool_raw_and_schema_rejects() -> None:
    out = translate_dsl_config(_editor_cfg(max_bars_in_trade=False), strategy_name="t_nbar_bool")
    assert out["max_bars_in_trade"] is False
    with pytest.raises(ValidationError) as exc:
        StrategyDSL(**out)
    assert any("max_bars_in_trade" in err["loc"] for err in exc.value.errors())


def test_canonical_input_passes_through() -> None:
    canonical = _base_dsl(max_bars_in_trade=12)
    out = translate_dsl_config(canonical, strategy_name="t_nbar_canon")
    assert out["max_bars_in_trade"] == 12
    assert StrategyDSL(**out).max_bars_in_trade == 12

    canonical_null = _base_dsl(max_bars_in_trade=None)
    out_null = translate_dsl_config(canonical_null, strategy_name="t_nbar_canon_null")
    assert StrategyDSL(**out_null).max_bars_in_trade is None
