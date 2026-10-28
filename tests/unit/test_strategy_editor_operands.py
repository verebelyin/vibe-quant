"""Every operand the editor offers for every catalog indicator must translate and compile.

The Conditions tab builds operand strings with ``buildOperandOptions`` (frontend
``editor/types.ts``) from each configured indicator: ``TYPE(v1,v2,...)`` over the param values in
order. The vitest ``operandOptions.test.ts`` pins that format in
``__fixtures__/operand_options.json``; ``_editor_operand`` below mirrors it and is checked against
that snapshot, then drives the whole live catalog (the same ``get_catalog`` the Indicators tab
loads) through the translator, the DSL validator and the compiler.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vibe_quant.api.routers.indicators import get_catalog
from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.parser import validate_strategy_dict
from vibe_quant.dsl.translator import translate_dsl_config

_SNAPSHOT: list[dict[str, Any]] = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "frontend/src/components/strategies/editor/__fixtures__/operand_options.json"
    ).read_text()
)
_CATALOG = get_catalog().indicators


def _js_str(value: Any) -> str:
    """JavaScript ``String(value)`` for the JSON scalars a param can hold."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _editor_operand(indicator: dict[str, Any]) -> str:
    """Mirror of frontend buildOperandOptions for one indicator."""
    values = [_js_str(v) for v in indicator["params"].values()]
    return f"{indicator['type']}({','.join(values)})" if values else str(indicator["type"])


def _config(indicator: dict[str, Any], operand: str) -> dict[str, Any]:
    return {
        "general": {"strategy_type": "momentum", "symbols": ["BTCUSDT"], "timeframe": "1h"},
        "indicators": [indicator],
        "conditions": {"entry": [{"left": "price", "operator": ">", "right": operand}]},
        "risk": {
            "stop_loss": {"type": "fixed_pct", "value": 2},
            "take_profit": {"type": "fixed_pct", "value": 4},
        },
        "time": {},
    }


def _assert_resolves_and_compiles(indicator: dict[str, Any], operand: str) -> None:
    out = translate_dsl_config(_config(indicator, operand), strategy_name="t_editor_operand")
    assert len(out["indicators"]) == 1
    (name, cfg), = out["indicators"].items()
    # The operand became a reference to the one configured indicator (not left as raw text)
    assert out["entry_conditions"]["long"] == [f"close > {name}"], operand
    expected_type = "BBANDS" if indicator["type"] == "BB" else indicator["type"]
    assert cfg["type"] == expected_type
    StrategyCompiler().compile_to_module(validate_strategy_dict(out))


def test_python_mirror_matches_the_frontend_operand_snapshot() -> None:
    assert len(_SNAPSHOT) >= 8
    for item in _SNAPSHOT:
        # numeric args compare as numbers: JavaScript prints 0.000001 / 1e-7, Python 1e-06 / 1e-07
        assert _operand_parts(_editor_operand(item["indicator"])) == _operand_parts(item["operand"])
        # and the operand the frozen fixture prints is one the translator still resolves
        _assert_resolves_and_compiles(item["indicator"], item["operand"])


def _operand_parts(operand: str) -> tuple[str, list[float | str]]:
    head, _, rest = operand.partition("(")
    args: list[float | str] = []
    for arg in rest.rstrip(")").split(",") if rest else []:
        try:
            args.append(float(arg))
        except ValueError:
            args.append(arg)
    return head, args


@pytest.mark.parametrize("entry", _CATALOG, ids=[e.type_name for e in _CATALOG])
def test_every_catalog_indicator_operand_translates_validates_and_compiles(entry: Any) -> None:
    """The one per-spec editor walk: catalog params -> editor operand -> translator -> import."""
    indicator = {"type": entry.type_name, "params": dict(entry.default_params)}
    _assert_resolves_and_compiles(indicator, _editor_operand(indicator))


def test_catalog_is_not_trivially_small() -> None:
    # Guards the parametrization above against silently iterating nothing
    assert len(_CATALOG) >= 30
    assert {"BBANDS", "MACD", "STOCH", "RSI"} <= {e.type_name for e in _CATALOG}
