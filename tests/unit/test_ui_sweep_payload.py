"""Sweep payloads built by the backtest launch form are accepted by the screening runner.

The JSON fixtures are written by frontend/src/components/backtest/SweepBuilder.test.ts
(sweepToPayload on every bundled template and on a UI-format strategy). Screening start does
``StrategyDSL.model_validate(dsl | {"sweep": payload})`` and ``NTScreeningRunner`` resolves every
sweep key onto the compiled strategy config; either rejecting a key fails the whole run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import msgspec
import pytest

from vibe_quant.api.routers.indicators import _spec_to_entry
from vibe_quant.dsl.indicators import indicator_registry
from vibe_quant.dsl.schema import StrategyDSL
from vibe_quant.dsl.translator import translate_dsl_config
from vibe_quant.screening.grid import build_parameter_grid
from vibe_quant.screening.nt_runner import (
    NTScreeningRunner,
    UnknownStrategyParamError,
    resolve_strategy_params,
)
from vibe_quant.strategies.templates._metadata import list_template_payloads

_FIXTURES = Path(__file__).parents[2] / "frontend/src/components/backtest/__fixtures__"


def _fixture(name: str) -> Any:
    return json.loads((_FIXTURES / name).read_text())


def _runner(dsl_dict: dict[str, Any]) -> NTScreeningRunner:
    return NTScreeningRunner(
        dsl_dict, symbols=["BTCUSDT"], start_date="2024-01-01", end_date="2024-01-05"
    )


def _decode_every_combo(runner: NTScreeningRunner, sweep: dict[str, Any]) -> int:
    """Decode the compiled StrategyConfig for every grid point, as NT's StrategyFactory does.

    Key resolution alone is not enough: an int-typed field given 12.5 fails every backtest
    with ``Expected int, got float`` (msgspec), long after the run was accepted.
    """
    config_cls = getattr(sys.modules[runner._module_path], runner._config_cls_name)
    grid = build_parameter_grid(sweep)
    for combo in grid:
        params = resolve_strategy_params(combo, runner._config_fields())
        config_cls.parse(msgspec.json.encode({"instrument_id": "BTCUSDT-PERP.BINANCE", **params}))
    return len(grid)


def _launch_like_screening_main(raw: dict[str, Any], name: str, sweep: dict[str, Any]) -> None:
    """Mirror vibe_quant/screening/__main__.py: translate, validate, override sweep, runner."""
    dsl = StrategyDSL.model_validate(translate_dsl_config(raw, strategy_name=name))
    dsl_dict = dsl.model_dump()
    dsl_dict["sweep"] = sweep
    dsl = StrategyDSL.model_validate(dsl_dict)  # sweep must be dict[str, list[number]]
    runner = _runner(dsl.model_dump())  # validate_param_keys -> resolve_strategy_params
    assert _decode_every_combo(runner, dsl.sweep) == 2 ** len(sweep)


_TEMPLATES: dict[str, dict[str, Any]] = {
    str(p["name"]): p["dsl_config"]
    for p in list_template_payloads()  # type: ignore[misc]
}


def test_template_payload_fixture_covers_every_template() -> None:
    assert sorted(_fixture("template_sweep_payloads.json")) == sorted(_TEMPLATES)


@pytest.mark.parametrize("name", sorted(_TEMPLATES))
def test_template_ui_sweep_resolves_on_compiled_config(name: str) -> None:
    sweep = _fixture("template_sweep_payloads.json")[name]
    assert sweep, "template has no sweepable params"
    _launch_like_screening_main(_TEMPLATES[name], name, sweep)


def test_ui_format_strategy_ui_sweep_resolves_on_compiled_config() -> None:
    sweep = _fixture("legacy_sweep_payload.json")
    _launch_like_screening_main(_fixture("legacy_ui_strategy.json"), "legacy_ui", sweep)


def test_catalog_default_params_ui_sweep_decodes_on_compiled_config() -> None:
    """Editor-shaped strategy built from the live catalog defaults (registry spellings, plugin
    alpha, ICHIMOKU, SQUEEZE_MOM length)."""
    sweep = _fixture("catalog_sweep_payload.json")
    assert {"macd_fast_period", "stoch_period", "ichimoku_tenkan", "adaptive_rsi_14_alpha"} <= set(
        sweep
    )
    _launch_like_screening_main(_fixture("catalog_ui_strategy.json"), "catalog_ui", sweep)


def test_stored_fractional_legacy_int_param_fails_translation() -> None:
    """A stored legacy ``period: 12.5`` used to translate to EMA(12) -- a different strategy than
    the user configured; it now fails at launch instead of silently backtesting the wrong period."""
    raw = _fixture("catalog_ui_strategy.json")
    raw["indicators"][0]["params"]["period"] = 12.5
    with pytest.raises(ValueError, match=r"EMA.*'period'"):
        translate_dsl_config(raw, strategy_name="catalog_frac")


def test_folded_synonym_spellings_ui_sweep_decodes_on_compiled_config() -> None:
    """Equal spellings (fast/fast_period, STOCH period/k_period/period_k) fold to one field."""
    sweep = _fixture("folded_sweep_payload.json")
    assert {"stoch_14_period", "stoch_14_d_period", "macd_fast_period"} <= set(sweep)
    _launch_like_screening_main(_fixture("folded_ui_strategy.json"), "folded_ui", sweep)


@pytest.mark.parametrize(
    ("ind_type", "params", "field"),
    [
        ("STOCH", {"period": 10, "k_period": 14, "d_period": 3}, "period"),
        ("MACD", {"fast": 12, "fast_period": 9, "slow": 26}, "fast_period"),
    ],
)
def test_conflicting_spellings_fail_translation(
    ind_type: str, params: dict[str, int], field: str
) -> None:
    """The launch fails in the translator, which is why the form flags the strategy inline."""
    raw = _fixture("folded_ui_strategy.json")
    raw["indicators"] = [{"type": ind_type, "params": params}]
    with pytest.raises(ValueError, match=f"Conflicting {ind_type} params for '{field}'"):
        translate_dsl_config(raw, strategy_name="conflict")


def test_catalog_fixture_matches_registry() -> None:
    """The frontend fixture is the live catalog (regenerate with vibe_quant.api.routers.indicators._spec_to_entry)."""
    live = {
        e.type_name: {"default_params": e.default_params, "param_schema": e.param_schema}
        for e in map(_spec_to_entry, indicator_registry.all_specs())
    }
    assert _fixture("indicator_catalog.json") == live


def test_fractional_value_on_int_field_fails_decode() -> None:
    """Why the form must reject fractional sweeps inline: DSL validation accepts them, the
    compiled config does not (msgspec 'Expected int, got float' on every backtest)."""
    raw = _TEMPLATES["dual_ema_crossover"]
    sweep = {"ema_fast_period": [12, 12.5, 13]}
    dsl = StrategyDSL.model_validate(
        {**StrategyDSL.model_validate(raw).model_dump(), "sweep": sweep}
    )
    assert dsl.sweep["ema_fast_period"] == [12.0, 12.5, 13.0]
    with pytest.raises(msgspec.ValidationError, match="Expected `int`"):
        _decode_every_combo(_runner(dsl.model_dump()), dsl.sweep)


def test_old_builder_shape_is_rejected() -> None:
    """The pre-fix payload ('EMA_0.period': {type,min,max,step}) never validated."""
    raw = _TEMPLATES["dual_ema_crossover"]
    dsl_dict = StrategyDSL.model_validate(raw).model_dump()
    dsl_dict["sweep"] = {"EMA_0.period": {"type": "range", "min": 5, "max": 10, "step": 1}}
    with pytest.raises(ValueError, match="sweep"):
        StrategyDSL.model_validate(dsl_dict)
    dsl_dict["sweep"] = {"EMA_0.period": [5, 10]}
    with pytest.raises(UnknownStrategyParamError, match="EMA_0.period"):
        _runner(dsl_dict)
