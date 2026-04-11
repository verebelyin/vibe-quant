"""End-to-end compile sanity for every registered built-in indicator.

This test is the Phase 3 integration gate: every built-in ``IndicatorSpec``
must still produce a DSL that validates, compiles, and imports cleanly.
It is parameterized over ``indicator_registry.all_specs()`` so new specs
(including plugins dropped into ``vibe_quant/dsl/plugins/`` once Phase 6
lands) automatically get covered.

Scope: compile-and-import, not backtest. Running a real NT backtest per
indicator would need ``data/catalog`` to be populated and would slow the
unit suite from sub-second to minutes — the dedicated backtest smoke is
left to the Phase 4 quality gate, which re-runs the known-good discovery
cycle on 1m data.
"""

from __future__ import annotations

from typing import Any

import pytest

from vibe_quant.dsl.compiler import StrategyCompiler
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.parser import validate_strategy_dict


def _threshold_for(spec: IndicatorSpec) -> float:
    """Pick a sensible numeric threshold for a condition against this indicator.

    Uses the spec's ``threshold_range`` midpoint when available, else 0.
    """
    if spec.threshold_range is not None:
        lo, hi = spec.threshold_range
        return (lo + hi) / 2
    return 0.0


def _sub_value_for(spec: IndicatorSpec) -> str | None:
    """Pick a valid sub-value reference for multi-output indicators."""
    if spec.output_names == ("value",):
        return None
    # Prefer the first output name that isn't the placeholder 'value'.
    for name in spec.output_names:
        if name != "value":
            return name
    return None


# IndicatorConfig accepts a fixed set of fields (see dsl/schema.py). We can't
# round-trip arbitrary spec.default_params keys — STOCH's ``period_k``/
# ``period_d`` and ICHIMOKU's ``tenkan``/``kijun``/``senkou`` are
# spec-internal. This mapping produces the smallest DSL config that still
# satisfies both the schema and the spec's @model_validator checks.
_DSL_CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    "STOCH": {"period": 14, "d_period": 3},
    "ICHIMOKU": {"period": 9},  # tenkan; kijun/senkou come from spec defaults
    "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    "BBANDS": {"period": 20, "std_dev": 2.0},
    "KC": {"period": 20, "atr_multiplier": 2.0},
    # Volume-only indicators need an explicit source override.
    "VOLSMA": {"period": 20, "source": "volume"},
    "OBV": {"source": "volume"},
    "VWAP": {},
}


def _build_indicator_config(spec: IndicatorSpec) -> dict[str, Any]:
    """Build the minimal IndicatorConfig dict for a spec.

    Special-cases multi-param indicators whose DSL field names diverge
    from their spec.default_params keys; everything else gets the
    canonical ``period`` field.
    """
    cfg: dict[str, Any] = {"type": spec.name}
    if spec.name in _DSL_CONFIG_OVERRIDES:
        cfg.update(_DSL_CONFIG_OVERRIDES[spec.name])
    elif "period" in spec.default_params:
        cfg["period"] = spec.default_params["period"]
    return cfg


def _build_minimal_dsl(spec: IndicatorSpec) -> dict[str, Any]:
    """Build a minimal DSL dict exercising this indicator.

    Entry condition: indicator < threshold (or sub-value < threshold for
    multi-output). Exit condition: indicator > threshold. This is not a
    sensible trading strategy — it is only intended to drive every branch
    of the compiler for one indicator.
    """
    ind_name = spec.name.lower()
    threshold = _threshold_for(spec)
    sub = _sub_value_for(spec)
    ref = f"{ind_name}.{sub}" if sub else ind_name

    return {
        "name": f"e2e_{ind_name}",
        "timeframe": "5m",
        "indicators": {ind_name: _build_indicator_config(spec)},
        "entry_conditions": {"long": [f"{ref} < {threshold}"]},
        "exit_conditions": {"long": [f"{ref} > {threshold}"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }


@pytest.mark.parametrize(
    "spec",
    indicator_registry.all_specs(),
    ids=lambda s: s.name,
)
def test_every_spec_compiles_minimal_strategy(spec: IndicatorSpec) -> None:
    """Every registered indicator must compile through the full DSL pipeline."""
    dsl_dict = _build_minimal_dsl(spec)
    dsl = validate_strategy_dict(dsl_dict)

    compiler = StrategyCompiler()
    source = compiler.compile(dsl)

    assert source, f"Empty compiled source for {spec.name}"
    assert "class " in source, f"Compiled source for {spec.name} missing class"
    # Every strategy class is named after the DSL name in CamelCase.
    camel = "".join(word.capitalize() for word in dsl.name.split("_"))
    assert f"class {camel}Strategy" in source, (
        f"Missing strategy class in compiled output for {spec.name}"
    )


@pytest.mark.parametrize(
    "spec",
    indicator_registry.all_specs(),
    ids=lambda s: s.name,
)
def test_every_spec_compiles_to_importable_module(spec: IndicatorSpec) -> None:
    """Every compiled module must import cleanly (valid Python + no name errors)."""
    dsl_dict = _build_minimal_dsl(spec)
    dsl = validate_strategy_dict(dsl_dict)

    compiler = StrategyCompiler()
    module = compiler.compile_to_module(dsl)

    camel = "".join(word.capitalize() for word in dsl.name.split("_"))
    strategy_cls = getattr(module, f"{camel}Strategy", None)
    config_cls = getattr(module, f"{camel}Config", None)
    assert strategy_cls is not None, f"No Strategy class on compiled module for {spec.name}"
    assert config_cls is not None, f"No Config class on compiled module for {spec.name}"


def test_all_specs_fixture_covers_full_registry() -> None:
    """Meta-guard: the parametrization picks up every built-in indicator."""
    specs = indicator_registry.all_specs()
    # 20 built-ins per CLAUDE.md + SPEC.md; Phase 6 may add plugins at runtime.
    assert len(specs) >= 20, f"Expected >= 20 specs, got {len(specs)}"
    # Smoke: a handful of well-known names are present.
    names = {s.name for s in specs}
    for required in ("RSI", "MACD", "BBANDS", "STOCH", "ATR", "ADX", "CCI"):
        assert required in names, f"Missing core indicator: {required}"
