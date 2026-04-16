"""CI regression test for the plugin contract (vibe-quant-gn83).

A single throwaway plugin, registered inside the test, must traverse
every integration point: registry, DSL schema, GA pool, API catalog, and
compiler. This is a structural contract test — distinct from
``test_plugin_end_to_end.py`` which tests the shipped ADAPTIVE_RSI
example.

If any layer silently drops a plugin registered at runtime, this test
fails. That gives us early warning when refactors in any of the four
integration points break the "drop a file, it just works" promise.
"""

from __future__ import annotations

import pytest

from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.plugins.example_adaptive_rsi import compute_adaptive_rsi

# ---------------------------------------------------------------------------
# Synthetic plugin — registered in a fixture, torn down after each test.
# ---------------------------------------------------------------------------

_PLUGIN_NAME = "CITEST_PLUGIN"


@pytest.fixture
def synthetic_plugin():
    """Register a throwaway plugin; always unregister on teardown.

    Uses a distinctive name so the test never collides with real plugins,
    and tears down via the private ``_indicators`` dict because
    ``IndicatorRegistry`` has no public unregister API.

    Reuses ADAPTIVE_RSI's compute_fn rather than declaring a test-local
    function: the compiler import allowlist only permits
    ``vibe_quant.*`` modules in generated source, so a test-module
    compute_fn would fail ``compile_to_module`` validation. What matters
    for the contract is the spec's *param_schema* (includes ``gain``,
    which ADAPTIVE_RSI does not) — not the algorithm.
    """
    spec = IndicatorSpec(
        name=_PLUGIN_NAME,
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 10, "gain": 1.5},
        param_schema={"period": int, "gain": float},
        compute_fn=compute_adaptive_rsi,
        display_name="CI Test Plugin",
        description="Synthetic plugin — do not ship.",
        category="Custom",
        param_ranges={"period": (5.0, 50.0), "gain": (0.5, 3.0)},
        threshold_range=(0.0, 1000.0),
    )
    indicator_registry.register_spec(spec)
    try:
        yield spec
    finally:
        indicator_registry._indicators.pop(_PLUGIN_NAME, None)


# ---------------------------------------------------------------------------
# 1. Registry: the plugin is reachable by name and by all_specs().
# ---------------------------------------------------------------------------


def test_contract_registry(synthetic_plugin) -> None:
    spec = indicator_registry.get(_PLUGIN_NAME)
    assert spec is synthetic_plugin
    assert _PLUGIN_NAME in indicator_registry.list_indicators()
    assert any(s.name == _PLUGIN_NAME for s in indicator_registry.all_specs())


# ---------------------------------------------------------------------------
# 2. DSL schema: plugin-declared params (``gain``) flow through
#    IndicatorConfig via extra='allow' + param_schema validation.
# ---------------------------------------------------------------------------


def test_contract_dsl_schema(synthetic_plugin) -> None:
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.dsl.schema import IndicatorConfig

    cfg = IndicatorConfig(type=_PLUGIN_NAME, period=10, gain=2.0)
    assert cfg.type == _PLUGIN_NAME
    assert (cfg.model_extra or {}).get("gain") == 2.0

    dsl = validate_strategy_dict(
        {
            "name": "citest_strategy",
            "timeframe": "5m",
            "indicators": {
                "ci": {"type": _PLUGIN_NAME, "period": 10, "gain": 2.0}
            },
            "entry_conditions": {"long": ["ci > 100"]},
            "exit_conditions": {"long": ["ci < 90"]},
            "stop_loss": {"type": "fixed_pct", "percent": 1.0},
            "take_profit": {"type": "fixed_pct", "percent": 2.0},
        }
    )
    ind = dsl.indicators["ci"]
    assert ind.type == _PLUGIN_NAME
    assert (ind.model_extra or {}).get("gain") == 2.0


# ---------------------------------------------------------------------------
# 3. GA pool: threshold_range + param_ranges → auto-enrollment.
# ---------------------------------------------------------------------------


def test_contract_ga_pool(synthetic_plugin) -> None:
    from vibe_quant.discovery.genome import build_indicator_pool

    # build_indicator_pool reads the live registry — must be called
    # fresh after the fixture registers the spec.
    pool = build_indicator_pool()
    assert _PLUGIN_NAME in pool
    entry = pool[_PLUGIN_NAME]
    assert entry.param_ranges == {"period": (5.0, 50.0), "gain": (0.5, 3.0)}
    assert entry.default_threshold_range == (0.0, 1000.0)


# ---------------------------------------------------------------------------
# 4. API catalog: /api/indicators/catalog surfaces the plugin with its
#    display_name / category / param_ranges.
# ---------------------------------------------------------------------------


def test_contract_api_catalog(synthetic_plugin) -> None:
    from fastapi.testclient import TestClient

    from vibe_quant.api.app import create_app

    client = TestClient(create_app())
    resp = client.get("/api/indicators/catalog")
    assert resp.status_code == 200

    indicators = resp.json()["indicators"]
    entry = next((e for e in indicators if e["type_name"] == _PLUGIN_NAME), None)
    assert entry is not None, "plugin missing from catalog endpoint"
    assert entry["display_name"] == "CI Test Plugin"
    assert entry["category"] == "Custom"

    # Frontend contract (bd-mgx7): the indicator picker renders params
    # dynamically from `default_params` (see IndicatorsTab.tsx's merged
    # catalog + IndicatorParamFields iterating Object.entries). Pin every
    # key the TS ApiIndicatorEntry interface declares so a schema drop
    # would be caught before it breaks the picker.
    expected_keys = {
        "type_name",
        "display_name",
        "description",
        "category",
        "popular",
        "chart_placement",
        "default_params",
        "param_schema",
        "output_names",
        "requires_high_low",
        "requires_volume",
    }
    assert expected_keys.issubset(entry.keys()), (
        f"missing keys: {expected_keys - entry.keys()}"
    )
    # Plugin-declared params flow into default_params + param_schema so
    # the picker can render them.
    assert entry["default_params"] == {"period": 10, "gain": 1.5}
    assert entry["param_schema"] == {"period": "int", "gain": "float"}


# ---------------------------------------------------------------------------
# 5. Compiler: DSL referencing the plugin compiles to a loadable module
#    whose generated source calls the registered compute_fn.
# ---------------------------------------------------------------------------


def test_contract_compiler(synthetic_plugin) -> None:
    from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
    from vibe_quant.dsl.parser import validate_strategy_dict

    dsl = validate_strategy_dict(
        {
            "name": "citest_compile",
            "timeframe": "5m",
            "indicators": {
                "ci": {"type": _PLUGIN_NAME, "period": 10, "gain": 2.0}
            },
            "entry_conditions": {"long": ["ci > 100"]},
            "exit_conditions": {"long": ["ci < 90"]},
            "stop_loss": {"type": "fixed_pct", "percent": 1.0},
            "take_profit": {"type": "fixed_pct", "percent": 2.0},
        }
    )
    src = StrategyCompiler().compile(dsl)
    # Source must be syntactically valid Python.
    compile(src, "<generated>", "exec")

    # Generated source imports the plugin's compute_fn by full module path.
    assert compute_adaptive_rsi.__name__ in src
    assert compute_adaptive_rsi.__module__ in src

    # Plugin-declared param `gain` must be merged into the compute_fn
    # params dict. Per compiler._merge_effective_params, plugin extras
    # appear in the params literal alongside period.
    assert '"gain": 2.0' in src

    # Also verify compile_to_module produces classes.
    mod = StrategyCompiler().compile_to_module(dsl)
    camel = _to_class_name(dsl.name)
    assert hasattr(mod, f"{camel}Strategy")
    assert hasattr(mod, f"{camel}Config")


# ---------------------------------------------------------------------------
# 6. Contract negative: a plugin that registers with an invalid param
#    type is rejected at IndicatorConfig validation.
# ---------------------------------------------------------------------------


def test_contract_rejects_wrong_param_type(synthetic_plugin) -> None:
    from vibe_quant.dsl.schema import IndicatorConfig

    # `gain` expected float; passing a string must fail.
    with pytest.raises(ValueError, match="expected float"):
        IndicatorConfig(type=_PLUGIN_NAME, period=10, gain="nope")

    # Unknown extras must fail.
    with pytest.raises(ValueError, match="unknown param 'bogus'"):
        IndicatorConfig(type=_PLUGIN_NAME, period=10, gain=1.0, bogus=True)
