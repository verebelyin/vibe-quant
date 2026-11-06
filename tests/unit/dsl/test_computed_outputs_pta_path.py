"""computed_outputs on the compute_fn (pta) path never read a silent 0.0.

An indicator forced onto the compute_fn path (a condition references an output
that NT cannot supply) used to read every ``computed_outputs`` entry (e.g.
``donchian.position``, ``bbands.percent_b``) as ``_pta_values.get(key, 0.0)``
unless its compute_fn happened to return that key.

Contract pinned here for EVERY registry spec that has ``computed_outputs`` and a
``compute_fn``: when the compute_fn does not return the output, the generated
``_get_indicator_value`` (and ``_update_prev_values``) must still return the
``derived.py`` helper's value computed from the raw pta outputs - or the compile
must fail with a ``CompilerError``. Never a constant 0.0.

The generated methods are plain functions, so they are run unbound against a
stub ``self`` carrying only ``_pta_values`` / ``_last_close`` / ``_prev_values``.

regression: vibe-quant-ho2hc
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl import compute_builtins, derived
from vibe_quant.dsl.compiler import CompilerError, StrategyCompiler, _to_class_name
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.parser import DSLValidationError, validate_strategy_dict

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def _synthetic_ohlcv(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, n)))
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.5,
            "low": close - 1.5,
            "close": close,
            "volume": 1_000.0,
        }
    )


_SPECS_WITH_COMPUTED: list[IndicatorSpec] = [
    s for s in indicator_registry.all_specs() if s.computed_outputs and s.compute_fn is not None
]
# (spec name, computed output) for every output a strategy can actually reference.
_CASES: list[tuple[str, str]] = [
    (s.name, out)
    for s in _SPECS_WITH_COMPUTED
    for out in s.computed_outputs
    if out in s.output_names
]


def _without_computed(spec: IndicatorSpec) -> Callable[..., dict[str, pd.Series]]:
    """The spec's compute_fn minus the computed-output keys (the exposed case)."""
    real = spec.compute_fn
    assert real is not None

    def stripped(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
        return {k: v for k, v in real(df, params).items() if k not in spec.computed_outputs}

    # Generated source may only import allowed modules: publish the wrapper
    # under compute_builtins (the caller monkeypatches it in and out).
    stripped.__name__ = f"stripped_{real.__name__}"
    stripped.__module__ = compute_builtins.__name__
    return stripped


@pytest.fixture
def pta_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[str], IndicatorSpec]]:
    """Register ``<NAME>_PTAVAR`` specs forced onto the compute_fn path whose
    compute_fn omits the computed outputs; unregistered on teardown."""
    made: list[str] = []

    def make(name: str) -> IndicatorSpec:
        base = indicator_registry.get(name)
        assert base is not None
        stripped = _without_computed(base)
        monkeypatch.setattr(compute_builtins, stripped.__name__, stripped, raising=False)
        variant = replace(
            base,
            name=f"{name}_PTAVAR",
            nt_class=None,
            compute_fn=stripped,
        )
        indicator_registry.register_spec(variant)
        made.append(variant.name)
        return variant

    yield make
    for n in made:
        indicator_registry.unregister(n)


def _generated_strategy_cls(indicator_type: str, conditions: list[str]) -> type:
    dsl = validate_strategy_dict(
        {
            "name": "computed_probe",
            "timeframe": "1m",
            "indicators": {"ind": {"type": indicator_type, "period": 10}},
            "entry_conditions": {"long": conditions},
            "exit_conditions": {},
            "stop_loss": {"type": "fixed_pct", "percent": 5.0},
            "take_profit": {"type": "fixed_pct", "percent": 5.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    cls: type = getattr(module, f"{_to_class_name(dsl.name)}Strategy")
    return cls


def _stub(cls: type, pta_values: dict[str, float], last_close: float) -> Any:
    """Minimal ``self`` for the generated unbound methods under test."""
    stub = SimpleNamespace(_pta_values=pta_values, _last_close=last_close, _prev_values={})
    for method in ("_get_indicator_value", "_update_prev_values"):
        setattr(stub, method, partial(getattr(cls, method), stub))
    return stub


def _raw_pta_values(spec: IndicatorSpec) -> tuple[dict[str, float], float]:
    """Latest raw outputs of the REAL compute_fn, keyed like ``_pta_values``."""
    assert spec.compute_fn is not None
    df = _synthetic_ohlcv()
    res = spec.compute_fn(df, {**spec.default_params, "period": 10})
    values = {f"ind_{k}": float(s.iloc[-1]) for k, s in res.items()}
    values["ind"] = values[f"ind_{spec.primary_output or spec.output_names[0]}"]
    return values, float(df["close"].iloc[-1])


def _expected(spec: IndicatorSpec, output: str, values: dict[str, float], close: float) -> float:
    helper = getattr(derived, spec.computed_outputs[output])
    raw = SimpleNamespace(
        **{o: values[f"ind_{o}"] for o in spec.output_names if f"ind_{o}" in values}
    )
    return float(helper(raw, close))


def test_registry_has_computed_output_specs_to_cover() -> None:
    # Guard against the sweep silently covering nothing.
    assert {"BBANDS", "DONCHIAN"} <= {name for name, _ in _CASES}


@pytest.mark.parametrize(("spec_name", "output"), _CASES)
def test_computed_output_on_pta_path_matches_derived_helper(
    spec_name: str, output: str, pta_variant: Callable[[str], IndicatorSpec]
) -> None:
    variant = pta_variant(spec_name)
    base = indicator_registry.get(spec_name)
    assert base is not None
    values, close = _raw_pta_values(base)
    expected = _expected(base, output, values, close)
    assert expected != 0.0, "synthetic data must make the helper value non-zero"

    try:
        cls = _generated_strategy_cls(variant.name, [f"ind.{output} > 0"])
    except CompilerError:
        pytest.skip("clear CompilerError is an accepted outcome")  # never a silent 0.0

    # Raw outputs only: the stripped compute_fn never produced the computed key.
    pta = {k: v for k, v in values.items() if k != f"ind_{output}"}
    stub = _stub(cls, pta, close)
    got = stub._get_indicator_value(f"ind_{output}")
    assert got == expected

    stub._update_prev_values(None)
    assert stub._prev_values[f"ind_{output}"] == expected


def test_donchian_forced_by_prev_output_reads_real_position(
    pta_variant: Callable[[str], IndicatorSpec],  # noqa: ARG001
) -> None:
    """The real (non-variant) forced path: ``upper_prev`` is not NT-supplied."""
    spec = indicator_registry.get("DONCHIAN")
    assert spec is not None
    values, close = _raw_pta_values(spec)
    cls = _generated_strategy_cls("DONCHIAN", ["ind.position > 0.1", "ind.upper_prev > 0"])
    stub = _stub(cls, values, close)
    got = stub._get_indicator_value("ind_position")
    assert got == _expected(spec, "position", values, close)
    assert got != 0.0


def test_computed_output_missing_from_pta_values_falls_back_to_helper_not_zero(
    pta_variant: Callable[[str], IndicatorSpec],
) -> None:
    """NaN rows are never stored in ``_pta_values``; a computed output whose
    key is absent must come from the helper while the raw bands are present."""
    variant = pta_variant("BBANDS")
    cls = _generated_strategy_cls(variant.name, ["ind.percent_b > 0.5"])
    stub = _stub(cls, {"ind": 10.0, "ind_upper": 12.0, "ind_middle": 10.0, "ind_lower": 8.0}, 11.0)
    assert stub._get_indicator_value("ind_percent_b") == pytest.approx(0.75)
    assert stub._get_indicator_value("ind_bandwidth") == pytest.approx(0.4)
    # Raw bands not computed yet (warm-up): not ready, reads 0.0 rather than a made-up value.
    cold = _stub(cls, {}, 11.0)
    assert cold._get_indicator_value("ind_percent_b") == 0.0


def test_stored_computed_output_wins_over_helper(
    pta_variant: Callable[[str], IndicatorSpec],
) -> None:
    """A value compute_fn stored under the computed key is returned as is, by the
    accessor and by ``_update_prev_values``; the helper is only the fallback.

    The stored value deliberately differs from the helper result, so deriving
    always would flip ``bandwidth > 0.5``.
    """
    variant = pta_variant("BBANDS")
    cls = _generated_strategy_cls(variant.name, ["ind.bandwidth > 0.5"])
    stub = _stub(
        cls,
        {
            "ind": 10.0,
            "ind_upper": 12.0,
            "ind_middle": 10.0,
            "ind_lower": 8.0,
            "ind_bandwidth": 3.5,  # helper would give 0.4
            "ind_percent_b": 0.9,  # helper would give 0.75
        },
        11.0,
    )
    assert stub._get_indicator_value("ind_bandwidth") == 3.5
    assert stub._get_indicator_value("ind_percent_b") == 0.9
    stub._update_prev_values(None)
    assert stub._prev_values["ind_bandwidth"] == 3.5
    assert stub._prev_values["ind_percent_b"] == 0.9


def test_kc_bandwidth_is_not_referenceable() -> None:
    """KC has no ``bandwidth`` output; pin that referencing it is rejected, not
    read as 0.0."""
    kc = indicator_registry.get("KC")
    assert kc is not None
    assert "bandwidth" not in kc.output_names
    with pytest.raises(DSLValidationError, match="Unknown indicator 'ind.bandwidth'"):
        _generated_strategy_cls("KC", ["ind.bandwidth > 0.1"])
