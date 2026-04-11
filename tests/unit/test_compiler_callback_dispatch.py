"""Regression coverage for the P4 callback-dispatch compiler refactor.

These tests pin the branches that the spec-driven dispatcher has to keep
working so future plugin additions (Phase 6+) don't regress the contract:

1. **compute_fn-only path** — a spec with ``nt_class=None`` + ``compute_fn``
   (e.g. TEMA) compiles straight onto the ``_update_pta_indicators`` +
   ``_pta_values`` code path. No NT indicator instantiation.

2. **NT-only path** — a spec with a live ``nt_class`` (e.g. RSI) compiles
   onto ``self.ind_{name} = {NtClass}(...)`` + ``_get_indicator_value``
   reading from the NT attribute. No pandas import, no compute_fn import.

3. **Partial NT-coverage / sub-output fallback** — a spec that has both
   ``nt_class`` AND ``output_names`` where one of the outputs is NOT in
   ``nt_output_attrs`` / ``computed_outputs``. When a strategy references
   the missing sub-output, the compiler must force the indicator to the
   compute_fn path (the generalized MACD quirk). We register a throwaway
   spec at test time to exercise this without waiting for a real NT
   indicator to grow a similar limitation.

4. **Derived-output import** — a strategy using a spec that declares
   ``computed_outputs`` (e.g. BBANDS) must import the named helper(s)
   from ``vibe_quant.dsl.derived`` at the top of the generated source
   and call them in ``_get_indicator_value``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.dsl import StrategyCompiler, parse_strategy_string
from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry

if TYPE_CHECKING:
    import pandas as pd


def _compile(yaml: str) -> str:
    """Parse + compile a DSL YAML string to Python source."""
    dsl = parse_strategy_string(yaml)
    return StrategyCompiler().compile(dsl)


# =============================================================================
# Case 1 — compute_fn-only path (TEMA: nt_class=None)
# =============================================================================


def test_compile_with_compute_fn_only() -> None:
    """A spec with ``nt_class=None`` compiles onto the compute_fn path.

    TEMA is the canonical example: NT has no TripleExponentialMovingAverage,
    so its spec declares ``nt_class=None`` and routes through
    ``compute_tema``. The generated source must:

    - NOT instantiate an NT indicator for TEMA
    - Import ``compute_tema`` from ``vibe_quant.dsl.compute_builtins``
    - Populate ``_pta_values["tema"]`` via the generic compute_fn loop
    """
    source = _compile(
        """
name: tema_only
timeframe: 5m
indicators:
  tema:
    type: TEMA
    period: 14
entry_conditions:
  long:
    - tema > 0
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
    )
    compile(source, "<generated>", "exec")
    assert "self.ind_tema = " not in source
    assert "from vibe_quant.dsl.compute_builtins import compute_tema" in source
    assert "compute_tema(" in source
    assert '_pta_values.get("tema"' in source
    # Should check readiness via _pta_values, not via .initialized
    assert '"tema" not in self._pta_values' in source
    assert "self.ind_tema.initialized" not in source


# =============================================================================
# Case 2 — NT-only path (RSI: nt_class present, compute_fn irrelevant)
# =============================================================================


def test_compile_with_nt_only() -> None:
    """A single-output NT spec compiles onto the pure NT path.

    RSI has both ``nt_class`` and ``compute_fn`` set, but the NT class is
    always preferred — the generated source should instantiate
    ``RelativeStrengthIndex``, read ``self.ind_rsi.value``, and NOT emit
    any compute_fn import or ``_pta_values`` buffer for rsi.
    """
    source = _compile(
        """
name: rsi_only
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
    )
    compile(source, "<generated>", "exec")
    assert "RelativeStrengthIndex(period=self.config.rsi_period)" in source
    assert "self.register_indicator_for_bars(self.bar_type_5m, self.ind_rsi)" in source
    assert "self.ind_rsi.value" in source
    assert "compute_rsi" not in source
    assert "from vibe_quant.dsl.compute_builtins" not in source
    # A pure-NT strategy does not need the pandas-ta buffer at all.
    assert "self._pta_values" not in source
    assert "def _update_pta_indicators" not in source


# =============================================================================
# Case 3 — Sub-output fallback (generalized MACD quirk)
# =============================================================================


class _MockNtInd:
    """Duck type the compiler treats as an NT indicator class."""

    __name__ = "MockNtInd"
    __module__ = "vibe_quant.dsl.tests.mock"


def _compute_dummy_multi(
    df: pd.DataFrame, params: dict[str, Any]
) -> dict[str, pd.Series]:
    """Return three identical series keyed as ("main", "aux", "extra")."""
    base = df["close"].rolling(int(params.get("period", 5))).mean()
    return {"main": base, "aux": base, "extra": base}


@pytest.fixture
def _partial_nt_spec_registered() -> Any:
    """Register a throwaway spec whose NT path covers only ``main`` + ``aux``.

    ``extra`` is in ``output_names`` but absent from ``nt_output_attrs`` and
    ``computed_outputs``, so any condition referencing ``extra`` has to flip
    the indicator onto the compute_fn path.
    """
    spec = IndicatorSpec(
        name="TESTPARTIAL",
        nt_class=_MockNtInd,  # type: ignore[arg-type]
        pandas_ta_func=None,
        default_params={"period": 5},
        param_schema={"period": int},
        output_names=("main", "aux", "extra"),
        compute_fn=_compute_dummy_multi,
        nt_kwargs_fn=lambda params: {"period": params.get("period", 5)},
        nt_codegen_kwargs=(("period", "period"),),
        nt_output_attrs={"main": "value", "aux": "aux_value"},
        computed_outputs={},
        primary_output="main",
    )
    indicator_registry.register_spec(spec)
    try:
        yield spec
    finally:
        # Leave the registry clean for other tests — since there's no
        # public unregister() we pop the entry directly from the dict.
        indicator_registry._indicators.pop("TESTPARTIAL", None)  # type: ignore[attr-defined]


def test_compile_multi_output_nt_partial_coverage(
    _partial_nt_spec_registered: Any,
) -> None:
    """Referencing a sub-output not in ``nt_output_attrs`` forces compute_fn.

    ``main`` and ``aux`` are covered by the mock NT path. ``extra`` isn't,
    so a condition on ``foo.extra`` must flip the whole indicator onto the
    compute_fn path — we should NOT see ``MockNtInd(`` in the output and
    we SHOULD see a ``_compute_dummy_multi`` import + call.

    Post-P5 the schema validator queries ``indicator_registry`` directly,
    so registering ``TESTPARTIAL`` via the fixture is enough — no schema
    monkey-patching needed.
    """
    source = _compile(
        """
name: partial_nt_cov
timeframe: 5m
indicators:
  foo:
    type: TESTPARTIAL
    period: 5
entry_conditions:
  long:
    - foo.extra > 0
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
    )

    # The mock "NT" class name must be absent — the indicator should have
    # been forced onto compute_fn by the sub-output fallback.
    assert "MockNtInd(" not in source
    # The spec's compute_fn must be imported + called. The generated
    # import line uses the compute_fn's real ``__module__``, which for a
    # function defined inside this test file is this test file itself.
    expected_import = (
        f"from {_compute_dummy_multi.__module__} import _compute_dummy_multi"
    )
    assert expected_import in source
    assert "_compute_dummy_multi(" in source
    # The readiness check should use _pta_values (compute_fn path), not
    # .initialized (NT path).
    assert '"foo" not in self._pta_values' in source
    assert "self.ind_foo.initialized" not in source


# =============================================================================
# Case 4 — Computed outputs route through vibe_quant.dsl.derived
# =============================================================================


def test_compile_computed_outputs_import_derived_module() -> None:
    """BBANDS ``percent_b`` and ``bandwidth`` route through derived helpers.

    After the P4 refactor the compiler no longer emits inline formulas for
    band-derived outputs — it imports ``compute_percent_b`` /
    ``compute_bandwidth`` from ``vibe_quant.dsl.derived`` and calls them
    with the live NT indicator instance + the latest close. This test
    pins that contract so plugins can plug in their own helpers.
    """
    source = _compile(
        """
name: bbands_derived
timeframe: 5m
indicators:
  bb:
    type: BBANDS
    period: 20
    std_dev: 2.0
entry_conditions:
  long:
    - bb.percent_b < 0.2
    - bb.bandwidth > 0.02
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""
    )
    compile(source, "<generated>", "exec")
    # Import line from derived module.
    assert "from vibe_quant.dsl.derived import" in source
    assert "compute_percent_b" in source
    assert "compute_bandwidth" in source
    # Helper calls threaded through _get_indicator_value.
    assert "compute_percent_b(self.ind_bb, self._last_close)" in source
    assert "compute_bandwidth(self.ind_bb, self._last_close)" in source
    # No inline band math should remain — the helpers own the formula now.
    assert "_range = _upper - _lower" not in source
