"""Dynamic schema validation regression tests (P5).

After P5 the DSL parser's indicator-type validator reads from
``indicator_registry`` at validation time instead of checking a static
``VALID_INDICATOR_TYPES`` frozenset. These tests pin the new contract so
the plugin discovery flow (P6+) can rely on "spec registered -> parser
accepts the name" holding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from vibe_quant.dsl.indicators import IndicatorSpec, indicator_registry
from vibe_quant.dsl.schema import IndicatorConfig

if TYPE_CHECKING:
    from collections.abc import Iterator


def _noop_compute(df: Any, params: Any) -> Any:  # noqa: ARG001
    """Stub compute_fn for test specs — never actually called."""
    return df


# =============================================================================
# 1. Unknown indicator type is rejected
# =============================================================================


def test_unknown_indicator_type_rejected() -> None:
    """An indicator type with no registry entry must fail validation."""
    with pytest.raises((ValueError, ValidationError)):
        IndicatorConfig(type="DEFINITELY_NOT_A_REAL_INDICATOR", period=14)


# =============================================================================
# 2. Error message lists currently-registered names
# =============================================================================


def test_error_message_lists_registered_names() -> None:
    """Validator error surface must cite the live registry contents so the
    developer running a broken strategy knows exactly which names are on
    the table — especially useful once plugins add to that list."""
    try:
        IndicatorConfig(type="FAKENAME")
    except (ValueError, ValidationError) as exc:
        msg = str(exc)
    else:
        pytest.fail("Expected validation to fail for FAKENAME")

    # A handful of well-known built-ins must all appear somewhere in the
    # error text. If someone reduces the registry in the future this also
    # catches accidental removal.
    for known in ("RSI", "EMA", "MACD", "BBANDS"):
        assert known in msg, f"Expected {known} in validator error, got: {msg[:200]}"


# =============================================================================
# 3. Runtime spec registration is immediately accepted by the parser
# =============================================================================


@pytest.fixture
def _tmp_registered_spec() -> Iterator[IndicatorSpec]:
    """Register a throwaway spec with a unique name and tear down after."""
    spec = IndicatorSpec(
        name="TESTP5DYNAMIC",
        nt_class=None,
        pandas_ta_func=None,
        default_params={"period": 7},
        param_schema={"period": int},
        compute_fn=_noop_compute,
    )
    indicator_registry.register_spec(spec)
    try:
        yield spec
    finally:
        # Private attr access; the registry intentionally has no public
        # unregister() and shipping one is out of scope for this test.
        indicator_registry._indicators.pop("TESTP5DYNAMIC", None)  # type: ignore[attr-defined]


def test_runtime_registration_is_accepted_by_schema(
    _tmp_registered_spec: IndicatorSpec,
) -> None:
    """The proof that P5 matters: zero schema edits for a new spec.

    Before P5 this would have required also adding TESTP5DYNAMIC to the
    static frozenset in schema.py. After P5 the validator reads the live
    registry, so the spec alone is enough.
    """
    cfg = IndicatorConfig(type="TESTP5DYNAMIC", period=7)
    assert cfg.type == "TESTP5DYNAMIC"


def test_runtime_spec_accepted_via_full_dsl_parse(
    _tmp_registered_spec: IndicatorSpec,
) -> None:
    """End-to-end: a full YAML-style DSL dict referencing the freshly
    registered spec must round-trip through validate_strategy_dict.
    This covers the path the real compiler runs through at backtest time.
    """
    from vibe_quant.dsl.parser import validate_strategy_dict

    dsl = validate_strategy_dict(
        {
            "name": "p5_runtime_spec",
            "timeframe": "5m",
            "indicators": {"dyn": {"type": "TESTP5DYNAMIC", "period": 7}},
            "entry_conditions": {"long": ["dyn > 0"]},
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
        }
    )
    assert dsl.indicators["dyn"].type == "TESTP5DYNAMIC"


# =============================================================================
# 4. Case-insensitive type lookup
# =============================================================================


@pytest.mark.parametrize("raw", ["rsi", "RSI", "Rsi", "rSi"])
def test_case_insensitive_type_lookup(raw: str) -> None:
    """The validator must normalize to upper-case and accept any casing.

    The registry itself stores upper-case keys; schema's responsibility is
    to normalize. We pin the canonical form in the validated model.
    """
    cfg = IndicatorConfig(type=raw, period=14)
    assert cfg.type == "RSI"
