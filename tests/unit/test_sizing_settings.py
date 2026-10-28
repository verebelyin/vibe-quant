"""One shared run-sizing rule: exposure cap, risk per trade and the margin limit.

This file owns the ONE input-domain table for all three fields: each limit's equality
edges, one interior point per field and each non-finite / type class once per guard
(``_is_amount`` and ``_is_margin`` are separate functions, so a margin row cannot be
assumed to cover a cap row). The HTTP routes, the screening runner and the discovery CLI
keep ONE thin wiring test each (the request is refused, nothing launches) and never
re-table the domain.

Binding contract rows exercised here: S1 (key absent -> defaults, nothing stored),
S2 (an explicit default stores nothing), S3/S4 (ints and fractions), S5-S9 and S12
(every other input -> ``SizingConfigError`` naming the raw value; the type and range
are checked BEFORE any ``float()``, so ``10**400`` never raises OverflowError),
S10 (both upper bounds are inclusive), S11, S13 (``max_position_pct`` may not exceed
the run's leverage), plus the chief rulings S1-M1..S1-M6 (exact error wording,
``check_leverage``, ``from_params``, ``recorded_sizing``, import isolation,
``to_params``).

Task T3 adds the margin limit: R26 (an absent ``max_margin_pct`` is OFF and stores
nothing), R26b (an explicit ``null`` is an error, never "off"), R26c (bool/string/0/above-1/
nan are the same error quoting the raw value) and R26d (an explicit 0.5 is stored, and
``strategy_params(leverage)`` rides the run leverage along as ``margin_leverage``).

The rule is an exposure CAP, not a target: ``min(risk/stop%, cap)`` is the template's
call, out of scope here.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vibe_quant.sizing_settings import (
    DEFAULT_MAX_POSITION_PCT,
    DEFAULT_RISK_PER_TRADE,
    SIZING_FIELDS,
    SizingConfigError,
    SizingSettings,
    recorded_sizing,
)

_INVALID = object()  # marker: the input must raise SizingConfigError

CAP_MESSAGE = "max_position_pct must be a number in (0, 125], got"
RISK_MESSAGE = (
    "risk_per_trade must be more than 0 and at most 0.5 (50% of equity per trade), got"
)
MARGIN_MESSAGE = "max_margin_pct must be a number in (0, 1], got"

_MESSAGES = {
    "max_position_pct": CAP_MESSAGE,
    "risk_per_trade": RISK_MESSAGE,
    "max_margin_pct": MARGIN_MESSAGE,
}


class _IntSubclass(int):
    """An int subclass must be accepted exactly like an int."""


# The ONE domain table: (field, value, expected). The three fields share the same
# type-first rule, so the non-finite / type classes appear once, not once per field.
INPUT_DOMAIN = [
    pytest.param("max_position_pct", 125, 125.0, id="cap-upper-bound-inclusive"),
    pytest.param("max_position_pct", 125.0000001, _INVALID, id="cap-just-above-bound"),
    pytest.param("max_position_pct", 1, 1.0, id="cap-int-interior"),
    pytest.param("max_position_pct", 1e-9, 1e-9, id="cap-tiny-fraction"),
    pytest.param("max_position_pct", 0, _INVALID, id="cap-zero"),
    pytest.param("max_position_pct", -1, _INVALID, id="cap-negative"),
    pytest.param("risk_per_trade", 0.5, 0.5, id="risk-upper-bound-inclusive"),
    pytest.param("risk_per_trade", 0.5000001, _INVALID, id="risk-just-above-bound"),
    pytest.param("risk_per_trade", 0.02, 0.02, id="risk-interior"),
    pytest.param("risk_per_trade", 0, _INVALID, id="risk-zero"),
    pytest.param("max_margin_pct", 1, 1.0, id="margin-upper-bound-inclusive"),
    pytest.param("max_margin_pct", 1.0000001, _INVALID, id="margin-just-above-bound"),
    pytest.param("max_margin_pct", 1e-9, 1e-9, id="margin-interior"),
    pytest.param("max_margin_pct", 0, _INVALID, id="margin-zero"),
    pytest.param("max_margin_pct", None, None, id="margin-none-is-off"),
    # _is_margin is its own guard, so its type classes are tabled too: a mutant in it
    # would survive a row asserted only through the cap/risk guard.
    pytest.param("max_margin_pct", True, _INVALID, id="margin-bool"),
    pytest.param("max_margin_pct", "0.5", _INVALID, id="margin-numeric-string"),
    pytest.param("max_margin_pct", math.nan, _INVALID, id="margin-nan"),
    pytest.param("max_position_pct", math.nan, _INVALID, id="nan"),
    pytest.param("max_position_pct", math.inf, _INVALID, id="inf"),
    pytest.param("max_position_pct", -math.inf, _INVALID, id="minus-inf"),
    pytest.param("max_position_pct", True, _INVALID, id="bool"),
    pytest.param("max_position_pct", None, _INVALID, id="none"),
    pytest.param("max_position_pct", "0.5", _INVALID, id="numeric-string"),
    pytest.param("max_position_pct", 10**400, _INVALID, id="10-to-400"),
    pytest.param("max_position_pct", [], _INVALID, id="list"),
    pytest.param("max_position_pct", _IntSubclass(2), 2.0, id="int-subclass-is-an-int"),
]


def _build(field: str, value: object) -> SizingSettings:
    if field == "max_position_pct":
        return SizingSettings(max_position_pct=value)  # type: ignore[arg-type]
    if field == "risk_per_trade":
        return SizingSettings(risk_per_trade=value)  # type: ignore[arg-type]
    return SizingSettings(max_margin_pct=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(("field", "value", "expected"), INPUT_DOMAIN)
def test_input_domain(field: str, value: object, expected: object) -> None:
    """S3-S12/R26c: a value in (0, upper] parses as a float; EVERY other input is a hard
    error quoting the raw value (the bool and 10**400 rows pin the type-first check)."""
    if expected is _INVALID:
        with pytest.raises(SizingConfigError) as excinfo:
            _build(field, value)
        assert str(excinfo.value) == f"{_MESSAGES[field]} {value!r}"
        return
    settings = _build(field, value)
    assert getattr(settings, field) == expected
    if expected is not None:
        assert type(getattr(settings, field)) is float


def test_absent_params_are_defaults_and_store_nothing() -> None:
    """S1/S2/R26: a missing key means (0.5, 0.02, off); an explicit default stores
    exactly what a default launch stores (nothing)."""
    settings = SizingSettings.from_params({})
    assert (
        settings.max_position_pct,
        settings.risk_per_trade,
        settings.max_margin_pct,
    ) == (0.5, 0.02, None)
    assert SizingSettings(max_margin_pct=None).max_margin_pct is None
    assert settings.to_params() == {}
    assert settings.strategy_params(10) == {}
    explicit = SizingSettings.from_params({"max_position_pct": 0.5, "risk_per_trade": 0.02})
    assert explicit == settings
    assert explicit.to_params() == SizingSettings().to_params() == {}
    assert (DEFAULT_MAX_POSITION_PCT, DEFAULT_RISK_PER_TRADE) == (0.5, 0.02)


def test_to_params_stores_only_non_defaults_as_floats() -> None:
    """S1-M6/R26d: only the keys a run moved off the default, stored as floats."""
    assert SizingSettings(0.5, 0.02).to_params() == {}
    assert SizingSettings(1, 0.02).to_params() == {"max_position_pct": 1.0}
    assert SizingSettings(0.5, 0.1).to_params() == {"risk_per_trade": 0.1}
    stored = SizingSettings(2, 0.25, 0.5).to_params()
    assert stored == {"max_position_pct": 2.0, "risk_per_trade": 0.25, "max_margin_pct": 0.5}
    assert all(type(value) is float for value in stored.values())


def test_from_params_reads_each_key_independently_and_never_mutates() -> None:
    """S1-M3: only the sizing keys are read, and the caller's mapping is untouched."""
    params: dict[str, object] = {"rsi_period": 14, "max_position_pct": 1}
    settings = SizingSettings.from_params(params)
    assert (settings.max_position_pct, settings.risk_per_trade) == (1.0, 0.02)
    assert settings.to_params() == {"max_position_pct": 1.0}
    assert params == {"rsi_period": 14, "max_position_pct": 1}


def test_from_params_reports_the_cap_first_when_both_are_bad() -> None:
    """S1-M3: both fields are validated, and max_position_pct is reported first."""
    with pytest.raises(SizingConfigError) as excinfo:
        SizingSettings.from_params({"max_position_pct": -1, "risk_per_trade": 0})
    assert str(excinfo.value) == f"{CAP_MESSAGE} -1"


def test_check_leverage_equality_edge() -> None:
    """S13/S1-M2: a cap exactly at the leverage is fully marginable; one step above is
    the first refused value."""
    SizingSettings().check_leverage(10.0)  # the 0.5 default is fine
    SizingSettings(max_position_pct=125).check_leverage(125)
    SizingSettings(max_position_pct=10.0).check_leverage(10.0)
    with pytest.raises(SizingConfigError) as just_above:
        SizingSettings(max_position_pct=10.000001).check_leverage(10)
    assert str(just_above.value) == (
        "max_position_pct (10.000001) must not exceed leverage (10.0)"
    )
    with pytest.raises(SizingConfigError) as above:
        SizingSettings.from_params({"max_position_pct": 20}).check_leverage(10)
    assert str(above.value) == "max_position_pct (20.0) must not exceed leverage (10.0)"


def test_error_is_a_value_error() -> None:
    """Callers (the API schema) turn this into a 422 through the ValueError path."""
    assert issubclass(SizingConfigError, ValueError)
    with pytest.raises(ValueError):
        SizingSettings(risk_per_trade=0.9)


# A ``params.get(key) or DEFAULT`` shortcut would silently turn a stored 0 / null into
# the default through BOTH readers (the constructor's own guard never sees them), so
# each reader is pinned separately and the message quotes the complete raw value.
_FALSEY_EXPLICIT = [
    pytest.param("max_position_pct", 0, CAP_MESSAGE, id="cap-zero"),
    pytest.param("risk_per_trade", 0, RISK_MESSAGE, id="risk-zero"),
    pytest.param("max_margin_pct", 0, MARGIN_MESSAGE, id="margin-zero"),
    pytest.param("max_margin_pct", None, MARGIN_MESSAGE, id="margin-null-is-not-off"),
]


def _read(reader: str, params: dict[str, object]) -> SizingSettings:
    if reader == "from_params":
        return SizingSettings.from_params(params)
    return recorded_sizing(params)


@pytest.mark.parametrize(("field", "value", "message"), _FALSEY_EXPLICIT)
@pytest.mark.parametrize("reader", ["from_params", "recorded_sizing"])
def test_explicit_falsey_value_is_never_defaulted(
    reader: str, field: str, value: object, message: str
) -> None:
    """S7/S9/R26b: a stored None or 0 must be rejected, never read back as the default."""
    with pytest.raises(SizingConfigError) as excinfo:
        _read(reader, {field: value})
    assert str(excinfo.value) == f"{message} {value!r}"


def test_recorded_sizing_defaults_legacy_rows_and_rejects_garbage() -> None:
    """S1-M4: a non-mapping reads as the defaults; a mapping is the launch rule (so a
    corrupt recorded value is an error, never a silent default)."""
    for value in (None, [], "params", 7):
        settings = recorded_sizing(value)
        assert settings == SizingSettings()
        assert settings.to_params() == {}
    assert recorded_sizing({}) == SizingSettings()
    assert recorded_sizing({"other": 1}) == SizingSettings()
    settings = recorded_sizing({"max_position_pct": 2.0, "risk_per_trade": 0.1})
    assert (settings.max_position_pct, settings.risk_per_trade) == (2.0, 0.1)
    with pytest.raises(SizingConfigError) as excinfo:
        recorded_sizing({"max_position_pct": -1})
    assert str(excinfo.value) == f"{CAP_MESSAGE} -1"


def test_sizing_fields_names_the_three_settings() -> None:
    """The registry later tasks use to reject a run-level sizing key inside a sweep."""
    assert SIZING_FIELDS == ("max_position_pct", "risk_per_trade", "max_margin_pct")


def test_max_margin_strategy_params() -> None:
    """R26d: an explicit 0.5 is stored and the run leverage rides along as
    ``margin_leverage`` (two unrelated leverages, literal expected values)."""
    settings = SizingSettings.from_params({"max_margin_pct": 0.5})
    assert settings.max_margin_pct == 0.5
    assert settings.to_params() == {"max_margin_pct": 0.5}
    params = settings.strategy_params(20)
    assert params == {"max_margin_pct": 0.5, "margin_leverage": 20.0}
    assert all(type(value) is float for value in params.values())
    assert settings.strategy_params(1) == {"max_margin_pct": 0.5, "margin_leverage": 1.0}


def test_importing_adds_no_heavy_modules() -> None:
    """S1-M5: the rule is stdlib only — importing it pulls no engine/validation stack."""
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys, vibe_quant.sizing_settings; "
        "print(sorted(m for m in sys.modules if m.startswith("
        "('nautilus_trader', 'vibe_quant.validation', 'vibe_quant.liquidation'))))"
    )
    env = {**os.environ, "PYTHONPATH": str(root)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"
