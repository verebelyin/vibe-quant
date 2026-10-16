"""One shared run-sizing rule: exposure cap x equity, risk per trade (task S1).

Binding contract rows exercised here: S1 (key absent -> defaults, nothing stored),
S2 (an explicit default stores nothing), S3/S4 (ints and fractions), S5-S9 and S12
(every other input -> ``SizingConfigError`` naming the raw value; the type and range
are checked BEFORE any ``float()``, so ``10**400`` never raises OverflowError),
S10 (both upper bounds are inclusive), S11, S13 (``max_position_pct`` may not exceed
the run's leverage), plus the chief rulings S1-M1..S1-M6 (exact error wording,
``check_leverage``, ``from_params``, ``recorded_sizing``, import isolation,
``to_params``).

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
    MAX_MAX_POSITION_PCT,
    MAX_RISK_PER_TRADE,
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

# Either field accepts the same domain: (0, upper] for finite numbers, nothing else.
INPUT_DOMAIN = [
    pytest.param("max_position_pct", 1, 1.0, id="cap-int-1"),
    pytest.param("max_position_pct", 2.0, 2.0, id="cap-integral-float"),
    pytest.param("max_position_pct", 0.75, 0.75, id="cap-fraction"),
    pytest.param("max_position_pct", 1e-9, 1e-9, id="cap-tiny-fraction"),
    pytest.param("max_position_pct", 100.0, 100.0, id="cap-100"),
    pytest.param("max_position_pct", 125, 125.0, id="cap-upper-bound-inclusive"),
    pytest.param("risk_per_trade", 0.02, 0.02, id="risk-default"),
    pytest.param("risk_per_trade", 0.25, 0.25, id="risk-fraction"),
    pytest.param("risk_per_trade", 1e-9, 1e-9, id="risk-tiny-fraction"),
    pytest.param("risk_per_trade", 0.5, 0.5, id="risk-upper-bound-inclusive"),
    pytest.param("max_position_pct", 125.0000001, _INVALID, id="cap-just-above-bound"),
    pytest.param("max_position_pct", 1e308, _INVALID, id="cap-1e308"),
    pytest.param("risk_per_trade", 0.5000001, _INVALID, id="risk-just-above-bound"),
    pytest.param("risk_per_trade", 1, _INVALID, id="risk-int-1-above-bound"),
    pytest.param("max_position_pct", 0, _INVALID, id="cap-zero"),
    pytest.param("max_position_pct", -0.0, _INVALID, id="cap-minus-zero"),
    pytest.param("max_position_pct", -1, _INVALID, id="cap-negative"),
    pytest.param("risk_per_trade", 0, _INVALID, id="risk-zero"),
    pytest.param("risk_per_trade", -0.5, _INVALID, id="risk-negative"),
    pytest.param("max_position_pct", True, _INVALID, id="cap-True"),
    pytest.param("risk_per_trade", False, _INVALID, id="risk-False"),
    pytest.param("max_position_pct", "0.5", _INVALID, id="cap-numeric-string"),
    pytest.param("risk_per_trade", "0.02", _INVALID, id="risk-numeric-string"),
    pytest.param("max_position_pct", None, _INVALID, id="cap-None"),
    pytest.param("risk_per_trade", None, _INVALID, id="risk-None"),
    pytest.param("max_position_pct", math.nan, _INVALID, id="cap-nan"),
    pytest.param("max_position_pct", math.inf, _INVALID, id="cap-inf"),
    pytest.param("max_position_pct", -math.inf, _INVALID, id="cap-minus-inf"),
    pytest.param("risk_per_trade", math.nan, _INVALID, id="risk-nan"),
    pytest.param("risk_per_trade", math.inf, _INVALID, id="risk-inf"),
    pytest.param("max_position_pct", 10**400, _INVALID, id="cap-10-to-400"),
    pytest.param("risk_per_trade", 10**400, _INVALID, id="risk-10-to-400"),
    pytest.param("max_position_pct", [], _INVALID, id="cap-list"),
    pytest.param("max_position_pct", {}, _INVALID, id="cap-dict"),
    pytest.param("max_position_pct", "abc", _INVALID, id="cap-text"),
    pytest.param("risk_per_trade", [], _INVALID, id="risk-list"),
]


class _FloatSubclass(float):
    """A float subclass must be accepted exactly like a float."""


class _IntSubclass(int):
    """An int subclass must be accepted exactly like an int."""


def _build(field: str, value: object) -> SizingSettings:
    if field == "max_position_pct":
        return SizingSettings(max_position_pct=value)  # type: ignore[arg-type]
    return SizingSettings(risk_per_trade=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(("field", "value", "expected"), INPUT_DOMAIN)
def test_input_domain(field: str, value: object, expected: object) -> None:
    """S3-S12: a finite number in (0, upper] parses; EVERY other input is a hard error."""
    if expected is _INVALID:
        message = CAP_MESSAGE if field == "max_position_pct" else RISK_MESSAGE
        with pytest.raises(SizingConfigError) as excinfo:
            _build(field, value)
        assert str(excinfo.value) == f"{message} {value!r}"
    else:
        settings = _build(field, value)
        assert getattr(settings, field) == expected
        assert type(getattr(settings, field)) is float


def test_absent_is_default() -> None:
    """S1: a missing key means (0.5, 0.02) and nothing is stored."""
    settings = SizingSettings.from_params({})
    assert settings.max_position_pct == 0.5
    assert settings.risk_per_trade == 0.02
    assert settings.to_params() == {}
    assert DEFAULT_MAX_POSITION_PCT == 0.5
    assert DEFAULT_RISK_PER_TRADE == 0.02


def test_explicit_default_stores_nothing() -> None:
    """S2: an explicit default stores exactly what a default launch stores (nothing)."""
    explicit = SizingSettings.from_params({"max_position_pct": 0.5, "risk_per_trade": 0.02})
    assert explicit.to_params() == {}
    assert explicit == SizingSettings()
    assert explicit.to_params() == SizingSettings().to_params()


def test_int_is_float() -> None:
    """S3: an int is accepted and stored as a float."""
    settings = SizingSettings(max_position_pct=1)
    assert settings.max_position_pct == 1.0
    assert type(settings.max_position_pct) is float


def test_fractions_accepted() -> None:
    """S4: the cap and the risk need not be whole numbers (a tiny cap just skips entries)."""
    assert SizingSettings(max_position_pct=2.0).max_position_pct == 2.0
    assert SizingSettings(max_position_pct=0.75).max_position_pct == 0.75
    assert SizingSettings(max_position_pct=1e-9).max_position_pct == 1e-9
    assert SizingSettings(risk_per_trade=0.75 * 0.5).risk_per_trade == 0.375


def test_string_rejected() -> None:
    """S5: a numeric string is not a number (the API's 422 quotes it back)."""
    with pytest.raises(SizingConfigError) as excinfo:
        SizingSettings(max_position_pct="0.5")  # type: ignore[arg-type]
    assert str(excinfo.value) == f"{CAP_MESSAGE} '0.5'"


def test_bool_rejected() -> None:
    """S6: bool is an int subclass but is NOT a number here — checked before the int path."""
    assert isinstance(True, int)  # the trap this test pins
    for value in (True, False):
        with pytest.raises(SizingConfigError) as excinfo:
            SizingSettings(max_position_pct=value)
        assert str(excinfo.value) == f"{CAP_MESSAGE} {value!r}"


def test_none_rejected() -> None:
    """S7: an explicit None is an error, never "use the default"."""
    with pytest.raises(SizingConfigError) as cap:
        SizingSettings(max_position_pct=None)  # type: ignore[arg-type]
    assert str(cap.value) == f"{CAP_MESSAGE} None"
    with pytest.raises(SizingConfigError) as risk:
        SizingSettings.from_params({"risk_per_trade": None})
    assert str(risk.value) == f"{RISK_MESSAGE} None"


def test_non_finite_rejected() -> None:
    """S8: nan/inf/-inf are not amounts (the boundary comparison rejects all three)."""
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(SizingConfigError):
            SizingSettings(risk_per_trade=value)


def test_zero_and_negative_rejected() -> None:
    """S9: zero and every negative are rejected — including -0.0."""
    for value in (0, 0.0, -0.0, -0.0000001, -1, -125):
        with pytest.raises(SizingConfigError) as excinfo:
            SizingSettings(max_position_pct=value)
        assert str(excinfo.value) == f"{CAP_MESSAGE} {value!r}"


def test_upper_bounds_inclusive() -> None:
    """S10: BOTH upper bounds are inclusive; one ulp above is rejected."""
    assert SizingSettings(max_position_pct=125).max_position_pct == 125.0
    assert SizingSettings(risk_per_trade=0.5).risk_per_trade == 0.5
    with pytest.raises(SizingConfigError):
        SizingSettings(max_position_pct=125.0000001)
    with pytest.raises(SizingConfigError):
        SizingSettings(risk_per_trade=0.5000001)
    assert MAX_MAX_POSITION_PCT == 125.0
    assert MAX_RISK_PER_TRADE == 0.5


def test_huge_int_no_overflow() -> None:
    """S11: 10**400 is rejected by comparison, never by float(10**400)'s OverflowError."""
    with pytest.raises(SizingConfigError) as excinfo:
        SizingSettings(max_position_pct=10**400)
    assert str(excinfo.value) == f"{CAP_MESSAGE} {10**400!r}"


def test_non_numeric_rejected() -> None:
    """S12: a list, a dict and a word are all rejected."""
    for value in ([], {}, "abc"):
        with pytest.raises(SizingConfigError):
            SizingSettings(risk_per_trade=value)


def test_subclasses_behave_like_their_base() -> None:
    """Edge: a float subclass is a float and an int subclass is an int (stored as float)."""
    from_float = SizingSettings(max_position_pct=_FloatSubclass(1.5))
    assert from_float.max_position_pct == 1.5
    assert type(from_float.max_position_pct) is float
    from_int = SizingSettings(max_position_pct=_IntSubclass(2))
    assert from_int.max_position_pct == 2.0
    assert type(from_int.max_position_pct) is float
    with pytest.raises(SizingConfigError):
        SizingSettings(max_position_pct=_FloatSubclass(math.nan))


def test_exposure_above_leverage() -> None:
    """S13: the exposure cap may not exceed the run's leverage (margin can't cover it)."""
    with pytest.raises(SizingConfigError) as excinfo:
        SizingSettings.from_params({"max_position_pct": 20}).check_leverage(10)
    assert str(excinfo.value) == "max_position_pct (20.0) must not exceed leverage (10.0)"
    SizingSettings().check_leverage(10.0)  # the 0.5 default is fine
    SizingSettings(max_position_pct=125).check_leverage(125)


def test_leverage_equal_is_allowed_and_just_above_is_not() -> None:
    """S1-M2: equality is OK; 10.000001 against a 10 cap is not."""
    SizingSettings(max_position_pct=10).check_leverage(10)
    SizingSettings(max_position_pct=10.0).check_leverage(10.0)
    with pytest.raises(SizingConfigError) as excinfo:
        SizingSettings(max_position_pct=10.000001).check_leverage(10)
    assert str(excinfo.value) == (
        "max_position_pct (10.000001) must not exceed leverage (10.0)"
    )


def test_error_texts() -> None:
    """S1-M1: the exact wording of all three messages."""
    with pytest.raises(SizingConfigError) as cap:
        SizingSettings(max_position_pct=0)
    assert str(cap.value) == f"{CAP_MESSAGE} 0"
    with pytest.raises(SizingConfigError) as risk:
        SizingSettings(risk_per_trade=-1)
    assert str(risk.value) == f"{RISK_MESSAGE} -1"
    with pytest.raises(SizingConfigError) as leverage:
        SizingSettings(max_position_pct=20).check_leverage(10)
    assert str(leverage.value) == "max_position_pct (20.0) must not exceed leverage (10.0)"


def test_error_is_a_value_error() -> None:
    """Callers (the API schema) turn this into a 422 through the ValueError path."""
    assert issubclass(SizingConfigError, ValueError)
    with pytest.raises(ValueError):
        SizingSettings(risk_per_trade=0.9)


def test_from_params_ignores_other_keys_and_never_mutates() -> None:
    """S1-M3: only the two sizing keys are read, and the caller's mapping is untouched."""
    params: dict[str, object] = {"rsi_period": 14, "max_position_pct": 1}
    settings = SizingSettings.from_params(params)
    assert settings.max_position_pct == 1.0
    assert settings.risk_per_trade == 0.02
    assert params == {"rsi_period": 14, "max_position_pct": 1}


def test_from_params_missing_risk_key_is_the_default() -> None:
    """S1-M3: each key is independent — a supplied cap keeps the default risk."""
    settings = SizingSettings.from_params({"max_position_pct": 3})
    assert settings.risk_per_trade == 0.02
    assert settings.to_params() == {"max_position_pct": 3.0}


def test_from_params_reports_the_cap_first_when_both_are_bad() -> None:
    """S1-M3: both fields are validated, and max_position_pct is reported first."""
    with pytest.raises(SizingConfigError) as excinfo:
        SizingSettings.from_params({"max_position_pct": -1, "risk_per_trade": 0})
    assert str(excinfo.value) == f"{CAP_MESSAGE} -1"


_FALSEY_EXPLICIT = [
    pytest.param("max_position_pct", None, CAP_MESSAGE, id="cap-None"),
    pytest.param("max_position_pct", 0, CAP_MESSAGE, id="cap-zero"),
    pytest.param("risk_per_trade", None, RISK_MESSAGE, id="risk-None"),
    pytest.param("risk_per_trade", 0, RISK_MESSAGE, id="risk-zero"),
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
    """S7/S9: a stored None or 0 must be rejected, never read back as the default.

    A ``params.get(key) or DEFAULT`` shortcut would silently turn both into 0.5 / 0.02 through
    BOTH readers (the constructor's own guard never sees them), so each reader is pinned
    separately and the assertion quotes the complete raw-value message.
    """
    with pytest.raises(SizingConfigError) as excinfo:
        _read(reader, {field: value})
    assert str(excinfo.value) == f"{message} {value!r}"


def test_recorded_sizing_non_dict_is_defaults() -> None:
    """S1-M4: a legacy row with no params at all reads as the defaults."""
    for value in (None, [], "params", 7):
        settings = recorded_sizing(value)
        assert settings == SizingSettings()
        assert settings.to_params() == {}


def test_recorded_sizing_dict_uses_from_params() -> None:
    """S1-M4: a dict goes through the same rule as a launch request."""
    settings = recorded_sizing({"max_position_pct": 2.0, "risk_per_trade": 0.1})
    assert settings.max_position_pct == 2.0
    assert settings.risk_per_trade == 0.1
    assert recorded_sizing({}) == SizingSettings()
    assert recorded_sizing({"other": 1}) == SizingSettings()


def test_recorded_sizing_garbage_raises() -> None:
    """S1-M4: an edited legacy row is an error, never a silent default."""
    with pytest.raises(SizingConfigError) as excinfo:
        recorded_sizing({"max_position_pct": -1})
    assert str(excinfo.value) == f"{CAP_MESSAGE} -1"


def test_to_params_stores_only_non_defaults() -> None:
    """S1-M6: only keys whose value differs from the default, as floats."""
    assert SizingSettings(0.5, 0.02).to_params() == {}
    assert SizingSettings(1, 0.02).to_params() == {"max_position_pct": 1.0}
    assert SizingSettings(0.5, 0.1).to_params() == {"risk_per_trade": 0.1}
    stored = SizingSettings(2, 0.25).to_params()
    assert stored == {"max_position_pct": 2.0, "risk_per_trade": 0.25}
    assert all(type(v) is float for v in stored.values())


def test_sizing_fields_names_the_two_settings() -> None:
    """The registry later tasks use to reject a sizing key inside a sweep."""
    assert SIZING_FIELDS == ("max_position_pct", "risk_per_trade")


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
