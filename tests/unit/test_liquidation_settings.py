"""The one parse rule for liquidation settings (bead vibe-quant-2gyk1.1.8).

This file owns the ONE domain table for the mode and the leverage: every mode branch
(bad modes, the three valid ones, off vs on storage), both leverage equality edges, one
interior value and each non-whole / non-finite / type class once. The launch routes and
the screening runner keep ONE thin wiring test each and never re-table the domain; the
mode itself has no effect on leverage validation (one ``__post_init__`` branch), so the
leverage rows are not multiplied by mode.

Contract rows P1-P4, L1-L4: the mode is case-sensitive, ``None`` is NOT "off", and the
leverage is validated in EVERY mode (L4) as a whole number in [1, 125].
"""

from __future__ import annotations

import pytest

from vibe_quant.liquidation_settings import (
    DEFAULT_LEVERAGE,
    LIQUIDATION_MODES,
    MAX_LEVERAGE,
    MIN_LEVERAGE,
    MISSING,
    LiquidationConfigError,
    LiquidationSettings,
    resolve_leverage,
)

BAD_MODES: list[object] = [None, "Isolated", "", 1, True, "margin"]

BAD_LEVERAGE: list[object] = [
    0,
    126,
    2.5,
    125.0001,
    0.999,
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    "20",
    None,
    10**400,
    -(10**400),
    1e308,
    -1e308,
]


@pytest.mark.parametrize("mode", BAD_MODES)
def test_bad_mode_is_rejected(mode: object) -> None:
    """P4: the mode is case-sensitive, and an explicit null is never a silent "off"."""
    with pytest.raises(LiquidationConfigError) as err:
        LiquidationSettings(mode=mode)  # type: ignore[arg-type]
    assert str(err.value) == (
        f"liquidation_mode must be one of {LIQUIDATION_MODES}, got {mode!r}"
    )


def test_every_valid_mode_is_accepted() -> None:
    for mode in LIQUIDATION_MODES:
        assert LiquidationSettings(mode=mode).mode == mode


def test_defaults_are_off_at_ten_x() -> None:
    settings = LiquidationSettings()
    assert (settings.mode, settings.leverage) == ("off", DEFAULT_LEVERAGE)
    assert settings.on is False
    assert settings.to_params() == {}
    assert settings.to_notes() is None
    assert (MIN_LEVERAGE, MAX_LEVERAGE) == (1.0, 125.0)


def test_p1_p2_off_stores_no_params_and_no_notes() -> None:
    """P1/P2: an absent mode key and an explicit "off" are the same stored row."""
    for params in ({}, {"liquidation_mode": "off"}):
        settings = LiquidationSettings.from_params(params)
        assert (settings.mode, settings.leverage, settings.on) == ("off", 10.0, False)
        assert settings.to_params() == {}
        assert settings.to_notes() is None


@pytest.mark.parametrize("mode", ["isolated", "cross"])
def test_p3_on_modes_round_trip_through_params_and_notes(mode: str) -> None:
    """P3: the mode value is forwarded verbatim to both stored shapes."""
    settings = LiquidationSettings(mode=mode, leverage=20)  # type: ignore[arg-type]
    assert settings.on is True
    assert settings.leverage == 20.0 and isinstance(settings.leverage, float)
    assert settings.to_params() == {"liquidation_mode": mode, "leverage": 20.0}
    assert settings.to_notes() == {"mode": mode, "leverage": 20.0}


def test_from_params_reads_both_keys_and_defaults_the_missing_leverage() -> None:
    """L1/P1: a missing leverage key means 10x; both keys are read when present."""
    assert LiquidationSettings.from_params({"liquidation_mode": "isolated"}).leverage == 10.0
    settings = LiquidationSettings.from_params({"liquidation_mode": "cross", "leverage": 50})
    assert settings.mode == "cross" and settings.leverage == 50.0


def test_valid_leverage_edges_are_stored_as_float() -> None:
    """L2/S10: [1, 125] is inclusive at both ends and every value is stored as a float."""
    for leverage in (1, 20, 125, 125.0):
        settings = LiquidationSettings(mode="isolated", leverage=leverage)
        assert settings.leverage == float(leverage)
        assert isinstance(settings.leverage, float)


@pytest.mark.parametrize("leverage", BAD_LEVERAGE)
def test_bad_leverage_is_rejected(leverage: object) -> None:
    """L3/L4: any non-whole or out-of-range value is a config error naming the raw value —
    never an ``OverflowError`` from ``float(10**400)`` — through both entry points."""
    with pytest.raises(LiquidationConfigError) as err:
        LiquidationSettings(mode="isolated", leverage=leverage)  # type: ignore[arg-type]
    assert str(err.value) == f"leverage must be a whole number in [1, 125], got {leverage!r}"
    with pytest.raises(LiquidationConfigError, match=r"\[1, 125\]"):
        LiquidationSettings.from_params({"liquidation_mode": "isolated", "leverage": leverage})


def test_l4_off_validates_leverage_too_and_rejects_null() -> None:
    """L4: the domain is checked in EVERY mode, not only while liquidation is on; an
    explicit null leverage is rejected rather than defaulted."""
    with pytest.raises(LiquidationConfigError, match=r"\[1, 125\]"):
        LiquidationSettings(mode="off", leverage=0)
    assert LiquidationSettings(mode="off", leverage=3).leverage == 3.0
    with pytest.raises(LiquidationConfigError, match=r"\[1, 125\]"):
        LiquidationSettings(leverage=None)  # type: ignore[arg-type]
    with pytest.raises(LiquidationConfigError, match=r"\[1, 125\]"):
        LiquidationSettings.from_params({"leverage": None})


def test_to_params_off_non_default_leverage() -> None:
    """Off at a non-default leverage stores just the value; off at 10 stays empty."""
    assert LiquidationSettings(mode="off", leverage=3).to_params() == {"leverage": 3.0}
    assert LiquidationSettings(mode="off", leverage=DEFAULT_LEVERAGE).to_params() == {}
    assert LiquidationSettings(mode="isolated", leverage=3).to_params() == {
        "liquidation_mode": "isolated",
        "leverage": 3.0,
    }


# --------------------------------------------------------------- resolve_leverage (H1)


def test_resolve_leverage_picks_the_first_supplied() -> None:
    assert resolve_leverage(MISSING, MISSING) is MISSING
    assert resolve_leverage(3, MISSING) == 3
    assert resolve_leverage(MISSING, 20) == 20
    assert resolve_leverage(3, 3.0) == 3  # 3 == 3.0; the first value wins unchanged


def test_resolve_leverage_conflict_raises() -> None:
    with pytest.raises(LiquidationConfigError) as err:
        resolve_leverage(3, 20)
    assert str(err.value) == "conflicting leverage 3 and liquidation_leverage 20"
