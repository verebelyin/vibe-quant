"""Trend-variant run spec: field ranges, combinations, daily schedule, arm hash.

Contract rows R6-R12 of arch-c9-2 (vibe-quant-clmmc.9) plus review findings
C9-7 (every economic arm input hashed, legacy serialization kept, family N
inside the finite DSR quantile domain) and C9-11 (funding refuses a spread).
Every rule fires at construction, before any data load.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from vibe_quant.portfolio.spec import (
    TREND_LOOKBACKS_DAYS,
    CarrySpec,
    decision_times,
    family_num_trials,
)

_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_START = "2022-06-01"
_END = "2026-09-28"
_HOUR_NS = 3_600 * 10**9
_DAY_NS = 24 * _HOUR_NS
_DEFAULT_HASH = "aeaa2d4b4a623be018aec2f7c385d6695936990705fa736ee6f7625bd6c896e3"


def _trend_kw(**over: Any) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "signal": "trend",
        "schedule": "daily",
        "weighting": "time_series",
        "direction": "long_flat",
        "vol_target": 0.40,
    }
    kw.update(over)
    return {key: value for key, value in kw.items() if value is not _DROP}


_DROP = object()


def _trend(symbols: tuple[str, ...] = _SYMBOLS, **over: Any) -> CarrySpec:
    return CarrySpec.from_dates(symbols, _START, _END, **_trend_kw(**over))


def _funding(**over: Any) -> CarrySpec:
    return CarrySpec.from_dates(("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"), _START, _END, **over)


# --- R6: trend needs its four fields; vol_target range; funding refuses trend fields ---


@pytest.mark.parametrize(
    ("over", "field"),
    [
        ({"weighting": _DROP}, "weighting"),
        ({"weighting": "equal"}, "weighting"),
        ({"weighting": "inverse_vol"}, "weighting"),
        ({"schedule": _DROP}, "schedule"),
        ({"schedule": "weekly"}, "schedule"),
        ({"direction": _DROP}, "direction"),
        ({"direction": "long"}, "direction"),
        ({"vol_target": _DROP}, "vol_target"),
        ({"vol_target": 0}, "vol_target"),
        ({"vol_target": 0.0}, "vol_target"),
        ({"vol_target": 2.0001}, "vol_target"),
        ({"vol_target": float("nan")}, "vol_target"),
        ({"vol_target": float("inf")}, "vol_target"),
        ({"vol_target": True}, "vol_target"),
        ({"vol_target": 10**400}, "vol_target"),
        ({"vol_target": "0.4"}, "vol_target"),
    ],
)
def test_trend_requires_its_fields(over: dict[str, Any], field: str) -> None:
    """R6: a trend spec without (or with a bad) required field names that field."""
    with pytest.raises(ValueError, match=field):
        _trend(**over)


def test_trend_vol_target_edge() -> None:
    """R6 edge: vol_target 2.0 is accepted (inclusive), stored as float."""
    spec = _trend(vol_target=2)
    assert spec.vol_target == 2.0
    assert isinstance(spec.vol_target, float)


@pytest.mark.parametrize(
    ("over", "field"),
    [
        ({"schedule": "daily"}, "schedule"),
        ({"weighting": "time_series"}, "weighting"),
        ({"fill_granularity": "1h"}, "fill_granularity"),
        ({"direction": "long_flat"}, "direction"),
        ({"vol_target": 0.4}, "vol_target"),
        ({"trend_filter": "BTCUSDT"}, "trend_filter"),
        ({"spread": 0.0001}, "spread"),
        ({"spread": 0.01}, "spread"),
    ],
)
def test_funding_refuses_trend_fields(over: dict[str, Any], field: str) -> None:
    """R6/R7/C9-11: funding with a trend-only field (incl. a valid spread) is refused."""
    with pytest.raises(ValueError, match=field):
        _funding(**over)
    with pytest.raises(ValueError, match=field):
        _funding(liquidation_mode="cross", **over)


@pytest.mark.parametrize(
    ("over", "field"),
    [
        ({"signal": "momentum"}, "signal"),
        ({"schedule": "hourly"}, "schedule"),
        ({"fill_granularity": "5m"}, "fill_granularity"),
        ({"allow_gross_above_one": 1}, "allow_gross_above_one"),
    ],
)
def test_unknown_enum_values(over: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _funding(**over)


# --- R7: trend refuses cross liquidation and the volume filter; spread range ---


@pytest.mark.parametrize(
    ("over", "field"),
    [
        ({"liquidation_mode": "cross"}, "liquidation_mode"),
        ({"min_quote_volume": 1_000_000.0}, "min_quote_volume"),
        ({"spread": float("nan")}, "spread"),
        ({"spread": -1e-9}, "spread"),
        ({"spread": 0.0101}, "spread"),
        ({"spread": "1"}, "spread"),
        ({"spread": True}, "spread"),
    ],
)
def test_trend_refuses_bad_combinations(over: dict[str, Any], field: str) -> None:
    """R7: ValueError at construction (before any load) naming the field."""
    with pytest.raises(ValueError, match=field):
        _trend(**over)


def test_spread_edges_and_storage() -> None:
    """R7 edges: 0.0 and 0.01 are accepted; an int spread is stored as float."""
    assert _trend(spread=0).spread == 0.0
    assert isinstance(_trend(spread=0).spread, float)
    assert _trend(spread=0.01).spread == 0.01
    assert _trend(fill_granularity="1h").fill_granularity == "1h"
    assert _funding(spread=0.0).spread == 0.0


# --- R8: the regime filter and the trend vol lookback ---


@pytest.mark.parametrize(
    ("over", "field"),
    [
        ({"trend_filter": "BTCUSDT", "direction": "long_short"}, "trend_filter"),
        ({"trend_filter": "SOLUSDT"}, "trend_filter"),
        ({"trend_filter": ""}, "trend_filter"),
        ({"vol_lookback_days": 1}, "vol_lookback_days"),
        ({"vol_lookback_days": 160}, "vol_lookback_days"),
        ({"vol_lookback_days": True}, "vol_lookback_days"),
        ({"vol_lookback_days": 30.0}, "vol_lookback_days"),
    ],
)
def test_trend_filter_and_lookback(over: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _trend(**over)


def test_trend_filter_and_lookback_edges() -> None:
    """R8 edges: lookback 2 and 159 accepted; a filter in symbols with long_flat accepted."""
    assert _trend(vol_lookback_days=2).vol_lookback_days == 2
    assert _trend(vol_lookback_days=159).vol_lookback_days == 159
    assert _trend(trend_filter="BTCUSDT").trend_filter == "BTCUSDT"


# --- R9: gross above one is an explicit opt-in ---


def test_gross_above_one_needs_opt_in() -> None:
    with pytest.raises(ValueError) as caught:
        _funding(gross=1.5)
    assert (
        str(caught.value) == "gross > 1.0 needs allow_gross_above_one (liquidation is not modelled)"
    )
    with pytest.raises(ValueError, match="allow_gross_above_one"):
        _trend(gross=1.0000000000000002)
    assert _funding(gross=1.5, allow_gross_above_one=True).gross == 1.5
    assert _trend(gross=1.5, allow_gross_above_one=True).gross == 1.5
    assert _funding(gross=1.0).gross == 1.0  # the equality edge needs no opt-in
    with pytest.raises(ValueError, match="gross"):
        _funding(gross=2.0000001, allow_gross_above_one=True)


def test_opt_in_is_never_hashed() -> None:
    """R9: allow_gross_above_one changes neither to_json nor the hashes."""
    plain, opted = _funding(), _funding(allow_gross_above_one=True)
    assert plain.to_json() == opted.to_json()
    assert opted.spec_hash() == _DEFAULT_HASH
    assert _trend().arm_hash() == _trend(allow_gross_above_one=True).arm_hash()


# --- R10: symbol counts ---


def test_symbol_counts() -> None:
    assert _trend(symbols=("BTCUSDT",)).symbols == ("BTCUSDT",)
    with pytest.raises(ValueError, match="at least 2 distinct symbols"):
        CarrySpec.from_dates(("BTCUSDT",), _START, _END)
    with pytest.raises(ValueError, match="symbols"):
        _trend(symbols=())


# --- R11: the daily 01:00 UTC schedule ---


def test_daily_schedule() -> None:
    start, end = _trend().start_ns, _trend().end_ns
    daily = decision_times(start, end, "daily")
    assert len(daily) == 1580
    assert daily[0] == 1_654_045_200_000_000_000  # 2022-06-01T01:00:00Z
    assert daily[-1] == 1_790_470_800_000_000_000  # 2026-09-27T01:00:00Z
    assert all(b - a == _DAY_NS for a, b in zip(daily, daily[1:], strict=False))
    weekly = decision_times(start, end, "weekly")
    assert weekly == decision_times(start, end)
    assert len(weekly) == 225
    assert weekly[0] == 1_654_473_900_000_000_000


def test_daily_schedule_cutoff_edges() -> None:
    """The last D is <= end - 61 min; a start past 01:00 begins the next day."""
    start = 1_710_115_200_000_000_000  # 2024-03-11T00:00Z
    end = start + _DAY_NS
    assert decision_times(start, end, "daily") == [start + _HOUR_NS]
    assert decision_times(start + _HOUR_NS, end + _DAY_NS, "daily") == [
        start + _HOUR_NS,
        start + _DAY_NS + _HOUR_NS,
    ]
    assert decision_times(start + _HOUR_NS + 1, end + _DAY_NS, "daily") == [
        start + _DAY_NS + _HOUR_NS
    ]
    with pytest.raises(ValueError, match="schedule"):
        decision_times(start, end, "hourly")


# --- R12 + C9-7: serialization and the arm hash ---


def test_trend_to_json_carries_every_economic_input() -> None:
    payload = _trend(vol_lookback_days=45, spread=0.0001, trend_filter="BTCUSDT").to_json()
    assert list(payload)[:7] == [
        "symbols",
        "start_ns",
        "end_ns",
        "initial_balance",
        "legs_divisor",
        "gross",
        "band",
    ]
    assert {key: payload[key] for key in list(payload)[7:]} == {
        "signal": "trend",
        "schedule": "daily",
        "fill_granularity": "1m",
        "weighting": "time_series",
        "direction": "long_flat",
        "trend_filter": "BTCUSDT",
        "vol_target": 0.4,
        "vol_lookback_days": 45,
        "trend_lookbacks_days": [10, 20, 40, 80, 160],
        "spread": 0.0001,
    }
    assert TREND_LOOKBACKS_DAYS == (10, 20, 40, 80, 160)


def test_legacy_serialization_kept() -> None:
    """C9-7: funding specs keep their exact keys and hashes (default and inverse_vol)."""
    default = _funding()
    assert len(default.to_json()) == 7
    assert default.spec_hash() == _DEFAULT_HASH
    inverse = _funding(weighting="inverse_vol", vol_lookback_days=60)
    assert list(inverse.to_json())[7:] == ["weighting", "vol_lookback_days"]
    expected = {key: value for key, value in default.to_json().items() if key not in _ARM_OMIT}
    assert (
        default.arm_hash()
        == hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    )


_ARM_OMIT = ("symbols", "start_ns", "end_ns", "initial_balance", "fill_granularity")


def test_arm_hash_ignores_window_cohort_tier() -> None:
    """R12: symbols, window, E0 and tier do not change the arm."""
    base = _trend()
    others = (
        CarrySpec.from_dates(
            ("ETHUSDT", "SOLUSDT", "XRPUSDT"), "2023-01-01", "2024-01-01", **_trend_kw()
        ),
        _trend(initial_balance=99_000.0),
        _trend(fill_granularity="1h"),
    )
    for other in others:
        assert other.arm_hash() == base.arm_hash()
        assert other.spec_hash() != base.spec_hash()
    assert len(base.arm_hash()) == 64


@pytest.mark.parametrize(
    "over",
    [
        {"vol_target": 0.41},
        {"vol_lookback_days": 31},
        {"direction": "long_short"},
        {"trend_filter": "BTCUSDT"},
        {"spread": 0.0002},
        {"gross": 0.9},
        {"band": 0.2},
        {"legs_divisor": 4},
    ],
)
def test_arm_hash_sees_every_economic_input(over: dict[str, Any]) -> None:
    """R12/C9-7: each economic arm input (incl. the trend vol lookback) moves the arm hash."""
    assert _trend(**over).arm_hash() != _trend().arm_hash()


def test_trend_and_funding_arms_differ() -> None:
    assert _trend().arm_hash() != CarrySpec.from_dates(_SYMBOLS, _START, _END).arm_hash()


# --- C9-7: the frozen family N stays inside the finite DSR quantile domain ---


def test_family_num_trials() -> None:
    assert family_num_trials(4, 3) == 7
    assert family_num_trials(0, 1) == 1
    top = 6_627_126_856_707_895  # the largest N whose DSR quantile is finite
    assert family_num_trials(top - 1, 1) == top
    for n_prior, arms in (
        (top, 1),
        (10**400, 3),
        (-1, 3),
        (True, 3),
        (4.0, 3),
        (4, 0),
        (4, False),
        (4, "3"),
    ):
        with pytest.raises(ValueError):
            family_num_trials(n_prior, arms)  # type: ignore[arg-type]
