"""Funding-carry signal rules (epic vibe-quant-x0axf, task T1).

Contract rows R1-R8c of the funding-carry semantics contract
(``data/swarm/20261015-funding-carry/design.md`` §2 and §7). The window is
``(L - 7 d, L]`` with ``L = D - 60 s``; the exact float literals below are the
contract's hand-computed sums and are the point of the tests.
"""

from __future__ import annotations

import pytest

from vibe_quant.portfolio.signal import (
    Settlements,
    carry_hours,
    rank_weights,
    trailing_funding,
)

_NANOS_PER_S = 10**9
_DAY_NS = 86_400 * _NANOS_PER_S
_EIGHT_HOURS_NS = 8 * 3_600 * _NANOS_PER_S
_FOUR_HOURS_NS = 4 * 3_600 * _NANOS_PER_S

_D = 1_654_473_900 * _NANOS_PER_S  # Monday 2022-06-06 00:05:00 UTC
_L = _D - 60 * _NANOS_PER_S
_LO = _L - 7 * _DAY_NS

_ON_GRID_SUM = 0.0021000000000000003


def _grid() -> list[int]:
    """The 21 8h UTC boundaries inside the signal window ``(L-7d, L]``."""
    first = (_LO // _EIGHT_HOURS_NS + 1) * _EIGHT_HOURS_NS
    return [first + i * _EIGHT_HOURS_NS for i in range(21)]


def _four_hour_times() -> list[int]:
    """The 42 4h settlement times inside the signal window."""
    first = (_LO // _FOUR_HOURS_NS + 1) * _FOUR_HOURS_NS
    return [first + i * _FOUR_HOURS_NS for i in range(42)]


def _settlements(times: list[int], rates: list[float]) -> Settlements:
    return Settlements(times_ns=tuple(times), rates=tuple(rates))


def test_settlement_at_safety_edge_is_included() -> None:
    """R1: a row exactly at ``L`` is in the window, at ``L+1ns`` it is out."""
    grid = _grid()
    at_l = _settlements(grid + [_L], [0.0001] * 21 + [0.001])
    assert trailing_funding(at_l, _D) == 0.0031000000000000003
    after_l = _settlements(grid + [_L + 1], [0.0001] * 21 + [0.001])
    assert trailing_funding(after_l, _D) == _ON_GRID_SUM


def test_settlement_at_lookback_edge_is_excluded() -> None:
    """R2: a row exactly at ``L-7d`` is out (open interval), at ``+1ns`` in."""
    grid = _grid()
    at_lo = _settlements([_LO, *grid], [0.0001] * 22)
    assert trailing_funding(at_lo, _D) == _ON_GRID_SUM
    after_lo = _settlements([_LO + 1, *grid], [0.0001] * 22)
    assert trailing_funding(after_lo, _D) == 0.0022


def test_uncovered_8h_boundary_excludes_symbol() -> None:
    """R3: every 8h boundary needs a row within 60 s, else ``funding_hole``."""
    grid = _grid()
    boundary = grid[5]
    hole = _settlements([*grid[:5], boundary + 61 * _NANOS_PER_S, *grid[6:]], [0.0001] * 21)
    assert trailing_funding(hole, _D) is None
    covered = _settlements([*grid[:5], boundary + 3_000_000, *grid[6:]], [0.0001] * 21)
    assert trailing_funding(covered, _D) == _ON_GRID_SUM


def test_8h_boundary_row_at_upper_safety_edge() -> None:
    """R3: a row exactly at ``b + 60 s`` covers the boundary, ``b + 61 s`` is a hole."""
    grid = _grid()
    boundary = grid[5]
    at_edge = _settlements([*grid[:5], boundary + 60 * _NANOS_PER_S, *grid[6:]], [0.0001] * 21)
    assert trailing_funding(at_edge, _D) == _ON_GRID_SUM
    beyond = _settlements([*grid[:5], boundary + 61 * _NANOS_PER_S, *grid[6:]], [0.0001] * 21)
    assert trailing_funding(beyond, _D) is None


def test_8h_boundary_row_at_lower_safety_edge() -> None:
    """R3: a row exactly at ``b - 60 s`` covers, ``b - 60 s - 1 ns`` is a hole."""
    grid = _grid()
    boundary = grid[5]
    at_edge = _settlements([*grid[:5], boundary - 60 * _NANOS_PER_S, *grid[6:]], [0.0001] * 21)
    assert trailing_funding(at_edge, _D) == _ON_GRID_SUM
    beyond = _settlements([*grid[:5], boundary - 60 * _NANOS_PER_S - 1, *grid[6:]], [0.0001] * 21)
    assert trailing_funding(beyond, _D) is None


def test_4h_and_8h_sums_tie_exactly() -> None:
    """R4: ``/(8/N)``-scaled intervals sum to the same total, tie -> key order."""
    grid = _grid()
    eight_hour = _settlements(grid, [0.0001] * 21)
    four_hour = _settlements(_four_hour_times(), [0.00005] * 42)
    signal_8h = trailing_funding(eight_hour, _D)
    signal_4h = trailing_funding(four_hour, _D)
    assert signal_8h == _ON_GRID_SUM
    assert signal_4h == _ON_GRID_SUM
    assert signal_8h == signal_4h
    weights = rank_weights({"BTCUSDT": signal_8h, "ETHUSDT": signal_4h}, 2, 1.0)
    assert weights == {"BTCUSDT": 0.5, "ETHUSDT": -0.5}


def test_missing_4h_row_is_not_detected() -> None:
    """R5: a missing NON-8h row is summed as archived (accepted limitation)."""
    times = _four_hour_times()
    del times[0]  # the 04:00 row: not an 8h boundary
    four_hour = _settlements(times, [0.00005] * 41)
    assert trailing_funding(four_hour, _D) == 0.00205


def test_rank_long_lowest_short_highest() -> None:
    """R6: total order by ``(signal, symbol)``; k = n // divisor legs a side."""
    weights = rank_weights(
        {
            "BTCUSDT": 0.0021,
            "ETHUSDT": 0.0030,
            "SOLUSDT": -0.0010,
            "BNBUSDT": 0.0005,
        },
        3,
        1.0,
    )
    assert weights == {
        "BTCUSDT": 0.0,
        "ETHUSDT": -0.5,
        "SOLUSDT": 0.5,
        "BNBUSDT": 0.0,
    }


def test_tie_broken_by_symbol() -> None:
    """R7: a tie on the signal is broken by the symbol, tying legs stay ties."""
    weights = rank_weights({"BTCUSDT": 0.0021, "ETHUSDT": 0.0021, "SOLUSDT": -0.0010}, 3, 1.0)
    assert weights["SOLUSDT"] == 0.5
    assert weights["ETHUSDT"] == -0.5
    assert weights["BTCUSDT"] == 0.0


def test_leg_count_edges() -> None:
    """R8/R8b/R8c: k = max(1, n // divisor) for n >= 2, flat for n < 2."""
    for n in (4, 3, 2):
        signals = {f"S{i}USDT": float(i) for i in range(n)}
        weights = rank_weights(signals, 3, 1.0)
        assert sum(1 for w in weights.values() if w > 0) == 1
        assert sum(1 for w in weights.values() if w < 0) == 1
    middle = rank_weights({"AUSDT": 1.0, "BUSDT": 2.0, "CUSDT": 3.0}, 3, 1.0)
    assert middle == {"AUSDT": 0.5, "BUSDT": 0.0, "CUSDT": -0.5}
    # R8b: n = 1 or 0 -> flat
    assert rank_weights({"AUSDT": 1.0}, 3, 1.0) == {"AUSDT": 0.0}
    assert rank_weights({}, 3, 1.0) == {}
    # R8c: n = 6 div 3 and n = 5 div 2 both give k = 2 -> +-0.25
    six = rank_weights({f"S{i}USDT": float(i) for i in range(6)}, 3, 1.0)
    assert sorted(w for w in six.values() if w) == [-0.25, -0.25, 0.25, 0.25]
    five = rank_weights({f"S{i}USDT": float(i) for i in range(5)}, 2, 1.0)
    assert sorted(w for w in five.values() if w) == [-0.25, -0.25, 0.25, 0.25]


def test_unrankable_symbols_are_not_ranked() -> None:
    """A ``None`` signal (funding hole / no price) is flat and out of the order."""
    weights = rank_weights({"AUSDT": 0.002, "BUSDT": None, "CUSDT": -0.002}, 3, 1.0)
    assert weights == {"AUSDT": -0.5, "BUSDT": 0.0, "CUSDT": 0.5}


def test_4h_symbol_missing_row_is_unrankable() -> None:
    """R34: a missing 4h row in the window is ``None``; with hours=None it is summed."""
    times = _four_hour_times()
    del times[0]  # the 04:00 row: not an 8h boundary, but 4h on this symbol
    rates = [0.00005] * len(times)
    four_hour = Settlements(times_ns=tuple(times), rates=tuple(rates), hours=(4,) * len(times))

    assert trailing_funding(four_hour, _D) is None

    # the same rows under the default 8h rule: summed as archived, the blind spot
    assert trailing_funding(Settlements(tuple(times), tuple(rates)), _D) == 0.00205


def test_settlements_hours_default_and_mismatch() -> None:
    """R35: ``hours`` defaults to None (8h rule exactly as today); mismatch raises."""
    grid = _grid()
    default = _settlements(grid, [0.0001] * 21)

    assert default.hours is None
    assert trailing_funding(default, _D) == _ON_GRID_SUM
    assert (
        trailing_funding(Settlements(tuple(grid), (0.0001,) * 21, (8,) * 21), _D) == _ON_GRID_SUM
    )
    with pytest.raises(ValueError, match=r"^Settlements\.hours has 3 entries for 4 settlements$"):
        Settlements(tuple(grid[:4]), (0.0001,) * 4, (2, 2, 2))


def test_carry_hours_collapses_all_8h() -> None:
    """R35b: all-8h (or empty) hours collapse to None; a real mix survives."""
    assert carry_hours(()) is None
    assert carry_hours((8, 8, 8)) is None
    assert carry_hours((8, 2, 8)) == (8, 2, 8)
    assert carry_hours((2, 2)) == (2, 2)
