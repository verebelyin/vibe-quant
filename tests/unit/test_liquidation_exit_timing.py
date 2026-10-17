"""1m refinement of boundary resting exits (V1a, bead vibe-quant-vjno2).

Pure functions; every case cites its contract row (R20-R32, design r2) and carries
hand-computed timestamps. The R22 fixture is the reference 4h strategy bar
2024-04-17 12:00 UTC: ``ts_init = close - 1 ms = 15:59:59.999``, minute ``i`` has
``ts_init = bar_open + (i + 1) min - 1 ms``.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibe_quant.liquidation.exit_timing import (
    MissingExitBars,
    RestingExit,
    refine_resting_exits,
)
from vibe_quant.metrics import BarSeries

_MIN = 60_000_000_000
_FOUR_H = 14_400_000_000_000  # 240 minutes: the execution bar of the R22 fixture
_BAR_OPEN = 1_713_355_200_000_000_000  # 2024-04-17 12:00 UTC
_FILL = 1_713_369_599_999_000_000  # 15:59:59.999, the boundary fill / the bar's ts_init


def _ts(minute: int) -> int:
    """``ts_init`` of the 1m bar opening ``minute`` minutes after ``_BAR_OPEN``."""
    return _BAR_OPEN + minute * _MIN + (_MIN - 1_000_000)


def _series(bar_open: int, lows: list[float], highs: list[float] | None = None) -> BarSeries:
    """1m bars opening at ``bar_open``; ``open``/``close`` are unused by the trigger."""
    n = len(lows)
    ts = np.array([bar_open + i * _MIN + (_MIN - 1_000_000) for i in range(n)], dtype=np.int64)
    low = np.asarray(lows, dtype=np.float64)
    high = low + 1.0 if highs is None else np.asarray(highs, dtype=np.float64)
    return BarSeries(ts=ts, open=low, high=high, low=low, close=low)


def _exit(**over: object) -> RestingExit:
    """A falling SELL stop at 100, live before the bar, at the R22 boundary fill."""
    base: dict[str, object] = {
        "key": "p@0",
        "fill_index": 0,
        "order_id": "o-0",
        "symbol": "BTCUSDT",
        "ts": _FILL,
        "kind": "stop",
        "falls": True,
        "level": 100.0,
        "active_from": _BAR_OPEN - 1,
    }
    return RestingExit(**{**base, **over})  # type: ignore[arg-type]


def test_stop_refines_to_first_1m_bar_reaching_the_level():
    """R22: 12:00 bar, NT fill 15:59:59.999, falling STOP at 100.

    Lows are 101 for minutes 0-138, exactly 100 at minute 139 (14:19), 99 at minute 140.
    Live before the bar, the first bar reaching the level is minute 139 ->
    1713363599999000000. With ``active_from`` at minute 139's ts_init the minute itself is
    excluded (R21), so minute 140 -> 1713363659999000000.
    """
    lows = [101.0] * 139 + [100.0] + [99.0] * (240 - 140)
    bars = {"BTCUSDT": _series(_BAR_OPEN, lows)}

    timing = refine_resting_exits([_exit()], bars, _FOUR_H)
    assert timing.scan_ts[("p@0", 0)] == 1_713_363_599_999_000_000
    assert timing.refined == 1
    assert timing.kept_bar_close == 0

    live_at_139 = refine_resting_exits([_exit(active_from=1_713_363_599_999_000_000)], bars, _FOUR_H)
    assert live_at_139.scan_ts[("p@0", 0)] == 1_713_363_659_999_000_000


def test_stop_refines_on_equality_for_both_directions():
    """R20: SELL stop low <= level, BUY stop high >= level (equality counts)."""
    sell_bars = {"BTCUSDT": _series(_BAR_OPEN, [101.0, 100.0] + [101.0] * 238)}
    sell = refine_resting_exits([_exit(falls=True)], sell_bars, _FOUR_H)
    assert sell.scan_ts[("p@0", 0)] == _ts(1)

    buy_bars = {
        "BTCUSDT": _series(
            _BAR_OPEN, [99.0, 99.5] + [99.5] * 238, highs=[99.5, 100.0] + [100.0] * 238
        )
    }
    buy = refine_resting_exits([_exit(falls=False)], buy_bars, _FOUR_H)
    assert buy.scan_ts[("p@0", 0)] == _ts(1)


def test_limits_need_strict_penetration_not_a_touch():
    """R20b probe: SELL LIMIT 105 rejects high 105.0, refines on the next minute 105.1."""
    sell_bars = {
        "BTCUSDT": _series(
            _BAR_OPEN, [104.0, 104.5, 105.0] + [105.5] * 237, highs=[105.0, 105.0, 105.1] + [106.0] * 237
        )
    }
    sell = refine_resting_exits([_exit(kind="limit", falls=False, level=105.0)], sell_bars, _FOUR_H)
    assert sell.scan_ts[("p@0", 0)] == _ts(2)

    buy_bars = {
        "BTCUSDT": _series(
            _BAR_OPEN, [106.0, 105.0, 104.9] + [104.0] * 237, highs=[107.0, 106.0, 105.5] + [105.0] * 237
        )
    }
    buy = refine_resting_exits([_exit(kind="limit", falls=True, level=105.0)], buy_bars, _FOUR_H)
    assert buy.scan_ts[("p@0", 0)] == _ts(2)


def test_candidate_window_is_the_execution_bar_and_strictly_after_active_from():
    """R21: opens in [bar_open, bar_open + E) and strictly after active_from."""
    # the hit is in the NEXT strategy bar's first minute -> outside the window -> R25
    next_bar = {"BTCUSDT": _series(_BAR_OPEN, [101.0] * 240 + [100.0])}
    outside = refine_resting_exits([_exit()], next_bar, _FOUR_H)
    assert outside.scan_ts[("p@0", 0)] == _FILL
    assert outside.kept_bar_close == 1

    # the last in-window minute refines to the bar's own ts_init (r == n)
    last_minute = {"BTCUSDT": _series(_BAR_OPEN, [101.0] * 239 + [100.0])}
    at_close = refine_resting_exits([_exit()], last_minute, _FOUR_H)
    assert at_close.scan_ts[("p@0", 0)] == _FILL
    assert at_close.refined == 1

    # minute 2 reaches the level but the order is only live from minute 2's ts_init
    touch_early = {"BTCUSDT": _series(_BAR_OPEN, [101.0, 101.0, 100.0, 100.0] + [101.0] * 236)}
    from_minute_2 = refine_resting_exits([_exit(active_from=_ts(2))], touch_early, _FOUR_H)
    assert from_minute_2.scan_ts[("p@0", 0)] == _ts(3)
    from_minute_1 = refine_resting_exits([_exit(active_from=_ts(1))], touch_early, _FOUR_H)
    assert from_minute_1.scan_ts[("p@0", 0)] == _ts(2)


def test_no_candidate_keeps_the_nt_timestamp_and_counts_kept_bar_close():
    """R25: a stop the 1m series never reaches keeps the NT ts and is counted."""
    bars = {"BTCUSDT": _series(_BAR_OPEN, [200.0] * 240)}
    timing = refine_resting_exits([_exit()], bars, _FOUR_H)
    assert timing.scan_ts[("p@0", 0)] == _FILL
    assert timing.refined == 0
    assert timing.kept_bar_close == 1


def test_to_notes_has_the_r25b_shape_including_by_symbol():
    """R25b: ``exit_timing`` notes are counts plus a per-symbol breakdown."""
    bars = {
        "BTCUSDT": _series(_BAR_OPEN, [100.0] * 240),
        "ETHUSDT": _series(_BAR_OPEN, [200.0] * 240),
    }
    exits = [
        _exit(key="a@0", symbol="BTCUSDT"),
        _exit(key="b@0", symbol="ETHUSDT"),
    ]
    timing = refine_resting_exits(exits, bars, _FOUR_H)
    assert timing.to_notes() == {
        "refined": 1,
        "kept_bar_close": 1,
        "by_symbol": {
            "BTCUSDT": {"refined": 1, "kept_bar_close": 0},
            "ETHUSDT": {"refined": 0, "kept_bar_close": 1},
        },
    }


def test_symbol_without_a_series_raises_and_an_empty_window_is_kept():
    """R26: no 1m series -> MissingExitBars; a series with an empty window -> R25."""
    with pytest.raises(MissingExitBars, match="no 1m LAST series"):
        refine_resting_exits([_exit()], {"ETHUSDT": _series(_BAR_OPEN, [1.0])}, _FOUR_H)

    elsewhere = {"BTCUSDT": _series(_BAR_OPEN - 400 * _MIN, [100.0] * 240)}
    timing = refine_resting_exits([_exit()], elsewhere, _FOUR_H)
    assert timing.kept_bar_close == 1
    assert timing.scan_ts[("p@0", 0)] == _FILL


def test_fills_of_one_order_at_one_nt_ts_share_one_refinement():
    """R28/R28b: fills of the SAME order at one NT ts share r; r <= n always."""
    bars = {"BTCUSDT": _series(_BAR_OPEN, [100.0, 50.0] + [200.0] * 238)}
    pair = [
        _exit(fill_index=0, order_id="A", level=100.0),  # minute 0
        _exit(fill_index=1, order_id="A", level=50.0),  # minute 1, same order
    ]
    timing = refine_resting_exits(pair, bars, _FOUR_H)
    assert timing.scan_ts[("p@0", 0)] == _ts(0)
    assert timing.scan_ts[("p@0", 1)] == _ts(0)
    assert all(r <= _FILL for r in timing.scan_ts.values())


def test_distinct_orders_at_one_nt_ts_refine_independently():
    """R28b (finding 1): distinct orders at the same (key, ts) refine independently.

    A SELL stop at 100 and one at 50, both filled at the boundary NT ts, reach their
    levels in different minutes; a cache keyed by position and NT ts alone assigned both
    the first order's timestamp.
    """
    bars = {"BTCUSDT": _series(_BAR_OPEN, [100.0, 50.0] + [200.0] * 238)}
    exits = [
        _exit(fill_index=0, order_id="A", level=100.0),  # minute 0 -> _ts(0)
        _exit(fill_index=1, order_id="B", level=50.0),  # minute 1 -> _ts(1)
    ]
    timing = refine_resting_exits(exits, bars, _FOUR_H)
    assert timing.scan_ts[("p@0", 0)] == _ts(0)
    assert timing.scan_ts[("p@0", 1)] == _ts(1)
    assert timing.refined == 2
    assert timing.kept_bar_close == 0


def test_a_distinct_order_refines_on_its_own_active_from():
    """R28b/R25 (finding 1): the second order's active_from is honoured, not inherited.

    Order B reaches its level in minute 1 but is only live from that minute's ts_init, so
    it has no candidate and keeps its NT timestamp instead of inheriting the first
    order's refinement.
    """
    bars = {"BTCUSDT": _series(_BAR_OPEN, [100.0, 50.0] + [200.0] * 238)}
    exits = [
        _exit(fill_index=0, order_id="A", level=100.0),
        _exit(fill_index=1, order_id="B", level=50.0, active_from=_ts(1)),
    ]
    timing = refine_resting_exits(exits, bars, _FOUR_H)
    assert timing.scan_ts[("p@0", 0)] == _ts(0)
    assert timing.scan_ts[("p@0", 1)] == _FILL
    assert timing.refined == 1
    assert timing.kept_bar_close == 1


def test_distinct_orders_at_one_nt_ts_must_advance_or_raise():
    """R28/R28b: r strictly increases across fills unless one order shares its NT ts."""
    bars = {"BTCUSDT": _series(_BAR_OPEN, [100.0, 50.0] + [200.0] * 238)}
    # order A reaches its level at minute 1, order B at minute 0 -> r goes backwards
    backwards_orders = [
        _exit(fill_index=0, order_id="A", level=50.0),
        _exit(fill_index=1, order_id="B", level=100.0),
    ]
    with pytest.raises(ValueError, match="not after the previous"):
        refine_resting_exits(backwards_orders, bars, _FOUR_H)

    # a later fill_index at an EARLIER NT ts would drag the scan ts backwards
    wide = {"BTCUSDT": _series(_BAR_OPEN - 300 * _MIN, [100.0] * 600)}
    backwards = [
        _exit(fill_index=0, ts=_FILL),
        _exit(fill_index=1, ts=_FILL - _FOUR_H),
    ]
    with pytest.raises(ValueError, match="not after the previous"):
        refine_resting_exits(backwards, wide, _FOUR_H)


def test_distinct_orders_refining_to_the_same_ts_raise():
    """R28/R28b: distinct orders must refine STRICTLY after; equal refined ts raises.

    Orders A and B both reach their level in minute 0, so their refined timestamps are
    equal. The shared-timestamp exemption belongs to fills of ONE order only (R28b), so
    equality across DISTINCT orders breaks the strict ordering of R28. Weakening the
    production ``r <= prev`` guard to ``r < prev`` lets this pair through.
    """
    bars = {"BTCUSDT": _series(_BAR_OPEN, [100.0] * 240)}
    equal_orders = [
        _exit(fill_index=0, order_id="A", level=100.0),  # minute 0 -> _ts(0)
        _exit(fill_index=1, order_id="B", level=100.0),  # minute 0 -> _ts(0), equal
    ]
    with pytest.raises(ValueError, match="not after the previous"):
        refine_resting_exits(equal_orders, bars, _FOUR_H)


def test_the_minute_before_the_execution_bar_is_not_a_candidate():
    """R21/R25 (finding 2): only the minute PRECEDING the bar reaches the level.

    The 1m bar opening at ``bar_open - 60 s`` closes at ``bar_open - 1 ms``, before the
    execution bar opens. The order is live from well before it, so only the R21 window
    bound can exclude it: the fill keeps its NT timestamp. A window that leaks one minute
    backwards would refine to that bar instead.
    """
    bars = {"BTCUSDT": _series(_BAR_OPEN - _MIN, [100.0] + [101.0] * 240)}
    timing = refine_resting_exits([_exit(active_from=_BAR_OPEN - 2 * _MIN)], bars, _FOUR_H)
    assert timing.scan_ts[("p@0", 0)] == _FILL
    assert timing.refined == 0
    assert timing.kept_bar_close == 1


def test_a_bar_opening_past_the_level_is_a_candidate():
    """R32: a bar already past the level counts when live before the bar, else R25."""
    stopped = {"BTCUSDT": _series(_BAR_OPEN, [95.0] + [101.0] * 239)}
    timing = refine_resting_exits([_exit()], stopped, _FOUR_H)
    assert timing.scan_ts[("p@0", 0)] == _ts(0)

    # the same gap is not a candidate once the order is only live AT the bar open
    late = refine_resting_exits([_exit(active_from=_BAR_OPEN)], stopped, _FOUR_H)
    assert late.kept_bar_close == 1
    assert late.scan_ts[("p@0", 0)] == _FILL

    gapped = {
        "BTCUSDT": _series(_BAR_OPEN, [104.0] + [98.0] * 239, highs=[105.1] + [99.0] * 239)
    }
    limit = refine_resting_exits([_exit(kind="limit", falls=False, level=105.0)], gapped, _FOUR_H)
    assert limit.scan_ts[("p@0", 0)] == _ts(0)


def test_missing_exit_bars_is_data_unavailable():
    """R18: ``MissingExitBars`` is re-parented to ``DataUnavailableError`` (still a
    ``ValueError``), so a MODE-OFF screening run that hits it fails closed instead of scoring a
    silent ``-inf`` row. The mode-on wrapping text is unchanged (see
    ``test_screening_liquidation_exit_timing``)."""
    from vibe_quant.errors import DataUnavailableError

    assert issubclass(MissingExitBars, DataUnavailableError)
    assert issubclass(MissingExitBars, ValueError)
    assert str(MissingExitBars("BTCUSDT: no 1m LAST series")) == "BTCUSDT: no 1m LAST series"

