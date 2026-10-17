"""Funding holes on each symbol's own settlement interval (bd vibe-quant-x0axf.13, T4).

Contract B rows R19, R19b and R21-R33 of the T4 brief. The rule under test:
archived rows are always charged as archived, and the hole rule only ADDS
fallback payments for settlements the symbol's own Binance Vision
``funding_interval_hours`` expects in ``(lo, hi]`` but the archive has no row
within 60 s of. ``H = max(h_prev, h_next)`` between two archived rows (never
``min``, which flags 4 false holes at the SOL 2022-11 switch); a missing
settlement pays ``default_rate * (H / 8)``. With all-8h intervals the rule is
exactly the pre-T4 fixed 8h UTC grid -- pinned by the copied old loop below.

Every comparison on a money path is pinned AT its equality edge (R30-R30d: a row
at ``B + 60 s`` covers, ``B + 60.001 s`` does not; likewise at ``B - 60 s``).
"""

from __future__ import annotations

import logging
import random
from bisect import bisect_left
from typing import TYPE_CHECKING

import pytest

from vibe_quant.data.archive import RawDataArchive
from vibe_quant.validation.funding import (
    DEFAULT_FUNDING_RATE_PER_PERIOD,
    FundingCalculator,
    clear_rate_cache,
    missing_settlements,
)

if TYPE_CHECKING:
    from pathlib import Path

SYMBOL = "SOLUSDT"
INSTRUMENT = "SOLUSDT-PERP.BINANCE"
MS_PER_HOUR = 3_600_000
HOUR_NS = 3_600 * 1_000_000_000
TOL_NS = 60 * 1_000_000_000
T0_MS = 1_668_038_400_000  # 2022-11-10T00:00:00Z
T0_NS = T0_MS * 1_000_000
RATE = 0.0001
NOTIONAL = 1500.0


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_rate_cache()


def _rows(base_ms: int, offsets_h: list[int], jitter_ms: int = 0) -> list[int]:
    """Archived funding times (ms) at ``offsets_h`` hours from ``base_ms``."""
    return [base_ms + h * MS_PER_HOUR + jitter_ms for h in offsets_h]


def _ns(times_ms: list[int]) -> list[int]:
    return [t * 1_000_000 for t in times_ms]


def _archive(
    path: Path,
    times_ms: list[int],
    hours: dict[int, int] | None = None,
    symbol: str = SYMBOL,
) -> FundingCalculator:
    """A tmp archive with one rate row per ``times_ms`` and optional Vision intervals."""
    archive = RawDataArchive(path)
    archive.insert_funding_rates(symbol, [(t, RATE) for t in times_ms], "test")
    if hours:
        archive.insert_funding_intervals(symbol, sorted(hours.items()), "test")
    archive.close()
    return FundingCalculator(path)


def _old_grid_missing(times: list[int], lo_ns: int, hi_ns: int, tol_ns: int = TOL_NS) -> list[int]:
    """The pre-T4 fixed-8h-grid loop, copied verbatim as the oracle (R19b)."""
    missing: list[int] = []
    step = 8 * HOUR_NS
    boundary = (lo_ns // step + 1) * step
    while boundary <= hi_ns:
        j = bisect_left(times, boundary - tol_ns)
        if j >= len(times) or times[j] > boundary + tol_ns:
            missing.append(boundary)
        boundary += step
    return missing


def test_all_8h_matches_old_grid(tmp_path: Path) -> None:
    """R19: on an all-8h symbol a deleted 16:00 row is exactly today's grid hole."""
    times_ms = _rows(T0_MS, [-24, -16, -8, 0, 8, 24, 32, 40], jitter_ms=4)
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 8))
    lo, hi = T0_NS - HOUR_NS, T0_NS + 24 * HOUR_NS
    times_ns = _ns(times_ms)
    hole = T0_NS + 16 * HOUR_NS

    assert calc.settlement_hours(SYMBOL) == (8,) * len(times_ms)
    assert missing_settlements(times_ns, list(calc.settlement_hours(SYMBOL)), lo, hi) == [(hole, 8)]

    acc = calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi)

    assert acc.fallback_timestamps == [hole]
    assert acc.fallback_settlements == 1
    # bit for bit: the old-rule payment list, built in the old order
    expected = [(t, 1.0 * RATE * NOTIONAL) for t in times_ns if lo < t <= hi]
    expected.append((hole, 1.0 * DEFAULT_FUNDING_RATE_PER_PERIOD * NOTIONAL))
    expected.sort(key=lambda p: p[0])
    assert acc.payments == expected
    assert acc.total == sum(amount for _, amount in expected)
    assert _old_grid_missing(times_ns, lo, hi) == [hole]


def test_all_8h_equivalence_property() -> None:
    """R19b: for all-8h hours the rule equals the old 8h loop, ts for ts (200 seeds)."""
    rng = random.Random(20261017)
    step = 8 * HOUR_NS
    for _ in range(200):
        count = rng.randint(0, 40)
        start = T0_NS - rng.randint(0, 8) * step
        kept = [
            start + i * step + rng.randint(0, TOL_NS)
            for i in range(count)
            if rng.random() > 0.25
        ]
        lo = T0_NS + rng.randint(-6, 6) * step
        hi = lo + rng.randint(0, 6) * step
        assert missing_settlements(kept, [8] * len(kept), lo, hi) == [
            (ts, 8) for ts in _old_grid_missing(kept, lo, hi)
        ]


def test_4h_missing_0400_detected(tmp_path: Path) -> None:
    """R21 (the AC): a deleted 04:00 row on a 4h symbol pays 4/8 of the fallback."""
    times_ms = _rows(T0_MS, [-8, -4, 0, 8, 12, 16])
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 4))
    lo, hi = T0_NS - TOL_NS, T0_NS + 12 * HOUR_NS
    times_ns = _ns(times_ms)
    hole = T0_NS + 4 * HOUR_NS

    assert missing_settlements(times_ns, [4] * 6, lo, hi) == [(hole, 4)]
    assert missing_settlements(times_ns, [8] * 6, lo, hi) == []  # pre-T4: free

    acc = calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi)

    assert acc.fallback_timestamps == [hole]
    assert acc.fallback_settlements == 1
    assert dict(acc.payments)[hole] == pytest.approx(RATE * (4 / 8) * NOTIONAL)  # 0.075
    assert acc.total == pytest.approx(3 * RATE * NOTIONAL + RATE * (4 / 8) * NOTIONAL)
    # short side: the fallback takes the trader's sign
    short = calc.accrue(INSTRUMENT, "SHORT", NOTIONAL, lo, hi)
    assert dict(short.payments)[hole] == pytest.approx(-RATE * (4 / 8) * NOTIONAL)


def test_2h_missing_row_detected(tmp_path: Path) -> None:
    """R22: a deleted 06:00 row on a 2h symbol pays 2/8 of the fallback."""
    times_ms = _rows(T0_MS, [-2, 0, 2, 4, 8, 10, 12, 14, 16])
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 2))
    lo, hi = T0_NS + 5 * HOUR_NS, T0_NS + 7 * HOUR_NS
    times_ns = _ns(times_ms)
    hole = T0_NS + 6 * HOUR_NS

    assert missing_settlements(times_ns, [2] * 9, lo, hi) == [(hole, 2)]
    assert missing_settlements(times_ns, [8] * 9, lo, hi) == []

    acc = calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi)

    assert acc.fallback_timestamps == [hole]
    assert acc.total == pytest.approx(RATE * (2 / 8) * NOTIONAL)  # 0.0375


def test_2h_hole_spanning_window_uses_neighbour_interval(tmp_path: Path) -> None:
    """R22b: a hole wider than the window still uses max(h) of its true neighbours.

    A padded-window scan (the r1 text) finds no row in [01:00, 19:00], falls back
    to the 8h grid and returns [] for the narrow trade; the neighbour scan does not.
    """
    times_ms = _rows(T0_MS, [-2, 0, 20])  # 02:00..18:00 are one REST hole
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 2))
    times_ns = _ns(times_ms)
    hours = [2, 2, 2]

    narrow_lo, narrow_hi = T0_NS + 9 * HOUR_NS, T0_NS + 11 * HOUR_NS
    assert missing_settlements(times_ns, hours, narrow_lo, narrow_hi) == [
        (T0_NS + 10 * HOUR_NS, 2)
    ]
    narrow = calc.accrue(INSTRUMENT, "LONG", NOTIONAL, narrow_lo, narrow_hi)
    assert narrow.fallback_settlements == 1
    assert narrow.total == pytest.approx(RATE * (2 / 8) * NOTIONAL)  # 0.0375

    wide_lo, wide_hi = T0_NS + HOUR_NS, T0_NS + 19 * HOUR_NS
    assert missing_settlements(times_ns, hours, wide_lo, wide_hi) == [
        (T0_NS + h * HOUR_NS, 2) for h in range(2, 20, 2)
    ]
    wide = calc.accrue(INSTRUMENT, "LONG", NOTIONAL, wide_lo, wide_hi)
    assert wide.fallback_settlements == 9
    assert wide.total == pytest.approx(9 * RATE * (2 / 8) * NOTIONAL)  # 0.3375


def test_hole_next_to_8_to_2_switch(tmp_path: Path) -> None:
    """R22c: max(h) at an 8->2 switch keeps the 8h gap clean and finds the 2h hole."""
    times_ms = _rows(T0_MS, [0, 8, 16, 22])
    calc = _archive(tmp_path / "a.db", times_ms, dict(zip(times_ms, [8, 8, 2, 2], strict=True)))
    times_ns = _ns(times_ms)
    hours = [8, 8, 2, 2]

    # the 08:00->16:00 gap is a legal 8h one: no false 10:00/12:00/14:00
    before_lo, before_hi = T0_NS + 9 * HOUR_NS, T0_NS + 15 * HOUR_NS
    assert missing_settlements(times_ns, hours, before_lo, before_hi) == []
    assert calc.accrue(INSTRUMENT, "LONG", NOTIONAL, before_lo, before_hi).fallback_settlements == 0

    after_lo, after_hi = T0_NS + 17 * HOUR_NS, T0_NS + 21 * HOUR_NS
    assert missing_settlements(times_ns, hours, after_lo, after_hi) == [
        (T0_NS + 18 * HOUR_NS, 2),
        (T0_NS + 20 * HOUR_NS, 2),
    ]
    after = calc.accrue(INSTRUMENT, "LONG", NOTIONAL, after_lo, after_hi)
    assert after.fallback_settlements == 2
    assert after.total == pytest.approx(2 * RATE * (2 / 8) * NOTIONAL)


def test_switch_8_to_4_and_masked_first_row(tmp_path: Path) -> None:
    """R23: an 8->4 switch has no hole; a deleted first 4h row is masked (known 1/105).

    The masked case is the documented limitation of ``missing_settlements``: 20:00
    is the first row of the 4h regime, and its neighbours' ``max(h)`` is 8, so its
    deletion is invisible. The measured cost is 1 of 105 single-row deletions in
    the SOL 2022-11-09..19 switch week.
    """
    times_ms = _rows(T0_MS, [0, 8, 16, 20, 24])
    hours = dict(zip(times_ms, [8, 8, 8, 4, 4], strict=True))
    calc = _archive(tmp_path / "a.db", times_ms, hours)
    lo, hi = T0_NS + 9 * HOUR_NS, T0_NS + 23 * HOUR_NS

    assert calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi).fallback_settlements == 0

    kept = [t for t in times_ms if t != times_ms[3]]  # 20:00 deleted
    masked = _archive(tmp_path / "m.db", kept, {t: hours[t] for t in kept})
    assert masked.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi).fallback_settlements == 0


def test_switch_4_to_2_no_false_hole(tmp_path: Path) -> None:
    """R24: 00:00 h=4 -> 04:00 h=2 does NOT flag 02:00 (min(h) would)."""
    times_ms = _rows(T0_MS, [0, 4, 6, 8])
    calc = _archive(tmp_path / "a.db", times_ms, dict(zip(times_ms, [4, 2, 2, 2], strict=True)))
    lo, hi = T0_NS + HOUR_NS, T0_NS + 5 * HOUR_NS

    assert missing_settlements(_ns(times_ms), [4, 2, 2, 2], lo, hi) == []
    assert calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi).fallback_settlements == 0


def test_switch_2_to_8_no_false_hole(tmp_path: Path) -> None:
    """R25: 08:00 h=2 -> 16:00 h=8 does NOT flag 10:00/12:00/14:00 (min(h) would)."""
    times_ms = _rows(T0_MS, [0, 2, 4, 6, 8, 16])
    calc = _archive(tmp_path / "a.db", times_ms, dict(zip(times_ms, [2, 2, 2, 2, 2, 8], strict=True)))
    lo, hi = T0_NS + 9 * HOUR_NS, T0_NS + 15 * HOUR_NS

    assert missing_settlements(_ns(times_ms), [2, 2, 2, 2, 2, 8], lo, hi) == []
    assert calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi).fallback_settlements == 0


def test_unknown_interval_inherits_previous(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """R26: a row without a Vision interval inherits the previous one (8 before any)."""
    times_ms = _rows(T0_MS, [0, 2, 4, 6, 8])
    calc = _archive(tmp_path / "a.db", times_ms, {times_ms[i]: 2 for i in (1, 2, 3)})

    with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.funding"):
        hours = calc.settlement_hours(SYMBOL)
        assert calc.settlement_hours(SYMBOL) == hours  # cached: no second warning

    assert hours == (8, 2, 2, 2, 2)
    messages = [r.getMessage() for r in caplog.records if "have no Vision interval" in r.getMessage()]
    assert len(messages) == 1
    assert "SOLUSDT: 2 of 5 archived funding rows have no Vision interval" in messages[0]


def test_symbol_absent_uses_8h_grid(tmp_path: Path) -> None:
    """R27: a symbol with no archived row keeps the 8h grid and the flat fallback."""
    calc = _archive(tmp_path / "a.db", [], symbol="ETHUSDT")
    lo, hi = T0_NS - HOUR_NS, T0_NS + 31 * HOUR_NS

    assert calc.settlement_hours("BTCUSDT") == ()
    assert missing_settlements((), (), lo, hi) == [
        (T0_NS + h * HOUR_NS, 8) for h in (0, 8, 16, 24)
    ]

    acc = calc.accrue("BTCUSDT-PERP.BINANCE", "LONG", 40_000.0, lo, hi)

    assert acc.fallback_settlements == 4
    assert acc.total == pytest.approx(4 * 0.0001 * 40_000.0)  # 16.0


def test_trailing_edge_uses_last_interval(tmp_path: Path) -> None:
    """R28: after the last row (12:00 h=2) the 2h grid continues (old rule: [16:00])."""
    times_ms = _rows(T0_MS, [8, 10, 12])
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 2))
    lo, hi = T0_NS + 11 * HOUR_NS, T0_NS + 18 * HOUR_NS
    times_ns = _ns(times_ms)

    assert missing_settlements(times_ns, [2] * 3, lo, hi) == [
        (T0_NS + h * HOUR_NS, 2) for h in (14, 16, 18)
    ]
    assert missing_settlements(times_ns, [8] * 3, lo, hi) == [(T0_NS + 16 * HOUR_NS, 8)]

    acc = calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi)

    assert acc.fallback_settlements == 3
    assert acc.total == pytest.approx(RATE * NOTIONAL + 3 * RATE * (2 / 8) * NOTIONAL)


def test_leading_edge_uses_first_interval(tmp_path: Path) -> None:
    """R29: before the first row (12:00 h=4) the 4h grid runs back (old rule: [08:00])."""
    times_ms = _rows(T0_MS, [12, 16, 20])
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 4))
    lo, hi = T0_NS + HOUR_NS, T0_NS + 15 * HOUR_NS
    times_ns = _ns(times_ms)

    assert missing_settlements(times_ns, [4] * 3, lo, hi) == [
        (T0_NS + 4 * HOUR_NS, 4),
        (T0_NS + 8 * HOUR_NS, 4),
    ]
    assert missing_settlements(times_ns, [8] * 3, lo, hi) == [(T0_NS + 8 * HOUR_NS, 8)]

    acc = calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi)

    assert acc.fallback_settlements == 2
    assert acc.total == pytest.approx(RATE * NOTIONAL + 2 * RATE * (4 / 8) * NOTIONAL)


def test_tolerance_boundary_inclusive(tmp_path: Path) -> None:
    """R30: a row at B+60 s covers B; at B+60.001 s B is a hole AND the row is charged."""
    lo, hi = T0_NS - HOUR_NS, T0_NS + 20 * HOUR_NS
    hole = T0_NS + 16 * HOUR_NS
    head = _rows(T0_MS, [-8, 0, 8])
    tail = _rows(T0_MS, [24, 32])
    at_edge = head + [T0_MS + 16 * MS_PER_HOUR + 60_000] + tail
    beyond = head + [T0_MS + 16 * MS_PER_HOUR + 60_001] + tail

    assert missing_settlements(_ns(at_edge), [8] * len(at_edge), lo, hi) == []
    assert _old_grid_missing(_ns(at_edge), lo, hi) == []
    assert missing_settlements(_ns(beyond), [8] * len(beyond), lo, hi) == [(hole, 8)]
    assert _old_grid_missing(_ns(beyond), lo, hi) == [hole]

    calc = _archive(tmp_path / "b.db", beyond, dict.fromkeys(beyond, 8))
    acc = calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi)

    assert acc.fallback_timestamps == [hole]
    # the late row is still charged as archived: the hole rule only ADDS
    assert dict(acc.payments)[hole + 60_001_000_000] == pytest.approx(1.0 * RATE * NOTIONAL)


def test_tolerance_before_boundary(tmp_path: Path) -> None:
    """R30b: a row at B-60 s covers B; at B-60.001 s B is missing."""
    lo, hi = T0_NS - HOUR_NS, T0_NS + 20 * HOUR_NS
    hole = T0_NS + 16 * HOUR_NS
    head = _rows(T0_MS, [-8, 0, 8])
    tail = _rows(T0_MS, [24, 32])
    at_edge = head + [T0_MS + 16 * MS_PER_HOUR - 60_000] + tail
    beyond = head + [T0_MS + 16 * MS_PER_HOUR - 60_001] + tail

    assert missing_settlements(_ns(at_edge), [8] * len(at_edge), lo, hi) == []
    assert _old_grid_missing(_ns(at_edge), lo, hi) == []
    assert missing_settlements(_ns(beyond), [8] * len(beyond), lo, hi) == [(hole, 8)]
    assert _old_grid_missing(_ns(beyond), lo, hi) == [hole]


def test_first_row_exactly_tol_after_boundary(tmp_path: Path) -> None:
    """R30c: a FIRST row exactly at B+60 s covers B, and B-8h is still missing."""
    times_ms = [T0_MS + 8 * MS_PER_HOUR + 60_000, *_rows(T0_MS, [16, 24])]
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 8))
    lo, hi = T0_NS - HOUR_NS, T0_NS + 24 * HOUR_NS
    times_ns = _ns(times_ms)

    assert missing_settlements(times_ns, [8] * 3, lo, hi) == [(T0_NS, 8)]
    assert _old_grid_missing(times_ns, lo, hi) == [T0_NS]

    acc = calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi)

    assert acc.fallback_timestamps == [T0_NS]
    assert acc.fallback_settlements == 1


def test_last_row_exactly_tol_before_boundary(tmp_path: Path) -> None:
    """R30d: a LAST row exactly at B-60 s covers B, and B+8h is still missing."""
    times_ms = [*_rows(T0_MS, [-8, 0, 8]), T0_MS + 16 * MS_PER_HOUR - 60_000]
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 8))
    lo, hi = T0_NS - HOUR_NS, T0_NS + 25 * HOUR_NS
    times_ns = _ns(times_ms)
    hole = T0_NS + 24 * HOUR_NS

    assert missing_settlements(times_ns, [8] * 4, lo, hi) == [(hole, 8)]
    assert _old_grid_missing(times_ns, lo, hi) == [hole]

    acc = calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi)

    assert acc.fallback_timestamps == [hole]
    assert acc.fallback_settlements == 1


def test_coverage_ignores_window_edge(tmp_path: Path) -> None:
    """R31: a row 30 s after B but 20 s after the exit covers B and is itself uncharged."""
    times_ms = [*_rows(T0_MS, [-8, 0, 8]), T0_MS + 16 * MS_PER_HOUR + 30_000, *_rows(T0_MS, [24, 32])]
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 8))
    lo, hi = T0_NS - HOUR_NS, T0_NS + 16 * HOUR_NS + 10_000_000_000
    times_ns = _ns(times_ms)

    assert missing_settlements(times_ns, [8] * len(times_ms), lo, hi) == []
    assert _old_grid_missing(times_ns, lo, hi) == []

    acc = calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi)

    assert acc.fallback_timestamps == []
    assert acc.fallback_settlements == 0
    assert [ts for ts, _ in acc.payments] == [T0_NS, T0_NS + 8 * HOUR_NS]


def test_book_flat_at_hole_not_charged(tmp_path: Path) -> None:
    """R32: accrue_book charges a missing settlement only while the leg is non-flat."""
    times_ms = _rows(T0_MS, [0, 8, 24])  # 16:00 is a hole
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 8))
    hole = T0_NS + 16 * HOUR_NS
    end_ns = T0_NS + 25 * HOUR_NS

    def mark(_ts: int) -> float:
        return 100.0

    flat = calc.accrue_book(
        INSTRUMENT, [(T0_NS + HOUR_NS, 1.0), (T0_NS + 10 * HOUR_NS, -1.0)], mark, end_ns=end_ns
    )
    assert flat.fallback_settlements == 0
    assert hole not in [ts for ts, _ in flat.payments]

    held = calc.accrue_book(
        INSTRUMENT, [(T0_NS + HOUR_NS, 1.0), (T0_NS + 20 * HOUR_NS, -1.0)], mark, end_ns=end_ns
    )
    assert held.fallback_settlements == 1
    assert dict(held.payments)[hole] == pytest.approx(1.0 * 100.0 * RATE)


def test_book_4h_hole_scaled_fallback_long_and_short(tmp_path: Path) -> None:
    """R35c (r2 SF1): accrue_book charges a 4h hole 4/8 of the fallback, both signs.

    R32's archive is all-8h, so the leg-book fallback factor ``h / 8`` is always
    1.0 there: dropping it from ``accrue_book`` survived every test. A 4h symbol
    with a deleted 08:00 row and a held leg is charged 2x too much by that mutant.
    """
    times_ms = _rows(T0_MS, [0, 4, 12, 16])  # 08:00 deleted
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 4))
    hole = T0_NS + 8 * HOUR_NS
    end_ns = T0_NS + 18 * HOUR_NS  # not on the 4h grid: no trailing-edge hole

    def mark(_ts: int) -> float:
        return 100.0

    assert missing_settlements(_ns(times_ms), [4] * 4, T0_NS + HOUR_NS, end_ns) == [(hole, 4)]

    long = calc.accrue_book(INSTRUMENT, [(T0_NS + HOUR_NS, 2.0)], mark, end_ns=end_ns)
    assert long.fallback_settlements == 1
    assert dict(long.payments)[hole] == pytest.approx(2.0 * 100.0 * RATE * (4 / 8))  # 0.01
    # a real row near the hole stays unscaled (control)
    assert dict(long.payments)[T0_NS + 4 * HOUR_NS] == pytest.approx(2.0 * 100.0 * RATE)

    short = calc.accrue_book(INSTRUMENT, [(T0_NS + HOUR_NS, -2.0)], mark, end_ns=end_ns)
    assert short.fallback_settlements == 1
    assert dict(short.payments)[hole] == pytest.approx(-2.0 * 100.0 * RATE * (4 / 8))


def test_book_2h_hole_scaled_fallback(tmp_path: Path) -> None:
    """R35c (r2 SF1): accrue_book charges a 2h hole 2/8 of the fallback (long leg)."""
    times_ms = _rows(T0_MS, [0, 2, 6, 8])  # 04:00 deleted
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 2))
    hole = T0_NS + 4 * HOUR_NS
    end_ns = T0_NS + 9 * HOUR_NS  # not on the 2h grid

    def mark(_ts: int) -> float:
        return 100.0

    assert missing_settlements(_ns(times_ms), [2] * 4, T0_NS + HOUR_NS, end_ns) == [(hole, 2)]

    held = calc.accrue_book(INSTRUMENT, [(T0_NS + HOUR_NS, 2.0)], mark, end_ns=end_ns)
    assert held.fallback_settlements == 1
    assert dict(held.payments)[hole] == pytest.approx(2.0 * 100.0 * RATE * (2 / 8))  # 0.005


def test_length_mismatch_raises() -> None:
    """R33: a misaligned hours list is a ValueError, never a silent 8h default."""
    with pytest.raises(ValueError, match=r"^hours has 3 entries for 4 settlements$"):
        missing_settlements([T0_NS + i for i in range(4)], [8, 8, 8], T0_NS, T0_NS + 1)
