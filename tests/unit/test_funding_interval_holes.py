"""Funding holes on each symbol's own settlement interval.

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
    missing_settlements_causal,
    switch_holes_upper,
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
    """On an all-8h symbol a deleted 16:00 row is exactly today's grid hole."""
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
    """For all-8h hours the rule equals the old 8h loop, ts for ts (200 seeds)."""
    rng = random.Random(20261017)
    step = 8 * HOUR_NS
    for _ in range(200):
        count = rng.randint(0, 40)
        start = T0_NS - rng.randint(0, 8) * step
        kept = [
            start + i * step + rng.randint(0, TOL_NS) for i in range(count) if rng.random() > 0.25
        ]
        lo = T0_NS + rng.randint(-6, 6) * step
        hi = lo + rng.randint(0, 6) * step
        assert missing_settlements(kept, [8] * len(kept), lo, hi) == [
            (ts, 8) for ts in _old_grid_missing(kept, lo, hi)
        ]


def test_4h_missing_0400_detected(tmp_path: Path) -> None:
    """A deleted 04:00 row on a 4h symbol pays 4/8 of the fallback."""
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
    """A deleted 06:00 row on a 2h symbol pays 2/8 of the fallback."""
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
    """A hole wider than the window still uses max(h) of its true neighbours.

    A padded-window scan (the r1 text) finds no row in [01:00, 19:00], falls back
    to the 8h grid and returns [] for the narrow trade; the neighbour scan does not.
    """
    times_ms = _rows(T0_MS, [-2, 0, 20])  # 02:00..18:00 are one REST hole
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 2))
    times_ns = _ns(times_ms)
    hours = [2, 2, 2]

    narrow_lo, narrow_hi = T0_NS + 9 * HOUR_NS, T0_NS + 11 * HOUR_NS
    assert missing_settlements(times_ns, hours, narrow_lo, narrow_hi) == [(T0_NS + 10 * HOUR_NS, 2)]
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
    """Max(h) at an 8->2 switch keeps the 8h gap clean and finds the 2h hole."""
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
    """An 8->4 switch has no hole; a deleted first 4h row is masked (known 1/105).

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
    """00:00 h=4 -> 04:00 h=2 does NOT flag 02:00 (min(h) would)."""
    times_ms = _rows(T0_MS, [0, 4, 6, 8])
    calc = _archive(tmp_path / "a.db", times_ms, dict(zip(times_ms, [4, 2, 2, 2], strict=True)))
    lo, hi = T0_NS + HOUR_NS, T0_NS + 5 * HOUR_NS

    assert missing_settlements(_ns(times_ms), [4, 2, 2, 2], lo, hi) == []
    assert calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi).fallback_settlements == 0


def test_switch_2_to_8_no_false_hole(tmp_path: Path) -> None:
    """08:00 h=2 -> 16:00 h=8 does NOT flag 10:00/12:00/14:00 (min(h) would)."""
    times_ms = _rows(T0_MS, [0, 2, 4, 6, 8, 16])
    calc = _archive(
        tmp_path / "a.db", times_ms, dict(zip(times_ms, [2, 2, 2, 2, 2, 8], strict=True))
    )
    lo, hi = T0_NS + 9 * HOUR_NS, T0_NS + 15 * HOUR_NS

    assert missing_settlements(_ns(times_ms), [2, 2, 2, 2, 2, 8], lo, hi) == []
    assert calc.accrue(INSTRUMENT, "LONG", NOTIONAL, lo, hi).fallback_settlements == 0


def test_unknown_interval_inherits_previous(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A row without a Vision interval inherits the previous one (8 before any)."""
    times_ms = _rows(T0_MS, [0, 2, 4, 6, 8])
    calc = _archive(tmp_path / "a.db", times_ms, {times_ms[i]: 2 for i in (1, 2, 3)})

    with caplog.at_level(logging.WARNING, logger="vibe_quant.validation.funding"):
        hours = calc.settlement_hours(SYMBOL)
        assert calc.settlement_hours(SYMBOL) == hours  # cached: no second warning

    assert hours == (8, 2, 2, 2, 2)
    messages = [
        r.getMessage() for r in caplog.records if "have no Vision interval" in r.getMessage()
    ]
    assert len(messages) == 1
    assert "SOLUSDT: 2 of 5 archived funding rows have no Vision interval" in messages[0]


def test_symbol_absent_uses_8h_grid(tmp_path: Path) -> None:
    """A symbol with no archived row keeps the 8h grid and the flat fallback."""
    calc = _archive(tmp_path / "a.db", [], symbol="ETHUSDT")
    lo, hi = T0_NS - HOUR_NS, T0_NS + 31 * HOUR_NS

    assert calc.settlement_hours("BTCUSDT") == ()
    assert missing_settlements((), (), lo, hi) == [(T0_NS + h * HOUR_NS, 8) for h in (0, 8, 16, 24)]

    acc = calc.accrue("BTCUSDT-PERP.BINANCE", "LONG", 40_000.0, lo, hi)

    assert acc.fallback_settlements == 4
    assert acc.total == pytest.approx(4 * 0.0001 * 40_000.0)  # 16.0


def test_trailing_edge_uses_last_interval(tmp_path: Path) -> None:
    """After the last row (12:00 h=2) the 2h grid continues (old rule: [16:00])."""
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
    """Before the first row (12:00 h=4) the 4h grid runs back (old rule: [08:00])."""
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
    """A row at B+60 s covers B; at B+60.001 s B is a hole AND the row is charged."""
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
    """A row at B-60 s covers B; at B-60.001 s B is missing."""
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
    """A FIRST row exactly at B+60 s covers B, and B-8h is still missing."""
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
    """A LAST row exactly at B-60 s covers B, and B+8h is still missing."""
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
    """A row 30 s after B but 20 s after the exit covers B and is itself uncharged."""
    times_ms = [
        *_rows(T0_MS, [-8, 0, 8]),
        T0_MS + 16 * MS_PER_HOUR + 30_000,
        *_rows(T0_MS, [24, 32]),
    ]
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
    """accrue_book charges a missing settlement only while the leg is non-flat."""
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
    """accrue_book charges a 4h hole 4/8 of the fallback, both signs.

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
    """accrue_book charges a 2h hole 2/8 of the fallback (long leg)."""
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
    """A misaligned hours list is a ValueError, never a silent 8h default."""
    with pytest.raises(ValueError, match=r"^hours has 3 entries for 4 settlements$"):
        missing_settlements([T0_NS + i for i in range(4)], [8, 8, 8], T0_NS, T0_NS + 1)


def _copied_missing_settlements(
    times_ns: list[int], hours: list[int], lo_ns: int, hi_ns: int, tol_ns: int = TOL_NS
) -> list[tuple[int, int]]:
    """The pre-optimisation body (copies ``times`` and every ``hours`` entry per call)."""
    from bisect import bisect_right

    from vibe_quant.validation.funding import _multiples

    times = list(times_ns)
    n = len(times)
    if len(hours) != n:
        raise ValueError("length")
    if n == 0:
        return [(ts, 8) for ts in _multiples(8 * HOUR_NS, lo_ns, hi_ns)]
    aligned = [int(h) for h in hours]
    i0 = max(0, bisect_left(times, lo_ns - tol_ns) - 1)
    i1 = min(n, bisect_right(times, hi_ns + tol_ns) + 1)
    found: list[tuple[int, int]] = []
    if i0 == 0:
        head = min(hi_ns, times[0] - tol_ns - 1)
        found.extend((ts, aligned[0]) for ts in _multiples(aligned[0] * HOUR_NS, lo_ns, head))
    for i in range(i0, i1 - 1):
        h = max(aligned[i], aligned[i + 1])
        low = max(lo_ns, times[i] + tol_ns)
        high = min(hi_ns, times[i + 1] - tol_ns - 1)
        found.extend((ts, h) for ts in _multiples(h * HOUR_NS, low, high))
    if i1 == n:
        found.extend(
            (ts, aligned[-1])
            for ts in _multiples(aligned[-1] * HOUR_NS, max(lo_ns, times[-1] + tol_ns), hi_ns)
        )
    found.sort()
    return found


def test_missing_settlements_equals_the_copying_body_on_random_series() -> None:
    """The in-place rewrite returns exactly what the copying body did, tuple or list input.

    Random series cover empty, single-row, unsorted and duplicated times, mixed 2/4/8h
    intervals, jittered rows (inside and just outside the 60 s tolerance) and windows
    starting/ending ON a row, on the tolerance edge, before the first row and past the last.
    """
    rng = random.Random(7)
    checked = 0
    for _ in range(600):
        n = rng.choice([0, 1, 2, 3, 8, 30])
        base = rng.randrange(0, 5) * 8 * HOUR_NS
        times = [
            base + i * rng.choice([2, 4, 8]) * HOUR_NS + rng.choice([0, 0, 59, 60, 61]) * 10**9
            for i in range(n)
        ]
        if rng.random() < 0.8:
            times.sort()
        if times and rng.random() < 0.2:
            times.append(times[rng.randrange(len(times))])  # a duplicate
        hours = [rng.choice([2, 4, 8, 8, 8]) for _ in times]
        edges = [base + k * HOUR_NS + d for k in range(-3, 40) for d in (0, TOL_NS, -TOL_NS, 1)]
        for _ in range(4):
            lo, hi = sorted(rng.sample(edges, 2))
            expected = _copied_missing_settlements(times, hours, lo, hi)
            assert missing_settlements(times, hours, lo, hi) == expected
            assert missing_settlements(tuple(times), tuple(hours), lo, hi) == expected
            checked += 1
    assert checked == 2400


# ---------------------------------------------------------------------------
# Causal grid (arch-c9-2 R35/R36, dreview C9-12): the trend path's fallback grid
# uses the interval of the LAST row at or before ``s + 60 s`` (8h before any row);
# ``switch_holes_upper`` counts, never charges, what that grid cannot see at a switch.
# ---------------------------------------------------------------------------


def _hold(qty: float = 1.0) -> list[tuple[int, float]]:
    """A leg opened at H1 and held through the window."""
    return [(T0_NS + HOUR_NS, qty)]


def _flat_mark(_ts: int) -> float:
    return 100.0


def test_causal_grid(tmp_path: Path) -> None:
    """R35: a later 8h row never removes the H4 fallback the 4h row before it expects.

    Legacy ``max(h_prev, h_next)`` reads the 08:00 row's 8h back into the gap, so its H4
    fallback disappears once that row is archived (pinned); the causal grid keeps it.
    """
    h4 = (T0_NS + 4 * HOUR_NS, 4)
    one = [T0_MS]
    two = [T0_MS, T0_MS + 8 * MS_PER_HOUR]
    lo, hi = T0_NS + HOUR_NS, T0_NS + 6 * HOUR_NS

    assert missing_settlements_causal(_ns(one), [4], lo, hi) == [h4]
    assert missing_settlements_causal(_ns(two), [4, 8], lo, hi) == [h4]
    assert missing_settlements(_ns(one), [4], lo, hi) == [h4]
    assert missing_settlements(_ns(two), [4, 8], lo, hi) == []

    for name, times, hours in (
        ("one", one, {T0_MS: 4}),
        ("two", two, dict(zip(two, [4, 8], strict=True))),
    ):
        calc = _archive(tmp_path / f"{name}.db", times, hours)
        causal = calc.accrue_book(INSTRUMENT, _hold(2.0), _flat_mark, end_ns=hi, causal=True)
        assert causal.fallback_timestamps == [h4[0]]
        assert dict(causal.payments)[h4[0]] == 2.0 * 100.0 * RATE * (4 / 8)
        legacy = calc.accrue_book(INSTRUMENT, _hold(2.0), _flat_mark, end_ns=hi)
        assert legacy.fallback_timestamps == ([h4[0]] if name == "one" else [])


def test_causal_grid_is_8h_before_any_row() -> None:
    """Before the first row the causal interval is 8h; legacy reads the first row's own."""
    times = [T0_NS + 8 * HOUR_NS]
    lo, hi = T0_NS - HOUR_NS, T0_NS + 5 * HOUR_NS
    assert missing_settlements_causal(times, [4], lo, hi) == [(T0_NS, 8)]
    assert missing_settlements(times, [4], lo, hi) == [(T0_NS, 4), (T0_NS + 4 * HOUR_NS, 4)]
    assert missing_settlements_causal([], [], lo, hi) == [(T0_NS, 8)]


def test_causal_grid_tolerance_edges() -> None:
    """A row covers ``s`` iff ``|t - s| <= 60 s``; it sets ``h(s)`` iff ``t <= s + 60 s``."""
    s = T0_NS + HOUR_NS
    lo, hi = T0_NS, T0_NS + 90 * 60 * 10**9  # (H0, H1:30]: only H1 is a candidate
    for offset, missing in (
        (TOL_NS, []),
        (TOL_NS + 1, [(s, 1)]),
        (-TOL_NS, []),
        (-TOL_NS - 1, [(s, 1)]),
    ):
        times = [T0_NS - HOUR_NS, s + offset]
        assert missing_settlements_causal(times, [1, 1], lo, hi) == missing, offset
    # the row at s + 60 s decides h(s) = 8 -> H1 not expected; 1 ns later the 1h row before decides
    for offset, missing in ((TOL_NS, []), (TOL_NS + 1, [(s, 1)])):
        times = [T0_NS - HOUR_NS, s + offset]
        assert missing_settlements_causal(times, [1, 8], lo, hi) == missing, offset
    with pytest.raises(ValueError, match=r"^hours has 1 entries for 2 settlements$"):
        missing_settlements_causal([T0_NS, s], [8], lo, hi)


def test_causal_switch_blind_spot_counted_held_only(tmp_path: Path) -> None:
    """R36 / C9-12: a missing first 4h row (12:00) after 8h rows is charged by neither grid.

    The design's figure "legacy: 0.005" is wrong for this fixture: legacy's
    ``max(8, 4)`` masks 12:00 too (the documented limitation pinned in
    ``test_switch_8_to_4_and_masked_first_row``). The retrospective counter sees it:
    the NEXT row's 4h interval expects 12:00 in the gap. It is counted only while the
    leg is held and never charged.
    """
    times_ms = _rows(T0_MS, [0, 8, 16, 20])
    hours = dict(zip(times_ms, [8, 8, 4, 4], strict=True))
    calc = _archive(tmp_path / "a.db", times_ms, hours)
    h12 = T0_NS + 12 * HOUR_NS
    lo, hi = T0_NS + 9 * HOUR_NS, T0_NS + 19 * HOUR_NS
    times_ns, interval = _ns(times_ms), [8, 8, 4, 4]

    assert missing_settlements_causal(times_ns, interval, lo, hi) == []
    assert missing_settlements(times_ns, interval, lo, hi) == []
    assert switch_holes_upper(times_ns, interval, lo, hi, horizon_ns=hi + 1) == [h12]

    held = calc.accrue_book(INSTRUMENT, [(lo, 1.0)], _flat_mark, end_ns=hi, causal=True)
    assert held.switch_hole_timestamps == [h12]
    assert held.fallback_settlements == 0
    assert h12 not in dict(held.payments)
    assert held.total == 1.0 * 100.0 * RATE  # the archived 16:00 row only

    flat = calc.accrue_book(
        INSTRUMENT, [(lo, 1.0), (T0_NS + 11 * HOUR_NS, -1.0)], _flat_mark, end_ns=hi, causal=True
    )
    assert flat.switch_hole_timestamps == []
    legacy = calc.accrue_book(INSTRUMENT, [(lo, 1.0)], _flat_mark, end_ns=hi)
    assert legacy.switch_hole_timestamps == []  # the counter is causal-path only
    assert legacy.payments == held.payments


def test_switch_holes_upper_is_an_upper_bound() -> None:
    """A lag-labelled legal 4h->2h switch (00:00 h4, 04:00 h2) is counted too (documented)."""
    times = _ns(_rows(T0_MS, [0, 4, 6, 8]))
    lo, hi = T0_NS + HOUR_NS, T0_NS + 5 * HOUR_NS
    assert missing_settlements(times, [4, 2, 2, 2], lo, hi) == []
    assert missing_settlements_causal(times, [4, 2, 2, 2], lo, hi) == []
    assert switch_holes_upper(times, [4, 2, 2, 2], lo, hi, horizon_ns=hi + 1) == [
        T0_NS + 2 * HOUR_NS
    ]
    # covered or causally expected settlements are never counted
    assert switch_holes_upper(times, [2, 2, 2, 2], lo, hi, horizon_ns=hi + 1) == []
    assert (
        switch_holes_upper(_ns(_rows(T0_MS, [0, 2, 4])), [4, 2, 2], lo, hi, horizon_ns=hi + 1) == []
    )


def test_causal_posting_boundary(tmp_path: Path) -> None:
    """C9-12: a settlement posts at its covering row's time, so a window split never loses it.

    H4's row is archived at H4 + 2 ms. A window ending at H4 + 1 ms charges neither the
    row (it posts later) nor a fallback (H4 is covered); the next window charges the row.
    Split windows equal one window, payment for payment.
    """
    times_ms = [T0_MS, T0_MS + 4 * MS_PER_HOUR + 2]
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 4))
    split = T0_NS + 4 * HOUR_NS + 1_000_000
    end = T0_NS + 6 * HOUR_NS

    first = calc.accrue_book(INSTRUMENT, _hold(), _flat_mark, end_ns=split, causal=True)
    assert first.payments == []
    assert first.fallback_settlements == 0
    second = calc.accrue_book(INSTRUMENT, [(split, 1.0)], _flat_mark, end_ns=end, causal=True)
    whole = calc.accrue_book(INSTRUMENT, _hold(), _flat_mark, end_ns=end, causal=True)
    assert second.payments == whole.payments == [(times_ms[1] * 1_000_000, 100.0 * RATE)]
    assert whole.fallback_settlements == 0


def test_causal_default_path_unchanged(tmp_path: Path) -> None:
    """``causal`` defaults to False: the legacy grid, fallback count and payments."""
    times_ms = _rows(T0_MS, [0, 4, 12, 16])  # 08:00 deleted, 4h
    calc = _archive(tmp_path / "a.db", times_ms, dict.fromkeys(times_ms, 4))
    end_ns = T0_NS + 18 * HOUR_NS
    default = calc.accrue_book(INSTRUMENT, _hold(2.0), _flat_mark, end_ns=end_ns)
    explicit = calc.accrue_book(INSTRUMENT, _hold(2.0), _flat_mark, end_ns=end_ns, causal=False)
    causal = calc.accrue_book(INSTRUMENT, _hold(2.0), _flat_mark, end_ns=end_ns, causal=True)
    assert default == explicit
    assert default.fallback_timestamps == causal.fallback_timestamps == [T0_NS + 8 * HOUR_NS]
    assert default.payments == causal.payments


def test_switch_holes_upper_never_reads_past_the_run_end(tmp_path: Path) -> None:
    """R39 / rev S3: rows at or after ``horizon_ns`` (the run end) are never read.

    8h rows at 00/08/16:00, then 4h rows at 24h and 28h. Unbounded, the 24h row's 4h
    interval flags 20:00 in a window ending before 24h; bounded at 24h the full archive
    and the archive cut at the run end agree. A next row 1 ns before the horizon is read.
    ``accrue_book`` without a horizon bounds at its own end (the call is the run).
    """
    full = [0, 8 * HOUR_NS, 16 * HOUR_NS, 24 * HOUR_NS, 28 * HOUR_NS]
    hours = [8, 8, 8, 4, 4]
    lo, hi, end = 0, 24 * HOUR_NS - 1, 24 * HOUR_NS
    assert switch_holes_upper(full, hours, lo, hi, horizon_ns=end + 1) == [20 * HOUR_NS]
    assert switch_holes_upper(full, hours, lo, hi, horizon_ns=end) == []
    assert switch_holes_upper(full[:3], hours[:3], lo, hi, horizon_ns=end) == []
    edge = [*full[:3], end - 1, *full[4:]]
    assert switch_holes_upper(edge, hours, lo, hi, horizon_ns=end) == [20 * HOUR_NS]

    times_ms = _rows(T0_MS, [0, 8, 16, 24, 28])
    calc = _archive(tmp_path / "a.db", times_ms, dict(zip(times_ms, hours, strict=True)))
    one = calc.accrue_book(INSTRUMENT, _hold(), _flat_mark, end_ns=T0_NS + hi, causal=True)
    assert one.switch_hole_timestamps == []
    longer = calc.accrue_book(INSTRUMENT, _hold(), _flat_mark, end_ns=T0_NS + end, causal=True)
    assert longer.switch_hole_timestamps == [T0_NS + 20 * HOUR_NS]
