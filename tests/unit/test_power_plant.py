"""Tests for scripts/research/power/plant.py (planted causal drift on 1m klines)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))
import plant  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

MIN_MS = 60_000
BAR4H_MIN = 240
T0_MS = (
    1_700_006_400_000  # multiple of 4h (1_700_006_400_000 / 14_400_000 = 118055.6 -> aligned below)
)
T0_MS -= T0_MS % (BAR4H_MIN * MIN_MS)


def make_klines(n_minutes: int = 40 * 1440, seed: int = 5, sigma: float = 0.0012) -> plant.Klines:
    """Random-walk 1m klines on a 0.1 price grid, starting on a 4h boundary."""
    rng = np.random.default_rng(seed)
    close = np.round(30_000.0 * np.exp(np.cumsum(rng.normal(0.0, sigma, n_minutes))), 1)
    open_ = np.concatenate(([30_000.0], close[:-1]))
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    return plant.Klines(
        open_time=T0_MS + np.arange(n_minutes, dtype=np.int64) * MIN_MS,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=rng.uniform(1.0, 10.0, n_minutes),
        close_time=T0_MS + np.arange(n_minutes, dtype=np.int64) * MIN_MS + 59_999,
    )


def window(k: plant.Klines, a_bars: int = 10, b_bars: int | None = None) -> dict[str, int]:
    """plant_start / plant_end (ms) a_bars into the series and b_bars (default: near end)."""
    n4 = len(k.open_time) // BAR4H_MIN
    b = n4 - 2 if b_bars is None else b_bars
    return {
        "plant_start_ms": int(k.open_time[0]) + a_bars * BAR4H_MIN * MIN_MS,
        "plant_end_ms": int(k.open_time[0]) + b * BAR4H_MIN * MIN_MS,
    }


def test_zero_delta_is_identity() -> None:
    k = make_klines()
    res = plant.plant_drift(k, delta=0.0, kappa=0.0, **window(k))
    for name in ("open_time", "open", "high", "low", "close", "volume", "close_time"):
        assert np.array_equal(getattr(res.klines, name), getattr(k, name)), name
    assert res.report.l_end == 0.0


DELTA = 0.00005
KAPPA = 0.00001


def level(k: plant.Klines, res: plant.PlantResult) -> np.ndarray:
    """Log level L = ln(planted close / input close), per minute."""
    return np.asarray(np.log(res.klines.close / k.close))


def test_triggers_are_causal_at_fixed_kappa() -> None:
    k = make_klines()
    w = window(k)
    full = plant.plant_drift(k, delta=DELTA, kappa=KAPPA, **w)
    assert full.report.n_triggers >= 4
    mid = full.report.trigger_close_ms[full.report.n_triggers // 2]
    cut = (mid - int(k.open_time[0])) // MIN_MS  # truncate right at a trigger's 4h close
    trunc_k = plant.Klines(**{f: getattr(k, f)[:cut] for f in plant.Klines.__dataclass_fields__})
    cut_end_ms = mid
    part = plant.plant_drift(trunc_k, delta=DELTA, kappa=KAPPA, **w)
    expect = tuple(c for c in full.report.trigger_close_ms if c <= cut_end_ms)
    assert 1 <= len(expect) < full.report.n_triggers
    assert part.report.trigger_close_ms == expect
    assert np.array_equal(part.klines.close, full.klines.close[:cut])
    assert np.array_equal(part.klines.high, full.klines.high[:cut])

    # before plant_start: bit-identical
    pre = k.open_time < w["plant_start_ms"]
    assert pre.sum() == 10 * BAR4H_MIN
    for f in ("open", "high", "low", "close"):
        assert np.array_equal(getattr(full.klines, f)[pre], getattr(k, f)[pre]), f
    # before the first trigger close: input * exp(-kappa*m), m = minutes since plant_start
    first = full.report.trigger_close_ms[0]
    seg = (k.open_time >= w["plant_start_ms"]) & (k.open_time < first)
    m = (k.open_time[seg] - w["plant_start_ms"]) // MIN_MS
    assert seg.sum() > 100
    np.testing.assert_allclose(
        full.klines.close[seg], k.close[seg] * np.exp(-KAPPA * m), rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        full.klines.low[seg], k.low[seg] * np.exp(-KAPPA * m), rtol=1e-12, atol=0
    )


def test_drift_minutes_and_union() -> None:
    k = make_klines()
    w = window(k)
    res = plant.plant_drift(k, delta=DELTA, kappa=KAPPA, **w)
    trig = res.report.trigger_close_ms
    span = plant.DRIFT_MIN * MIN_MS
    t = k.open_time
    step = np.diff(level(k, res), prepend=0.0)
    inwin = (t > w["plant_start_ms"]) & (t < w["plant_end_ms"])
    # log-step: delta-kappa inside a drift span, -kappa elsewhere in the window, 0 outside
    active = np.zeros(len(t), dtype=bool)
    for c in trig:
        active |= (t >= c) & (t < c + span)
    active &= inwin
    expect = np.where(inwin, np.where(active, DELTA, 0.0) - KAPPA, 0.0)
    np.testing.assert_allclose(step, expect, rtol=0, atol=1e-12)
    assert res.report.drift_minutes == int(active.sum())

    gaps = np.diff(trig)
    iso = [
        i
        for i, c in enumerate(trig)
        if (i == 0 or gaps[i - 1] >= span) and (i == len(trig) - 1 or gaps[i] >= span)
    ]
    assert iso, "fixture needs an isolated trigger"
    c = trig[iso[0]]
    mine = (t >= c) & (t < c + span) & inwin
    assert mine.sum() == plant.DRIFT_MIN
    assert t[mine][0] == c  # first minute with open_time >= trigger close
    np.testing.assert_allclose(step[mine], DELTA - KAPPA, rtol=0, atol=1e-12)

    ov = [i for i in range(len(gaps)) if gaps[i] < span]
    assert ov, "fixture needs overlapping triggers"
    c1, c2 = trig[ov[0]], trig[ov[0] + 1]
    both = (t >= c2) & (t < c1 + span)
    assert both.sum() == (c1 + span - c2) // MIN_MS > 0
    np.testing.assert_allclose(
        step[both], DELTA - KAPPA, rtol=0, atol=1e-12
    )  # union, never 2*delta
    union = (t >= c1) & (t < c2 + span) & inwin
    assert union.sum() == (c2 + span - c1) // MIN_MS
    assert active[union].all()  # one contiguous drift span from c1 to c2 + 1440 min


def test_solve_kappa_compensates_and_returns_a_consistent_pair() -> None:
    k = make_klines()
    w = window(k)
    raw = plant.plant_drift(k, delta=DELTA, kappa=0.0, **w)
    assert (
        raw.report.l_end == DELTA * raw.report.drift_minutes > 0.1
    )  # plant_drift never compensates
    kappa, passes, l_end = plant.solve_kappa(k, DELTA, **w)
    assert 1 <= passes <= 5
    assert abs(l_end) <= 0.005
    assert kappa > 0
    again = plant.plant_drift(k, delta=DELTA, kappa=kappa, **w)
    assert again.report.l_end == l_end
    assert (
        abs(
            float(
                np.log(
                    again.klines.close[again.klines.open_time < w["plant_end_ms"]][-1]
                    / k.close[k.open_time < w["plant_end_ms"]][-1]
                )
            )
        )
        <= 0.005
    )

    # (kappa, l_end) always belongs together, also when the pass budget runs out
    for delta, cap in ((0.0002, 1), (0.0002, 5), (0.0003, 5), (DELTA, 1), (DELTA, 5)):
        kappa, passes, l_end = plant.solve_kappa(
            k, delta, max_passes=cap, max_total_passes=cap, **w
        )
        assert passes <= cap
        assert plant.plant_drift(k, delta=delta, kappa=kappa, **w).report.l_end == l_end, (
            delta,
            cap,
        )
    kappa, passes, l_end = plant.solve_kappa(k, 0.0002, max_passes=1, max_total_passes=1, **w)
    assert (kappa, passes) == (0.0, 1)
    assert abs(l_end) > 0.005
    _, passes5, l5 = plant.solve_kappa(k, 0.0002, max_total_passes=5, **w)
    assert passes5 == 5 and abs(l5) > 0.005  # capped inside the fixed point, not converged


def bar_series(closes: list[float]) -> plant.Klines:
    """One flat 1m price per 4h bar (each bar = 240 equal minutes)."""
    n = len(closes) * BAR4H_MIN
    px = np.repeat(np.array(closes, dtype=np.float64), BAR4H_MIN)
    t = T0_MS + np.arange(n, dtype=np.int64) * MIN_MS
    return plant.Klines(t, px, px, px, px, np.ones(n), t + 59_999)


def test_crosses_below_semantics() -> None:
    assert plant.crosses_below(30.0, 29.9) is True
    assert plant.crosses_below(30.1, 30.0) is False
    assert plant.crosses_below(29.9, 29.0) is False  # already below
    assert plant.crosses_below(None, 10.0) is False
    assert plant.crosses_below(50.0, None) is False


def test_uninitialized_rsi_never_triggers() -> None:
    w = {"plant_start_ms": T0_MS - MIN_MS, "plant_end_ms": T0_MS + 10**9 * 1000}
    # steady fall: RSI is 0 from its first reading, never a cross from >= 30 -> no trigger
    fall = bar_series([1000.0 - 10.0 * i for i in range(40)])
    assert plant.plant_drift(fall, delta=0.0, kappa=0.0, **w).report.n_triggers == 0
    # rise then fall: exactly one cross, and not before the RSI is initialized (14 updates)
    ups = [1000.0 + 20.0 * i for i in range(20)]
    cross = bar_series(ups + [ups[-1] - 30.0 * j for j in range(1, 40)])
    rep = plant.plant_drift(cross, delta=0.0, kappa=0.0, **w).report
    assert rep.n_triggers == 1
    first_close = rep.trigger_close_ms[0]
    assert first_close >= T0_MS + 20 * BAR4H_MIN * MIN_MS


def test_load_and_write_scratch_roundtrip(tmp_path: Path) -> None:
    """Archive opened ro -> planted -> scratch catalog (all timeframes) + funding rows copied."""
    import sqlite3

    from vibe_quant.data.archive import RawDataArchive
    from vibe_quant.data.catalog import CatalogManager

    day0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
    n = 2 * 1440
    k = make_klines(n_minutes=n)
    src = RawDataArchive(tmp_path / "src.db")
    src.insert_klines(
        "BTCUSDT",
        "1m",
        [
            (
                int(day0 + i * MIN_MS),
                k.open[i],
                k.high[i],
                k.low[i],
                k.close[i],
                k.volume[i],
                int(day0 + i * MIN_MS + 59_999),
                0,
                0,
                0,
                0,
            )
            for i in range(n)
        ],
        "test",
    )
    src.insert_funding_rates(
        "BTCUSDT", [(day0, 0.0001, 30_000.0), (day0 + 8 * 3_600_000, -0.0002, 30_100.0)], "test"
    )
    src.insert_funding_intervals("BTCUSDT", [(day0, 8)], "test")
    src.close()
    before = (tmp_path / "src.db").read_bytes()

    loaded = plant.load_klines(tmp_path / "src.db", start="2024-01-01", end="2024-01-02")
    assert len(loaded.open_time) == n
    assert np.array_equal(loaded.close, k.close) and np.array_equal(
        loaded.close_time, loaded.open_time + 59_999
    )
    plant.write_scratch(tmp_path / "scratch", loaded, tmp_path / "src.db")
    assert (tmp_path / "src.db").read_bytes() == before  # source untouched

    cat = CatalogManager(tmp_path / "scratch" / "data" / "catalog")
    assert cat.get_bar_count("BTCUSDT", "1m") == n
    assert cat.get_bar_count("BTCUSDT", "4h") == n // 240
    assert cat.get_bar_count("BTCUSDT", "1d") == 2
    con = sqlite3.connect(tmp_path / "scratch" / "data" / "archive" / "raw_data.db")
    assert con.execute("SELECT count(*) FROM raw_funding_rates").fetchone()[0] == 2
    assert con.execute("SELECT interval_hours FROM raw_funding_intervals").fetchall() == [(8,)]
    con.close()


def test_solve_kappa_refine_above_gates_phase2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 2 starts only past ``refine_above`` (default tol), so a caller with a looser
    acceptance threshold keeps the phase-1 kappa bit for bit."""
    k = make_klines()
    w = window(k)
    seen: list[float] = []
    real_drift = plant.plant_drift

    def spy(*args: Any, **kwargs: Any) -> plant.PlantResult:
        res = real_drift(*args, **kwargs)
        seen.append(res.report.l_end)
        return res

    monkeypatch.setattr(plant, "plant_drift", spy)
    kappa1, passes1, l1 = plant.solve_kappa(k, 0.0002, max_total_passes=5, **w)
    assert passes1 == 5 and abs(l1) > 0.002
    last = seen[-1]  # the last phase-1 iterate: the L_end the phase-2 gate reads
    assert abs(l1) <= abs(last)

    seen.clear()
    # equality edge: |L_end| == refine_above is not refined (the caller calls that converged)
    kappa_eq, passes_eq, l_eq = plant.solve_kappa(k, 0.0002, refine_above=abs(last), **w)
    assert (passes_eq, l_eq, kappa_eq) == (5, l1, kappa1)

    seen.clear()
    # a hair past the gate: refined to tol
    _, passes_up, l_up = plant.solve_kappa(
        k, 0.0002, refine_above=math.nextafter(abs(last), 0.0), **w
    )
    assert passes_up > 5 and abs(l_up) <= 0.002

    # a gate looser than the miss: not refined
    _, passes_lo, l_lo = plant.solve_kappa(k, 0.0002, refine_above=1.0, **w)
    assert (passes_lo, l_lo) == (5, l1)


def test_solve_kappa_brackets_a_stalled_fixed_point() -> None:
    """5 fixed-point passes do not reach tol at this delta; the bracket phase does.

    delta=2e-4 stalls the way btc5.0 did: kappa feeds back through the planted closes ->
    more RSI triggers -> a larger D, so L_end stays positive after the 5 fixed-point
    passes. Phase 2 spends the rest of the budget on a bracket + bisection instead.
    """
    k = make_klines()
    w = window(k)
    kappa5, passes5, l5 = plant.solve_kappa(k, 0.0002, max_total_passes=5, **w)
    assert passes5 == 5 and abs(l5) > 0.005  # phase 1 alone is short of tol

    kappa, passes, l_end = plant.solve_kappa(k, 0.0002, **w)
    assert 5 < passes <= 40  # phase 2 ran, and only up to the total cap
    assert abs(l_end) <= 0.002
    assert (kappa, l_end) != (kappa5, l5)
    again = plant.plant_drift(k, delta=0.0002, kappa=kappa, **w)
    assert again.report.l_end == l_end  # the returned pair belongs together


W60 = {"plant_start_ms": 0, "plant_end_ms": 61 * MIN_MS}  # window_min = 60


def patch_l_end(
    monkeypatch: pytest.MonkeyPatch, fn: Callable[[float], float]
) -> list[tuple[float, float]]:
    """Replace ``plant.plant_drift`` with a pure L_end(kappa) = fn(kappa); return the call log."""
    calls: list[tuple[float, float]] = []

    def fake(*args: Any, **kwargs: Any) -> Any:
        l_val = float(fn(kwargs["kappa"]))
        calls.append((kwargs["kappa"], l_val))
        return SimpleNamespace(report=SimpleNamespace(l_end=l_val))

    monkeypatch.setattr(plant, "plant_drift", fake)
    return calls


def test_solve_kappa_total_budget_caps_phase_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_total_passes caps phase 1 + phase 2 together, so a budget < max_passes stops early."""
    k = make_klines()
    w = window(k)
    real = plant.plant_drift
    calls: list[tuple[float, float]] = []

    def spy(*args: Any, **kwargs: Any) -> plant.PlantResult:
        res = real(*args, **kwargs)
        calls.append((kwargs["kappa"], res.report.l_end))
        return res

    monkeypatch.setattr(plant, "plant_drift", spy)
    for cap in (1, 3):
        calls.clear()
        kappa, passes, l_end = plant.solve_kappa(
            k, 0.0002, max_passes=5, max_total_passes=cap, **w
        )
        assert passes == cap == len(calls)  # exact count, bounded by the total budget
        assert passes <= cap
        assert (kappa, l_end) == min(calls, key=lambda pair: abs(pair[1]))  # true best pair
        assert l_end == real(k, delta=0.0002, kappa=kappa, **w).report.l_end
        assert abs(l_end) > 0.002  # not converged at this budget


def test_solve_kappa_without_a_root_spends_the_cap_and_returns_the_best_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No acceptable root: spend exactly the cap and return the smallest-|L_end| pair seen.

    Three unrelated L_end shapes that all fail to yield a root: an unbracketable
    constant, a bracketed discontinuity with no acceptable root, and a non-monotone
    piecewise curve.
    """

    def piecewise(kappa: float) -> float:
        if kappa < 0.01:
            return 0.5
        return -0.3 if kappa < 0.02 else 0.5

    cases: list[tuple[Callable[[float], float], float]] = [
        (lambda _kappa: 1.0, 1.0),
        (lambda kappa: 1.0 if kappa < 0.01 else -1.0, 1.0),
        (piecewise, 0.3),
    ]
    for fn, want_abs in cases:
        calls = patch_l_end(monkeypatch, fn)
        kappa, passes, l_end = plant.solve_kappa(
            make_klines(), 0.0002, max_total_passes=40, **W60
        )
        assert passes == len(calls) == 40  # every pass is one evaluation, exactly the cap
        assert (kappa, l_end) == min(calls, key=lambda pair: abs(pair[1]))
        assert abs(l_end) == want_abs  # never a root: fail closed on the best pair seen
        assert abs(l_end) > 0.002


# --- bar_min / drift_bars / rsi_period / rsi_level (B1-B5) --------------------------------

DAY_MIN = 1440
DAY_MS = DAY_MIN * MIN_MS


def rise_then_fall(n_up: int = 20, n_down: int = 20) -> list[float]:
    """Daily closes: ``n_up`` days stepping +20, then ``n_down`` days stepping -30."""
    ups = [1000.0 + 20.0 * i for i in range(n_up)]
    return ups + [ups[-1] - 30.0 * j for j in range(1, n_down + 1)]


def day_series(closes: list[float], last_minutes: int | None = None) -> plant.Klines:
    """One flat 1m price per day; ``last_minutes`` truncates the final day (incomplete bar)."""
    parts = [np.full(DAY_MIN, float(c)) for c in closes]
    if last_minutes is not None:
        parts[-1] = np.full(last_minutes, float(closes[-1]))
    px = np.concatenate(parts)
    t = T0_MS + np.arange(len(px), dtype=np.int64) * MIN_MS
    return plant.Klines(t, px, px, px, px, np.ones(len(px)), t + 59_999)


def day_window(n_days: int, a_days: int = 2) -> dict[str, int]:
    """plant_start (a_days into the series) and plant_end (at the series end)."""
    return {
        "plant_start_ms": T0_MS + a_days * DAY_MS,
        "plant_end_ms": T0_MS + n_days * DAY_MS,
    }


def test_daily_bar_plant_triggers_on_day_close() -> None:
    """bar_min=1440 feeds the RSI one close per day: triggers land on day closes and the drift
    span is drift_bars * 1440 minutes (8640 at the default drift_bars=6)."""
    assert T0_MS % DAY_MS == 0  # day-aligned fixture: the bar keys are UTC-day boundaries
    closes = rise_then_fall(20, 20)  # 40 days, exactly one cross below 30
    k = day_series(closes)
    w = day_window(len(closes))
    rep = plant.plant_drift(k, delta=0.0, kappa=0.0, bar_min=DAY_MIN, **w).report
    assert rep.n_triggers == 1
    assert rep.trigger_close_ms == (T0_MS + 27 * DAY_MS,)
    assert all(c % 86_400_000 == 0 for c in rep.trigger_close_ms)
    assert rep.drift_minutes == 6 * DAY_MIN  # drift_bars=6 (default) * bar_min=1440

    # drift_bars is forwarded: 3 days * 1440 minutes for the same isolated trigger
    rep3 = plant.plant_drift(k, delta=0.0, kappa=0.0, bar_min=DAY_MIN, drift_bars=3, **w).report
    assert rep3.trigger_close_ms == rep.trigger_close_ms
    assert rep3.drift_minutes == 3 * DAY_MIN


def test_incomplete_day_bar_never_updates_rsi() -> None:
    """A bar with fewer than bar_min rows is skipped: it never feeds the RSI, so it cannot
    trigger (the cross here falls exactly on the final, complete day)."""
    closes = rise_then_fall(20, 7)  # 27 days; the cross lands on the last day
    w = day_window(len(closes))
    full = day_series(closes)
    rep = plant.plant_drift(full, delta=0.0, kappa=0.0, bar_min=DAY_MIN, **w).report
    assert rep.trigger_close_ms == (T0_MS + 27 * DAY_MS,)

    partial = day_series(closes, last_minutes=1000)  # same closes, final day truncated
    rep_part = plant.plant_drift(partial, delta=0.0, kappa=0.0, bar_min=DAY_MIN, **w).report
    assert rep_part.n_triggers == 0  # the incomplete day's close never reached the RSI


def test_rsi_period_and_level_are_honoured() -> None:
    """rsi_period/rsi_level reach the RSI: a shorter period / higher level crosses on an
    earlier day than the 14/30 default, on the same constructed close series."""
    closes = rise_then_fall(20, 12)
    k = day_series(closes)
    w = day_window(len(closes))

    def triggers(period: int, level: float) -> tuple[int, ...]:
        rep = plant.plant_drift(
            k, delta=0.0, kappa=0.0, bar_min=DAY_MIN, rsi_period=period, rsi_level=level, **w
        ).report
        assert rep.n_triggers == 1
        return rep.trigger_close_ms

    default = triggers(14, 30.0)
    assert default == (T0_MS + 27 * DAY_MS,)
    assert triggers(14, 35.0) == (T0_MS + 26 * DAY_MS,)
    assert triggers(7, 30.0) == (T0_MS + 24 * DAY_MS,)
    assert triggers(7, 35.0) == (T0_MS + 23 * DAY_MS,)
    assert triggers(5, 35.0) == (T0_MS + 22 * DAY_MS,)
    # period 7 / level 35 fires on a bar where the 14/30 default does not
    assert (T0_MS + 23 * DAY_MS) not in default
