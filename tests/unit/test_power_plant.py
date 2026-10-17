"""Tests for scripts/research/power/plant.py (planted causal drift on 1m klines)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))
import plant  # noqa: E402

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


def test_solve_kappa_compensates() -> None:
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


def test_crosses_below_semantics() -> None:
    assert plant.crosses_below(30.0, 29.9) is True
    assert plant.crosses_below(30.1, 30.0) is False
    assert plant.crosses_below(29.9, 29.0) is False  # already below
    assert plant.crosses_below(None, 10.0) is False
    assert plant.crosses_below(50.0, None) is False


def bar_series(closes: list[float]) -> plant.Klines:
    """One flat 1m price per 4h bar (each bar = 240 equal minutes)."""
    n = len(closes) * BAR4H_MIN
    px = np.repeat(np.array(closes, dtype=np.float64), BAR4H_MIN)
    t = T0_MS + np.arange(n, dtype=np.int64) * MIN_MS
    return plant.Klines(t, px, px, px, px, np.ones(n), t + 59_999)


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


def test_solve_kappa_returns_last_evaluated_pair() -> None:
    """(kappa, l_end) always belongs together, also when the passes run out."""
    k = make_klines()
    w = window(k)
    for delta, max_passes in ((0.0002, 1), (0.0002, 5), (0.0003, 5), (DELTA, 1), (DELTA, 5)):
        kappa, passes, l_end = plant.solve_kappa(k, delta, max_passes=max_passes, **w)
        assert passes <= max_passes
        assert plant.plant_drift(k, delta=delta, kappa=kappa, **w).report.l_end == l_end, (
            delta,
            max_passes,
        )
    kappa, passes, l_end = plant.solve_kappa(k, 0.0002, max_passes=1, **w)
    assert (kappa, passes) == (0.0, 1)
    assert abs(l_end) > 0.005
    _, passes5, l5 = plant.solve_kappa(k, 0.0002, **w)
    assert passes5 == 5 and abs(l5) > 0.005  # exhausted, not converged
