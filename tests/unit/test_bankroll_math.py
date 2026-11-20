"""Continuous Kelly, half-Kelly and vol-target numbers on daily MTM returns.

Contract rows C1-C6 of design 2gyk1.3 (bankroll sizing report), plus import isolation.
Closed-form expectations to 1e-9.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from vibe_quant.bankroll.kelly import (
    DAY_NS,
    FAT_TAIL_EXCESS_KURTOSIS,
    KellyStats,
    day_buckets,
    is_fat_tail,
    kelly_stats,
)
from vibe_quant.metrics import mark_to_market_daily_returns

_F_FIELDS = ("f_full", "f_half", "vol_target")


def _two_point(mean: float, var: float, n: int) -> list[float]:
    """``n`` (even) values with sample mean ``mean`` and sample variance (ddof=1) ``var``."""
    d = math.sqrt(var * (n - 1) / n)
    return [mean + d, mean - d] * (n // 2)


def test_kelly_closed_form() -> None:
    stats = kelly_stats(_two_point(0.001, 0.0004, 400))
    assert stats.status == "ok"
    assert stats.reason is None
    assert stats.n == 400
    assert stats.mean == pytest.approx(0.001, abs=1e-9)
    assert stats.variance == pytest.approx(0.0004, abs=1e-9)
    assert stats.f_full == pytest.approx(2.5, abs=1e-9)
    assert stats.f_half == pytest.approx(1.25, abs=1e-9)
    assert stats.sigma_ann == pytest.approx(math.sqrt(0.146), abs=1e-9)
    assert stats.target_vol == 0.20

    # A second, unrelated sample: mean 0.003, var 0.0009, n 100.
    other = kelly_stats(_two_point(0.003, 0.0009, 100))
    assert other.f_full == pytest.approx(0.003 / 0.0009, abs=1e-9)
    assert other.f_half == pytest.approx(0.003 / 0.0009 / 2, abs=1e-9)
    assert other.sigma_ann == pytest.approx(math.sqrt(365 * 0.0009), abs=1e-9)


def test_min_days_grid() -> None:
    sample = _two_point(0.001, 0.0004, 60)
    short = kelly_stats(sample[:59])
    assert (short.status, short.reason, short.n) == ("insufficient_data", "insufficient_data", 59)
    assert all(getattr(short, name) is None for name in _F_FIELDS)
    assert short.mean is None and short.sigma_ann is None and short.excess_kurtosis is None

    enough = kelly_stats(sample)
    assert (enough.status, enough.n) == ("ok", 60)
    assert all(getattr(enough, name) is not None for name in _F_FIELDS)

    # min_days is the edge: n == min_days computes, n == min_days - 1 does not.
    assert kelly_stats(sample[:10], min_days=10).status == "ok"
    assert kelly_stats(sample[:9], min_days=10).status == "insufficient_data"

    # 59 elapsed days starting and ending at noon touch 60 UTC buckets (partial first and
    # last days count), on the same grid as the MTM daily series.
    noon = 19_723 * DAY_NS + DAY_NS // 2
    end = noon + 59 * DAY_NS
    assert day_buckets(noon, end) == 60
    daily = mark_to_market_daily_returns(1000.0, [], noon, end)
    assert daily is not None and len(daily) == 60
    # Exact midnight end: the end day itself is not a bucket.
    midnight = 19_723 * DAY_NS
    assert day_buckets(midnight, midnight + 59 * DAY_NS) == 59
    assert day_buckets(midnight, midnight + 59 * DAY_NS + 1) == 60


def test_non_positive_mean() -> None:
    zero = kelly_stats(_two_point(0.0, 0.0004, 400))
    assert zero.status == "ok"
    assert zero.f_full == 0.0
    assert zero.f_half == 0.0
    negative = kelly_stats(_two_point(-0.0005, 0.0004, 400))
    assert negative.status == "ok"
    assert negative.f_full == pytest.approx(-1.25, abs=1e-9)
    assert negative.f_half == pytest.approx(-0.625, abs=1e-9)
    # Recommendation side (ceiling 0, null, non_positive_mean): test_bankroll_recommend.


def test_numeric_domain() -> None:
    flat = kelly_stats([0.001] * 80)
    assert (flat.status, flat.reason) == ("zero_variance", "zero_variance")
    assert all(getattr(flat, name) is None for name in _F_FIELDS)

    huge = kelly_stats([1e200, 0.0] * 40)
    assert (huge.status, huge.reason) == ("numeric", "numeric")
    tiny = kelly_stats([1e-200, 0.0] * 40)
    assert (tiny.status, tiny.reason) == ("numeric", "numeric")
    nan = kelly_stats([0.01, float("nan")] * 40)
    assert nan.status == "numeric"
    inf = kelly_stats([0.01, float("inf")] * 40)
    assert inf.status == "numeric"

    for stats in (flat, huge, tiny, nan, inf, kelly_stats([0.01] * 3)):
        payload = json.dumps(stats.to_dict(), allow_nan=False)
        decoded = json.loads(payload)
        assert decoded["f_full"] is None
        assert decoded["reason"] == stats.status
    ok = json.loads(
        json.dumps(kelly_stats(_two_point(0.001, 0.0004, 400)).to_dict(), allow_nan=False)
    )
    assert ok["reason"] is None and ok["status"] == "ok"
    assert ok["f_full"] == pytest.approx(2.5, abs=1e-9)


def test_vol_target() -> None:
    var = 0.40**2 / 365
    stats = kelly_stats(_two_point(0.001, var, 400), target_vol=0.20)
    assert stats.sigma_ann == pytest.approx(0.40, abs=1e-9)
    assert stats.vol_target == pytest.approx(0.5, abs=1e-9)
    other = kelly_stats(_two_point(0.001, var, 400), target_vol=0.30)
    assert other.vol_target == pytest.approx(0.75, abs=1e-9)
    # Display only: vol_target never moves the Kelly fractions.
    assert other.f_full == stats.f_full and other.f_half == stats.f_half


def _hand_excess_kurtosis(values: list[int]) -> Fraction:
    xs = [Fraction(v) for v in values]
    mean = sum(xs, Fraction(0)) / len(xs)
    m2 = sum(((x - mean) * (x - mean) for x in xs), Fraction(0)) / len(xs)
    m4 = sum(((x - mean) * (x - mean) * (x - mean) * (x - mean) for x in xs), Fraction(0)) / len(xs)
    return m4 / (m2 * m2) - 3


def test_kurtosis() -> None:
    sample = [1, 2, 3, 4, 5, 6, 7, 30]
    stats = kelly_stats([float(v) for v in sample], min_days=8)
    expected = float(_hand_excess_kurtosis(sample))
    assert stats.excess_kurtosis is not None
    assert abs(stats.excess_kurtosis - expected) <= 1e-12
    assert stats.fat_tail is False  # 2.64 <= 3

    assert FAT_TAIL_EXCESS_KURTOSIS == 3.0
    # Exactly 3.0 (ten zeros, +1, -1: population kurtosis 6) is not fat-tailed ...
    at_edge = kelly_stats([0.0] * 10 + [1.0, -1.0], min_days=12)
    assert at_edge.excess_kurtosis == 3.0
    assert at_edge.fat_tail is False
    # ... and above it is (twelve zeros: kurtosis 7, excess 4).
    above = kelly_stats([0.0] * 12 + [1.0, -1.0], min_days=14)
    assert above.excess_kurtosis == pytest.approx(4.0, abs=1e-12)
    assert above.fat_tail is True
    assert is_fat_tail(3.0) is False
    assert is_fat_tail(math.nextafter(3.0, math.inf)) is True


def test_kelly_stats_is_frozen_record() -> None:
    stats = kelly_stats(_two_point(0.001, 0.0004, 400))
    assert isinstance(stats, KellyStats)
    with pytest.raises(AttributeError):
        stats.f_full = 1.0  # type: ignore[misc]


def test_import_isolation() -> None:
    """kelly/stress/recommend pull in no engine, validation or DB module."""
    code = (
        "import sys\n"
        "import vibe_quant.bankroll.kelly, vibe_quant.bankroll.stress\n"
        "import vibe_quant.bankroll.recommend\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'nautilus_trader'"
        " or m.startswith(('vibe_quant.validation', 'vibe_quant.db')))\n"
        "print(bad)\n"
    )
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
        env={"PYTHONPATH": str(root), "PATH": "/usr/bin:/bin"},
    )
    assert out.stdout.strip() == "[]"
