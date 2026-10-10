"""compute_kama must be bit-identical to pandas_ta_classic.kama.

The plugin ports the library's math to numpy for speed (the library loops with
``.iloc`` per element and the pandas prep is rebuilt on every call). Any
arithmetic divergence would silently change screening/validation results, so
equality is asserted exactly (not approximately) for every period in the
registry's param range across buffer sizes, flat bars, and NaN gaps, and the
fast path's one-shot self-check fallback is exercised with a perturbed port.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pta_stream_harness import (  # noqa: E402
    compile_indicators,
    memo_env,
    random_walk_bars,
    run_stream,
)

from vibe_quant.dsl.indicators import indicator_registry  # noqa: E402
from vibe_quant.dsl.plugins import kama as kama_mod  # noqa: E402
from vibe_quant.dsl.plugins.kama import compute_kama  # noqa: E402
from vibe_quant.dsl.prefix_memo import MEMO  # noqa: E402

FAST, SLOW = 2, 30  # the plugin's fixed fast/slow (Kaufman canonical)


def _df(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = pd.Series(100.0 + np.cumsum(rng.randn(n) * 0.5))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": pd.Series(np.full(n, 100.0)),
        }
    )


def _flat_df(n: int) -> pd.DataFrame:
    """Constant price: every diff is exactly 0, exercising non_zero_range's
    zero-guard epsilon add."""
    close = pd.Series(np.full(n, 100.0))
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": pd.Series(np.full(n, 1.0))}
    )


def _gap_df(seed: int, n: int) -> pd.DataFrame:
    """Random walk with NaN gaps in close."""
    df = _df(seed, n)
    close = df["close"].to_numpy(copy=True)
    close[:: 7] = np.nan
    df["close"] = pd.Series(close)
    return df


def _walk_with_zero_diff(seed: int, n: int) -> pd.DataFrame:
    """Random walk with one exactly-equal consecutive pair (epsilon path)."""
    df = _df(seed, n)
    if n >= 3:
        close = df["close"].to_numpy(copy=True)
        close[2] = close[1]
        df["close"] = pd.Series(close)
    return df


def _shapes(seed: int, n: int) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("walk", _df(seed, n)),
        ("zerodiff", _walk_with_zero_diff(seed, n)),
        ("flat", _flat_df(n)),
        ("gaps", _gap_df(seed, n)),
    ]


def _periods() -> list[int]:
    spec = indicator_registry.get("KAMA")
    assert spec is not None
    lo, hi = spec.param_ranges["period"]
    return list(range(int(lo), int(hi) + 1))


PERIODS = _periods()


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("n", [40, 120, 500])
@pytest.mark.parametrize("period", [5, 10, 21, 50])
def test_bit_identical_to_library(seed: int, n: int, period: int) -> None:
    df = _df(seed, n)
    expected = ta.kama(df["close"], length=period)
    actual = compute_kama(df, {"period": period})
    if expected is None:
        # Series shorter than warmup: plugin returns all-NaN (not ready), never 0
        assert actual.isna().all()
        return
    # Exact — no tolerance. NaN positions must also match.
    np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
    assert actual.name == expected.name


def test_too_short_series_returns_nan() -> None:
    """Warmup must read as not-ready: a 0.0 KAMA made ``close > kama`` true
    throughout warmup (vibe-quant-e70tl.23)."""
    df = _df(7, 8)  # shorter than slow=30 warmup
    result = compute_kama(df, {"period": 10})
    assert len(result) == 8
    assert result.isna().all()


def test_flat_series_exact() -> None:
    """Constant price exercises non_zero_range's zero-guard epsilon."""
    n = 100
    df = _flat_df(n)
    expected = ta.kama(df["close"], length=10)
    actual = compute_kama(df, {"period": 10})
    assert expected is not None
    np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())


def test_fast_slow_params_match_library() -> None:
    """The plugin fixes fast/slow at Kaufman's canonical 2/30 — the output must
    match the library with those params made explicit."""
    df = _df(3, 300)
    expected = ta.kama(df["close"], length=10, fast=FAST, slow=SLOW)
    actual = compute_kama(df, {"period": 10})
    assert expected is not None
    np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
    assert actual.name == f"KAMA_10_{FAST}_{SLOW}" == expected.name


@pytest.mark.parametrize("period", PERIODS)
def test_new_equals_old_all_periods_and_shapes(period: int) -> None:
    """Zero-tolerance new-vs-old sweep: every period in the registry's param
    range at buffer sizes {15, 40, 400, period-1, period+1, 2*period} over
    random walks, flat bars, and NaN gaps. Both the raw port and the public
    entry point must equal the pre-port pandas implementation exactly."""
    sizes = sorted({15, 40, 400, period - 1, period + 1, 2 * period})
    for n in sizes:
        for shape, df in _shapes(period * 31 + n, n):
            expected = kama_mod._kama_pandas(df, period)
            np.testing.assert_array_equal(
                kama_mod._kama_port(df, period).to_numpy(),
                expected.to_numpy(),
                err_msg=f"port != old: period={period} n={n} shape={shape}",
            )
            np.testing.assert_array_equal(
                compute_kama(df, {"period": period}).to_numpy(),
                expected.to_numpy(),
                err_msg=f"compute_kama != old: period={period} n={n} shape={shape}",
            )


def test_selfcheck_passes_on_this_platform() -> None:
    """On this platform the fast path must verify clean — otherwise the port is
    dead code and every compute_kama call silently takes the slow fallback."""
    assert kama_mod._kama_port_ok() is True


def test_selfcheck_fallback_on_perturbed_fast_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If the fast path disagrees with the old path the self-check must fall
    back to the old output and warn exactly once (one-shot cached check)."""
    real_port = kama_mod._kama_port

    def perturbed(df: pd.DataFrame, period: int) -> pd.Series:
        out = real_port(df, period)
        out.iloc[period + 5] = out.iloc[period + 5] + 1e-9
        return out

    monkeypatch.setattr(kama_mod, "_kama_port", perturbed)
    kama_mod._kama_port_ok.cache_clear()
    df = _df(11, 300)
    try:
        with caplog.at_level(logging.WARNING, logger=kama_mod.__name__):
            first = compute_kama(df, {"period": 10})
            second = compute_kama(df, {"period": 15})
        expected_10 = kama_mod._kama_pandas(df, 10)
        expected_15 = kama_mod._kama_pandas(df, 15)
        np.testing.assert_array_equal(first.to_numpy(), expected_10.to_numpy())
        np.testing.assert_array_equal(second.to_numpy(), expected_15.to_numpy())
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "falling back" in warnings[0].getMessage()
    finally:
        kama_mod._kama_port_ok.cache_clear()


@pytest.fixture(autouse=True)
def _clean_memo() -> object:
    MEMO.clear()
    yield
    MEMO.clear()


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)


def _stress_closes(n: int, period: int) -> np.ndarray:
    """Walk with the hostile appends: an equal close (diff==0 flips the whole
    window's eps guard, so prior outputs change), a close equal to the one
    `period` bars back (abs_diff zero, lag period), and a NaN gap."""
    rng = np.random.RandomState(5)
    c = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    for i in (150, 301, 470):
        c[i] = c[i - 1]
    for i in (220, 520):
        c[i] = c[i - period]
    c[390] = np.nan
    return c


def test_kama_memo_stream_bitwise() -> None:
    period = 10
    spec = {"kama": ("KAMA", {"period": period}, "1h")}
    _, cls = compile_indicators({"kama": {"type": "KAMA", "period": period}})
    close = _stress_closes(700, period)  # > buffer cap: several 25%-slack trims
    events = [("1h", c, c + 0.5, c - 0.5, c, 1.0) for c in close.tolist()]
    on = run_stream(cls, events, True, specs=spec)
    hits = MEMO.hits
    off = run_stream(cls, events, False, specs=spec)
    assert hits > len(events) // 2  # memo really engaged, not always-miss
    assert len(on) == len(off) == len(events)
    for a, b in zip(on, off, strict=True):
        assert a.keys() == b.keys()
        for k in a:
            assert _bits(np.array([a[k]]))[0] == _bits(np.array([b[k]]))[0]

    # Full arrays over a trimmed sliding window vs the unmemoized pandas reference.
    MEMO.clear()
    buf: list[float] = []
    for i, v in enumerate(close.tolist()):
        buf.append(v)
        if len(buf) > 60:  # compiler-style trim: drop the oldest slack at once
            del buf[:15]
        if len(buf) <= period:
            continue
        df = pd.DataFrame({"close": buf})
        got = compute_kama(df, {"period": period}).to_numpy()
        ref = kama_mod._kama_pandas(df, period).to_numpy()
        assert np.array_equal(_bits(got), _bits(ref)), i
    assert MEMO.hits > 400


def test_kama_flip_append_changes_prefix_but_matches_reference() -> None:
    """An appended equal close rewrites 290/400 prior outputs; the memo must
    not extend its stale prefix."""
    rng = np.random.RandomState(7)
    c = 100.0 + np.cumsum(rng.randn(400) * 0.5)
    with memo_env(True):
        a = compute_kama(pd.DataFrame({"close": c}), {"period": 10}).to_numpy()
        c2 = np.append(c, c[-1])
        b = compute_kama(pd.DataFrame({"close": c2}), {"period": 10}).to_numpy()
        ref = kama_mod._kama_pandas(pd.DataFrame({"close": c2}), 10).to_numpy()
        assert (a != b[:-1]).sum() > 100  # the flip really rewrote history
        assert np.array_equal(_bits(b), _bits(ref))
        assert MEMO.hits == 0  # sc prefix changed -> bitwise check rejected it


def test_kama_random_walk_stream_hits_every_bar() -> None:
    """Steady state with no flips: every bar after the first is a hit."""
    cl = [e[4] for e in random_walk_bars(300, seed=2)]
    with memo_env(True):
        for end in range(40, len(cl) + 1):
            compute_kama(pd.DataFrame({"close": cl[:end]}), {"period": 10})
        assert MEMO.hits == len(cl) - 40


def test_eps_constant_matches_pandas_ta() -> None:
    from pandas_ta_classic.utils import sflt

    assert kama_mod._EPS == float(np.finfo(np.float64).eps) == sflt.epsilon
