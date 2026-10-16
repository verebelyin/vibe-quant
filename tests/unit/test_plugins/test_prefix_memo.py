"""RecurrenceMemo: exact, bounded, read-only, killable; ADX results unchanged.

The real exactness proof is the pandas-reference test (``test_adx_exactness``)
and the full-array comparisons below against ``_adx_pandas``: the memo on/off
switch cannot catch a bug in the rewritten full path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pta_stream_harness import (  # noqa: E402
    compile_indicators,
    memo_env,
    random_walk_bars,
    run_stream,
)

from vibe_quant.dsl import prefix_memo  # noqa: E402
from vibe_quant.dsl.compute_builtins import (  # noqa: E402
    _adx_pandas,
    _first_valid,
    _rma_np,
    _wilder_np,
    _wilder_start,
    compute_adx,
)
from vibe_quant.dsl.prefix_memo import MEMO, RecurrenceMemo, memo_run  # noqa: E402


def _bits(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def _ohlc(n: int, seed: int = 3) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + np.abs(rng.randn(n)) * 0.4
    low = close - np.abs(rng.randn(n)) * 0.4
    return pd.DataFrame({"high": high, "low": low, "close": close})


def _schedule(n: int, cap: int, lookback: int, prefill: int = 0) -> Iterator[tuple[int, int]]:
    """Yield the buffer window after each bar: grow to cap+cap//4+1, cut to last cap.

    ``prefill`` starts the stream at the steady state -- a buffer already holding
    ``prefill`` rows, exactly as right after a trim -- instead of from an empty
    one; the two ``prefill=0`` parametrizations keep the empty-start growth
    (window 12 -> cap) covered.
    """
    start = 0
    if prefill:  # steady state: a buffer already holding ``prefill`` rows
        yield start, prefill
    for end in range(prefill + 1, n + 1):
        if end - start > cap + cap // 4:
            start = end - cap
        if end - start >= lookback:
            yield start, end


def _assert_stream_matches_pandas(df: pd.DataFrame, period: int, cap: int, prefill: int = 0) -> int:
    frames = 0
    with memo_env(True):
        for s, e in _schedule(len(df), cap, period, prefill):
            win = df.iloc[s:e].reset_index(drop=True)
            got = compute_adx(win, {"period": period}).to_numpy()
            ref = _adx_pandas(win, period).to_numpy()
            assert got.shape == ref.shape
            assert np.array_equal(_bits(got), _bits(ref)), f"end={e}"
            frames += 1
        assert MEMO.hits > frames  # the extend path really ran (4 rma calls per frame)
    return frames


@pytest.mark.parametrize(
    ("period", "cap", "prefill"),
    [(12, 400, 0), (12, 1330, 1330), (26, 400, 0)],
    ids=["12-400", "12-1330", "26-400"],
)
def test_full_array_bitwise_over_buffer_schedule(period: int, cap: int, prefill: int) -> None:
    # Crosses at least two trims (cap//4 slack) and every cut re-seeds the series.
    # [12-1330] starts from the steady state (prefill=cap): it crosses the same two
    # production-scale trim boundaries (window 1663 > 1330 + 1330//4 -> reseed to
    # 1330, twice) in ~680 frames instead of ~2000. The empty-start growth (window
    # 12 -> cap) stays covered by the two prefill=0 cases, [12-400] and [26-400].
    # Length: from a start buffer of cap rows (prefill=0) or prefill rows, two trims
    # need cap//4 further bars each.
    n = (prefill or cap) + 2 * (cap // 4) + 10
    assert _assert_stream_matches_pandas(_ohlc(n), period, cap, prefill) > 0


def test_eps_flip_mid_segment() -> None:
    """A zero-range bar adds eps to the WHOLE window's high-low: earlier values change."""
    df = _ohlc(700)
    for i in (500, 650):
        df.loc[i, ["high", "low"]] = df.loc[i, "close"]
    _assert_stream_matches_pandas(df, 12, 400)


def test_dm_below_eps_is_zeroed_at_tiny_scale() -> None:
    """At a scale where the raw +DM/-DM sit below eps the pandas reference blanks them
    (``utils._math.zero`` == ``abs(x) < float64 eps``) before Wilder smoothing, so the
    port must blank them too. 3e-16 keeps a few DMs above eps (finite ADX values)
    while the rest are below: an unguarded ``pos`` carries those tiny nonzero DMs into
    the Wilder seed and every ADX bit differs from the reference."""
    frames = _assert_stream_matches_pandas(_ohlc(100) * 3e-16, 12, 60, 60)
    assert frames > 0


def test_leading_nan_buffer_stream() -> None:
    """A buffer whose first rows are NaN (chained input): the TR blanking, the
    Wilder seed and the dx-seeded rma all shift with the first valid bar."""
    df = _ohlc(700)
    df.loc[:5, ["high", "low", "close"]] = np.nan
    _assert_stream_matches_pandas(df, 12, 400)


def test_nan_gaps() -> None:
    df = _ohlc(700)
    df.loc[450, "high"] = np.nan
    df.loc[520:522, "close"] = np.nan
    df.loc[600, ["high", "low", "close"]] = np.nan
    _assert_stream_matches_pandas(df, 12, 400)


def test_two_same_length_series_both_hit() -> None:
    """R1: popping the matched slot keeps both live series hot (stale slots starve them)."""
    rng = np.random.RandomState(0)
    a = rng.randn(60) + 5.0
    b = rng.randn(60) - 5.0
    with memo_env(True):
        for n in range(15, 60):
            for series in (a, b):
                _rma_np(series[:n], 12)
        steps = 59 - 15
        assert MEMO.misses == 2  # only the two first calls
        assert MEMO.hits == 2 * steps
        assert MEMO.n_slots(("rma", 12)) == 2  # live series only, no stale prefixes


def test_slots_keys_bytes_bounded() -> None:
    memo = RecurrenceMemo(slots=16, max_keys=32, max_bytes=40_000)
    x = np.arange(500, dtype=np.float64)
    for k in range(100):
        for j in range(20):
            memo.store(("k", k), (x + j,), x, None)
        assert memo.n_slots(("k", k)) <= 16
        assert memo.n_keys <= 32
        assert memo.bytes <= 40_000
    assert memo.bytes == sum(
        sum(a.nbytes for a in e[0]) + e[1].nbytes for lst in memo._d.values() for e in lst
    )
    assert prefix_memo.MAX_KEYS == 32
    assert prefix_memo.SLOTS_PER_KEY == 16


def test_oversized_input_bypasses_memo() -> None:
    x = np.zeros(prefix_memo.MAX_LEN + 1)
    with memo_env(True):
        memo_run(("big",), (x,), lambda i: (i[0].copy(), None), lambda s, i, k: (0.0, s))
        assert MEMO.n_keys == 0
        assert MEMO.misses == 0


def test_entries_read_only() -> None:
    with memo_env(True):
        _rma_np(np.random.RandomState(1).randn(40), 12)
        for lst in MEMO._d.values():
            for pins, out, _ in lst:
                assert not out.flags.writeable
                assert all(not p.flags.writeable for p in pins)


def test_stored_input_is_a_copy() -> None:
    """Mutating the caller's array after the call must not corrupt the memo."""
    x = np.random.RandomState(2).randn(40)
    with memo_env(True):
        _rma_np(x, 12)
        x2 = np.append(x, 1.0)
        x[5] += 1.0  # stale prefix no longer matches -> must miss, not hit
        before = MEMO.hits
        got = _rma_np(np.append(x, 1.0), 12)
        assert MEMO.hits == before
    ref = _rma_np_reference(np.append(x, 1.0), 12)
    assert np.array_equal(_bits(got), _bits(ref))
    assert len(x2) == 41


def _rma_np_reference(x: np.ndarray, length: int) -> np.ndarray:
    with memo_env(False):
        return np.asarray(_rma_np(x, length))


def test_kill_switch_disables_memo() -> None:
    x = np.random.RandomState(4).randn(50)
    with memo_env(False):
        for n in range(15, 50):
            _rma_np(x[:n], 12)
        assert MEMO.n_keys == 0
        assert MEMO.hits == 0
        assert MEMO.misses == 0


def test_rma_matches_pandas_rma_directly() -> None:
    import pandas_ta_classic as ta

    rng = np.random.RandomState(5)
    x = rng.randn(300)
    x[100] = np.nan
    with memo_env(True):
        for n in range(20, 300):
            got = _rma_np(x[:n], 12)
            ref = ta.rma(pd.Series(x[:n]), length=12).to_numpy()
            assert np.array_equal(_bits(got), _bits(ref)), n


def test_rma_seeds_at_first_valid_value() -> None:
    """Chained input (a NaN prefix): the SMA seed window starts at the first
    non-NaN value, every append is bit-identical to ``ta.rma``, and the memo hits."""
    import pandas_ta_classic as ta

    rng = np.random.RandomState(9)
    x = rng.randn(200)
    x[:7] = np.nan
    x[90] = np.nan
    with memo_env(True):
        for n in range(8, 200):  # incl. n < first_valid + length (undefined, all NaN)
            got = _rma_np(x[:n], 12)
            ref = ta.rma(pd.Series(x[:n]), length=12)
            ref_arr = np.full(n, np.nan) if ref is None else ref.to_numpy()
            assert np.array_equal(_bits(got), _bits(ref_arr)), n
        assert MEMO.hits > 150


def test_rma_memo_extends_at_seed_boundary() -> None:
    """The rma extension guard is strict: ``len(x) - 1 < first_valid + length``. At
    ``len(x) == first_valid + length + 1`` the seed is already inside the prefix, so the
    call must go through the memo (a real miss that stores) and every later append must
    HIT. A ``<=`` guard sends that boundary call down the full path, so nothing is
    stored and the next append misses and re-seeds instead of extending."""
    length = 12
    x = np.random.RandomState(12).randn(80)
    x[:4] = np.nan
    n0 = _first_valid(x) + length + 1  # 17: first length with the seed inside the prefix
    ref = _rma_np_reference(x[: n0 + 3], length)
    with memo_env(True):
        for n in range(1, n0):  # seed not in the prefix yet: the memo is bypassed
            _rma_np(x[:n], length)
        assert MEMO.hits == MEMO.misses == 0
        for n in range(n0, n0 + 4):
            got = _rma_np(x[:n], length)
        assert MEMO.hits == 3  # boundary append stores, the next three extend
        assert MEMO.misses == 1
    assert np.array_equal(_bits(got), _bits(ref))


@pytest.mark.parametrize("length", [1, 2, 14])
def test_wilder_matches_library_wilder_smooth(length: int) -> None:
    """``_wilder_np`` (memoized) vs the library's TA-Lib-seeded smoother, bit for
    bit, over growing prefixes with a NaN prefix (index 0 is always skipped), an
    interior NaN (carried forward) and a zero stretch."""
    from pandas_ta_classic.utils._wilder import wilder_smooth

    rng = np.random.RandomState(10)
    for lead in (1, 6):
        x = np.abs(rng.randn(160))
        x[:lead] = np.nan
        x[70] = np.nan
        x[100:110] = 0.0
        with memo_env(True):
            for n in range(1, 160):
                got = _wilder_np(x[:n], length)
                ref = wilder_smooth(pd.Series(x[:n]), length).to_numpy()
                assert np.array_equal(_bits(got), _bits(ref)), (lead, n)
            assert MEMO.hits > 100


@pytest.mark.parametrize("length", [1, 2, 14])
def test_wilder_seed_skips_index_zero(length: int) -> None:
    """``_wilder_start`` clamps to 1 (the library: ``max(int(finite[0]), 1)``): the seed
    is the sum of the ``length - 1`` values AFTER the first finite bar, even when that
    bar is index 0. Dropping the clamp sums the wrong window (``x[0:length-1]``), places
    the seed one bar early and indexes -1 at length 1, so the memoized stream stops
    matching ``wilder_smooth`` (which also skips index 0) from the first append."""
    from pandas_ta_classic.utils._wilder import wilder_smooth

    assert _wilder_start(np.array([1.0, 2.0, 3.0])) == 1  # finite[0] == 0 -> clamp to 1
    rng = np.random.RandomState(10)
    x = np.abs(rng.randn(160))  # no leading NaN: the first bar is finite
    x[70] = np.nan
    with memo_env(True):
        for n in range(1, 160):
            got = _wilder_np(x[:n], length)
            ref = wilder_smooth(pd.Series(x[:n]), length).to_numpy()
            assert np.array_equal(_bits(got), _bits(ref)), n
        assert MEMO.hits > 100


def test_wilder_memo_extends_at_seed_boundary() -> None:
    """The Wilder extension guard is strict: ``n - 1 < start + length - 1``. At
    ``n == start + length`` the seed bar is already inside the prefix, so the call must
    go through the memo (a real miss that stores) and every later append must HIT. A
    ``<=`` guard sends that boundary call down the full path, so nothing is stored and
    the next append misses and re-seeds instead of extending."""
    length = 12
    x = np.random.RandomState(12).randn(80)
    x[:4] = np.nan
    start = _wilder_start(x)  # 4: the NaN prefix is skipped
    n0 = start + length  # 16: first length with the seed bar inside the prefix
    with memo_env(False):
        ref = np.asarray(_wilder_np(x[: n0 + 3], length))
    with memo_env(True):
        for n in range(1, n0):  # seed not in the prefix yet: the memo is bypassed
            _wilder_np(x[:n], length)
        assert MEMO.hits == MEMO.misses == 0
        for n in range(n0, n0 + 4):
            got = _wilder_np(x[:n], length)
        assert MEMO.hits == 3  # boundary append stores, the next three extend
        assert MEMO.misses == 1
    assert np.array_equal(_bits(got), _bits(ref))


def test_two_timeframe_adx_value_stream_identical_memo_on_off() -> None:
    specs = {
        "adx_1h": ("ADX", {"period": 12}, "1h"),
        "adx_4h": ("ADX", {"period": 26}, "4h"),
    }
    _, cls = compile_indicators(
        {
            "adx_1h": {"type": "ADX", "period": 12, "timeframe": "1h"},
            "adx_4h": {"type": "ADX", "period": 26, "timeframe": "4h"},
        },
        additional_timeframes=["4h"],
    )
    base = random_walk_bars(900, seed=11, tf="1h")
    events = []
    for i, ev in enumerate(base):
        events.append(ev)
        if i % 4 == 3:
            events.append(("4h", *ev[1:]))
    on = run_stream(cls, events, True, specs=specs)
    hits = MEMO.hits
    off = run_stream(cls, events, False, specs=specs)
    assert hits > len(events)  # memo really engaged
    assert len(on) == len(off) == len(events)
    assert any("adx_4h" in d for d in on)
    for a, b in zip(on, off, strict=True):
        assert a.keys() == b.keys()
        for k in a:
            assert np.float64(a[k]).view(np.uint64) == np.float64(b[k]).view(np.uint64)


@pytest.fixture(autouse=True)
def _clean_memo() -> object:
    MEMO.clear()
    yield
    MEMO.clear()


def test_negative_zero_prefix_misses() -> None:
    """+0.0 == -0.0 numerically but the bits differ: the memo must not hit."""
    a = np.random.RandomState(6).randn(40)
    a[10] = 0.0
    b = a.copy()
    b[10] = -0.0
    with memo_env(True):
        _rma_np(a[:30], 12)
        got = _rma_np(b[:31], 12)
        assert MEMO.hits == 0
    assert np.array_equal(_bits(got), _bits(_rma_np_reference(b[:31], 12)))


@pytest.mark.timing
def test_thrash_more_series_than_slots_is_exact_and_not_slower() -> None:
    """More interleaved same-length series than slots: exact results, miss path ~ unmemoized."""
    import time

    import pandas_ta_classic as ta

    nser = prefix_memo.SLOTS_PER_KEY + 8
    rng = np.random.RandomState(8)
    series = [100.0 + np.cumsum(rng.randn(1400) * 0.5) for _ in range(nser)]
    with memo_env(True):
        for n in range(1000, 1010):
            for x in series:
                got = _rma_np(x[:n], 14)
                if n in (1000, 1009):
                    ref = ta.rma(pd.Series(x[:n]), length=14).to_numpy()
                    assert np.array_equal(_bits(got), _bits(ref))
        assert MEMO.hits == 0  # thrashing: every LRU slot is evicted before reuse

    def cost(enabled: bool) -> float:
        with memo_env(enabled):
            t = time.perf_counter()
            for n in range(1200, 1210):
                for x in series:
                    _rma_np(x[:n], 14)
            return time.perf_counter() - t

    cost(True)
    best_on = min(cost(True) for _ in range(3))
    best_off = min(cost(False) for _ in range(3))
    # memo OFF runs the same full path (kill switch), so ON-while-thrashing may
    # only add lookup+store overhead, not another recompute
    assert best_on < best_off * 1.5


@pytest.mark.timing
def test_rma_full_no_nan_uses_inline_fast_path() -> None:
    """B1: the miss path must stay near main's inline loop, not a per-element helper call."""
    import time
    from math import fma

    from vibe_quant.dsl.compute_builtins import _rma_full

    x = 100.0 + np.cumsum(np.random.RandomState(9).randn(1330) * 0.5)
    length = 14
    alpha = 1.0 / length
    factor = 1.0 - alpha

    def inline_main() -> list[float]:  # main's pre-memo no-NaN loop
        w = float(x[:length].sum() / length)
        out = [float("nan")] * (length - 1) + [w]
        denom = factor + alpha
        for c in x.tolist()[length:]:
            if w != c:
                w = fma(factor, w, alpha * c) / denom
            out.append(w)
        return out

    def best(fn: object) -> float:
        times = []
        for _ in range(5):
            t = time.perf_counter()
            for _ in range(20):
                fn()  # type: ignore[operator]
            times.append(time.perf_counter() - t)
        return min(times)

    assert best(lambda: _rma_full((x,), length)) < best(inline_main) * 2.0
