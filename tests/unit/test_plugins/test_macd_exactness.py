"""compute_macd must be bit-identical to ``pandas_ta_classic.macd`` (3 outputs).

The compute_fn is a numpy port of the library's ``_ema_aligned`` path (numpy
``.mean()`` seed, plain-float ``k*x + (1-k)*prev`` loop, slow<fast swap), with
each EMA recurrence memoized (``prefix_memo``). Equality is exact (no tolerance).
"""

from __future__ import annotations

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

from vibe_quant.dsl import compute_builtins as cb  # noqa: E402
from vibe_quant.dsl.compute_builtins import compute_macd  # noqa: E402
from vibe_quant.dsl.indicators import indicator_registry  # noqa: E402
from vibe_quant.dsl.prefix_memo import MEMO  # noqa: E402

_spec = indicator_registry.get("MACD")
assert _spec is not None and _spec.param_ranges is not None
FAST = list(
    range(
        int(_spec.param_ranges["fast_period"][0]), int(_spec.param_ranges["fast_period"][1]) + 1, 3
    )
)
SLOW = list(
    range(
        int(_spec.param_ranges["slow_period"][0]), int(_spec.param_ranges["slow_period"][1]) + 1, 7
    )
)
SIG = [5, 9, 13]


@pytest.fixture(autouse=True)
def _clean_memo() -> object:
    MEMO.clear()
    yield
    MEMO.clear()


def _walk(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"close": 100.0 + np.cumsum(rng.randn(n) * 0.5)})


def _frames(n: int) -> list[pd.DataFrame]:
    flat = pd.DataFrame({"close": np.full(n, 50.0)})
    nan = _walk(4, n)
    nan.loc[n // 3, "close"] = np.nan
    nan.loc[n // 2 : n // 2 + 2, "close"] = np.nan
    stretch = _walk(3, n)
    stretch.loc[n // 4 : n // 2, "close"] = stretch.loc[n // 4, "close"]
    return [_walk(1, n), _walk(2, n), flat, nan, stretch]


def _ref(df: pd.DataFrame, f: int, s: int, g: int) -> dict[str, pd.Series]:
    res = ta.macd(df["close"], fast=f, slow=s, signal=g)
    if res is None:
        return {k: pd.Series(np.full(len(df), np.nan)) for k in ("macd", "histogram", "signal")}
    return {"macd": res.iloc[:, 0], "histogram": res.iloc[:, 1], "signal": res.iloc[:, 2]}


def _bits(a: pd.Series) -> np.ndarray:
    return np.asarray(a.to_numpy(dtype=np.float64).view(np.uint64))


def _check(df: pd.DataFrame, f: int, s: int, g: int) -> None:
    ref = _ref(df, f, s, g)
    # the public fn can silently fall back to ta when the self-check fails, so
    # also pin the port itself (and that the self-check passed)
    assert cb._macd_port_ok()
    got = compute_macd(df, {"fast_period": f, "slow_period": s, "signal_period": g})
    port = cb._macd_port(df["close"], f, s, g)
    for k in ("macd", "histogram", "signal"):
        assert got[k].index.equals(df.index)
        assert np.array_equal(_bits(got[k]), _bits(ref[k])), f"{k} f={f} s={s} g={g} n={len(df)}"
        if len(df) >= max(f, s, g) and min(f, s, g) > 0:
            assert np.array_equal(_bits(port[k]), _bits(ref[k])), f"port {k} f={f} s={s} g={g}"


def test_macd_port_matches_ta_macd() -> None:
    # GA ranges incl. fast>slow (library swaps), every sampled combination
    combos = [(f, s, g) for f in FAST for s in SLOW for g in SIG]
    combos += [(40, 10, 5), (30, 21, 9), (26, 26, 9), (5, 8, 13), (0, 0, 0), (-2, 5, 3), (3, 2, 1)]
    for f, s, g in combos:
        for n in sorted({3, s, max(f, s, g) - 1, max(f, s, g), s + g - 1, s + g, 60, 300}):
            if n < 1:
                continue
            for df in _frames(n):
                _check(df, f, s, g)


def test_macd_port_incremental_growth_bitwise() -> None:
    """Memo hits on every growing prefix (and across a trim) stay bit-exact."""
    full = _walk(8, 400)
    full.loc[150, "close"] = np.nan
    with memo_env(True):
        for n in list(range(30, 400)) + list(range(100, 400)):  # second pass = trim/restart
            _check(full.iloc[:n].reset_index(drop=True), 12, 26, 9)
        assert MEMO.hits > 300


def test_macd_memo_off_equals_on() -> None:
    full = _walk(9, 300)
    outs = {}
    for flag in (True, False):
        with memo_env(flag):
            outs[flag] = [
                compute_macd(
                    full.iloc[:n], {"fast_period": 9, "slow_period": 30, "signal_period": 7}
                )
                for n in range(40, 300)
            ]
    for a, b in zip(outs[True], outs[False], strict=True):
        for k in a:
            assert np.array_equal(_bits(a[k]), _bits(b[k]))


def test_macd_stream_matches_ta_across_buffer_trims(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = {"macd_1h": ("MACD", {"fast_period": 10, "slow_period": 24, "signal_period": 8}, "1h")}
    _, cls = compile_indicators(
        {"macd_1h": {"type": "MACD", "fast_period": 10, "slow_period": 24, "signal_period": 8}}
    )
    bars = random_walk_bars(1500, seed=21)
    cb._macd_port_ok.cache_clear()
    on = run_stream(cls, bars, True, specs=specs)
    assert MEMO.hits > 1000  # memo engaged (3 EMAs per bar)
    monkeypatch.setattr(cb, "_macd_port_ok", lambda: False)  # reference = real ta.macd
    ref = run_stream(cls, bars, False, specs=specs)
    assert any(k.startswith("macd_1h") for d in on for k in d)
    for a, b in zip(on, ref, strict=True):
        assert a.keys() == b.keys()
        for k in a:
            assert np.float64(a[k]).view(np.uint64) == np.float64(b[k]).view(np.uint64)


def test_self_check_passes_here() -> None:
    cb._macd_port_ok.cache_clear()
    assert cb._macd_port_ok() is True
    assert MEMO.n_keys == 0


def test_self_check_leaves_other_memo_entries() -> None:
    from vibe_quant.dsl.compute_builtins import _rma_np

    x = np.random.RandomState(2).randn(40)
    with memo_env(True):
        _rma_np(x, 12)  # an ADX-style entry
        before = (MEMO.n_keys, MEMO.n_slots(("rma", 12)), MEMO.bytes)
        cb._macd_port_ok.cache_clear()
        assert cb._macd_port_ok() is True
        assert (MEMO.n_keys, MEMO.n_slots(("rma", 12)), MEMO.bytes) == before
        _rma_np(np.append(x, 1.0), 12)
        assert MEMO.hits == 1  # entry still usable


def test_mismatch_falls_back_to_ta(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = cb._macd_port

    def perturbed(close: pd.Series, f: int, s: int, g: int) -> dict[str, pd.Series]:
        out = real(close, f, s, g)
        out["macd"].iloc[-1] = out["macd"].iloc[-1] + 1e-9
        return out

    monkeypatch.setattr(cb, "_macd_port", perturbed)
    cb._macd_port_ok.cache_clear()
    try:
        df = _walk(11, 90)
        with caplog.at_level("WARNING"):
            got = compute_macd(df, {"fast_period": 12, "slow_period": 26, "signal_period": 9})
        assert any("falling back" in r.message for r in caplog.records)
        ref = _ref(df, 12, 26, 9)
        for k in ref:
            assert np.array_equal(_bits(got[k]), _bits(ref[k]))
    finally:
        monkeypatch.undo()
        cb._macd_port_ok.cache_clear()


def test_non_float64_uses_ta_path() -> None:
    df = _walk(5, 80).astype("float32")
    got = compute_macd(df, {"fast_period": 12, "slow_period": 26, "signal_period": 9})
    res = ta.macd(df["close"], fast=12, slow=26, signal=9)
    assert res is not None
    assert np.array_equal(_bits(got["macd"]), _bits(res.iloc[:, 0]))


def test_names_and_index_preserved() -> None:
    df = _walk(0, 60)
    df.index = pd.RangeIndex(500, 560)
    got = compute_macd(df, {"fast_period": 30, "slow_period": 10, "signal_period": 5})
    res = ta.macd(df["close"], fast=30, slow=10, signal=5)
    assert res is not None
    assert [got["macd"].name, got["histogram"].name, got["signal"].name] == list(res.columns)
    assert got["macd"].index.equals(df.index)
