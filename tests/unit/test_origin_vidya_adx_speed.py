"""Catalog-origin VIDYA and ADX: faster kernels, the same bits (vibe-quant-ivlr6).

The references are frozen copies of the kernels before ivlr6 (``_rma_full``'s fast loop,
``_wilder_full``, ``_vidya_port``'s loop). Patching them back in gives the old values
through the public ``compute_adx`` / ``compute_vidya`` entry points, compared bit for bit
(``tobytes``, NaN payloads included).
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from vibe_quant.dsl import compute_builtins
from vibe_quant.dsl.compute_builtins import _rma_np, _wilder_np, compute_adx
from vibe_quant.dsl.plugins import vidya
from vibe_quant.dsl.plugins.vidya import compute_vidya
from vibe_quant.screening import nt_runner

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO = Path(__file__).resolve().parents[2]
REAL_CATALOG = REPO / "data" / "catalog"
BTC_1M = "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"
RUN_START_NS = 1772323200 * 10**9  # 2026-03-01 00:00 UTC
RUN_END_NS = 1772928000 * 10**9  # 2026-03-08 00:00 UTC


# --- frozen pre-ivlr6 kernels ------------------------------------------------------------


def _old_rma_full(inputs: Any, length: int) -> tuple[Any, Any]:
    x = inputs[0]
    n = len(x)
    alpha, factor = compute_builtins._rma_consts(length)
    fv = compute_builtins._first_valid(x)
    if fv + length > n:
        return np.full(n, np.nan), (float("nan"), 1.0, alpha, factor)
    lead = fv + length - 1
    seed_win = x[fv : fv + length]
    nan_mask = np.isnan(seed_win)
    count = length - int(nan_mask.sum())
    seed = float(np.where(nan_mask, 0.0, seed_win).sum() / count)
    vals = x.tolist()
    for i in range(lead):
        vals[i] = float("nan")
    vals[lead] = seed
    if seed == seed and not np.isnan(x[lead + 1 :]).any():
        denom = factor + alpha
        w = seed
        fast = [float("nan")] * lead
        fast.append(w)
        for c in vals[lead + 1 :]:
            if w != c:
                w = math.fma(factor, w, alpha * c) / denom
            fast.append(w)
        return np.array(fast, dtype=np.float64), (w, 1.0, alpha, factor)
    seeded = (np.array(vals, dtype=np.float64),)
    out = [float("nan")] * n
    state: Any = (vals[0], 1.0, alpha, factor)
    out[0] = vals[0]
    for i in range(1, n):
        out[i], state = compute_builtins._rma_step(state, seeded, i)
    return np.array(out, dtype=np.float64), state


def _old_wilder_full(inputs: Any, length: int) -> tuple[Any, Any]:
    x = inputs[0]
    n = len(x)
    start = compute_builtins._wilder_start(x)
    if start + length - 1 > n:
        return np.full(n, np.nan), (float("nan"), length)
    seed = float(np.nansum(x[start : start + length - 1]))
    res = [float("nan")] * n
    res[start + length - 2] = seed
    value = seed
    vals = x.tolist()
    for i in range(start + length - 1, n):
        raw = vals[i]
        if raw == raw:
            value = value - value / length + raw
        res[i] = value
    return np.array(res, dtype=np.float64), (value, length)


def _old_vidya_port(df: pd.DataFrame, period: int) -> pd.Series | None:
    period = period if period > 0 else 14
    n = len(df)
    if n < period:
        return None
    close = df["close"].to_numpy(dtype=np.float64)
    mom = np.empty(n, dtype=np.float64)
    mom[0] = np.nan
    mom[1:] = close[1:] - close[:-1]
    positive = np.clip(mom, 0.0, None)
    negative = np.abs(np.clip(mom, None, 0.0))
    sums = (
        pd.DataFrame(np.column_stack((positive, negative)), copy=False)
        .rolling(period)
        .sum()
        .to_numpy()
    )
    psum = sums[:, 0]
    nsum = sums[:, 1]
    with np.errstate(all="ignore"):
        abs_cmo = np.abs((psum - nsum) / (psum + nsum))
    abs_cmo[psum + nsum == 0.0] = 0.0
    out = np.full(n, np.nan)
    out[period - 1] = np.mean(close[:period])
    alpha = 2.0 / (period + 1)
    cmo_list = abs_cmo.tolist()
    close_list = close.tolist()
    out_list = out.tolist()
    prev = out_list[period - 1]
    for i in range(period, n):
        a = cmo_list[i]
        prev = alpha * a * close_list[i] + prev * (1 - alpha * a)
        out_list[i] = prev
    return pd.Series(np.asarray(out_list, dtype=np.float64), index=df.index, name=f"VIDYA_{period}")


@contextmanager
def _old_kernels() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(compute_builtins, "_rma_full", _old_rma_full)
        mp.setattr(compute_builtins, "_wilder_full", _old_wilder_full)
        mp.setattr(vidya, "_vidya_port", _old_vidya_port)
        yield


def _values(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "adx": compute_adx(frame, {"period": 26}).to_numpy(),
        "vidya": compute_vidya(frame, {"period": 14}).to_numpy(),
    }


def _old_and_new(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with _old_kernels():
        old = _values(frame)
    return old, _values(frame)


def _bits(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a, dtype=np.float64).tobytes()


def _bit_different(a: np.ndarray, b: np.ndarray) -> int:
    return int((np.asarray(a).view(np.uint64) != np.asarray(b).view(np.uint64)).sum())


def _walk(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    high = close + np.abs(rng.normal(0.0, 0.2, n))
    low = close - np.abs(rng.normal(0.0, 0.2, n))
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 1.0})


# --- bit identity ------------------------------------------------------------------------


@pytest.mark.slow(reason="3.2M real bars x old and new ADX + VIDYA kernels, ~5 s")
@pytest.mark.skipif(not (REAL_CATALOG / "data" / "bar" / BTC_1M).is_dir(), reason="no real catalog")
def test_real_btc_1m_origin_history_vidya_and_adx26_are_bit_identical() -> None:
    """Every bar of the BTC 1m origin history up to 2026-03-08 (the 03-01..03-08 run included)."""
    hist = nt_runner._decode_bar_file(str(REAL_CATALOG), BTC_1M)
    assert hist is not None
    keep = hist["ts_init"] < RUN_END_NS
    frame = pd.DataFrame({k: hist[k][keep] for k in ("open", "high", "low", "close", "volume")})
    old, new = _old_and_new(frame)
    run = hist["ts_init"][keep] >= RUN_START_NS
    assert int(run.sum()) == 10080
    for name in ("adx", "vidya"):
        assert _bit_different(old[name][run], new[name][run]) == 0, name
        assert _bits(new[name]) == _bits(old[name]), name
        assert np.isfinite(new[name][run]).all(), name


def test_flat_stretches_hold_vidya_and_keep_adx_bit_identical() -> None:
    """A flat run of ``period`` equal closes is VIDYA 0/0 (weight 0: the value holds); a run
    one bar shorter never reaches 0/0. Both edges, plus a NaN close late in the series."""
    frame = _walk(40_000, 3)
    for col in ("open", "high", "low", "close"):
        frame.loc[10_000:10_060, col] = frame.loc[10_000, "close"]  # 61 equal bars: 0/0
        frame.loc[20_000:20_013, col] = frame.loc[20_000, "close"]  # 14 equal: 13 zero moves
    frame.loc[39_990, "close"] = np.nan
    old, new = _old_and_new(frame)
    for name in ("adx", "vidya"):
        assert _bits(new[name]) == _bits(old[name]), name
    v = new["vidya"]
    held = v[10_015:10_061]
    assert (held == held[0]).all() and np.isfinite(held).all()  # the 0/0 hold
    assert v[20_013] != v[20_012]  # 13 zero moves never make 0/0: still moving
    assert np.isnan(v[39_990:]).all() and np.isfinite(v[39_989])  # NaN close sticks


@pytest.mark.parametrize("nan_at", [None, "first", "last"])
def test_recursion_kernels_at_the_nan_edges(nan_at: str | None) -> None:
    """RMA and Wilder take a fast path only without a NaN after the seed: a NaN right after
    it, at the very end, or none at all; plus a constant run (the ``w == c`` skip)."""
    x = _walk(30_000, 5)["close"].to_numpy().copy()
    x[:3] = np.nan  # leading NaN run (chained input such as dx)
    x[5_000:5_300] = x[5_000]
    length = 26
    if nan_at == "first":
        x[3 + length] = np.nan
    elif nan_at == "last":
        x[-1] = np.nan
    with _old_kernels():
        old_rma, old_wilder = _rma_np(x, length), _wilder_np(x, length)
    assert _bits(_rma_np(x, length)) == _bits(old_rma)
    assert _bits(_wilder_np(x, length)) == _bits(old_wilder)
    assert np.isfinite(old_rma[-2]) and np.isfinite(old_wilder[-2])


# --- speed -------------------------------------------------------------------------------


@pytest.mark.timing
@pytest.mark.slow(reason="best of 3 x old and new kernels over 1.5M bars, ~6 s")
def test_origin_length_vidya_and_adx_faster() -> None:
    """1.5M bars (a catalog-origin 1m history is ~3.2M): best of 3, new vs the frozen kernels.
    MEASURED at ivlr6: ~0.63x (0.59 vs 0.94 s); the CPython loop floor bounds the rest."""
    frame = _walk(1_500_000, 7)

    def best(old: bool) -> float:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            if old:
                with _old_kernels():
                    _values(frame)
            else:
                _values(frame)
            times.append(time.perf_counter() - t0)
        return min(times)

    old_s, new_s = best(old=True), best(old=False)
    assert new_s <= 0.8 * old_s, (old_s, new_s)
