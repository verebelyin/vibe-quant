"""Pure-Python compute_fn implementations for every built-in indicator.

Each function takes a rolling OHLCV DataFrame (columns: open, high, low,
close, volume) plus the merged params dict, and returns either a single
``pd.Series`` (for one-output indicators) or a ``dict[str, pd.Series]``
keyed by output name (for multi-output indicators). The return shape
mirrors the per-indicator layout the compiler already produces in its
``_generate_update_pta_indicators`` method.

``pandas_ta_classic`` is imported lazily so that merely loading
``vibe_quant.dsl.indicators`` (which eagerly references these functions
at spec-registration time) does not pay the ~100 MB / several-hundred-ms
pandas + pandas-ta startup cost. The first call to any compute_fn
triggers the import; subsequent calls hit ``sys.modules`` cache.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pandas as pd


@cache
def _ta() -> Any:
    """Lazy accessor for ``pandas_ta_classic`` — defers the heavy import."""
    import pandas_ta_classic as ta

    return ta


def int_param(params: dict[str, object], key: str, default: int) -> int:
    """Pull an int from a param dict with a fallback when the key is
    missing or the value isn't a real number.

    Shared by the compute functions here and the NT-kwargs helpers in
    ``indicators.py``.
    """
    val = params.get(key, default)
    return int(val) if isinstance(val, (int, float)) else default


def float_param(params: dict[str, object], key: str, default: float) -> float:
    val = params.get(key, default)
    return float(val) if isinstance(val, (int, float)) else default


def nan_like(df: pd.DataFrame) -> pd.Series:
    """All-NaN series on ``df``'s index: the "not enough data yet" result.

    Never return zeros for warmup -- the compiled strategy treats NaN as
    not-ready, but a 0.0 would be stored as a real value (e.g. KAMA = 0 makes
    ``close > kama`` true throughout warmup).
    """
    import numpy as np
    import pandas as pd

    return pd.Series(np.full(len(df.index), np.nan, dtype=np.float64), index=df.index)


# ---------------------------------------------------------------------------
# Single-output indicators — Series return
# ---------------------------------------------------------------------------


def compute_rsi(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", _ta().rsi(df["close"], length=int_param(params, "period", 14)))


def compute_ema(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", _ta().ema(df["close"], length=int_param(params, "period", 14)))


def compute_sma(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", _ta().sma(df["close"], length=int_param(params, "period", 14)))


def compute_wma(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", _ta().wma(df["close"], length=int_param(params, "period", 14)))


def compute_dema(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """DEMA — all-NaN below ``2 * period - 1`` rows (the first valid bar).

    pandas-ta returns None there (after logging an inner-EMA "Series has N rows"
    warning at ``n == period``); the guard gives the same NaN without the call.
    """
    period = int_param(params, "period", 14)
    if len(df) < 2 * (period if period > 0 else 10) - 1:  # pandas-ta: length<=0 -> 10
        return nan_like(df)
    result = _ta().dema(df["close"], length=period)
    return nan_like(df) if result is None else cast("pd.Series", result)


def compute_tema(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", _ta().tema(df["close"], length=int_param(params, "period", 14)))


def compute_cci(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast(
        "pd.Series",
        _ta().cci(df["high"], df["low"], df["close"], length=int_param(params, "period", 20)),
    )


def compute_willr(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast(
        "pd.Series",
        _ta().willr(df["high"], df["low"], df["close"], length=int_param(params, "period", 14)),
    )


def compute_roc(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", _ta().roc(df["close"], length=int_param(params, "period", 10)))


def compute_atr(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast(
        "pd.Series",
        _ta().atr(df["high"], df["low"], df["close"], length=int_param(params, "period", 14)),
    )


_PTA_COLUMNS = ("open", "high", "low", "close", "volume")


def _mfi_port(df: pd.DataFrame, length: int) -> pd.Series | None:
    """Numpy port of ``pandas_ta_classic.mfi`` (default drift 1, no offset).

    Operation-for-operation with the library: ``hlc3 = (high + low + close) / 3``,
    ``raw_money_flow = hlc3 * volume``, the one-bar typical-price diff decides
    the positive/negative flow, then pandas' own ``rolling(length).sum()`` (the
    library's Cython accumulator -- a hand-rolled add/subtract loop is NOT bit
    identical) and ``100 * psum / (psum + nsum)``. Only the DataFrame/``.loc``
    prep is removed; every float values goes through the same operations in the
    same order, so the result equals the library bit for bit.

    Assumes float64 columns; ``None`` when the library would return ``None``
    (fewer than ``length`` rows). ``tests/unit/test_indicator_fast_paths.py``
    is the zero-tolerance reference.
    """
    import numpy as np
    import pandas as pd

    length = length if length > 0 else 14
    n = len(df)
    if n < length:
        return None
    cols = df.columns
    if len(cols) == 5 and tuple(cols) == _PTA_COLUMNS:
        block = df.to_numpy(copy=False)
        high = block[:, 1]
        low = block[:, 2]
        close = block[:, 3]
        volume = block[:, 4]
    else:
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        volume = df["volume"].to_numpy(dtype=np.float64)

    typical = (high + low + close) / 3.0
    raw_money_flow = typical * volume
    diff = np.empty(n, dtype=np.float64)
    diff[0] = np.nan
    diff[1:] = typical[1:] - typical[:-1]
    positive = np.where(diff > 0.0, raw_money_flow, 0.0)
    negative = np.where(diff < 0.0, raw_money_flow, 0.0)
    sums = (
        pd.DataFrame(np.column_stack((positive, negative)), copy=False)
        .rolling(length)
        .sum()
        .to_numpy()
    )
    psum = sums[:, 0]
    nsum = sums[:, 1]
    with np.errstate(invalid="ignore"):
        mfi = 100 * psum / (psum + nsum)
    return pd.Series(mfi, index=df.index, name=f"MFI_{length}")


def _mfi_fast_ok(df: pd.DataFrame) -> bool:
    """True when the port's float64 arithmetic is exactly the library's own."""
    import numpy as np

    if not df.index.is_unique:  # the library assigns by label
        return False
    cols = df.columns
    if len(cols) == 5 and tuple(cols) == _PTA_COLUMNS:
        return bool(df.to_numpy(copy=False).dtype == np.float64)
    return all(df[col].dtype == "float64" for col in ("high", "low", "close", "volume"))


def compute_mfi(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """Money Flow Index — numpy fast path over the same pandas arithmetic.

    Falls back to ``pandas_ta_classic.mfi`` for non-float64 columns or a
    non-unique index. ``None`` below ``length`` rows, exactly like the library.
    """
    length = int_param(params, "period", 14)
    if _mfi_fast_ok(df):
        return cast("pd.Series", _mfi_port(df, length))
    return cast(
        "pd.Series",
        _ta().mfi(
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            length=length,
        ),
    )


def compute_obv(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:  # noqa: ARG001
    return cast("pd.Series", _ta().obv(df["close"], df["volume"]))


def compute_vwap(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:  # noqa: ARG001
    """VWAP — pandas-ta-classic requires a DatetimeIndex for its groupby.

    If the input frame has a plain integer index (as the compiler's rolling
    buffer does), we synthesize a minute-resolution datetime index so the
    groupby-cumsum works, using a shallow copy (column data is shared) to
    avoid a per-bar deep-copy of the OHLCV buffer. The result series is
    returned with the caller's original index so downstream code stays
    indexing-agnostic.
    """
    import pandas as pd

    if not isinstance(df.index, pd.DatetimeIndex):
        synthetic = pd.date_range("2000-01-01", periods=len(df), freq="min")
        view = df.copy(deep=False)
        view.index = synthetic
        result = _ta().vwap(view["high"], view["low"], view["close"], view["volume"])
        if result is None:
            return nan_like(df)
        return cast("pd.Series", pd.Series(result.to_numpy(), index=df.index))
    return cast("pd.Series", _ta().vwap(df["high"], df["low"], df["close"], df["volume"]))


def compute_volsma(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", _ta().sma(df["volume"], length=int_param(params, "period", 20)))


def _rma_consts(length: int) -> tuple[float, float]:
    base = 1.0 / length
    com = (1.0 - base) / base  # pandas get_center_of_mass(alpha)
    alpha = 1.0 / (1.0 + com)
    return alpha, 1.0 - alpha


def _rma_step(state: Any, inputs: Any, i: int) -> tuple[float, Any]:
    """One pandas ``ewm(adjust=False, ignore_na=False)`` step (shared by full and extend paths).

    The blend uses a fused multiply-add: pandas' arm64 wheel compiles
    ``old_wt * weighted + new_wt * cur`` with FMA contraction (verified bitwise
    on this platform). x86 wheels may not contract; the exactness test would
    flag it there.
    """
    from math import fma

    weighted, old_wt, alpha, factor = state
    cur = float(inputs[0][i])
    if weighted == weighted:
        old_wt *= factor  # interior NaN decays the old weight
        if cur == cur:
            if weighted != cur:
                weighted = fma(old_wt, weighted, alpha * cur)
                weighted /= old_wt + alpha
            old_wt = 1.0
    elif cur == cur:
        weighted = cur
    return weighted, (weighted, old_wt, alpha, factor)


def _first_valid(x: Any) -> int:
    """Position of the first non-NaN value (``len(x)`` when there is none)."""
    import numpy as np

    valid = ~np.isnan(x)
    return int(valid.argmax()) if valid.any() else len(x)


def _rma_full(inputs: Any, length: int) -> tuple[Any, Any]:
    import numpy as np

    x = inputs[0]
    n = len(x)
    alpha, factor = _rma_consts(length)
    fv = _first_valid(x)
    if fv + length > n:  # no SMA seed window: undefined everywhere (never extended)
        return np.full(n, np.nan), (float("nan"), 1.0, alpha, factor)
    lead = fv + length - 1  # seed position; everything before it is NaN
    seed_win = x[fv : fv + length]
    nan_mask = np.isnan(seed_win)
    count = length - int(nan_mask.sum())
    seed = float(np.where(nan_mask, 0.0, seed_win).sum() / count)
    vals = x.tolist()
    for i in range(lead):
        vals[i] = float("nan")
    vals[lead] = seed
    if seed == seed and not np.isnan(x[lead + 1 :]).any():
        # Fast path (no NaN after the seed): every step is an observation with
        # old_wt == factor, so _rma_step reduces to this recurrence (same
        # arithmetic; state after an observation has old_wt == 1.0).
        from math import fma

        denom = factor + alpha
        w = seed
        fast = [float("nan")] * lead
        fast.append(w)
        append = fast.append
        for c in vals[lead + 1 :]:
            if w != c:
                w = fma(factor, w, alpha * c) / denom
            append(w)
        return np.array(fast, dtype=np.float64), (w, 1.0, alpha, factor)
    seeded = (np.array(vals, dtype=np.float64),)
    out = [float("nan")] * n
    state: Any = (vals[0], 1.0, alpha, factor)
    out[0] = vals[0]
    for i in range(1, n):
        out[i], state = _rma_step(state, seeded, i)
    return np.array(out, dtype=np.float64), state


def _rma_np(x: Any, length: int) -> Any:
    """Port of ``pandas_ta_classic.rma`` (SMA-seeded Wilder smoothing, 0.8).

    Operation-for-operation with the library: the seed is the NaN-skipping mean
    (sum of NaN->0 slots / non-NaN count) of the ``length`` values starting at
    the FIRST non-NaN value (chained input such as dx has a NaN prefix), then
    ``ewm(alpha, adjust=False)`` with ``ignore_na=False`` as in pandas' cython
    ``ewm`` (alpha re-derived via com, interior NaN decays the old weight).
    Undefined (all NaN) when no full seed window exists.

    The per-bar rebuild is memoized on the actual input array: when ``x`` is a
    previously seen input plus one appended bar, only one :func:`_rma_step` runs
    (``prefix_memo``; kill switch ``VIBE_QUANT_PTA_MEMO=0``). A miss recomputes
    fully through the same step helper.
    """
    import numpy as np

    from vibe_quant.dsl.prefix_memo import memo_run

    x = np.asarray(x, dtype=np.float64)

    def full(inputs: Any) -> tuple[Any, Any]:
        return _rma_full(inputs, length)

    if len(x) - 1 < _first_valid(x) + length:  # seed not placed before the appended element
        return full((x,))[0]
    return memo_run(("rma", length), (x,), full, _rma_step)


def _wilder_start(x: Any) -> int:
    """First raw position summed into the Wilder seed (index 0 is always skipped)."""
    import numpy as np

    finite = np.flatnonzero(np.isfinite(x))
    return max(int(finite[0]), 1) if finite.size else len(x)


def _wilder_step(state: Any, inputs: Any, i: int) -> tuple[float, Any]:
    """One Wilder step (``value - value / length + raw``, NaN input carries forward).

    Plain Python floats: the library's loop never contracts to FMA either
    (and the self-check ``_adx_port_ok`` flags a numba-compiled platform that does).
    """
    value, length = state
    raw = float(inputs[0][i])
    if raw == raw:
        value = value - value / length + raw
    return value, (value, length)


def _wilder_full(inputs: Any, length: int) -> tuple[Any, Any]:
    import numpy as np

    x = inputs[0]
    n = len(x)
    start = _wilder_start(x)
    if start + length - 1 > n:  # no seed window: all NaN (never extended)
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


def _wilder_np(x: Any, length: int) -> Any:
    """Port of ``pandas_ta_classic.utils._wilder.wilder_smooth`` (TA-Lib seeding).

    Seed = ``np.nansum`` of the ``length - 1`` raw values after the first finite
    one (never position 0), placed at ``start + length - 2``; then
    ``value - value / length + raw`` with NaN raw carrying the value forward.
    Memoized like :func:`_rma_np` (extension needs the seed inside the prefix).
    """
    import numpy as np

    from vibe_quant.dsl.prefix_memo import memo_run

    x = np.asarray(x, dtype=np.float64)
    start = _wilder_start(x)

    def full(inputs: Any) -> tuple[Any, Any]:
        return _wilder_full(inputs, length)

    n = len(x)
    if n - 1 < start + length - 1:  # seed not in the prefix
        return full((x,))[0]
    return memo_run(("wilder", length), (x,), full, _wilder_step)


def _adx_pandas(df: pd.DataFrame, length: int) -> pd.Series:
    """Reference path: the pandas-ta-classic ADX column (the pre-port code)."""
    result = _ta().adx(df["high"], df["low"], df["close"], length=length)
    if result is None:
        return nan_like(df)
    return cast("pd.Series", result.iloc[:, 0])


@cache
def _adx_port_ok() -> bool:
    """One-shot runtime check that the numpy port is bit-identical here.

    The rma blend relies on pandas' compiled ``ewm`` contracting it into an FMA
    (true on pandas 3.0.3 arm64 wheels), and the Wilder recurrence on the
    library loop not being FMA-contracted (it is plain Python unless numba is
    installed). Other platforms or versions may differ by ~1 ulp per step, so
    verify once per process and fall back to the pandas path (with a warning)
    on any mismatch.
    """
    import logging

    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(12345)
    close = 100.0 + np.cumsum(rng.randn(120) * 0.5)
    high = close + np.abs(rng.randn(120)) * 0.4
    low = close - np.abs(rng.randn(120)) * 0.4
    high[40:60] = low[40:60] = close[40:60] = close[40]  # flat stretch
    high[80] = np.nan
    df = pd.DataFrame({"high": high, "low": low, "close": close})
    ref = _adx_pandas(df, 14).to_numpy()
    ok = bool(np.array_equal(_adx_port(df, 14).to_numpy(), ref, equal_nan=True))
    if not ok:
        logging.getLogger(__name__).warning(
            "numpy ADX port differs from pandas-ta here (FMA/pandas version); "
            "falling back to the slower pandas path"
        )
    return ok


def compute_adx(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """ADX — bit-identical numpy port of ``pandas_ta_classic.adx`` (ADX column).

    Falls back to pandas-ta for non-float64 input, a non-unique index (the
    library blanks the DI seed bar by label) or if the one-shot self-check
    (``_adx_port_ok``) fails on this platform.
    """
    length = int_param(params, "period", 14)
    if (
        any(df[c].dtype != "float64" for c in ("high", "low", "close"))
        or not df.index.is_unique
        or not _adx_port_ok()
    ):
        return _adx_pandas(df, length)
    return _adx_port(df, length if length > 0 else 14)


def _adx_port(df: pd.DataFrame, length: int) -> pd.Series:
    """Numpy port of ``ta.adx`` (rma mode, drift 1): true_range -> Wilder-smoothed
    TR / +DM / -DM -> DI -> dx -> rma(dx); see
    ``tests/unit/test_plugins/test_adx_exactness.py``. Any arithmetic change must
    keep that test (zero tolerance) green.
    """
    import numpy as np
    import pandas as pd

    n = len(df)
    name = f"ADX_{length}"
    if n < length:
        return nan_like(df)
    eps = float(np.finfo(np.float64).eps)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    nan = np.full(1, np.nan)
    with np.errstate(all="ignore"):
        # true_range: pointwise non_zero_range, skipna max, rows up to the first
        # all-finite bar (plus the drift) blanked.
        hl = high - low
        hl = np.where(hl != 0, hl, eps)
        prev_close = np.concatenate((nan, close[:-1]))
        tr = np.fmax(np.fmax(np.abs(hl), np.abs(high - prev_close)), np.abs(prev_close - low))
        finite = ~(np.isnan(high) | np.isnan(low) | np.isnan(close))
        tr[: (int(finite.argmax()) if finite.any() else n) + 1] = np.nan
        tr_s = _wilder_np(tr, length)

        prev_high = np.concatenate((nan, high[:-1]))
        prev_low = np.concatenate((nan, low[:-1]))
        up = high - prev_high
        dn = prev_low - low
        pos = ((up > dn) & (up > 0)).astype(np.float64) * up
        neg = ((dn > up) & (dn > 0)).astype(np.float64) * dn
        pos = np.where(np.abs(pos) < eps, 0.0, pos)
        neg = np.where(np.abs(neg) < eps, 0.0, neg)

        dmp = 100.0 * _wilder_np(pos, length) / tr_s
        dmn = 100.0 * _wilder_np(neg, length) / tr_s
        seeded = np.flatnonzero(~np.isnan(tr_s))
        if seeded.size:  # wilder_di: the seed bar itself is not reported
            dmp[seeded[0]] = np.nan
            dmn[seeded[0]] = np.nan
        dx = 100.0 * np.abs(dmp - dmn) / (dmp + dmn)
        adx = _rma_np(dx, length)
    return pd.Series(adx, index=df.index, name=name)


# ---------------------------------------------------------------------------
# Multi-output indicators — dict return
# ---------------------------------------------------------------------------


def _macd_ta(df: pd.DataFrame, fast: int, slow: int, signal: int) -> dict[str, pd.Series]:
    """Reference path: the pandas-ta-classic MACD columns (the pre-port code)."""
    result = _ta().macd(df["close"], fast=fast, slow=slow, signal=signal)
    if result is None:
        empty = nan_like(df)
        return {"macd": empty, "histogram": empty, "signal": empty}
    return {
        "macd": result.iloc[:, 0],
        "histogram": result.iloc[:, 1],
        "signal": result.iloc[:, 2],
    }


def _ema_step(state: Any, inputs: Any, i: int) -> tuple[float, Any]:
    """One ``_ema_aligned`` recurrence step; ``state = (k, prev)``, no FMA."""
    k, prev = state
    v = k * float(inputs[0][i]) + (1 - k) * prev
    return v, (k, v)


def _ema_aligned_np(arr: Any, period: int, seed_end: int) -> Any:
    """Port of ``pandas_ta_classic.momentum.macd._ema_aligned``.

    NumPy ``.mean()`` seed over the ``period`` bars ending at ``seed_end``, then
    ``k*x + (1-k)*prev`` on plain floats (Python never contracts to FMA, like the
    library's scalar loop). Memoized on the actual input array (``prefix_memo``).
    """
    import numpy as np

    from vibe_quant.dsl.prefix_memo import memo_run

    m = len(arr)
    k = 2.0 / (period + 1)

    def full(inputs: Any) -> tuple[Any, Any]:
        x = inputs[0]
        out = np.full(m, np.nan)
        if seed_end - period + 1 < 0 or seed_end >= m:
            return out, (k, float("nan"))
        prev = float(x[seed_end - period + 1 : seed_end + 1].mean())
        out[seed_end] = prev
        k1 = 1 - k
        res: list[float] = []
        append = res.append
        for c in x[seed_end + 1 :].tolist():
            prev = k * c + k1 * prev
            append(prev)
        out[seed_end + 1 :] = res
        return out, (k, prev)

    # an extension is only valid once the seed was placed before the new bar
    if m - 1 <= seed_end or seed_end - period + 1 < 0:
        return full((arr,))[0]
    return memo_run(("ema_aligned", period, seed_end), (arr,), full, _ema_step)


def _macd_port(close: pd.Series, fast: int, slow: int, signal: int) -> dict[str, pd.Series]:
    """Numpy port of ``ta.macd`` (default, non-talib, non-asmode path).

    Mirrors the library: non-positive periods -> 12/26/9, slow<fast swapped,
    fast/slow EMAs seeded at their own lookback, signal EMA seeded at
    ``slow-1+signal-1`` over the MACD line. See
    ``tests/unit/test_plugins/test_macd_exactness.py`` (zero tolerance).
    """
    import numpy as np
    import pandas as pd

    fast = fast if fast > 0 else 12
    slow = slow if slow > 0 else 26
    signal = signal if signal > 0 else 9
    if slow < fast:
        fast, slow = slow, fast
    c = close.to_numpy(dtype=np.float64)
    fast_ema = _ema_aligned_np(c, fast, fast - 1)
    slow_ema = _ema_aligned_np(c, slow, slow - 1)
    macd = fast_ema - slow_ema
    sig = _ema_aligned_np(macd, signal, slow - 1 + signal - 1)
    hist = macd - sig
    idx = close.index
    props = f"_{fast}_{slow}_{signal}"
    return {
        "macd": pd.Series(macd, index=idx, name=f"MACD{props}"),
        "histogram": pd.Series(hist, index=idx, name=f"MACDh{props}"),
        "signal": pd.Series(sig, index=idx, name=f"MACDs{props}"),
    }


@cache
def _macd_port_ok() -> bool:
    """One-shot runtime check that the numpy MACD port is bit-identical here.

    Compared on fixed vectors (random walk, flat stretch, NaN gap; normal and
    swapped periods) against ``ta.macd``; any mismatch falls back to it.
    """
    import logging

    import numpy as np
    import pandas as pd

    from vibe_quant.dsl import prefix_memo

    rng = np.random.RandomState(4321)
    walk = 100.0 + np.cumsum(rng.randn(150) * 0.5)
    flat = walk.copy()
    flat[50:80] = flat[50]
    gap = walk.copy()
    gap[100] = np.nan
    ok = True
    # memo off: the check neither reads nor writes the shared (e.g. ADX) entries
    was_enabled = prefix_memo._enabled
    prefix_memo._enabled = False
    try:
        for arr in (walk, flat, gap):
            df = pd.DataFrame({"close": arr})
            for fast, slow, sig in ((12, 26, 9), (30, 8, 5), (5, 5, 3)):
                ref = _macd_ta(df, fast, slow, sig)
                got = _macd_port(df["close"], fast, slow, sig)
                for name in ref:
                    ok = ok and bool(
                        np.array_equal(got[name].to_numpy(), ref[name].to_numpy(), equal_nan=True)
                    )
    finally:
        prefix_memo._enabled = was_enabled
    if not ok:
        logging.getLogger(__name__).warning(
            "numpy MACD port differs from pandas-ta here; falling back to ta.macd"
        )
    return ok


def compute_macd(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """MACD — returns ``{"macd", "signal", "histogram"}`` keyed series.

    Bit-identical numpy port of ``ta.macd`` (whose columns are ordered
    ``[MACD, histogram, signal]``); falls back to it for non-float64 close or if
    the one-shot self-check (``_macd_port_ok``) fails. Series shorter than the
    library's minimum length give all-NaN.
    """
    fast = int_param(params, "fast_period", 12)
    slow = int_param(params, "slow_period", 26)
    signal = int_param(params, "signal_period", 9)
    close = df["close"]
    if close.dtype != "float64" or not _macd_port_ok():
        return _macd_ta(df, fast, slow, signal)
    need = max(fast if fast > 0 else 12, slow if slow > 0 else 26, signal if signal > 0 else 9)
    if len(close) < need:
        empty = nan_like(df)
        return {"macd": empty, "histogram": empty, "signal": empty}
    return _macd_port(close, fast, slow, signal)


def compute_stoch(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Stochastic — ``{"k", "d"}`` keyed series.

    The DSL fields ``period``/``d_period`` win over the spec-default aliases
    ``period_k``/``period_d`` (the compiler drops the aliases once the DSL
    field is set, so a swept ``stoch_period`` is what gets computed).
    """
    k_period = int_param(params, "period", int_param(params, "period_k", 14))
    d_period = int_param(params, "d_period", int_param(params, "period_d", 3))
    # First valid %D is at ``k + smooth_k + d - 2`` rows (smooth_k fixed at
    # pandas-ta's 3); below that pandas-ta returns None after logging an inner
    # "Series has N rows" warning. Same NaN result, no call (pandas-ta:
    # k<=0 -> 14, d<=0 -> 3).
    if len(df) < (k_period if k_period > 0 else 14) + 3 + (d_period if d_period > 0 else 3) - 2:
        empty = nan_like(df)
        return {"k": empty, "d": empty}
    result = _ta().stoch(df["high"], df["low"], df["close"], k=k_period, d=d_period)
    if result is None:
        empty = nan_like(df)
        return {"k": empty, "d": empty}
    return {"k": result.iloc[:, 0], "d": result.iloc[:, 1]}


def compute_bbands(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Bollinger Bands — ``{"lower", "middle", "upper", "bandwidth", "percent_b"}``.

    pandas-ta-classic's ``ta.bbands`` column order: BBL, BBM, BBU, BBB, BBP
    (iloc 0/1/2/3/4).

    ``bandwidth`` and ``percent_b`` are derived from the returned bands with the
    exact formulas and guards of ``derived.compute_bandwidth`` /
    ``compute_percent_b`` on the NT path: bandwidth is a FRACTION
    ``(upper - lower) / middle`` (0.0 when ``middle <= 0``; the GA samples
    [0, 0.2]) and ``percent_b`` is 0.5 on zero-width bands. pandas-ta's own
    BBB (percent, x100) and BBP columns add an epsilon that breaks both on a flat
    window, so they are not used. Warm-up rows (bands NaN) stay NaN.

    The bands are built on the typical price ``(high + low + close) / 3`` — what
    NT's ``BollingerBands.update_raw(high, low, close)`` averages — so both paths
    read the same bands (feeding ``close`` put the upper band ~0.1% off on real
    bars). ``percent_b`` still compares the bar close with those bands, as
    ``derived.compute_percent_b`` does.
    """
    period = int_param(params, "period", 20)
    std_dev = float_param(params, "std_dev", 2.0)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    result = _ta().bbands(typical, length=period, std=std_dev)
    if result is None:
        empty = nan_like(df)
        return {
            "lower": empty,
            "middle": empty,
            "upper": empty,
            "bandwidth": empty,
            "percent_b": empty,
        }
    lower, middle, upper = result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]
    band_range = upper - lower
    bandwidth = (band_range / middle).where(middle > 0, 0.0).where(middle.notna())
    percent_b = ((df["close"] - lower) / band_range).where(band_range > 0, 0.5)
    return {
        "lower": lower,
        "middle": middle,
        "upper": upper,
        "bandwidth": bandwidth,
        "percent_b": percent_b.where(band_range.notna()),
    }


def compute_kc(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Keltner Channel — ``{"lower", "middle", "upper"}`` keyed series."""
    period = int_param(params, "period", 20)
    scalar = float_param(params, "atr_multiplier", 2.0)
    # pandas-ta returns None (after logging "Series has N rows") below ``period``
    # rows; at ``n == period`` it returns the EMA basis with NaN bands. Same NaN
    # result, no call. pandas-ta: length<=0 -> 20.
    if len(df) < (period if period > 0 else 20):
        empty = nan_like(df)
        return {"lower": empty, "middle": empty, "upper": empty}
    result = _ta().kc(df["high"], df["low"], df["close"], length=period, scalar=scalar)
    if result is None:
        empty = nan_like(df)
        return {"lower": empty, "middle": empty, "upper": empty}
    return {
        "lower": result.iloc[:, 0],
        "middle": result.iloc[:, 1],
        "upper": result.iloc[:, 2],
    }


def compute_donchian(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Donchian Channel — ``{lower, middle, upper, position, upper_prev, lower_prev}``.

    ``upper_prev``/``lower_prev`` are the PREVIOUS bar's channel
    (``upper.shift(1)`` / ``lower.shift(1)``) — highest high / lowest low of
    the N bars before the current bar — so ``close crosses_above
    donchian.upper_prev`` is a genuine breakout. They are opt-in: the spec
    omits them from ``nt_output_attrs``/``computed_outputs``, so only a
    strategy referencing them is forced onto this compute_fn path and
    ``upper``/``middle``/``lower``/``position`` keep their NT values otherwise.

    ``position`` = ``(close - lower) / (upper - lower)`` (0.5 when the channel
    has zero width) — identical to ``derived.compute_position`` on the NT path
    (same-bar bands), so a strategy mixing ``donchian.position`` with a
    ``*_prev`` output (compute_fn path) reads the same value as the NT path.
    """
    period = int_param(params, "period", 20)
    result = _ta().donchian(df["high"], df["low"], lower_length=period, upper_length=period)
    if result is None:
        empty = nan_like(df)
        return {
            "lower": empty,
            "middle": empty,
            "upper": empty,
            "position": empty,
            "upper_prev": empty,
            "lower_prev": empty,
        }
    lower = result.iloc[:, 0]
    upper = result.iloc[:, 2]
    channel_range = upper - lower
    # Zero-width channel -> neutral 0.5 (mirrors derived.compute_position).
    position = ((df["close"] - lower) / channel_range).where(channel_range > 0, 0.5)
    # NaN warm-up rows must stay NaN (``where`` would turn them into 0.5).
    position = position.where(channel_range.notna())
    return {
        "lower": lower,
        "middle": result.iloc[:, 1],
        "upper": upper,
        "position": position,
        "upper_prev": upper.shift(1),
        "lower_prev": lower.shift(1),
    }


def compute_ichimoku(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Ichimoku Cloud — ``{"conversion", "base", "span_a", "span_b"}``.

    pandas-ta-classic's ``ta.ichimoku(as_dataframe=True)`` returns the visible
    period frame (the legacy tuple's ``core_df``); its columns are ``ISA_t, ISB_k, ITS_t, IKS_k, ICS_k``. Outputs map
    BY NAME: conversion = Tenkan-sen (ITS), base = Kijun-sen (IKS), span_a /
    span_b = Senkou spans as plotted AT the current bar (ISA/ISB, i.e. the
    cloud price trades against, computed ``kijun`` bars ago). The projected
    frame (cloud ahead of price) and Chikou (ICS, built from FUTURE closes)
    are never exposed. Before vibe-quant-e70tl.23 positional indexing
    labelled span A as "conversion", span B as "base" and the forward
    projection as the spans.
    """
    tenkan = int_param(params, "tenkan", 9)
    kijun = int_param(params, "kijun", 26)
    senkou = int_param(params, "senkou", 52)
    # as_dataframe=True: the visible-period frame only (None below max(tenkan,
    # kijun, senkou) rows); the default tuple return is deprecated.
    core = _ta().ichimoku(
        df["high"],
        df["low"],
        df["close"],
        tenkan=tenkan,
        kijun=kijun,
        senkou=senkou,
        as_dataframe=True,
    )
    empty = nan_like(df)
    if core is None:
        return {"conversion": empty, "base": empty, "span_a": empty, "span_b": empty}

    def col(prefix: str) -> pd.Series:
        for name in core.columns:
            if str(name).startswith(prefix):
                return cast("pd.Series", core[name])
        return empty

    return {
        "conversion": col("ITS_"),
        "base": col("IKS_"),
        "span_a": col("ISA_"),
        "span_b": col("ISB_"),
    }
