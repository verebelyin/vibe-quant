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
    return cast("pd.Series", _ta().dema(df["close"], length=int_param(params, "period", 14)))


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


def compute_mfi(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast(
        "pd.Series",
        _ta().mfi(
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            length=int_param(params, "period", 14),
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


def _rma_np(x: Any, length: int) -> Any:
    """Port of ``pandas_ta_classic.rma`` (SMA-seeded Wilder smoothing).

    Operation-for-operation with pandas: NaN-skipping seed mean (sum of
    NaN->0 slots / non-NaN count), then ``ewm(alpha, adjust=False)`` with
    ``ignore_na=False`` as in pandas' cython ``ewm`` (alpha re-derived via
    com, interior NaN decays the old weight). Requires ``len(x) >= length``.

    The blend uses a fused multiply-add: pandas' arm64 wheel compiles
    ``old_wt * weighted + new_wt * cur`` with FMA contraction (verified bitwise
    on this platform). x86 wheels may not contract; the exactness test would
    flag it there.
    """
    from math import fma

    import numpy as np

    n = len(x)
    seed_win = x[:length]
    nan_mask = np.isnan(seed_win)
    count = length - int(nan_mask.sum())
    seed = float(np.where(nan_mask, 0.0, seed_win).sum() / count) if count else float("nan")
    vals = x.tolist()
    for i in range(length - 1):
        vals[i] = float("nan")
    vals[length - 1] = seed
    base = 1.0 / length
    com = (1.0 - base) / base  # pandas get_center_of_mass(alpha)
    alpha = 1.0 / (1.0 + com)
    factor = 1.0 - alpha
    s = length - 1
    if seed == seed and not np.isnan(x[length:]).any():
        # Fast path (no NaN after the seed): every step is an observation with
        # old_wt == factor, so the general loop below reduces to this recurrence.
        denom = factor + alpha
        w = seed
        out = [float("nan")] * s
        out.append(w)
        append = out.append
        for c in vals[length:]:
            if w != c:
                w = fma(factor, w, alpha * c) / denom
            append(w)
        return np.array(out, dtype=np.float64)
    out = [float("nan")] * n
    weighted = vals[0]
    old_wt = 1.0
    out[0] = weighted
    for i in range(1, n):
        cur = vals[i]
        obs = cur == cur
        if weighted == weighted:
            old_wt *= factor
            if obs:
                if weighted != cur:
                    weighted = fma(old_wt, weighted, alpha * cur)
                    weighted /= old_wt + alpha
                old_wt = 1.0
        elif obs:
            weighted = cur
        out[i] = weighted
    return np.array(out, dtype=np.float64)


def _adx_pandas(df: pd.DataFrame, length: int) -> pd.Series:
    """Reference path: the pandas-ta-classic ADX column (the pre-port code)."""
    result = _ta().adx(df["high"], df["low"], df["close"], length=length)
    if result is None:
        return nan_like(df)
    return cast("pd.Series", result.iloc[:, 0])


@cache
def _adx_port_ok() -> bool:
    """One-shot runtime check that the numpy port is bit-identical here.

    The port relies on pandas' compiled ``ewm`` contracting the blend into an
    FMA (true on pandas 3.0.3 arm64 wheels). Other platforms or pandas versions
    may differ by ~1 ulp per step, so verify once per process and fall back to
    the pandas path (with a warning) on any mismatch.
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

    Falls back to pandas-ta for non-float64 input or if the one-shot
    self-check (``_adx_port_ok``) fails on this platform.
    """
    length = int_param(params, "period", 14)
    if any(df[c].dtype != "float64" for c in ("high", "low", "close")) or not _adx_port_ok():
        return _adx_pandas(df, length)
    return _adx_port(df, length if length > 0 else 14)


def _adx_port(df: pd.DataFrame, length: int) -> pd.Series:
    """Numpy port: true_range -> rma ATR, +DM/-DM via rma, dx, rma(dx); see
    ``tests/unit/test_plugins/test_adx_exactness.py``. Any arithmetic change
    must keep that test (zero tolerance) green.
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
        hl = high - low
        if (hl == 0).any():
            hl = hl + eps
        prev_close = np.concatenate((nan, close[:-1]))
        tr = np.fmax(np.fmax(np.abs(hl), np.abs(high - prev_close)), np.abs(prev_close - low))
        tr[0] = np.nan
        atr = _rma_np(tr, length)

        prev_high = np.concatenate((nan, high[:-1]))
        prev_low = np.concatenate((nan, low[:-1]))
        up = high - prev_high
        dn = prev_low - low
        pos = ((up > dn) & (up > 0)).astype(np.float64) * up
        neg = ((dn > up) & (dn > 0)).astype(np.float64) * dn
        pos = np.where(np.abs(pos) < eps, 0.0, pos)
        neg = np.where(np.abs(neg) < eps, 0.0, neg)

        k = 100.0 / atr
        dmp = k * _rma_np(pos, length)
        dmn = k * _rma_np(neg, length)
        dx = 100.0 * np.abs(dmp - dmn) / (dmp + dmn)
        adx = _rma_np(dx, length)
    return pd.Series(adx, index=df.index, name=name)


# ---------------------------------------------------------------------------
# Multi-output indicators — dict return
# ---------------------------------------------------------------------------


def compute_macd(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """MACD — returns ``{"macd", "signal", "histogram"}`` keyed series.

    pandas-ta-classic's ``ta.macd`` returns a DataFrame with columns in the
    order ``[MACD, histogram, signal]`` (iloc 0/1/2).
    """
    fast = int_param(params, "fast_period", 12)
    slow = int_param(params, "slow_period", 26)
    signal = int_param(params, "signal_period", 9)
    result = _ta().macd(df["close"], fast=fast, slow=slow, signal=signal)
    if result is None:
        empty = nan_like(df)
        return {"macd": empty, "histogram": empty, "signal": empty}
    return {
        "macd": result.iloc[:, 0],
        "histogram": result.iloc[:, 1],
        "signal": result.iloc[:, 2],
    }


def compute_stoch(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Stochastic — ``{"k", "d"}`` keyed series.

    The DSL fields ``period``/``d_period`` win over the spec-default aliases
    ``period_k``/``period_d`` (the compiler drops the aliases once the DSL
    field is set, so a swept ``stoch_period`` is what gets computed).
    """
    k_period = int_param(params, "period", int_param(params, "period_k", 14))
    d_period = int_param(params, "d_period", int_param(params, "period_d", 3))
    result = _ta().stoch(df["high"], df["low"], df["close"], k=k_period, d=d_period)
    if result is None:
        empty = nan_like(df)
        return {"k": empty, "d": empty}
    return {"k": result.iloc[:, 0], "d": result.iloc[:, 1]}


def compute_bbands(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Bollinger Bands — ``{"lower", "middle", "upper", "bandwidth", "percent_b"}``.

    pandas-ta-classic's ``ta.bbands`` column order: BBL, BBM, BBU, BBB, BBP
    (iloc 0/1/2/3/4).
    """
    period = int_param(params, "period", 20)
    std_dev = float_param(params, "std_dev", 2.0)
    result = _ta().bbands(df["close"], length=period, std=std_dev)
    if result is None:
        empty = nan_like(df)
        return {
            "lower": empty,
            "middle": empty,
            "upper": empty,
            "bandwidth": empty,
            "percent_b": empty,
        }
    return {
        "lower": result.iloc[:, 0],
        "middle": result.iloc[:, 1],
        "upper": result.iloc[:, 2],
        "bandwidth": result.iloc[:, 3],
        "percent_b": result.iloc[:, 4],
    }


def compute_kc(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Keltner Channel — ``{"lower", "middle", "upper"}`` keyed series."""
    period = int_param(params, "period", 20)
    scalar = float_param(params, "atr_multiplier", 2.0)
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
    """Donchian Channel — ``{"lower", "middle", "upper"}`` keyed series.

    The derived output ``position`` is computed at runtime by
    ``vibe_quant/dsl/derived.py::compute_position`` from the raw bands plus
    the latest close, so it is not returned here.
    """
    period = int_param(params, "period", 20)
    result = _ta().donchian(df["high"], df["low"], lower_length=period, upper_length=period)
    if result is None:
        empty = nan_like(df)
        return {"lower": empty, "middle": empty, "upper": empty}
    return {
        "lower": result.iloc[:, 0],
        "middle": result.iloc[:, 1],
        "upper": result.iloc[:, 2],
    }


def compute_ichimoku(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Ichimoku Cloud — ``{"conversion", "base", "span_a", "span_b"}``.

    pandas-ta-classic's ``ta.ichimoku`` returns ``(core_df, projected_df)``;
    ``core_df`` columns are ``ISA_t, ISB_k, ITS_t, IKS_k, ICS_k``. Outputs map
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
    ichi = _ta().ichimoku(
        df["high"], df["low"], df["close"], tenkan=tenkan, kijun=kijun, senkou=senkou
    )
    empty = nan_like(df)
    core = ichi[0] if isinstance(ichi, tuple) and len(ichi) >= 1 else None
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
