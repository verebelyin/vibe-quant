"""Pure-Python compute_fn implementations for every built-in indicator.

Each function takes a rolling OHLCV DataFrame (columns: open, high, low,
close, volume) plus the merged params dict, and returns either a single
``pd.Series`` (for one-output indicators) or a ``dict[str, pd.Series]``
keyed by output name (for multi-output indicators).

These are referenced from ``vibe_quant/dsl/indicators.py`` via the
``IndicatorSpec.compute_fn`` field. The compiler ignores them until Phase 4
of the plugin-system refactor — Phase 3 populates the spec, Phase 4 deletes
the hardcoded elif chain in the compiler and dispatches through these
callbacks instead.

The return shape mirrors the per-indicator layout the compiler already
produces in its ``_generate_update_pta_indicators`` method, so Phase 4's
byte-identical compiled-source check holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas_ta_classic as ta

if TYPE_CHECKING:
    import pandas as pd


def _int_param(params: dict[str, object], key: str, default: int) -> int:
    val = params.get(key, default)
    return int(val) if isinstance(val, (int, float)) else default


def _float_param(params: dict[str, object], key: str, default: float) -> float:
    val = params.get(key, default)
    return float(val) if isinstance(val, (int, float)) else default


# ---------------------------------------------------------------------------
# Single-output indicators — Series return
# ---------------------------------------------------------------------------


def compute_rsi(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", ta.rsi(df["close"], length=_int_param(params, "period", 14)))


def compute_ema(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", ta.ema(df["close"], length=_int_param(params, "period", 14)))


def compute_sma(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", ta.sma(df["close"], length=_int_param(params, "period", 14)))


def compute_wma(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", ta.wma(df["close"], length=_int_param(params, "period", 14)))


def compute_dema(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", ta.dema(df["close"], length=_int_param(params, "period", 14)))


def compute_tema(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", ta.tema(df["close"], length=_int_param(params, "period", 14)))


def compute_cci(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast(
        "pd.Series",
        ta.cci(df["high"], df["low"], df["close"], length=_int_param(params, "period", 20)),
    )


def compute_willr(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast(
        "pd.Series",
        ta.willr(df["high"], df["low"], df["close"], length=_int_param(params, "period", 14)),
    )


def compute_roc(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", ta.roc(df["close"], length=_int_param(params, "period", 10)))


def compute_atr(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast(
        "pd.Series",
        ta.atr(df["high"], df["low"], df["close"], length=_int_param(params, "period", 14)),
    )


def compute_mfi(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast(
        "pd.Series",
        ta.mfi(
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            length=_int_param(params, "period", 14),
        ),
    )


def compute_obv(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:  # noqa: ARG001
    return cast("pd.Series", ta.obv(df["close"], df["volume"]))


def compute_vwap(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:  # noqa: ARG001
    """VWAP — pandas-ta-classic requires a DatetimeIndex for its groupby.

    If the input frame has a plain integer index (as the compiler's rolling
    buffer does), we synthesize a minute-resolution datetime index so the
    groupby-cumsum works. The result series is returned with the caller's
    original index to keep downstream code indexing-agnostic.
    """
    import pandas as pd

    if not isinstance(df.index, pd.DatetimeIndex):
        synthetic = pd.date_range("2000-01-01", periods=len(df), freq="min")
        view = df.copy()
        view.index = synthetic
        result = ta.vwap(view["high"], view["low"], view["close"], view["volume"])
        if result is None:
            return cast("pd.Series", df["close"] * 0)
        return cast("pd.Series", pd.Series(result.to_numpy(), index=df.index))
    return cast("pd.Series", ta.vwap(df["high"], df["low"], df["close"], df["volume"]))


def compute_volsma(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    return cast("pd.Series", ta.sma(df["volume"], length=_int_param(params, "period", 20)))


def compute_adx(df: pd.DataFrame, params: dict[str, object]) -> pd.Series:
    """ADX — pandas-ta returns a DataFrame with ADX/DMP/DMN; surface just ADX."""
    result = ta.adx(df["high"], df["low"], df["close"], length=_int_param(params, "period", 14))
    if result is None:
        return cast("pd.Series", df["close"] * 0)
    # First column is the ADX line itself, e.g. "ADX_14".
    return cast("pd.Series", result.iloc[:, 0])


# ---------------------------------------------------------------------------
# Multi-output indicators — dict return
# ---------------------------------------------------------------------------


def compute_macd(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """MACD — returns ``{"macd", "signal", "histogram"}`` keyed series.

    pandas-ta-classic's ``ta.macd`` returns a DataFrame with columns in the
    order ``[MACD, histogram, signal]`` (iloc 0/1/2). This matches the
    compiler's existing extraction path at ``compiler.py:1298-1317``.
    """
    fast = _int_param(params, "fast_period", 12)
    slow = _int_param(params, "slow_period", 26)
    signal = _int_param(params, "signal_period", 9)
    result = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if result is None:
        empty = df["close"] * 0
        return {"macd": empty, "histogram": empty, "signal": empty}
    return {
        "macd": result.iloc[:, 0],
        "histogram": result.iloc[:, 1],
        "signal": result.iloc[:, 2],
    }


def compute_stoch(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Stochastic — ``{"k", "d"}`` keyed series."""
    k_period = _int_param(params, "period_k", _int_param(params, "period", 14))
    d_period = _int_param(params, "period_d", 3)
    result = ta.stoch(df["high"], df["low"], df["close"], k=k_period, d=d_period)
    if result is None:
        empty = df["close"] * 0
        return {"k": empty, "d": empty}
    return {"k": result.iloc[:, 0], "d": result.iloc[:, 1]}


def compute_bbands(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Bollinger Bands — ``{"lower", "middle", "upper", "bandwidth", "percent_b"}``.

    pandas-ta-classic's ``ta.bbands`` column order: BBL, BBM, BBU, BBB, BBP
    (iloc 0/1/2/3/4). Matches ``compiler.py:1345-1365``.
    """
    period = _int_param(params, "period", 20)
    std_dev = _float_param(params, "std_dev", 2.0)
    result = ta.bbands(df["close"], length=period, std=std_dev)
    if result is None:
        empty = df["close"] * 0
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
    period = _int_param(params, "period", 20)
    scalar = _float_param(params, "atr_multiplier", 2.0)
    result = ta.kc(df["high"], df["low"], df["close"], length=period, scalar=scalar)
    if result is None:
        empty = df["close"] * 0
        return {"lower": empty, "middle": empty, "upper": empty}
    return {
        "lower": result.iloc[:, 0],
        "middle": result.iloc[:, 1],
        "upper": result.iloc[:, 2],
    }


def compute_donchian(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Donchian Channel — ``{"lower", "middle", "upper"}`` keyed series.

    Note: the derived output ``position`` is computed at runtime by
    ``vibe_quant/dsl/derived.py::compute_position`` from the raw bands plus
    the latest close, so it is not returned here.
    """
    period = _int_param(params, "period", 20)
    result = ta.donchian(df["high"], df["low"], lower_length=period, upper_length=period)
    if result is None:
        empty = df["close"] * 0
        return {"lower": empty, "middle": empty, "upper": empty}
    return {
        "lower": result.iloc[:, 0],
        "middle": result.iloc[:, 1],
        "upper": result.iloc[:, 2],
    }


def compute_ichimoku(df: pd.DataFrame, params: dict[str, object]) -> dict[str, pd.Series]:
    """Ichimoku Cloud — ``{"conversion", "base", "span_a", "span_b"}``.

    pandas-ta-classic's ``ta.ichimoku`` returns a tuple ``(ichimoku_df,
    span_df)``. The conversion/base lines are in the first DataFrame's
    columns 0/1 and the span_a/span_b lines are in the span DataFrame's
    columns 0/1. Matches ``compiler.py:1260-1286``.
    """
    tenkan = _int_param(params, "tenkan", 9)
    kijun = _int_param(params, "kijun", 26)
    senkou = _int_param(params, "senkou", 52)
    ichi = ta.ichimoku(
        df["high"], df["low"], df["close"], tenkan=tenkan, kijun=kijun, senkou=senkou
    )
    empty = df["close"] * 0
    if ichi is None or not isinstance(ichi, tuple) or len(ichi) < 1:
        return {"conversion": empty, "base": empty, "span_a": empty, "span_b": empty}
    core = ichi[0]
    span = ichi[1] if len(ichi) >= 2 else None
    return {
        "conversion": core.iloc[:, 0] if core is not None and core.shape[1] >= 1 else empty,
        "base": core.iloc[:, 1] if core is not None and core.shape[1] >= 2 else empty,
        "span_a": span.iloc[:, 0] if span is not None and span.shape[1] >= 1 else empty,
        "span_b": span.iloc[:, 1] if span is not None and span.shape[1] >= 2 else empty,
    }
