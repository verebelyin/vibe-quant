"""On-demand indicator computation from OHLCV data using pandas-ta-classic.

Used by the /api/data/indicators/{symbol} endpoint to compute indicator
series for chart overlays. No storage — computed fresh each request.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from vibe_quant.api.schemas.data import IndicatorSeries, IndicatorSeriesPoint
from vibe_quant.dsl.indicators import indicator_registry

logger = logging.getLogger(__name__)


def _classify_pane(indicator_type: str) -> str:
    """Classify indicator as overlay or oscillator.

    Reads ``IndicatorSpec.chart_placement`` directly from the registry
    (post-P7 single source of truth). Falls back to ``"oscillator"`` for
    unknown types — matches the pre-refactor behavior for anything that
    slipped through the old OVERLAY/OSCILLATOR sets.
    """
    spec = indicator_registry.get(indicator_type.upper())
    if spec is None:
        return "oscillator"
    return spec.chart_placement


def _make_display_label(indicator_type: str, output_name: str, params: dict[str, Any]) -> str:
    """Build human-readable label like 'EMA(20)' or 'BB Upper(20, 2.0)'."""
    upper = indicator_type.upper()
    period = params.get("period")

    if upper == "BBANDS":
        std = params.get("std_dev", 2.0)
        label_map = {"upper": "BB Upper", "middle": "BB Mid", "lower": "BB Lower"}
        prefix = label_map.get(output_name, "BB")
        return f"{prefix}({period}, {std})"
    if upper == "MACD":
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        sig = params.get("signal_period", 9)
        label_map = {"macd": "MACD", "signal": "Signal", "histogram": "Histogram"}
        prefix = label_map.get(output_name, "MACD")
        return f"{prefix}({fast},{slow},{sig})"
    if upper == "STOCH":
        k_period = params.get("period", params.get("period_k", 14))
        label_map = {"k": "%K", "d": "%D"}
        prefix = label_map.get(output_name, "STOCH")
        return f"{prefix}({k_period})"
    if upper in {"KC", "DONCHIAN"}:
        label_map = {"upper": f"{upper} Upper", "middle": f"{upper} Mid", "lower": f"{upper} Lower"}
        prefix = label_map.get(output_name, upper)
        return f"{prefix}({period})"
    if upper == "ICHIMOKU":
        tenkan = params.get("tenkan", 9)
        label_map = {
            "conversion": "Tenkan",
            "base": "Kijun",
            "span_a": "Span A",
            "span_b": "Span B",
        }
        prefix = label_map.get(output_name, "ICH")
        return f"{prefix}({tenkan})"
    if period is not None:
        return f"{upper}({period})"
    return upper


def _safe_value(v: Any) -> float | None:
    """Convert to float or None for NaN/inf."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _series_to_points(times: pd.Series, values: pd.Series) -> list[IndicatorSeriesPoint]:
    """Convert pandas series pair to list of IndicatorSeriesPoint."""
    points: list[IndicatorSeriesPoint] = []
    for t, v in zip(times, values, strict=True):
        points.append(IndicatorSeriesPoint(time=int(t), value=_safe_value(v)))
    return points


def _compute_single(
    df: pd.DataFrame,
    indicator_type: str,
    params: dict[str, Any],
) -> list[tuple[str, pd.Series]]:
    """Compute a single indicator, returning list of (output_name, series) pairs."""
    import pandas_ta_classic  # noqa: F401 — registers .ta accessor

    upper = indicator_type.upper()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    if upper in {"SMA", "EMA", "WMA", "DEMA", "TEMA"}:
        period = params.get("period", 20)
        func = getattr(df.ta, upper.lower(), None)
        if func is None:
            return []
        result = func(close=close, length=period)
        if result is None:
            return []
        return [("value", result)]

    if upper == "RSI":
        period = params.get("period", 14)
        result = df.ta.rsi(close=close, length=period)
        if result is None:
            return []
        return [("value", result)]

    if upper == "MACD":
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        signal = params.get("signal_period", 9)
        result = df.ta.macd(close=close, fast=fast, slow=slow, signal=signal)
        if result is None:
            return []
        cols = result.columns.tolist()
        # pandas-ta returns columns like MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        outputs: list[tuple[str, pd.Series]] = []
        for col in cols:
            col_lower = col.lower()
            if "macdh" in col_lower or "histogram" in col_lower:
                outputs.append(("histogram", result[col]))
            elif "macds" in col_lower or "signal" in col_lower:
                outputs.append(("signal", result[col]))
            elif "macd" in col_lower:
                outputs.append(("macd", result[col]))
        return outputs

    if upper == "BBANDS":
        period = params.get("period", 20)
        std_dev = params.get("std_dev", 2.0)
        result = df.ta.bbands(close=close, length=period, std=std_dev)
        if result is None:
            return []
        cols = result.columns.tolist()
        outputs = []
        for col in cols:
            col_lower = col.lower()
            if "bbl" in col_lower:
                outputs.append(("lower", result[col]))
            elif "bbm" in col_lower:
                outputs.append(("middle", result[col]))
            elif "bbu" in col_lower:
                outputs.append(("upper", result[col]))
        return outputs

    if upper == "STOCH":
        k_period = params.get("period", params.get("period_k", 14))
        d_period = params.get("d_period", 3)
        result = df.ta.stoch(high=high, low=low, close=close, k=k_period, d=d_period)
        if result is None:
            return []
        cols = result.columns.tolist()
        outputs = []
        for col in cols:
            col_lower = col.lower()
            if "stochd" in col_lower or "d_" in col_lower:
                outputs.append(("d", result[col]))
            elif "stochk" in col_lower or "k_" in col_lower:
                outputs.append(("k", result[col]))
        return outputs

    if upper == "ATR":
        period = params.get("period", 14)
        result = df.ta.atr(high=high, low=low, close=close, length=period)
        if result is None:
            return []
        return [("value", result)]

    if upper == "ADX":
        period = params.get("period", 14)
        result = df.ta.adx(high=high, low=low, close=close, length=period)
        if result is None:
            return []
        for col in result.columns:
            if col.lower().startswith("adx_") and "dm" not in col.lower():
                return [("value", result[col])]
        return []

    if upper in {"CCI", "WILLR", "ROC", "MFI"}:
        period = params.get("period", 14)
        func = getattr(df.ta, upper.lower(), None)
        if func is None:
            return []
        if upper == "MFI":
            result = func(high=high, low=low, close=close, volume=volume, length=period)
        elif upper in {"CCI", "WILLR"}:
            result = func(high=high, low=low, close=close, length=period)
        else:
            result = func(close=close, length=period)
        if result is None:
            return []
        return [("value", result)]

    if upper == "OBV":
        result = df.ta.obv(close=close, volume=volume)
        if result is None:
            return []
        return [("value", result)]

    if upper == "VWAP":
        result = df.ta.vwap(high=high, low=low, close=close, volume=volume)
        if result is None:
            return []
        return [("value", result)]

    if upper in {"KC", "DONCHIAN"}:
        period = params.get("period", 20)
        if upper == "KC":
            atr_mult = params.get("atr_multiplier", 1.5)
            result = df.ta.kc(high=high, low=low, close=close, length=period, scalar=atr_mult)
        else:
            result = df.ta.donchian(high=high, low=low, lower_length=period, upper_length=period)
        if result is None:
            return []
        cols = result.columns.tolist()
        outputs = []
        for col in cols:
            col_lower = col.lower()
            if "lower" in col_lower or "dcl" in col_lower:
                outputs.append(("lower", result[col]))
            elif "upper" in col_lower or "dcu" in col_lower:
                outputs.append(("upper", result[col]))
            elif "mid" in col_lower or "basis" in col_lower or "dcm" in col_lower:
                outputs.append(("middle", result[col]))
        return outputs

    if upper == "ICHIMOKU":
        tenkan = params.get("tenkan", 9)
        kijun = params.get("kijun", 26)
        senkou = params.get("senkou", 52)
        result_tuple = df.ta.ichimoku(tenkan=tenkan, kijun=kijun, senkou=senkou)
        if result_tuple is None:
            return []
        # ichimoku returns (span_df, ...) — first element is the main DataFrame
        result = result_tuple[0] if isinstance(result_tuple, tuple) else result_tuple
        if result is None:
            return []
        cols = result.columns.tolist()
        outputs = []
        name_map = {
            "isa": "span_a",
            "isb": "span_b",
            "its": "conversion",
            "iks": "base",
        }
        for col in cols:
            col_lower = col.lower()
            for key, out_name in name_map.items():
                if key in col_lower:
                    outputs.append((out_name, result[col]))
                    break
        return outputs

    if upper == "VOLSMA":
        period = params.get("period", 20)
        result = df.ta.sma(close=volume, length=period)
        if result is None:
            return []
        return [("value", result)]

    logger.warning("Unknown indicator type for computation: %s", indicator_type)
    return []


def compute_indicators(
    df: pd.DataFrame,
    indicators: list[dict[str, Any]],
) -> list[IndicatorSeries]:
    """Compute multiple indicators from OHLCV DataFrame.

    Args:
        df: DataFrame with columns: open_time, open, high, low, close, volume
        indicators: List of indicator configs (dicts with 'type', 'period', etc.)

    Returns:
        List of IndicatorSeries ready for API response.
    """
    times = df["open_time"]
    results: list[IndicatorSeries] = []

    for ind_cfg in indicators:
        ind_type = ind_cfg.get("type", "").upper()
        if not ind_type:
            continue

        params = {k: v for k, v in ind_cfg.items() if k != "type" and v is not None}
        pane = _classify_pane(ind_type)

        period = params.get("period")
        name_suffix = f"_{period}" if period else ""
        base_name = f"{ind_type.lower()}{name_suffix}"

        try:
            outputs = _compute_single(df, ind_type, params)
        except Exception:
            logger.exception("Failed to compute indicator %s", ind_type)
            continue

        for output_name, series_data in outputs:
            display_label = _make_display_label(ind_type, output_name, params)
            points = _series_to_points(times, series_data)

            results.append(
                IndicatorSeries(
                    name=base_name,
                    output_name=output_name,
                    indicator_type=ind_type,
                    display_label=display_label,
                    pane=pane,
                    params=params,
                    data=points,
                )
            )

    return results
