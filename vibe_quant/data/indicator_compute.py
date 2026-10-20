"""On-demand indicator computation from OHLCV data using pandas-ta-classic.

Used by the /api/data/indicators/{symbol} endpoint to compute indicator
series for chart overlays. No storage — computed fresh each request.

Registry-fallback indicators (plugins and compute_fn-only built-ins) replay
the strategy runtime's capped-buffer schedule: the generated strategy feeds
each compute_fn a rolling ``PtaBuffer`` (trim to the cap once it exceeds
``cap + cap // 4``) and reads the last value per bar, so the overlay must
reproduce that stream rather than compute over the full request history.
Hard-coded pandas-ta branches below keep their full-history formulas.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from vibe_quant.api.schemas.data import IndicatorSeries, IndicatorSeriesPoint
from vibe_quant.dsl.indicators import indicator_registry, pta_buffer_cap, pta_lookback
from vibe_quant.dsl.pta_replay import replay_pta_series

logger = logging.getLogger(__name__)


class _UnsupportedIndicatorError(Exception):
    """Requested type has no OHLCV-only computation path.

    Raised by :func:`_compute_single` for a registry spec that needs non-price
    context or only exists as a NautilusTrader runtime object, and for a type
    that is not in the registry at all. :func:`compute_indicators` turns it into
    an explicit per-indicator ``unsupported`` marker instead of dropping it.
    """


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
        typical = (high + low + close) / 3.0
        result = df.ta.bbands(close=typical, length=period, std=std_dev)
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
            # pandas-ta-classic names: KCL*/KCB*/KCU* (lower/basis/upper), DCL*/DCM*/DCU*
            if col_lower.startswith(("kcl", "dcl")):
                outputs.append(("lower", result[col]))
            elif col_lower.startswith(("kcu", "dcu")):
                outputs.append(("upper", result[col]))
            elif col_lower.startswith(("kcb", "dcm")):
                outputs.append(("middle", result[col]))
        return outputs

    if upper == "ICHIMOKU":
        tenkan = params.get("tenkan", 9)
        kijun = params.get("kijun", 26)
        senkou = params.get("senkou", 52)
        # the accessor returns the visible-period DataFrame (pandas-ta-classic >= 0.8)
        result = df.ta.ichimoku(tenkan=tenkan, kijun=kijun, senkou=senkou)
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

    # Generic registry fallback: any registered type without a hard-coded
    # branch above is computed through the spec's own compute_fn — the same
    # function (and params) the strategy compiler calls, replayed over the
    # runtime's capped buffer schedule and warmup gate. Specs that cannot run
    # on OHLCV alone raise _UnsupportedIndicatorError, which compute_indicators
    # reports explicitly.
    spec = indicator_registry.get(upper)
    if spec is None:
        logger.warning("Unknown indicator type for computation: %s", indicator_type)
        raise _UnsupportedIndicatorError("unknown indicator type")
    if spec.needs_context:
        raise _UnsupportedIndicatorError(
            "needs external context (not available in chart overlays)"
        )
    if spec.compute_fn is None:
        raise _UnsupportedIndicatorError(
            "computed by NautilusTrader at runtime; no chart formula"
        )

    # Merge the spec's defaults under the request params — the same base the
    # compiler's StrategyCompiler._merge_effective_params starts from. Without
    # this, a request that omits params ({} or a partial set) lets a plugin's
    # internal fallback (e.g. PRICE_POSITION's period 14) differ from the
    # spec default the strategy path uses (52), so chart and strategy values
    # diverge. Response params still echo the request verbatim.
    merged = {**spec.default_params, **params}
    cap = pta_buffer_cap([pta_lookback(spec, merged)], full_history=spec.requires_full_history)
    ohlcv = df[["open", "high", "low", "close", "volume"]].astype("float64")
    replayed = replay_pta_series(spec, ohlcv, merged, cap)
    return [
        (name, pd.Series(replayed[name], index=df.index))
        for name in spec.output_names
        if name in replayed
    ]


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
        except _UnsupportedIndicatorError as exc:
            spec = indicator_registry.get(ind_type)
            output_name = spec.output_names[0] if spec is not None else "value"
            results.append(
                IndicatorSeries(
                    name=base_name,
                    output_name=output_name,
                    indicator_type=ind_type,
                    display_label=_make_display_label(ind_type, output_name, params),
                    pane=pane,
                    params=params,
                    data=[],
                    unsupported=True,
                    unsupported_reason=str(exc),
                )
            )
            continue
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
