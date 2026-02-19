"""Translate frontend DslConfig format to canonical StrategyDSL format.

The frontend visual/YAML editor stores strategies in a nested DslConfig format:
  {general: {...}, indicators: [...], conditions: {...}, risk: {...}, time: {...}}

The backend StrategyDSL expects a flat canonical format:
  {name, timeframe, indicators: {name: config}, entry_conditions, stop_loss, take_profit}

This module bridges the two formats so frontend-created strategies can be backtested.
"""

from __future__ import annotations

from typing import Any


def _is_dsl_config_format(raw: dict[str, Any]) -> bool:
    """Return True if raw uses the frontend DslConfig nested format."""
    return "general" in raw or (
        isinstance(raw.get("indicators"), list) and "general" not in raw
    )


def _generate_indicator_name(ind_type: str, params: dict[str, Any], existing: set[str]) -> str:
    """Generate a unique snake_case indicator name from type + params."""
    base = ind_type.lower()
    period = params.get("period")
    if period is not None:
        base = f"{base}_{int(period)}"
    name = base
    counter = 2
    while name in existing:
        name = f"{base}_{counter}"
        counter += 1
    return name


def _translate_indicators(indicators_raw: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert DslConfig indicator array to StrategyDSL indicators dict."""
    result: dict[str, dict[str, Any]] = {}
    for ind in indicators_raw:
        ind_type = str(ind.get("type", "EMA")).upper()
        params = ind.get("params", {}) or {}
        name = _generate_indicator_name(ind_type, params, set(result.keys()))

        config: dict[str, Any] = {"type": ind_type}

        # Map flat params to StrategyDSL field names
        if "period" in params:
            config["period"] = int(params["period"])
        # MACD
        if "fast" in params:
            config["fast_period"] = int(params["fast"])
        if "slow" in params:
            config["slow_period"] = int(params["slow"])
        if "signal" in params:
            config["signal_period"] = int(params["signal"])
        # Bollinger Bands
        if "std_dev" in params:
            config["std_dev"] = float(params["std_dev"])
        # ATR-based
        if "atr_multiplier" in params:
            config["atr_multiplier"] = float(params["atr_multiplier"])

        result[name] = config
    return result


def _condition_to_str(cond: dict[str, Any] | str) -> str:
    """Convert a DslCondition object or raw string to expression string."""
    if isinstance(cond, str):
        return cond
    left = cond.get("left", "price")
    op = cond.get("operator", ">")
    right = cond.get("right", "price")
    return f"{left} {op} {right}"


def _translate_conditions(conditions: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (long_entry_conditions, short_entry_conditions)."""
    long_conds: list[str] = []
    short_conds: list[str] = []

    if isinstance(conditions.get("long_entry"), list):
        long_conds = [_condition_to_str(c) for c in conditions["long_entry"]]
    if isinstance(conditions.get("short_entry"), list):
        short_conds = [_condition_to_str(c) for c in conditions["short_entry"]]

    # Fall back: generic entry → long
    if not long_conds and not short_conds and isinstance(conditions.get("entry"), list):
        long_conds = [_condition_to_str(c) for c in conditions["entry"]]

    return long_conds, short_conds


def _translate_stop_loss(sl_raw: dict[str, Any]) -> dict[str, Any]:
    """Translate DslStopLoss {type, value} to StrategyDSL StopLossConfig."""
    sl_type = sl_raw.get("type", "fixed_pct")
    value = float(sl_raw.get("value", 2.0))
    result: dict[str, Any] = {"type": sl_type}
    if sl_type == "fixed_pct":
        result["percent"] = value
    elif sl_type in ("atr_fixed", "atr_trailing"):
        result["atr_multiplier"] = value
    return result


def _translate_take_profit(tp_raw: dict[str, Any]) -> dict[str, Any]:
    """Translate DslTakeProfit {type, value} to StrategyDSL TakeProfitConfig."""
    tp_type = tp_raw.get("type", "fixed_pct")
    value = float(tp_raw.get("value", 4.0))
    result: dict[str, Any] = {"type": tp_type}
    if tp_type == "fixed_pct":
        result["percent"] = value
    elif tp_type == "atr_fixed":
        result["atr_multiplier"] = value
    elif tp_type == "risk_reward":
        result["risk_reward_ratio"] = value
    return result


def translate_dsl_config(raw: dict[str, Any], strategy_name: str = "strategy") -> dict[str, Any]:
    """Translate frontend DslConfig dict to canonical StrategyDSL dict.

    If the input is already in StrategyDSL format (no 'general' key, indicators is a dict),
    it is returned unchanged.

    Args:
        raw: The dsl_config dict from the database.
        strategy_name: The strategy name (from the strategies table, not dsl_config).

    Returns:
        Dict suitable for StrategyDSL.model_validate().
    """
    if not _is_dsl_config_format(raw):
        # Already canonical StrategyDSL format
        return raw

    general = raw.get("general", {}) or {}
    timeframe = str(general.get("timeframe", "1h"))
    additional_tfs = list(general.get("additional_timeframes") or [])

    # Indicators
    indicators_raw = raw.get("indicators", []) or []
    if isinstance(indicators_raw, list):
        indicators = _translate_indicators(indicators_raw)
    else:
        indicators = indicators_raw  # already dict

    # Conditions
    conditions_raw = raw.get("conditions", {}) or {}
    long_entry, short_entry = _translate_conditions(conditions_raw)

    # Ensure at least one condition (StrategyDSL requires it)
    if not long_entry and not short_entry:
        long_entry = ["price > price"]  # trivially-true placeholder

    # Risk
    risk = raw.get("risk", {}) or {}
    sl_raw = risk.get("stop_loss", {}) or {}
    tp_raw = risk.get("take_profit", {}) or {}
    stop_loss = _translate_stop_loss(sl_raw)
    take_profit = _translate_take_profit(tp_raw)

    # Time filters
    time_raw = raw.get("time", {}) or {}
    time_filters: dict[str, Any] = {}
    if time_raw.get("funding_avoidance"):
        time_filters["avoid_around_funding"] = {"enabled": True}
    if time_raw.get("trading_hours"):
        hours = time_raw["trading_hours"]
        time_filters["allowed_sessions"] = [
            {"start": hours.get("start", "00:00"), "end": hours.get("end", "23:59")}
        ]
    if time_raw.get("trading_days"):
        time_filters["blocked_days"] = [
            d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if d not in (time_raw.get("trading_days") or [])
        ]

    result: dict[str, Any] = {
        "name": strategy_name,
        "timeframe": timeframe,
        "indicators": indicators,
        "entry_conditions": {
            "long": long_entry,
            "short": short_entry,
        },
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }
    if additional_tfs:
        result["additional_timeframes"] = additional_tfs
    if time_filters:
        result["time_filters"] = time_filters

    return result
