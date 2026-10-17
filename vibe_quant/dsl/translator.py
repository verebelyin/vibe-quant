"""Translate frontend DslConfig format to canonical StrategyDSL format.

The frontend visual/YAML editor stores strategies in a nested DslConfig format:
  {general: {...}, indicators: [...], conditions: {...}, risk: {...}, time: {...}}

The backend StrategyDSL expects a flat canonical format:
  {name, timeframe, indicators: {name: config}, entry_conditions, stop_loss, take_profit}

This module bridges the two formats so frontend-created strategies can be backtested.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

# Editor indicator types whose StrategyDSL registry name differs. Only the config `type`
# is renamed: generated indicator names and condition refs keep the editor spelling
# (the UI sweep builder mirrors `<editor type lowercased>_<period>`).
_TYPE_ALIASES = {"BB": "BBANDS"}

# Alternate spellings of one StrategyDSL param: legacy editor names (types.ts) and the registry
# `default_params` the catalog hands out. Canonical field -> accepted spellings.
_PARAM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "fast_period": ("fast_period", "fast"),
    "slow_period": ("slow_period", "slow"),
    "signal_period": ("signal_period", "signal"),
}
# STOCH's k/d periods; IndicatorConfig.normalize_param_aliases folds the same spellings
_STOCH_SYNONYMS: dict[str, tuple[str, ...]] = {
    "period": ("period", "k_period", "period_k"),
    "d_period": ("d_period", "period_d"),
}


def _is_dsl_config_format(raw: dict[str, Any]) -> bool:
    """Return True if raw uses the frontend DslConfig nested format."""
    return "general" in raw or (isinstance(raw.get("indicators"), list) and "general" not in raw)


_DECIMAL = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")


def _as_float(value: Any) -> float | None:
    """``float(value)`` when it is a finite number (or a numeric string), else ``None``.

    ``bool`` is excluded: ``float(True)`` is ``1.0``, but a flag is not a period. A
    non-finite value (``inf``/``nan``, including those strings) is not a number either, and
    neither is an int too large for a float. A string must be a plain decimal numeral, the
    one spelling the editor's ``Number()`` agrees on: ``float`` also takes ``1_000``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    if isinstance(value, str) and _DECIMAL.fullmatch(value.strip()) is None:
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _conflict_key(value: Any) -> Any:
    """Comparison key for two spellings of one param: numbers compare numerically, a bool
    never equals a number (``_as_float`` rejects ``True``, so it compares by ``repr``), anything
    else compares by ``repr`` so it never crashes the check."""
    number = _as_float(value)
    return number if number is not None else repr(value)


def _generate_indicator_name(ind_type: str, params: dict[str, Any], existing: set[str]) -> str:
    """Generate a unique snake_case indicator name from type + params."""
    base = ind_type.lower()
    period = params.get("period")
    if period is not None:
        # best-effort: a bad period still reaches ``_int_param`` and fails there with a message
        # naming the indicator and param, never a raw ``int()``/``TypeError``
        number = _as_float(period)
        if number is not None:
            base = f"{base}_{int(number)}"
    name = base
    counter = 2
    while name in existing:
        name = f"{base}_{counter}"
        counter += 1
    return name


def _native_param_names(ind_type: str) -> set[str]:
    """Param names the indicator's registry spec declares natively.

    A name the spec declares belongs to that indicator, so it is not a legacy
    editor alias for a canonical DSL field (a plugin's ``fast``). See
    ``_canonical_params``.
    """
    from vibe_quant.dsl.indicators import indicator_registry

    spec = indicator_registry.get(_TYPE_ALIASES.get(ind_type, ind_type))
    return set(spec.param_schema) if spec is not None else set()


def _canonical_params(ind_type: str, params: dict[str, Any]) -> dict[str, Any]:
    """Fold param spellings onto StrategyDSL field names (MACD periods, STOCH k/d).

    A legacy alias the indicator's registry spec declares natively is that
    indicator's own param, not a spelling of the canonical field: folding it
    would hand the value to a field the plugin ignores. Such a key is left for
    ``_spec_params`` to type and pass through.

    Raises:
        ValueError: if two spellings of one param disagree -- picking one would silently
            run different periods than the user configured (IndicatorConfig rejects the same).
    """
    native = _native_param_names(ind_type)
    groups: dict[str, tuple[tuple[str, ...], bool]] = {
        target: (spellings, True) for target, spellings in _PARAM_SYNONYMS.items()
    }
    if ind_type == "STOCH":
        # STOCH's spellings are the registry's own names (period_k/period_d) mapping onto
        # the DSL fields period/d_period: always folded, native despite the spec declaring them.
        for target, spellings in _STOCH_SYNONYMS.items():
            groups[target] = (spellings, False)
    out: dict[str, Any] = {}
    for target, (spellings, legacy_aliases) in groups.items():
        if legacy_aliases and target in params:
            # A spelling the spec declares natively is that indicator's own param, not an alias of
            # the canonical field: setting both asks for two different things on one indicator.
            shadowed = next(
                (key for key in spellings if key != target and key in native and key in params),
                None,
            )
            if shadowed is not None:
                msg = (
                    f"{ind_type} sets both its native param '{shadowed}'={params[shadowed]!r} and "
                    f"the legacy alias '{target}'={params[target]!r}; '{shadowed}' is a native "
                    f"{ind_type} param, not an alias of '{target}': remove one"
                )
                raise ValueError(msg)
        found = {
            key: params[key]
            for key in spellings
            if key in params and not (legacy_aliases and key != target and key in native)
        }
        if len({_conflict_key(v) for v in found.values()}) > 1:
            shown = ", ".join(f"'{k}'={v!r}" for k, v in found.items())
            msg = f"Conflicting {ind_type} params for '{target}': {shown}"
            raise ValueError(msg)
        if found:
            out[target] = next(iter(found.values()))
    return out


def _spec_param(ind_type: str, key: str, value: Any, expected: type) -> Any:
    """Type one registry-declared param the way ``IndicatorConfig`` validates it.

    Raises:
        ValueError: if the value cannot be represented as ``expected`` without changing it
            (``9.5`` for an int param would silently run ``9``).
    """
    shown = f"{ind_type} param '{key}'={value!r}"
    if expected in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"{shown} must be a number"
            raise ValueError(msg)
        if expected is int:
            if not float(value).is_integer():
                msg = f"{shown} must be a whole number"
                raise ValueError(msg)
            return int(value)
        return float(value)
    return value


def _int_param(ind_type: str, key: str, value: Any) -> int:
    """Legacy canonical int param: numeric strings parse, bools and fractions are rejected.

    The editor's YAML tab and LLM-extracted DSL spell periods ``"14"``, which ``int()`` used
    to accept, so a numeric string is parsed. A fractional value is rejected instead of being
    truncated (``int(12.5)`` would silently run 12 while the user configured 12.5) and an
    integral float (12.0) becomes 12. A bool is rejected (``True`` used to become period 1).
    """
    shown = f"{ind_type} param '{key}'={value!r}"
    number = _as_float(value)
    if number is None:
        msg = f"{shown} must be a number"
        raise ValueError(msg)
    if not number.is_integer():
        msg = f"{shown} must be a whole number"
        raise ValueError(msg)
    return int(number)


def _spec_params(ind_type: str, params: dict[str, Any], consumed: set[str]) -> dict[str, Any]:
    """Params beyond the canonical mapping, taken from the indicator's registry spec.

    ``consumed`` holds the keys ``_translate_indicators`` already folded onto StrategyDSL
    fields. Everything else must be declared by the spec's ``param_schema`` (ICHIMOKU
    ``tenkan``, plugin ``alpha``, SQUEEZE ``length``, ...) and is passed through typed.

    Raises:
        ValueError: if a param is neither canonical nor declared by the spec -- dropping it
            would silently backtest the spec default instead of the user's value.
    """
    from vibe_quant.dsl.indicators import indicator_registry

    spec = indicator_registry.get(_TYPE_ALIASES.get(ind_type, ind_type))
    if spec is None:  # unknown type: StrategyDSL validation rejects it with the valid list
        return {}
    out: dict[str, Any] = {}
    for key, value in params.items():
        if key in consumed:
            continue
        expected = spec.param_schema.get(key)
        if expected is None:
            valid = sorted(spec.param_schema)
            msg = f"Indicator '{ind_type}' got unknown param '{key}'. Valid params: {valid}"
            raise ValueError(msg)
        out[key] = _spec_param(ind_type, key, value, expected)
    return out


def _js_str(value: Any) -> str:
    """JavaScript ``String(value)`` for a param value (the editor joins them into operands)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _ref_int(value: Any) -> int:
    """Canonical int period as an int for an operand ref; a numeric string is int-able.

    ``_build_ref_map`` re-derives canonical params from the raw editor values, so a spell
    like ``"12.0"`` reaches here as a string (``_translate_indicators`` validated it first).

    Raises:
        ValueError: if the value is not a number -- a placeholder period would register a
            ref (``MACD(0,26,9)``) no condition means.
    """
    number = _as_float(value)
    if number is None:
        msg = f"operand ref period {value!r} must be a number"
        raise ValueError(msg)
    return int(number)


_CALL_REF = re.compile(r"^([A-Za-z_]\w*)\((.*)\)$")


def _norm_ref(ref: str) -> str:
    """Canonical form of ``TYPE(a,b,...)``: spaces dropped, numeric args compared as numbers.

    The editor prints args with JavaScript ``String(number)`` (``0.000001``, ``1e-7``), which
    Python spells differently (``1e-06``); parsing both sides as floats makes them equal.
    """
    m = _CALL_REF.match(ref)
    if m is None:
        return ref
    args = []
    for arg in (a.strip() for a in m.group(2).split(",")):
        try:
            num = float(arg)
        except ValueError:
            args.append(arg)
            continue
        args.append(repr(num) if math.isfinite(num) else arg)
    return f"{m.group(1)}({','.join(args)})"


def _drop_none_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Drop ``None``-valued params: an explicit null means "not set" (T7-R4).

    The editor emits ``std_dev: null`` and a stored config may carry a legacy null;
    treating it as absent makes such a config translate and launch exactly like the
    same config without the key. Must run before any ``float()``/``int()`` coercion.
    """
    return {key: value for key, value in params.items() if value is not None}


def _translate_indicators(indicators_raw: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert DslConfig indicator array to StrategyDSL indicators dict."""
    result: dict[str, dict[str, Any]] = {}
    for ind in indicators_raw:
        ind_type = str(ind.get("type", "EMA")).upper()
        params = _drop_none_params(ind.get("params", {}) or {})
        name = _generate_indicator_name(ind_type, params, set(result.keys()))

        config: dict[str, Any] = {"type": _TYPE_ALIASES.get(ind_type, ind_type)}

        # Map flat params to StrategyDSL field names
        canonical = _canonical_params(ind_type, params)
        if "period" in params and "period" not in canonical:
            config["period"] = _int_param(ind_type, "period", params["period"])
        # MACD periods; STOCH k/d (editor spells them k_period / d_period, registry period_k /
        # period_d). Contradictory spellings were rejected above.
        for field, value in canonical.items():
            config[field] = _int_param(ind_type, field, value)
        # Bollinger Bands
        if "std_dev" in params:
            config["std_dev"] = float(params["std_dev"])
        # ATR-based
        if "atr_multiplier" in params:
            config["atr_multiplier"] = float(params["atr_multiplier"])
        # Spec-declared params (ICHIMOKU tenkan/kijun/senkou, plugin alpha, SQUEEZE length, ...)
        # A legacy alias the spec declares natively (a plugin's ``fast``) is the indicator's
        # own param, not a synonym: leave it unconsumed so ``_spec_params`` passes it through.
        native = _native_param_names(ind_type)
        consumed = {"period", "std_dev", "atr_multiplier"}
        for target, spellings in _PARAM_SYNONYMS.items():
            consumed.update(
                key for key in spellings if not (key != target and key in native)
            )
        if ind_type == "STOCH":
            for spellings in _STOCH_SYNONYMS.values():
                consumed.update(spellings)
        config.update(_spec_params(ind_type, params, consumed))
        # Per-indicator timeframe (was silently dropped)
        tf_override = ind.get("timeframe_override")
        if tf_override:
            config["timeframe"] = str(tf_override)

        result[name] = config
    return result


# Public alias of the indicator-only translation path, used by
# ``schema.check_strategy_indicator_params`` (the save guard).
translate_indicators = _translate_indicators


def _type_refs(type_name: str, params: dict[str, Any], canonical: dict[str, Any]) -> list[str]:
    """Ref spellings one indicator answers to when written with ``type_name``.

    Covers the bare type, ``TYPE(period)``, the editor's ``TYPE(v1,v2,...)`` (every param value
    in order, see frontend buildOperandOptions), and the legacy STOCH(k,d) /
    MACD(fast,slow,signal) forms built from the canonical periods.
    """
    refs = [type_name, type_name.lower()]
    # TYPE(period) e.g. "EMA(50)", "RSI(14)"; a numeric-string period ("14", "12.0") is int-able
    period = _as_float(params.get("period"))
    if period is not None:
        refs.append(f"{type_name}({int(period)})")
    # TYPE(v1,v2,...) exactly as buildOperandOptions joins the configured param values
    if params:
        values = [_js_str(v) for v in params.values()]
        refs.append(f"{type_name}({','.join(values)})")
    # STOCH(k,d) e.g. "STOCH(14,3)"
    if type_name == "STOCH" and "period" in canonical and "d_period" in canonical:
        k, d = _ref_int(canonical["period"]), _ref_int(canonical["d_period"])
        refs.append(f"STOCH({k},{d})")
    # MACD(fast,slow,signal) e.g. "MACD(12,26,9)"
    if type_name == "MACD" and "fast_period" in canonical and "slow_period" in canonical:
        sig = canonical.get("signal_period", 9)
        refs.append(
            f"MACD({_ref_int(canonical['fast_period'])},{_ref_int(canonical['slow_period'])},"
            f"{_ref_int(sig)})"
        )
    return refs


def _add_ref(refs: dict[str, list[str]], ref: str, name: str) -> None:
    owners = refs.setdefault(_norm_ref(ref), [])
    if name not in owners:
        owners.append(name)


def _build_ref_map(
    indicators_raw: list[dict[str, Any]], canonical_names: list[str]
) -> dict[str, list[str]]:
    """Map human-readable refs like 'RSI(14)' or 'EMA(50)' to the canonical names they denote.

    A ref owned by several indicators is ambiguous (two ``BBANDS(20,2)`` on different timeframes
    print the same operand); ``_resolve_ref`` rejects it. Refs spelled with an indicator's own
    type always beat refs that reach it only through a registry alias (``BBANDS(20)`` names the
    BBANDS indicator even when a ``BB(20)`` one exists).
    """
    explicit: dict[str, list[str]] = {}
    via_alias: dict[str, list[str]] = {}
    for ind, canonical in zip(indicators_raw, canonical_names, strict=True):
        ind_type = str(ind.get("type", "EMA")).upper()
        params = _drop_none_params(ind.get("params", {}) or {})
        canonical_params = _canonical_params(ind_type, params)

        for ref in _type_refs(ind_type, params, canonical_params):
            _add_ref(explicit, ref, canonical)
        alias = _TYPE_ALIASES.get(ind_type)
        if alias:
            for ref in _type_refs(alias, params, canonical_params):
                _add_ref(via_alias, ref, canonical)

    # Canonical names are unique and always map to themselves
    for canonical in canonical_names:
        explicit[canonical] = [canonical]

    return via_alias | explicit


_PRICE_ALIASES = {"price": "close", "Price": "close", "PRICE": "close"}


def _resolve_ref(token: str, ref_map: dict[str, list[str]]) -> str:
    """Replace a human-readable indicator ref with its canonical name.

    Raises:
        ValueError: if the ref matches several indicators -- compiling against one of them
            would silently trade the wrong configuration.
    """
    token = _PRICE_ALIASES.get(token, token)
    owners = ref_map.get(_norm_ref(token))
    if not owners:
        return token
    if len(owners) > 1:
        msg = (
            f"Ambiguous operand '{token}' matches indicators {', '.join(owners)}; "
            f"reference one by its name (e.g. '{owners[0]}') or give them different params"
        )
        raise ValueError(msg)
    return owners[0]


def _condition_to_str(cond: dict[str, Any] | str, ref_map: dict[str, list[str]] | None = None) -> str:
    """Convert a DslCondition object or raw string to expression string."""
    if isinstance(cond, str):
        return cond
    left = str(cond.get("left", "price"))
    op = cond.get("operator", ">")
    right = str(cond.get("right", "price"))
    if ref_map:
        left = _resolve_ref(left, ref_map)
        right = _resolve_ref(right, ref_map)
    return f"{left} {op} {right}"


def _condition_list(
    conditions: dict[str, Any], key: str, ref_map: dict[str, list[str]] | None
) -> list[str]:
    """Translate one editor condition list; AND-only (StrategyDSL semantics).

    Raises:
        ValueError: if any condition after the first is joined with ``or`` --
            silently AND-ing it would trade a different strategy.
    """
    raw = conditions.get(key)
    if not isinstance(raw, list):
        return []
    for i, cond in enumerate(raw):
        if i > 0 and isinstance(cond, dict) and str(cond.get("logic") or "and").lower() == "or":
            msg = (
                f"conditions.{key}[{i}] uses 'or' logic, which the backtest engine does not "
                "support (conditions are AND-ed); split it into separate strategies"
            )
            raise ValueError(msg)
    return [_condition_to_str(c, ref_map) for c in raw]


def _translate_conditions(
    conditions: dict[str, Any], ref_map: dict[str, list[str]] | None = None
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (long_entry, short_entry, long_exit, short_exit) condition lists."""
    long_conds = _condition_list(conditions, "long_entry", ref_map)
    short_conds = _condition_list(conditions, "short_entry", ref_map)

    # Fall back: generic entry → long
    if not long_conds and not short_conds:
        long_conds = _condition_list(conditions, "entry", ref_map)

    long_exit = _condition_list(conditions, "long_exit", ref_map)
    short_exit = _condition_list(conditions, "short_exit", ref_map)
    # Generic exit applies to every direction that has entries (was dropped)
    if not long_exit and not short_exit:
        generic_exit = _condition_list(conditions, "exit", ref_map)
        if long_conds:
            long_exit = list(generic_exit)
        if short_conds:
            short_exit = list(generic_exit)

    return long_conds, short_conds, long_exit, short_exit


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


def _ensure_atr_reference(
    indicators: dict[str, dict[str, Any]], *risk_configs: dict[str, Any]
) -> None:
    """Point every ATR stop / take-profit at an ATR indicator (StrategyDSL requires one).

    The editor's risk model has no indicator field, so reuse the strategy's first ATR
    indicator or add a default ATR(14) when it has none.
    """
    atr_users = [c for c in risk_configs if str(c.get("type", "")).startswith("atr")]
    if not atr_users:
        return
    ref = next((n for n, cfg in indicators.items() if cfg.get("type") == "ATR"), None)
    if ref is None:
        ref = _generate_indicator_name("ATR", {"period": 14}, set(indicators))
        indicators[ref] = {"type": "ATR", "period": 14}
    for cfg in atr_users:
        cfg["indicator"] = ref


def _strip_none_indicator_params(raw: dict[str, Any]) -> dict[str, Any]:
    """Canonical-format config with a stored null builtin param made equal to the key being absent.

    ``translate_dsl_config`` returns canonical input unchanged; a legacy ``std_dev: null``
    must not survive into ``IndicatorConfig``/launch, so it is dropped here (T7-R4).
    """
    indicators = raw.get("indicators")
    if not isinstance(indicators, Mapping):
        return raw
    return {
        **raw,
        "indicators": {
            name: _drop_none_params(cfg) if isinstance(cfg, Mapping) else cfg
            for name, cfg in indicators.items()
        },
    }


def translate_dsl_config(raw: dict[str, Any], strategy_name: str = "strategy") -> dict[str, Any]:
    """Translate frontend DslConfig dict to canonical StrategyDSL dict.

    If the input is already in StrategyDSL format (no 'general' key, indicators is a dict),
    it is returned with any ``None``-valued indicator param dropped (T7-R4).

    Args:
        raw: The dsl_config dict from the database.
        strategy_name: The strategy name (from the strategies table, not dsl_config).

    Returns:
        Dict suitable for StrategyDSL.model_validate().
    """
    if not _is_dsl_config_format(raw):
        # Already canonical StrategyDSL format
        return _strip_none_indicator_params(raw)

    general = raw.get("general", {}) or {}
    timeframe = str(general.get("timeframe", "1h"))
    additional_tfs = list(general.get("additional_timeframes") or [])

    # Indicators
    indicators_raw = raw.get("indicators", []) or []
    if isinstance(indicators_raw, list):
        indicators = _translate_indicators(indicators_raw)
        ref_map = _build_ref_map(indicators_raw, list(indicators.keys()))
    else:
        indicators = indicators_raw  # already dict
        ref_map = None

    # Conditions
    conditions_raw = raw.get("conditions", {}) or {}
    long_entry, short_entry, long_exit, short_exit = _translate_conditions(
        conditions_raw, ref_map
    )

    # Ensure at least one condition (StrategyDSL requires it)
    if not long_entry and not short_entry:
        long_entry = ["price > price"]  # trivially-true placeholder

    # Risk
    risk = raw.get("risk", {}) or {}
    sl_raw = risk.get("stop_loss", {}) or {}
    tp_raw = risk.get("take_profit", {}) or {}
    stop_loss = _translate_stop_loss(sl_raw)
    take_profit = _translate_take_profit(tp_raw)
    if isinstance(indicators, dict):
        _ensure_atr_reference(indicators, stop_loss, take_profit)
    if risk.get("trailing_stop_pct"):
        msg = (
            "risk.trailing_stop_pct is not supported by the backtest engine (only ATR "
            "trailing stops: stop_loss.type 'atr_trailing'); remove it or switch the stop type"
        )
        raise ValueError(msg)

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
            d
            for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
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
    if long_exit or short_exit:
        result["exit_conditions"] = {"long": long_exit, "short": short_exit}
    # Indicator timeframe overrides must be declared as additional timeframes
    for cfg in indicators.values() if isinstance(indicators, dict) else []:
        tf = cfg.get("timeframe") if isinstance(cfg, dict) else None
        if tf and tf != timeframe and tf not in additional_tfs:
            additional_tfs.append(tf)
    if additional_tfs:
        result["additional_timeframes"] = additional_tfs
    if time_filters:
        result["time_filters"] = time_filters

    return result
