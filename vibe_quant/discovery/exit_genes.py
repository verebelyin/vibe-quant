"""GA exit genes: stop, target and time-exit rules a chromosome can carry.

A legacy chromosome trades one exit shape (``stop_loss_pct`` / ``take_profit_pct``,
both ``fixed_pct``). A chromosome with ``exit_rules`` set instead picks stop and target
TYPES (fixed %, ATR, trailing ATR, risk:reward, a 50 % safety net, no target) and an
optional time exit; ``rules_to_dsl`` renders them to DSL the compiler already runs.
``exit_rules is None`` is the legacy genome and never reaches this module's code.

Every rule an extended genome needs lives here (types, ranges, canonical form, render,
serialize, validate). ``genome.py`` calls it only when ``exit_rules`` is set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

from vibe_quant.discovery.operators import Direction
from vibe_quant.utils import TIMEFRAME_MINUTES

StopType = Literal["fixed_pct", "atr_fixed", "atr_trailing", "safety"]
TargetType = Literal["fixed_pct", "atr_fixed", "risk_reward", "none"]
STOP_TYPES: tuple[StopType, ...] = ("fixed_pct", "atr_fixed", "atr_trailing", "safety")
TARGET_TYPES: tuple[TargetType, ...] = ("fixed_pct", "atr_fixed", "risk_reward", "none")

STOP_PCT = (0.5, 10.0)
STOP_ATR_MULT = (1.0, 6.0)
TARGET_PCT = (0.5, 20.0)
TARGET_ATR_MULT = (1.0, 10.0)
RISK_REWARD = (1.0, 5.0)
ATR_PERIOD = (7, 30)
SAFETY_PCT = 50.0  # a "safety" stop renders as this fixed_pct net; exits come from signals
NONE_PCT = (
    100.0  # a "none" target renders as this fixed_pct: never fills a long, not placed on a short
)
TIME_EXIT_SPAN_H = (12, 480)  # 12 h .. 20 d, converted to bars per timeframe
MAX_BARS_LIMIT = 100_000  # StrategyDSL.max_bars_in_trade upper bound
PINNED_SL_PCT, PINNED_TP_PCT = 2.0, 3.0  # legacy fields of an extended genome (dead, pinned)

_ATR_STOPS: frozenset[str] = frozenset({"atr_fixed", "atr_trailing"})

SAFETY_NONE_DESCRIPTION = (
    "Exit genes: safety = 50% stop net, exits come from signals; "
    "none = 100% target, never fills on a long and is not placed on a short (price 0)."
)


@dataclass(frozen=True, slots=True)
class ExitBlock:
    """One side's stop and target."""

    stop_type: StopType
    stop_value: float | None  # % (fixed_pct) | ATR multiple (atr_*) | None (safety); 2 dp
    target_type: TargetType
    target_value: float | None  # % | ATR multiple | R:R ratio | None (none); 2 dp
    atr_period: int | None  # set iff the stop is atr_* or the target is atr_fixed


@dataclass(frozen=True, slots=True)
class ExitRules:
    """A genome's exit genes (frozen: operators replace it, never edit it)."""

    long: ExitBlock  # the genome's own side (LONG, SHORT, or BOTH's long side)
    short: ExitBlock | None  # BOTH only
    max_bars: int | None  # time exit (max_bars_in_trade), None = off


def max_bars_range(timeframe: str) -> tuple[int, int]:
    """Time-exit bounds in bars: 12 h .. 480 h of ``timeframe``."""
    minutes = TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        msg = f"unknown timeframe {timeframe!r} for a time exit"
        raise ValueError(msg)
    lo_h, hi_h = TIME_EXIT_SPAN_H
    return max(1, math.ceil(lo_h * 60 / minutes)), (hi_h * 60) // minutes


def _needs_atr(block: ExitBlock) -> bool:
    return block.stop_type in _ATR_STOPS or block.target_type == "atr_fixed"


def _round2(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def canonical_block(block: ExitBlock) -> ExitBlock:
    """Pin parameters the block does not use (no RNG).

    - ``atr_period`` is None unless the stop is ATR or the target is ``atr_fixed``.
    - ``risk_reward`` under a ``safety`` stop becomes ``none``: the TP would sit
      >= 50 % away (same trades, different DSL = fake diversity).
    - values are rounded to 2 dp; a ``safety`` stop / ``none`` target carries no value.
    """
    target_type = (
        "none"
        if block.stop_type == "safety" and block.target_type == "risk_reward"
        else block.target_type
    )
    pinned = ExitBlock(
        stop_type=block.stop_type,
        stop_value=None if block.stop_type == "safety" else _round2(block.stop_value),
        target_type=target_type,
        target_value=None if target_type == "none" else _round2(block.target_value),
        atr_period=block.atr_period,
    )
    if _needs_atr(pinned):
        return pinned
    return ExitBlock(
        pinned.stop_type, pinned.stop_value, pinned.target_type, pinned.target_value, None
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _stop_range(stop_type: StopType) -> tuple[float, float] | None:
    if stop_type == "fixed_pct":
        return STOP_PCT
    if stop_type in _ATR_STOPS:
        return STOP_ATR_MULT
    return None


def _target_range(target_type: TargetType) -> tuple[float, float] | None:
    if target_type == "fixed_pct":
        return TARGET_PCT
    if target_type == "atr_fixed":
        return TARGET_ATR_MULT
    if target_type == "risk_reward":
        return RISK_REWARD
    return None


def _value_problems(
    label: str, kind: str, value: float | None, bounds: tuple[float, float] | None
) -> list[str]:
    """Problems of one stop/target value: a typed kind needs a value inside ``bounds``, else none."""
    if bounds is None:
        return [f"{label}: {kind} takes no value (got {value})"] if value is not None else []
    if value is None:
        return [f"{label}: required for {kind}"]
    lo, hi = bounds
    if not lo <= value <= hi:
        return [f"{label}: {value} outside [{lo}, {hi}] for {kind}"]
    return []


def _block_problems(block: ExitBlock, label: str) -> list[str]:
    problems = _value_problems(
        f"{label}.stop_value", block.stop_type, block.stop_value, _stop_range(block.stop_type)
    )
    problems += _value_problems(
        f"{label}.target_value",
        block.target_type,
        block.target_value,
        _target_range(block.target_type),
    )
    if not _needs_atr(block):
        if block.atr_period is not None:
            problems.append(
                f"{label}.atr_period: only valid with an ATR stop or target (got {block.atr_period})"
            )
    elif block.atr_period is None:
        problems.append(f"{label}.atr_period: required for an ATR stop or target")
    elif not ATR_PERIOD[0] <= block.atr_period <= ATR_PERIOD[1]:
        problems.append(
            f"{label}.atr_period: {block.atr_period} outside [{ATR_PERIOD[0]}, {ATR_PERIOD[1]}]"
        )
    if not problems and canonical_block(block) != block:
        problems.append(
            f"{label}: not canonical (a risk_reward target under a safety stop must be none; values 2 dp)"
        )
    return problems


def _core_problems(rules: ExitRules) -> list[str]:
    problems = _block_problems(rules.long, "exit_rules.long")
    if rules.short is not None:
        problems += _block_problems(rules.short, "exit_rules.short")
    if rules.max_bars is not None and not 1 <= rules.max_bars <= MAX_BARS_LIMIT:
        problems.append(f"exit_rules.max_bars: {rules.max_bars} outside [1, {MAX_BARS_LIMIT}]")
    return problems


def _direction_problems(rules: ExitRules, direction: Direction) -> list[str]:
    if direction == Direction.BOTH and rules.short is None:
        return ["exit_rules.short: required for direction=both"]
    if direction != Direction.BOTH and rules.short is not None:
        return ["exit_rules.short: only valid for direction=both"]
    return []


def rules_problems(rules: ExitRules, direction: Direction) -> list[str]:
    """Why ``rules`` is not a valid exit gene set for ``direction`` ([] = valid)."""
    return _direction_problems(rules, direction) + _core_problems(rules)


# ---------------------------------------------------------------------------
# DSL rendering
# ---------------------------------------------------------------------------


def _stop_dsl(block: ExitBlock, atr_name: str) -> dict[str, object]:
    if block.stop_type == "safety":
        return {"type": "fixed_pct", "percent": SAFETY_PCT}
    if block.stop_type == "fixed_pct":
        return {"type": "fixed_pct", "percent": block.stop_value}
    return {"type": block.stop_type, "atr_multiplier": block.stop_value, "indicator": atr_name}


def _target_dsl(block: ExitBlock, atr_name: str) -> dict[str, object]:
    if block.target_type == "none":
        return {"type": "fixed_pct", "percent": NONE_PCT}
    if block.target_type == "fixed_pct":
        return {"type": "fixed_pct", "percent": block.target_value}
    if block.target_type == "atr_fixed":
        return {"type": "atr_fixed", "atr_multiplier": block.target_value, "indicator": atr_name}
    return {"type": "risk_reward", "risk_reward_ratio": block.target_value}


def rules_to_dsl(rules: ExitRules, direction: Direction) -> dict[str, object]:
    """DSL fragment for ``rules``: ``indicators`` to merge, the stop/target keys, time exit, description.

    LONG/SHORT: the generic ``stop_loss`` / ``take_profit`` are the block. BOTH: the generic
    keys are the long block and ``*_long`` / ``*_short`` carry each side (the compiler reads
    the directional override first). The ATR indicator is ``exit_atr`` (long / only block) or
    ``exit_atr_short`` (BOTH's short block) and is emitted only when its block needs one.
    """
    problems = _direction_problems(rules, direction)
    if problems:
        raise ValueError("; ".join(problems))
    indicators: dict[str, object] = {}
    blocks = [("long", "exit_atr", rules.long)]
    if rules.short is not None:
        blocks.append(("short", "exit_atr_short", rules.short))
    frag: dict[str, object] = {}
    for side, atr_name, block in blocks:
        if _needs_atr(block):
            indicators[atr_name] = {"type": "ATR", "period": block.atr_period}
        stop, target = _stop_dsl(block, atr_name), _target_dsl(block, atr_name)
        if side == "long":
            frag["stop_loss"], frag["take_profit"] = stop, target
        if direction == Direction.BOTH:
            frag[f"stop_loss_{side}"], frag[f"take_profit_{side}"] = stop, target
    frag["indicators"] = indicators
    if rules.max_bars is not None:
        frag["max_bars_in_trade"] = rules.max_bars
    if any(b.stop_type == "safety" or b.target_type == "none" for _, _, b in blocks):
        frag["description"] = SAFETY_NONE_DESCRIPTION
    return frag


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _block_to_dict(block: ExitBlock) -> dict[str, object]:
    return {
        "stop_type": block.stop_type,
        "stop_value": block.stop_value,
        "target_type": block.target_type,
        "target_value": block.target_value,
        "atr_period": block.atr_period,
    }


def rules_to_dict(rules: ExitRules) -> dict[str, object]:
    """JSON-safe form of ``rules`` (inverse: ``rules_from_dict``)."""
    return {
        "long": _block_to_dict(rules.long),
        "short": None if rules.short is None else _block_to_dict(rules.short),
        "max_bars": rules.max_bars,
    }


def _number(label: str, raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        msg = f"{label}: must be a number, got {raw!r}"
        raise ValueError(msg)
    return float(raw)


def _whole(label: str, raw: object) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        msg = f"{label}: must be a whole number, got {raw!r}"
        raise ValueError(msg)
    return raw


def _unknown_keys(label: str, raw: dict[str, object], allowed: set[str]) -> None:
    extra = sorted(set(raw) - allowed)
    if extra:
        msg = f"{label}: unknown key {extra[0]!r}"
        raise ValueError(msg)


def _block_from_dict(raw: object, label: str) -> ExitBlock:
    if not isinstance(raw, dict):
        msg = f"{label}: expected an object, got {type(raw).__name__}"
        raise ValueError(msg)
    _unknown_keys(
        label, raw, {"stop_type", "stop_value", "target_type", "target_value", "atr_period"}
    )
    stop_type, target_type = raw.get("stop_type"), raw.get("target_type")
    if stop_type not in STOP_TYPES:
        msg = f"{label}.stop_type: unknown type {stop_type!r}"
        raise ValueError(msg)
    if target_type not in TARGET_TYPES:
        msg = f"{label}.target_type: unknown type {target_type!r}"
        raise ValueError(msg)
    return ExitBlock(
        stop_type=cast("StopType", stop_type),
        stop_value=_number(f"{label}.stop_value", raw.get("stop_value")),
        target_type=cast("TargetType", target_type),
        target_value=_number(f"{label}.target_value", raw.get("target_value")),
        atr_period=_whole(f"{label}.atr_period", raw.get("atr_period")),
    )


def rules_from_dict(raw: object) -> ExitRules:
    """Parse the stored form; ``ValueError`` naming the field on any bad shape or value."""
    if not isinstance(raw, dict):
        msg = f"exit_rules: expected an object, got {type(raw).__name__}"
        raise ValueError(msg)
    _unknown_keys("exit_rules", raw, {"long", "short", "max_bars"})
    if "long" not in raw:
        msg = "exit_rules.long: required"
        raise ValueError(msg)
    short_raw = raw.get("short")
    rules = ExitRules(
        long=_block_from_dict(raw["long"], "exit_rules.long"),
        short=None if short_raw is None else _block_from_dict(short_raw, "exit_rules.short"),
        max_bars=_whole("exit_rules.max_bars", raw.get("max_bars")),
    )
    problems = _core_problems(rules)
    if problems:
        raise ValueError("; ".join(problems))
    return rules


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


def _block_summary(block: ExitBlock) -> str:
    atr = f"atr{block.atr_period}"
    if block.stop_type == "safety":
        stop = "safety"
    elif block.stop_type == "fixed_pct":
        stop = f"fixed_pct({block.stop_value})"
    else:
        stop = f"{block.stop_type}({block.stop_value},{atr})"
    if block.target_type == "none":
        target = "none"
    elif block.target_type == "atr_fixed":
        target = f"atr_fixed({block.target_value},{atr})"
    else:
        target = f"{block.target_type}({block.target_value})"
    return f"{stop}/{target}"


def rules_summary(rules: ExitRules) -> str:
    """One-line description for logs, e.g. ``exits=long:atr_trailing(2.5,atr14)/risk_reward(2.0) time=12``."""
    parts = [f"exits=long:{_block_summary(rules.long)}"]
    if rules.short is not None:
        parts.append(f"short:{_block_summary(rules.short)}")
    if rules.max_bars is not None:
        parts.append(f"time={rules.max_bars}")
    return " ".join(parts)
