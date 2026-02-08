"""Strategy validation summary component.

Provides a pre-backtest health check with:
- Validation status and error details
- Smart warnings for suboptimal configurations
- Strategy complexity score
- Ready-to-backtest checklist
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from vibe_quant.dsl.schema import StrategyDSL


def render_validation_summary(
    dsl: StrategyDSL | None,
    error: str | None,
) -> None:
    """Render the validation summary panel.

    Args:
        dsl: Validated StrategyDSL model (None if validation failed)
        error: Error message (None if validation succeeded)
    """
    if error:
        st.error(f"**Validation Failed**\n\n{error}")
        return

    if dsl is None:
        return

    st.success("**Strategy is valid**")

    # Strategy overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Indicators", len(dsl.indicators))
    with col2:
        entry_count = len(dsl.entry_conditions.long) + len(dsl.entry_conditions.short)
        st.metric("Entry Rules", entry_count)
    with col3:
        exit_count = len(dsl.exit_conditions.long) + len(dsl.exit_conditions.short)
        st.metric("Exit Rules", exit_count)
    with col4:
        total_combos = 1
        for v in dsl.sweep.values():
            total_combos *= len(v)
        st.metric("Sweep Combos", f"{total_combos:,}" if dsl.sweep else "0")

    # Complexity score
    complexity = _calculate_complexity(dsl)
    complexity_label = "Simple" if complexity <= 3 else "Moderate" if complexity <= 6 else "Complex"
    complexity_color = "green" if complexity <= 3 else "orange" if complexity <= 6 else "red"
    st.markdown(f"**Complexity:** :{complexity_color}[{complexity_label}] ({complexity}/10)")

    # Smart warnings
    warnings = _generate_warnings(dsl)
    if warnings:
        with st.expander(f"**Warnings** ({len(warnings)})", expanded=True):
            for level, msg in warnings:
                if level == "warning":
                    st.warning(msg)
                elif level == "info":
                    st.info(msg)
                elif level == "error":
                    st.error(msg)

    # Ready-to-backtest checklist
    checklist = _generate_checklist(dsl)
    with st.expander("**Backtest Readiness**", expanded=False):
        for item, ok in checklist:
            icon = "white_check_mark" if ok else "x"
            st.markdown(f":{icon}: {item}")


def _calculate_complexity(dsl: StrategyDSL) -> int:
    """Calculate strategy complexity score (1-10)."""
    score = 0

    # Indicators
    n_indicators = len(dsl.indicators)
    if n_indicators <= 2:
        score += 1
    elif n_indicators <= 4:
        score += 2
    else:
        score += 3

    # Conditions
    n_conditions = (
        len(dsl.entry_conditions.long)
        + len(dsl.entry_conditions.short)
        + len(dsl.exit_conditions.long)
        + len(dsl.exit_conditions.short)
    )
    if n_conditions <= 3:
        score += 1
    elif n_conditions <= 6:
        score += 2
    else:
        score += 3

    # Multi-timeframe
    if dsl.additional_timeframes:
        score += 2

    # Time filters
    if dsl.time_filters.allowed_sessions:
        score += 1
    if dsl.time_filters.blocked_days:
        score += 1

    return min(10, score)


def _generate_warnings(dsl: StrategyDSL) -> list[tuple[str, str]]:
    """Generate smart warnings for the strategy.

    Returns list of (level, message) tuples.
    """
    warnings: list[tuple[str, str]] = []

    # Check entry/exit coverage
    if dsl.entry_conditions.long and not dsl.exit_conditions.long:
        warnings.append(("warning", "No exit conditions for long positions. "
                         "Positions will only close via stop-loss or take-profit."))

    if dsl.entry_conditions.short and not dsl.exit_conditions.short:
        warnings.append(("warning", "No exit conditions for short positions. "
                         "Positions will only close via stop-loss or take-profit."))

    if not dsl.entry_conditions.long and not dsl.entry_conditions.short:
        warnings.append(("error", "No entry conditions defined at all."))

    # Check for one-sided strategy
    if dsl.entry_conditions.long and not dsl.entry_conditions.short:
        warnings.append(("info", "Long-only strategy. No short entries defined."))
    elif dsl.entry_conditions.short and not dsl.entry_conditions.long:
        warnings.append(("info", "Short-only strategy. No long entries defined."))

    # Check time filters
    if not dsl.time_filters.blocked_days and not dsl.time_filters.allowed_sessions:
        warnings.append(("info", "No time filters configured. Strategy will trade 24/7 "
                         "including weekends."))

    if not dsl.time_filters.avoid_around_funding.enabled:
        warnings.append(("info", "Funding avoidance is disabled. Entries may occur during "
                         "the volatile funding settlement period."))

    # Check stop loss
    if dsl.stop_loss.type == "fixed_pct" and dsl.stop_loss.percent and dsl.stop_loss.percent > 5.0:
        warnings.append(("warning", f"Stop loss is {dsl.stop_loss.percent}% -- "
                       "this is high risk per trade. Consider using ATR-based stops."))

    # Check risk/reward
    if (dsl.take_profit.type == "risk_reward"
            and dsl.take_profit.risk_reward_ratio
            and dsl.take_profit.risk_reward_ratio < 1.0):
        warnings.append(("warning", f"Risk/reward ratio is {dsl.take_profit.risk_reward_ratio} "
                       "(<1). You need a high win rate to be profitable."))

    # Check sweep
    if dsl.sweep:
        total_combos = 1
        for v in dsl.sweep.values():
            total_combos *= len(v)
        if total_combos > 5000:
            warnings.append(("warning", f"Sweep has {total_combos:,} combinations. "
                           "This may take a very long time. Consider narrowing ranges."))

    # Check for single condition entries
    for side in ("long", "short"):
        conditions = getattr(dsl.entry_conditions, side)
        if len(conditions) == 1:
            warnings.append(("info", f"Only 1 {side} entry condition. "
                           "Single conditions may produce noisy signals."))

    # Check for orphan indicators
    used_indicators: set[str] = set()
    all_conditions = (
        dsl.entry_conditions.long
        + dsl.entry_conditions.short
        + dsl.exit_conditions.long
        + dsl.exit_conditions.short
    )
    for cond in all_conditions:
        parts = cond.split()
        for part in parts:
            if part in dsl.indicators:
                used_indicators.add(part)

    # Also check stop_loss and take_profit indicator refs
    if dsl.stop_loss.indicator:
        used_indicators.add(dsl.stop_loss.indicator)
    if dsl.take_profit.indicator:
        used_indicators.add(dsl.take_profit.indicator)

    orphans = set(dsl.indicators.keys()) - used_indicators
    for orphan in orphans:
        warnings.append(("info", f"Indicator '{orphan}' is defined but not used in any "
                       "condition or stop-loss/take-profit reference."))

    return warnings


def _generate_checklist(dsl: StrategyDSL) -> list[tuple[str, bool]]:
    """Generate a backtest readiness checklist.

    Returns list of (item description, is_ok).
    """
    checks: list[tuple[str, bool]] = []

    # Basic requirements
    checks.append(("Strategy name defined", bool(dsl.name)))
    checks.append(("At least one indicator defined", len(dsl.indicators) > 0))
    checks.append((
        "Entry conditions defined",
        bool(dsl.entry_conditions.long or dsl.entry_conditions.short),
    ))
    checks.append((
        "Exit conditions or stop-loss defined",
        bool(
            dsl.exit_conditions.long
            or dsl.exit_conditions.short
            or dsl.stop_loss
        ),
    ))
    checks.append(("Stop-loss configured", dsl.stop_loss is not None))
    checks.append(("Take-profit configured", dsl.take_profit is not None))

    # Recommended
    checks.append((
        "Time filters configured (recommended)",
        bool(dsl.time_filters.blocked_days or dsl.time_filters.allowed_sessions),
    ))
    checks.append((
        "Funding avoidance enabled (recommended for perps)",
        dsl.time_filters.avoid_around_funding.enabled,
    ))
    checks.append((
        "Sweep parameters defined (needed for screening)",
        bool(dsl.sweep),
    ))

    return checks
