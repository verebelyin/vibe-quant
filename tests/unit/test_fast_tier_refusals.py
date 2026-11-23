"""Fast tier refuses features it cannot replay (design-v2 rows R18, arch-c4-1 R20).

``position_mode: reverse``, ``signal_level`` stops/targets and any stop/target that is
not ``fixed_pct`` are never silently replayed as a different rule: the fast tier raises
``FastTierUnsupported`` before any runner or engine exists, so discovery falls back to
NT (or records the reason). A DSL without those features, or a plain directional
override, runs as today.
"""

from __future__ import annotations

from typing import Any

import pytest

from vibe_quant.discovery import genome as genome_mod
from vibe_quant.discovery.operators import ConditionType as C
from vibe_quant.discovery.operators import Direction, StrategyChromosome, StrategyGene
from vibe_quant.fast_tier import FastTierUnsupported
from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

START, END = "2026-03-02", "2026-03-04"

#: The four directional stop/target fields that may carry ``signal_level``.
_SIGNAL_LEVEL_FIELDS = (
    "stop_loss_long",
    "stop_loss_short",
    "take_profit_long",
    "take_profit_short",
)


class _ReachedRunner(Exception):
    """Sentinel: execution got past every refusal to the runner build."""


def _chromosome() -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, C.GT, 50.0)],
        exit_genes=[],
        stop_loss_pct=1.0,
        take_profit_pct=2.0,
        direction=Direction.LONG,
    )


def _fn(monkeypatch: pytest.MonkeyPatch, dsl: dict[str, Any]) -> FastBacktestFn:
    """A 1m single-symbol fast fn whose chromosome converts to ``dsl``.

    ``_make_runner`` is stubbed: reaching it means the DSL was not refused.
    """
    monkeypatch.setattr(genome_mod, "chromosome_to_dsl", lambda _: dsl)

    def _reached(*_args: object, **_kwargs: object) -> None:
        raise _ReachedRunner

    monkeypatch.setattr(FastBacktestFn, "_make_runner", _reached)
    return FastBacktestFn(["BTCUSDT"], "1m", START, END)


def test_fast_tier_refuses_reverse_position_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """``position_mode: reverse`` never reaches the runner."""
    fn = _fn(monkeypatch, {"position_mode": "reverse"})
    with pytest.raises(FastTierUnsupported, match="position_mode reverse"):
        fn._run_single(_chromosome(), START, END)


@pytest.mark.parametrize("field", _SIGNAL_LEVEL_FIELDS)
def test_fast_tier_refuses_directional_signal_level(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Each of the four directional fields refuses a signal_level config."""
    indicator = "dc.lower" if field.startswith("stop_loss") else "dc.upper"
    dsl = {field: {"type": "signal_level", "indicator": indicator}}
    fn = _fn(monkeypatch, dsl)
    with pytest.raises(FastTierUnsupported, match="signal_level stops/targets"):
        fn._run_single(_chromosome(), START, END)


def test_fast_tier_refuses_signal_target_beside_a_plain_generic_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain generic stop does not hide a directional signal_level target."""
    dsl = {
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit_short": {"type": "signal_level", "indicator": "dc.upper"},
    }
    fn = _fn(monkeypatch, dsl)
    with pytest.raises(FastTierUnsupported, match="signal_level stops/targets"):
        fn._run_single(_chromosome(), START, END)


#: One non-``fixed_pct`` stop/target per exit field the fast tier reads.
_NON_FIXED_EXITS = [
    pytest.param(
        "stop_loss",
        {"type": "atr_fixed", "atr_multiplier": 2.0, "indicator": "exit_atr"},
        id="generic-atr-stop",
    ),
    pytest.param(
        "stop_loss",
        {"type": "atr_trailing", "atr_multiplier": 2.5, "indicator": "exit_atr"},
        id="generic-trailing-stop",
    ),
    pytest.param(
        "stop_loss_long",
        {"type": "atr_fixed", "atr_multiplier": 3.0, "indicator": "exit_atr"},
        id="long-atr-stop",
    ),
    pytest.param(
        "stop_loss_short",
        {"type": "atr_trailing", "atr_multiplier": 1.5, "indicator": "exit_atr_short"},
        id="short-trailing-stop",
    ),
    pytest.param(
        "take_profit",
        {"type": "atr_fixed", "atr_multiplier": 3.0, "indicator": "exit_atr"},
        id="generic-atr-target",
    ),
    pytest.param(
        "take_profit",
        {"type": "risk_reward", "risk_reward_ratio": 2.0},
        id="generic-risk-reward",
    ),
    pytest.param(
        "take_profit_long",
        {"type": "risk_reward", "risk_reward_ratio": 1.5},
        id="long-risk-reward",
    ),
    pytest.param(
        "take_profit_short",
        {"type": "atr_fixed", "atr_multiplier": 4.0, "indicator": "exit_atr_short"},
        id="short-atr-target",
    ),
]


@pytest.mark.parametrize(("field", "cfg"), _NON_FIXED_EXITS)
def test_fast_tier_refuses_non_fixed_stop_and_target(
    monkeypatch: pytest.MonkeyPatch, field: str, cfg: dict[str, Any]
) -> None:
    """Every non-fixed_pct stop/target refuses, naming the field that carries it."""
    fn = _fn(monkeypatch, {field: cfg})
    with pytest.raises(FastTierUnsupported, match=f"{field} type {cfg['type']}"):
        fn._run_single(_chromosome(), START, END)


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({}, id="legacy"),
        pytest.param({"position_mode": None}, id="position-mode-null"),
        pytest.param(
            {"stop_loss_long": {"type": "fixed_pct", "percent": 1.5}},
            id="directional-plain-override",
        ),
        pytest.param(
            {
                "stop_loss": {"type": "fixed_pct", "percent": 50.0},
                "take_profit": {"type": "fixed_pct", "percent": 100.0},
            },
            id="safety-and-none-render-as-fixed-pct",
        ),
    ],
)
def test_fast_tier_runs_as_today_without_the_features(
    monkeypatch: pytest.MonkeyPatch, over: dict[str, Any]
) -> None:
    """Neither feature, or a plain directional override, reaches the runner."""
    fn = _fn(monkeypatch, dict(over))
    with pytest.raises(_ReachedRunner):
        fn._run_single(_chromosome(), START, END)
