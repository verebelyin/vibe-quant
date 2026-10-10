"""DONCHIAN update-order semantics inside a compiled strategy (bead vibe-quant-yul7u.20).

Question: does the ``upper`` a compiled strategy reads in ``on_bar(t)`` include bar t's own
high? If so ``close > donchian.upper`` can never fire (close <= high <= upper).

Answer pinned here by running the real generated strategy in a BacktestEngine: DONCHIAN is
the NT built-in ``DonchianChannel`` registered via ``register_indicator_for_bars``; NT updates
registered indicators BEFORE ``on_bar``, so upper[t] = max(high[t-period+1..t]) -- the
current bar IS included. Consequences pinned below:

* ``close > donchian.upper`` and ``crosses_above donchian.upper`` never fire;
* bare ``donchian`` resolves to the MIDDLE band (``primary_output="middle"``), so the
  template's ``close > donchian`` is a close-vs-midline trend filter, not a breakout;
* GA genes use ``donchian.position`` = (close-lower)/(upper-lower), capped at 1.0, so any
  ``position > 1`` style threshold is unreachable and ``>= 1`` fires only when close == high.

Observation is by subclassing the generated strategy; indicator state is read before
``super().on_bar`` (the same point the generated condition code reads it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.unit.test_fill_timing import _INSTRUMENT_ID, _make_bar, _new_engine
from vibe_quant.data.catalog import get_bar_type
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import validate_strategy_dict

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar

PERIOD = 3
# Strictly rising series: every close exceeds ALL prior highs (clear breakout) yet stays
# below its own bar high (close = high - 10).
_CLOSES = [100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0]


def _bars() -> list[Bar]:
    bar_type = get_bar_type("BTCUSDT", "1m")
    return [
        _make_bar(bar_type, c - 5.0, c + 10.0, c - 10.0, c, minute=i) for i, c in enumerate(_CLOSES)
    ]


def _run(condition: str) -> dict[str, list[float]]:
    """Compile a DONCHIAN long-entry strategy; return per-bar observations.

    Keys: ``upper``/``lower``/``middle`` (as read at on_bar entry, None-free once
    initialized), ``high``/``close``, and ``fired`` (entry-signal bool per bar as 0/1).
    """
    dsl = validate_strategy_dict(
        {
            "name": "donchian_probe",
            "timeframe": "1m",
            "indicators": {"donchian": {"type": "DONCHIAN", "period": PERIOD}},
            "entry_conditions": {"long": [condition]},
            "exit_conditions": {},
            "stop_loss": {"type": "fixed_pct", "percent": 50.0},
            "take_profit": {"type": "fixed_pct", "percent": 50.0},
        }
    )
    module = StrategyCompiler().compile_to_module(dsl)
    camel = _to_class_name(dsl.name)
    base_cls = getattr(module, f"{camel}Strategy")
    config_cls = getattr(module, f"{camel}Config")

    obs: dict[str, list[float]] = {
        k: [] for k in ("upper", "lower", "middle", "high", "close", "fired")
    }

    class Probe(base_cls):  # type: ignore[valid-type, misc]
        def on_bar(self, bar: Bar) -> None:
            if bar.bar_type == self.bar_type_1m:
                obs["upper"].append(float(self.ind_donchian.upper))
                obs["lower"].append(float(self.ind_donchian.lower))
                obs["middle"].append(float(self.ind_donchian.middle))
                obs["high"].append(float(bar.high))
                obs["close"].append(float(bar.close))
            super().on_bar(bar)

        def _check_long_entry(self, bar: Bar) -> bool:
            fired: bool = super()._check_long_entry(bar)
            obs["fired"].append(1.0 if fired else 0.0)
            return fired

    strategy = Probe(
        config=config_cls(
            instrument_id=_INSTRUMENT_ID,
            donchian_period=PERIOD,
            execution_delay_probability=0.0,
        )
    )
    engine = _new_engine()
    engine.add_data(_bars())
    engine.add_strategy(strategy)
    try:
        engine.run()
    finally:
        engine.reset()
        engine.dispose()
    return obs


def test_upper_includes_current_bar_high() -> None:
    """upper read in on_bar(t) == max(high[t-P+1..t]); on a rising series that is high[t]."""
    obs = _run("close > donchian.upper")
    assert len(obs["upper"]) == len(_CLOSES)
    highs = obs["high"]
    for t in range(PERIOD - 1, len(highs)):
        expected = max(highs[t - PERIOD + 1 : t + 1])
        assert obs["upper"][t] == pytest.approx(expected), f"bar {t}"
        # Rising series: the current bar's high IS the channel top.
        assert obs["upper"][t] == pytest.approx(highs[t]), f"bar {t}"
        # And it is NOT the previous-bars-only max (that would lag by one bar).
        assert obs["upper"][t] > max(highs[t - PERIOD + 1 : t])


def test_close_above_upper_never_fires_on_clear_new_highs() -> None:
    obs = _run("close > donchian.upper")
    assert obs["fired"], "entry check never ran"
    assert sum(obs["fired"]) == 0
    # The series does make a new high every bar -- the breakout exists, the condition misses it.
    assert all(
        obs["close"][t] > max(obs["high"][max(0, t - PERIOD) : t]) for t in range(1, len(_CLOSES))
    )


def test_crosses_above_upper_never_fires() -> None:
    obs = _run("close crosses_above donchian.upper")
    assert obs["fired"], "entry check never ran"
    assert sum(obs["fired"]) == 0


def test_bare_donchian_is_the_middle_band_so_template_condition_fires() -> None:
    """Template ``close > donchian`` compares to the MIDDLE band (primary_output), not upper."""
    obs = _run("close > donchian")
    assert sum(obs["fired"]) > 0
    t = len(_CLOSES) - 1
    assert obs["middle"][t] == pytest.approx((obs["upper"][t] + obs["lower"][t]) / 2)
    assert obs["close"][t] > obs["middle"][t]
    assert obs["close"][t] < obs["upper"][t]


def test_position_is_capped_at_one_so_ga_breakout_threshold_cannot_fire_above_one() -> None:
    # position > 1.0 (a breakout beyond the channel) is unreachable...
    assert sum(_run("donchian.position > 1.0")["fired"]) == 0
    # ...and ">= 1" needs close == high, which this series (close = high - 10) never has.
    assert sum(_run("donchian.position >= 1.0")["fired"]) == 0
    # A sub-1 threshold fires (close sits in the top of the channel).
    assert sum(_run("donchian.position > 0.8")["fired"]) > 0
