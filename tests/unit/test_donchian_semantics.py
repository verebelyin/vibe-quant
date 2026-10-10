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

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.unit.test_fill_timing import _INSTRUMENT_ID, _make_bar, _new_engine
from vibe_quant.data.catalog import get_bar_type
from vibe_quant.dsl.compiler import StrategyCompiler, _to_class_name
from vibe_quant.dsl.parser import parse_strategy, validate_strategy_dict

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar

    from vibe_quant.dsl.schema import StrategyDSL

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


# ---------------------------------------------------------------------------
# vibe-quant-yul7u.25: opt-in ``upper_prev`` / ``lower_prev`` (channel of the
# PREVIOUS bar) so a real breakout is expressible.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE_PATH = _REPO_ROOT / "vibe_quant" / "strategies" / "templates" / "donchian_breakout.yaml"


def _donchian_dsl(condition: str | list[str], name: str = "donchian_prev_probe") -> StrategyDSL:
    return validate_strategy_dict(
        {
            "name": name,
            "timeframe": "1m",
            "indicators": {"donchian": {"type": "DONCHIAN", "period": PERIOD}},
            "entry_conditions": {"long": [condition] if isinstance(condition, str) else condition},
            "exit_conditions": {},
            "stop_loss": {"type": "fixed_pct", "percent": 50.0},
            "take_profit": {"type": "fixed_pct", "percent": 50.0},
        }
    )


def _run_prev(condition: str, bars: list[Bar]) -> dict[str, list[float | None]]:
    """Compile a DONCHIAN strategy and observe the compute_fn-path prev outputs.

    Unlike :func:`_run`, values are read AFTER ``super().on_bar``: the
    ``*_prev`` outputs live on the compute_fn path, whose bar buffer is fed
    inside ``on_bar`` (the NT-path probe reads before, since NT updates
    registered indicators first).
    """
    dsl = _donchian_dsl(condition)
    module = StrategyCompiler().compile_to_module(dsl)
    camel = _to_class_name(dsl.name)
    base_cls = getattr(module, f"{camel}Strategy")
    config_cls = getattr(module, f"{camel}Config")

    obs: dict[str, list[float | None]] = {
        k: [] for k in ("upper_prev", "lower_prev", "high", "low", "close", "fired")
    }

    class Probe(base_cls):  # type: ignore[valid-type, misc]
        def on_bar(self, bar: Bar) -> None:
            super().on_bar(bar)
            if bar.bar_type == self.bar_type_1m:
                obs["upper_prev"].append(self._pta_values.get("donchian_upper_prev"))
                obs["lower_prev"].append(self._pta_values.get("donchian_lower_prev"))
                obs["high"].append(float(bar.high))
                obs["low"].append(float(bar.low))
                obs["close"].append(float(bar.close))

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
    engine.add_data(bars)
    engine.add_strategy(strategy)
    try:
        engine.run()
    finally:
        engine.reset()
        engine.dispose()
    return obs


def _breakout_bars() -> list[Bar]:
    """Ten flat bars (high 100, close 95) then one bar closing at 130."""
    bar_type = get_bar_type("BTCUSDT", "1m")
    bars = [_make_bar(bar_type, 94.0, 100.0, 90.0, 95.0, minute=i) for i in range(10)]
    bars.append(_make_bar(bar_type, 125.0, 135.0, 125.0, 130.0, minute=len(bars)))
    return bars


# Distinct highs/lows so the trailing max/min is not trivially constant.
_PREV_HIGHS = [10.0, 12.0, 11.0, 15.0, 14.0, 13.0, 20.0, 19.0, 18.0, 25.0]
_PREV_LOWS = [8.0, 9.0, 7.0, 11.0, 10.0, 9.0, 14.0, 13.0, 12.0, 18.0]
_PREV_CLOSES = [9.0, 10.5, 9.5, 13.0, 12.0, 11.0, 17.0, 16.0, 15.0, 22.0]


def _prev_value_bars() -> list[Bar]:
    bar_type = get_bar_type("BTCUSDT", "1m")
    return [
        _make_bar(bar_type, c - 0.5, h, lo, c, minute=i)
        for i, (h, lo, c) in enumerate(zip(_PREV_HIGHS, _PREV_LOWS, _PREV_CLOSES, strict=True))
    ]


def test_crosses_above_upper_prev_fires_on_breakout() -> None:
    """A genuine N-bar breakout is expressible and fires at least once."""
    obs = _run_prev("close crosses_above donchian.upper_prev", _breakout_bars())
    assert obs["fired"], "entry check never ran"
    assert sum(obs["fired"]) >= 1, obs


def test_upper_prev_and_lower_prev_are_prior_n_bar_channel() -> None:
    obs = _run_prev("close crosses_above donchian.upper_prev", _prev_value_bars())
    highs = obs["high"]
    lows = obs["low"]
    assert len(obs["upper_prev"]) == len(highs)
    for t in range(PERIOD, len(highs)):
        expected_upper = max(highs[t - PERIOD : t])
        expected_lower = min(lows[t - PERIOD : t])
        assert obs["upper_prev"][t] == pytest.approx(expected_upper), f"upper_prev bar {t}"
        assert obs["lower_prev"][t] == pytest.approx(expected_lower), f"lower_prev bar {t}"
    # Genuinely lagged: at the first valid bar the prior-window high is NOT the
    # current-bar-inclusive high (what the NT path would expose).
    assert obs["upper_prev"][PERIOD] == pytest.approx(max(highs[0:PERIOD]))
    assert obs["upper_prev"][PERIOD] < highs[PERIOD]


def test_prev_outputs_are_opt_in_and_leave_the_nt_path_untouched() -> None:
    """Referencing ``*_prev`` forces the compute_fn path; otherwise NT is kept."""
    src_nt = StrategyCompiler().compile(_donchian_dsl("close > donchian.upper"))
    src_prev = StrategyCompiler().compile(_donchian_dsl("close crosses_above donchian.upper_prev"))
    assert "DonchianChannel(" in src_nt
    assert "self.ind_donchian" in src_nt
    assert "DonchianChannel(" not in src_prev
    assert "self.ind_donchian" not in src_prev
    assert "compute_donchian" in src_prev
    assert "donchian_upper_prev" in src_prev


def test_donchian_breakout_template_is_a_real_prior_bar_breakout() -> None:
    dsl = parse_strategy(_TEMPLATE_PATH)
    assert "upper_prev" in " ".join(dsl.entry_conditions.long)
    assert "lower_prev" in " ".join(dsl.entry_conditions.short)
    src = StrategyCompiler().compile(dsl)
    # A *_prev read is not an NT output → the donchian instance runs compute_fn.
    assert "donchian_upper_prev" in src
    assert "compute_donchian" in src
    assert "DonchianChannel(" not in src


def test_donchian_breakout_template_exits_use_prev_channel() -> None:
    """Pin the exits: long exits on lower_prev, short exits on upper_prev."""
    dsl = parse_strategy(_TEMPLATE_PATH)
    assert dsl.exit_conditions.long == ["close crosses_below donchian.lower_prev"]
    assert dsl.exit_conditions.short == ["close crosses_above donchian.upper_prev"]


def test_first_valid_prev_output_appears_at_bar_index_period() -> None:
    """shift(1) needs period+1 bars: first non-NaN upper_prev/lower_prev is bar index N."""
    obs = _run_prev("close crosses_above donchian.upper_prev", _prev_value_bars())
    for key in ("upper_prev", "lower_prev"):
        first = next(i for i, v in enumerate(obs[key]) if v is not None)
        assert first == PERIOD, f"{key} first valid at {first}"


def test_donchian_spec_lookback_is_period_plus_one() -> None:
    """``shift(1)`` outputs need N+1 buffered bars (the NaN guard hides a too-small value
    behaviourally, so pin the spec value directly)."""
    from vibe_quant.dsl.indicators import pta_lookback

    assert pta_lookback("DONCHIAN", {"period": PERIOD}) == PERIOD + 1
    assert pta_lookback("DONCHIAN", {}) == 21


def _run_position_mix(condition: str | list[str], bars: list[Bar]) -> dict[str, list[float | None]]:
    """Observe ``donchian.position`` on whichever path the strategy's conditions select.

    ``nt`` = ``compute_position`` on the live NT ``DonchianChannel`` (read before
    ``super().on_bar``, like the generated condition code on the NT path);
    ``read`` = what the generated ``_get_indicator_value("donchian_position")``
    returns after ``super().on_bar`` (the NT-path strategy computes it from NT
    state, the compute_fn-path strategy from ``_pta_values``).
    """
    dsl = _donchian_dsl(condition, name="donchian_pos_mix")
    module = StrategyCompiler().compile_to_module(dsl)
    camel = _to_class_name(dsl.name)
    base_cls = getattr(module, f"{camel}Strategy")
    config_cls = getattr(module, f"{camel}Config")
    obs: dict[str, list[float | None]] = {"read": [], "fired": []}

    class Probe(base_cls):  # type: ignore[valid-type, misc]
        def on_bar(self, bar: Bar) -> None:
            super().on_bar(bar)
            if bar.bar_type == self.bar_type_1m:
                obs["read"].append(self._get_indicator_value("donchian_position"))

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
    engine.add_data(bars)
    engine.add_strategy(strategy)
    try:
        engine.run()
    finally:
        engine.reset()
        engine.dispose()
    return obs


def test_position_on_compute_fn_path_equals_nt_path_position() -> None:
    """Mixing ``upper_prev`` with ``position`` must not zero ``position``.

    The ``upper_prev`` reference forces the compute_fn path, where ``position``
    was never produced (``_pta_values.get(..., 0.0)`` -> constant 0.0, so
    ``position > x`` never fired). Values must match the NT-path position
    (``derived.compute_position``) on the same bars, bar for bar.
    """
    bars = _prev_value_bars()
    nt = _run_position_mix("donchian.position > 0.0", bars)["read"]
    mixed = _run_position_mix(
        ["close crosses_above donchian.upper_prev", "donchian.position > 0.5"], bars
    )
    assert len(nt) == len(mixed["read"]) == len(bars)
    # Expected value from first principles (same-bar bands, close of that bar).
    # compute_fn path buffers period+1 bars (shift(1) outputs), so it is ready from bar N.
    for t in range(PERIOD, len(bars)):
        hi = max(_PREV_HIGHS[t - PERIOD + 1 : t + 1])
        lo = min(_PREV_LOWS[t - PERIOD + 1 : t + 1])
        expected = (_PREV_CLOSES[t] - lo) / (hi - lo) if hi > lo else 0.5
        assert nt[t] == pytest.approx(expected), f"NT bar {t}"
        assert mixed["read"][t] == pytest.approx(expected), f"compute_fn bar {t}"
        assert mixed["read"][t] == pytest.approx(nt[t]), f"parity bar {t}"
    values = [v for v in mixed["read"][PERIOD:] if v is not None]
    assert any(v > 0.0 for v in values), "position stuck at 0.0 on the compute_fn path"
    assert len({round(v, 9) for v in values}) > 1, "position constant on the compute_fn path"


def test_position_neutral_half_when_channel_collapses_on_compute_fn_path() -> None:
    """Zero-width channel -> 0.5 (derived.compute_position), not a divide-by-zero."""
    bar_type = get_bar_type("BTCUSDT", "1m")
    flat = [_make_bar(bar_type, 50.0, 50.0, 50.0, 50.0, minute=i) for i in range(6)]
    mixed = _run_position_mix("close crosses_above donchian.upper_prev", flat)["read"]
    assert all(v == pytest.approx(0.5) for v in mixed[PERIOD:])
