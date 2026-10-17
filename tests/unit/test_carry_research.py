"""Research evaluator: hooks, persistence null, p-values, cap, DSR (task T8).

Contract rows R35 (the ``simulate_reference`` hooks default to a byte-identical
run), R36 (a ruined null draw is Sharpe ``-inf``, counted, and the p-value is
``(1 + #{null >= strat}) / (1 + draws)``), R36b (``p_keep`` from the strategy's
own picks; a seeded null is reproducible), R37 (the funding cap applies to the
signal AND to paid funding while the module rate cache stays untouched) and
R38/R41 (the DSR trial count is ``3 + |recorded spec hashes u {evaluated}|``) of
``data/swarm/20261015-funding-carry/design.md`` §6, §8-T8, plus the §8-T8
``evaluate`` output keys and the CLI ``evaluate`` subcommand.

The fixture is copied from the CLI test builders on purpose (not imported): a
test that reaches into another test file's fixture breaks when that file changes.
It is a four-week two-symbol run (BTCUSDT + ETHUSDT) with a weekly rate switch
(so the book flips and its equity moves) and flat sizing ticks at the daily
boundary marks, which gives four weekly returns -- a real headline Sharpe, not
the R50 null.
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pytest

import vibe_quant.portfolio.__main__ as carry_cli
import vibe_quant.portfolio.research as research
from vibe_quant.data.catalog import create_instrument
from vibe_quant.data.mark_bars import MarkSeries
from vibe_quant.overfitting.dsr import (
    DeflatedSharpeRatio,
    calculate_dsr,
    daily_sharpe_inputs,
    mtm_dsr_calculate,
)
from vibe_quant.portfolio.metrics import CarryReport, carry_metrics
from vibe_quant.portfolio.parity import compare_runs
from vibe_quant.portfolio.reference import simulate_reference
from vibe_quant.portfolio.signal import Settlements
from vibe_quant.portfolio.sizing import Decision, InstrumentSpec, nt_commission
from vibe_quant.portfolio.spec import (
    CarryDataError,
    CarryEquityTooSmall,
    CarrySpec,
    decision_times,
)
from vibe_quant.portfolio.ticks import CarryInputs, PriceTick
from vibe_quant.validation.funding import (
    _RATE_CACHE,
    FundingCalculator,
    clear_rate_cache,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

_MIN = 60 * 10**9
_HOUR = 3_600 * 10**9
_DAY = 86_400 * 10**9
_WEEK = 7 * _DAY
_EIGHT_HOURS = 8 * _HOUR
_BAR_TS_INIT = 59_999_000_000  # catalog 1m bar: ts_init = open + 59.999 s
_JITTER_MS = 3  # archived funding_time carries a few ms after the boundary
_JITTER_NS = _JITTER_MS * 10**6

_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_SIZING = {"BTCUSDT": 51000.0, "ETHUSDT": 3000.0}
_FILL = {
    "BTCUSDT": (51010.0, 52010.0, 51010.0, 52010.0),
    "ETHUSDT": (3010.0, 3110.0, 3010.0, 3110.0),
}
# The low-funding leg alternates every week, so the book flips each week.
_ALT = {"BTCUSDT": (0.0001, 0.0002), "ETHUSDT": (0.0002, 0.0001)}
_START = int(datetime.fromisoformat("2025-01-06").replace(tzinfo=UTC).timestamp()) * 10**9


def _ns(date: str) -> int:
    """``YYYY-MM-DD`` 00:00 UTC -> epoch ns."""
    return int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp()) * 10**9


def _alt_rate(symbol: str, boundary: int) -> float:
    """The default archive: low/high funding swap every week (the book flips)."""
    return _ALT[symbol][((boundary - _START) // _WEEK) % 2]


def _marks(price: float, end_ns: int) -> MarkSeries:
    """A flat 1m MARK series over the fixture window (one bar opening per minute)."""
    ts = np.arange(_START - _EIGHT_HOURS, end_ns, _MIN, dtype=np.int64)
    values = np.full(ts.shape, price, dtype=np.float64)
    return MarkSeries(ts=ts, low=values, high=values, close=values, open=values)


def _inputs(
    tmp_path: Path,
    *,
    weeks: int = 4,
    rate_fn: Callable[[str, int], float] = _alt_rate,
    name: str = "a",
) -> CarryInputs:
    """A ``weeks``-long two-symbol fixture: sizing ticks flat, fill ticks move."""
    end_ns = _START + weeks * _WEEK
    decisions = tuple(decision_times(_START, end_ns))
    boundaries = list(range(_START - 9 * _DAY, end_ns, _EIGHT_HOURS))
    spec = CarrySpec(symbols=_SYMBOLS, start_ns=_START, end_ns=end_ns)
    instruments = {symbol: create_instrument(symbol) for symbol in _SYMBOLS}
    path = tmp_path / f"raw_{name}.db"
    clear_rate_cache()
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE raw_funding_rates (
            id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, funding_time INTEGER NOT NULL,
            funding_rate REAL NOT NULL, mark_price REAL, source TEXT,
            UNIQUE(symbol, funding_time))"""
    )
    for symbol in _SYMBOLS:
        for boundary in boundaries:
            conn.execute(
                "INSERT INTO raw_funding_rates (symbol, funding_time, funding_rate) VALUES (?,?,?)",
                (symbol, boundary // 1_000_000 + _JITTER_MS, rate_fn(symbol, boundary)),
            )
    conn.commit()
    conn.close()
    return CarryInputs(
        spec=spec,
        decisions=decisions,
        instruments=instruments,
        specs={
            symbol: InstrumentSpec.from_instrument(instrument)
            for symbol, instrument in instruments.items()
        },
        ticks={
            symbol: tuple(
                tick
                for index, decision_ns in enumerate(decisions)
                for tick in (
                    PriceTick(decision_ns - 1_000_000, _SIZING[symbol]),
                    PriceTick(decision_ns + _BAR_TS_INIT, _FILL[symbol][index]),
                )
            )
            for symbol in _SYMBOLS
        },
        settlements={
            symbol: Settlements(
                times_ns=tuple(boundary + _JITTER_NS for boundary in boundaries),
                rates=tuple(rate_fn(symbol, boundary) for boundary in boundaries),
            )
            for symbol in _SYMBOLS
        },
        marks={symbol: _marks(_SIZING[symbol], end_ns) for symbol in _SYMBOLS},
        calc=FundingCalculator(path),
        boundary_prices={
            symbol: {boundary: _SIZING[symbol] for boundary in range(_START, end_ns + _DAY, _DAY)}
            for symbol in _SYMBOLS
        },
    )


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    clear_rate_cache()


def _cache_snapshot() -> dict[tuple[str, str], tuple[list[int], list[float]]]:
    """A deep copy of the module funding-rate cache (lists are mutable)."""
    return {key: (list(times), list(rates)) for key, (times, rates) in _RATE_CACHE.items()}


def _decision(ts: int, signals: Mapping[str, float | None], weights: Mapping[str, float]) -> Decision:
    """A minimal ``Decision`` carrying only the fields ``_persistence`` reads."""
    return Decision(
        ts=ts,
        equity=10_000.0,
        signals=dict(signals),
        excluded={},
        weights=dict(weights),
        prices={},
        target_steps={},
        orders=(),
        band_skips=0,
        min_notional_skips=0,
        rounding_drift=0.0,
    )


# --- R35: the hooks default to a byte-identical run ------------------------------------


def test_hooks_default_is_identical(tmp_path: Path) -> None:
    """R35: ``simulate_reference(inputs)`` == ``simulate_reference(inputs, ranker=None, fee_multiplier=1.0)``."""
    inputs = _inputs(tmp_path)

    default = simulate_reference(inputs)
    hooked = simulate_reference(inputs, ranker=None, fee_multiplier=1.0)

    assert default.fills and default.decisions
    assert (
        compare_runs(
            default, hooked, carry_metrics(default, inputs), carry_metrics(hooked, inputs)
        )
        == []
    )


# --- R36: null accounting --------------------------------------------------------------


_SEED0 = 7  # the harness reads a draw's index off its seed: ``seed - _SEED0``
_STRATEGY_POST, _STRATEGY_PRE, _FEE2X, _CAPPED, _PLACEBO = 1.0, 1.5, 0.4, 0.3, -1.0


class _SentinelRanker:
    """A stand-in for ``PersistenceMatchedRanker`` that records what ``evaluate`` hands it."""

    def __init__(self, p_keep: float, seed: int) -> None:
        self.p_keep = p_keep
        self.seed = seed

    def __call__(self, decision_ns: int, signals: Mapping[str, float | None]) -> dict[str, float | None]:
        return dict(signals)


class _FakeRun:
    """A run stand-in: ``decisions`` feeds ``p_keep``; ``sharpe`` is what the metrics fake reports."""

    def __init__(self, sharpe: float | None, decisions: tuple[Decision, ...] = ()) -> None:
        self.sharpe = sharpe
        self.decisions = decisions


class _Harness:
    """Fake seams of ``evaluate`` + a log of every ``simulate_reference`` call."""

    def __init__(
        self,
        inputs: CarryInputs,
        post: list[float | str],
        pre: list[float | str],
        decisions: tuple[Decision, ...],
    ) -> None:
        self.inputs = inputs
        self.post = post
        self.pre = pre
        self.decisions = decisions
        self.factory_calls: list[tuple[float, int]] = []
        self.draw_fees: list[float] = []  # fee_multiplier of every null-draw simulate call
        self.strategy_fees: list[float] = []  # fee_multiplier of every ranker=None, same-inputs call
        self.other_kinds: list[tuple[str, float]] = []  # placebo / capped, with their fee
        self.weekly_runs: list[float | None] = []  # every draw's ``weekly_sharpe_only`` run
        self.report_calls = 0  # the §4 reports a draw must never build

    def factory(self, p_keep: float, seed: int) -> _SentinelRanker:
        self.factory_calls.append((p_keep, seed))
        return _SentinelRanker(p_keep, seed)

    def simulate(
        self, inputs: CarryInputs, *, ranker: object = None, fee_multiplier: float = 1.0
    ) -> _FakeRun:
        if isinstance(ranker, _SentinelRanker):
            self.draw_fees.append(fee_multiplier)
            value = (self.post if fee_multiplier == 1.0 else self.pre)[ranker.seed - _SEED0]
            if value == "ruined":
                raise CarryEquityTooSmall("<book>", 0, 1.0, 2.0)
            if value == "data_error":
                raise CarryDataError(f"no price for held BTCUSDT at draw {ranker.seed - _SEED0}")
            return _FakeRun(float(value))
        if ranker is not None:
            self.other_kinds.append(("placebo", fee_multiplier))
            return _FakeRun(_PLACEBO)
        if inputs is not self.inputs:
            self.other_kinds.append(("capped", fee_multiplier))
            return _FakeRun(_CAPPED)
        self.strategy_fees.append(fee_multiplier)
        sharpe = {1.0: _STRATEGY_POST, 0.0: _STRATEGY_PRE, 2.0: _FEE2X}[fee_multiplier]
        return _FakeRun(sharpe, self.decisions)

    def metrics(self, run: _FakeRun, inputs: CarryInputs) -> CarryReport:
        self.report_calls += 1
        return CarryReport(
            summary={"weekly_sharpe": run.sharpe, "weeks": 4},
            daily=[],
            weekly=[],
            legs=[],
            by_year=[],
        )

    def weekly(self, run: _FakeRun, inputs: CarryInputs) -> float | None:
        """The metrics-level seam a draw reads instead of the full report (T8).

        A draw handed ``-inf`` stands for one whose ruin is reported HERE (the CF1 daily
        scan) rather than inside ``simulate_reference``.
        """
        if run.sharpe == float("-inf"):
            raise CarryEquityTooSmall("<book>", 0, 1.0, 2.0)
        self.weekly_runs.append(run.sharpe)
        return run.sharpe


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    inputs: CarryInputs,
    post: list[float | str],
    pre: list[float | str],
    decisions: tuple[Decision, ...] = (),
) -> _Harness:
    """Patch the seams ``evaluate`` drives: the factory, simulate, metrics and the weekly Sharpe."""
    harness = _Harness(inputs, post, pre, decisions)
    monkeypatch.setattr(research, "PersistenceMatchedRanker", harness.factory)
    monkeypatch.setattr(research, "simulate_reference", harness.simulate)
    monkeypatch.setattr(research, "carry_metrics", harness.metrics)
    monkeypatch.setattr(research, "weekly_sharpe_only", harness.weekly)
    return harness


def test_null_draws_read_the_weekly_only_sharpe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T8: a draw reads the Monday-only weekly Sharpe; only the §4 reports build a full report."""
    inputs = _inputs(tmp_path, name="weeklyonly")
    harness = _install_fakes(monkeypatch, inputs, [0.5, 0.6, 0.7], [1.0, 1.1, 1.2])

    result = research.evaluate(inputs, draws=3, seed=_SEED0, trials_path=tmp_path / "trials.jsonl")

    assert harness.weekly_runs == [0.5, 0.6, 0.7, 1.0, 1.1, 1.2]  # the post-cost draws, then pre-cost
    assert harness.report_calls == 5  # strategy, pre-cost strategy, placebo, 2x fees, capped
    assert result["p_value"] == 1 / 4  # 1.0 strategy beats all three post-cost draws
    assert result["p_value_pre_cost"] == 1 / 4  # 1.5 beats 1.0/1.1/1.2


def test_ruined_draw_from_the_weekly_seam_is_inf_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R36/CF1: a ruin the weekly seam reports is ``-inf`` and counted, never dropped."""
    inputs = _inputs(tmp_path, name="weeklyruin")
    ruin = float("-inf")  # the harness's marker: this draw's ruin comes from the weekly seam
    harness = _install_fakes(monkeypatch, inputs, [0.5, ruin, 2.0], [1.6, ruin, 1.5])

    result = research.evaluate(inputs, draws=3, seed=_SEED0, trials_path=tmp_path / "trials.jsonl")

    assert harness.weekly_runs == [0.5, 2.0, 1.6, 1.5]  # the ruined draw reports no value
    assert result["ruined_draws"] == 1  # the pre-cost pass's ruin is not double counted
    assert result["p_value"] == 0.5  # only the 2.0 draw reaches the 1.0 strategy: (1 + 1) / 4
    assert result["p_value_pre_cost"] == 0.75  # 1.6 and the equal 1.5 beat the 1.5 strategy


def test_ruined_draw_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R36: a ruined draw is ``-inf``, counted, and p = (1 + #{null >= strat}) / (1 + draws)."""
    inputs = _inputs(tmp_path, name="ruin")
    harness = _install_fakes(monkeypatch, inputs, [0.5, "ruined", 2.0], [1.6, "ruined", 1.5])

    result = research.evaluate(
        inputs, draws=3, seed=_SEED0, trials_path=tmp_path / "trials.jsonl"
    )

    assert result["draws"] == 3
    assert result["ruined_draws"] == 1  # the pre-cost pass's ruined draw is not double counted
    # strategy 1.0; only the 2.0 draw is >= 1.0 -> (1 + 1) / (1 + 3)
    assert result["p_value"] == 0.5
    # pre-cost strategy 1.5; the 1.6 draw and the equal 1.5 draw count -> (1 + 2) / (1 + 3);
    # reading the post-cost draws here (or the post-cost strategy) would give 0.5 / 1.0
    assert result["p_value_pre_cost"] == 0.75
    assert harness.factory_calls  # the factory really built the draws


def test_data_error_in_one_null_draw_aborts_the_whole_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5: a ``CarryDataError`` in a later null draw (k>0) propagates -- never skipped.

    Only a ruined draw (``CarryEquityTooSmall``) is absorbed; a missing price is a
    fail-closed data error, so ``evaluate`` must not catch it and keep going.
    """
    inputs = _inputs(tmp_path, name="abort")
    _install_fakes(monkeypatch, inputs, [0.5, "data_error", 2.0], [1.6, "ruined", 1.5])

    with pytest.raises(CarryDataError, match="draw 1"):
        research.evaluate(inputs, draws=3, seed=_SEED0, trials_path=tmp_path / "trials.jsonl")


def test_cost_passes_use_the_right_fee_multiplier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6: pre-cost = ``fee_multiplier=0.0`` for the strategy AND every draw; post-cost = 1.0."""
    inputs = _inputs(tmp_path, name="fees")
    harness = _install_fakes(monkeypatch, inputs, [0.5, 0.6, 0.7], [1.0, 1.1, 1.2])

    result = research.evaluate(
        inputs, draws=3, seed=_SEED0, trials_path=tmp_path / "trials.jsonl"
    )

    assert sorted(harness.draw_fees) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert sorted(harness.strategy_fees) == [0.0, 1.0, 2.0]  # pre-cost, headline, fee stress
    assert harness.other_kinds == [("placebo", 1.0), ("capped", 1.0)]
    # each stress run feeds its own key
    assert result["fee2x_weekly_sharpe"] == _FEE2X
    assert result["capped_weekly_sharpe"] == _CAPPED
    assert result["placebo_weekly_sharpe"] == _PLACEBO
    # strategy 1.0 beats every post draw, 1.5 beats every pre draw
    assert result["p_value"] == 1 / 4
    assert result["p_value_pre_cost"] == 1 / 4


def test_null_draws_get_the_strategys_p_keep_and_seed_plus_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6/R36b: draw ``i`` is ``PersistenceMatchedRanker(strategy p_keep, seed + i)``."""
    inputs = _inputs(tmp_path, name="seeds")
    same = {"A": 0.0, "B": 0.1, "C": 0.2}
    # picks A/B, A/B, then C/B: one repeat out of two consecutive pairs -> p_keep 1/2
    decisions = (
        _decision(1, same, {"A": 0.5, "B": -0.5, "C": 0.0}),
        _decision(2, same, {"A": 0.5, "B": -0.5, "C": 0.0}),
        _decision(3, same, {"A": 0.0, "B": -0.5, "C": 0.5}),
    )
    harness = _install_fakes(monkeypatch, inputs, [0.5] * 3, [0.5] * 3, decisions)

    result = research.evaluate(
        inputs, draws=3, seed=_SEED0, trials_path=tmp_path / "trials.jsonl"
    )

    assert result["p_keep"] == 0.5
    # the post-cost pass first, then the pre-cost pass: both reuse seeds seed + 0..2
    expected = [(0.5, _SEED0 + index) for index in range(3)]
    assert harness.factory_calls == expected + expected


# --- R36b: persistence ----------------------------------------------------------------


def test_persistence_from_strategy_picks() -> None:
    """R36b: picks [A/B, A/B, C/B, C/B] -> 2/3; a flat decision breaks the chain."""
    d1 = _decision(1, {"A": 0.0, "B": 0.1, "C": 0.2}, {"A": 0.5, "B": -0.5, "C": 0.0})
    d2 = _decision(2, {"A": 0.0, "B": 0.1, "C": 0.2}, {"A": 0.5, "B": -0.5, "C": 0.0})
    d3 = _decision(3, {"A": None, "B": 0.1, "C": 0.0}, {"A": 0.0, "B": -0.5, "C": 0.5})
    d4 = _decision(4, {"A": None, "B": 0.1, "C": 0.0}, {"A": 0.0, "B": -0.5, "C": 0.5})

    assert research._persistence((d1, d2, d3, d4)) == 2 / 3

    # a decision with n < 2 contributes to neither side of the fraction
    flat = _decision(5, {"A": 0.0, "B": None, "C": None}, {"A": 0.0, "B": 0.0, "C": 0.0})
    assert research._persistence((d1, flat, d1)) == 0.0
    assert research._persistence((d1,)) == 0.0

    # n == 2 is still a "decision with n >= 2": the pair counts
    e1 = _decision(6, {"A": 0.0, "B": 0.1}, {"A": 0.5, "B": -0.5})
    e2 = _decision(7, {"A": 0.0, "B": 0.1}, {"A": 0.5, "B": -0.5})
    assert research._persistence((e1, e2)) == 1.0


def test_p_value_counts_equality() -> None:
    """§6: a null Sharpe EXACTLY equal to the strategy's counts as ``>=`` it."""
    assert research._p_value(1.0, [1.0, 0.5, float("-inf")]) == (1 + 1) / 4
    assert research._p_value(None, [1.0]) is None


def test_persistence_matched_null_seeded() -> None:
    """R36b: the null ranker is seeded (reproducible) and honours ``p_keep``."""
    signals = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

    assert research.PersistenceMatchedRanker(0.0, 3)(0, signals) == research.PersistenceMatchedRanker(
        0.0, 3
    )(0, signals)

    kept = research.PersistenceMatchedRanker(1.0, 3)
    first = kept(0, signals)
    assert kept(7, signals) == first  # p_keep == 1.0 always keeps the previous ordering
    assert sorted(v for v in first.values() if v is not None) == [0.0, 1.0, 2.0, 3.0]  # rank positions, distinct floats

    reshuffled = research.PersistenceMatchedRanker(0.0, 3)
    assert sorted(v for v in reshuffled(0, signals).values() if v is not None) == [0.0, 1.0, 2.0, 3.0]
    # a different seed draws a different permutation of the same four symbols
    assert research.PersistenceMatchedRanker(0.0, 4)(0, signals) != research.PersistenceMatchedRanker(
        0.0, 3
    )(0, signals)


# --- R37: capped funding --------------------------------------------------------------


def test_cap_applies_to_signal_and_paid(tmp_path: Path) -> None:
    """R37: a ±0.1% cap hits the signal and paid funding; the module cache is untouched."""
    settlement = _START + 8 * _HOUR  # inside (D1, end); paid while the book is held

    def rate_fn(symbol: str, boundary: int) -> float:
        # a signal row inside D1's lookback window and a paid row after D1
        if boundary == _START - 8 * _HOUR:
            return 0.003 if symbol == "BTCUSDT" else -0.004
        if boundary == settlement:
            return 0.003 if symbol == "BTCUSDT" else -0.004
        return 0.0

    inputs = _inputs(tmp_path, weeks=1, rate_fn=rate_fn, name="cap")
    # warm the process cache with the TRUE (unclipped) rates, then snapshot it
    inputs.calc.settlements("BTCUSDT")
    inputs.calc.settlements("ETHUSDT")
    before = _cache_snapshot()
    assert any(0.003 in rates for _, rates in before.values())  # the true rates are cached

    capped = research.capped_inputs(inputs, 0.001)

    assert isinstance(capped.calc, research.CappedFundingCalculator)
    assert set(capped.settlements["BTCUSDT"].rates) == {0.0, 0.001}
    assert set(capped.settlements["ETHUSDT"].rates) == {0.0, -0.001}
    assert _cache_snapshot() == before  # clipped values never reach the module cache
    assert any(0.003 in rates for _, rates in _cache_snapshot().values())

    run = simulate_reference(capped)

    # the signal itself is clipped (long ETH / short BTC under either value)
    assert run.decisions[0].signals == {"BTCUSDT": 0.001, "ETHUSDT": -0.001}
    # BTC short -0.098 @ 51000, ETH long 1.666 @ 3000: both paid (clipped) 4.998
    assert [payment.amount for payment in run.payments if payment.amount != 0.0] == [
        -4.998,
        -4.998,
    ]

    uncapped = simulate_reference(inputs)
    assert [payment.amount for payment in uncapped.payments if payment.amount != 0.0] == [
        -14.994,
        -19.992,
    ]


# --- R38 / R41: the DSR trial count ----------------------------------------------------


def test_dsr_trial_count(tmp_path: Path) -> None:
    """R38/R41: N = 3 prior + |recorded spec hashes u {evaluated}|."""
    inputs = _inputs(tmp_path, name="dsr")
    spec_hash = inputs.spec.spec_hash()

    empty = research.evaluate(inputs, draws=0, seed=0, trials_path=tmp_path / "absent.jsonl")
    assert empty["draws"] == 0
    assert empty["ruined_draws"] == 0
    assert empty["p_value"] == 1.0  # (1 + 0) / (1 + 0)
    assert empty["dsr_num_trials"] == 4  # 3 prior + the evaluated spec alone
    assert empty["trials_sharpe_variance"] is None  # < 2 recorded variants

    recorded = tmp_path / "trials.jsonl"
    recorded.write_text(json.dumps({"spec_hash": spec_hash, "weekly_sharpe": 0.5}) + "\n")
    one = research.evaluate(inputs, draws=0, seed=0, trials_path=recorded)
    assert one["dsr_num_trials"] == 4  # the recorded hash IS the evaluated hash

    two = tmp_path / "trials2.jsonl"
    two.write_text(
        json.dumps({"spec_hash": "deadbeef", "weekly_sharpe": 0.5})
        + "\n"
        + json.dumps({"spec_hash": spec_hash, "weekly_sharpe": 0.5})
        + "\n"
    )
    both = research.evaluate(inputs, draws=0, seed=0, trials_path=two)
    assert both["dsr_num_trials"] == 5  # |{deadbeef, evaluated}| = 2 -> 3 + 2
    assert both["trials_sharpe_variance"] == 0.0  # two equal recorded Sharpes


def test_evaluate_output_keys(tmp_path: Path) -> None:
    """§8-T8 + R56: the result dict carries every ruled key."""
    inputs = _inputs(tmp_path, name="keys")

    result = research.evaluate(inputs, draws=0, seed=0, trials_path=tmp_path / "absent.jsonl")

    assert set(result) == {
        "p_value",
        "p_value_pre_cost",
        "ruined_draws",
        "draws",
        "p_keep",
        "placebo_weekly_sharpe",
        "beta_btc",
        "alpha_t_btc",
        "beta_ew",
        "alpha_t",
        "fee2x_weekly_sharpe",
        "capped_weekly_sharpe",
        "drop_top3_weekly_sharpe",
        "placebo_kind",
        "dsr",
        "dsr_p_value",
        "dsr_error",
        "dsr_raw_p_value",
        "dsr_num_trials",
        "trials_sharpe_variance",
        "by_year",
    }
    assert result["placebo_kind"] == "reversal"  # R59: longs the LOWEST trailing return
    assert result["by_year"] == carry_metrics(simulate_reference(inputs), inputs).by_year


def test_fee2x_doubles_the_commission(tmp_path: Path) -> None:
    """§6 stress: ``fee_multiplier=2.0`` doubles the NT taker commission exactly."""
    instrument = create_instrument("BTCUSDT")
    quantity = instrument.make_qty(Decimal("0.098"))
    price = instrument.make_price(51010.0)
    single_fee = nt_commission(instrument, quantity, price)

    assert nt_commission(instrument, quantity, price, Decimal(2)) == single_fee * 2

    inputs = _inputs(tmp_path, name="fee2x")
    single = simulate_reference(inputs, fee_multiplier=1.0)
    doubled = simulate_reference(inputs, fee_multiplier=2.0)

    # the first decision sizes off equity 10,000 in both runs, so its fills' fees double
    assert [fill.fee for fill in doubled.fills[:2]] == [fill.fee * 2.0 for fill in single.fills[:2]]
    assert math.fsum(fill.fee for fill in doubled.fills) > math.fsum(fill.fee for fill in single.fills)


def test_beta_ols_on_exact_line(tmp_path: Path) -> None:
    """§6/R56: a returns series exactly ``0.5 * x`` gives beta 0.5 and no alpha t-stat."""
    assert research._ols([0.5 * x for x in (1.0, 2.0, 3.0, 4.0)], [1.0, 2.0, 3.0, 4.0]) == (
        0.5,
        None,
    )
    # n == 3 is the smallest regression the helper accepts
    assert research._ols([0.5, 1.0, 1.5], [1.0, 2.0, 3.0]) == (0.5, None)
    beta, alpha_t = research._ols([1.0, 2.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0, 4.0])
    assert beta is not None and alpha_t is not None


def test_alpha_t_stat_hand_computed() -> None:
    """R56: y = [2, 1, 4, 3, 5] on x = 1..5.

    mean x = mean y = 3, sxy = 8, sxx = 10 -> beta 0.8, alpha 0.6; residuals
    [0.6, -1.2, 1.0, -0.8, 0.4] -> rss 3.6, s^2 = 3.6 / 3 = 1.2;
    se(alpha)^2 = 1.2 * (1/5 + 3^2/10) = 1.32 -> t = 0.6 / sqrt(1.32).
    """
    beta, alpha_t = research._ols([2.0, 1.0, 4.0, 3.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])

    assert beta == pytest.approx(0.8, rel=1e-12)
    assert alpha_t == pytest.approx(0.6 / math.sqrt(1.32), rel=1e-12)
    assert alpha_t != pytest.approx(0.8 / math.sqrt(1.32), rel=1e-6)  # alpha, not beta


def test_betas_wiring_two_separate_regressions(tmp_path: Path) -> None:
    """R56: ``beta_btc/alpha_t_btc`` regress on BTC alone, ``beta_ew/alpha_t`` on the basket."""
    inputs = _inputs(tmp_path, name="betas")
    weekly_ts = [_START + (k + 1) * _WEEK for k in range(5)]
    btc_returns = [0.01, 0.02, -0.01, 0.03, 0.0]
    eth_returns = [0.0, 0.01, 0.05, -0.02, 0.02]
    portfolio = [0.004, 0.012, -0.003, 0.02, 0.001]  # the strategy's weekly returns

    def prices(returns: list[float]) -> dict[int, float]:
        series: dict[int, float] = {_START: 100.0}
        for ts, move in zip(weekly_ts, returns, strict=True):
            series[ts] = series[ts - _WEEK] * (1.0 + move)
        return series

    shaped = replace(
        inputs,
        boundary_prices={"BTCUSDT": prices(btc_returns), "ETHUSDT": prices(eth_returns)},
    )
    report = CarryReport(
        summary={}, daily=[], weekly=list(zip(weekly_ts, portfolio, strict=True)), legs=[], by_year=[]
    )

    beta_btc, alpha_t_btc, beta_ew, alpha_t = research._betas(shaped, report)

    basket = [(b + e) / 2 for b, e in zip(btc_returns, eth_returns, strict=True)]
    expected_btc = research._ols(portfolio, btc_returns)
    expected_ew = research._ols(portfolio, basket)
    assert expected_btc[1] != pytest.approx(expected_ew[1], rel=1e-3)  # the regressions differ
    assert (beta_btc, alpha_t_btc) == pytest.approx(expected_btc, rel=1e-9)
    assert (beta_ew, alpha_t) == pytest.approx(expected_ew, rel=1e-9)


def test_drop_top3_removes_the_three_best_weeks() -> None:
    """§6 stress: the Sharpe of the remaining weeks, annualised with sqrt(52)."""
    returns = [0.01, -0.02, 0.05, 0.03, 0.04, -0.01, 0.02]
    kept = [-0.02, -0.01, 0.01, 0.02]  # drop 0.05, 0.04, 0.03

    expected = statistics.mean(kept) / statistics.stdev(kept) * math.sqrt(52.0)

    assert research._drop_top3(returns) == pytest.approx(expected, rel=1e-12)
    # the best three are dropped, not the worst three
    assert research._drop_top3(returns) != pytest.approx(
        statistics.mean([0.01, 0.02, 0.03, 0.04]) / statistics.stdev([0.01, 0.02, 0.03, 0.04]) * math.sqrt(52.0),
        rel=1e-3,
    )
    # the edge: 6 weeks leave exactly 3 (the Sharpe minimum); 3 weeks leave none
    assert research._drop_top3([0.01, -0.02, 0.05, 0.03, 0.04, -0.01]) is not None
    assert research._drop_top3([0.01, 0.02, 0.03]) is None
    assert research._drop_top3([0.01, 0.02, 0.03, 0.04]) is None  # 1 left -> R50b null


def test_placebo_ranks_by_trailing_price_return_with_the_right_sign() -> None:
    """§6: a symbol's placebo signal is price(D_k) / price(D_{k-1}) - 1 (positive when it rose)."""
    first_ns = 10 * _WEEK
    second_ns = first_ns + _WEEK

    def series(before: float, after: float) -> tuple[PriceTick, ...]:
        return (PriceTick(first_ns - 1_000_000, before), PriceTick(second_ns - 1_000_000, after))

    ranker = research._PlaceboRanker(
        {"A": series(100.0, 110.0), "B": series(200.0, 180.0), "C": series(10.0, 11.0)}
    )
    signals: dict[str, float | None] = {"A": 0.5, "B": 0.5, "C": None}

    assert ranker(first_ns, signals) == {"A": None, "B": None, "C": None}  # no previous tick yet
    got = ranker(second_ns, signals)
    assert got["A"] == pytest.approx(0.1, abs=1e-12)  # rose 10% -> positive (so it is SHORTED)
    assert got["B"] == pytest.approx(-0.1, abs=1e-12)  # fell 10% -> negative (so it is LONGED)
    assert got["C"] is None  # an excluded symbol stays excluded


def test_dsr_matches_the_gate_helper_and_the_raw_record(tmp_path: Path) -> None:
    """R57: headline = ``mtm_dsr_calculate`` (clamped moments); ``dsr_raw_p_value`` = raw moments."""
    inputs = _inputs(tmp_path, name="dsrpin")
    report = carry_metrics(simulate_reference(inputs), inputs)
    weekly = [move for _, move in report.weekly]
    sharpe = report.summary["weekly_sharpe"]
    assert sharpe is not None and len(weekly) == 4
    _, skew, kurt = daily_sharpe_inputs(weekly)
    assert kurt < 3.0  # the clamp bites in this fixture (kurtosis floors at 1.0)

    result = research.evaluate(inputs, draws=0, seed=0, trials_path=tmp_path / "absent.jsonl")

    expected, reason = mtm_dsr_calculate(
        DeflatedSharpeRatio(), sharpe / math.sqrt(52.0), skew, kurt, 4, 4
    )
    raw = calculate_dsr(
        observed_sharpe=sharpe / math.sqrt(52.0),
        num_trials=4,
        num_observations=4,
        skewness=skew,
        kurtosis=kurt,
    )
    assert reason is None and expected is not None
    assert result["dsr"] == expected.deflated_sharpe
    assert result["dsr_p_value"] == expected.p_value
    assert result["dsr_error"] is None
    assert result["dsr_raw_p_value"] == raw.p_value
    assert raw.p_value != expected.p_value  # raw and gate-clamped moments disagree here
    assert result["dsr_num_trials"] == 4


def test_dsr_fails_closed_without_enough_weeks(tmp_path: Path) -> None:
    """R57: one week -> no weekly Sharpe -> ``dsr`` null with the gate helper's reason."""
    inputs = _inputs(tmp_path, weeks=1, name="dsr1")

    result = research.evaluate(inputs, draws=0, seed=0, trials_path=tmp_path / "absent.jsonl")

    assert result["dsr"] is None
    assert result["dsr_p_value"] is None
    assert result["dsr_raw_p_value"] is None
    assert isinstance(result["dsr_error"], str)
    assert result["dsr_error"].startswith("DSR input missing")


# --- CLI: the evaluate subcommand ------------------------------------------------------


def _patch_inputs(monkeypatch: pytest.MonkeyPatch, inputs: CarryInputs) -> None:
    monkeypatch.setattr(carry_cli, "load_carry_inputs", lambda spec, catalog, archive: inputs)


def _evaluate_args(tmp_path: Path, *extra: str, out: Path | None = None) -> list[str]:
    args = [
        "evaluate",
        "--symbols",
        "BTCUSDT",
        "ETHUSDT",
        "--start",
        "2025-01-06",
        "--end",
        "2025-02-03",
        "--draws",
        "1",
    ]
    if out is not None:
        args += ["--out", str(out)]
    return [*args, *extra]


def test_cli_evaluate_writes_research_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§8-T8/R55: one stdout line with the wall time, ``research.json`` written."""
    inputs = _inputs(tmp_path, name="cli")
    _patch_inputs(monkeypatch, inputs)
    out_dir = tmp_path / "research"

    code = carry_cli.main(
        _evaluate_args(
            tmp_path,
            "--trials",
            str(tmp_path / "trials.jsonl"),
            out=out_dir,
        )
    )

    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert captured.out.startswith("research[reference]:")
    assert "wall=" in captured.out
    assert captured.out.count("\n") == 1

    payload = json.loads((out_dir / "research.json").read_text())
    assert payload["spec"] == inputs.spec.to_json()
    assert payload["spec_hash"] == inputs.spec.spec_hash()
    assert payload["draws"] == 1
    assert payload["placebo_kind"] == "reversal"
    assert payload["dsr_num_trials"] == 4
    assert not (tmp_path / "trials.jsonl").exists()  # evaluate NEVER appends (R38)


def test_cli_evaluate_strategy_ruined_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R36: a ruined strategy exits 2 (never a traceback, never a partial output)."""
    inputs = _inputs(tmp_path, name="ruined")

    def boom(inputs: CarryInputs, **kw: object) -> object:
        raise CarryEquityTooSmall("<book>", inputs.spec.start_ns, 0.0, 0.0)

    _patch_inputs(monkeypatch, inputs)
    monkeypatch.setattr(research, "simulate_reference", boom)
    out_dir = tmp_path / "research"

    code = carry_cli.main(_evaluate_args(tmp_path, "--trials", str(tmp_path / "t.jsonl"), out=out_dir))

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1
    assert "CarryEquityTooSmall" in captured.err
    assert not out_dir.exists()


def _run_args(*extra: str) -> list[str]:
    return [
        "run",
        "--engine",
        "reference",
        "--symbols",
        "BTCUSDT",
        "ETHUSDT",
        "--start",
        "2025-01-06",
        "--end",
        "2025-02-03",
        *extra,
    ]


def _root_at(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Point every default path of the CLI at ``root`` (never the real ``data/portfolio``)."""
    monkeypatch.setattr(carry_cli, "REPO_ROOT", root)
    return root / "data/portfolio"


def test_evaluate_then_run_on_default_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SF2/R58: ``evaluate`` and ``run`` default to different dirs; neither deletes the other's files."""
    inputs = _inputs(tmp_path, name="order1")
    _patch_inputs(monkeypatch, inputs)
    portfolio = _root_at(monkeypatch, tmp_path / "root1")

    assert carry_cli.main(_evaluate_args(tmp_path)) == 0
    assert carry_cli.main(_run_args()) == 0
    assert capsys.readouterr().err == ""

    research_files = list(portfolio.glob("*/research.json"))
    summary_files = list(portfolio.glob("*/summary.json"))
    assert len(research_files) == 1 and len(summary_files) == 1
    assert research_files[0].parent != summary_files[0].parent
    assert json.loads(research_files[0].read_text())["spec_hash"] == inputs.spec.spec_hash()
    assert (summary_files[0].parent / "weekly.parquet").is_file()  # the run's outputs intact
    assert len((portfolio / "trials.jsonl").read_text().splitlines()) == 1  # only `run` appends


def test_run_then_evaluate_on_default_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SF2/R58: the reverse order -- ``evaluate`` after ``run`` must not touch the run dir."""
    inputs = _inputs(tmp_path, name="order2")
    _patch_inputs(monkeypatch, inputs)
    portfolio = _root_at(monkeypatch, tmp_path / "root2")

    assert carry_cli.main(_run_args()) == 0
    run_dir = next(portfolio.glob("*/summary.json")).parent
    before = sorted(path.name for path in run_dir.iterdir())
    assert carry_cli.main(_evaluate_args(tmp_path)) == 0
    assert capsys.readouterr().err == ""

    assert sorted(path.name for path in run_dir.iterdir()) == before  # no research.json inside
    assert not (run_dir / "research.json").exists()
    assert len(list(portfolio.glob("*/research.json"))) == 1
    # the evaluate default dir is named for the research kind, not the reference run's
    assert next(portfolio.glob("*/research.json")).parent.name.startswith("research-")
    assert run_dir.name.startswith("reference-")
    # evaluate read the ledger the run wrote: one recorded variant == the evaluated spec -> N = 4
    payload = json.loads(next(portfolio.glob("*/research.json")).read_text())
    assert payload["dsr_num_trials"] == 4


@pytest.mark.parametrize("draws", ["0", "-1"])
def test_cli_evaluate_rejects_draws_below_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    draws: str,
) -> None:
    """R58: ``--draws < 1`` is a usage error (exit 2) raised before any input load or output."""
    inputs = _inputs(tmp_path, name="draws")
    _patch_inputs(monkeypatch, inputs)
    out_dir = tmp_path / "research"
    args = _evaluate_args(tmp_path, out=out_dir)
    args[args.index("--draws") + 1] = draws

    code = carry_cli.main(args)

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "ValueError" in captured.err and "--draws" in captured.err
    assert not out_dir.exists()


def test_cli_evaluate_research_json_requires_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R58: an existing research.json is refused (exit 2, untouched) unless ``--force``."""
    inputs = _inputs(tmp_path, name="force")
    _patch_inputs(monkeypatch, inputs)
    out_dir = tmp_path / "research"
    out_dir.mkdir()
    existing = out_dir / "research.json"
    existing.write_text("old")

    code = carry_cli.main(_evaluate_args(tmp_path, out=out_dir))

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "exists" in captured.err and "--force" in captured.err
    assert existing.read_text() == "old"

    assert carry_cli.main(_evaluate_args(tmp_path, "--force", "--trials", str(tmp_path / "t.jsonl"), out=out_dir)) == 0
    assert json.loads(existing.read_text())["draws"] == 1
