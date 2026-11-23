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

import itertools
import json
import math
import sqlite3
import statistics
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
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


# --- The hooks default to a byte-identical run ------------------------------------


def test_hooks_default_is_identical(tmp_path: Path) -> None:
    """``simulate_reference(inputs)`` == ``simulate_reference(inputs, ranker=None, fee_multiplier=1.0)``."""
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


def _hole_at_second_decision(inputs: CarryInputs) -> CarryInputs:
    """ETH loses its rows in ``[D1 - 3 d, D1 - 2 d]``: a funding hole at decision 1 only."""
    d1 = inputs.decisions[1]
    eth = inputs.settlements["ETHUSDT"]
    keep = [
        index
        for index, ts in enumerate(eth.times_ns)
        if not d1 - 3 * _DAY <= ts <= d1 - 2 * _DAY
    ]
    holed = Settlements(
        times_ns=tuple(eth.times_ns[i] for i in keep), rates=tuple(eth.rates[i] for i in keep)
    )
    return replace(inputs, settlements={**inputs.settlements, "ETHUSDT": holed})


def test_ranker_keeps_exclusions(tmp_path: Path) -> None:
    """R8: a ranker that signals an excluded symbol fails closed; built-in rankers comply."""
    inputs = _hole_at_second_decision(_inputs(tmp_path, name="ranker"))

    def leaky(decision_ns: int, signals: Mapping[str, float | None]) -> dict[str, float | None]:
        return {symbol: 1.0 for symbol in signals}

    with pytest.raises(ValueError, match=r"^ranker returned a signal for excluded ETHUSDT$"):
        simulate_reference(inputs, ranker=leaky)

    for ranker in (
        research.PersistenceMatchedRanker(0.5, 11),
        research._PlaceboRanker(inputs.ticks),
    ):
        run = simulate_reference(inputs, ranker=ranker)
        assert run.decisions[1].excluded == {"ETHUSDT": "funding_hole"}
        assert run.decisions[1].signals["ETHUSDT"] is None
        assert all(
            decision.signals[symbol] is None
            for decision in run.decisions
            for symbol in decision.excluded
        )


# --- Null accounting --------------------------------------------------------------


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
            summary={
                "weekly_sharpe": run.sharpe,
                "weeks": 4,
                "total_return": 0.0,
                "max_drawdown_daily": 0.0,
            },
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
    """A ruin the weekly seam reports is ``-inf`` and counted, never dropped."""
    inputs = _inputs(tmp_path, name="weeklyruin")
    ruin = float("-inf")  # the harness's marker: this draw's ruin comes from the weekly seam
    harness = _install_fakes(monkeypatch, inputs, [0.5, ruin, 2.0], [1.6, ruin, 1.5])

    result = research.evaluate(inputs, draws=3, seed=_SEED0, trials_path=tmp_path / "trials.jsonl")

    assert harness.weekly_runs == [0.5, 2.0, 1.6, 1.5]  # the ruined draw reports no value
    assert result["ruined_draws"] == 1  # the pre-cost pass's ruin is not double counted
    assert result["p_value"] == 0.5  # only the 2.0 draw reaches the 1.0 strategy: (1 + 1) / 4
    assert result["p_value_pre_cost"] == 0.75  # 1.6 and the equal 1.5 beat the 1.5 strategy


def test_ruined_draw_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ruined draw is ``-inf``, counted, and p = (1 + #{null >= strat}) / (1 + draws)."""
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
    """A ``CarryDataError`` in a later null draw (k>0) propagates -- never skipped.

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


# --- Persistence ----------------------------------------------------------------


def test_persistence_from_strategy_picks() -> None:
    """Picks [A/B, A/B, C/B, C/B] -> 2/3; a flat decision breaks the chain."""
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
    """The null ranker is seeded (reproducible) and honours ``p_keep``."""
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


# --- Capped funding --------------------------------------------------------------


def test_cap_applies_to_signal_and_paid(tmp_path: Path) -> None:
    """A ±0.1% cap hits the signal and paid funding; the module cache is untouched."""
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


# --- The DSR trial count ----------------------------------------------------


def test_dsr_trial_count(tmp_path: Path) -> None:
    """N = 3 prior + |recorded spec hashes u {evaluated}|."""
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
    assert result["placebo_kind"] == "reversal"  # Longs the LOWEST trailing return
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
    """Y = [2, 1, 4, 3, 5] on x = 1..5.

    mean x = mean y = 3, sxy = 8, sxx = 10 -> beta 0.8, alpha 0.6; residuals
    [0.6, -1.2, 1.0, -0.8, 0.4] -> rss 3.6, s^2 = 3.6 / 3 = 1.2;
    se(alpha)^2 = 1.2 * (1/5 + 3^2/10) = 1.32 -> t = 0.6 / sqrt(1.32).
    """
    beta, alpha_t = research._ols([2.0, 1.0, 4.0, 3.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])

    assert beta == pytest.approx(0.8, rel=1e-12)
    assert alpha_t == pytest.approx(0.6 / math.sqrt(1.32), rel=1e-12)
    assert alpha_t != pytest.approx(0.8 / math.sqrt(1.32), rel=1e-6)  # alpha, not beta


def test_betas_wiring_two_separate_regressions(tmp_path: Path) -> None:
    """``beta_btc/alpha_t_btc`` regress on BTC alone, ``beta_ew/alpha_t`` on the basket."""
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
    """Headline = ``mtm_dsr_calculate`` (clamped moments); ``dsr_raw_p_value`` = raw moments."""
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
    """One week -> no weekly Sharpe -> ``dsr`` null with the gate helper's reason."""
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
    # evaluate records its own spec once (chief ruling 2026-11-19), before its DSR read
    records = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert [(r["spec_hash"], r["outcome"], r["engine"]) for r in records] == [
        (inputs.spec.spec_hash(), "completed", "reference")
    ]


def test_cli_evaluate_strategy_ruined_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ruined strategy exits 2 (never a traceback, never a partial output)."""
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
    """``evaluate`` and ``run`` default to different dirs; neither deletes the other's files."""
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
    # evaluate's own completion, then the run's history row (same hash: still one trial)
    lines = [json.loads(line) for line in (portfolio / "trials.jsonl").read_text().splitlines()]
    assert [(r["engine"], r["outcome"]) for r in lines] == [
        ("reference", "completed"),
        ("reference", "completed"),
    ]
    assert {r["spec_hash"] for r in lines} == {inputs.spec.spec_hash()}


def test_run_then_evaluate_on_default_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reverse order -- ``evaluate`` after ``run`` must not touch the run dir."""
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
    """``--draws < 1`` is a usage error (exit 2) raised before any input load or output."""
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
    """An existing research.json is refused (exit 2, untouched) unless ``--force``."""
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


# --- The DSR trial ledger: evaluate ensures its own trial (x0axf.14 T3) -----------------


def _ledger(path: Path) -> list[tuple[str, str]]:
    return [
        (record["spec_hash"], record["outcome"])
        for record in (json.loads(line) for line in path.read_text().splitlines())
    ]


def _patch_spec_inputs(monkeypatch: pytest.MonkeyPatch, inputs: CarryInputs) -> None:
    """The CLI's spec (band from argv) on the fixture's data."""
    monkeypatch.setattr(
        carry_cli, "load_carry_inputs", lambda spec, catalog, archive: replace(inputs, spec=spec)
    )


def test_evaluate_ensures_own_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R13: A, B, A on the default ledger -> N 4, 5, 5; two lines; the real ledger untouched."""
    real = carry_cli.REPO_ROOT / "data/portfolio/trials.jsonl"
    real_before = real.read_bytes() if real.exists() else None
    inputs = _inputs(tmp_path, name="r13")
    _patch_spec_inputs(monkeypatch, inputs)
    root = tmp_path / "portfolio"
    monkeypatch.setattr(carry_cli, "DEFAULT_OUT_ROOT", str(root))
    hash_a = inputs.spec.spec_hash()
    hash_b = replace(inputs.spec, band=0.2).spec_hash()

    def num_trials(*extra: str) -> int:
        assert carry_cli.main(_evaluate_args(tmp_path, *extra)) == 0
        out = capsys.readouterr()
        assert out.err == ""
        research_dir = Path(out.out.split(" out=")[1].split(" wall=")[0])
        return int(json.loads((research_dir / "research.json").read_text())["dsr_num_trials"])

    assert num_trials() == 4
    assert num_trials("--band", "0.2") == 5
    assert num_trials("--force") == 5
    assert num_trials("--out", str(tmp_path / "fresh")) == 5
    assert _ledger(root / "trials.jsonl") == [(hash_a, "completed"), (hash_b, "completed")]
    assert (real.read_bytes() if real.exists() else None) == real_before


def _flaky_simulate(inputs: CarryInputs, fail: str) -> Callable[..., object]:
    """The real reference run, except the ``fail`` kind raises."""
    real = simulate_reference

    def simulate(
        run_inputs: CarryInputs, *, ranker: object = None, fee_multiplier: float = 1.0
    ) -> object:
        if isinstance(ranker, research.PersistenceMatchedRanker):
            kind = "draw"
        elif ranker is not None:
            kind = "placebo"
        elif run_inputs is not inputs:
            kind = "capped"
        else:
            kind = {1.0: "own", 0.0: "precost", 2.0: "fee2x"}[fee_multiplier]
        if kind == fail == "draw":
            raise CarryDataError("no price for held BTCUSDT in a null draw")
        if kind == fail:
            raise CarryEquityTooSmall("<book>", inputs.decisions[2], 1.0, 2.0)
        return real(run_inputs, ranker=ranker, fee_multiplier=fee_multiplier)  # type: ignore[arg-type]

    return simulate


@pytest.mark.parametrize("fail", ["precost", "placebo", "fee2x", "capped", "draw"])
def test_evaluate_keeps_own_trial_on_later_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: str
) -> None:
    """R13a: the own completion lands right after its metrics; no auxiliary run registers."""
    inputs = _inputs(tmp_path, name=f"r13a{fail}")
    ledger = tmp_path / "trials.jsonl"
    monkeypatch.setattr(research, "simulate_reference", _flaky_simulate(inputs, fail))

    with pytest.raises((CarryEquityTooSmall, CarryDataError)):
        research.evaluate(inputs, draws=2, seed=0, trials_path=ledger)

    own = inputs.spec.spec_hash()
    assert _ledger(ledger) == [(own, "completed")]
    record = json.loads(ledger.read_text())
    assert math.isfinite(record["weekly_sharpe"])
    before = ledger.read_bytes()

    monkeypatch.setattr(research, "simulate_reference", simulate_reference)
    result = research.evaluate(inputs, draws=2, seed=0, trials_path=ledger)
    assert result["dsr_num_trials"] == 4
    assert ledger.read_bytes() == before  # the rerun found the recorded completion


def test_same_hash_one_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R14: run then evaluate appends nothing more; two runs = two lines, ONE trial."""
    inputs = _inputs(tmp_path, name="r14")
    _patch_inputs(monkeypatch, inputs)
    ledger = tmp_path / "trials.jsonl"
    trials_args = ("--trials", str(ledger))

    assert carry_cli.main(_run_args("--out", str(tmp_path / "run1"), *trials_args)) == 0
    assert len(ledger.read_text().splitlines()) == 1
    out_dir = tmp_path / "research"
    assert carry_cli.main(_evaluate_args(tmp_path, *trials_args, out=out_dir)) == 0
    assert len(ledger.read_text().splitlines()) == 1  # evaluate found the run's completion
    assert carry_cli.main(_run_args("--out", str(tmp_path / "run2"), *trials_args)) == 0
    assert _ledger(ledger) == [(inputs.spec.spec_hash(), "completed")] * 2
    assert carry_cli.main(_evaluate_args(tmp_path, "--force", *trials_args, out=out_dir)) == 0
    assert capsys.readouterr().err == ""
    assert json.loads((out_dir / "research.json").read_text())["dsr_num_trials"] == 4
    assert len(ledger.read_text().splitlines()) == 2


def test_liquidated_draw_is_ruined(tmp_path: Path) -> None:
    """R43: a cross bust inside a null draw is a ruined draw (-inf, counted)."""
    base = _inputs(tmp_path, name="liqdraw")
    spike = base.decisions[0] + _DAY
    marks = {}
    for symbol, series in base.marks.items():
        index = int(np.searchsorted(series.ts, spike))
        low, high = series.low.copy(), series.high.copy()
        low[index], high[index] = 1.0, _SIZING[symbol] * 10.0  # adverse for either side
        marks[symbol] = series._replace(low=low, high=high)
    spiked = replace(base, marks=marks)
    cross = replace(spiked, spec=replace(base.spec, liquidation_mode="cross"))

    sharpes, ruined = research._draw_sharpes(cross, 0.5, _SEED0, 2, 1.0)
    assert (sharpes, ruined) == ([float("-inf"), float("-inf")], 2)
    sharpes, ruined = research._draw_sharpes(spiked, 0.5, _SEED0, 2, 1.0)
    assert ruined == 0
    assert all(math.isfinite(s) or s is None for s in sharpes)


# ---------------------------------------------------------------------------
# arch-c9-2 trend evaluator (T5): R55-R63; dreview-b2-2 C9-6 (joint rotation null on
# each coin's own eligible mask), C9-7 (frozen family N, order independence), C9-8
# (the run's evidence block in research.json), C9-17 (NC1 signal-only lag)
# ---------------------------------------------------------------------------

_T_PAIR = ("BTCUSDT", "ETHUSDT")


def _climb(first_ms: int, last_ms: int, base: float) -> list[tuple[float | int, ...]]:
    """Hourly closes rising 0.01 per hour: every trend term of the coin is +1."""
    from tests.unit.test_carry_bars import HOUR_MS, kline

    return [
        kline(open_ms, round(base + 0.01 * index, 2))
        for index, open_ms in enumerate(range(first_ms, last_ms, HOUR_MS))
    ]


def _trend_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    direction: str = "long_short",
    trend_filter: str | None = None,
    vol_target: float = 0.4,
    start: str = "2024-03-01",
    end: str = "2024-03-20",
    name: str = "trend.db",
) -> CarryInputs:
    """A 1h trend run on a tmp archive: BTC a 500-hour sawtooth (its signal and the
    BTC regime change sign), ETH climbing; 8h funding throughout."""
    from tests.unit.test_carry_bars import hourly, ms, use_snapshot, write_archive
    from vibe_quant.portfolio.ticks import load_carry_inputs

    first, last = ms("2023-09-01T00:00:00"), ms("2024-03-25T00:00:00")
    archive = tmp_path / name
    if not archive.exists():
        write_archive(
            archive,
            {"BTCUSDT": hourly(first, last, 100.0), "ETHUSDT": _climb(first, last, 50.0)},
            dict.fromkeys(_T_PAIR, first),
            funding_to_ms=last,
        )
    use_snapshot(tmp_path, monkeypatch, archive)
    spec = CarrySpec.from_dates(
        _T_PAIR,
        start,
        end,
        signal="trend",
        schedule="daily",
        weighting="time_series",
        direction=direction,
        trend_filter=trend_filter,
        vol_target=vol_target,
        fill_granularity="1h",
        spread=0.0001,
    )
    clear_rate_cache()
    return load_carry_inputs(spec, tmp_path / "no-catalog", archive)


def _family_file(path: Path, n_prior: object, arms: object) -> Path:
    path.write_text(json.dumps({"n_prior": n_prior, "arms": arms}))
    return path


def _table(rows: list[dict[str, float | None]], regime: list[float | None] | None = None):
    decisions = tuple(_START + _HOUR + index * _DAY for index in range(len(rows)))
    return research.SignalTable(
        decisions=decisions,
        signals=tuple(dict(row) for row in rows),
        regime=tuple(regime if regime is not None else [None] * len(rows)),
    )


def _column(table: research.SignalTable, symbol: str) -> list[float | None]:
    return [row.get(symbol) for row in table.signals]


def test_rotation_null() -> None:
    """R57 + C9-6 (R-c9t5-1): one u per draw; every segment of decisions with the same
    eligible set (and regime presence) shifts all its rows together by
    ``k = 1 + floor(u (n - 1))``, so each coin stays inside its own mask, None stays
    None and two coins keep their co-movement on the overlap."""
    assert research.rotate(["v0", "v1", "v2", "v3", "v4"], 0.5) == ["v3", "v4", "v0", "v1", "v2"]
    assert research.rotate(["v0", "v1", "v2", "v3", "v4"], 0.0) == ["v1", "v2", "v3", "v4", "v0"]
    assert research.rotate(["only"], 0.99) == ["only"]

    # a coin eligible only at decisions 50..99 rotates within its own 50
    late = [None] * 50 + [float(i) for i in range(50, 100)]
    early = [float(-i) for i in range(100)]
    table = _table([{"A": a, "B": b} for a, b in zip(early, late, strict=True)])
    rotated, degenerate = research.rotate_table(table, 0.5)
    b_path = _column(rotated, "B")
    assert b_path[:50] == [None] * 50
    assert b_path[50:] == late[50 + 25 :] + late[50 : 50 + 25]  # k = 1 + floor(0.5 * 49) = 25
    assert sorted(_column(rotated, "A")[:50]) == sorted(early[:50])  # A's first segment
    assert degenerate == 0

    # two identical paths and masks rotate identically
    same = _table([{"A": float(i % 4), "B": float(i % 4)} for i in range(9)])
    twin, _ = research.rotate_table(same, 0.37)
    assert _column(twin, "A") == _column(twin, "B")
    assert _column(twin, "A") != _column(same, "A")

    # a one-decision segment is the identity and is counted
    lone = _table([{"A": 1.0, "B": None}, {"A": 0.5, "B": 0.2}, {"A": 0.1, "B": 0.3}])
    lone_rotated, lone_degenerate = research.rotate_table(lone, 0.9)
    assert lone_rotated.signals[0] == {"A": 1.0, "B": None}
    assert lone_degenerate == 1


def test_rotation_keeps_staggered_listing_co_movement() -> None:
    """C9-6 reviewer probe: B lists at decision 4; on the overlap A and B agree 4/4 and
    are both long on 2 days. A per-coin rotation drops that to 0/4 and 0; the joint
    segment rotation keeps 4/4 and 2, for both probe fractions."""
    first = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    listed_late: list[float | None] = [None] * 4 + first[4:]
    table = _table([{"A": a, "B": b} for a, b in zip(first, listed_late, strict=True)])

    def agreement(a: list[float | None], b: list[float | None]) -> tuple[int, int]:
        pairs = [(x, y) for x, y in zip(a, b, strict=True) if y is not None]
        return sum(x == y for x, y in pairs), sum(x == y == 1.0 for x, y in pairs)

    assert agreement(first, listed_late) == (4, 2)
    for u in (0.3, 0.5):
        rotated, _ = research.rotate_table(table, u)
        a, b = _column(rotated, "A"), _column(rotated, "B")
        assert agreement(a, b) == (4, 2)
        assert [value is None for value in b] == [value is None for value in listed_late]
        assert sorted(x for x in a[:4] if x is not None) == sorted(first[:4])


def test_rotation_moves_the_regime_with_its_segment() -> None:
    """C9-6: the filter regime is one more column of the segment (None stays None)."""
    table = _table(
        [{"A": float(i)} for i in range(5)], regime=[None, 0.2, -0.2, 0.6, 1.0]
    )
    rotated, degenerate = research.rotate_table(table, 0.5)
    # segments: [0] (regime None, degenerate), [1..4] (k = 1 + floor(0.5 * 3) = 2)
    assert rotated.regime == (None, 0.6, 1.0, 0.2, -0.2)
    assert _column(rotated, "A") == [0.0, 3.0, 4.0, 1.0, 2.0]
    assert degenerate == 1


def test_rotation_null_draws_are_seeded() -> None:
    """R57: draw i uses ``Random(seed + i).random()``; seed 0 twice gives the same draws."""
    import random

    assert research.draw_fraction(0, 3) == random.Random(3).random()
    assert research.draw_fraction(7, 0) == research.draw_fraction(7, 0)
    assert research.draw_fraction(0, 1) != research.draw_fraction(0, 2)


def test_cross_wired_placebo_keeps_the_true_regime() -> None:
    """R60: each sorted eligible symbol takes the NEXT one's signal (cyclic); the
    regime is untouched; a lone eligible symbol keeps its own."""
    table = _table(
        [{"A": 0.2, "B": -0.4, "C": 1.0}, {"A": None, "B": 0.6, "C": 0.8}, {"A": 0.4}],
        regime=[0.2, -0.6, None],
    )
    wired = research.cross_wire(table)
    assert wired.signals == (
        {"A": -0.4, "B": 1.0, "C": 0.2},
        {"A": None, "B": 0.8, "C": 0.6},
        {"A": 0.4},
    )
    assert wired.regime == table.regime


def test_null_percentile_and_p_value_edges() -> None:
    """R58: ``p = (1 + #{null >= own}) / (1 + draws)``; ``percentile = #{null < own} /
    draws``; a tie counts for the null in both; a ruined draw (-inf) is below any own."""
    nulls = [0.5, 1.0, float("-inf"), 2.0]
    assert research._p_value(1.0, nulls) == 3 / 5
    assert research.null_percentile(1.0, nulls) == 2 / 4
    assert research.null_percentile(2.0, nulls) == 3 / 4
    assert research.null_percentile(None, nulls) is None
    assert research.null_percentile(1.0, []) is None


def test_filter_zeroing_equals_the_native_filter_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-c9t5-1 Q2: a draw runs the spec without its filter and the ranker zeroes every
    eligible signal while the (rotated) regime is None or <= 0. With the TRUE table that
    is the native filter run bit for bit: equities, weights, targets, orders, fills and
    funding (only the book_state label differs: flat vs filter_off)."""
    inputs = _trend_inputs(
        tmp_path, monkeypatch, direction="long_flat", trend_filter="BTCUSDT"
    )
    native = simulate_reference(inputs)
    table, mismatches = research.signal_table(inputs)
    zeroed = research.table_run(inputs, table)

    states = {decision.book_state for decision in native.decisions}
    assert "filter_off" in states and None in states  # the fixture exercises both regimes
    assert mismatches == 0
    assert zeroed.spec.trend_filter is None
    assert zeroed.fills == native.fills
    assert zeroed.payments == native.payments
    for mine, theirs in zip(zeroed.decisions, native.decisions, strict=True):
        assert (mine.equity, mine.weights, mine.target_steps, mine.orders) == (
            theirs.equity,
            theirs.weights,
            theirs.target_steps,
            theirs.orders,
        )
        assert mine.book_state == ("flat" if theirs.book_state == "filter_off" else theirs.book_state)


def test_signal_table_lags_only_the_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C9-17: a K-day lag reads ``trend_signal(closes, D - K d)`` for the coins eligible
    at D (and the regime the same way); the eligibility, prices and book-vol closes are
    those of D. A lagged window without history is counted as a mask mismatch."""
    from vibe_quant.portfolio.signal import trend_signal

    base = _trend_inputs(tmp_path, monkeypatch, direction="long_flat", trend_filter="BTCUSDT")
    longer = _trend_inputs(
        tmp_path, monkeypatch, direction="long_flat", trend_filter="BTCUSDT", start="2024-02-25"
    )
    assert longer.spec == research.lag_spec(base.spec, 5)
    assert base.trend_closes is not None and longer.trend_closes is not None
    inputs = replace(
        base, trend_closes=research.extend_closes(base.trend_closes, longer.trend_closes)
    )
    native, _ = research.signal_table(inputs)
    lagged, mismatches = research.signal_table(inputs, lag_days=5)
    closes = inputs.trend_closes
    assert closes is not None
    for index, decision_ns in enumerate(lagged.decisions):
        for symbol, value in native.signals[index].items():
            expected = None if value is None else trend_signal(closes[symbol], decision_ns - 5 * _DAY)
            assert lagged.signals[index][symbol] == expected
        assert lagged.regime[index] == trend_signal(closes["BTCUSDT"], decision_ns - 5 * _DAY)
    assert mismatches == 0
    # without the longer load the closes start at start - 160 d: decisions 0..3 lag past them
    _, short = research.signal_table(base, lag_days=5)
    assert short == 4


def test_extended_closes_must_agree_on_shared_days() -> None:
    """C9-17: the lag's earlier closes come from a longer load; on every day both loads
    hold, the values must be identical (else the price/vol inputs would move)."""
    base = {"BTCUSDT": {_START: 100.0, _START + _DAY: 101.0}}
    longer = {"BTCUSDT": {_START - _DAY: 99.0, _START: 100.0, _START + _DAY: 101.0}}
    assert research.extend_closes(base, longer) == longer
    with pytest.raises(ValueError, match="BTCUSDT"):
        research.extend_closes(base, {"BTCUSDT": {_START: 100.5, _START + _DAY: 101.0}})
    with pytest.raises(ValueError, match="BTCUSDT"):
        research.extend_closes(base, {"BTCUSDT": {_START - _DAY: 99.0}})


def test_nc1_lag_set_and_verdict() -> None:
    """C9-17 / R-c9-8: 50 lags 365..708 step 7; FAIL iff X = #{percentile >= 0.95} >= 6
    (an undefined percentile counts, so a broken null never certifies)."""
    assert tuple(range(365, 709, 7)) == research.NC1_LAGS_DAYS
    assert len(research.NC1_LAGS_DAYS) == 50
    assert research.NC1_LAGS_DAYS[-1] == 708
    five = [0.95] * 5 + [0.9499] * 45
    assert research.nc1_verdict(five) == {"lags": 50, "flagged": 5, "fail_at": 6, "verdict": "PASS"}
    six = [0.95] * 5 + [None] + [0.1] * 44
    assert research.nc1_verdict(six) == {"lags": 50, "flagged": 6, "fail_at": 6, "verdict": "FAIL"}
    with pytest.raises(ValueError, match="50"):
        research.nc1_verdict([0.5] * 49)


def test_nc1_warmup_reaches_back_to_2021_01_14() -> None:
    """C9-17: the 708-day arm on a 2023-06-01 window reads closes from 2021-01-14."""
    spec = CarrySpec.from_dates(
        ("BTCUSDT",),
        "2023-06-01",
        "2026-09-28",
        signal="trend",
        schedule="daily",
        weighting="time_series",
        direction="long_flat",
        vol_target=0.4,
    )
    longer = research.lag_spec(spec, 708)
    assert longer.start_ns == _ns("2023-06-01") - 708 * _DAY
    assert longer.start_ns - 160 * _DAY == _ns("2021-01-14")
    assert replace(longer, start_ns=spec.start_ns) == spec


def test_family_file_validation(tmp_path: Path) -> None:
    """R55 / C9-7: ``{n_prior, arms}``; n_prior an int >= 0 (bool refused), arms a
    non-empty list of distinct sha256 hex strings, N inside the DSR quantile domain."""
    arm = "a" * 64
    good = research.load_family(_family_file(tmp_path / "f.json", 4, [arm, "b" * 64]))
    assert (good.n_prior, good.arms, good.num_trials) == (4, (arm, "b" * 64), 6)
    raw = (tmp_path / "f.json").read_bytes()
    import hashlib

    assert good.to_json() == {
        "path": str(tmp_path / "f.json"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_prior": 4,
        "arms": [arm, "b" * 64],
    }
    for n_prior, arms, match in (
        (-1, [arm], "n_prior"),
        (True, [arm], "n_prior"),
        ("3", [arm], "n_prior"),
        (3, [], "arms"),
        (3, [arm, arm], "duplicate"),
        (3, ["xyz"], "sha256"),
        (3, "a" * 64, "arms"),
        (10**400, [arm], "DSR quantile"),
    ):
        with pytest.raises(ValueError, match=match):
            research.load_family(_family_file(tmp_path / "bad.json", n_prior, arms))
    with pytest.raises(ValueError, match="family"):
        research.load_family(tmp_path / "absent.json")
    (tmp_path / "junk.json").write_text("{not json")
    with pytest.raises(ValueError, match="family"):
        research.load_family(tmp_path / "junk.json")


def test_family_arm_identity_hashes_every_signal_param(tmp_path: Path) -> None:
    """C9-7: the arm identity moves with the vol lookback and every economic input and
    not with the window, cohort, E0 or tier; an arm outside the family is refused."""
    base = dict(
        signal="trend",
        schedule="daily",
        weighting="time_series",
        direction="long_flat",
        vol_target=0.4,
        spread=0.0001,
    )
    spec = CarrySpec.from_dates(("BTCUSDT", "ETHUSDT"), "2024-03-01", "2024-03-20", **base)
    family = research.load_family(_family_file(tmp_path / "f.json", 4, [spec.arm_hash()]))
    research.require_arm(family, spec)
    research.require_arm(
        family,
        CarrySpec.from_dates(
            ("SOLUSDT",), "2022-01-01", "2023-01-01", initial_balance=5.0, fill_granularity="1h", **base
        ),
    )
    for change in (
        {"vol_lookback_days": 31},
        {"direction": "long_short"},
        {"trend_filter": "BTCUSDT"},
        {"vol_target": 0.41},
        {"spread": 0.0002},
        {"gross": 0.9},
        {"band": 0.2},
    ):
        other = CarrySpec.from_dates(
            ("BTCUSDT", "ETHUSDT"), "2024-03-01", "2024-03-20", **{**base, **change}
        )
        with pytest.raises(ValueError, match=f"arm {other.arm_hash()} not in frozen family"):
            research.require_arm(family, other)


def test_family_order_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R56: three arms evaluated a1, a2, a3 on one scratch ledger and a3, a2, a1 on
    another give each arm the same N (= n_prior + 3) and the same DSR p-value, with the
    theoretical variance; the ledger still records each own trial once."""
    arms = [
        _trend_inputs(tmp_path, monkeypatch, vol_target=target, name="order.db")
        for target in (0.3, 0.4, 0.5)
    ]
    family = research.load_family(
        _family_file(tmp_path / "f.json", 4, [inputs.spec.arm_hash() for inputs in arms])
    )

    def sweep(order: list[int], ledger: Path) -> dict[int, tuple[object, ...]]:
        results = {}
        for index in order:
            clear_rate_cache()
            result = research.evaluate(
                arms[index], draws=0, seed=0, trials_path=ledger, family=family
            )
            results[index] = (
                result["dsr_num_trials"],
                result["dsr_p_value"],
                result["dsr"],
                result["dsr_variance"],
            )
        return results

    forward = sweep([0, 1, 2], tmp_path / "forward.jsonl")
    backward = sweep([2, 1, 0], tmp_path / "backward.jsonl")
    assert forward == backward
    assert {row[0] for row in forward.values()} == {7}
    assert {row[3] for row in forward.values()} == {"theoretical"}
    assert all(row[1] is not None for row in forward.values())
    records = read_lines(tmp_path / "forward.jsonl")
    assert sorted(r["spec_hash"] for r in records) == sorted(i.spec.spec_hash() for i in arms)


def read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_trend_evaluate_needs_a_family_with_its_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R55: no family, or a family without this arm, is refused before any run."""
    inputs = _trend_inputs(tmp_path, monkeypatch)
    ledger = tmp_path / "t.jsonl"
    with pytest.raises(ValueError, match="frozen family"):
        research.evaluate(inputs, draws=0, seed=0, trials_path=ledger)
    other = research.load_family(_family_file(tmp_path / "f.json", 3, ["c" * 64]))
    with pytest.raises(ValueError, match=f"arm {inputs.spec.arm_hash()} not in frozen family"):
        research.evaluate(inputs, draws=0, seed=0, trials_path=ledger, family=other)
    assert not ledger.exists()


def test_criteria(tmp_path: Path) -> None:
    """R62: per criterion ``{value, op, bar, pass}``; an undefined value (None/nan)
    fails; PASS needs every criterion and the 1m tier; 1h gives ``research_only`` with
    ``criteria_all_pass``; no criteria never PASS; a bad file is a ValueError."""
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps(
            {
                "sharpe": {"op": ">=", "bar": 1.0},
                "null_percentile": {"op": ">=", "bar": 0.95},
                "max_drawdown_daily": {"op": "<=", "bar": 0.3},
                "dsr_p_value": {"op": "<", "bar": 0.05},
                "coins_with_data": {"op": ">", "bar": 3},
            }
        )
    )
    criteria = research.load_criteria(path)
    edge = {
        "sharpe": 1.0,
        "null_percentile": 0.95,
        "max_drawdown_daily": 0.3,
        "dsr_p_value": 0.0499,
        "coins_with_data": 4,
    }
    judged = research.judge(criteria, edge, "1m")
    assert judged["verdict"] == "PASS"
    assert judged["criteria_all_pass"] is True
    assert judged["criteria"]["sharpe"] == {"value": 1.0, "op": ">=", "bar": 1.0, "pass": True}
    for metric, value in (
        ("sharpe", 0.9999999),
        ("max_drawdown_daily", 0.3000001),
        ("dsr_p_value", 0.05),
        ("coins_with_data", 3),
        ("sharpe", None),
        ("null_percentile", float("nan")),
    ):
        failed = research.judge(criteria, {**edge, metric: value}, "1m")
        assert failed["verdict"] == "FAIL"
        assert failed["criteria"][metric]["pass"] is False
        assert failed["criteria_all_pass"] is False
    hour = research.judge(criteria, edge, "1h")
    assert (hour["verdict"], hour["criteria_all_pass"]) == ("research_only", True)
    # N2: an NC1 (lagged) run is a calibration, never PASS, on either tier
    for tier in ("1m", "1h"):
        calibration = research.judge(criteria, edge, tier, calibration=True)
        assert (calibration["verdict"], calibration["criteria_all_pass"]) == ("calibration", True)
        assert research.judge(None, edge, tier, calibration=True)["verdict"] == "calibration"
    assert research.judge(None, edge, "1m") == {
        "criteria": None,
        "criteria_all_pass": None,
        "verdict": "no_criteria",
    }
    assert research.judge(None, edge, "1h")["verdict"] == "research_only"
    for bad in (
        {},
        {"sharpe_weekly": {"op": ">=", "bar": 1.0}},
        {"sharpe": {"op": "=>", "bar": 1.0}},
        {"sharpe": {"op": ">=", "bar": float("nan")}},
        {"sharpe": {"op": ">=", "bar": True}},
        {"sharpe": {"op": ">=", "bar": 1.0, "extra": 1}},
        [],
    ):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="criteria"):
            research.load_criteria(path)


def test_exposure_stats_hand_computed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C9-6: own and null runs report the same exposure block from the executed book:
    post-plan gross/net/active legs at each decision (sizing prices, decision equity),
    the regime's duty cycle, annual turnover and the signal repeat rate."""
    inputs = _trend_inputs(tmp_path, monkeypatch, direction="long_flat", trend_filter="BTCUSDT")
    run = simulate_reference(inputs)
    table, _ = research.signal_table(inputs)
    stats = research.exposure_stats(run, inputs, table)

    held: dict[str, int] = dict.fromkeys(inputs.spec.symbols, 0)
    gross, net, active = [], [], []
    for decision in run.decisions:
        for order in decision.orders:
            held[order.symbol] += order.steps
        values = [
            float(Decimal(steps) * inputs.specs[symbol].step) * decision.prices[symbol]
            for symbol, steps in held.items()
            if steps
        ]
        gross.append(math.fsum(abs(v) for v in values) / decision.equity)
        net.append(math.fsum(values) / decision.equity)
        active.append(len(values))
    report = carry_metrics(run, inputs)
    on = [value is not None and value > 0.0 for value in table.regime]
    assert stats == {
        "mean_gross": math.fsum(gross) / len(gross),
        "mean_net": math.fsum(net) / len(net),
        "mean_active": math.fsum(active) / len(active),
        "filter_duty": sum(on) / len(on),
        "turnover_annual": report.summary["turnover_annual"],
        "signal_repeat": stats["signal_repeat"],
    }
    assert 0.0 < stats["filter_duty"] < 1.0
    assert 0.0 < stats["mean_gross"] <= 1.0
    pairs = [
        (before[s], after[s])
        for before, after in itertools.pairwise(table.signals)
        for s in before
        if before[s] is not None and after.get(s) is not None
    ]
    assert stats["signal_repeat"] == sum(a == b for a, b in pairs) / len(pairs)
    no_filter = research.exposure_stats(
        run, replace(inputs, spec=replace(inputs.spec, trend_filter=None)), table
    )
    assert no_filter["filter_duty"] is None


def test_trend_research_json_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R61 / C9-8: the trend result carries the daily statistic everywhere, the null
    (p, percentile, pre-cost p, counts), exposure own vs null mean, daily betas, the
    stresses, the family DSR and the run's own evidence block (provenance)."""
    from vibe_quant.portfolio.metrics import provenance

    inputs = _trend_inputs(tmp_path, monkeypatch)
    family = research.load_family(_family_file(tmp_path / "f.json", 4, [inputs.spec.arm_hash()]))
    result = research.evaluate(
        inputs,
        draws=3,
        seed=5,
        trials_path=tmp_path / "t.jsonl",
        family=family,
        spread_source="DEFAULT_SPREAD (spread table not landed)",
    )
    own = simulate_reference(inputs)
    report = carry_metrics(own, inputs)
    evidence = provenance(
        own, inputs, spread_source="DEFAULT_SPREAD (spread table not landed)", universe=None
    )
    assert result["statistic"] == "daily_sharpe_365"
    assert result["sharpe"] == report.summary["daily_sharpe_365"]
    assert result["total_return"] == report.summary["total_return"]
    assert result["max_drawdown_daily"] == report.summary["max_drawdown_daily"]
    for key, value in evidence.items():
        assert result[key] == value, key
    assert result["family"] == family.to_json()
    assert (result["draws"], result["seed"]) == (3, 5)
    assert result["dsr_num_trials"] == 5
    assert result["dsr_variance"] == "theoretical"
    assert result["placebo_kind"] == "cross_wired"
    assert result["signal_lag_days"] == 0
    assert result["verdict"] == "research_only"
    assert result["criteria"] is None
    assert result["ledger_num_trials"] == 4  # trend records only (R-c9t4-2)
    assert set(result["exposure"]) == {"own", "null_mean"}
    assert result["exposure"]["own"] == research.exposure_stats(
        own, inputs, research.signal_table(inputs)[0]
    )
    for key in (
        "p_value",
        "null_percentile",
        "p_value_pre_cost",
        "ruined_draws",
        "undefined_draws",
        "degenerate_segments",
        "beta_btc",
        "alpha_t_btc",
        "beta_ew",
        "alpha_t",
        "fee2x_sharpe",
        "capped_sharpe",
        "placebo_sharpe",
        "drop_top3_weeks_sharpe",
        "dsr",
        "dsr_p_value",
        "dsr_error",
        "dsr_raw_p_value",
        "lag_mask_mismatches",
        "by_year",
    ):
        assert key in result, key
    assert result["by_year"] == report.by_year
    assert "weekly_sharpe" not in json.dumps(sorted(result))


def test_trend_cost_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R59: pre-cost = fees x0 AND spread 0 for the own run and every draw, the funding
    calculator kept; fee2x doubles the fee; every draw runs the filter-free spec."""
    inputs = _trend_inputs(tmp_path, monkeypatch)
    family = research.load_family(_family_file(tmp_path / "f.json", 4, [inputs.spec.arm_hash()]))
    calls: list[tuple[float, float, bool, bool]] = []
    real = research.simulate_reference

    def spy(given: CarryInputs, **kw: object) -> object:
        fee = kw.get("fee_multiplier", 1.0)
        assert isinstance(fee, float)
        calls.append((fee, given.spec.spread, kw.get("ranker") is not None, given.calc is inputs.calc))
        return real(given, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(research, "simulate_reference", spy)
    research.evaluate(inputs, draws=2, seed=0, trials_path=tmp_path / "t.jsonl", family=family)
    precost = [call for call in calls if call[0] == 0.0]
    assert len(precost) == 3  # own + 2 draws
    assert all(call[1] == 0.0 and call[3] for call in precost)
    assert sorted(call[0] for call in calls if call[1] == 0.0001) == [1.0] * 5 + [2.0]
    assert sum(1 for call in calls if not call[3]) == 1  # the capped run's calculator
    assert sum(1 for call in calls if call[2]) == 5  # 2 + 2 draws + placebo


def test_trend_drop_top3_weeks_drops_their_days() -> None:
    """R61: the daily Sharpe without the days of the three best Monday weeks."""
    daily = [(_START + (i + 1) * _DAY, 0.0, 0.001 * ((i * 7) % 5 - 2)) for i in range(35)]
    weekly = [(_START + 7 * (w + 1) * _DAY, move) for w, move in enumerate([0.03, -0.01, 0.05, 0.02, 0.0])]
    kept = [
        row[2]
        for row in daily
        if not any(end - 7 * _DAY < row[0] <= end for end in (weekly[0][0], weekly[2][0], weekly[3][0]))
    ]
    from vibe_quant.portfolio.metrics import _annualised_sharpe

    assert research._drop_top3_daily(daily, weekly) == _annualised_sharpe(kept, 365.0)
    assert research._drop_top3_daily(daily, weekly[:3]) is None


def test_trend_evaluate_counts_trend_trials_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-c9t4-2: research passes the spec's signal to trial_counts -- funding lines in
    the ledger never move a trend evaluate's ledger count, and vice versa."""
    inputs = _trend_inputs(tmp_path, monkeypatch)
    family = research.load_family(_family_file(tmp_path / "f.json", 4, [inputs.spec.arm_hash()]))
    ledger = tmp_path / "t.jsonl"
    ledger.write_text(
        json.dumps({"spec_hash": "f" * 64, "weekly_sharpe": 0.5, "spec": {"symbols": []}}) + "\n"
    )
    result = research.evaluate(inputs, draws=0, seed=0, trials_path=ledger, family=family)
    assert result["ledger_num_trials"] == 4
    funding = research.evaluate(
        _inputs(tmp_path, name="fund"), draws=0, seed=0, trials_path=ledger
    )
    assert funding["dsr_num_trials"] == 5  # deadbeef-style funding line + own; trend ignored


def test_regime_gate_and_duty_at_zero() -> None:
    """R27 rule in the ranker: the filter is on only for a regime > 0 -- exactly 0.0 and
    None zero the book (eligible coins 0.0, ineligible stay None) and count as off in the
    duty cycle; the smallest positive regime passes the table's signals."""
    from types import SimpleNamespace

    table = _table(
        [{"A": 0.6, "B": None}, {"A": 0.4, "B": 0.2}, {"A": -0.2, "B": 1.0}],
        regime=[0.0, None, 5e-324],
    )
    gated = research.TableRanker(table, gate=True)
    live = {"A": 0.0, "B": None}
    assert gated(table.decisions[0], live) == {"A": 0.0, "B": None}
    assert gated(table.decisions[1], {"A": 0.0, "B": 0.0}) == {"A": 0.0, "B": 0.0}
    assert gated(table.decisions[2], {"A": 0.0, "B": 0.0}) == {"A": -0.2, "B": 1.0}
    assert research.TableRanker(table, gate=False)(table.decisions[0], live) == {"A": 0.6, "B": None}
    with pytest.raises(ValueError, match="B eligible"):
        gated(table.decisions[2], {"A": 0.0, "B": None})

    spec = CarrySpec.from_dates(
        ("A", "B"),
        "2025-01-06",
        "2025-01-10",
        signal="trend",
        schedule="daily",
        weighting="time_series",
        direction="long_flat",
        trend_filter="A",
        vol_target=0.4,
    )
    empty = SimpleNamespace(decisions=(), fills=())
    inputs = SimpleNamespace(spec=spec, specs={})
    stats = research.exposure_stats(empty, inputs, table)  # type: ignore[arg-type]
    assert stats["filter_duty"] == 1 / 3


def _lag_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CarryInputs:
    """The filter fixture with closes reaching 5 days earlier (a 5-day NC1 lag)."""
    base = _trend_inputs(tmp_path, monkeypatch, direction="long_flat", trend_filter="BTCUSDT")
    longer = _trend_inputs(
        tmp_path, monkeypatch, direction="long_flat", trend_filter="BTCUSDT", start="2024-02-25"
    )
    assert base.trend_closes is not None and longer.trend_closes is not None
    return replace(
        base, trend_closes=research.extend_closes(base.trend_closes, longer.trend_closes)
    )


def test_lagged_evaluate_never_touches_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1: the lag is not in the spec hash, so a lagged (NC1) evaluate must not write the
    trial ledger -- neither its completion nor a ruin. A real evaluate afterwards records
    the REAL (unlagged) metrics under the spec hash."""
    inputs = _lag_inputs(tmp_path, monkeypatch)
    spec_hash = inputs.spec.spec_hash()
    family = research.load_family(_family_file(tmp_path / "f.json", 4, [inputs.spec.arm_hash()]))
    ledger = tmp_path / "t.jsonl"
    seeded = json.dumps({"spec_hash": "e" * 64, "weekly_sharpe": 0.25, "spec": {"signal": "trend"}})
    ledger.write_text(seeded + "\n")
    before = ledger.read_bytes()

    lagged = research.evaluate(
        inputs, draws=2, seed=0, trials_path=ledger, family=family, signal_lag_days=5
    )
    assert ledger.read_bytes() == before
    native = carry_metrics(simulate_reference(inputs), inputs).summary
    assert lagged["total_return"] != native["total_return"]  # the lag really moved the own run
    assert lagged["ledger_num_trials"] == 5  # reads the ledger: 3 + {seeded, own}

    real_table_run = research.table_run

    def ruin_own(given: CarryInputs, table: research.SignalTable, **kw: float) -> object:
        if table.signals == research.signal_table(given, 5)[0].signals:
            raise CarryEquityTooSmall("BTCUSDT", given.decisions[2], 1.0, 2.0)
        return real_table_run(given, table, **kw)

    monkeypatch.setattr(research, "table_run", ruin_own)
    with pytest.raises(CarryEquityTooSmall):
        research.evaluate(
            inputs, draws=2, seed=0, trials_path=ledger, family=family, signal_lag_days=5
        )
    assert ledger.read_bytes() == before  # a lagged ruin is no failure trial either
    monkeypatch.setattr(research, "table_run", real_table_run)

    clear_rate_cache()
    real = research.evaluate(inputs, draws=2, seed=0, trials_path=ledger, family=family)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]["spec_hash"] == spec_hash
    assert (rows[1]["outcome"], rows[1]["total_return"], rows[1]["weekly_sharpe"]) == (
        "completed",
        native["total_return"],
        native["weekly_sharpe"],
    )
    assert real["total_return"] == native["total_return"]


def test_trend_p_value_and_percentile_from_the_rotation_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S1 / R58: p, null percentile and pre-cost p recomputed draw by draw from
    rotate_table + table_run + daily_sharpe_only -- the post-cost own Sharpe against the
    post-cost draws, the pre-cost own (fees x0, spread 0) against the pre-cost draws."""
    from vibe_quant.portfolio.metrics import daily_sharpe_only

    inputs = _trend_inputs(tmp_path, monkeypatch, direction="long_flat", trend_filter="BTCUSDT")
    family = research.load_family(_family_file(tmp_path / "f.json", 4, [inputs.spec.arm_hash()]))
    draws, seed = 6, 0  # seed 0: 3 post-cost draws reach the own Sharpe, 4 pre-cost ones
    result = research.evaluate(
        inputs, draws=draws, seed=seed, trials_path=tmp_path / "t.jsonl", family=family
    )
    table, _ = research.signal_table(inputs)
    precost = replace(inputs, spec=replace(inputs.spec, spread=0.0))
    nulls: list[float] = []
    precost_nulls: list[float] = []
    for index in range(draws):
        rotated, _ = research.rotate_table(table, research.draw_fraction(seed, index))
        for given, fee, out in ((inputs, 1.0, nulls), (precost, 0.0, precost_nulls)):
            sharpe = daily_sharpe_only(research.table_run(given, rotated, fee_multiplier=fee), given)
            assert sharpe is not None
            out.append(sharpe)
    own = result["sharpe"]
    own_precost = daily_sharpe_only(simulate_reference(precost, fee_multiplier=0.0), precost)
    assert own_precost is not None
    assert result["p_value"] == (1 + sum(n >= own for n in nulls)) / (1 + draws)
    assert result["null_percentile"] == sum(n < own for n in nulls) / draws
    assert result["p_value_pre_cost"] == (1 + sum(n >= own_precost for n in precost_nulls)) / (
        1 + draws
    )
    # teeth: the post-cost own against the pre-cost draws would give another p
    assert result["p_value"] != (1 + sum(n >= own for n in precost_nulls)) / (1 + draws)
    assert (result["ruined_draws"], result["undefined_draws"]) == (0, 0)


def test_trend_ruined_and_undefined_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R58 / R-c9t5-3: a draw raising CarryEquityTooSmall is ruined: -inf (below own,
    it lowers p and raises the percentile), counted, never dropped. N1: a draw whose
    Sharpe is undefined (None) counts conservatively as >= own (raises p, never below
    own in the percentile), counted in undefined_draws. Both passes see both draws."""
    from vibe_quant.portfolio import metrics

    inputs = _trend_inputs(tmp_path, monkeypatch, direction="long_flat", trend_filter="BTCUSDT")
    family = research.load_family(_family_file(tmp_path / "f.json", 4, [inputs.spec.arm_hash()]))
    draws, seed, ruined_at, undefined_at = 6, 3, 1, 4
    table, _ = research.signal_table(inputs)
    # marked by the draw's u (two draws can rotate to the same table): the objects
    # rotate_table returns for the ruined / undefined draw, and the runs built from them
    marks = {
        research.draw_fraction(seed, ruined_at): "ruined",
        research.draw_fraction(seed, undefined_at): "undefined",
    }
    marked: list[tuple[object, str]] = []
    real_rotate, real_table_run = research.rotate_table, research.table_run
    real_sharpe = metrics.daily_sharpe_only

    def rotate_table(given: research.SignalTable, u: float, *rest: object) -> object:
        rotated = real_rotate(given, u, *rest)  # type: ignore[arg-type]
        if u in marks:
            marked.append((rotated[0], marks[u]))
        return rotated

    def mark_of(item: object) -> str | None:
        return next((mark for obj, mark in marked if obj is item), None)

    def table_run(given: CarryInputs, rotated: research.SignalTable, **kw: float) -> object:
        mark = mark_of(rotated)
        if mark == "ruined":
            raise CarryEquityTooSmall("BTCUSDT", given.decisions[3], 1.0, 2.0)
        run = real_table_run(given, rotated, **kw)
        if mark == "undefined":
            marked.append((run, "undefined"))
        return run

    def daily_sharpe_only(run: object, given: CarryInputs) -> float | None:
        return None if mark_of(run) == "undefined" else real_sharpe(run, given)  # type: ignore[arg-type]

    monkeypatch.setattr(research, "rotate_table", rotate_table)
    monkeypatch.setattr(research, "table_run", table_run)
    monkeypatch.setattr(research, "daily_sharpe_only", daily_sharpe_only)
    result = research.evaluate(
        inputs, draws=draws, seed=seed, trials_path=tmp_path / "t.jsonl", family=family
    )
    monkeypatch.undo()

    precost = replace(inputs, spec=replace(inputs.spec, spread=0.0))
    kept = [i for i in range(draws) if i not in (ruined_at, undefined_at)]
    nulls, precost_nulls = [], []
    for index in kept:
        rotated, _ = research.rotate_table(table, research.draw_fraction(seed, index))
        nulls.append(real_sharpe(real_table_run(inputs, rotated), inputs))
        precost_nulls.append(
            real_sharpe(real_table_run(precost, rotated, fee_multiplier=0.0), precost)
        )
    own = result["sharpe"]
    own_precost = real_sharpe(simulate_reference(precost, fee_multiplier=0.0), precost)
    assert own is not None and own_precost is not None
    # ruined: in the denominator, never >= own, always < own; undefined: always >= own
    assert result["p_value"] == (1 + 1 + sum(n >= own for n in nulls)) / (1 + draws)  # type: ignore[operator]
    assert result["null_percentile"] == (1 + sum(n < own for n in nulls)) / draws  # type: ignore[operator]
    assert result["p_value_pre_cost"] == (
        1 + 1 + sum(n >= own_precost for n in precost_nulls)  # type: ignore[operator]
    ) / (1 + draws)
    assert (result["ruined_draws"], result["undefined_draws"]) == (1, 1)


def test_lagged_evaluate_verdict_is_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N2: a lagged (NC1) evaluate reports verdict "calibration" even when every
    criterion passes; the per-criterion results stay."""
    inputs = _lag_inputs(tmp_path, monkeypatch)
    family = research.load_family(_family_file(tmp_path / "f.json", 4, [inputs.spec.arm_hash()]))
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"sharpe": {"op": ">=", "bar": -1e9}}))
    criteria = research.load_criteria(path)
    result = research.evaluate(
        inputs,
        draws=1,
        seed=0,
        trials_path=tmp_path / "t.jsonl",
        family=family,
        criteria=criteria,
        signal_lag_days=5,
    )
    assert (result["verdict"], result["criteria_all_pass"]) == ("calibration", True)
