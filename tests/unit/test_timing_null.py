"""Timing-null kernel: as-traded account sim, day-close marking, data fail-closed, verdict rule.

The as-traded sim is checked against the engine's own mode-off MTM series: NT-shaped positions
(a trade still open at the run end has its entry fill only) through
``extraction.mark_to_market_paths`` + ``mark_to_market_daily_returns``, exactly as the screening
runner builds it, so its day grid, marking, open-position valuation and arithmetic order are
the engine's, not a second opinion.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import math
import sys
from array import array
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from nautilus_trader.model.enums import OrderSide

from tests.unit.fake_engine import FakePosition, fake_engine
from vibe_quant.discovery import timing_null as tn
from vibe_quant.metrics import (
    DailyReturns,
    LedgerBook,
    LedgerEvent,
    mark_to_market_daily_returns,
    pool_daily_returns,
)
from vibe_quant.validation.extraction import mark_to_market_paths
from vibe_quant.validation.funding import FundingCalculator

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

MS = 10**6
H4 = 4 * 3_600 * 10**9
DAY = 86_400 * 10**9
B0 = 1000.0
ZERO = tn.TradeCosts(fee=0.0)


def _ns(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp()) * 10**9


def _tape(
    opens: Sequence[float],
    closes: Sequence[float] | None = None,
    *,
    symbol: str = "BTCUSDT",
    start: str = "2024-01-01",
    tf_ns: int = H4,
    timeframe: str = "4h",
    drop: Sequence[int] = (),
) -> tn.WindowTape:
    """Bars opening every ``tf_ns`` from ``start``; ``ts`` = close - 1 ms (Binance kline)."""
    o = np.asarray(opens, dtype=np.float64)
    c = o.copy() if closes is None else np.asarray(closes, dtype=np.float64)
    start_ns = _ns(start)
    open_ns = start_ns + np.arange(len(o), dtype=np.int64) * tf_ns
    keep = np.setdiff1d(np.arange(len(o)), np.asarray(drop, dtype=np.int64))
    return tn.WindowTape(
        symbol=symbol,
        timeframe=timeframe,
        tf_ns=tf_ns,
        start_ns=start_ns,
        end_ns=start_ns + len(o) * tf_ns,
        open_ns=open_ns[keep],
        ts=open_ns[keep] + tf_ns - MS,
        open=o[keep],
        close=c[keep],
    )


def _span(side: int, frac: float, entry: int, hold: int, carried: bool = False) -> tn.TradeSpan:
    return tn.TradeSpan(side=side, frac=frac, entry_bar=entry, hold_bars=hold, carried=carried)


def _reference(
    tape: tn.WindowTape, spans: Sequence[tn.TradeSpan], costs: tn.TradeCosts
) -> DailyReturns:
    """The engine's mode-off MTM series of the same fills (nt_runner: ``mark_to_market_paths``
    over NT positions + funding ``extra_cash``, then ``mark_to_market_daily_returns``).

    A span reaching the tape end is an OPEN position: entry fill only, valued by the engine's
    own marking at the last close, plus the run-end mark's estimated taker exit fee booked as
    modeled cash at the last bar (``mark_open_positions`` -> ``accumulate`` ->
    ``extra_cash``; no exit fill or slippage). ``LedgerBook`` only sizes each entry from the
    account's realized equity (``frac`` x equity / fill price).
    """
    iid = f"{tape.symbol}-PERP.BINANCE"
    book = LedgerBook()
    positions: list[FakePosition] = []
    cash: list[tuple[int, float]] = []
    n = len(tape.open)
    for k, sp in enumerate(sorted(spans, key=lambda s: s.entry_bar)):
        i, s = sp.entry_bar, sp.side
        is_open = i + sp.hold_bars >= n
        xb = n - 1 if is_open else i + sp.hold_bars
        p_in = float(tape.open[i])
        if costs.fill_price is not None:
            p_in = costs.fill_price(tape.symbol, s, p_in)
        q = sp.frac * (B0 + book.realized) / p_in
        side = OrderSide.BUY if s > 0 else OrderSide.SELL
        entry = {"entry_ns": int(tape.ts[i]), "entry_price": p_in, "qty": q, "entry_side": side}
        book.apply(
            LedgerEvent(ts=int(tape.ts[i]), qty=s * q, price=p_in, cash=-costs.fee * q * p_in)
        )
        pays: Sequence[tuple[int, float]] = []
        if costs.funding is not None:
            # the engine funds from the FILL instants (bar open + 59.999 s), a carried hold to
            # the run-end mark (the oracle is spelled out, not tn.funding_window)
            f_in = int(tape.open_ns[i]) + 59_999_000_000
            f_out = int(tape.ts[xb]) if is_open else int(tape.open_ns[xb]) + 59_999_000_000
            pays = costs.funding(tape.symbol, s, q * p_in, f_in, f_out)
            cash += [(t, -a) for t, a in pays]
        for t, a in pays:
            book.apply(LedgerEvent(ts=t, cash=-a))
        if is_open:
            # the runner's mark: -taker x |qty| x last close at the last bar (modeled cash)
            cash.append((int(tape.ts[n - 1]), -(abs(q) * float(tape.close[n - 1]) * costs.fee)))
            pos = FakePosition.open(f"P-{k}", iid, entry_fee=costs.fee * q * p_in, **entry)
        else:
            p_out = float(tape.open[xb])
            if costs.fill_price is not None:
                p_out = costs.fill_price(tape.symbol, -s, p_out)
            exit_ev = LedgerEvent(
                ts=int(tape.ts[xb]), qty=-s * q, price=p_out, cash=-costs.fee * q * p_out
            )
            book.apply(exit_ev)
            pos = FakePosition.closed(
                f"P-{k}",
                iid,
                exit_ns=int(tape.ts[xb]),
                exit_price=p_out,
                entry_fee=costs.fee * q * p_in,
                exit_fee=costs.fee * q * p_out,
                **entry,
            )
        positions.append(pos)
    hi, lo = np.maximum(tape.open, tape.close), np.minimum(tape.open, tape.close)
    bars = [
        SimpleNamespace(ts_init=int(t), open=float(o), high=float(h), low=float(lw), close=float(c))
        for t, o, h, lw, c in zip(tape.ts, tape.open, hi, lo, tape.close, strict=True)
    ]
    step = {H4: "4-HOUR", DAY: "1-DAY"}[tape.tf_ns]
    engine = fake_engine(positions, bars_by_type={f"{iid}-{step}-LAST-EXTERNAL": bars})
    paths = mark_to_market_paths(
        engine,
        execution_timeframe=tape.timeframe,
        start_ns=tape.start_ns,
        end_ns=tape.end_ns,
        extra_cash={iid: cash} if cash else None,
    )
    assert paths is not None
    out = mark_to_market_daily_returns(B0, paths, tape.start_ns, tape.end_ns)
    assert out is not None
    return out


def _series(first_day: int, values: Sequence[float]) -> DailyReturns:
    return DailyReturns(first_day, array("d", values))


def _closes(
    symbols: Sequence[str], first_day: int, n: int, noisy: bool = True
) -> dict[str, dict[int, float]]:
    """1d closes for days ``first_day - 1 .. first_day + n - 1`` (constant when not noisy)."""
    rng = np.random.default_rng(0)
    out: dict[str, dict[int, float]] = {}
    for s in symbols:
        steps = 1.0 + (rng.normal(0, 0.01, n + 1) if noisy else np.zeros(n + 1))
        out[s] = {first_day - 1 + k: float(v) for k, v in enumerate(100.0 * np.cumprod(steps))}
    return out


def _check(
    tapes: dict[str, list[tn.WindowTape]],
    spans: dict[str, list[list[tn.TradeSpan]]],
    *,
    real: DailyReturns | None = None,
    costs: tn.TradeCosts = ZERO,
    draws: int = 5,
    closes: dict[str, dict[int, float]] | None = None,
    exposure_max: float = tn.EXPOSURE_MAX,
) -> tn.TimingNullRecord:
    """Run the check with a real series on the tapes' own chained grid (flat unless given)."""
    first = next(iter(tapes.values()))
    grid_first = first[0].start_ns // DAY + 1
    n_days = sum((w.end_ns - 1) // DAY - w.start_ns // DAY + 1 for w in first) - 1
    if real is None:
        real = _series(grid_first, [0.0] * n_days)
    agg: Callable[[Sequence[DailyReturns | None]], DailyReturns | None] = (
        tn.single_series if len(tapes) == 1 else pool_daily_returns
    )
    return tn.timing_null_check(
        real,
        tapes,
        spans,
        costs=costs,
        b0=B0,
        draws=draws,
        seeds={"+".join(tapes): 7},  # one seed per book: the symbols joined by "+"
        aggregate=agg,
        closes=closes if closes is not None else _closes(list(tapes), real.first_day, len(real)),
        exposure_max=exposure_max,
    )


# --------------------------------------------------------------------------- moved kernel (B1)


def test_moved_kernel_keeps_the_report_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    null = tn.null_draws(
        np.linspace(100, 120, 41),
        np.array([3, 5, 2]),
        np.array([1, -1, 1]),
        0.0005,
        500,
        np.random.default_rng(7),
    )[0]
    assert tn.null_percentile(-0.001, null) == 0.98
    spec = importlib.util.spec_from_file_location(
        "oos_analysis_moved",
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "research"
        / "tv_ports"
        / "oos_analysis.py",
    )
    assert spec is not None and spec.loader is not None
    oa = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "oos_analysis_moved", oa)
    spec.loader.exec_module(oa)
    for name in ("trade_multiplier", "place_trades", "null_draws", "null_percentile", "alpha_beta"):
        assert getattr(oa, name) is getattr(tn, name), name


# --------------------------------------------------------------------------- verdict rule (B2)


def test_percentile_threshold_edges() -> None:
    policy = tn.TimingNullPolicy(enforced=True, draws=500, pct=0.95, exposure_max=0.9, base_seed=7)
    assert tn.null_percentile(0.1, np.full(500, 0.1)) == 0.0
    at = tn.null_percentile(0.1, np.array([0.0] * 475 + [0.2] * 25))
    below = tn.null_percentile(0.1, np.array([0.0] * 474 + [0.2] * 26))
    assert (at, below) == (0.95, 0.948)
    assert tn.timing_null_decision(policy, {"percentile": at, "fallback": None}) == (True, [])
    assert tn.timing_null_decision(policy, {"percentile": below, "fallback": None}) == (
        False,
        ["Timing null: percentile 0.9480 < 0.95"],
    )
    assert tn.timing_null_decision(policy, {"percentile": 0.0, "fallback": None}) == (
        False,
        ["Timing null: percentile 0.0000 < 0.95"],
    )


def test_decision_fails_closed_and_policy_parse() -> None:
    policy = tn.TimingNullPolicy(enforced=True, draws=500, pct=0.95, exposure_max=0.9, base_seed=7)
    assert tn.timing_null_decision(None, None) == (True, [])
    assert tn.timing_null_decision(policy, None) == (
        False,
        ["Timing null: record unreadable (missing)"],
    )
    for bad in ("x", float("nan"), 1.2, None, True):
        ok, reasons = tn.timing_null_decision(policy, {"percentile": bad, "fallback": None})
        assert (ok, reasons) == (False, [f"Timing null: record unreadable (percentile {bad!r})"])
    over = {"percentile": None, "exposure": 0.9401, "fallback": "Timing null: untestable"}
    line = "Timing null: exposure 0.9401 > 0.90, timing untestable (not certified)"
    # R3: ANY record carrying a fallback rejects -- a legacy stored buy-and-hold pass too
    for legacy in (False, True, None):
        assert tn.timing_null_decision(policy, {**over, "fallback_passed": legacy}) == (
            False,
            [line],
        )
    legacy_pass = {**over, "fallback": "passed", "fallback_passed": True}
    assert tn.timing_null_decision(policy, legacy_pass) == (False, [line])
    wide = tn.TimingNullPolicy(enforced=True, draws=500, pct=0.95, exposure_max=0.95, base_seed=7)
    assert tn.timing_null_decision(wide, legacy_pass)[1] == [
        "Timing null: exposure 0.9401 > 0.95, timing untestable (not certified)"
    ]
    off = tn.TimingNullPolicy(enforced=False, draws=500, pct=0.95, exposure_max=0.9, base_seed=7)
    assert tn.timing_null_decision(off, legacy_pass) == (True, [])  # R4
    assert tn.parse_timing_null_policy({}) is None
    assert tn.parse_timing_null_policy(
        {
            "timing_null_gate": {
                "enforced": True,
                "draws": 500,
                "pct": 0.96,
                "exposure_max": 0.9,
                "base_seed": 3,
            }
        }
    ) == tn.TimingNullPolicy(enforced=True, draws=500, pct=0.96, exposure_max=0.9, base_seed=3)
    for bad_gate in ("on", {"enforced": "yes", "pct": 0.95}, {"enforced": True, "pct": 1.5}):
        with pytest.raises(tn.TimingNullPolicyError):
            tn.parse_timing_null_policy({"timing_null_gate": bad_gate})


# --------------------------------------------------------------------------- as-traded sizing (B3-B5)


def test_as_traded_size_compounds_entry_equity_fractions() -> None:
    day_tape = _tape([100, 180, 100, 50], [180, 100, 50, 50], tf_ns=DAY, timeframe="1d")
    half = tn.sim_daily(day_tape, [_span(1, 0.5, 0, 1), _span(1, 0.5, 2, 1)], ZERO, B0)
    full = tn.sim_daily(day_tape, [_span(1, 1.0, 0, 1), _span(1, 1.0, 2, 1)], ZERO, B0)
    # account probe (equity ratio): 0.050000000000000044 / -0.09999999999999998; T compounds the
    # day-close returns, so the same trades land within 1 ulp-scale of it
    assert list(half.values) == [0.3999999999999999, 0.0, -0.25, 0.0]  # 1 + 0.5*0.8, then 0.75x
    assert list(full.values) == [0.8, 0.0, -0.5, 0.0]
    assert tn.total_return(half.values) == pytest.approx(0.050000000000000044, abs=1e-15)
    assert tn.total_return(full.values) == pytest.approx(-0.09999999999999998, abs=1e-15)
    short = _tape([100, 220, 100, 50], [220, 100, 50, 50], tf_ns=DAY, timeframe="1d")
    ruined = tn.sim_daily(short, [_span(-1, 1.0, 0, 1), _span(-1, 1.0, 2, 1)], ZERO, B0)
    assert list(ruined.values) == [-1.0, 0.0, 0.0, 0.0]  # absorbs at 0, never +0.04
    assert tn.total_return(ruined.values) == -1.0


def test_pooled_compound_uses_the_aggregate_for_real_and_null() -> None:
    a, b = _series(10, [0.5, -0.5]), _series(10, [-0.5, 0.5])
    pooled = pool_daily_returns([a, b])
    assert pooled is not None
    assert tn.total_return(pooled.values) == 0.0  # not the mean terminal -0.25
    calls: list[list[str]] = []

    def spy(parts: Sequence[DailyReturns | None]) -> DailyReturns | None:
        calls.append([f"{len(p)}" for p in parts if p is not None])
        return pool_daily_returns(parts)

    tapes = {s: [_tape([100.0] * 18, symbol=s)] for s in ("BTCUSDT", "ETHUSDT")}
    spans = {s: [[_span(1, 0.5, 2, 3)]] for s in tapes}
    real = _series(tapes["BTCUSDT"][0].start_ns // DAY + 1, [0.5, -0.5])
    rec = tn.timing_null_check(
        real,
        tapes,
        spans,
        costs=ZERO,
        b0=B0,
        draws=4,
        seeds={"BTCUSDT+ETHUSDT": 1},
        aggregate=spy,
        closes=_closes(list(tapes), real.first_day, 2),
    )
    assert calls == [["2", "2"]] * 4  # one pooled series per draw, both symbols, chained grid
    assert rec.real_T == tn.total_return(real.values) == -0.25
    assert rec.null_median == 0.0 and rec.percentile == 0.0


def test_triples_kept_in_every_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    # 3 long + 2 short by count (60/40) and 60/40 by held bars: both mixes are pinned
    spans = [
        _span(1, 0.2, 0, 20),
        _span(-1, 0.4, 25, 22),
        _span(1, 0.1, 50, 25),
        _span(-1, 0.3, 80, 18),
        _span(1, 0.25, 100, 15),
    ]
    want = Counter((sp.hold_bars, sp.side, sp.frac) for sp in spans)
    seen: list[Counter[tuple[int, int, float]]] = []
    real_sim = tn.sim_daily

    def spy(
        tape: tn.WindowTape, placed: Sequence[tn.TradeSpan], costs: tn.TradeCosts, b0: float
    ) -> DailyReturns:
        seen.append(Counter((sp.hold_bars, sp.side, sp.frac) for sp in placed))
        return real_sim(tape, placed, costs, b0)

    monkeypatch.setattr(tn, "sim_daily", spy)
    rng = np.random.default_rng(3)
    tape = _tape(100.0 + rng.normal(0, 1, 240).cumsum())
    rec = _check({"BTCUSDT": [tape]}, {"BTCUSDT": [spans]}, draws=500)
    assert len(seen) == 500 and rec.draws == 500
    assert all(c == want for c in seen)
    assert sum(1 for sp in spans if sp.side == 1) == 3
    assert sum(sp.hold_bars for sp in spans if sp.side == 1) == 60
    assert sum(sp.hold_bars for sp in spans if sp.side == -1) == 40


def test_twelve_trades_run_the_null() -> None:
    spans = [_span(1 if k % 2 else -1, 0.5, 10 * k, 3) for k in range(12)]
    rng = np.random.default_rng(11)
    tape = _tape(100.0 + rng.normal(0, 1, 120).cumsum())
    rec = _check({"BTCUSDT": [tape]}, {"BTCUSDT": [spans]}, draws=20)
    assert rec.fallback is None and rec.draws == 20
    assert rec.percentile is not None and 0.0 <= rec.percentile <= 1.0


def test_single_symbol_draws_are_bit_identical_to_main() -> None:
    # Values pinned on main efc19e47 (identical on the costed-null base): a one-symbol book's
    # units are its trades and its book seed is the symbol's, so joint re-timing moves nothing.
    rng = np.random.default_rng(11)
    twelve = _check(
        {"BTCUSDT": [_tape(100.0 + rng.normal(0, 1, 120).cumsum())]},
        {"BTCUSDT": [[_span(1 if k % 2 else -1, 0.5, 10 * k, 3) for k in range(12)]]},
        draws=20,
    )
    rng = np.random.default_rng(3)
    walk = _tape(100.0 + rng.normal(0, 1, 240).cumsum())
    five = [
        _span(1, 0.2, 0, 20),
        _span(-1, 0.4, 25, 22),
        _span(1, 0.1, 50, 25),
        _span(-1, 0.3, 80, 18),
        _span(1, 0.25, 100, 15),
    ]
    placed: list[list[tuple[int, float, int, int]]] = []
    real_sim = tn.sim_daily

    def spy(
        tape: tn.WindowTape, spans: Sequence[tn.TradeSpan], costs: tn.TradeCosts, b0: float
    ) -> DailyReturns:
        placed.append([(sp.side, sp.frac, sp.entry_bar, sp.hold_bars) for sp in spans])
        return real_sim(tape, spans, costs, b0)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tn, "sim_daily", spy)
        walked = _check({"BTCUSDT": [walk]}, {"BTCUSDT": [five]}, draws=40)
    w1 = _tape([100.0] * 12, [100.0] * 5 + [50.0] * 7)
    w2 = _tape([100.0] * 12, [100.0] * 5 + [75.0] * 7, start="2024-01-03")
    carried = [_span(1, 1.0, 0, 12, True)]
    first = _check(
        {"BTCUSDT": [w1, w2]},
        {"BTCUSDT": [carried, carried]},
        real=_series(w1.start_ns // DAY + 1, [0.0, -0.25, 0.0]),
        draws=3,
        exposure_max=1.0,
    )
    flat = _tape([100.0] * 60)
    short = {"BTCUSDT": [[_span(-1, 1.0, 10, 6)]]}

    def credit(
        sym: str, side: int, notional: float, entry_ns: int, exit_ns: int
    ) -> list[tuple[int, float]]:
        return [(exit_ns, side * 0.002 * notional)]

    fee = _check({"BTCUSDT": [flat]}, short, costs=tn.TradeCosts(fee=0.0005), draws=6)
    funded = _check(
        {"BTCUSDT": [flat]}, short, costs=tn.TradeCosts(fee=0.0005, funding=credit), draws=6
    )
    got = {
        name: (r.percentile, r.null_median, r.null_p95, r.seeds, r.exposure, r.carried)
        for name, r in (
            ("twelve", twelve),
            ("walked", walked),
            ("first", first),
            ("fee", fee),
            ("funded", funded),
        )
    }
    assert got == {
        "twelve": (0.35, 0.010742485952267788, 0.04277775511838379, {"BTCUSDT": 7}, 0.3, 0),
        "walked": (
            0.35,
            0.013750785264188381,
            0.04551901586183882,
            {"BTCUSDT": 7},
            0.4166666666666667,
            0,
        ),
        "first": (0.0, -0.25, -0.25, {"BTCUSDT": 7}, 1.0, 2),
        "fee": (1.0, -0.0009999999999998899, -0.0009999999999998899, {"BTCUSDT": 7}, 0.1, 0),
        "funded": (0.0, 0.001000000000000112, 0.001000000000000112, {"BTCUSDT": 7}, 0.1, 0),
    }
    assert twelve.zero_hold_clipped == walked.zero_hold_clipped == 0
    # the very span lists main handed to sim_daily, draw by draw (first and last shown)
    assert len(placed) == 40
    assert placed[0] == [(1, 0.1, 8, 25), (1, 0.2, 56, 20), (1, 0.25, 124, 15)] + [
        (-1, 0.4, 167, 22),
        (-1, 0.3, 197, 18),
    ]
    assert placed[39] == [(1, 0.25, 25, 15), (-1, 0.3, 58, 18), (1, 0.1, 85, 25)] + [
        (1, 0.2, 178, 20),
        (-1, 0.4, 199, 22),
    ]
    assert hashlib.sha256(repr(placed).encode()).hexdigest()[:16] == "2f200ab234a975d9"


# --------------------------------------------------------------------------- day-close marking (GATES-5)


def test_overnight_trade_is_marked_at_each_utc_day_close() -> None:
    # long 1.0 enters day 0 20:00 at 100, exits day 2 20:00 at 100: same terminal, different days
    flat = _tape([100.0] * 18)
    closes = [100.0] * 18
    closes[11] = 120.0  # day 1's last 4h bar closes at 120
    gap = _tape([100.0] * 18, closes)
    trade = [_span(1, 1.0, 5, 12)]
    assert list(tn.sim_daily(flat, trade, ZERO, B0).values) == [0.0, 0.0, 0.0]
    marked = tn.sim_daily(gap, trade, ZERO, B0)
    assert list(marked.values) == [0.0, 0.19999999999999996, -0.16666666666666663]
    assert marked == _reference(gap, trade, ZERO)


def _funding(
    sym: str, side: int, notional: float, entry_ns: int, exit_ns: int
) -> list[tuple[int, float]]:
    """Signed 8h settlements in (entry, exit], positive = paid (FundingModel.accrue's shape)."""
    rate = 0.0003 if sym == "BTCUSDT" else -0.0002
    first = (entry_ns // (8 * 3_600 * 10**9) + 1) * 8 * 3_600 * 10**9
    return [(t, side * rate * notional) for t in range(first, exit_ns + 1, 8 * 3_600 * 10**9)]


COSTED = tn.TradeCosts(
    fee=0.0005, fill_price=lambda _sym, side, px: px + side * 0.25, funding=_funding
)


@pytest.mark.parametrize("costs", [ZERO, COSTED], ids=["zero-cost", "fee-slippage-funding"])
@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_unshuffled_schedule_matches_the_engine_ledger(costs: tn.TradeCosts, symbol: str) -> None:
    rng = np.random.default_rng(5)
    path = 100.0 + rng.normal(0, 2, 37).cumsum()
    tape = _tape(path[:-1], path[1:] + rng.normal(0, 0.5, 36), symbol=symbol)
    # back-to-back exit/re-entry at bar 9, overnight holds, and a carried trade to the last close
    spans = [
        _span(1, 0.5, 1, 8),
        _span(-1, 0.8, 9, 7),
        _span(1, 0.3, 20, 5),
        _span(-1, 1.0, 30, 6, True),
    ]
    assert tn.sim_daily(tape, spans, costs, B0) == _reference(tape, spans, costs)


def test_trade_open_at_the_window_end_pays_the_mark_exit_fee_like_the_engine() -> None:
    # long 1.0 at 100 from bar 10, still open at the end; flat price, fee 5 bp, tick slippage:
    # the engine charges the entry fee and its slipped fill, marks the open position at the
    # last close and books the mark's estimated taker exit fee there (no exit fill, no exit
    # slippage) -- real NT: test_timing_null_gate::test_carried_trade_kernel_equals_engine_mtm
    tape = _tape([100.0] * 18)
    carried = [_span(1, 1.0, 10, 8, True)]
    fee_only = tn.TradeCosts(fee=0.0005)
    got = tn.sim_daily(tape, carried, fee_only, B0)
    assert got == _reference(tape, carried, fee_only)
    assert tn.total_return(got.values) == -0.0009999999999998899
    slipped = tn.TradeCosts(fee=0.0005, fill_price=lambda _sym, side, px: px + side * 0.25)
    got = tn.sim_daily(tape, carried, slipped, B0)
    assert got == _reference(tape, carried, slipped)
    # entry fee 0.5 + 0.25/100.25 of the 1000 notional lost to the slipped entry, plus the
    # exit fee on the mark's value (q x 100 at the unslipped last close)
    q = 1000.0 / 100.25
    assert tn.total_return(got.values) == pytest.approx(
        -(0.5 + 1000.0 * 0.25 / 100.25 + 0.0005 * q * 100.0) / B0
    )
    # a null copy placed flush with the end is the same open position: same valuation
    flush = [_span(1, 1.0, 10, 8)]
    assert tn.sim_daily(tape, flush, fee_only, B0) == tn.sim_daily(tape, carried, fee_only, B0)


def test_ruin_books_one_full_loss_then_zero_and_the_next_window_resets() -> None:
    # short 1.0 at 100; day 1 closes at 210 (equity -100): ruin there, even though price returns
    closes = [100.0] * 18
    closes[11] = 210.0
    tape = _tape([100.0] * 18, closes)
    ruin = [_span(-1, 1.0, 0, 15)]
    assert list(tn.sim_daily(tape, ruin, ZERO, B0).values) == [0.0, -1.0, 0.0]
    window2 = _tape([100.0] * 18, closes, start="2024-01-04")
    assert list(tn.sim_daily(window2, ruin, ZERO, B0).values) == [0.0, -1.0, 0.0]  # fresh b0
    assert list(tn.sim_daily(window2, [_span(1, 1.0, 0, 15)], ZERO, B0).values) == [
        0.0,
        2100.0 / 1000.0 - 1.0,
        1000.0 / 2100.0 - 1.0,
    ]
    # ruin at an EXIT: a 2x long loses 110% at its bar-2 exit; the same-day 3x re-entry must
    # never trade (without the exit checkpoint it books -0.9333 instead of the absorbing -1.0)
    crash = _tape([100, 100, 45, 45, 20, 20, 20, 20, 20, 20, 20, 20])
    after_ruin = [_span(1, 2.0, 0, 2), _span(1, 3.0, 3, 2)]
    assert list(tn.sim_daily(crash, after_ruin, ZERO, B0).values) == [-1.0, 0.0]


def test_first_window_day_zero_is_trimmed_like_the_engine_chain() -> None:
    # each window: one long held over both days; day-0 close is the window's loss day
    w1 = _tape([100.0] * 12, [100.0] * 5 + [50.0] * 7)
    w2 = _tape([100.0] * 12, [100.0] * 5 + [75.0] * 7, start="2024-01-03")
    full = {"BTCUSDT": [[_span(1, 1.0, 0, 12, True)], [_span(1, 1.0, 0, 12, True)]]}
    real = _series(w1.start_ns // DAY + 1, [0.0, -0.25, 0.0])
    rec = _check({"BTCUSDT": [w1, w2]}, full, real=real, draws=3, exposure_max=1.0)
    # w1's day-0 -50% is dropped (engine trim); w2's day 0 is kept vs a fresh balance
    assert (rec.null_median, rec.null_p95, rec.real_T) == (-0.25, -0.25, -0.25)
    assert rec.carried == 2 and rec.exposure == 1.0


# --------------------------------------------------------------------------- production costs (GATES-9)


def test_signed_funding_and_fees_reach_every_null_draw() -> None:
    tape = _tape([100.0] * 60)
    spans = {"BTCUSDT": [[_span(-1, 1.0, 10, 6)]]}

    def credit(
        sym: str, side: int, notional: float, entry_ns: int, exit_ns: int
    ) -> list[tuple[int, float]]:
        return [(exit_ns, side * 0.002 * notional)]  # short on a positive rate: +2.0 credit

    fee_only = _check({"BTCUSDT": [tape]}, spans, costs=tn.TradeCosts(fee=0.0005), draws=6)
    funded = _check(
        {"BTCUSDT": [tape]}, spans, costs=tn.TradeCosts(fee=0.0005, funding=credit), draws=6
    )
    # fees 0.5 per leg on 1000 notional; a short on a positive rate is credited 2.0
    assert fee_only.null_median == pytest.approx(-0.001, rel=1e-12)
    assert fee_only.null_p95 == pytest.approx(-0.001, rel=1e-12)
    assert funded.null_median == pytest.approx(0.001, rel=1e-12)
    assert funded.null_p95 == pytest.approx(0.001, rel=1e-12)
    assert fee_only.fee == funded.fee == 0.0005


FILL_DELAY = 59_999_000_000  # bar open + 59.999 s: where the engine stamps the fill


def _recorder(
    calls: list[tuple[str, int, float, int, int]],
) -> Callable[[str, int, float, int, int], list[tuple[int, float]]]:
    """An 8h-grid funding hook (rate 0.0003, positive = paid) that remembers its arguments."""

    def hook(
        sym: str, side: int, notional: float, entry_ns: int, exit_ns: int
    ) -> list[tuple[int, float]]:
        calls.append((sym, side, notional, entry_ns, exit_ns))
        return _funding("BTCUSDT", side, notional, entry_ns, exit_ns)

    return hook


@pytest.mark.parametrize(
    ("tf_ns", "timeframe", "bar_at"),
    [
        (DAY, "1d", "2024-01-02"),
        (H4, "4h", "2024-01-02T08:00"),
        (60 * 10**9, "1m", "2024-01-01T00:03"),
    ],
    ids=["1d", "4h", "1m"],
)
def test_funding_runs_between_the_fill_instants(tf_ns: int, timeframe: str, bar_at: str) -> None:
    """The engine funds from the FILL instant (bar open + 59.999 s) to the exit fill, not from
    the bar close: on 1d bars the old (ts[i], ts[xb]) window was one day late."""
    tape = _tape([100.0] * 40, tf_ns=tf_ns, timeframe=timeframe)
    i = int((_ns(bar_at) - tape.start_ns) // tf_ns)
    calls: list[tuple[str, int, float, int, int]] = []
    costs = tn.TradeCosts(fee=0.0, funding=_recorder(calls))
    tn.sim_daily(tape, [_span(1, 1.0, i, 2)], costs, B0)  # long, 2 bars
    open_i = _ns(bar_at)
    assert calls == [("BTCUSDT", 1, 1000.0, open_i + FILL_DELAY, open_i + 2 * tf_ns + FILL_DELAY)]
    if timeframe == "1m":  # a 1m bar's fill instant is the bar's own ts (close - 1 ms)
        assert calls[0][3] == int(tape.ts[i])
    # a hold reaching the tape end is valued at the run-end mark: its exit is the last ts
    calls.clear()
    tn.sim_daily(tape, [_span(-1, 1.0, 36, 4, True)], costs, B0)
    assert calls == [("BTCUSDT", -1, 1000.0, int(tape.open_ns[36]) + FILL_DELAY, int(tape.ts[-1]))]


def test_a_1d_hold_pays_the_settlements_after_its_entry_fill_booked_on_their_day() -> None:
    # long 1 bar from 2024-01-02: the fill is 01-02 00:00:59.999, the exit fill 01-03
    # 00:00:59.999 -> settlements 01-02 08:00, 01-02 16:00 and 01-03 00:00 (0.3 each; the old
    # window charged 01-03 00:00, 08:00 and 16:00); the first two book on the 01-02 close,
    # the third on the exit bar's day
    tape = _tape([100.0] * 4, tf_ns=DAY, timeframe="1d")
    costs = tn.TradeCosts(fee=0.0, funding=_funding)
    spans = [_span(1, 1.0, 1, 1)]
    got = tn.sim_daily(tape, spans, costs, B0)
    assert list(got.values) == pytest.approx(
        [0.0, 999.4 / 1000.0 - 1.0, 999.1 / 999.4 - 1.0, 0.0], rel=1e-12
    )
    assert got == _reference(tape, spans, costs)


# --------------------------------------------------------------------------- cost inputs (.28)


def _archive(
    monkeypatch: pytest.MonkeyPatch,
    calc: FundingCalculator,
    rows: Sequence[tuple[int, float, int]],
) -> None:
    """Back ``calc`` with ``(settlement_ns, rate, interval hours)`` rows, no archive file."""
    times = [r[0] for r in rows]
    rates = [r[1] for r in rows]
    hours = [r[2] for r in rows]
    monkeypatch.setattr(calc, "_load_rates", lambda _sym: (times, rates))
    monkeypatch.setattr(calc, "_load_intervals", lambda _sym: hours)


def _rows(start: str, end: str, step_h: int = 8) -> list[tuple[int, float, int]]:
    out: list[tuple[int, float, int]] = []
    for k, t in enumerate(range(_ns(start), _ns(end), step_h * 3_600 * 10**9)):
        out.append((t, 0.0001 * (1 + k % 7) - 0.0002 * (k % 3 == 0), step_h))
    return out


def _with_costs(tape: tn.WindowTape, calc: FundingCalculator, tick: float = 0.1) -> tn.WindowTape:
    iid = f"{tape.symbol}-PERP.BINANCE"
    ts, unit = tn.funding_table(calc, iid, tape.start_ns, int(tape.ts[-1]))
    return dataclasses.replace(tape, tick=tick, funding_ts=ts, funding_unit=unit)


def _no_funding(tape: tn.WindowTape, tick: float | None = 0.1) -> tn.WindowTape:
    """``tape`` with a tick and an empty (but present) funding table."""
    return dataclasses.replace(
        tape, tick=tick, funding_ts=np.zeros(0, np.int64), funding_unit=np.zeros(0)
    )


def test_funding_table_slices_equal_accrue_with_holes(monkeypatch: pytest.MonkeyPatch) -> None:
    """R5: the table hook pays EXACTLY what ``FundingCalculator.accrue`` charges the same hold
    (ts and amounts), through a 3-day archive hole, a 4h-interval stretch and the rows around
    the window."""
    rows = [
        r
        for r in _rows("2024-01-01", "2024-02-10")
        if not _ns("2024-01-08") <= r[0] < _ns("2024-01-11")  # interior hole
        and not _ns("2024-01-14") <= r[0] < _ns("2024-01-16")  # replaced below by 4h rows
    ]
    rows += _rows("2024-01-14", "2024-01-16", step_h=4)
    rows.sort()
    calc = FundingCalculator("/nonexistent/raw_data.db")
    _archive(monkeypatch, calc, rows)
    tape = _tape([100.0] * 120, start="2024-01-05")  # 2024-01-05 .. 2024-01-25
    costs = tn.capture_costs(0.0005, {"BTCUSDT": [_with_costs(tape, calc)]}, 0.5)
    assert costs.funding is not None
    rng = np.random.default_rng(4)
    fallbacks = charged = 0
    for _ in range(300):
        i = int(rng.integers(0, 118))
        xb = int(rng.integers(i + 1, 120))
        side = int(rng.choice([-1, 1]))
        notional = float(rng.uniform(10, 900))
        entry, exit_ = int(tape.open_ns[i]) + FILL_DELAY, int(tape.open_ns[xb]) + FILL_DELAY
        want = calc.accrue(
            "BTCUSDT-PERP.BINANCE", "LONG" if side > 0 else "SHORT", notional, entry, exit_
        )
        assert costs.funding("BTCUSDT", side, notional, entry, exit_) == want.payments
        fallbacks += want.fallback_settlements
        charged += len(want.payments)
    assert fallbacks > 0 and charged > 300  # the hole and the edges were really crossed


def test_each_hold_gets_its_own_windows_settlements_and_only_those(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NULL-4: two eval windows of one symbol; a hold in window 2 receives only window 2's
    settlements (== accrue on the same span), a hold in window 1 only window 1's, and the
    slice keeps accrue's (entry, exit] edges."""
    calc = FundingCalculator("/nonexistent/raw_data.db")
    _archive(monkeypatch, calc, _rows("2024-01-01", "2024-03-01"))
    w1 = _with_costs(_tape([100.0] * 60, start="2024-01-02"), calc)  # 01-02 .. 01-12
    w2 = _with_costs(_tape([100.0] * 60, start="2024-02-01"), calc)  # 02-01 .. 02-11
    assert w1.funding_ts is not None and w2.funding_ts is not None
    assert len(w1.funding_ts) > 0 and len(w2.funding_ts) > 0
    costs = tn.capture_costs(0.0005, {"BTCUSDT": [w1, w2]}, 0.5)
    assert costs.funding is not None
    for tape, lo, hi in ((w2, 3, 40), (w1, 5, 20)):
        entry, exit_ = int(tape.open_ns[lo]) + FILL_DELAY, int(tape.open_ns[hi]) + FILL_DELAY
        got = costs.funding("BTCUSDT", -1, 250.0, entry, exit_)
        assert got == calc.accrue("BTCUSDT-PERP.BINANCE", "SHORT", 250.0, entry, exit_).payments
        assert got and all(tape.start_ns < t <= int(tape.ts[-1]) for t, _ in got)
    # edges: a settlement exactly at the entry is not charged, one exactly at the exit is
    ts = [int(t) for t in w2.funding_ts[:3]]
    assert [t for t, _ in costs.funding("BTCUSDT", 1, 1.0, ts[0], ts[2])] == ts[1:]
    assert [t for t, _ in costs.funding("BTCUSDT", 1, 1.0, ts[0] - 1, ts[1])] == ts[:2]
    assert costs.funding("BTCUSDT", 1, 1.0, ts[0], ts[0]) == []
    # window membership: the window's first instant belongs to it, the end and the gap do not
    assert costs.funding("BTCUSDT", 1, 1.0, w2.start_ns, w2.start_ns + 1) == []
    assert costs.funding("BTCUSDT", 1, 1.0, w1.end_ns - 1, w1.end_ns) == []
    with pytest.raises(tn.TimingNullError, match="no funding window"):
        costs.funding("BTCUSDT", 1, 1.0, w1.end_ns, w1.end_ns + 1)
    with pytest.raises(tn.TimingNullError, match="no funding window"):
        costs.funding("BTCUSDT", 1, 1.0, w1.start_ns - 1, w1.start_ns)


def test_expected_tick_slippage_per_market_leg() -> None:
    """R8: a market fill moves half a tick against the trader (NT's prob_slippage 0.5 as a
    mean); a carried hold's run-end mark has no exit fill, so no exit slippage."""
    opens = [50000.0] * 6 + [51000.0] * 12
    tape = _no_funding(_tape(opens))
    costs = tn.capture_costs(0.0, {"BTCUSDT": [tape]}, 0.5)
    assert costs.fee == 0.0 and costs.fill_price is not None
    # long: buys 50000.05, sells 50999.95; short: sells 49999.95, buys 51000.05
    assert costs.fill_price("BTCUSDT", 1, 50000.0) == pytest.approx(50000.05, rel=1e-15)
    assert costs.fill_price("BTCUSDT", -1, 51000.0) == pytest.approx(50999.95, rel=1e-15)
    long_ = tn.sim_daily(tape, [_span(1, 1.0, 2, 4)], costs, B0)
    short = tn.sim_daily(tape, [_span(-1, 1.0, 2, 4)], costs, B0)
    carried = tn.sim_daily(tape, [_span(1, 1.0, 2, 16, True)], costs, B0)
    assert tn.total_return(long_.values) == pytest.approx(0.01999798000201988, rel=1e-12)
    assert tn.total_return(short.values) == pytest.approx(-0.020002020002020122, rel=1e-12)
    assert tn.total_return(carried.values) == pytest.approx(0.019998980001019937, rel=1e-12)


def test_slippage_uses_each_symbols_own_tick() -> None:
    """One ``TradeCosts`` carries per-coin ticks: BTC 0.1, ETH 0.01, SOL 0.001."""
    tapes = {
        sym: [_no_funding(_tape([100.0] * 4, symbol=sym), tick)]
        for sym, tick in (("BTCUSDT", 0.1), ("ETHUSDT", 0.01), ("SOLUSDT", 0.001))
    }
    fill = tn.capture_costs(0.0005, tapes, 0.5).fill_price
    assert fill is not None
    assert fill("ETHUSDT", 1, 3000.0) == pytest.approx(3000.005, rel=1e-15)
    assert fill("SOLUSDT", -1, 150.0) == pytest.approx(149.9995, rel=1e-15)
    assert fill("BTCUSDT", -1, 60000.0) == pytest.approx(59999.95, rel=1e-15)


def test_cost_inputs_fail_closed() -> None:
    """R9: a tape without its tick or its funding table rejects (never a silent zero cost);
    the tick is checked first."""
    ok = _no_funding(_tape([100.0] * 4))
    no_table = dataclasses.replace(ok, funding_ts=None, funding_unit=None)
    with pytest.raises(tn.TimingNullError) as err:
        tn.capture_costs(0.0005, {"BTCUSDT": [dataclasses.replace(ok, tick=None)]}, 0.5)
    assert str(err.value) == "Timing null: no price increment for BTCUSDT"
    with pytest.raises(tn.TimingNullError) as err:
        tn.capture_costs(0.0005, {"BTCUSDT": [ok, no_table]}, 0.5)
    assert str(err.value) == "Timing null: no funding table for BTCUSDT window 2"
    both = dataclasses.replace(no_table, tick=None, symbol="ETHUSDT")
    with pytest.raises(tn.TimingNullError) as err:
        tn.capture_costs(0.0005, {"ETHUSDT": [both]}, 0.5)
    assert str(err.value) == "Timing null: no price increment for ETHUSDT"
    # a zero, negative or non-finite tick would price every fill at the reference
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(tn.TimingNullError, match="no price increment"):
            tn.capture_costs(0.0005, {"BTCUSDT": [dataclasses.replace(ok, tick=bad)]}, 0.5)


# --------------------------------------------------------------------------- exposure fallback (B8, GATES-7)


def test_exposure_above_the_cap_is_refused_before_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tape = _tape([100.0] * 10_000)
    runs = _check(
        {"BTCUSDT": [tape]},
        {"BTCUSDT": [[_span(1, 0.5, 0, 4500), _span(-1, 0.5, 5000, 4500)]]},
        draws=2,
    )
    assert (runs.exposure, runs.fallback, runs.draws) == (0.9, None, 2)  # R2: strict >
    half = {"BTCUSDT": [[_span(1, 0.5, 0, 5000)]]}
    assert _check({"BTCUSDT": [tape]}, half, draws=2).fallback is None

    def boom(*_a: object) -> None:
        raise AssertionError("place_trades called on an untestable book")

    monkeypatch.setattr(tn, "place_trades", boom)
    over = {"BTCUSDT": [[_span(1, 0.5, 0, 4501), _span(-1, 0.5, 5000, 4500)]]}
    refused = _check({"BTCUSDT": [tape]}, over)
    # R1: no percentile, no draws, a refusal text, never a pass
    assert (refused.exposure, refused.percentile, refused.draws) == (0.9001, None, 0)
    assert refused.fallback_passed is False
    assert refused.fallback is not None and "untestable" in refused.fallback
    assert refused.null_median is None and refused.null_p95 is None
    # a tighter cap refuses a book the default cap would test
    tight = _check({"BTCUSDT": [tape]}, half, exposure_max=0.4999)
    assert (tight.exposure, tight.draws, tight.fallback_passed) == (0.5, 0, False)
    policy = tn.TimingNullPolicy(enforced=True, draws=500, pct=0.95, exposure_max=0.9, base_seed=7)
    assert tn.timing_null_decision(policy, refused.as_dict()) == (
        False,
        ["Timing null: exposure 0.9001 > 0.90, timing untestable (not certified)"],
    )


def test_mixed_exposure_of_equally_covered_coins_is_exactly_the_cap() -> None:
    tapes = {"BTCUSDT": [_tape([100.0] * 200)], "ETHUSDT": [_tape([100.0] * 200, symbol="ETHUSDT")]}
    spans = {"BTCUSDT": [[_span(1, 0.5, 0, 200)]], "ETHUSDT": [[_span(-1, 0.5, 20, 160)]]}
    rec = _check(tapes, spans, draws=3)
    assert rec.exposure == 0.9 and rec.exposure_by_symbol == {"BTCUSDT": 1.0, "ETHUSDT": 0.8}
    assert rec.fallback is None and rec.draws == 3  # 0.90 is not > 0.90: the null runs


# --------------------------------------------------------------------------- joint re-timing (.23)

BOOK = "BTCUSDT+ETHUSDT"
Layout = dict[str, list[tuple[int, int]]]


def _layouts(
    spans: dict[str, list[list[tn.TradeSpan]]], bars: int = 60, draws: int = 30
) -> list[Layout]:
    """Per draw, each symbol's placed ``(entry_bar, hold)`` pairs, read off the sim calls."""
    tapes = {s: [_tape([100.0] * bars, symbol=s)] for s in spans}
    seen: list[tuple[str, list[tuple[int, int]]]] = []
    real_sim = tn.sim_daily

    def spy(
        tape: tn.WindowTape, placed: Sequence[tn.TradeSpan], costs: tn.TradeCosts, b0: float
    ) -> DailyReturns:
        seen.append((tape.symbol, [(sp.entry_bar, sp.hold_bars) for sp in placed]))
        return real_sim(tape, placed, costs, b0)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tn, "sim_daily", spy)
        _check(tapes, spans, draws=draws)
    by_symbol = {sym: [pairs for s, pairs in seen if s == sym] for sym in spans}
    assert all(len(v) == draws for v in by_symbol.values())  # one sim per symbol per draw
    return [{sym: by_symbol[sym][d] for sym in spans} for d in range(draws)]


def _entry_gap(layouts: list[Layout], a: tuple[str, int], b: tuple[str, int]) -> set[int]:
    """Entry bar of the ``b`` trade minus the ``a`` trade (each named by symbol and hold)."""
    gaps: set[int] = set()
    for layout in layouts:
        (ea,) = [e for e, h in layout[a[0]] if h == a[1]]
        (eb,) = [e for e, h in layout[b[0]] if h == b[1]]
        gaps.add(eb - ea)
    return gaps


def test_coinciding_trades_move_together() -> None:
    mix = [_span(1, 0.5, 10, 6), _span(-1, 0.3, 30, 4)]
    layouts = _layouts({"BTCUSDT": [mix], "ETHUSDT": [mix]}, draws=20)
    for layout in layouts:
        assert layout["BTCUSDT"] == layout["ETHUSDT"]  # one unit per trade: both coins, same bars
        assert sorted(h for _, h in layout["BTCUSDT"]) == [4, 6]
    # the placement is random, not a fixed layout, and the two units trade places
    assert len({tuple(layout["BTCUSDT"]) for layout in layouts}) > 5
    assert {layout["BTCUSDT"][0][1] for layout in layouts} == {4, 6}


def test_overlapping_trades_across_symbols_form_one_unit() -> None:
    spans = {
        "BTCUSDT": [[_span(1, 0.5, 10, 6)]],
        "ETHUSDT": [[_span(-1, 0.4, 12, 8), _span(1, 0.3, 30, 2)]],
    }
    layouts = _layouts(spans)
    assert _entry_gap(layouts, ("BTCUSDT", 6), ("ETHUSDT", 8)) == {2}  # one unit of length 10
    assert len(_entry_gap(layouts, ("BTCUSDT", 6), ("ETHUSDT", 2))) > 5  # its own unit
    for layout in layouts:  # the 10-bar unit and the 2-bar unit never overlap in book time
        (b,) = [e for e, _ in layout["BTCUSDT"]]
        (e2,) = [e for e, h in layout["ETHUSDT"] if h == 2]
        assert e2 >= b + 10 or e2 + 2 <= b


# (symbol/hold of the first trade, of the second, spans, whether they are one unit)
JOIN_EDGES = [
    pytest.param(
        ("BTCUSDT", 5), ("ETHUSDT", 3),
        {"BTCUSDT": [[_span(1, 0.5, 0, 5)]], "ETHUSDT": [[_span(1, 0.5, 5, 3)]]},
        None, id="adjacent-bars-stay-separate",
    ),
    pytest.param(
        ("BTCUSDT", 5), ("ETHUSDT", 4),
        {"BTCUSDT": [[_span(1, 0.5, 0, 5)]], "ETHUSDT": [[_span(1, 0.5, 4, 4)]]},
        4, id="one-shared-bar-joins",
    ),
    pytest.param(
        ("ETHUSDT", 3), ("ETHUSDT", 2),
        {
            "BTCUSDT": [[_span(1, 0.5, 5, 0), _span(1, 0.5, 5, 3)]],
            "ETHUSDT": [[_span(1, 0.5, 4, 3), _span(1, 0.5, 8, 2)]],
        },
        4, id="pushed-end-9-joins-entry-8",
    ),
    pytest.param(
        ("ETHUSDT", 3), ("ETHUSDT", 2),
        {
            "BTCUSDT": [[_span(1, 0.5, 5, 0), _span(1, 0.5, 5, 3)]],
            "ETHUSDT": [[_span(1, 0.5, 4, 3), _span(1, 0.5, 9, 2)]],
        },
        None, id="pushed-end-9-leaves-entry-9-separate",
    ),
]  # fmt: skip


@pytest.mark.parametrize(("a", "b", "spans", "joined"), JOIN_EDGES)
def test_units_join_on_overlap_not_adjacency_and_on_pushed_ends(
    a: tuple[str, int],
    b: tuple[str, int],
    spans: dict[str, list[list[tn.TradeSpan]]],
    joined: int | None,
) -> None:
    gaps = _entry_gap(_layouts(spans), a, b)
    assert gaps == {joined} if joined is not None else len(gaps) > 5


def test_zero_hold_clip_pushes_the_same_symbol_right() -> None:
    # ETH [4,7); BTC zero-hold at 5 (clipped [5,6)); BTC re-entry [5,8) pushed to [6,9); ETH [8,10)
    # joins the pushed end (8 < 9): ONE unit, ETH 0..3, BTC 1..2, BTC 2..5, ETH 4..6
    spans = {
        "BTCUSDT": [[_span(1, 0.5, 5, 0), _span(1, 0.5, 5, 3)]],
        "ETHUSDT": [[_span(-1, 0.4, 4, 3), _span(1, 0.3, 8, 2)]],
    }
    for layout in _layouts(spans, bars=24):
        u = layout["ETHUSDT"][0][0]
        assert sorted(layout["ETHUSDT"]) == [(u, 3), (u + 4, 2)]
        assert sorted(layout["BTCUSDT"]) == [(u + 1, 1), (u + 2, 3)]  # never overlapping
    tapes = {s: [_tape([100.0] * 5, symbol=s)] for s in spans}
    with pytest.raises(tn.TimingNullError) as err:  # the unit is 6 bars, not 5 + 2
        _check(tapes, spans)
    assert str(err.value) == f"Timing null: holds need 6 bars, {BOOK} window 1 has 5"
    ok = _check({s: [_tape([100.0] * 24, symbol=s)] for s in spans}, spans, draws=2)
    assert ok.zero_hold_clipped == 1 and ok.draws == 2


def test_joint_needs_one_bar_grid() -> None:
    btc = _tape([100.0] * 60)
    eth = _tape([100.0] * 60, symbol="ETHUSDT", start="2024-01-01T04:00")
    spans = {"BTCUSDT": [[_span(1, 0.5, 2, 3)]], "ETHUSDT": [[_span(1, 0.5, 2, 3)]]}
    with pytest.raises(tn.TimingNullError) as err:
        _check({"BTCUSDT": [btc], "ETHUSDT": [eth]}, spans)
    assert str(err.value) == (
        "Timing null: symbol windows differ in window 1 (joint re-timing needs one bar grid)"
    )
    assert _check({"BTCUSDT": [btc], "ETHUSDT": [_tape([100.0] * 60, symbol="ETHUSDT")]}, spans)


def test_book_seed_drives_the_joint_draws() -> None:
    spans = {"BTCUSDT": [[_span(1, 0.5, 10, 6)]], "ETHUSDT": [[_span(1, 0.5, 30, 4)]]}
    rng = np.random.default_rng(5)
    tapes = {
        s: [_tape(100.0 + rng.normal(0, 1, 60).cumsum(), symbol=s)] for s in ("BTCUSDT", "ETHUSDT")
    }
    real = _series(tapes["BTCUSDT"][0].start_ns // DAY + 1, [0.0] * 9)

    def run(seed: int) -> tn.TimingNullRecord:
        return tn.timing_null_check(
            real, tapes, spans, costs=ZERO, b0=B0, draws=40, seeds={BOOK: seed},
            aggregate=pool_daily_returns, closes=_closes(list(tapes), real.first_day, 9),
        )  # fmt: skip

    a, b, again = run(7), run(8), run(7)
    assert a.seeds == {BOOK: 7} and b.seeds == {BOOK: 8}
    assert (a.null_median, a.null_p95) == (again.null_median, again.null_p95)
    assert (a.null_median, a.null_p95) != (b.null_median, b.null_p95)


def test_book_exposure_never_moves_the_refusal() -> None:
    # mean per-coin exposure keeps deciding (NULL-1): a book that holds something on every bar
    # through staggered coins still gets the null
    btc, eth = _tape([100.0] * 200), _tape([100.0] * 200, symbol="ETHUSDT")
    tapes = {"BTCUSDT": [btc], "ETHUSDT": [eth]}
    full = _check(
        tapes,
        {"BTCUSDT": [[_span(1, 0.5, 0, 200)]], "ETHUSDT": [[_span(-1, 0.5, 20, 160)]]},
        draws=3,
    )
    assert (full.exposure, full.book_exposure) == (0.9, 1.0)
    assert full.exposure_by_symbol == {"BTCUSDT": 1.0, "ETHUSDT": 0.8}
    assert full.fallback is None and full.draws == 3
    staggered = _check(
        tapes,
        {"BTCUSDT": [[_span(1, 0.5, 0, 100)]], "ETHUSDT": [[_span(1, 0.5, 100, 95)]]},
        draws=3,
    )
    assert (staggered.exposure, staggered.book_exposure) == (0.4875, 0.975)
    assert staggered.fallback is None and staggered.draws == 3
    assert staggered.percentile is not None
    over = _check(
        tapes,
        {"BTCUSDT": [[_span(1, 0.5, 0, 200)]], "ETHUSDT": [[_span(-1, 0.5, 19, 162)]]},
    )
    assert (over.exposure, over.book_exposure) == (0.905, 1.0)  # mean 0.905 > 0.9: refused
    assert (over.fallback_passed, over.draws) == (False, 0)


# --------------------------------------------------------------------------- data fail closed (B9, B10, GATES-7)


def _six_months(**kw: object) -> tn.WindowTape:
    n = (_ns("2024-07-01") - _ns("2024-01-01")) // H4
    return _tape([100.0] * n, **kw)  # type: ignore[arg-type]


def _bar(iso: str) -> int:
    return (_ns(iso) - _ns("2024-01-01")) // H4


def _with_price(tape: tn.WindowTape, bar: int, value: float) -> tn.WindowTape:
    opens = tape.open.copy()
    opens[bar] = value
    return dataclasses.replace(tape, open=opens)


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (
            lambda: _six_months(drop=[_bar("2024-03-10T12:00")]),
            "Timing null: bar gap BTCUSDT 4h after 2024-03-10T08:00Z",
        ),
        (
            lambda: _six_months(drop=[_bar("2024-06-30T20:00")]),
            "Timing null: no 4h bar BTCUSDT at window end 2024-06-30T20:00Z",
        ),
        (
            lambda: _six_months(drop=[0]),
            "Timing null: no 4h bar BTCUSDT at window start 2024-01-01T00:00Z",
        ),
        (
            lambda: _with_price(_six_months(), _bar("2024-03-10T08:00"), 0.0),
            "Timing null: invalid price BTCUSDT 2024-03-10T08:00Z: 0.0",
        ),
        (
            lambda: _with_price(_six_months(), _bar("2024-03-10T08:00"), float("nan")),
            "Timing null: invalid price BTCUSDT 2024-03-10T08:00Z: nan",
        ),
    ],
    ids=["interior-gap", "window-end", "leading-gap", "zero-open", "nan-open"],
)
def test_bad_bars_fail_closed(make: Callable[[], tn.WindowTape], reason: str) -> None:
    with pytest.raises(tn.TimingNullError) as err:
        _check({"BTCUSDT": [make()]}, {"BTCUSDT": [[_span(1, 0.5, 2, 3)]]})
    assert str(err.value) == reason


def test_missing_daily_close_fails_closed_without_the_buy_hold_gate() -> None:
    tapes = {s: [_tape([100.0] * 1092, symbol=s)] for s in ("BTCUSDT", "ETHUSDT")}
    spans = {s: [[_span(1, 0.5, 2, 3)]] for s in tapes}
    closes = _closes(list(tapes), _ns("2024-01-01") // DAY, 182)
    del closes["ETHUSDT"][_ns("2024-02-29") // DAY]
    with pytest.raises(tn.TimingNullError) as err:
        _check(tapes, spans, closes=closes)
    assert (
        str(err.value)
        == "Timing null: buy-and-hold record unavailable: missing close ETHUSDT 2024-02-29"
    )


def test_null_off_the_real_day_grid_fails_closed() -> None:
    tape = _tape([100.0] * 60)
    spans = {"BTCUSDT": [[_span(1, 0.5, 2, 3)]]}
    first = tape.start_ns // DAY + 1
    for real in (_series(first, [0.0] * 10), _series(first + 1, [0.0] * 9)):
        with pytest.raises(tn.TimingNullError) as err:
            _check({"BTCUSDT": [tape]}, spans, real=real)
        assert str(err.value) == "Timing null: null series off the real series' day grid"
    assert _check({"BTCUSDT": [tape]}, spans, real=_series(first, [0.0] * 9)).draws == 5


@pytest.mark.parametrize(
    ("side", "frac", "reason"),
    [
        (1, float("nan"), "frac nan"),
        (1, float("inf"), "frac inf"),
        (1, 0.0, "frac 0.0"),
        (1, -0.5, "frac -0.5"),
        (0, 0.5, "side 0"),
        (2, 0.5, "side 2"),
    ],
)
def test_invalid_trade_side_or_size_fails_closed(side: int, frac: float, reason: str) -> None:
    # a NaN size used to ruin every null draw to -1.0: a flat real series then ranked 0.9
    rng = np.random.default_rng(1)
    tape = _tape(100.0 + rng.normal(0, 1, 120).cumsum())
    spans = {"BTCUSDT": [[_span(1, 0.5, 5, 4), _span(side, frac, 40, 6)]]}
    with pytest.raises(tn.TimingNullError) as err:
        _check({"BTCUSDT": [tape]}, spans)
    assert str(err.value) == f"Timing null: invalid trade BTCUSDT window 1 bar 40: {reason}"
    tiny = {"BTCUSDT": [[_span(1, 0.5, 5, 4), _span(-1, 1e-12, 40, 6)]]}
    assert _check({"BTCUSDT": [tape]}, tiny).draws == 5


def test_missing_seed_or_no_symbols_fail_closed() -> None:
    tape = _tape([100.0] * 60)
    real = _series(tape.start_ns // DAY + 1, [0.0] * 9)
    common = {
        "costs": ZERO,
        "b0": B0,
        "aggregate": tn.single_series,
        "closes": _closes(["BTCUSDT"], real.first_day, 9),
        "draws": 2,
    }
    with pytest.raises(tn.TimingNullError) as err:
        tn.timing_null_check(real, {"BTCUSDT": [tape]}, {"BTCUSDT": [[]]}, seeds={}, **common)  # type: ignore[arg-type]
    assert str(err.value) == "Timing null: no null seed for BTCUSDT"
    with pytest.raises(tn.TimingNullError) as err:
        tn.timing_null_check(real, {}, {}, seeds={}, **common)  # type: ignore[arg-type]
    assert str(err.value) == "Timing null: no symbols"
    both = {s: [tape] for s in ("BTCUSDT", "ETHUSDT")}
    with pytest.raises(tn.TimingNullError) as err:  # per-coin seeds are not the book's
        tn.timing_null_check(
            real,
            both,
            {s: [[]] for s in both},
            seeds={"BTCUSDT": 1, "ETHUSDT": 2},
            **common,  # type: ignore[arg-type]
        )
    assert str(err.value) == "Timing null: no null seed for BTCUSDT+ETHUSDT"


def test_overlapping_spans_are_refused_by_the_sim() -> None:
    tape = _tape([100.0] * 18)
    with pytest.raises(ValueError, match="overlaps or leaves the tape"):
        tn.sim_daily(tape, [_span(1, 0.5, 10, 8), _span(1, 0.5, 17, 1)], ZERO, B0)
    with pytest.raises(ValueError, match="overlaps or leaves the tape"):
        tn.sim_daily(tape, [_span(1, 0.5, 2, 5), _span(1, 0.5, 6, 1)], ZERO, B0)


def test_zero_hold_is_clipped_and_an_overfull_window_rejects() -> None:
    w1, w2 = _tape([100.0] * 240), _tape([100.0] * 240, start="2024-02-10")
    ok = _check(
        {"BTCUSDT": [w1, w2]}, {"BTCUSDT": [[_span(1, 0.5, 4, 0)], [_span(1, 0.5, 4, 3)]]}, draws=2
    )
    assert ok.zero_hold_clipped == 1 and ok.exposure == 4 / 480
    over = {"BTCUSDT": [[_span(1, 0.5, 0, 3)], [_span(1, 0.5, 0, 200), _span(-1, 0.5, 200, 41)]]}
    with pytest.raises(tn.TimingNullError) as err:
        _check({"BTCUSDT": [w1, w2]}, over)
    assert str(err.value) == "Timing null: holds need 241 bars, BTCUSDT window 2 has 240"
    # two coins: units of 29 and 32 bars (BTC's zero-hold clip pushes its re-entry right one bar)
    both = {s: [_tape([100.0] * 60, symbol=s)] for s in ("BTCUSDT", "ETHUSDT")}
    full = {
        "BTCUSDT": [[_span(1, 0.5, 29, 0), _span(1, 0.5, 29, 31)]],
        "ETHUSDT": [[_span(-1, 0.5, 0, 29)]],
    }
    with pytest.raises(tn.TimingNullError) as err:
        _check(both, full)
    assert str(err.value) == f"Timing null: holds need 61 bars, {BOOK} window 1 has 60"


# --------------------------------------------------------------------------- alpha / beta (B18)


@pytest.mark.parametrize(
    ("slope", "icpt", "alpha_ann"), [(0.5, 0.001, 0.365), (-1.5, -0.002, -0.73)]
)
def test_alpha_beta_on_the_real_series_days(slope: float, icpt: float, alpha_ann: float) -> None:
    tape = _tape([100.0] * 60)
    first = tape.start_ns // DAY + 1
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.02, 9)
    closes = {"BTCUSDT": {first - 1: 100.0}}
    for k, r in enumerate(x):
        closes["BTCUSDT"][first + k] = closes["BTCUSDT"][first + k - 1] * (1.0 + r)
    bh = np.array(
        [closes["BTCUSDT"][first + k] / closes["BTCUSDT"][first + k - 1] - 1.0 for k in range(9)]
    )
    real = _series(first, list(icpt + slope * bh))
    rec = _check(
        {"BTCUSDT": [tape]}, {"BTCUSDT": [[_span(1, 0.5, 2, 3)]]}, real=real, closes=closes
    )
    assert rec.beta == pytest.approx(slope, rel=1e-9)
    assert rec.alpha_ann == pytest.approx(alpha_ann, rel=1e-9)
    assert rec.alpha_note is None


def test_alpha_beta_degenerate_when_benchmark_is_flat() -> None:
    tape = _tape([100.0] * 60)
    rec = _check({"BTCUSDT": [tape]}, {"BTCUSDT": [[_span(1, 0.5, 2, 3)]]}, closes=None)
    flat = _check(
        {"BTCUSDT": [tape]},
        {"BTCUSDT": [[_span(1, 0.5, 2, 3)]]},
        closes=_closes(["BTCUSDT"], tape.start_ns // DAY + 1, 9, noisy=False),
    )
    assert rec.alpha_note is None and rec.beta is not None and math.isfinite(rec.beta)
    assert (flat.alpha_ann, flat.beta, flat.alpha_note) == (None, None, "degenerate regression")


def test_record_keys_and_seeds() -> None:
    rec = _check({"BTCUSDT": [_tape([100.0] * 60)]}, {"BTCUSDT": [[_span(1, 0.5, 2, 3)]]}, draws=2)
    assert set(rec.as_dict()) == {
        "percentile",
        "draws",
        "seeds",
        "exposure",
        "exposure_by_symbol",
        "book_exposure",
        "carried",
        "zero_hold_clipped",
        "real_T",
        "null_median",
        "null_p95",
        "fallback",
        "fallback_passed",
        "alpha_ann",
        "beta",
        "alpha_note",
        "fee",
    }
    assert rec.seeds == {"BTCUSDT": 7} and rec.book_exposure == 3 / 60
    pair = {s: [_tape([100.0] * 60, symbol=s)] for s in ("BTCUSDT", "ETHUSDT")}
    two = _check(pair, {s: [[_span(1, 0.5, 2, 3)]] for s in pair}, draws=2)
    assert two.seeds == {BOOK: 7} and two.exposure == 0.05 and two.book_exposure == 0.05
    # sha256("<base>|<dsl_key>|<sym>")[:8], big-endian
    assert tn.null_seed(20261122, "abc", "BTCUSDT") == 13677930538711399009
    assert tn.null_seed(1, "k2", "ETHUSDT") == 9744413252148747248
