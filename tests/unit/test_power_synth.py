"""Tests for the L1 synthetic trade/return generator (scripts/research/power/synth.py).

Task T1a of job 20261017-power-paper; contract rows R5, R5b, R6, R7, R8, R9 of
data/swarm/20261017-power-paper/design-power.md. R6 (the generator's population
Sharpe is the label) is asserted by the CLOSED FORM in
``test_chained_active_days_and_population_sharpe`` rather than by pooling 4000
Monte-Carlo reps: the exact reading is the stronger proof and costs nothing.

R5/R5b pin the chief's W-R2 values (the chained series length T = 875 for W3 /
1689 for W6), which override the design's T_train-based literals (876/1690).
W-R2b pins the round-2 exactness requirement: the series the gates see carries
exactly ``n*hold`` active days, and the holdout is generated at the train rate.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from vibe_quant.metrics import daily_balance_returns
from vibe_quant.validation.extraction import sharpe_sortino_from_returns

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))

import synth  # noqa: E402


def test_hold_exposure_and_mu_at_the_chained_length() -> None:
    # R5/R5b (W-R2): hold = max(1, round(0.4*T/n)); f = n*hold/T with T the chained
    # length; mu_for_sharpe(S, f_real, 0.012) recomputed at that exposure.
    assert synth.hold_for(70, 875) == 5
    assert synth.hold_for(150, 875) == 2
    assert synth.hold_for(500, 875) == 1
    assert synth.hold_for(70, 1689) == 10
    assert synth.hold_for(150, 1689) == 5

    assert synth.realized_exposure(70, 5, 875) == 0.4
    assert synth.realized_exposure(150, 2, 875) == 0.34285714285714286
    assert synth.realized_exposure(500, 1, 875) == 0.5714285714285714
    assert synth.realized_exposure(70, 10, 1689) == 0.4144464179988159
    assert synth.realized_exposure(150, 5, 1689) == 0.44404973357015987

    cases = [
        (1.2, 0.4, 0.0011952944893735414),
        (1.2, 0.34285714285714286, 0.0012921343999060216),
        (1.2, 0.5714285714285714, 0.0009985700922125256),
        (1.2, 0.4144464179988159, 0.0011740744357494068),
        (1.2, 0.44404973357015987, 0.0011339019107760495),
        (2.0, 0.34285714285714286, 0.0021682925396065526),
        (2.0, 0.44404973357015987, 0.0018982307744029728),
    ]
    for S, f_real, expected in cases:
        mu = synth.mu_for_sharpe(S, f_real, 0.012)
        assert abs(mu - expected) <= 1e-12 * max(1.0, abs(expected)), (S, f_real, mu, expected)
        # The closed form is the real check: the zero-inflated series at f_real
        # has population daily Sharpe exactly S/sqrt(365).
        daily = synth.zero_inflated_daily_sharpe(f_real, mu, 0.012)
        assert abs(daily - S / math.sqrt(365.0)) <= 1e-12 * max(1.0, S / math.sqrt(365.0))

    assert synth.mu_for_sharpe(0.0, 0.4, 0.012) == 0.0
    assert synth.mu_for_sharpe(0.0, 0.44404973357015987, 0.012) == 0.0


#: R5 cells of the contract: (plan, train trade count).
R5_CELLS: list[tuple[str, int]] = [
    ("W3", 70),
    ("W3", 150),
    ("W3", 500),
    ("W6", 70),
    ("W6", 150),
]


def test_chained_active_days_and_population_sharpe() -> None:
    # F1 (W-R2b): the series the gates see -- the PUBLIC train_mtm_values after
    # the real chaining -- carries exactly n*hold active days, and the
    # closed-form population Sharpe at that realized f equals the label to
    # 1e-12, for every R5 cell. Round 1 scheduled activity on the first window's
    # calendar day 0, which _chain_window_series drops: 299/875, not 300/875.
    label = 1.2
    target = label / math.sqrt(365.0)
    for plan, n in R5_CELLS:
        for seed in (20261017, 7):
            sched = synth.build_schedule(plan, n, label, seed)
            T = synth.PLANS[plan].chained_days
            hold = synth.hold_for(n, T)
            f_real = synth.realized_exposure(n, hold, T)
            assert sched.hold == hold
            assert sched.f_real == f_real
            assert sched.mu == synth.mu_for_sharpe(label, f_real)
            values = synth.train_mtm_values(sched)
            assert values.size == T
            assert int(np.count_nonzero(values)) == n * hold
            closed = synth.zero_inflated_daily_sharpe(f_real, sched.mu)
            assert abs(closed - target) <= 1e-12

    # W3 n150 at the round-1 exposure reads 1.1979904533781507; at the realized
    # 300/875 it reads the label.
    assert (
        abs(
            synth.zero_inflated_daily_sharpe(299 / 875, synth.mu_for_sharpe(1.2, 300 / 875))
            * math.sqrt(365.0)
            - 1.1979904533781507
        )
        <= 1e-12
    )
    assert (
        abs(
            synth.zero_inflated_daily_sharpe(300 / 875, synth.mu_for_sharpe(1.2, 300 / 875))
            * math.sqrt(365.0)
            - 1.2
        )
        <= 1e-12
    )


def test_holdout_generation_matches_train_rate() -> None:
    # F2 (W-R2b): the holdout emits trades at the train rate (trades per day)
    # and the train hold, over the holdout window, with the same exposure/Sharpe
    # exactness on the public holdout output. Round 1 emitted 0 trades across
    # the 220 W3 holdout days (a train-only schedule read on the holdout span).
    # Cases: (plan, train n, holdout days, holdout trades, hold).
    label = 1.2
    target = label / math.sqrt(365.0)
    cases = [("W3", 150, 220, 38, 2), ("W6", 70, 423, 18, 10)]
    for plan, n, holdout_days, n_holdout, hold in cases:
        gen = synth.holdout(plan, n, label, 11)
        assert gen.plan.name == plan
        assert gen.hold == hold == synth.hold_for(n, synth.PLANS[plan].chained_days)
        assert gen.n == n_holdout
        assert gen.n == synth.holdout_trades(n, synth.PLANS[plan].chained_days, holdout_days)
        assert gen.f_real == synth.realized_exposure(n_holdout, hold, holdout_days)
        assert gen.mu == synth.mu_for_sharpe(label, gen.f_real)
        assert gen.window.window == synth.PLANS[plan].holdout
        series = gen.window.bt["mtm_daily_returns"]
        assert len(series) == holdout_days
        values = np.frombuffer(series.values, dtype=np.float64)
        assert values[0] == 0.0  # first calendar day of the window stays flat
        assert int(np.count_nonzero(values)) == n_holdout * hold
        assert abs(synth.zero_inflated_daily_sharpe(gen.f_real, gen.mu) - target) <= 1e-12
        assert gen.window.bt["total_trades"] == n_holdout == len(gen.window.trades)
        # the R13 holdout gate reads >= 6 trades; round 1 gave it 0
        assert gen.window.bt["total_trades"] >= 6


def _mtm(window: synth.WindowGen) -> np.ndarray:
    return np.frombuffer(window.bt["mtm_daily_returns"].values, dtype=np.float64)


def test_window_edges_and_chain_length() -> None:
    # R7: inclusion by entry day; open at we-1 closed there; series includes day ws;
    # the two chained W3 eval windows give length 875.
    ws, we = synth.day_index("2024-01-01"), synth.day_index("2024-01-11")
    assert synth.window_spans([ws], 5, ws, we) == [(ws, ws + 4)]  # entering ON ws included
    assert synth.window_spans([we], 5, ws, we) == []  # entering ON we excluded
    assert synth.window_spans([we - 1], 5, ws, we) == [(we - 1, we - 1)]  # closed at we-1

    generated = synth.generate("W3", 150, 1.2, 7)
    for window in generated.windows:
        start, end = window.window
        series = window.bt["mtm_daily_returns"]
        assert series.first_day == synth.day_index(start)  # includes day ws
        assert len(series) == synth.day_index(end) - synth.day_index(start)
    chained = generated.bt["mtm_daily_returns"]
    assert chained is not None
    assert len(chained) == 875


def test_mtm_equals_realized_pnl() -> None:
    # R8: per window, compounded MTM == summed trade PnL == 1000*total_return;
    # total_trades == len(trade_returns); flat days are exactly 0.0.
    generated = synth.generate("W3", 150, 1.2, 11)
    for window in generated.windows:
        mtm = _mtm(window)
        compounded = 1000.0 * (float(np.prod(1.0 + mtm)) - 1.0)
        pnls = sum(t.pnl for t in window.trades)
        assert abs(compounded - pnls) <= 1e-9
        assert abs(pnls - 1000.0 * window.bt["total_return"]) <= 1e-9
        assert window.bt["total_trades"] == len(window.bt["trade_returns"]) == len(window.trades)
        active = {d for t in window.trades for d in range(t.entry_day, t.exit_day + 1)}
        first_day = window.bt["mtm_daily_returns"].first_day
        assert all(v == 0.0 for i, v in enumerate(mtm) if first_day + i not in active)


def test_sharpe_uses_nt_realized_path() -> None:
    # R9: sharpe_ratio is the realized-balance Sharpe of cash events at
    # exit_day*DAY_NS + DAY_NS//2, over two unrelated inputs.
    for n, seed in [(150, 3), (500, 4)]:
        generated = synth.generate("W3", n, 1.2, seed)
        for window in generated.windows:
            ws_day = synth.day_index(window.window[0])
            we_day = synth.day_index(window.window[1])
            expected = sharpe_sortino_from_returns(
                daily_balance_returns(
                    1000.0, window.cash_events, ws_day * synth.DAY_NS, we_day * synth.DAY_NS
                )
            )[0]
            assert window.bt["sharpe_ratio"] == expected
            for trade, (ts, pnl) in zip(window.trades, window.cash_events, strict=True):
                assert ts == trade.exit_day * synth.DAY_NS + synth.DAY_NS // 2
                assert pnl == trade.pnl
