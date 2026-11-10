"""TradingView-port OOS analysis: the random-timing null, pooled Sharpe and the pre-registered bars.

Hand-computed references use binary-exact numbers (powers of two) so the deterministic math is
checked with zero tolerance; only irrational results (square roots) use a 1e-12 relative bound.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

TV_DIR = Path(__file__).resolve().parents[2] / "scripts" / "research" / "tv_ports"
_spec = importlib.util.spec_from_file_location("oos_analysis", TV_DIR / "oos_analysis.py")
assert _spec is not None and _spec.loader is not None
oa = importlib.util.module_from_spec(_spec)
sys.modules["oos_analysis"] = oa  # dataclasses resolve their module by name
_spec.loader.exec_module(oa)

FEE = 2.0**-7  # 0.0078125, binary-exact


# --------------------------------------------------------------------------- trade arithmetic


def test_trade_multiplier_matches_hand_ledger() -> None:
    # equity 1, long notional 1 at 128 -> 160 (R = 1.25): pnl +0.25, fees FEE * (1 + 1.25)
    assert oa.trade_multiplier(1, 128.0, 160.0, FEE) == 1.25 - FEE * 2.25 == 1.232421875
    # short the same move loses the 25% and pays the same fees
    assert oa.trade_multiplier(-1, 128.0, 160.0, FEE) == 0.75 - FEE * 2.25 == 0.732421875
    assert oa.trade_multiplier(-1, 160.0, 120.0, 0.0) == 1.25


def test_compound_and_max_drawdown() -> None:
    assert oa.compound([0.5, -0.5]) == -0.25
    assert oa.compound([]) == 0.0
    # equity 1.5 -> 0.75 -> 0.9375: worst fall 0.75 / 1.5
    assert oa.max_drawdown([0.5, -0.5, 0.25]) == 0.5
    assert oa.max_drawdown([0.25, 0.5]) == 0.0


# --------------------------------------------------------------------------- null generator


def _all_placements(n_bars: int, holds: list[int]) -> set[tuple[int, ...]]:
    """Every non-overlapping in-order placement, by brute force."""
    out: set[tuple[int, ...]] = set()

    def rec(i: int, earliest: int, acc: tuple[int, ...]) -> None:
        if i == len(holds):
            out.add(acc)
            return
        for s in range(earliest, n_bars - sum(holds[i:]) + 1):
            rec(i + 1, s + holds[i], (*acc, s))

    rec(0, 0, ())
    return out


def test_place_trades_is_uniform_over_every_valid_placement() -> None:
    holds = np.array([2, 1])
    expected = _all_placements(5, [2, 1])
    assert expected == {(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4)}
    rng = np.random.default_rng(7)
    draws = 6_000
    counts: dict[tuple[int, ...], int] = {}
    for _ in range(draws):
        key = tuple(int(x) for x in oa.place_trades(5, holds, rng))
        counts[key] = counts.get(key, 0) + 1
    assert set(counts) == expected
    # each of the 6 placements ~ 1000 times; 5 sigma of Binomial(6000, 1/6) is ~144
    assert all(abs(c - draws / 6) < 150 for c in counts.values())


def test_place_trades_exact_fit_and_overfill() -> None:
    rng = np.random.default_rng(0)
    assert oa.place_trades(6, np.array([3, 1, 2]), rng).tolist() == [0, 3, 4]
    assert oa.place_trades(4, np.array([], dtype=np.int64), rng).tolist() == []
    with pytest.raises(ValueError):
        oa.place_trades(3, np.array([2, 2]), rng)


def test_place_trades_never_overlaps_on_a_long_segment() -> None:
    rng = np.random.default_rng(1)
    holds = np.array([5, 1, 9, 3, 3, 40, 2])
    for _ in range(500):
        starts = oa.place_trades(100, holds, rng)
        ends = starts + holds
        assert starts[0] >= 0 and ends[-1] <= 100
        assert np.all(starts[1:] >= ends[:-1])


def test_null_draws_match_hand_computed_orders() -> None:
    # bars open at 128, 160 and the segment ends at 120: ratios 1.25 then 0.75
    prices = np.array([128.0, 160.0, 120.0])
    holds = np.array([1, 1])
    sides = np.array([1.0, -1.0])
    long_first = (1.25 - FEE * 2.25) * (1.25 - FEE * 1.75)  # long 128->160, short 160->120
    short_first = (0.75 - FEE * 2.25) * (0.75 - FEE * 1.75)  # short 128->160, long 160->120
    assert long_first == 1.232421875 * 1.236328125
    totals, cells = oa.null_draws(
        prices, holds, sides, FEE, 200, np.random.default_rng(3), np.array([0, 1]), 2
    )
    assert set(totals.tolist()) == {long_first - 1.0, short_first - 1.0}
    for total, (c0, c1) in zip(totals, cells, strict=True):
        if total == long_first - 1.0:
            assert (c0, c1) == (0.25 - FEE * 2.25, 0.25 - FEE * 1.75)
        else:
            assert (c0, c1) == (-0.25 - FEE * 2.25, -0.25 - FEE * 1.75)


def test_null_on_flat_prices_is_the_fees() -> None:
    totals, _ = oa.null_draws(np.full(11, 64.0), np.array([3, 2]), np.array([1.0, -1.0]), FEE, 50,
                              np.random.default_rng(0))
    assert set(totals.tolist()) == {(1 - 2 * FEE) ** 2 - 1.0}


def test_null_percentile_counts_strictly_below() -> None:
    null = np.array([-0.5, 0.0, 0.0, 0.25])
    assert oa.null_percentile(0.0, null) == 0.25
    assert oa.null_percentile(0.5, null) == 1.0


# --------------------------------------------------------------------------- pooled Sharpe / bootstrap


def test_pooled_daily_averages_the_coins_present_each_day() -> None:
    days, pooled = oa.pooled_daily({"A": {1: 0.25, 2: 0.5}, "B": {2: 0.25, 3: -0.125}})
    assert days == [1, 2, 3]
    assert pooled.tolist() == [0.25, 0.375, -0.125]


def test_annual_sharpe_hand_reference() -> None:
    # mean 1/6, sample std sqrt(39)/24 -> daily Sharpe 4/sqrt(39), x sqrt(365)
    got = oa.annual_sharpe(np.array([0.25, 0.375, -0.125, 1 / 6]))
    x = [0.25, 0.375, -0.125, 1 / 6]
    mean = sum(x) / 4
    std = math.sqrt(sum((v - mean) ** 2 for v in x) / 3)
    assert math.isclose(got, mean / std * math.sqrt(365), rel_tol=1e-12)
    assert oa.annual_sharpe(np.array([0.0, 0.0, 0.0, 0.0])) == 0.0


def test_stationary_bootstrap_block_extremes() -> None:
    n, b = 7, 4
    # p = 1: every position starts a new block -> the uniform starts themselves
    idx = oa.stationary_bootstrap_indices(n, 1.0, b, np.random.default_rng(5))
    rng = np.random.default_rng(5)
    assert np.array_equal(idx, rng.integers(0, n, size=(b, n)))
    # p ~ 0: one circular block per resample
    idx = oa.stationary_bootstrap_indices(n, 1e18, b, np.random.default_rng(5))
    for row in idx:
        assert np.array_equal(row, (row[0] + np.arange(n)) % n)


def test_stationary_bootstrap_mean_block_length() -> None:
    idx = oa.stationary_bootstrap_indices(2_000, 10.0, 20, np.random.default_rng(2))
    breaks = np.sum(idx[:, 1:] != (idx[:, :-1] + 1) % 2_000)
    # new-block draws (p = 0.1) that land on the continuation index are invisible: ~0.1 * 39,980
    assert 3_700 < breaks < 4_300


def test_bootstrap_ci_brackets_a_drifting_series() -> None:
    x = np.random.default_rng(4).normal(0.004, 0.02, 1_000)
    lo, hi = oa.bootstrap_sharpe_ci(x, seed=1)
    assert lo < oa.annual_sharpe(x) < hi
    assert lo > 0  # true annual Sharpe 0.2 * sqrt(365) ~ 3.8 over 1,000 days
    assert oa.bootstrap_sharpe_ci(x[:5]) == (float("-inf"), float("inf"))


def test_alpha_beta_recovers_an_exact_line() -> None:
    x = [0.5, 0.25, -0.25, 0.125]
    y = [0.25 + 2 * v for v in x]
    alpha, beta = oa.alpha_beta(y, x)
    assert math.isclose(beta, 2.0, rel_tol=1e-12)
    assert math.isclose(alpha, 0.25 * 365, rel_tol=1e-12)


# --------------------------------------------------------------------------- segments / bars


def test_oos_segments_per_universe() -> None:
    d = oa.day_of
    assert oa.oos_segments("U1", "2023-01-01", "2026-10-20") == [
        (d("2023-01-01"), d("2026-01-01")), (d("2026-08-01"), d("2026-10-20"))
    ]
    assert oa.oos_segments("U1", "2025-05-31", "2026-10-20")[0] == (d("2025-05-31"), d("2026-01-01"))
    assert oa.oos_segments("U2", "2023-05-04", "2026-11-07") == [(d("2026-01-01"), d("2026-11-07"))]
    assert oa.oos_segments("FWD", "2023-01-01", "2026-10-20") == []
    assert oa.half_of(d("2026-06-30")) == "2026H1" and oa.half_of(d("2026-07-01")) == "2026H2"


def _coin(ret: float, null: list[float], cells: dict[str, float], null_cells: dict[str, list[float]]) -> object:
    return oa.CoinOOS(
        coin="X", ret=ret, trades=10, sharpe=0.0, pf=1.0, dd=0.0, exposure=0.5, alpha=0.0, beta=0.0,
        days=100, cells=cells, null=np.array(null), null_cells={k: np.array(v) for k, v in null_cells.items()},
        null_pct=0.0, bh=None,
    )


def test_judge_applies_the_pre_registered_bars() -> None:
    # 8 coins: 6 positive is enough, 5 is not; cell share 0.6 passes; CI needs a positive drift
    null = [-0.1, 0.0, 0.1] * 10
    coins = [_coin(0.2 if i < 6 else -0.1, null, {"2026H1": 0.1, "2026H2": 0.1 if i < 4 else -0.1},
                   {"2026H1": null, "2026H2": null}) for i in range(8)]
    series = {f"C{i}": dict(enumerate(np.random.default_rng(i).normal(0.01, 0.01, 200))) for i in range(8)}
    v = oa.judge(coins, series, seed=0)
    assert (v["need"], v["positive"], v["cells"]) == (6, 6, 16)
    assert v["cell_share"] == 12 / 16
    assert v["robust"] and v["ci_lo"] > 0
    assert v["port_ret"] == 0.125  # (6 * 0.2 - 2 * 0.1) / 8
    assert v["port_null_pct"] == 1.0 and v["beats_null"] and v["candidate"]
    losing = {f"C{i}": dict(enumerate(np.random.default_rng(i).normal(-0.01, 0.01, 200))) for i in range(8)}
    v = oa.judge(coins, losing, seed=0)
    assert v["ci_lo"] < 0 and not v["robust"] and not v["candidate"]  # coins and cells alone are not enough
    coins[5] = _coin(-0.1, null, {"2026H1": 0.1, "2026H2": 0.1}, {"2026H1": null, "2026H2": null})
    assert not oa.judge(coins, series, seed=0)["robust"]


def test_need_rounds_up_for_fewer_coins() -> None:
    null = [0.0] * 4
    coins = [_coin(0.1, null, {}, {}) for _ in range(3)]
    v = oa.judge(coins, {"A": {1: 0.0, 2: 0.0}}, seed=0)
    assert v["need"] == 3  # ceil(0.75 * 3)
    coins7 = [_coin(0.1, null, {}, {}) for _ in range(7)]
    assert oa.judge(coins7, {"A": {1: 0.0, 2: 0.0}}, seed=0)["need"] == 6  # ceil(5.25)


def test_coin_oos_on_synthetic_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    start, end = "2026-01-01", "2026-01-03"
    t0 = oa.day_of(start) * oa.DAY_NS
    hour = oa.TF_NS["1h"]
    opens = np.full(48, 64.0)
    monkeypatch.setitem(oa._BARS, ("XUSDT", "1h"), (t0 + np.arange(48) * hour, opens, opens.copy()))
    row = {
        "coin": "XUSDT", "tf": "1h", "start": start, "end": end, "first_day": oa.day_of(start),
        "daily": [0.25, -0.5],
        "positions": [{"side": 1, "ts_open": t0, "ts_close": t0 + 24 * hour, "px_open": 64.0,
                       "px_close": 80.0, "qty": 1.0, "realized_pnl": 16.0}],
    }
    c = oa.coin_oos(row, "U2", {}, {}, 20, FEE, seed=0)
    assert c is not None
    assert (c.ret, c.trades, c.days, c.exposure, c.dd) == (-0.375, 1, 2, 0.5, 0.5)
    assert c.cells == {}  # 2 OOS days < 30
    assert set(c.null.tolist()) == {(1 - 2 * FEE) - 1.0}  # flat prices: the null pays fees only
    assert c.null_pct == 0.0  # -0.375 is below every null draw
    from vibe_quant.metrics import PROFIT_FACTOR_CAP

    assert c.pf == PROFIT_FACTOR_CAP  # one winning closed trade, no loser
