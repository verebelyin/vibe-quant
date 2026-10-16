"""Daily mark-to-market return series (bd vibe-quant-wcn55, task D2-2).

The realized-balance series (``FitnessResult.daily_returns`` until it was
removed, bd vibe-quant-f9zzq) books PnL only on exit days, so a multi-day hold
shows a zero return until it closes — that inflates Sharpe vs a fair
buy-and-hold comparison. ``mark_to_market_daily_returns`` values the same UTC
day grid at each execution bar's close, so an overnight position shows its move
on the day it happens. Output only: nothing existing may move numerically.
"""

from __future__ import annotations

from array import array
from typing import Any

import numpy as np
import pytest

from vibe_quant.metrics import (
    BarSeries,
    DailyReturns,
    LedgerEvent,
    dense_daily_returns,
    mark_to_market_daily_returns,
    pnl_path,
)

DAY_NS = 86_400 * 1_000_000_000
HOUR_NS = 3_600 * 1_000_000_000
D0 = 19_000


def _daily_bars(closes: dict[int, float]) -> BarSeries:
    """One flat OHLC bar per UTC day, at 12:00 (so ``ts // DAY`` keys it)."""
    days = sorted(closes)
    ts = np.array([d * DAY_NS + 12 * HOUR_NS for d in days], dtype=np.int64)
    c = np.array([closes[d] for d in days], dtype=float)
    return BarSeries(ts, c.copy(), c.copy(), c.copy(), c)


def _path(bars: BarSeries, events: list[LedgerEvent]) -> Any:
    pnl_close, pnl_min = pnl_path(bars, events)
    return (bars.ts, pnl_close, pnl_min)


def _ns(date: str) -> int:
    from vibe_quant.validation.extraction import date_to_ns

    ns = date_to_ns(date)
    assert ns is not None
    return ns


def _chrom(**kw: Any) -> Any:
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    base: dict[str, Any] = {
        "entry_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0, None)],
        "exit_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        "stop_loss_pct": 2.0, "take_profit_pct": 4.0, "direction": Direction.LONG,
    }  # fmt: skip
    base.update(kw)
    return StrategyChromosome(**base)


# --- the metric itself ------------------------------------------------------


def test_mtm_daily_marks_overnight_position() -> None:
    """A long held 2 days on a +5% move: the move lands on day 1, not day 2."""
    bars = _daily_bars({D0: 100.0, D0 + 1: 105.0, D0 + 2: 105.0})
    qty = 10.0
    events = [
        LedgerEvent(ts=int(bars.ts[0]), qty=qty, price=100.0),
        LedgerEvent(ts=int(bars.ts[2]), qty=-qty, price=105.0),
    ]
    start_ns = D0 * DAY_NS
    end_ns = (D0 + 3) * DAY_NS

    mtm = mark_to_market_daily_returns(1000.0, [_path(bars, events)], start_ns, end_ns)
    assert mtm is not None
    assert mtm.values[1] == pytest.approx(0.05)  # day 1 marks the +5% move

    # The realized series books PnL only on the exit day: day 1 is exactly 0.
    realized = dense_daily_returns(1000.0, [(int(bars.ts[2]), 50.0)], start_ns, end_ns)
    assert realized is not None
    assert realized.values[1] == 0.0


def test_mtm_daily_equals_realized_for_intraday_trades() -> None:
    """Flat at every day boundary -> the MTM series equals the realized one."""
    bars = _daily_bars({D0: 100.0, D0 + 1: 100.0})
    events = [
        LedgerEvent(ts=int(bars.ts[0]), qty=10.0, price=100.0),
        LedgerEvent(ts=int(bars.ts[0]), qty=-10.0, price=102.0),  # +20 on day 0
        LedgerEvent(ts=int(bars.ts[1]), qty=10.0, price=102.0),
        LedgerEvent(ts=int(bars.ts[1]), qty=-10.0, price=100.0),  # -20 on day 1
    ]
    start_ns = D0 * DAY_NS
    end_ns = (D0 + 2) * DAY_NS
    mtm = mark_to_market_daily_returns(1000.0, [_path(bars, events)], start_ns, end_ns)
    realized = dense_daily_returns(
        1000.0, [(int(bars.ts[0]), 20.0), (int(bars.ts[1]), -20.0)], start_ns, end_ns
    )
    assert mtm is not None and realized is not None
    assert list(mtm.values) == pytest.approx(list(realized.values), abs=1e-12)
    assert mtm.values[0] == pytest.approx(0.02)


def test_mtm_daily_same_grid_as_dense() -> None:
    start_ns = D0 * DAY_NS
    end_ns = (D0 + 5) * DAY_NS
    mtm = mark_to_market_daily_returns(1000.0, [], start_ns, end_ns)
    dense = dense_daily_returns(1000.0, [], start_ns, end_ns)
    assert mtm is not None and dense is not None
    assert mtm.first_day == dense.first_day == D0
    assert len(mtm) == len(dense) == 5


def test_mtm_no_positions_is_zero_series() -> None:
    mtm = mark_to_market_daily_returns(1000.0, [], D0 * DAY_NS, (D0 + 4) * DAY_NS)
    assert mtm is not None
    assert len(mtm) == 4
    assert all(v == 0.0 for v in mtm.values)


def test_mtm_sparse_asynchronous_instruments_carry_forward_without_lookahead() -> None:
    """Two instruments on different bar timestamps; oracle worked out by hand.

    Grid D0..D0+4, start 1000. X has bars on D0+1 (+10) and D0+3 (-20); Y has
    two bars on D0+2 (06:00 -> 30, 18:00 -> 50) and one on D0+4 (-> 0).
    Per-day total pnl (last bar with day <= d per instrument, 0 before the
    first bar, nothing from later bars):
      D0: 0+0=0 | D0+1: 10+0=10 | D0+2: 10+50=60 | D0+3: -20+50=30 | D0+4: -20+0=-20
    """

    def _p(points: list[tuple[int, int, float]]) -> Any:
        ts = np.array([d * DAY_NS + h * HOUR_NS for d, h, _ in points], dtype=np.int64)
        pc = np.array([v for _, _, v in points], dtype=float)
        return (ts, pc, pc.copy())

    x = _p([(D0 + 1, 12, 10.0), (D0 + 3, 12, -20.0)])
    y = _p([(D0 + 2, 6, 30.0), (D0 + 2, 18, 50.0), (D0 + 4, 1, 0.0)])
    mtm = mark_to_market_daily_returns(1000.0, [x, y], D0 * DAY_NS, (D0 + 5) * DAY_NS)
    assert mtm is not None
    equity = [1000.0, 1010.0, 1060.0, 1030.0, 980.0]
    expected = [equity[0] / 1000.0 - 1.0] + [b / a - 1.0 for a, b in zip(equity, equity[1:], strict=False)]
    assert mtm.first_day == D0
    assert list(mtm.values) == pytest.approx(expected, abs=1e-12)
    assert mtm.values[0] == 0.0  # before either instrument's first bar
    # Same answer whatever the instrument order (the sum is symmetric).
    swapped = mark_to_market_daily_returns(1000.0, [y, x], D0 * DAY_NS, (D0 + 5) * DAY_NS)
    assert swapped is not None and list(swapped.values) == list(mtm.values)


@pytest.mark.parametrize(
    ("start", "pnl"),
    [
        (1000.0, -1000.0),  # equity exactly zero
        (100.0, -101.0),  # negative equity (an abs(balance) bug would give [-0.99])
        (1000.0, float("nan")),
        (1000.0, float("inf")),
        (1000.0, float("-inf")),
        (0.0, 5.0),  # non-positive starting balance
        (-100.0, 5.0),
        (float("nan"), 5.0),
        (float("inf"), 5.0),
    ],
)
def test_mtm_invalid_equity_fails_closed(start: float, pnl: float) -> None:
    ts = np.array([D0 * DAY_NS + 12 * HOUR_NS], dtype=np.int64)
    path = (ts, np.array([pnl]), np.array([pnl]))
    assert mark_to_market_daily_returns(start, [path], D0 * DAY_NS, (D0 + 2) * DAY_NS) is None


def test_mtm_equity_turning_negative_on_a_later_day_fails_closed() -> None:
    """Healthy day 0 then wiped out on day 1 -> the whole series is None, not truncated."""
    ts = np.array([D0 * DAY_NS + 12 * HOUR_NS, (D0 + 1) * DAY_NS + 12 * HOUR_NS], dtype=np.int64)
    pnl = np.array([50.0, -150.0])
    assert mark_to_market_daily_returns(100.0, [(ts, pnl, pnl)], D0 * DAY_NS, (D0 + 2) * DAY_NS) is None


# --- runner output reset ----------------------------------------------------


def test_runner_failed_run_does_not_keep_previous_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused runner whose next run fails must not expose the previous series."""
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    runner = NTScreeningRunner({"name": "x"}, ["BTCUSDT"], "2024-01-01", "2024-02-01",
                               collect_daily_returns=True)  # fmt: skip
    stale = DailyReturns(D0, array("d", [0.01]))
    runner.last_mtm_daily_returns = stale

    def boom(self: Any, params: Any, start_time: float) -> Any:
        raise RuntimeError("engine blew up")

    monkeypatch.setattr(NTScreeningRunner, "_run_backtest", boom)
    result = runner({})
    assert result.sharpe_ratio == float("-inf")
    assert runner.last_mtm_daily_returns is None


def test_runner_unavailable_mtm_after_success_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real engine twice on one runner: run 2 cannot build MTM paths -> None, not run 1's series."""
    from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    bar_dir = DEFAULT_CATALOG_PATH / "data" / "bar"
    if not bar_dir.is_dir() or not any("BTCUSDT-PERP.BINANCE" in d.name for d in bar_dir.iterdir()):
        pytest.skip("BTCUSDT catalog data not available")
    start, end = "2024-01-01", "2024-02-01"
    fn = NTBacktestFn(["BTCUSDT"], "4h", start, end, collect_daily_returns=True)

    captured: dict[str, NTScreeningRunner] = {}
    real_init = NTScreeningRunner.__init__

    def spy_init(self: NTScreeningRunner, *a: Any, **kw: Any) -> None:
        real_init(self, *a, **kw)
        captured["r"] = self

    monkeypatch.setattr(NTScreeningRunner, "__init__", spy_init)
    out = fn._run_single(_chrom(stop_loss_pct=10.0), start, end)
    runner = captured["r"]
    assert out["mtm_daily_returns"] is not None
    assert runner.last_mtm_daily_returns is not None

    import vibe_quant.validation.extraction as extraction

    monkeypatch.setattr(extraction, "mark_to_market_paths", lambda *a, **kw: None)
    runner({})
    assert runner.last_mtm_daily_returns is None


# --- discovery wiring -------------------------------------------------------


def test_mtm_chain_windows_length_is_train_days_minus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worst mode, 2 windows: the per-symbol MTM series chains across the windows."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.utils import compute_day_count, split_into_windows

    def fake_single(self: NTBacktestFn, chrom: Any, s: str, e: str, symbols: Any = None) -> Any:
        n = compute_day_count(s, e) or 0
        out: dict[str, Any] = {
            "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
            "total_trades": 60, "total_return": 0.1,
        }  # fmt: skip
        if self.collect_daily_returns:
            out["mtm_daily_returns"] = DailyReturns(_ns(s) // DAY_NS, array("d", [0.002] * n))
        return out

    monkeypatch.setattr(NTBacktestFn, "_run_single", fake_single)
    start, end = "2024-01-01", "2024-07-01"
    windows = split_into_windows(start, end, 2)
    fn = NTBacktestFn(
        ["A", "B"], "4h", start, end, windows=windows, min_trades=10,
        symbol_agg="worst", collect_daily_returns=True,
    )  # fmt: skip
    out = fn(_chrom())
    n_days = compute_day_count(start, end) or 0
    assert len(out["mtm_daily_returns"]) == n_days - 1
    assert out["mtm_daily_returns"].first_day == (_ns(start) // DAY_NS) + 1  # day 0 trimmed
    assert set(out["symbol_mtm_daily"]) == {"A", "B"}


def _sym_mtm(sr: DailyReturns | None, ret: float = 0.2) -> dict[str, Any]:
    return {
        "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 100, "total_return": ret, "mtm_daily_returns": sr,
    }  # fmt: skip


def test_worst_mode_pools_and_keeps_symbol_mtm() -> None:
    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    a = DailyReturns(100, array("d", [0.01, 0.02]))
    b = DailyReturns(100, array("d", [0.03, 0.00]))
    out = NTBacktestFn._aggregate_symbols([_sym_mtm(a), _sym_mtm(b)], ["A", "B"])
    pooled = out["mtm_daily_returns"]
    assert isinstance(pooled, DailyReturns)
    assert list(pooled.values) == pytest.approx([0.02, 0.01])
    sm = out["symbol_mtm_daily"]
    assert sm["A"] == a and sm["B"] == b
    # one symbol without a series -> no pool, that symbol flagged missing
    broken = NTBacktestFn._aggregate_symbols([_sym_mtm(a), _sym_mtm(None)], ["A", "B"])
    assert broken["mtm_daily_returns"] is None and broken["symbol_mtm_daily"]["B"] is None


def test_mtm_not_in_averaged_symbol_metrics() -> None:
    """Nit (a): the per-symbol MTM series must not ride along in symbol_results."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    d = DailyReturns(100, array("d", [0.01, 0.02]))
    r = {
        "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 100, "total_return": 0.2, "mtm_daily_returns": d,
    }  # fmt: skip
    out = NTBacktestFn._aggregate_symbols([dict(r), dict(r)], ["A", "B"])
    for sr in out["symbol_results"].values():
        assert "mtm_daily_returns" not in sr
    assert out["sharpe_ratio"] == 1.0 and out["total_return"] == 0.2


def test_mtm_fields_follow_keep_series() -> None:
    """Nit (b): the mtm fields obey the keep_series rule."""
    from vibe_quant.discovery.fitness import _evaluate_single

    d = DailyReturns(100, array("d", [0.01, 0.02]))
    sym = {"A": d}
    base: dict[str, Any] = {
        "sharpe_ratio": 1.0, "max_drawdown": 0.1, "profit_factor": 1.5,
        "total_trades": 100, "total_return": 0.2,
        "mtm_daily_returns": d, "symbol_mtm_daily": sym,
    }  # fmt: skip
    kept = _evaluate_single(_chrom(), lambda _c: base, None, 50)
    assert kept.mtm_daily_returns is d and kept.symbol_mtm_daily == sym
    dropped = _evaluate_single(_chrom(), lambda _c: {**base, "total_return": -0.1}, None, 50)
    assert dropped.mtm_daily_returns is None and dropped.symbol_mtm_daily is None
    few = _evaluate_single(_chrom(), lambda _c: {**base, "total_trades": 10}, None, 50)
    assert few.mtm_daily_returns is None
