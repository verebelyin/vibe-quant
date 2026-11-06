"""Fail-closed buy-and-hold champion gate.

The gate compares a candidate's daily Sharpe against the equal-weight,
daily-rebalanced buy-and-hold of the same coins over the candidate's EXACT UTC
day set, with a direction-dependent bar (long -> bh, short -> -bh, both ->
|bh|). Pure math: catalog daily closes in, verdict out. No pipeline wiring
(that is D2-3).

``test_buy_hold_matches_h2b_reference`` is the pre-registered parity anchor: the
new module must reproduce the H-2b script's formula (pandas ``pct_change`` mean
over symbols, ``mean / std(ddof=1) * sqrt(365)``) to 1e-12.
"""

from __future__ import annotations

import math
from array import array
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from vibe_quant.discovery import buyhold
from vibe_quant.metrics import BarSeries, DailyReturns, dense_daily_returns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytest import MonkeyPatch

_DAY_NS = buyhold.DAY_NS


def _dr(first_day: int, values: Sequence[float] | np.ndarray) -> DailyReturns:
    return DailyReturns(first_day, array("d", (float(v) for v in values)))


def _prices(
    base_day: int, returns: Sequence[float] | np.ndarray, start: float = 100.0
) -> dict[int, float]:
    """Closes on ``base_day .. base_day + len(returns)`` from a return series."""
    prices = {base_day: start}
    level = start
    for i, r in enumerate(returns):
        level *= 1.0 + r
        prices[base_day + i + 1] = level
    return prices


# ---------------------------------------------------------------------------
# Parity with the H-2b pre-registered benchmark script
# ---------------------------------------------------------------------------


def test_buy_hold_matches_h2b_reference() -> None:
    """EW daily B&H Sharpe must equal the H-2b pandas formula to 1e-12."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    n_days = 400
    base = 20_000
    rng = np.random.default_rng(20261012)
    closes: dict[str, dict[int, float]] = {}
    cols: dict[str, np.ndarray] = {}
    for symbol in symbols:
        rets = rng.normal(0.0004, 0.02, size=n_days)
        prices = 1000.0 * np.cumprod(np.concatenate([[1.0], 1.0 + rets]))
        cols[symbol] = prices
        closes[symbol] = {base + k: float(prices[k]) for k in range(n_days + 1)}

    # H-2b formula, verbatim (data/swarm/20261011-h2b/buyhold.py).
    index = pd.to_datetime(np.arange(base, base + n_days + 1) * _DAY_NS, utc=True)
    frame = pd.DataFrame(cols, index=index)
    rets_frame = frame.pct_change().dropna(how="all")
    ew = rets_frame.mean(axis=1, skipna=True)
    h2b = float(ew.mean() / ew.std(ddof=1) * np.sqrt(365))

    bh, error = buyhold.buy_hold_daily(closes, symbols, base + 1, n_days)
    assert error is None
    assert bh is not None
    ours = buyhold.daily_sharpe(bh)
    assert ours is not None
    assert abs(ours * np.sqrt(365) - h2b) < 1e-12
    assert np.allclose(bh, ew.to_numpy(), atol=1e-15, rtol=0)


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------


def test_missing_close_rejects() -> None:
    """ANY missing close fails closed, naming the symbol and the UTC date."""
    closes = {"BTCUSDT": {0: 100.0, 1: 101.0, 2: 102.0}}
    values, error = buyhold.buy_hold_daily(closes, ["BTCUSDT"], first_day=1, n=3)
    assert values is None
    assert error == "missing close BTCUSDT 1970-01-04"

    # The previous day (d-1) missing for the FIRST requested day.
    values, error = buyhold.buy_hold_daily({"BTCUSDT": {0: 100.0, 2: 102.0}}, ["BTCUSDT"], 2, 1)
    assert values is None
    assert error == "missing close BTCUSDT 1970-01-02"


def test_missing_close_rejects_any_symbol() -> None:
    """One gapped symbol poisons the day, even if the others are complete."""
    closes = {
        "BTCUSDT": {10: 100.0, 11: 101.0, 12: 102.0},
        "ETHUSDT": {10: 200.0, 11: 202.0},
    }
    values, error = buyhold.buy_hold_daily(closes, ["BTCUSDT", "ETHUSDT"], first_day=11, n=2)
    assert values is None
    assert error == "missing close ETHUSDT 1970-01-13"


def test_day_alignment_uses_prev_close() -> None:
    """Day d's return uses close[d-1], never close[d+1] or close[d] itself."""
    closes = {"BTCUSDT": {5: 100.0, 6: 110.0, 7: 100.0}}
    values, error = buyhold.buy_hold_daily(closes, ["BTCUSDT"], first_day=6, n=1)
    assert error is None
    assert values == pytest.approx([110.0 / 100.0 - 1.0])

    values, error = buyhold.buy_hold_daily(closes, ["BTCUSDT"], first_day=7, n=1)
    assert error is None
    assert values == pytest.approx([100.0 / 110.0 - 1.0])


def test_equal_weight_mean_over_symbols() -> None:
    """bh_d is the plain arithmetic mean of the per-symbol daily returns."""
    closes = {
        "BTCUSDT": _prices(0, [0.02, -0.01]),
        "ETHUSDT": _prices(0, [0.01, 0.03]),
    }
    values, error = buyhold.buy_hold_daily(closes, ["BTCUSDT", "ETHUSDT"], first_day=1, n=2)
    assert error is None
    assert values == pytest.approx([0.015, 0.01])


# ---------------------------------------------------------------------------
# Daily Sharpe
# ---------------------------------------------------------------------------


def test_flat_or_short_series_rejects() -> None:
    """< 30 days, flat (std 0), non-finite and a None candidate all reject."""
    assert buyhold.daily_sharpe(np.zeros(10)) is None
    assert buyhold.daily_sharpe(np.full(40, 0.001)) is None
    bad = np.zeros(40)
    bad[3] = np.inf
    assert buyhold.daily_sharpe(bad) is None

    closes = {"BTCUSDT": _prices(0, [0.01] * 40)}
    short = _dr(1, [0.01] * 10)
    passed, _record, reasons = buyhold.buy_hold_gate(short, "long", closes, ["BTCUSDT"])
    assert passed is False
    assert any("candidate" in r for r in reasons)

    flat = _dr(1, [0.0] * 40)
    passed, _record, reasons = buyhold.buy_hold_gate(flat, "long", closes, ["BTCUSDT"])
    assert passed is False
    assert any("candidate" in r for r in reasons)

    passed, record, reasons = buyhold.buy_hold_gate(None, "long", closes, ["BTCUSDT"])
    assert passed is False
    assert record == {}
    assert reasons == ["Buy-hold: no daily series"]


def test_sharpe_is_daily_mean_over_std_ddof1() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(0.001, 0.02, size=120)
    assert buyhold.daily_sharpe(values) == pytest.approx(float(values.mean() / values.std(ddof=1)))
    # Display factor is sqrt(365) (NOT NT/DSR's sqrt(252)).
    assert buyhold.ANN_FACTOR == 365


def test_min_observations_boundary_is_30() -> None:
    """The cutoff is exactly 30: 29 values -> None, 30 -> a finite Sharpe.

    Kills the MIN_OBSERVATIONS 30 -> 29 mutant: with the bound at 29 the
    29-value series would return a Sharpe and the first assert would fail.
    """
    assert buyhold.MIN_OBSERVATIONS == 30
    rng = np.random.default_rng(13)
    below = rng.normal(0.001, 0.02, size=buyhold.MIN_OBSERVATIONS - 1)  # 29
    at = rng.normal(0.001, 0.02, size=buyhold.MIN_OBSERVATIONS)  # 30

    assert buyhold.daily_sharpe(below) is None
    sharpe = buyhold.daily_sharpe(at)
    assert sharpe is not None and math.isfinite(sharpe)
    assert sharpe == pytest.approx(float(at.mean() / at.std(ddof=1)))


def test_gate_min_observations_boundary() -> None:
    """The gate fails closed with a reason at 29 observations; 30 gives a Sharpe."""
    base = 12000
    rng = np.random.default_rng(17)
    rets = list(rng.normal(0.001, 0.02, size=31))  # 32 closes -> up to 31 daily returns
    closes = {"BTCUSDT": _prices(base, rets)}

    short = _dr(base + 1, rets[:29])
    passed, record, reasons = buyhold.buy_hold_gate(short, "long", closes, ["BTCUSDT"])
    assert passed is False
    assert record["observations"] == 29
    assert record["sharpe_daily"] is None
    assert any("candidate" in r and "T=29" in r for r in reasons)

    ok = _dr(base + 1, np.asarray(rets[:30]) + 0.004)
    passed, record, reasons = buyhold.buy_hold_gate(ok, "long", closes, ["BTCUSDT"])
    assert record["observations"] == 30
    assert isinstance(record["sharpe_daily"], float)
    assert math.isfinite(record["sharpe_daily"])
    assert isinstance(record["bh_sharpe_daily"], float)
    assert math.isfinite(record["bh_sharpe_daily"])


# ---------------------------------------------------------------------------
# Direction-dependent bar
# ---------------------------------------------------------------------------


def test_short_bar_is_negated() -> None:
    assert buyhold.direction_bar(0.8, "short") == pytest.approx(-0.8)


def test_both_bar_is_abs() -> None:
    assert buyhold.direction_bar(-0.6, "both") == pytest.approx(0.6)
    assert buyhold.direction_bar(0.6, "both") == pytest.approx(0.6)


def test_direction_bar_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="direction"):
        buyhold.direction_bar(0.5, "sideways")


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def test_equal_sharpe_rejects() -> None:
    """The comparison is STRICT: a candidate exactly at the bar fails."""
    n = 120
    base = 1000
    rng = np.random.default_rng(3)
    rets = rng.normal(0.001, 0.02, size=n)
    closes = {"BTCUSDT": _prices(base, rets)}
    bh, error = buyhold.buy_hold_daily(closes, ["BTCUSDT"], base + 1, n)
    assert error is None and bh is not None
    candidate = _dr(base + 1, bh)  # identical to its own benchmark

    passed, record, reasons = buyhold.buy_hold_gate(candidate, "long", closes, ["BTCUSDT"])
    assert passed is False
    assert record["passed"] is False
    assert record["bar_daily"] == pytest.approx(record["sharpe_daily"])
    assert len(reasons) == 1
    assert "<=" in reasons[0] and "BTCUSDT" in reasons[0] and "ann √365" in reasons[0]
    assert f"T={n}" in reasons[0]


def test_passing_candidate_and_reason_format() -> None:
    """A higher-Sharpe series passes; the record carries every documented key."""
    n = 120
    base = 3000
    rng = np.random.default_rng(11)
    bh_rets = rng.normal(0.0002, 0.02, size=n)
    closes = {"BTCUSDT": _prices(base, bh_rets), "ETHUSDT": _prices(base, bh_rets[::-1])}
    candidate = _dr(base + 1, np.asarray(bh_rets) + 0.004)  # shift mean -> higher Sharpe

    passed, record, reasons = buyhold.buy_hold_gate(
        candidate, "long", closes, ["BTCUSDT", "ETHUSDT"]
    )
    assert passed is True
    assert reasons == []
    assert record["series"] == "mtm"
    assert record["direction"] == "long"
    assert record["observations"] == n
    assert record["first_day"] == base + 1
    assert record["passed"] is True
    assert record["sharpe_ann"] == pytest.approx(record["sharpe_daily"] * np.sqrt(365))
    assert record["bh_sharpe_ann"] == pytest.approx(record["bh_sharpe_daily"] * np.sqrt(365))
    assert record["bar_ann"] == pytest.approx(record["bar_daily"] * np.sqrt(365))
    assert record["bar_daily"] == pytest.approx(record["bh_sharpe_daily"])
    assert "symbols" not in record


def test_per_symbol_bar_uses_own_symbol() -> None:
    """Worst mode: pooled passes, but a symbol under its OWN bar rejects it."""
    base = 5000
    n = 60
    r_a = np.array([0.01, 0.02] * (n // 2))  # own B&H Sharpe ~2.97 daily
    r_b = np.array([0.02, -0.02] * (n // 2))  # own B&H Sharpe ~0.0 daily
    closes = {"AAAUSDT": _prices(base, r_a), "BBBUSDT": _prices(base, r_b)}
    pooled_candidate = _dr(base + 1, (r_a + r_b) / 2.0 + 0.004)  # beats the pooled bar

    passed, record, reasons = buyhold.buy_hold_gate(
        pooled_candidate,
        "long",
        closes,
        ["AAAUSDT", "BBBUSDT"],
        per_symbol={
            "AAAUSDT": _dr(base + 1, r_a + 0.002),  # above its own bar
            "BBBUSDT": _dr(base + 1, r_b - 0.005),  # below its own bar -> reject
        },
    )
    assert passed is False
    assert len(reasons) == 1
    assert "BBBUSDT" in reasons[0]
    assert "AAAUSDT" not in reasons[0]
    symbols = record["symbols"]
    assert isinstance(symbols, dict)
    assert symbols["AAAUSDT"]["passed"] is True
    assert symbols["BBBUSDT"]["passed"] is False
    assert symbols["BBBUSDT"]["bar_daily"] == pytest.approx(symbols["BBBUSDT"]["bh_sharpe_daily"])


def test_per_symbol_missing_series_rejects() -> None:
    base = 6000
    n = 60
    r_a = np.array([0.01, 0.02] * (n // 2))
    closes = {"AAAUSDT": _prices(base, r_a)}
    candidate = _dr(base + 1, r_a + 0.002)
    passed, _record, reasons = buyhold.buy_hold_gate(
        candidate, "long", closes, ["AAAUSDT"], per_symbol={"AAAUSDT": None}
    )
    assert passed is False
    assert any("AAAUSDT" in r for r in reasons)


# ---------------------------------------------------------------------------
# Catalog bars -> day keyed closes
# ---------------------------------------------------------------------------


def test_complete_bar_day_keys(monkeypatch: MonkeyPatch) -> None:
    """Both complete forms map to their day; a partial bar is dropped."""
    seen: list[str] = []
    ts = np.array(
        [
            _DAY_NS,  # close exactly at midnight -> previous day (0)
            2 * _DAY_NS - 1_000_000,  # close at 23:59:59.999 -> day 1
            3 * _DAY_NS + 43_200 * 10**9,  # in-progress bar (noon) -> dropped
        ],
        dtype=np.int64,
    )
    close = np.array([100.0, 110.0, 120.0])
    fake = BarSeries(ts=ts, open=close, high=close, low=close, close=close)

    def _load(catalog_path: object, bar_type: str) -> BarSeries:
        seen.append(bar_type)
        return fake

    monkeypatch.setattr(buyhold, "load_catalog_bars", _load)
    got = buyhold.load_daily_closes(["BTCUSDT"], "/tmp/catalog")
    assert got == {"BTCUSDT": {0: 100.0, 1: 110.0}}
    assert seen == ["BTCUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL"]


def test_load_daily_closes_missing_bars_raises(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(buyhold, "load_catalog_bars", lambda c, b: None)
    with pytest.raises(ValueError, match="BTCUSDT"):
        buyhold.load_daily_closes(["BTCUSDT"], "/tmp/catalog")


def test_gate_benchmark_uses_candidate_exact_day_set() -> None:
    """The benchmark is built on the candidate's first_day/len, not the catalog's."""
    base = 7000
    n = 90
    rng = np.random.default_rng(21)
    rets = rng.normal(0.0005, 0.02, size=n + 50)
    closes = {"BTCUSDT": _prices(base, rets)}
    candidate = _dr(base + 51, np.asarray(rets)[50:] + 0.004)

    passed, record, reasons = buyhold.buy_hold_gate(candidate, "long", closes, ["BTCUSDT"])
    assert passed is True
    assert record["first_day"] == base + 51
    assert record["observations"] == n
    bh, _ = buyhold.buy_hold_daily(closes, ["BTCUSDT"], base + 51, n)
    assert bh is not None
    assert record["bh_sharpe_daily"] == pytest.approx(buyhold.daily_sharpe(bh))


def test_gate_candidate_from_dense_daily_returns() -> None:
    """A real metrics.DailyReturns candidate flows through the gate unchanged."""
    base = 8000
    n = 120
    rng = np.random.default_rng(31)
    cash = []
    balance = 10_000.0
    start_ns = base * _DAY_NS
    for i in range(1, n + 1):
        delta = balance * float(rng.normal(0.0004, 0.02))
        balance += delta
        cash.append((start_ns + i * _DAY_NS, delta))
    candidate = dense_daily_returns(10_000.0, cash, start_ns, start_ns + n * _DAY_NS)
    assert candidate is not None
    assert candidate.first_day == base
    assert len(candidate) == n

    closes = {"BTCUSDT": _prices(base, list(rng.normal(0.0, 0.03, size=n)))}
    passed, record, _reasons = buyhold.buy_hold_gate(candidate, "long", closes, ["BTCUSDT"])
    assert isinstance(passed, bool)
    assert record["observations"] == n
    assert record["first_day"] == base


def test_closes_mapping_is_not_mutated() -> None:
    closes = {"BTCUSDT": {0: 100.0, 1: 101.0, 2: 102.0}}
    snapshot = {s: dict(d) for s, d in closes.items()}
    buyhold.buy_hold_daily(closes, ["BTCUSDT"], 1, 2)
    assert closes == snapshot


# ---------------------------------------------------------------------------
# Fail-closed close validation (rev-d2-1 findings)
# ---------------------------------------------------------------------------

_RECORD_KEYS = {
    "series",
    "sharpe_daily",
    "sharpe_ann",
    "bh_sharpe_daily",
    "bh_sharpe_ann",
    "bar_daily",
    "bar_ann",
    "direction",
    "observations",
    "first_day",
    "passed",
}


@pytest.mark.parametrize(
    ("day", "value", "expected"),
    [
        (9, float("inf"), "invalid close BTCUSDT 1970-01-10: inf"),  # prev: inf/inf-1 was -1
        (9, 0.0, "invalid close BTCUSDT 1970-01-10: 0.0"),  # prev: was ZeroDivisionError
        (9, float("nan"), "invalid close BTCUSDT 1970-01-10: nan"),
        (10, float("nan"), "invalid close BTCUSDT 1970-01-11: nan"),
        (10, -5.0, "invalid close BTCUSDT 1970-01-11: -5.0"),
        (10, float("inf"), "invalid close BTCUSDT 1970-01-11: inf"),
    ],
)
def test_invalid_close_fails_closed(day: int, value: float, expected: str) -> None:
    """A non-finite or non-positive close (prev OR cur) rejects, never a return."""
    n = 40
    rng = np.random.default_rng(5)
    prices = _prices(9, list(rng.normal(0.001, 0.02, size=n)))
    prices[day] = value
    closes = {"BTCUSDT": prices}

    values, error = buyhold.buy_hold_daily(closes, ["BTCUSDT"], first_day=10, n=n)
    assert values is None
    assert error == expected

    # Through the gate: an otherwise-ordinary candidate must not pass.
    candidate = _dr(10, rng.normal(0.003, 0.02, size=n))
    passed, record, reasons = buyhold.buy_hold_gate(candidate, "long", closes, ["BTCUSDT"])
    assert passed is False
    assert record["bh_sharpe_daily"] is None
    assert record["bar_daily"] is None
    assert any(expected in r for r in reasons)


def test_invalid_close_in_any_symbol_rejects() -> None:
    """A bad close in the SECOND symbol still poisons the day."""
    n = 40
    rng = np.random.default_rng(6)
    good = _prices(9, list(rng.normal(0.0, 0.02, size=n)))
    bad = _prices(9, list(rng.normal(0.0, 0.02, size=n)))
    bad[20] = float("nan")
    values, error = buyhold.buy_hold_daily(
        {"BTCUSDT": good, "ETHUSDT": bad}, ["BTCUSDT", "ETHUSDT"], first_day=10, n=n
    )
    assert values is None
    assert error == "invalid close ETHUSDT 1970-01-21: nan"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -3.0])
def test_load_daily_closes_rejects_invalid_close(monkeypatch: MonkeyPatch, bad: float) -> None:
    """A complete catalog bar with a non-finite / non-positive close fails the load."""
    ts = np.array([_DAY_NS, 2 * _DAY_NS], dtype=np.int64)
    close = np.array([100.0, bad])
    fake = BarSeries(ts=ts, open=close, high=close, low=close, close=close)
    monkeypatch.setattr(buyhold, "load_catalog_bars", lambda c, b: fake)
    with pytest.raises(ValueError, match="invalid close BTCUSDT 1970-01-02"):
        buyhold.load_daily_closes(["BTCUSDT"], "/tmp/catalog")


def test_load_daily_closes_ignores_invalid_in_progress_bar(monkeypatch: MonkeyPatch) -> None:
    """Only COMPLETE bars are validated; a junk in-progress bar is just dropped."""
    ts = np.array([_DAY_NS, 3 * _DAY_NS + 43_200 * 10**9], dtype=np.int64)
    close = np.array([100.0, float("nan")])
    fake = BarSeries(ts=ts, open=close, high=close, low=close, close=close)
    monkeypatch.setattr(buyhold, "load_catalog_bars", lambda c, b: fake)
    assert buyhold.load_daily_closes(["BTCUSDT"], "/tmp/catalog") == {"BTCUSDT": {0: 100.0}}


def test_invalid_candidate_valid_benchmark_names_only_candidate() -> None:
    """Unavailable side is named exactly; the valid benchmark is still reported."""
    n = 40
    rng = np.random.default_rng(8)
    closes = {"AAAUSDT": _prices(10, list(rng.normal(0.001, 0.02, size=n)))}
    flat = _dr(11, [0.0] * n)

    passed, record, reasons = buyhold.buy_hold_gate(flat, "long", closes, ["AAAUSDT"])
    assert passed is False
    assert reasons == [f"Buy-hold: candidate daily sharpe unavailable (T={n})"]
    assert set(record) >= _RECORD_KEYS
    assert record["sharpe_daily"] is None and record["sharpe_ann"] is None
    assert isinstance(record["bh_sharpe_daily"], float)
    assert record["bar_daily"] == record["bh_sharpe_daily"]
    assert record["passed"] is False


def test_invalid_benchmark_valid_candidate_names_only_benchmark() -> None:
    n = 40
    rng = np.random.default_rng(9)
    prices = _prices(10, list(rng.normal(0.001, 0.02, size=n)))
    del prices[20]
    candidate = _dr(11, rng.normal(0.003, 0.02, size=n))

    passed, record, reasons = buyhold.buy_hold_gate(
        candidate, "long", {"AAAUSDT": prices}, ["AAAUSDT"]
    )
    assert passed is False
    assert len(reasons) == 1
    assert reasons[0].startswith("Buy-hold: benchmark missing close AAAUSDT")
    assert "candidate" not in reasons[0]
    assert isinstance(record["sharpe_daily"], float)
    assert record["bh_sharpe_daily"] is None and record["bar_daily"] is None


def test_worst_mode_pooled_unavailable_still_reports_symbols() -> None:
    """Worst mode keeps the full record and per-symbol verdicts when the pooled side fails."""
    n = 60
    base = 5000
    r_a = np.array([0.01, 0.02] * (n // 2))
    r_b = np.array([0.02, -0.02] * (n // 2))
    closes = {"AAAUSDT": _prices(base, r_a), "BBBUSDT": _prices(base, r_b)}
    flat = _dr(base + 1, [0.0] * n)  # pooled candidate unusable

    passed, record, reasons = buyhold.buy_hold_gate(
        flat,
        "long",
        closes,
        ["AAAUSDT", "BBBUSDT"],
        per_symbol={
            "AAAUSDT": _dr(base + 1, r_a + 0.002),  # above own bar
            "BBBUSDT": _dr(base + 1, r_b - 0.005),  # below own bar
        },
    )
    assert passed is False
    assert set(record) >= _RECORD_KEYS
    assert record["bh_sharpe_daily"] is not None
    symbols = record["symbols"]
    assert isinstance(symbols, dict)
    assert symbols["AAAUSDT"]["passed"] is True
    assert symbols["BBBUSDT"]["passed"] is False
    assert reasons[0] == f"Buy-hold: candidate daily sharpe unavailable (T={n})"
    assert any("BBBUSDT" in r for r in reasons)
    assert not any("AAAUSDT" in r for r in reasons)
    assert not any("benchmark" in r.split("(")[0] for r in reasons)


def test_per_symbol_equal_to_own_benchmark_rejects() -> None:
    """Worst mode is STRICT too: pooled passes, one symbol exactly at its own bar rejects.

    The symbol candidate IS that symbol's benchmark return array (no compounding
    round trip), so its Sharpe equals the bar bit for bit.
    """
    base = 5000
    n = 60
    r_a = np.array([0.01, 0.02] * (n // 2))
    r_b = np.array([0.02, -0.02] * (n // 2))
    symbols = ["AAAUSDT", "BBBUSDT"]
    closes = {"AAAUSDT": _prices(base, r_a), "BBBUSDT": _prices(base, r_b)}
    bh_a, err_a = buyhold.buy_hold_daily(closes, ["AAAUSDT"], base + 1, n)
    assert err_a is None and bh_a is not None
    pooled_candidate = _dr(base + 1, (r_a + r_b) / 2.0 + 0.004)  # beats the pooled bar

    passed, record, reasons = buyhold.buy_hold_gate(
        pooled_candidate,
        "long",
        closes,
        symbols,
        per_symbol={
            "AAAUSDT": _dr(base + 1, bh_a),  # exactly its own benchmark
            "BBBUSDT": _dr(base + 1, r_b + 0.01),  # clearly above its own bar
        },
    )
    pooled_sharpe, pooled_bar = record["sharpe_daily"], record["bar_daily"]
    assert isinstance(pooled_sharpe, float) and isinstance(pooled_bar, float)
    assert pooled_sharpe > pooled_bar  # pooled side really passes
    sym = record["symbols"]
    assert isinstance(sym, dict)
    assert sym["AAAUSDT"]["sharpe_daily"] == sym["AAAUSDT"]["bar_daily"]
    assert sym["AAAUSDT"]["passed"] is False
    assert sym["BBBUSDT"]["passed"] is True
    assert passed is False
    assert len(reasons) == 1 and "AAAUSDT" in reasons[0] and "<=" in reasons[0]


def test_empty_symbols_fails_closed_without_exception() -> None:
    """An empty symbol list must fail closed, never raise (rev-d2-1-r2 finding 2)."""
    values, error = buyhold.buy_hold_daily({}, [], first_day=10, n=40)
    assert values is None
    assert error == "no symbols"

    # A non-empty symbol list with no closes also fails closed (missing close).
    values, error = buyhold.buy_hold_daily({}, ["BTCUSDT"], first_day=10, n=40)
    assert values is None
    assert error == "missing close BTCUSDT 1970-01-10"


def test_gate_empty_symbols_fails_closed_without_exception() -> None:
    """The gate rejects a symbol-less symbol list with a reason naming it."""
    n = 40
    rng = np.random.default_rng(19)
    candidate = _dr(11, rng.normal(0.003, 0.02, size=n))
    passed, record, reasons = buyhold.buy_hold_gate(candidate, "long", {}, [])
    assert passed is False
    assert record["passed"] is False
    assert record["bh_sharpe_daily"] is None and record["bar_daily"] is None
    assert any("no symbols" in r for r in reasons)


def test_flat_benchmark_valid_candidate_names_only_benchmark() -> None:
    """Valid closes but a zero-variance benchmark: only the benchmark is unavailable."""
    n = 40
    rng = np.random.default_rng(12)
    closes = {"AAAUSDT": {d: 100.0 for d in range(10, 10 + n + 1)}}
    candidate = _dr(11, rng.normal(0.003, 0.02, size=n))

    passed, record, reasons = buyhold.buy_hold_gate(candidate, "long", closes, ["AAAUSDT"])
    assert passed is False
    assert reasons == [f"Buy-hold: benchmark daily sharpe unavailable (T={n})"]
    assert isinstance(record["sharpe_daily"], float)
    assert record["bh_sharpe_daily"] is None and record["bar_daily"] is None
