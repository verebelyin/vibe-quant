"""NTBacktestFn liquidation plumbing + report-only counts (bead vibe-quant-2gyk1.1.9).

The discovery backtest callable is the ONE place a genome is evaluated, so it must
carry the run's :class:`LiquidationSettings` into every ``NTScreeningRunner`` it builds
(train GA, holdout, cross-window / WFA factories) and report the mode-on liquidation
counts/flows the aggregates sum (R2, R3, R4). A mode-on integrity failure must fail the
run instead of scoring a silent -inf (I9), on both the sequential and the worst-split
paths.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.errors import DataUnavailableError, LiquidationIntegrityError
from vibe_quant.liquidation_settings import LiquidationSettings

if TYPE_CHECKING:
    from collections.abc import Callable

WINDOWS = [
    ("2024-01-01", "2024-02-01"),
    ("2024-02-01", "2024-03-01"),
    ("2024-03-01", "2024-04-01"),
]

ISOLATED_20 = LiquidationSettings(mode="isolated", leverage=20.0)


def _window_result(
    *,
    trades: int = 5,
    total_return: float = 0.05,
    liquidation_count: int = 0,
    liquidation_forfeited: float = 0.0,
) -> dict[str, float | int]:
    """One mode-on ``_run_single`` dict: every field the aggregates read."""
    return {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": trades,
        "total_return": total_return,
        "skewness": 0.0,
        "kurtosis": 3.0,
        "trade_returns": (),
        "total_fees": 1.0,
        "gross_pnl": 10.0,
        "starting_balance": 1000.0,
        "days": 30,
        "liquidation_count": liquidation_count,
        "liquidation_forfeited": liquidation_forfeited,
    }


def _metrics(**over: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": 5,
        "total_return": 0.05,
        "skewness": 0.0,
        "kurtosis": 3.0,
        "trade_returns": (),
        "total_fees": 1.0,
        "gross_pnl": 10.0,
        "liquidation_count": 7,
        "liquidation_forfeited": 3.0,
    }
    base.update(over)
    return SimpleNamespace(**base)


class _FakeRunner:
    """Minimal stand-in for NTScreeningRunner: one call returns fixed metrics."""

    last_starting_balance = 1000.0
    last_days = 30

    def __init__(self, metrics: SimpleNamespace) -> None:
        self._metrics = metrics

    def __call__(self, _params: dict[str, float | int]) -> SimpleNamespace:
        return self._metrics


def _patch_runner(monkeypatch: pytest.MonkeyPatch, metrics: SimpleNamespace) -> None:
    """Fake only the runner; the DSL build is stubbed so a bare uid suffices."""
    monkeypatch.setattr(
        NTBacktestFn, "_make_runner", lambda self, *a, **k: _FakeRunner(metrics)
    )
    monkeypatch.setattr(
        "vibe_quant.discovery.genome.chromosome_to_dsl",
        lambda chromosome: {"name": "probe", "timeframe": "4h"},
    )


def _patch_single(
    monkeypatch: pytest.MonkeyPatch,
    fn: Callable[..., dict[str, float | int]],
) -> None:
    monkeypatch.setattr(NTBacktestFn, "_run_single", fn)


# --------------------------------------------------------------------------- R2


def test_window_counts_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    """eval_windows=3 with hits 1/0/2 -> count 3, forfeited summed (R2)."""
    fn = NTBacktestFn(
        ["BTCUSDT"], "4h", "2024-01-01", "2024-04-01",
        windows=WINDOWS, min_trades=0, liquidation=ISOLATED_20,
    )
    by_start = {
        "2024-01-01": (1, 0.5),
        "2024-02-01": (0, 0.0),
        "2024-03-01": (2, 1.5),
    }

    def fake_run_single(self: Any, chromosome: Any, start: str, end: str, symbols: Any = None) -> dict[str, float | int]:
        count, forfeited = by_start[start]
        return _window_result(liquidation_count=count, liquidation_forfeited=forfeited)

    _patch_single(monkeypatch, fake_run_single)
    out = fn(SimpleNamespace(uid="x"))  # type: ignore[arg-type]

    assert out["liquidation_count"] == 3
    assert out["liquidation_forfeited"] == pytest.approx(2.0)


def test_early_exit_sums_only_windows_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A return <= 0 stop at window 2 counts neither window 3 nor its hits (R2)."""
    fn = NTBacktestFn(
        ["BTCUSDT"], "4h", "2024-01-01", "2024-04-01",
        windows=WINDOWS, min_trades=0, liquidation=ISOLATED_20,
    )
    ran: list[str] = []

    def fake_run_single(self: Any, chromosome: Any, start: str, end: str, symbols: Any = None) -> dict[str, float | int]:
        ran.append(start)
        if start == "2024-02-01":
            return _window_result(total_return=-0.01, liquidation_count=4)
        return _window_result(liquidation_count=1, liquidation_forfeited=0.5)

    _patch_single(monkeypatch, fake_run_single)
    out = fn(SimpleNamespace(uid="x"))  # type: ignore[arg-type]

    assert ran == ["2024-01-01", "2024-02-01"]  # window 3 never ran
    assert out["early_exit"] == 1
    assert out["liquidation_count"] == 5
    assert out["liquidation_forfeited"] == pytest.approx(0.5)


def test_low_trades_last_window_sums_every_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A last-window trade-gate miss takes the low-trades aggregate branch and
    still sums EVERY window that ran -- same rule as the other branches (R2).

    With min_trades=12 and N=3 the per-window gate is 2; windows 1-2 pass and
    stay positive (no early exit), so the miss on the LAST window cannot early
    exit and lands in ``_aggregate_multi_window``'s low-trades branch.
    """
    fn = NTBacktestFn(
        ["BTCUSDT"], "4h", "2024-01-01", "2024-04-01",
        windows=WINDOWS, min_trades=12, liquidation=ISOLATED_20,
    )
    # start -> (trades, liquidation_count, liquidation_forfeited)
    by_start = {
        "2024-01-01": (2, 2, 2.0),
        "2024-02-01": (3, 3, 3.0),
        "2024-03-01": (1, 1, 4.0),  # below the per-window gate, but it is last
    }

    def fake_run_single(self: Any, chromosome: Any, start: str, end: str, symbols: Any = None) -> dict[str, float | int]:
        trades, count, forfeited = by_start[start]
        return _window_result(
            trades=trades, liquidation_count=count, liquidation_forfeited=forfeited
        )

    _patch_single(monkeypatch, fake_run_single)
    out = fn(SimpleNamespace(uid="x"))  # type: ignore[arg-type]

    assert "early_exit" not in out  # no early exit: the miss is the last window
    assert out["sharpe_ratio"] == -1.0  # the low-trades failure branch was taken
    assert out["total_trades"] == 6  # 2 + 3 + 1, all windows ran
    assert out["liquidation_count"] == 2 + 3 + 1
    assert out["liquidation_forfeited"] == pytest.approx(2.0 + 3.0 + 4.0)


# --------------------------------------------------------------------------- R3


def test_worst_mode_sums_symbols_keeps_per_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    """symbol_agg=worst: the aggregate sums the symbols, each keeps its own count (R3)."""
    fn = NTBacktestFn(
        ["A", "B"], "4h", "2024-01-01", "2024-04-01",
        symbol_agg="worst", liquidation=ISOLATED_20,
    )
    per_symbol = {"A": (1, 0.5), "B": (2, 1.0)}

    def fake_run_single(self: Any, chromosome: Any, start: str, end: str, symbols: Any = None) -> dict[str, float | int]:
        count, forfeited = per_symbol[symbols[0]]
        return _window_result(liquidation_count=count, liquidation_forfeited=forfeited)

    _patch_single(monkeypatch, fake_run_single)
    out = fn(SimpleNamespace(uid="x"))  # type: ignore[arg-type]

    assert out["liquidation_count"] == 3
    assert out["liquidation_forfeited"] == pytest.approx(1.5)
    symbol_results = out["symbol_results"]
    assert isinstance(symbol_results, dict)
    assert symbol_results["A"]["liquidation_count"] == 1
    assert symbol_results["B"]["liquidation_count"] == 2


# --------------------------------------------------------------------------- R4


def test_default_construction_is_mode_off() -> None:
    """NTBacktestFn() without the keyword runs mode off (chief ruling on B2)."""
    fn = NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-06-01")

    assert fn.liquidation == LiquidationSettings()
    assert fn.liquidation.on is False


def test_mode_off_emits_no_liquidation_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mode-off run drops the counts the runner may have computed (R4/X1)."""
    fn = NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-06-01")
    _patch_runner(monkeypatch, _metrics(liquidation_count=7, liquidation_forfeited=3.0))

    out = fn(SimpleNamespace(uid="x"))  # type: ignore[arg-type]

    assert "liquidation_count" not in out
    assert "liquidation_forfeited" not in out


def test_mode_on_emits_the_runner_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mode on: the runner's liquidation metrics reach the bt dict (R1/R4)."""
    fn = NTBacktestFn(
        ["BTCUSDT"], "4h", "2024-01-01", "2024-06-01", liquidation=ISOLATED_20
    )
    _patch_runner(monkeypatch, _metrics(liquidation_count=7, liquidation_forfeited=3.0))

    out = fn(SimpleNamespace(uid="x"))  # type: ignore[arg-type]

    assert out["liquidation_count"] == 7
    assert out["liquidation_forfeited"] == pytest.approx(3.0)


# -------------------------------------------------------- setting reaches every runner


def test_setting_reaches_the_runner_and_venue_leverage() -> None:
    """_make_runner carries the mode/leverage; the venue runs at L when on (V1)."""
    on = NTBacktestFn(
        ["BTCUSDT"], "4h", "2024-01-01", "2024-06-01", liquidation=ISOLATED_20
    )
    runner = on._make_runner({"name": "probe", "timeframe": "4h"}, "2024-01-01", "2024-06-01")
    assert runner._liquidation_mode == "isolated"
    assert runner._leverage == 20.0
    assert runner._venue_config().default_leverage == Decimal("20")
    # R33: discovery runners run at the launch default, never at a params balance.
    assert runner.launch.starting_balance == 1000

    off = NTBacktestFn(["BTCUSDT"], "4h", "2024-01-01", "2024-06-01")
    off_runner = off._make_runner(
        {"name": "probe", "timeframe": "4h"}, "2024-01-01", "2024-06-01"
    )
    assert off_runner._liquidation_mode == "off"
    assert off_runner._venue_config().default_leverage == Decimal("10")


def test_settings_survive_pickling() -> None:
    """The setting must reach runners built in worker processes (bd-cm14)."""
    import pickle

    fn = NTBacktestFn(
        ["BTCUSDT"], "4h", "2024-01-01", "2024-06-01", liquidation=ISOLATED_20
    )
    loaded = pickle.loads(pickle.dumps(fn))

    assert loaded.liquidation == ISOLATED_20


# --------------------------------------------------------------------------- I9


class MarkDataMissing(Exception):
    """Shaped like the real MARK-bar failure the integrity wrap names."""


def test_integrity_error_fails_run_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sequential eval: a wrapped integrity error propagates out of __call__ (I9)."""
    fn = NTBacktestFn(
        ["BTCUSDT"], "4h", "2024-01-01", "2024-06-01", liquidation=ISOLATED_20
    )

    def boom(self: Any, *a: Any, **k: Any) -> dict[str, float | int]:
        raise LiquidationIntegrityError.wrap(MarkDataMissing("no marks"), "isolated")

    _patch_single(monkeypatch, boom)

    with pytest.raises(LiquidationIntegrityError, match="liquidation_mode='isolated': MarkDataMissing:"):
        fn(SimpleNamespace(uid="x"))  # type: ignore[arg-type]


def test_integrity_error_fails_run_worst_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worst-split (per-symbol parallel task) fails the same way, not -inf (I9)."""
    fn = NTBacktestFn(
        ["A", "B"], "4h", "2024-01-01", "2024-06-01",
        symbol_agg="worst", liquidation=ISOLATED_20,
    )
    assert fn.splits_by_symbol

    def boom(self: Any, chromosome: Any, start: str, end: str, symbols: Any = None) -> dict[str, float | int]:
        raise LiquidationIntegrityError.wrap(MarkDataMissing("no marks"), "isolated")

    _patch_single(monkeypatch, boom)

    with pytest.raises(LiquidationIntegrityError, match="liquidation_mode='isolated': MarkDataMissing:"):
        fn(SimpleNamespace(uid="x"))  # type: ignore[arg-type]


def test_integrity_error_is_data_unavailable() -> None:
    """The guarantee __call__ relies on: the subclass is the re-raised base."""
    assert issubclass(LiquidationIntegrityError, DataUnavailableError)
