"""Per-symbol task split of worst-of-symbols evaluation (vibe-quant-yul7u.6)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.fitness import _evaluate_single, evaluate_population
from vibe_quant.errors import DataUnavailableError

from .test_fitness import _make_chromosome

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MIN_TRADES = 10


def _metrics(sharpe: float, ret: float, trades: int) -> dict[str, float | int]:
    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": 0.1,
        "profit_factor": 1.5,
        "total_trades": trades,
        "total_return": ret,
        "skewness": 0.1,
        "kurtosis": 3.5,
        "trade_returns": (0.01, -0.02),  # type: ignore[dict-item]
    }


PER_SYMBOL = {
    "BTCUSDT": _metrics(1.2, 0.3, 40),
    "ETHUSDT": _metrics(0.4, 0.1, 25),
    "SOLUSDT": _metrics(-0.5, -0.1, 30),
}


class FakeFn(NTBacktestFn):
    """Per-symbol results from a table; ``fail`` maps symbol -> exception."""

    def __init__(self, fail: dict[str, Exception] | None = None) -> None:
        super().__init__(SYMS, "4h", "2024-01-01", "2024-06-01", symbol_agg="worst", min_trades=MIN_TRADES)
        self.fail = fail or {}
        self.calls: list[list[str] | None] = []

    def _run_single(self, chromosome, start_date, end_date, symbols=None):  # type: ignore[no-untyped-def]
        self.calls.append(symbols)
        assert symbols is not None and len(symbols) == 1
        if symbols[0] in self.fail:
            raise self.fail[symbols[0]]
        return dict(PER_SYMBOL[symbols[0]])


class CountingPool(ThreadPoolExecutor):
    submits = 0

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.submits += 1
        return super().submit(fn, *args, **kwargs)


def _run_split(fn: NTBacktestFn, n: int = 2, expect_submits: int | None = None):  # type: ignore[no-untyped-def]
    chroms = [_make_chromosome() for _ in range(n)]
    pool = CountingPool(max_workers=4)
    try:
        out = evaluate_population(
            chroms, fn, None, max_workers=4, executor=pool, min_trades=MIN_TRADES, timeframe="4h"
        )
    finally:
        if expect_submits is not None:
            assert pool.submits == expect_submits
        pool.shutdown()
    return chroms, out


def test_split_equals_sequential_worst_of() -> None:
    fn = FakeFn()
    chroms, got = _run_split(fn, 2, expect_submits=2 * len(SYMS))
    fn.calls.clear()
    for chrom, fr in zip(chroms, got, strict=True):
        want = _evaluate_single(chrom, fn, None, MIN_TRADES, "4h")
        assert fr == want
    assert want.symbol_scores is not None and set(want.symbol_scores) == set(SYMS)
    # all symbols present as individual tasks
    fn2 = FakeFn()
    _run_split(fn2, 1)
    assert sorted(c[0] for c in fn2.calls if c) == sorted(SYMS)


def test_one_symbol_raises_scores_zero_with_error_dict() -> None:
    fn = FakeFn(fail={"ETHUSDT": RuntimeError("boom")})
    chroms, got = _run_split(fn, 2, expect_submits=2 * len(SYMS))
    for chrom, fr in zip(chroms, got, strict=True):
        assert fr.adjusted_score == 0.0
        assert fr.error == "RuntimeError: boom"
        assert fr.symbol_scores is None  # not aggregated as a synthetic symbol
        assert fr == _evaluate_single(chrom, fn, None, MIN_TRADES, "4h")


def test_data_unavailable_propagates() -> None:
    fn = FakeFn(fail={"SOLUSDT": DataUnavailableError("no data")})
    with pytest.raises(DataUnavailableError):
        _run_split(fn, 2, expect_submits=2 * len(SYMS))


def test_non_ntbacktestfn_uses_generic_path() -> None:
    seen: list[str] = []

    def plain(chrom):  # type: ignore[no-untyped-def]
        seen.append(chrom.uid)
        return _metrics(1.0, 0.2, 40)

    chroms, got = _run_split(plain, 2, expect_submits=2)  # type: ignore[arg-type]
    assert sorted(seen) == sorted(c.uid for c in chroms)
    assert all(fr.symbol_scores is None for fr in got)


def test_first_failure_in_symbol_order_wins_over_data_unavailable() -> None:
    # Sequential semantics: BTC ok, ETH raises RuntimeError -> error dict; SOL is never reached.
    fn = FakeFn(fail={"ETHUSDT": RuntimeError("eth"), "SOLUSDT": DataUnavailableError("sol")})
    chroms, got = _run_split(fn, 2, expect_submits=2 * len(SYMS))
    for chrom, fr in zip(chroms, got, strict=True):
        assert fr.error == "RuntimeError: eth"
        assert fr == _evaluate_single(chrom, fn, None, MIN_TRADES, "4h")


class TwoArgError(Exception):
    """Cannot be unpickled in the parent: __init__ takes 2 args, args holds 1."""

    def __init__(self, a: str, b: str) -> None:
        super().__init__(f"{a}-{b}")


class ProcFake(NTBacktestFn):
    """Module-level (picklable) fake; raises TwoArgError for ``bad_uid`` on ETH."""

    def __init__(self, bad_uid: str) -> None:
        super().__init__(SYMS, "4h", "2024-01-01", "2024-06-01", symbol_agg="worst", min_trades=MIN_TRADES)
        self.bad_uid = bad_uid

    def _run_single(self, chromosome, start_date, end_date, symbols=None):  # type: ignore[no-untyped-def]
        assert symbols is not None
        if chromosome.uid == self.bad_uid and symbols[0] == "ETHUSDT":
            raise TwoArgError("a", "b")
        return dict(PER_SYMBOL[symbols[0]])


def test_unpicklable_worker_exception_does_not_break_pool() -> None:
    from concurrent.futures import ProcessPoolExecutor

    chroms = [_make_chromosome() for _ in range(4)]
    fn = ProcFake(chroms[0].uid)
    with ProcessPoolExecutor(max_workers=2) as pool:
        got = evaluate_population(
            chroms, fn, None, max_workers=2, executor=pool, min_trades=MIN_TRADES, timeframe="4h"
        )
        again = evaluate_population(
            chroms[1:], fn, None, max_workers=2, executor=pool, min_trades=MIN_TRADES, timeframe="4h"
        )
    assert got[0].error == "TwoArgError: a-b" and got[0].adjusted_score == 0.0
    for fr in [*got[1:], *again]:
        assert fr.error is None and fr.adjusted_score > 0
