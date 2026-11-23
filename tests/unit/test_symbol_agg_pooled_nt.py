"""``symbol_agg="pooled"`` on the real NautilusTrader engine (vibe-quant-clmmc.5 C-T1, C3/C4).

Pooled scores ONE hypothesis over N coins: every coin runs on its own account (all eval
windows, no early exit), the per-window daily mark-to-market series are pooled equal weight
(symbols sorted once) and the pooled window metrics feed the usual worst-of-windows aggregate.
The reference here is built by hand from plain single-symbol screening runs, so every entry
point (inline call, process pool, holdout, cross-window/WFA factory, full range) must match it
exactly, with DSR's series collection on or off.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from vibe_quant.discovery import fitness as fitness_mod
from vibe_quant.discovery.fitness import FitnessResult, _evaluate_single, evaluate_population
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.pipeline import _confirm_key
from vibe_quant.discovery.run import NTBacktests
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.metrics import DailyReturns, profit_factor
from vibe_quant.validation.extraction import sharpe_sortino_from_returns

REPO = Path(__file__).resolve().parents[2]
BAR = REPO / "data" / "catalog" / "data" / "bar"
needs_catalog = pytest.mark.skipif(
    not all(
        (BAR / f"{s}-PERP.BINANCE-4-HOUR-LAST-EXTERNAL").is_dir() for s in ("BTCUSDT", "ETHUSDT")
    ),
    reason="BTCUSDT/ETHUSDT 4h catalog not available",
)

DAY_NS = 86_400 * 10**9
SYMS = ["BTCUSDT", "ETHUSDT"]  # sorted: the pooled order
JAN, FEB, MAR = (
    ("2024-01-01", "2024-02-01"),
    ("2024-02-01", "2024-03-01"),
    ("2024-03-01", "2024-04-01"),
)
FULL = ("2024-01-01", "2024-04-01")
MIN_TRADES = 50
# Long RSI momentum: trades every month on both coins; Feb and Mar are profitable on both.
CHROM = StrategyChromosome(
    entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 55.0, None)],
    exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 45.0, None)],
    stop_loss_pct=2.0, take_profit_pct=3.0, direction=Direction.LONG,
)  # fmt: skip


def _dsl() -> dict[str, Any]:
    from vibe_quant.discovery.genome import chromosome_to_dsl

    return {**chromosome_to_dsl(CHROM), "timeframe": "4h"}


@dataclass(frozen=True)
class Raw:
    """One plain single-symbol screening run: the hand reference's only input."""

    series: DailyReturns
    pnls: tuple[float, ...]
    trades: int


def raw_run(sym: str, span: tuple[str, str], launch: LaunchSettings | None = None) -> Raw:
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    runner = NTScreeningRunner(
        dsl_dict=_dsl(), symbols=[sym], start_date=span[0], end_date=span[1],
        collect_daily_returns=True, launch=launch or LaunchSettings(),
    )  # fmt: skip
    metrics = runner({})
    assert runner.last_mtm_daily_returns is not None
    return Raw(
        runner.last_mtm_daily_returns, tuple(runner.last_trade_pnls), int(metrics.total_trades)
    )


def hand_window(raws: list[Raw]) -> dict[str, Any]:
    """Pool one window by hand: equal-weight day means, then the pooled window metrics."""
    first = raws[0].series.first_day
    assert all(r.series.first_day == first and len(r.series) == len(raws[0].series) for r in raws)
    n = len(raws)
    pooled = [sum(r.series.values[i] for r in raws) / n for i in range(len(raws[0].series))]
    equity, peak, dd = 1.0, 1.0, 0.0
    for v in pooled:
        equity *= 1.0 + v
        peak = max(peak, equity)
        dd = max(dd, (peak - equity) / peak)
    sharpe = sharpe_sortino_from_returns(
        {(first + 1 + i) * DAY_NS: v for i, v in enumerate(pooled[1:])}
    )[0]
    return {
        "pooled": pooled,
        "first_day": first,
        "sharpe_ratio": sharpe,
        "total_return": equity - 1.0,
        "max_drawdown": dd,
        "profit_factor": profit_factor([p for r in raws for p in r.pnls]),
        "total_trades": sum(r.trades for r in raws),
    }


def symbol_return(raws: list[Raw]) -> float:
    equity = 1.0
    for r in raws:
        for v in r.series.values:
            equity *= 1.0 + v
    return equity - 1.0


@pytest.fixture(scope="module")
def ref() -> dict[tuple[str, tuple[str, str]], Raw]:
    return {(s, span): raw_run(s, span) for s in SYMS for span in (JAN, FEB, MAR, FULL)}


def _expected_train(ref: dict[tuple[str, tuple[str, str]], Raw]) -> dict[str, Any]:
    wins = [hand_window([ref[(s, span)] for s in SYMS]) for span in (FEB, MAR)]
    chain = wins[0]["pooled"][1:] + wins[1]["pooled"]
    return {
        "sharpe_ratio": min(w["sharpe_ratio"] for w in wins),
        "total_return": min(w["total_return"] for w in wins),
        "max_drawdown": max(w["max_drawdown"] for w in wins),
        "profit_factor": min(w["profit_factor"] for w in wins),
        "total_trades": sum(w["total_trades"] for w in wins),
        "chain": chain,
        "chain_first": wins[0]["first_day"] + 1,
        "stats": {
            s: (
                symbol_return([ref[(s, FEB)], ref[(s, MAR)]]),
                ref[(s, FEB)].trades + ref[(s, MAR)].trades,
            )
            for s in SYMS
        },
    }


def _assert_single_window(bt: dict[str, Any], w: dict[str, Any]) -> None:
    for key in ("sharpe_ratio", "total_return", "max_drawdown", "profit_factor", "total_trades"):
        assert bt[key] == w[key], key
    series = bt["mtm_daily_returns"]
    assert series is not None
    assert (series.first_day, list(series.values)) == (w["first_day"] + 1, w["pooled"][1:])


def _cell_fns(symbols: list[str]) -> Any:
    return NTBacktests().cell_fns(
        symbols=symbols, timeframe="4h", symbol_agg="pooled", launch=LaunchSettings(),
        train_start=FEB[0], train_end=MAR[1], eval_windows=[FEB, MAR], min_trades=MIN_TRADES,
        holdout=JAN, needs_factory=True, evaluator="nt",
    )  # fmt: skip


def _assert_train(fr: FitnessResult, exp: dict[str, Any]) -> None:
    for key in ("sharpe_ratio", "total_return", "max_drawdown", "profit_factor", "total_trades"):
        assert getattr(fr, key) == exp[key], key
    assert fr.error is None
    assert fr.pooled_symbol_stats == exp["stats"]
    assert fr.mtm_daily_returns is not None
    assert fr.mtm_daily_returns.first_day == exp["chain_first"]
    assert list(fr.mtm_daily_returns.values) == exp["chain"]
    # the pooled bootstrap input is the pooled daily series, never per-trade returns
    assert fr.trade_returns == tuple(exp["chain"])


@needs_catalog
@pytest.mark.slow(reason="real NT: 8 reference + 10 pooled single-symbol runs, ~3 s")
@pytest.mark.parametrize("dsr_on", [False, True], ids=["dsr-off", "dsr-on"])
def test_matches_hand_pool(ref: dict[tuple[str, tuple[str, str]], Raw], dsr_on: bool) -> None:
    """C3: inline call, holdout fn, factory and full-range fn == the hand pool, DSR on or off."""
    exp = _expected_train(ref)
    assert exp["total_return"] > 0 and exp["total_trades"] >= MIN_TRADES  # a gate passer
    fns = _cell_fns(list(SYMS))
    train = fns.backtest_fn
    if dsr_on:  # what DiscoveryPipeline does when DSR (or buy-hold / null gate) is on
        train.collect_daily_returns = True
    fr = _evaluate_single(CHROM, train, None, MIN_TRADES, "4h")
    _assert_train(fr, exp)
    assert fr.adjusted_score > 0

    bt = train(CHROM)
    assert set(bt["symbol_mtm_daily"]) == set(SYMS)
    for s in SYMS:
        assert list(bt["symbol_mtm_daily"][s].values) == [
            *ref[(s, FEB)].series.values[1:],
            *ref[(s, MAR)].series.values,
        ]
    assert "symbol_results" not in bt
    assert bt["trades_sum"] == exp["total_trades"]

    _assert_single_window(
        fns.holdout_backtest_fn(CHROM), hand_window([ref[(s, JAN)] for s in SYMS])
    )
    _assert_single_window(
        fns.backtest_fn_factory(*MAR)(CHROM), hand_window([ref[(s, MAR)] for s in SYMS])
    )
    full_fn = NTBacktests().nt_fn(
        symbols=list(SYMS), timeframe="4h", start_date=FULL[0], end_date=FULL[1],
        symbol_agg="pooled", launch=LaunchSettings(),
    )  # fmt: skip
    _assert_single_window(full_fn(CHROM), hand_window([ref[(s, FULL)] for s in SYMS]))


@needs_catalog
@pytest.mark.slow(reason="spawns a real process pool that imports NT, ~4 s")
def test_process_pool_matches_hand_pool(
    ref: dict[tuple[str, tuple[str, str]], Raw], monkeypatch: pytest.MonkeyPatch
) -> None:
    """C3: the per-(chromosome, symbol) process-pool split reduces to the same FitnessResult."""
    monkeypatch.setattr(fitness_mod, "_parallel_broken", False)
    train = _cell_fns(list(SYMS)).backtest_fn
    assert train.splits_by_symbol
    with ProcessPoolExecutor(max_workers=2) as pool:
        (fr,) = evaluate_population(
            [CHROM],
            train,
            None,
            max_workers=2,
            executor=pool,
            min_trades=MIN_TRADES,
            timeframe="4h",
        )
    _assert_train(fr, _expected_train(ref))
    assert _confirm_key(fr) == _confirm_key(_evaluate_single(CHROM, train, None, MIN_TRADES, "4h"))


@pytest.mark.slow(reason="real NT: 4 pooled runs on reversed symbol lists, ~2 s alone")
@needs_catalog
def test_reversed_symbols_identical(ref: dict[tuple[str, tuple[str, str]], Raw]) -> None:
    """C4: the symbol list order never reaches a number or the buy-and-hold record."""
    from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH
    from vibe_quant.discovery.buyhold import buy_hold_gate, load_daily_closes
    from vibe_quant.discovery.genome import genome_direction

    fwd = _evaluate_single(CHROM, _cell_fns(list(SYMS)).backtest_fn, None, MIN_TRADES, "4h")
    rev = _evaluate_single(CHROM, _cell_fns(SYMS[::-1]).backtest_fn, None, MIN_TRADES, "4h")
    assert _confirm_key(fwd) == _confirm_key(rev)
    assert list(fwd.symbol_mtm_daily or {}) == list(rev.symbol_mtm_daily or {}) == SYMS
    closes = load_daily_closes(list(SYMS), DEFAULT_CATALOG_PATH)

    def record(fr: FitnessResult) -> str:
        _passed, rec, _reasons = buy_hold_gate(
            fr.mtm_daily_returns, genome_direction(CHROM), closes, list(SYMS)
        )
        return json.dumps(rec, sort_keys=False)

    assert record(fwd) == record(rev)


@needs_catalog
def test_runner_trade_pnls_reset_per_run() -> None:
    """The per-trade net PnLs ride on the runner for one run only (a 0-trade rerun sees ())."""
    from vibe_quant.screening.nt_runner import NTScreeningRunner

    runner = NTScreeningRunner(
        dsl_dict=_dsl(),
        symbols=["BTCUSDT"],
        start_date=FEB[0],
        end_date=FEB[1],
        launch=LaunchSettings(),
    )
    first = runner({})
    assert first.total_trades > 0
    assert len(runner.last_trade_pnls) == first.total_trades
    assert profit_factor(runner.last_trade_pnls) == first.profit_factor
    assert all(math.isfinite(p) for p in runner.last_trade_pnls)
    second = runner({"rsi_entry_0_55_0_threshold": 101.0})  # RSI never exceeds 101
    assert second.total_trades == 0
    assert runner.last_trade_pnls == ()
