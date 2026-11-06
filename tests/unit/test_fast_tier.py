"""Fast tier: the replay equals NT screening fill-for-fill on real 1m data.

The parity tests run the same chromosome through ``NTBacktestFn`` and ``FastBacktestFn``
on two real BTCUSDT days and require identical fills and identical metrics (no
tolerance). ``scripts/agents/fast_tier_parity.py`` runs the same comparison over seeded
random genomes and longer windows.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vibe_quant.discovery.operators import ConditionType as C
from vibe_quant.discovery.operators import Direction, StrategyChromosome, StrategyGene
from vibe_quant.fast_tier import FastTierUnsupported
from vibe_quant.fast_tier.fills import bar_quarter_sizes, bar_ticks

REPO = Path(__file__).resolve().parents[2]
CATALOG_BARS = REPO / "data" / "catalog" / "data" / "bar" / "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"
METRICS = ("total_trades", "total_return", "sharpe_ratio", "profit_factor", "max_drawdown")
START, END = "2026-03-02", "2026-03-04"
# the memo tests compare runs with each other, not with NT: one day is enough
MEMO_START, MEMO_END = "2026-03-02", "2026-03-03"

needs_catalog = pytest.mark.skipif(not CATALOG_BARS.is_dir(), reason="needs the BTCUSDT 1m catalog")


def test_bar_quarter_sizes_matches_nt_rounding() -> None:
    # volume 10.0 at size increment 0.001 (raw scale 1e16 cancels out)
    assert bar_quarter_sizes(10_000, 1) == (2_500, 2_500)
    assert bar_quarter_sizes(10_003, 2) == (2_500, 2_502)  # quarter rounded down to the increment
    assert bar_quarter_sizes(1, 5) == (5, 5)  # never below the increment
    assert bar_quarter_sizes(12, 5) == (5, 5)  # 3 quarters exceed the volume -> close = increment


def test_bar_ticks_order_and_skips() -> None:
    q, cs = 7, 9
    # |H-O| = 5 < |L-O| = 10: high first
    assert bar_ticks(100, 105, 90, 95, 99, q, cs) == [
        (100, "O", q), (105, "H", q), (90, "L", q), (95, "C", cs),
    ]
    # tie -> low first; an open equal to the last price is not a tick
    assert bar_ticks(100, 110, 90, 90, 100, q, cs) == [(90, "L", q), (110, "H", q), (90, "C", cs)]
    # a close equal to the running last price is not a tick
    assert bar_ticks(100, 104, 100, 104, 100, q, cs) == [(104, "H", q)]
    # a flat bar at the last price processes nothing
    assert bar_ticks(100, 100, 100, 100, 100, q, cs) == []
    # first bar of the run: the open is always processed
    assert bar_ticks(100, 100, 100, 100, None, q, cs) == [(100, "O", q)]


def _genomes() -> list[StrategyChromosome]:
    stoch = {"period_k": 14.0, "period_d": 3.0}
    return [
        # many short trades: tight SL/TP so stops, at-limit TP draws and exits all occur
        StrategyChromosome(
            entry_genes=[StrategyGene("CCI", {"period": 20.0}, C.CROSSES_ABOVE, 100.0)],
            exit_genes=[StrategyGene("STOCH", stoch, C.GT, 95.0)],
            stop_loss_pct=0.5, take_profit_pct=0.6, direction=Direction.LONG,
        ),
        # both directions with a compute_fn (pandas) indicator on the exit side
        StrategyChromosome(
            entry_genes=[StrategyGene("RSI", {"period": 14.0}, C.CROSSES_BELOW, 30.0)],
            exit_genes=[StrategyGene("WILLR", {"period": 14.0}, C.GT, -20.0)],
            stop_loss_pct=0.8, take_profit_pct=1.0, direction=Direction.BOTH,
            short_entry_genes=[StrategyGene("RSI", {"period": 14.0}, C.CROSSES_ABOVE, 70.0)],
            short_exit_genes=[StrategyGene("WILLR", {"period": 14.0}, C.LT, -80.0)],
        ),
    ]


def _fills(positions: list[Any]) -> list[tuple[Any, ...]]:
    return [
        (int(f.ts_event), f.order_side.name, f.order_type.name, str(f.last_qty), str(f.last_px),
         str(f.commission))
        for pos in sorted(positions, key=lambda p: int(p.ts_opened))
        for f in pos.events
    ]


@pytest.fixture
def scratch_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBE_QUANT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.chdir(REPO)
    assert os.environ["VIBE_QUANT_DB"] != str(REPO / "data" / "state" / "vibe_quant.db")


@needs_catalog
@pytest.mark.parametrize("liquidation", ["off", "isolated"])
@pytest.mark.parametrize("genome_index", [0, 1])
def test_fast_tier_matches_nt_fill_for_fill(
    scratch_env: None, monkeypatch: pytest.MonkeyPatch, genome_index: int, liquidation: str
) -> None:
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.screening import nt_runner
    from vibe_quant.validation.extraction import all_positions

    captured: dict[str, list[Any]] = {}
    original = nt_runner.NTScreeningRunner._extract_metrics

    def grab(self: Any, params: Any, bt_result: Any, engine: Any, *a: Any, **k: Any) -> Any:
        captured["positions"] = all_positions(engine)
        return original(self, params, bt_result, engine, *a, **k)

    monkeypatch.setattr(nt_runner.NTScreeningRunner, "_extract_metrics", grab)
    kw: dict[str, Any] = {}
    if liquidation != "off":
        kw["liquidation"] = LiquidationSettings(mode="isolated", leverage=10)
    chromosome = _genomes()[genome_index]
    # collect_daily_returns: discovery's DSR / buy-and-hold gates read the MTM series
    fast_fn = FastBacktestFn(["BTCUSDT"], "1m", START, END, collect_daily_returns=True, **kw)
    fast = fast_fn._run_single(chromosome, START, END)
    nt = NTBacktestFn(["BTCUSDT"], "1m", START, END, collect_daily_returns=True, **kw)._run_single(
        chromosome, START, END
    )

    assert nt["total_trades"] > 5  # the window exercises fills, not an empty run
    assert _fills(fast_fn.last_positions) == _fills(captured["positions"])
    assert {m: fast[m] for m in METRICS} == {m: nt[m] for m in METRICS}
    # every other key too (skew, kurtosis, trade returns, fees, days, liquidation counts)
    assert {k: repr(v) for k, v in fast.items() if k != "mtm_daily_returns"} == {
        k: repr(v) for k, v in nt.items() if k != "mtm_daily_returns"
    }
    f_mtm, n_mtm = fast["mtm_daily_returns"], nt["mtm_daily_returns"]
    assert f_mtm.first_day == n_mtm.first_day  # type: ignore[union-attr]
    assert f_mtm.values.tobytes() == n_mtm.values.tobytes()  # type: ignore[union-attr]


FUNDED_SYMBOL = "ZECUSDT"
FUNDED_BARS = REPO / "data" / "catalog" / "data" / "bar" / "ZECUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL"
FUNDED_ARCHIVE = REPO / "data" / "archive" / "raw_data.db"
# One day on an 8h-funding symbol: three settlements, and every genome trades.
FUNDED_START, FUNDED_END = "2026-03-06", "2026-03-07"


def _funded_genome() -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[StrategyGene("FUNDING", {"period": 6.0}, C.GT, 0.0)],
        exit_genes=[StrategyGene("WILLR", {"period": 14.0}, C.GT, -20.0)],
        stop_loss_pct=0.8, take_profit_pct=1.0, direction=Direction.LONG,
    )


def _entry_quantities(positions: list[Any]) -> list[str]:
    """Qty of each position's entry fill, in fill order."""
    return [
        str(pos.events[0].last_qty)
        for pos in sorted(positions, key=lambda p: int(p.ts_opened))
    ]


@needs_catalog
def test_fast_tier_sizes_on_costed_balance(
    scratch_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay sizes and margin-checks on the costed balance NT uses.

    A funded 1m window: the replay's fills and metrics equal NT's bit for bit, and a
    second-or-later entry's qty is SMALLER than the same run with funding booked as zero
    (the raw NT balance). Equal fills alone would also pass a replay that ignores costs
    only if NT ignored them; the zero-funding twin pins that the ledger moved the size.
    """
    if not FUNDED_BARS.is_dir() or not FUNDED_ARCHIVE.is_file():
        pytest.skip("needs the ZECUSDT 1m catalog and the funding archive")

    from vibe_quant.dsl import aux_data
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn
    from vibe_quant.screening import nt_runner
    from vibe_quant.validation import funding as funding_module
    from vibe_quant.validation.extraction import all_positions

    captured: dict[str, list[Any]] = {}
    original = nt_runner.NTScreeningRunner._extract_metrics

    def grab(self: Any, params: Any, bt_result: Any, engine: Any, *a: Any, **k: Any) -> Any:
        captured["positions"] = all_positions(engine)
        return original(self, params, bt_result, engine, *a, **k)

    monkeypatch.setattr(nt_runner.NTScreeningRunner, "_extract_metrics", grab)
    aux_data.configure(FUNDED_ARCHIVE, REPO / "data" / "catalog")
    try:
        chromosome = _funded_genome()
        fast_fn = FastBacktestFn([FUNDED_SYMBOL], "1m", FUNDED_START, FUNDED_END)
        nt = fast_fn.nt_twin()._run_single(chromosome, FUNDED_START, FUNDED_END)
        nt_fills = _fills(captured["positions"])
        funded = fast_fn._run_single(chromosome, FUNDED_START, FUNDED_END)
        funded_fills = _fills(fast_fn.last_positions)

        assert nt["total_trades"] > 20
        assert funded_fills == nt_fills
        assert {m: funded[m] for m in METRICS} == {m: nt[m] for m in METRICS}
        funded_entries = _entry_quantities(fast_fn.last_positions)

        # Raw balance: the same replay with no funding ever charged (paid_through == 0).
        monkeypatch.setattr(
            funding_module.FundingCalculator,
            "accrue",
            lambda *_a, **_k: funding_module.FundingAccrual(),
        )
        raw_fn = FastBacktestFn([FUNDED_SYMBOL], "1m", FUNDED_START, FUNDED_END)
        raw_fn._run_single(chromosome, FUNDED_START, FUNDED_END)
    finally:
        aux_data.configure(None, None)

    raw_entries = _entry_quantities(raw_fn.last_positions)
    assert len(raw_entries) == len(funded_entries)
    differ = [i for i, (raw, costed) in enumerate(zip(raw_entries, funded_entries, strict=True))
              if raw != costed]
    assert differ, "funding never changed a size: the ledger is not wired into sizing"
    assert min(differ) >= 1, "the first entry must be sized on the untouched balance"
    assert float(funded_entries[differ[0]]) < float(raw_entries[differ[0]])


def _replay_for_ledger(
    qty: str | None, *, sl: float | None = None, tp: float | None = None
) -> Any:
    """A 3-bar replay over a real BTC instrument whose fake strategy enters at bar 1."""
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.objects import Price, Quantity

    from vibe_quant.fast_tier.fills import Replay
    from vibe_quant.fast_tier.signals import Signals, _Cache, load_instrument

    instrument = load_instrument(str(REPO / "data" / "catalog"), "BTCUSDT-PERP.BINANCE")
    bar_type = BarType.from_str("BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL")
    px = Price.from_str("100.0")
    volume = Quantity.from_str("1")
    bars = [
        Bar(bar_type, px, px, px, px, volume, ts, ts) for ts in (60 * 10**9, 120 * 10**9, 180 * 10**9)
    ]

    def size(_bar: Any, _is_long: bool) -> Any:
        return None if qty is None else Quantity.from_str(qty)

    strategy = SimpleNamespace(
        config=SimpleNamespace(max_margin_pct=0.5, margin_leverage=100),
        _calculate_position_size=size,
        _calculate_sl_price=lambda *_a: sl,
        _calculate_tp_price=lambda *_a: tp,
    )
    n = len(bars)
    long_entry = np.zeros(n, dtype=bool)
    long_entry[0] = True
    signals = Signals(
        np.ones(n, dtype=bool),
        long_entry,
        np.zeros(n, dtype=bool),
        np.zeros(n, dtype=bool),
        np.zeros(n, dtype=bool),
    )
    return Replay(
        strategy=strategy,
        cache=_Cache(instrument),
        instrument=instrument,
        bars=bars,
        signals=signals,
        cost_model=SimpleNamespace(),
        starting_balance=1000.0,
        fill_seed=1,
        prob_slippage=0.0,
        prob_fill_on_limit=1.0,
        end_ts=int(bars[-1].ts_init),
    )


@needs_catalog
def test_fast_tier_margin_replica_reads_costed_equity(scratch_env: None) -> None:
    """The margin replica reads NT balance minus ``paid_through``.

    Costed equity 1000 - 996.1 = 3.9 and a 0.5 limit: a 1.95 margin (notional 195 at
    leverage 100) lands exactly on the limit -- allowed, equality is marginable -- while
    1.96 is refused. On the raw 1000 balance both are allowed, so the subtraction is what
    refuses the second.
    """

    def run(qty: str, paid: float) -> Any:
        replay = _replay_for_ledger(qty)
        replay.ledger = SimpleNamespace(paid_through=lambda *_a: paid)
        return replay._entry_qty(0, True)

    assert run("1.95", 996.1) is not None  # exactly on the costed limit
    assert run("1.96", 996.1) is None  # above the costed limit
    assert run("1.96", 0.0) is not None  # the raw balance allows it


@needs_catalog
def test_fast_tier_records_the_stop_before_the_end_close(scratch_env: None) -> None:
    """The run-end force close comes AFTER ``ledger.stopped`` (NT's on_stop order)."""
    replay = _replay_for_ledger("0.1")
    calls: list[tuple[str, int]] = []
    replay.ledger = SimpleNamespace(
        paid_through=lambda *_a: 0.0,
        on_position_closed=lambda pos: calls.append(("closed", int(pos.ts_closed))),
        stopped=lambda now: calls.append(("stopped", int(now))),
    )
    positions = replay.run()

    end_ts = int(replay.bars[-1].ts_init)
    assert [pos.is_closed for pos in positions] == [True]
    assert int(positions[0].ts_opened) < end_ts  # the entry is not the end-of-run close
    assert calls == [("stopped", end_ts), ("closed", end_ts)]


def test_unsupported_shapes_raise_before_any_engine() -> None:
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

    chromosome = _genomes()[0]
    with pytest.raises(FastTierUnsupported, match="1m only"):
        FastBacktestFn(["BTCUSDT"], "4h", START, END)._run_single(chromosome, START, END)
    with pytest.raises(FastTierUnsupported, match="one symbol"):
        FastBacktestFn(["BTCUSDT", "ETHUSDT"], "1m", START, END)._run_single(chromosome, START, END)


def test_unsupported_becomes_an_error_row_through_call() -> None:
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

    row = FastBacktestFn(["BTCUSDT"], "4h", START, END)(_genomes()[0])
    assert "error" in row
    assert row["total_trades"] == 0


def _pool_genomes() -> list[StrategyChromosome]:
    """One genome per non-context GA indicator (entry and a re-parameterised exit) + MA genes."""
    import random

    from vibe_quant.discovery import operators
    from vibe_quant.discovery.operators import PriceVsMAConditionGene
    from vibe_quant.dsl.indicators import indicator_registry

    operators._ensure_pool()
    rng_state = random.getstate()
    random.seed(11)
    try:
        genomes = []
        for name in operators.INDICATOR_POOL:
            if getattr(indicator_registry.get(name), "needs_context", False):
                continue
            lo, hi = operators.THRESHOLD_RANGES.get(name, (0.0, 100.0))
            sub = {"BBANDS": "percent_b", "DONCHIAN": "position"}.get(name)
            genomes.append(StrategyChromosome(
                entry_genes=[StrategyGene(name, operators._random_params(name), C.GT, (lo + hi) / 2, sub)],
                exit_genes=[StrategyGene(name, operators._random_params(name), C.LT, (lo + hi) / 2, sub)],
                stop_loss_pct=0.8, take_profit_pct=1.0, direction=Direction.LONG,
            ))
        from vibe_quant.discovery.genome import MA_POOL

        for name in MA_POOL:
            fast, slow = operators._random_ma_params(name), operators._random_ma_params(name)
            genomes.append(StrategyChromosome(
                entry_genes=[StrategyGene("RSI", {"period": 14.0}, C.GT, 50.0)], exit_genes=[],
                ma_entry_genes=[PriceVsMAConditionGene(name, fast, C.CROSSES_ABOVE)],
                ma_exit_genes=[PriceVsMAConditionGene(name, slow, C.CROSSES_BELOW)],
                stop_loss_pct=0.8, take_profit_pct=1.0, direction=Direction.LONG,
            ))
        return genomes
    finally:
        random.setstate(rng_state)


def _run_fast(chromosome: StrategyChromosome) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

    fn = FastBacktestFn(["BTCUSDT"], "1m", MEMO_START, MEMO_END)
    res = fn._run_single(chromosome, MEMO_START, MEMO_END)
    return _fills(fn.last_positions), {m: res[m] for m in METRICS}


def _cold(monkeypatch: pytest.MonkeyPatch) -> None:
    from collections import OrderedDict

    from vibe_quant.fast_tier import signals

    monkeypatch.setattr(signals, "_MEMO", OrderedDict())
    monkeypatch.setattr(signals, "_memo_bytes", 0)


@needs_catalog
def test_replay_memo_hits_equal_cold_runs_for_every_pool_indicator(
    scratch_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibe_quant.fast_tier import signals

    genomes = _pool_genomes()
    cold = []
    for g in genomes:
        _cold(monkeypatch)
        cold.append(_run_fast(g))
    _cold(monkeypatch)
    first = [_run_fast(g) for g in genomes]
    entries = len(signals._MEMO)
    again = [_run_fast(g) for g in reversed(genomes)][::-1]  # every replay is now a hit

    assert entries > len(genomes)  # replays (and the frame) were memoised
    assert len(signals._MEMO) == entries  # the second pass added nothing: all hits
    assert first == cold
    assert again == cold
    assert sum(len(fills) for fills, _ in cold) > 20  # the window exercises fills


@needs_catalog
def test_replay_memo_never_shares_an_entry_across_params(
    scratch_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def genome(period: float) -> StrategyChromosome:
        return StrategyChromosome(
            entry_genes=[StrategyGene("WILLR", {"period": period}, C.LT, -90.0)],
            exit_genes=[StrategyGene("WILLR", {"period": period}, C.GT, -10.0)],
            stop_loss_pct=0.8, take_profit_pct=1.0, direction=Direction.LONG,
        )

    _cold(monkeypatch)
    cold_b = _run_fast(genome(15.0))
    _cold(monkeypatch)
    warm_a = _run_fast(genome(14.0))
    warm_b = _run_fast(genome(15.0))
    assert warm_a[0] != cold_b[0]  # the param changes the trades, so a shared entry would show
    assert warm_b == cold_b


def test_replay_memo_evicts_least_recent_under_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    from vibe_quant.fast_tier import signals

    _cold(monkeypatch)
    monkeypatch.setenv(signals.MEMO_MAX_MB_ENV, "1")
    bars: list[Any] = []
    mb = 1024 * 1024
    signals._memo_put(("a",), bars, "A", mb // 2)
    signals._memo_put(("b",), bars, "B", mb // 2)
    assert signals._memo_get(("a",)) == "A"  # a is now the most recent
    signals._memo_put(("c",), bars, "C", mb // 2)  # evicts b, the least recent
    assert list(signals._MEMO) == [("a",), ("c",)]
    assert signals._memo_bytes <= mb
    signals._memo_put(("huge",), bars, "H", 2 * mb)  # larger than the cap: not stored, no error
    assert signals._memo_get(("huge",)) is None
    monkeypatch.setenv(signals.MEMO_MAX_MB_ENV, "0")
    with pytest.raises(ValueError, match="integer >= 1"):
        signals._memo_put(("d",), bars, "D", 1)
