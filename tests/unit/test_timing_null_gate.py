"""Timing-null gate (``--null-gate``, vibe-quant-clmmc.14 B-T2): capture + gate + persistence.

The gate re-runs a candidate's train windows with the screening runner's span capture on
(the priced, run-end-marked mode-off ledger), checks the re-run equals the scored result,
then ranks the run's own as-traded MTM compound return among random re-timings of its own
trades (``timing_null.timing_null_check``). Exposure above 0.90 hands the verdict to the
buy-and-hold record. Every input problem rejects (fails closed). Off by default: a seeded
default run's notes carry no timing-null key (``test_discovery_notes_golden.py``).
"""

from __future__ import annotations

import dataclasses
import random
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vibe_quant.discovery import record
from vibe_quant.discovery import timing_null as tn
from vibe_quant.discovery.backtest_fn import CapturedRun
from vibe_quant.discovery.fitness import FitnessResult, _evaluate_single
from vibe_quant.discovery.gate_chain import GateChain, GateEvidence
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.pipeline import DiscoveryConfig, DiscoveryPipeline, DiscoveryResult
from vibe_quant.metrics import DailyReturns, pool_daily_returns

if TYPE_CHECKING:
    from collections.abc import Sequence

MS = 10**6
H4 = 4 * 3_600 * 10**9
B0 = 1000.0
FEE = 0.0005
COSTS = tn.TradeCosts(fee=FEE)
REPO = Path(__file__).resolve().parents[2]
BTC_4H = REPO / "data" / "catalog" / "data" / "bar" / "BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL"
needs_btc = pytest.mark.skipif(not BTC_4H.is_dir(), reason="BTCUSDT 4h catalog not available")

# the record keys of design arch-gates-2 §2.2, plus the GATES-7 window-boundary clip counter
RECORD_KEYS = {
    "percentile", "draws", "seeds", "exposure", "exposure_by_symbol", "carried",
    "zero_hold_clipped", "real_T", "null_median", "null_p95", "fallback", "fallback_passed",
    "alpha_ann", "beta", "alpha_note", "fee", "boundary_clipped",
}  # fmt: skip

# Real engine: BTC 4h 2024-01-01..04-01, enters once (2024-02-13) and is still long at the end.
START, END = "2024-01-01", "2024-04-01"
HELD = StrategyChromosome(
    entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 1.0, None)],
    exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 0.5, None)],
    stop_loss_pct=50.0, take_profit_pct=99.0, direction=Direction.LONG,
)  # fmt: skip
HELD_FRAC = 0.001 * 50001.8 / 1000.0  # entry notional / equity at entry (flat account)
HELD_SPAN = tn.TradeSpan(side=1, frac=HELD_FRAC, entry_bar=260, hold_bars=286, carried=True)


def _ns(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp()) * 10**9


def _held_dsl() -> dict[str, Any]:
    from vibe_quant.discovery.genome import chromosome_to_dsl

    return {**chromosome_to_dsl(HELD), "timeframe": "4h"}


# --------------------------------------------------------------------------- real engine


@needs_btc
def test_carried_trade_kernel_equals_engine_mtm(monkeypatch: pytest.MonkeyPatch) -> None:
    """R-gb1-1, B11, GATES-5/6: the captured span of a trade open at the run end replays to the
    engine's own MTM series -- overnight marks, funding and the run-end mark's taker exit fee."""
    import vibe_quant.screening.nt_runner as ntr

    seen: list[Any] = []
    real_cost = ntr.cost_positions

    def spy(*a: Any, **kw: Any) -> Any:
        out = real_cost(*a, **kw)
        seen.append(out)
        return out

    monkeypatch.setattr(ntr, "cost_positions", spy)
    runner = ntr.NTScreeningRunner(
        _held_dsl(), ["BTCUSDT"], START, END, collect_daily_returns=True, capture_spans=True
    )
    runner({})
    (item,) = seen[-1]
    # the probe numbers (/tmp/cc-gb2-1/probe_carried.py, chief R-gb2-1), verbatim
    assert int(item.pos.ts_opened) == 1707811259999000000
    assert float(item.pos.avg_px_open) == 50001.8
    assert float(item.pos.peak_qty) == 0.001
    assert item.mark.ts == 1711929599999000000
    assert item.mark.price == 71363.0
    assert item.mark.exit_fee == 0.0356815
    assert runner.last_trade_spans is not None
    tape, spans, boundary_clipped = runner.last_trade_spans["BTCUSDT"]
    assert (len(tape), tape.start_ns, tape.end_ns) == (546, _ns(START), _ns(END))
    assert spans == [HELD_SPAN] and boundary_clipped == 0
    # R7b: the capture hands the null its cost inputs: BTCUSDT-PERP's tick and the unit funding
    # table, every settlement inside (start, last ts]
    assert tape.tick == 0.1
    assert tape.funding_ts is not None and tape.funding_unit is not None
    assert len(tape.funding_ts) > 0 and len(tape.funding_ts) == len(tape.funding_unit)
    assert all(tape.start_ns < int(t) <= int(tape.ts[-1]) for t in tape.funding_ts)
    # R7: the production funding hook on the captured span == the engine's own accrual, exactly
    # (notional = avg_px_open x peak_qty, fill instant to the run-end mark)
    prod = tn.capture_costs(FEE, {"BTCUSDT": [tape]}, 0.5)
    assert prod.funding is not None
    notional = float(item.pos.avg_px_open) * abs(float(item.pos.peak_qty))
    entry_ns, exit_ns = int(item.pos.ts_opened), int(item.mark.ts)
    assert (entry_ns, exit_ns) == tn.funding_window(tape, 260, len(tape) - 1, True)
    pays = prod.funding("BTCUSDT", 1, notional, entry_ns, exit_ns)
    assert pays == list(item.accrual.payments) and len(pays) > 10
    # the real fill price stands in for the null's expected slippage: the kernel then equals
    # the engine's MTM with the production funding hook, no injected payments
    costs = dataclasses.replace(prod, fill_price=lambda _sym, _side, _px: 50001.8)
    engine = runner.last_mtm_daily_returns
    assert engine is not None
    got = tn.sim_daily(tape, spans, costs, B0)
    assert got.first_day == engine.first_day and len(got) == len(engine)
    # frac round-trips the quantity through equity / fill price: equal to the last bits
    assert list(got.values) == pytest.approx(list(engine.values), rel=0, abs=1e-12)
    # the last day holds the exit fee: the kernel without it would be off by fee x q x close
    no_fee_last = got.values[-1] + 0.0356815 / (B0 * (1 + tn.total_return(got.values[:-1])))
    assert abs(no_fee_last - engine.values[-1]) > 3e-5


@needs_btc
def test_capture_rerun_is_identical_and_keeps_global_random() -> None:
    """B7: two captures give the same spans and result; the global RNG never moves."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    fn = NTBacktestFn(["BTCUSDT"], "4h", START, END, collect_daily_returns=True)
    random.seed(123)
    before = random.getstate()
    first = fn.capture_spans(HELD)
    assert random.getstate() == before
    second = fn.capture_spans(HELD)
    assert first.spans == second.spans == {"BTCUSDT": [[HELD_SPAN]]}
    assert first.bt["total_return"] == second.bt["total_return"] == 0.019246313151302104
    assert (first.b0, first.fee, first.boundary_clipped) == (B0, FEE, 0)
    (tape_a,), (tape_b,) = first.tapes["BTCUSDT"], second.tapes["BTCUSDT"]
    assert np.array_equal(tape_a.open, tape_b.open) and np.array_equal(tape_a.ts, tape_b.ts)


@needs_btc
def test_an_unreadable_funding_archive_leaves_the_table_unset_without_the_error_text(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R9: when the unit funding table cannot be built the run still completes, the tape keeps
    its tick but no table (the gate then rejects the candidate), and the warning names the
    exception type, never its text."""
    import vibe_quant.screening.nt_runner as ntr
    from vibe_quant.validation.funding import FundingArchiveUnavailableError

    def boom(*a: Any, **kw: Any) -> Any:
        raise FundingArchiveUnavailableError("MARKER-archive-boom")

    monkeypatch.setattr(tn, "funding_table", boom)
    runner = ntr.NTScreeningRunner(
        _held_dsl(), ["BTCUSDT"], START, END, collect_daily_returns=True, capture_spans=True
    )
    with caplog.at_level("WARNING"):
        runner({})
    assert runner.last_trade_spans is not None
    tape, spans, _ = runner.last_trade_spans["BTCUSDT"]
    assert spans == [HELD_SPAN]
    assert tape.tick == 0.1 and tape.funding_ts is None and tape.funding_unit is None
    with pytest.raises(tn.TimingNullError) as err:
        tn.capture_costs(FEE, {"BTCUSDT": [tape]}, 0.5)
    assert str(err.value) == "Timing null: no funding table for BTCUSDT window 1"
    assert "no funding table for BTCUSDT (FundingArchiveUnavailableError)" in caplog.text
    assert "MARKER-archive-boom" not in caplog.text


@needs_btc
def test_a_run_without_span_capture_builds_no_cost_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """R17: the tick and the funding table exist only on the opt-in capture path."""
    import vibe_quant.screening.nt_runner as ntr

    monkeypatch.setattr(tn, "funding_table", lambda *a, **kw: pytest.fail("table built"))
    runner = ntr.NTScreeningRunner(_held_dsl(), ["BTCUSDT"], START, END, collect_daily_returns=True)
    runner({})
    assert runner.last_trade_spans is None


class _Ops:
    """A fixed one-generation population (no GA draws)."""

    def __init__(self, chroms: list[StrategyChromosome]) -> None:
        self.chroms = chroms

    def initialize(self, size: int, seeds: object = None) -> list[StrategyChromosome]:
        return [c.clone() for c in self.chroms[:size]]

    def random(self) -> StrategyChromosome:  # pragma: no cover - one generation
        raise AssertionError("no evolution")

    mutate = crossover = random  # type: ignore[assignment]

    def is_valid(self, genome: StrategyChromosome) -> bool:  # pragma: no cover
        return True


@needs_btc
@pytest.mark.slow(reason="real NT discovery run of 7 genomes + timing null, ~5 s")
def test_real_engine_cli_null_gate_with_dsr_and_buy_hold_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B13 + GATES-16: a real ``--null-gate --no-dsr --no-buy-hold-gate`` run still collects the
    train MTM series and persists every champion with its own record (keys exact, real_T the
    compound of the gated series). Fixture: a dip-buy port that beats its re-timings."""
    from tests.unit.test_param_genome_consumers import _genome, _write_port
    from tests.unit.test_run_discovery import drive_run_discovery
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import run

    port = _write_port(tmp_path, {
        "name": "timing_null_b13_fixture",
        "params": {"low": {"kind": "int", "low": 39100, "high": 40900, "step": 300,
                           "default": 39400}},
        "dsl": {
            "name": "timing_null_b13_fixture", "timeframe": "4h", "indicators": {},
            "entry_conditions": {"long": ["close < {{low}}"], "short": []},
            "exit_conditions": {"long": ["close > 72700"], "short": []},
            "stop_loss": {"type": "fixed_pct", "percent": 9.0},
            "take_profit": {"type": "fixed_pct", "percent": 95.0},
        },
    })  # fmt: skip
    chroms = [_genome(port, low=low) for low in range(39100, 40901, 300)]
    gated: dict[str, Any] = {}

    class Pipe(DiscoveryPipeline):
        def __init__(self, config: DiscoveryConfig, *a: Any, **kw: Any) -> None:
            kw["genome_ops"] = _Ops(chroms)  # type: ignore[arg-type]
            super().__init__(dataclasses.replace(config, min_trades=1), *a, **kw)
            gated["pipe"] = self

    monkeypatch.setattr(run, "DiscoveryPipeline", Pipe)
    db = tmp_path / "state.db"
    state = StateManager(db)
    run_id = state.create_backtest_run(None, "discovery", ["BTCUSDT"], "4h", START, END, {})
    state.close()
    argv = [
        "--seed", "7", "--population-size", str(len(chroms)), "--max-generations", "1",
        "--elite-count", "1", "--max-workers", "-1", "--start-date", START, "--end-date", END,
        "--timeframe", "4h", "--symbols", "BTCUSDT", "--eval-windows", "1",
        "--train-test-split", "0", "--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate",
        "--gate-pool", "all", "--null-gate", "--db", str(db), "--run-id", str(run_id),
    ]  # fmt: skip
    outcome = drive_run_discovery(db, run_id, argv, run.NTBacktests())
    assert outcome.exit_code == 0, outcome.message
    notes = outcome.notes
    assert notes is not None
    assert notes["timing_null_gate"] == {
        "enforced": True,
        "draws": 500,
        "pct": 0.95,
        "exposure_max": 0.9,
        "base_seed": 7,
        "statistic": "as-traded MTM compound",
        "null_costs": (
            "the real side's costs at the re-timed holds: taker fee per leg, expected screening "
            "tick slippage (0.5 x one tick per market fill) and signed archived funding per "
            "held settlement from the fill instants"
        ),
        "limits": [*tn.NULL_GATE_LIMITS],
    }
    assert len(tn.NULL_GATE_LIMITS) == 4 and tn.NULL_GATE_LIMITS[-1] == (
        "null fills are bar opens with the fill model's expected slippage; the engine fills at "
        "the boundary 1m close with a seeded 0/1-tick draw"
    )
    pipe: DiscoveryPipeline = gated["pipe"]
    records = pipe.gates.timing_null_records
    policy = tn.parse_timing_null_policy(notes)
    champions = notes["top_strategies"]
    assert len(champions) == 4  # the probe's outcome (rev-gb2-1 probe_b13_champion.py)
    for row in champions:
        rec = row["timing_null"]
        assert set(rec) == RECORD_KEYS
        (match,) = [c for c in chroms if records.get(c.uid) == rec]
        key = pipe._dsl_key(match)
        fit = pipe.fitness_cache[key]
        assert fit.mtm_daily_returns is not None  # collected although DSR and buy-hold are off
        assert rec["real_T"] == tn.total_return(fit.mtm_daily_returns.values)
        assert rec["seeds"] == {"BTCUSDT": tn.null_seed(7, key, "BTCUSDT")}
        assert rec["draws"] == 500 and rec["fee"] == FEE
        assert tn.timing_null_decision(policy, rec) == (True, [])


@needs_btc
def test_capture_rerun_scores_with_the_runs_filter() -> None:
    """A pipeline's filter_fn scores the capture re-run too: the re-run equals the scored
    result instead of failing on ``filter_results`` (rev-gb2-1 probe_filter_confirmation)."""
    from vibe_quant.discovery.backtest_fn import NTBacktestFn

    fn = NTBacktestFn(["BTCUSDT"], "4h", START, END)
    cfg = _config(["BTCUSDT"], start_date=START, end_date=END, min_trades=1, max_workers=None,
                  require_dsr=False, require_bootstrap_ci=False)  # fmt: skip
    pipe = DiscoveryPipeline(cfg, fn, filter_fn=lambda chrom, bt: {"custom": True})
    (fit,) = pipe._evaluate_new([HELD])
    assert fit.passed_filters and fit.filter_results == {"custom": True}
    pipe.gates.apply_timing_null([(HELD, fit)])
    reasons = [r["reasons"] for r in pipe.gates.rejections]
    assert not any("capture re-run differs" in str(r) for r in reasons), reasons
    assert pipe.gates.timing_null_records[HELD.uid]["carried"] == 1


# --------------------------------------------------------------------------- seam fixtures


def _tape(
    symbol: str, opens: Sequence[float], start: str = "2024-03-10", drop: Sequence[int] = ()
) -> tn.WindowTape:
    o = np.asarray(opens, dtype=np.float64)
    start_ns = _ns(start)
    open_ns = start_ns + np.arange(len(o), dtype=np.int64) * H4
    keep = np.setdiff1d(np.arange(len(o)), np.asarray(drop, dtype=np.int64))
    return tn.WindowTape(
        symbol=symbol, timeframe="4h", tf_ns=H4, start_ns=start_ns,
        end_ns=start_ns + len(o) * H4, open_ns=open_ns[keep], ts=open_ns[keep] + H4 - MS,
        open=o[keep], close=o[keep] * 1.001,
        tick=0.01, funding_ts=np.zeros(0, np.int64), funding_unit=np.zeros(0),  # no settlements
    )  # fmt: skip


def _walk(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))


def _closes(symbols: Sequence[str], start: str, days: int) -> dict[str, dict[int, float]]:
    d0 = _ns(start) // tn.DAY_NS
    return {
        s: {d0 - 1 + k: float(v) for k, v in enumerate(_walk(days + 2, 9 + i))}
        for i, s in enumerate(symbols)
    }


def _chrom(i: int) -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 20.0 + i / 10, None)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        stop_loss_pct=2.0, take_profit_pct=4.0, direction=Direction.LONG,
    )  # fmt: skip


def _candidate(
    chrom: StrategyChromosome,
    tapes: dict[str, list[tn.WindowTape]],
    spans: dict[str, list[list[tn.TradeSpan]]],
    *,
    rerun_return: float | None = None,
) -> tuple[tuple[StrategyChromosome, FitnessResult], CapturedRun]:
    """A scored candidate whose MTM series IS the unshuffled kernel replay of its spans."""
    per_symbol: dict[str, DailyReturns | None] = {}
    for sym in tapes:
        parts = [tn.sim_daily(t, s, COSTS, B0) for t, s in zip(tapes[sym], spans[sym], strict=True)]
        per_symbol[sym] = tn._chain(parts)
    symbols = list(tapes)
    bt: dict[str, Any] = {
        "sharpe_ratio": 2.0, "max_drawdown": 0.1, "profit_factor": 1.5, "total_trades": 60,
        "total_return": 0.2, "skewness": 0.0, "kurtosis": 3.0,
    }  # fmt: skip
    if len(symbols) == 1:
        bt["mtm_daily_returns"] = per_symbol[symbols[0]]
    else:
        bt["mtm_daily_returns"] = pool_daily_returns(list(per_symbol.values()))
        bt["symbol_mtm_daily"] = per_symbol
        bt["symbol_results"] = {s: dict(bt, mtm_daily_returns=None) for s in symbols}
        for s in symbols:
            del bt["symbol_results"][s]["mtm_daily_returns"]
    fit = _evaluate_single(chrom, lambda c: dict(bt), None, 50, "4h")
    rerun = dict(bt) if rerun_return is None else dict(bt, total_return=rerun_return)
    cap = CapturedRun(bt=rerun, tapes=tapes, spans=spans, b0=B0, fee=FEE, boundary_clipped=0)
    return (chrom, fit), cap


def _config(symbols: list[str], **kw: Any) -> DiscoveryConfig:
    base: dict[str, Any] = {
        "symbols": symbols, "timeframe": "4h", "start_date": "2024-03-10",
        "end_date": "2024-04-10", "min_trades": 50, "require_buy_hold": False,
        "null_gate": True, "null_base_seed": 7,
        "symbol_agg": "worst" if len(symbols) > 1 else "portfolio",
    }  # fmt: skip
    base.update(kw)
    return DiscoveryConfig(**base)


def _gates(
    cfg: DiscoveryConfig, caps: dict[str, CapturedRun], days: int, start: str = "2024-03-10"
) -> GateChain:
    closes = _closes(cfg.symbols, start, days)
    return GateChain(
        cfg,
        GateEvidence(
            SimpleNamespace(symbols=cfg.symbols),
            False,
            buy_hold_closes_fn=lambda syms: closes,
            timing_null_capture_fn=lambda chrom: caps[chrom.uid],
        ),
    )


def _fake_buy_hold(monkeypatch: pytest.MonkeyPatch, passed: bool) -> None:
    import vibe_quant.discovery.buyhold as bh

    reasons = [] if passed else ["Buy-hold: MTM Sharpe 0.10 <= benchmark 0.50"]
    monkeypatch.setattr(bh, "buy_hold_gate", lambda *a, **kw: (passed, {"bh": 1}, reasons))


# --------------------------------------------------------------------------- gate (seams)


@pytest.mark.parametrize(
    ("bh_passed", "survives"), [(False, False), (True, True)], ids=["bh-fails", "bh-passes"]
)
def test_exposure_above_090_hands_the_verdict_to_buy_and_hold(
    monkeypatch: pytest.MonkeyPatch, bh_passed: bool, survives: bool
) -> None:
    """GATES-7 / B8 edge: 9001 held bars of 10000 -> fallback, exact 4-decimal message."""
    _fake_buy_hold(monkeypatch, bh_passed)
    monkeypatch.setattr(tn, "place_trades", lambda *a: pytest.fail("null placed trades"))
    chrom = _chrom(1)
    tapes = {"BTCUSDT": [_tape("BTCUSDT", _walk(10_000, 1), start="2020-01-01")]}
    spans = {"BTCUSDT": [[tn.TradeSpan(1, 0.5, 0, 9001)]]}
    cand, cap = _candidate(chrom, tapes, spans)
    days = 10_000 // 6 + 2
    gates = _gates(_config(["BTCUSDT"]), {chrom.uid: cap}, days, start="2020-01-01")
    out = gates.validate([cand])
    rec = gates.timing_null_records[chrom.uid]
    assert rec["exposure"] == 0.9001 and rec["percentile"] is None
    assert rec["fallback_passed"] is bh_passed
    assert [c.uid for c, _ in out.survivors] == ([chrom.uid] if survives else [])
    if not survives:
        (row,) = gates.rejections
        assert row["stage"] == "timing_null"
        assert row["reasons"] == [
            "Timing null: exposure 0.9001 > 0.90, null degenerate; buy-and-hold decides "
            "(Buy-hold: MTM Sharpe 0.10 <= benchmark 0.50)"
        ]
        assert row["timing_null"] == rec


def test_mixed_exposure_exactly_090_runs_the_null(monkeypatch: pytest.MonkeyPatch) -> None:
    """GATES-7: 100 % and 80 % on two equally covered coins is exactly 0.90 -> the null runs."""
    monkeypatch.setattr(tn, "NULL_DRAWS", 20)
    _fake_buy_hold(monkeypatch, False)
    chrom = _chrom(2)
    tapes = {s: [_tape(s, _walk(60, i))] for i, s in enumerate(["BTCUSDT", "ETHUSDT"])}
    spans = {
        "BTCUSDT": [[tn.TradeSpan(1, 0.5, 0, 60, True)]],
        "ETHUSDT": [[tn.TradeSpan(-1, 0.4, 0, 30), tn.TradeSpan(1, 0.3, 40, 18)]],
    }
    cand, cap = _candidate(chrom, tapes, spans)
    gates = _gates(_config(["BTCUSDT", "ETHUSDT"]), {chrom.uid: cap}, 12)
    gates.validate([cand])
    rec = gates.timing_null_records[chrom.uid]
    assert rec["exposure"] == 0.9 and rec["exposure_by_symbol"] == {"BTCUSDT": 1.0, "ETHUSDT": 0.8}
    assert rec["fallback"] is None and rec["draws"] == 20
    assert isinstance(rec["percentile"], float)


@pytest.mark.parametrize(
    ("drop", "reason"),
    [
        ((0,), "Timing null: no 4h bar BTCUSDT at window start 2024-03-10T00:00Z"),
        ((7,), "Timing null: bar gap BTCUSDT 4h after 2024-03-11T00:00Z"),
    ],
    ids=["leading-gap", "interior-gap"],
)
def test_bar_gaps_reject(monkeypatch: pytest.MonkeyPatch, drop: tuple[int], reason: str) -> None:
    """GATES-7 leading-gap row: the gate rejects with the kernel's data reason, never passes."""
    _fake_buy_hold(monkeypatch, True)
    chrom = _chrom(3)
    full = _tape("BTCUSDT", _walk(60, 3))
    cand, cap = _candidate(chrom, {"BTCUSDT": [full]}, {"BTCUSDT": [[tn.TradeSpan(1, 0.5, 10, 5)]]})
    gapped = _tape("BTCUSDT", _walk(60, 3), drop=drop)
    cap = dataclasses.replace(cap, tapes={"BTCUSDT": [gapped]})
    gates = _gates(_config(["BTCUSDT"]), {chrom.uid: cap}, 12)
    assert gates.validate([cand]).survivors == []
    (row,) = gates.rejections
    assert (row["stage"], row["reasons"]) == ("timing_null", [reason])


def test_capture_rerun_that_differs_from_the_scored_result_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B12 (B-T2 half): the re-run must reproduce the scored FitnessResult field by field."""
    _fake_buy_hold(monkeypatch, True)
    chrom = _chrom(4)
    tapes = {"BTCUSDT": [_tape("BTCUSDT", _walk(60, 4))]}
    cand, cap = _candidate(
        chrom, tapes, {"BTCUSDT": [[tn.TradeSpan(1, 0.5, 10, 5)]]}, rerun_return=0.21
    )
    gates = _gates(_config(["BTCUSDT"]), {chrom.uid: cap}, 12)
    assert gates.validate([cand]).survivors == []
    (row,) = gates.rejections
    assert row["reasons"] == [
        "Timing null: capture re-run differs from the scored result (total_return)"
    ]


def test_missing_funding_table_rejects_one_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """R9: a candidate whose capture has no funding table rejects at ``timing_null`` with
    that reason; the next candidate is still judged."""
    monkeypatch.setattr(tn, "NULL_DRAWS", 6)
    _fake_buy_hold(monkeypatch, True)
    bad, good = _chrom(7), _chrom(8)
    tape = _tape("BTCUSDT", _walk(60, 7))
    spans = {"BTCUSDT": [[tn.TradeSpan(1, 0.5, 10, 5)]]}
    bad_tape = dataclasses.replace(tape, funding_ts=None, funding_unit=None)
    cand_bad, cap_bad = _candidate(bad, {"BTCUSDT": [bad_tape]}, spans)
    cand_good, cap_good = _candidate(good, {"BTCUSDT": [tape]}, spans)
    gates = _gates(_config(["BTCUSDT"]), {bad.uid: cap_bad, good.uid: cap_good}, 12)
    gates.validate([cand_bad, cand_good])
    reasons = [r["reasons"] for r in gates.rejections if r["stage"] == "timing_null"]
    missing = ["Timing null: no funding table for BTCUSDT window 1"]
    assert reasons.count(missing) == 1  # exactly the candidate without a table
    assert bad.uid not in gates.timing_null_records
    assert gates.timing_null_records[good.uid]["draws"] == 6  # the other one was judged


def test_non_engine_fn_without_capture_rejects_every_candidate() -> None:
    """GATES-16 fail-closed: no NT engine and no capture -> nothing passes."""
    chrom = _chrom(5)
    cand, _ = _candidate(chrom, {"BTCUSDT": [_tape("BTCUSDT", _walk(60, 5))]}, {"BTCUSDT": [[]]})
    gates = GateChain(_config(["BTCUSDT"]), GateEvidence(object(), False))
    assert gates.validate([cand]).survivors == []
    assert gates.rejections[0]["reasons"] == [
        "Timing null: needs the NT engine (non-NT backtest fn)"
    ]


def test_unloadable_closes_reject_without_the_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fail closed when the daily closes cannot load; the log names the type, not the text."""
    chrom = _chrom(6)
    cand, cap = _candidate(chrom, {"BTCUSDT": [_tape("BTCUSDT", _walk(60, 6))]}, {"BTCUSDT": [[]]})

    def boom(symbols: list[str]) -> dict[str, dict[int, float]]:
        raise RuntimeError("MARKER-closes-boom")

    gates = GateChain(
        _config(["BTCUSDT"]),
        GateEvidence(
            SimpleNamespace(symbols=["BTCUSDT"]),
            False,
            buy_hold_closes_fn=boom,
            timing_null_capture_fn=lambda c: cap,
        ),
    )
    with caplog.at_level("WARNING"):
        assert gates.validate([cand]).survivors == []
    assert gates.rejections[0]["reasons"] == [
        "Timing null: benchmark closes unavailable: RuntimeError"
    ]
    assert "MARKER-closes-boom" not in caplog.text
    assert "benchmark closes unavailable: RuntimeError" in caplog.text


def test_record_is_independent_of_pool_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """B7: the same genome gets the same seeds and percentile among 5 or 600 candidates."""
    monkeypatch.setattr(tn, "NULL_DRAWS", 8)
    _fake_buy_hold(monkeypatch, True)
    tape = _tape("BTCUSDT", _walk(60, 6))
    pool: list[tuple[tuple[StrategyChromosome, FitnessResult], CapturedRun]] = []
    for i in range(600):
        frac = 0.1 + (i % 7) / 20
        spans = {"BTCUSDT": [[tn.TradeSpan(1, frac, 3, 6), tn.TradeSpan(-1, frac, 20, 9)]]}
        pool.append(_candidate(_chrom(i), {"BTCUSDT": [tape]}, spans))
    caps = {c.uid: cap for (c, _), cap in pool}
    target = pool[3][0][0]

    def record_of(n: int) -> dict[str, object]:
        gates = _gates(_config(["BTCUSDT"]), caps, 12)
        gates.validate([cand for cand, _ in pool[:n]])
        return gates.timing_null_records[target.uid]

    small, big = record_of(5), record_of(600)
    assert small == big
    key = DiscoveryPipeline._dsl_key(target)
    assert small["seeds"] == {"BTCUSDT": tn.null_seed(7, key, "BTCUSDT")}


# --------------------------------------------------------------------------- persistence


class _MockFn:
    """Mock backtest whose result carries the unshuffled replay series of its spans."""

    def __init__(self, cands: dict[str, tuple[Any, CapturedRun]]) -> None:
        self.cands = cands

    def __call__(self, chrom: StrategyChromosome) -> dict[str, float | int]:
        return dict(self.cands[DiscoveryPipeline._dsl_key(chrom)][1].bt)


def test_rows_carry_the_record_after_later_rejections_and_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B16: a null passer that fails the holdout, a capped survivor and a null rejection all
    carry ``timing_null``; only the null rejection counts as kind ``Timing null``."""
    import vibe_quant.discovery.buyhold as bh

    monkeypatch.setattr(tn, "NULL_DRAWS", 8)
    chroms = [_chrom(10 * i) for i in range(4)]
    tape = _tape("BTCUSDT", _walk(60, 7))
    cands: dict[str, tuple[Any, CapturedRun]] = {}
    for i, c in enumerate(chroms):
        # chrom 3 holds 58 of 60 bars (exposure 0.9667): its verdict is the fake buy-hold's
        hold = 58 if i == 3 else 5
        cand, cap = _candidate(
            c, {"BTCUSDT": [tape]}, {"BTCUSDT": [[tn.TradeSpan(1, 0.5, 1, hold)]]}
        )
        cands[DiscoveryPipeline._dsl_key(c)] = (cand, cap)

    def fake_bh(*a: Any, **kw: Any) -> tuple[bool, dict[str, int], list[str]]:
        return (False, {"bh": 1}, ["Buy-hold: MTM Sharpe 0.10 <= benchmark 0.50"])

    monkeypatch.setattr(bh, "buy_hold_gate", fake_bh)
    real_decision = tn.timing_null_decision
    monkeypatch.setattr(  # every null draw passes; the fallback keeps the real rule
        tn,
        "timing_null_decision",
        lambda policy, rec: (
            (True, []) if rec.get("fallback") is None else real_decision(policy, rec)
        ),
    )
    holdout_fail = DiscoveryPipeline._dsl_key(chroms[0])

    def holdout(chrom: StrategyChromosome) -> dict[str, float | int]:
        bad = DiscoveryPipeline._dsl_key(chrom) == holdout_fail
        return {"sharpe_ratio": -1.0 if bad else 2.0, "max_drawdown": 0.1, "profit_factor": 1.5,
                "total_trades": 30, "total_return": -0.1 if bad else 0.1}  # fmt: skip

    closes = _closes(["BTCUSDT"], "2024-03-10", 12)
    cfg = _config(
        ["BTCUSDT"], population_size=4, max_generations=1, elite_count=1, top_k=1,
        max_workers=None, require_dsr=False, require_bootstrap_ci=False, gate_pool="all",
        train_test_split=0.8, holdout_start_date="2024-04-10", holdout_end_date="2024-05-10",
        holdout_min_trades=1, min_diversity_distance=0.0,
    )  # fmt: skip
    random.seed(1)
    pipe = DiscoveryPipeline(
        cfg,
        _MockFn(cands),
        holdout_backtest_fn=holdout,
        genome_ops=_Ops(chroms),  # type: ignore[arg-type]
        buy_hold_closes_fn=lambda syms: closes,
        timing_null_capture_fn=lambda c: cands[DiscoveryPipeline._dsl_key(c)][1],
    )
    result = pipe.run()
    by_stage = {r["uid"]: r for r in result.guardrail_rejections}
    rows = {
        "timing_null": [r for r in by_stage.values() if r["stage"] == "timing_null"],
        "holdout": [r for r in by_stage.values() if r["stage"] == "holdout"],
        "gate_pool_cap": [r for r in by_stage.values() if r["stage"] == "gate_pool_cap"],
    }
    assert [len(v) for v in rows.values()] == [1, 1, 1]
    for stage_rows in rows.values():
        (row,) = stage_rows
        assert row["timing_null"] == result.timing_null_records[str(row["uid"])]
    assert rows["timing_null"][0]["uid"] == chroms[3].uid
    ((champ, _),) = result.top_strategies
    assert champ.uid in result.timing_null_records
    _, counts = record.rejection_ledger(result.guardrail_rejections, "all")
    assert counts is not None
    assert counts["by_kind"] == {"Holdout": 1, "Timing null": 1, "Top-K cap": 1}
    # a 51st row still counts by kind although only the first 50 rows are kept
    many = [{"stage": "holdout", "reasons": ["Holdout: x"]}] * 50 + rows["timing_null"]
    kept, counts = record.rejection_ledger(many, "all")
    assert len(kept) == 50 and counts is not None
    assert counts["by_kind"] == {"Holdout": 50, "Timing null": 1}


def test_multi_seed_champion_keeps_its_own_seed_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """B17 (reachable multi-seed case): the champion's record is the winning seed's own, though
    a later losing seed writes the same uid (a last-write union would keep the loser's)."""
    from vibe_quant.discovery import pooling

    chrom = _chrom(1)
    rec1, rec2 = {"percentile": 0.97, "seed": 1}, {"percentile": 0.99, "seed": 2}
    fit = _candidate(chrom, {"BTCUSDT": [_tape("BTCUSDT", _walk(60, 8))]}, {"BTCUSDT": [[]]})[0][1]
    better = dataclasses.replace(fit, adjusted_score=fit.adjusted_score + 1.0)
    results = iter(
        [
            DiscoveryResult([], [(chrom, better)], 3, False, None,
                            timing_null_records={chrom.uid: rec1}, timing_null_gate={"enforced": True}),
            DiscoveryResult([], [(chrom.clone(), fit)], 3, False, None,
                            timing_null_records={chrom.uid: rec2}, timing_null_gate={"enforced": True}),
        ]
    )  # fmt: skip

    class FakePipe:
        def __init__(self, **kw: Any) -> None:
            self.fitness_cache: dict[str, FitnessResult] = {}

        def run(self) -> DiscoveryResult:
            return next(results)

    monkeypatch.setattr(pooling, "DiscoveryPipeline", FakePipe)
    cfg = _config(["BTCUSDT"], require_dsr=False, require_bootstrap_ci=False)
    merged = pooling._run_multi_seed(2, cfg, object(), "/tmp/cc-gb2-1/progress.json", base_seed=7)
    assert [c.uid for c, _ in merged.top_strategies] == [chrom.uid]
    assert merged.timing_null_records == {chrom.uid: rec1}
    assert merged.timing_null_gate == {"enforced": True}


def test_pooled_dsr_rejection_keeps_the_selected_seeds_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B16: a pick that passes DSR in its seed (N=450, p=0.0472) and fails the pooled re-check
    (N=900, p=0.0709) persists its pooled_guardrails row with the SELECTED seed's timing-null
    record, not the later seed's write of the same uid (rev-gb2-1 repro_pooled_rejection)."""
    from tests.unit.test_discovery_dsr_pooled import _fit, _series
    from vibe_quant.discovery import pooling
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.discovery.gate_chain import TrialBudget

    cfg = _config(["BTCUSDT"], start_date="2022-06-01", end_date="2025-11-17",
                  require_bootstrap_ci=False, gate_pool="all", dsr_rule="emax")  # fmt: skip
    nt = NTBacktestFn(cfg.symbols, cfg.timeframe, cfg.start_date, cfg.end_date)
    chrom = _chrom(1)
    fit = _fit(2.1, mtm_daily_returns=_series(2.1))
    fitness_cache = {chrom.uid: fit}
    seed_gate = GateChain(cfg, GateEvidence(nt, True))
    assert seed_gate.screen([(chrom, fit)], TrialBudget(450, fitness_cache))
    assert seed_gate.dsr_records[chrom.uid]["p_value"] == 0.04717929858610217
    selected, loser = {"percentile": 0.97}, {"percentile": 0.99}
    worse = dataclasses.replace(fit, adjusted_score=fit.adjusted_score - 0.1)
    results = iter([(fit, selected), (worse, loser)])

    class Seed:
        def __init__(self, **kw: Any) -> None:
            self.fitness_cache = fitness_cache

        def run(self) -> DiscoveryResult:
            f, rec = next(results)
            return DiscoveryResult([], [(chrom.clone(), f)], 450, False, None,
                                   timing_null_records={chrom.uid: rec},
                                   timing_null_gate={"enforced": True})  # fmt: skip

    monkeypatch.setattr(pooling, "DiscoveryPipeline", Seed)
    merged = pooling._run_multi_seed(2, cfg, nt, "/tmp/impl-gb2-2/progress.json", base_seed=7)
    rows, counts = record.rejection_ledger(merged.guardrail_rejections, "all")
    (row,) = rows
    assert not merged.top_strategies
    assert row["stage"] == "pooled_guardrails"
    assert row["dsr"]["p_value"] == 0.07085404751487628  # type: ignore[index]
    assert counts is not None and counts["by_kind"] == {"DSR": 1}
    assert row["timing_null"] == selected


# --------------------------------------------------------------------------- capture + launch


def _pos(side: str, qty: float, px: float, opened: int) -> Any:
    return SimpleNamespace(
        entry=SimpleNamespace(name=side), peak_qty=qty, avg_px_open=px, ts_opened=opened
    )


def _costed(pos: Any, exit_ns: int, realized: float, funding: float = 0.0, mark: Any = None) -> Any:
    return SimpleNamespace(
        pos=pos, mark=mark, realized_pnl=realized, exit_ns=exit_ns, slippage_cost=0.0,
        accrual=SimpleNamespace(total=funding) if funding else None,
    )  # fmt: skip


def test_spans_size_from_the_pre_entry_costed_ledger() -> None:
    """GATES-6: entry equity = B0 + net PnL (after fees and funding) of trades closed at or
    before the entry ts (a same-ts close-and-reopen counts); GATES-7: a hold crossing the
    half-open window end is clipped and counted apart from zero holds."""
    tape = _tape("BTCUSDT", _walk(60, 9))
    t = tape.open_ns
    a = _costed(_pos("BUY", 1.0, 100.0, int(t[2]) + 60_000 * MS), int(t[5]) + MS, 52.0, 2.0)
    b = _costed(_pos("SELL", 2.0, 101.0, int(t[5]) + MS), int(t[5]) + 2 * MS, -10.0)  # same bar
    c = _costed(_pos("BUY", 0.5, 99.0, int(t[8])), tape.end_ns + H4, 5.0)  # exits past the end
    d = _costed(_pos("SELL", 1.5, 98.0, int(t[50])), int(t[59]), 1.0, mark=object())  # carried
    spans, clipped = tn.spans_from_ledger([c, a, d, b], tape, B0)
    assert spans == [
        tn.TradeSpan(1, 100.0 / 1000.0, 2, 3),
        tn.TradeSpan(-1, 202.0 / 1050.0, 5, 0),
        tn.TradeSpan(1, 49.5 / 1040.0, 8, 52),
        tn.TradeSpan(-1, 147.0 / 1040.0, 50, 10, True),
    ]
    assert clipped == 1


def test_spans_never_size_from_their_own_or_a_later_exit() -> None:
    """GATES-6 equality edge, event order = ``ts_opened`` then input order: a round trip that
    opens and closes on one ts never feeds its own size; the next position opened on that ts
    (netting: the close filled first) does see it; a later-ordered exit never counts."""
    tape = _tape("BTCUSDT", _walk(60, 9))
    t3, t4 = int(tape.open_ns[3]), int(tape.open_ns[4])
    p = _costed(_pos("BUY", 1.0, 100.0, t3), t3, -7.0)  # opens and closes on t3
    q = _costed(_pos("SELL", 2.0, 100.0, t3), t4, 3.0)  # reopened on t3 after p closed
    spans, _ = tn.spans_from_ledger([p, q], tape, B0)
    assert [s.frac for s in spans] == [100.0 / 1000.0, 200.0 / 993.0]
    spans, _ = tn.spans_from_ledger([q, p], tape, B0)  # q first in event order: p never feeds q
    assert [s.frac for s in spans] == [200.0 / 1000.0, 100.0 / 1000.0]


def test_real_engine_last_bar_round_trip_sizes_from_pre_entry_equity(tmp_path: Path) -> None:
    """GATES-6 on the real 1m engine (rev-gb2-1 repro_last_bar_equity): the entry fills on the
    last bar's close, the run-end close shares its ts; its -0.0923156 exit PnL must not size it."""
    from tests.unit.test_reverse_runners import reverse_dsl
    from tests.unit.test_screening_fill_mode import _write_catalog
    from vibe_quant.screening import nt_runner

    catalog = _write_catalog(tmp_path, n_min=1440, timeframes=("1m",))
    dsl = reverse_dsl("timing_null_last_bar", tf="1m", reverse=None,
                      long=("close > 11438",), short=("close < 0",))  # fmt: skip
    runner = nt_runner.NTScreeningRunner(
        dsl, ["BTCUSDT"], "2024-01-01", "2024-01-02", catalog_path=str(catalog),
        collect_daily_returns=True, capture_spans=True,
    )  # fmt: skip
    priced: list[Any] = []
    real_cost = nt_runner.cost_positions

    def spy(*a: Any, **kw: Any) -> Any:
        out = real_cost(*a, **kw)
        priced.extend(out)
        return out

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nt_runner, "cost_positions", spy)
        runner({})
    (item,) = priced
    assert runner.last_trade_spans is not None
    tape, (span,), _ = runner.last_trade_spans["BTCUSDT"]
    assert int(item.pos.ts_opened) == int(item.exit_ns) == int(tape.ts[-1]) == 1704153599999000000
    assert item.realized_pnl == -0.0923156
    assert (span.entry_bar, span.hold_bars, span.carried) == (len(tape) - 1, 0, False)
    assert span.frac == 0.091516


@pytest.mark.parametrize(
    ("extra", "problem"),
    [
        (["--symbols", "BTCUSDT,ETHUSDT"],
         "--null-gate needs one symbol or --symbol-agg worst (a shared account couples "
         "symbol sizing)"),
        (["--liquidation-mode", "isolated"],
         "--null-gate needs --liquidation-mode off (liquidated trades have no re-placeable exit)"),
        (["--bankroll-modes", "isolated", "--bankroll-leverages", "5",
          "--bankroll-max-margin-pcts", "0.5"],
         "--null-gate cannot run a bankroll grid (its cells run liquidation on)"),
        ([], "--null-gate needs real NT backtests (no --mock, no --evaluator fast)"),
        (["--timeframe", "2m"],
         "--null-gate needs a timeframe with a bar tape (1s|5s|1m|5m|15m|1h|4h|1d), got 2m"),
        (["--symbols", "BTCUSDT,ETHUSDT", "--symbol-agg", "pooled"],
         "--null-gate cannot judge --symbol-agg pooled (no pooled timing null yet)"),
    ],
    ids=["portfolio-2-symbols", "liquidation-on", "bankroll-grid", "mock", "no-tape-timeframe",
         "pooled"],
)  # fmt: skip
def test_launch_refusals_before_any_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: list[str], problem: str
) -> None:
    """GATES-6 + B12 texts: the real ``--null-gate`` flag is refused with exit 2 before a
    pipeline exists, and the refusal writes no strategies or backtest_runs row."""
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import run
    from vibe_quant.discovery.run_spec import build_parser
    from vibe_quant.jobs.manager import BacktestJobManager

    monkeypatch.setattr(run, "DiscoveryPipeline", lambda *a, **kw: pytest.fail("pipeline built"))
    db = tmp_path / "state.db"
    state = StateManager(db)
    run_id = state.create_backtest_run(None, "discovery", ["BTCUSDT"], "4h", START, END, {})
    # pooled refuses --mock first (its own launch check), so that row launches real NT
    mock = [] if "pooled" in extra else ["--mock"]
    argv = [*mock, "--seed", "7", "--start-date", "2024-01-01", "--end-date", "2025-01-01",
            "--timeframe", "4h", "--symbols", "BTCUSDT", "--db", str(db), "--run-id", str(run_id),
            "--null-gate", *extra]  # fmt: skip
    args = build_parser().parse_args(argv)
    counts = _row_counts(state)
    jobs = BacktestJobManager(db)
    try:
        outcome = run.run_discovery(
            args, run.MockBacktests() if args.mock else run.NTBacktests(),
            run.StateRecordStore(state, jobs), run_id=run_id, started_at=0.0,
        )  # fmt: skip
        assert _row_counts(state) == counts == (0, 1)
    finally:
        state.close()
        jobs.close()
    assert outcome.exit_code == 2
    assert problem in outcome.message


def _row_counts(state: Any) -> tuple[int, int]:
    conn = state.conn
    return (
        int(conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]),
        int(conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]),
    )
