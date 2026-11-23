"""G-null harness (clmmc.14 B-T4): seeded zero-edge controls for the timing-null gate.

``scripts/research/power/timing_null_power.py`` draws a shared synthetic market tape per
replicate FIRST, then zero-edge trade schedules independent of it, plays the real side
through the production as-traded kernel and scores the production null and gate chain.
Pinned here: the tape-before-schedule determinism (design arch-gates-2 §5 G-null), the
as-traded real-side costs incl. the review's flat-short funding probe (GATES-9), the
five-trade triple fixture (GATES-10), the verdict rules (R-gb4-1) and the quick stream.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "power"))

import timing_null_power as tnp  # noqa: E402

from vibe_quant.discovery import timing_null as tn  # noqa: E402

H4 = 4 * 3_600 * 10**9
DAY = 86_400 * 10**9
MS = 10**6


def _cfg(key: str) -> tnp.Config:
    return next(c for c in tnp.configs() if c.key == key)


def _flat_tape(n: int, start_day: int = 20_000) -> tn.WindowTape:
    start = start_day * DAY
    on = start + np.arange(n, dtype=np.int64) * H4
    px = np.full(n, 100.0)
    return tn.WindowTape("BTCUSDT", "4h", H4, start, start + n * H4, on, on + H4 - MS, px, px)


def _same_tapes(a: tnp.Tapes, b: tnp.Tapes) -> bool:
    if list(a.train) != list(b.train):
        return False
    for sym in a.train:
        for x, y in zip(a.train[sym], b.train[sym], strict=True):
            if not (np.array_equal(x.open, y.open) and np.array_equal(x.close, y.close)):
                return False
        if not np.array_equal(a.holdout[sym].open, b.holdout[sym].open):
            return False
    return True


# --------------------------------------------------------------------------- seeds and tape


def test_shared_tape_is_drawn_first_and_never_depends_on_the_schedules() -> None:
    cfg = _cfg("W3:e=0.1:6async")
    seq = tnp.replicate_seq(cfg, 3, tnp.TAG)
    assert list(seq.entropy) == [20261122, 0x7141, cfg.index, 3]
    alone = tnp.draw_tapes(cfg, seq)
    tapes_1, scheds_1 = tnp.replicate(cfg, 3, tnp.TAG, candidates=1)
    tapes_5, scheds_5 = tnp.replicate(cfg, 3, tnp.TAG, candidates=5)
    assert _same_tapes(alone, tapes_1) and _same_tapes(alone, tapes_5)
    assert scheds_1[0] == scheds_5[0]  # candidate 0 does not depend on how many follow
    assert scheds_5[0] != scheds_5[1]
    again_tapes, again = tnp.replicate(cfg, 3, tnp.TAG, candidates=5)
    assert _same_tapes(again_tapes, tapes_5) and again == scheds_5
    other, _ = tnp.replicate(cfg, 4, tnp.TAG, candidates=1)
    assert not _same_tapes(other, tapes_5)
    topup, _ = tnp.replicate(cfg, 3, tnp.TOPUP_TAG, candidates=1)
    assert not _same_tapes(topup, tapes_5)


def test_streams_are_frozen_and_quick_never_touches_the_binding_ones() -> None:
    assert (tnp.SEED_BASE, tnp.TAG, tnp.TOPUP_TAG, tnp.QUICK_TAG) == (
        20261122,
        0x7141,
        0x7142,
        0x71A0,
    )
    assert tnp.stream_tag(quick=True, topup=False) == 0x71A0
    assert tnp.stream_tag(quick=True, topup=True) == 0x71A0
    assert tnp.stream_tag(quick=False, topup=False) == 0x7141
    assert tnp.stream_tag(quick=False, topup=True) == 0x7142
    keys = [c.key for c in tnp.configs()]
    assert len(keys) == 12 and [c.index for c in tnp.configs()] == list(range(12))
    assert keys[:3] == ["W3:e=0.1:1sym", "W3:e=0.1:6sync", "W3:e=0.1:6async"]
    assert keys[-1] == "W6:e=0.3:6async"


def test_tape_is_a_correlated_zero_drift_t5_market_at_sixty_percent_vol() -> None:
    cfg = _cfg("W3:e=0.3:6sync")
    tapes = tnp.draw_tapes(cfg, tnp.replicate_seq(cfg, 0, tnp.QUICK_TAG))
    assert list(tapes.train) == list(tnp.SYMBOLS6)
    logs = []
    for sym in tnp.SYMBOLS6:
        w1, w2 = tapes.train[sym]
        assert len(w1) == 438 * 6 and len(w2) == 438 * 6  # 2023-06-01..2024-08-12..2025-10-24
        assert w2.open[0] == w1.close[-1] and w1.end_ns == w2.start_ns  # one continuous path
        tn.validate_tape(w1)
        tn.validate_tape(w2)
        assert len(tapes.holdout[sym]) == 365 * 6 and tapes.holdout[sym].start_ns == w2.end_ns
        path = np.concatenate([w1.open, w2.open, [w2.close[-1]]])
        logs.append(np.diff(np.log(path)))
    x = np.vstack(logs)
    ann_vol = x.std(axis=1) * np.sqrt(365 * 6)
    assert np.all(np.abs(ann_vol - 0.60) < 0.03)
    corr = np.corrcoef(x)[np.triu_indices(6, 1)]
    assert np.all(np.abs(corr - 0.6) < 0.05)
    assert np.all(np.abs(x.mean(axis=1)) < 3 * x.std(axis=1) / np.sqrt(x.shape[1]))
    one = tnp.draw_tapes(_cfg("W6:e=0.1:1sym"), tnp.replicate_seq(_cfg("W6:e=0.1:1sym"), 0, 1))
    assert list(one.train) == ["BTCUSDT"] and one.train["BTCUSDT"][0].tf_ns == DAY


@pytest.mark.parametrize(("key", "mean_hold"), [("W3:e=0.3:6sync", 6.0), ("W6:e=0.1:6async", 2.0)])
def test_zero_edge_schedule_law(key: str, mean_hold: float) -> None:
    cfg = _cfg(key)
    tapes, scheds = tnp.replicate(cfg, 0, tnp.QUICK_TAG, candidates=6)
    holds, longs, fracs = [], 0, []
    for sched in scheds:
        per_sym = [sched.train[s] for s in tnp.SYMBOLS6]
        if cfg.mode == "6sync":
            assert all(p == per_sym[0] for p in per_sym)
        else:
            assert per_sym[0] != per_sym[1]
        for sym in tnp.SYMBOLS6:
            for tape, spans in zip(tapes.train[sym], sched.train[sym], strict=True):
                held = sum(sp.hold_bars for sp in spans)
                assert cfg.exposure * len(tape) - 60 < held <= round(cfg.exposure * len(tape))
                ends = [sp.entry_bar + sp.hold_bars for sp in spans]
                assert all(e <= s.entry_bar for e, s in zip(ends, spans[1:], strict=False))
                holds += [sp.hold_bars for sp in spans]
                longs += sum(sp.side == 1 for sp in spans)
                fracs += [sp.frac for sp in spans]
            assert sched.holdout[sym]  # the holdout tape gets its own schedule
    assert abs(np.mean(holds) - mean_hold) < 0.15 * mean_hold
    assert abs(longs / len(holds) - 0.6) < 0.03
    assert min(fracs) >= 0.05 and max(fracs) <= 0.5


# --------------------------------------------------------------------------- costs (GATES-9)


def test_flat_short_funding_credit_is_charged_on_the_real_side_only() -> None:
    # The review's probe: on a flat market a short with a positive funding rate is credited.
    # Fee-only, it loses both fees and every null draw ties it (percentile 0.0); with the
    # credit it beats every fee-only null draw (percentile 1.0) although it has no edge, so
    # a fee-only harness understates the production false-positive rate.
    tape = _flat_tape(60)
    short = [tn.TradeSpan(-1, 1.0, 10, 24)]  # 4 days: crosses 12 settlements (8h grid)
    fee_only = tn.sim_daily(tape, short, tnp.real_costs("fee"), 1000.0)
    assert tn.total_return(fee_only.values) == pytest.approx(-0.001, rel=1e-12)
    credited = tnp.real_costs("credit")
    got = tn.total_return(tn.sim_daily(tape, short, credited, 1000.0).values)
    assert got == pytest.approx((12 * 0.1 - 1.0) / 1000.0, rel=1e-12)
    closes = tnp.daily_closes({"BTCUSDT": [tape]})
    sched = tnp.Schedule(train={"BTCUSDT": [short]}, holdout={"BTCUSDT": []})
    reals = {}
    for name in ("fee", "credit"):
        real = tn._chain([tn.sim_daily(tape, short, tnp.real_costs(name), 1000.0)])
        assert real is not None
        reals[name] = real
    record, null = tnp.null_record(
        {"BTCUSDT": [tape]},
        sched,
        reals["fee"],
        seed_base=1,
        key="cand0",
        closes=closes,
        buy_hold=(True, []),
        draws=20,
    )
    assert record["fallback"] is None and record["percentile"] == 0.0
    # the credit line is re-ranked on the SAME null draws, never handed the fee percentile
    credit = tnp.ranked(record, null, reals["credit"])
    assert credit["percentile"] == tn.null_percentile(tn.total_return(reals["credit"].values), null)
    assert credit["percentile"] == 1.0 != record["percentile"]
    assert credit["real_T"] == pytest.approx((12 * 0.1 - 1.0) / 1000.0, rel=1e-12)
    assert tnp.ranked(record, null, reals["fee"])["percentile"] == 0.0


@pytest.mark.parametrize("side", [1, -1])
def test_credit_case_credits_every_trade_and_pays_only_the_taker_fee(side: int) -> None:
    # credit case (R-gb4-3): BOTH sides receive 0.0001 per 8h on held notional, taker fee per
    # leg, no slippage; the null stays fee-only (production)
    tape = _flat_tape(60)
    costs = tnp.real_costs("credit")
    assert tn.TradeCosts(fee=0.0005) == tnp.NULL_COSTS and costs.fee == 0.0005
    assert costs.fill_price is None
    spans = [tn.TradeSpan(side, 0.5, 10, 24)]  # notional 500, 12 settlements
    got = tn.total_return(tn.sim_daily(tape, spans, costs, 1000.0).values)
    assert got == pytest.approx((12 * 0.0001 * 500.0 - 2 * 0.0005 * 500.0) / 1000.0, rel=1e-12)
    assert tnp.funding_credit_hook(0.0001)("BTCUSDT", side, 500.0, 0, 16 * 3_600 * 10**9) == [
        (8 * 3_600 * 10**9, -0.05),
        (16 * 3_600 * 10**9, -0.05),
    ]


def test_credit_case_favours_the_real_side_on_every_replicate() -> None:
    # the reviewer's sign check made permanent: a case that claims to bound FP from above must
    # raise the real side's T over fee-only on held positions (own throwaway stream 0x5C5C)
    cfg = _cfg("W3:e=0.3:1sym")
    diffs = []
    for rep in range(20):
        seq = tnp.replicate_seq(cfg, rep, 0x5C5C)
        tapes = tnp.draw_tapes(cfg, seq)
        sched = tnp.draw_schedule(cfg, seq, 0, tapes)
        t = {
            c: tn.total_return(tnp.real_series(cfg, tapes, sched, c).values)
            for c in ("fee", "credit")
        }
        diffs.append(t["credit"] - t["fee"])
    assert np.mean(diffs) > 0
    assert np.mean(np.asarray(diffs) > 0) == 1.0


def test_trade_pnls_add_up_to_the_as_traded_daily_path() -> None:
    cfg = _cfg("W3:e=0.3:1sym")
    tapes, scheds = tnp.replicate(cfg, 1, tnp.QUICK_TAG, candidates=1)
    tape, spans = tapes.train["BTCUSDT"][0], scheds[0].train["BTCUSDT"][0]
    costs = tnp.real_costs("credit")
    trades = tnp.trade_pnls(tape, spans, costs, 1000.0)
    assert len(trades) == len(spans) > 50
    end = 1000.0 * (1.0 + tn.total_return(tn.sim_daily(tape, spans, costs, 1000.0).values))
    assert 1000.0 + sum(pnl for _eq, pnl, _ts in trades) == pytest.approx(end, rel=1e-12)
    bt = tnp.window_bt(tape, spans, costs)
    assert bt["total_trades"] == len(spans)
    assert bt["total_return"] == pytest.approx(end / 1000.0 - 1.0, rel=1e-12)
    assert bt["trade_returns"][0] == pytest.approx(trades[0][1] / 1000.0, rel=1e-12)


# --------------------------------------------------------------------------- null (GATES-10)


def test_five_trade_fixture_keeps_triples_and_the_sixty_forty_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # three longs, two shorts; long holds 2+4+6 = 12 of 20 held bars: 60/40 by count AND by
    # elapsed exposure. Every production null draw keeps the multiset of (hold, side, frac).
    spans = [
        tn.TradeSpan(1, 0.1, 0, 2),
        tn.TradeSpan(-1, 0.4, 5, 3),
        tn.TradeSpan(1, 0.2, 10, 4),
        tn.TradeSpan(-1, 0.5, 20, 5),
        tn.TradeSpan(1, 0.3, 30, 6),
    ]
    triples = Counter((s.hold_bars, s.side, s.frac) for s in spans)
    seen: list[list[tn.TradeSpan]] = []
    real_sim = tn.sim_daily

    def spy(tape: tn.WindowTape, placed: Any, costs: tn.TradeCosts, b0: float) -> Any:
        seen.append(list(placed))
        return real_sim(tape, placed, costs, b0)

    rng = np.random.default_rng(2)
    path = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(rng.normal(0, 0.01, 60))]))
    start = 20_000 * DAY
    on = start + np.arange(60, dtype=np.int64) * H4
    tape = tn.WindowTape(
        "BTCUSDT", "4h", H4, start, start + 60 * H4, on, on + H4 - MS, path[:-1], path[1:]
    )
    sched = tnp.Schedule(train={"BTCUSDT": [spans]}, holdout={"BTCUSDT": []})
    real = tn._chain([tn.sim_daily(tape, spans, tnp.real_costs("fee"), 1000.0)])
    assert real is not None
    monkeypatch.setattr(tn, "sim_daily", spy)
    record, null = tnp.null_record(
        {"BTCUSDT": [tape]},
        sched,
        real,
        seed_base=7,
        key="cand0",
        closes=tnp.daily_closes({"BTCUSDT": [tape]}),
        buy_hold=(True, []),
        draws=40,
    )
    assert len(seen) == 40 and len(null) == 40
    for placed in seen:
        assert Counter((s.hold_bars, s.side, s.frac) for s in placed) == triples
        assert sum(s.side == 1 for s in placed) == 3
        assert sum(s.hold_bars for s in placed if s.side == 1) == 12
        assert sum(s.hold_bars for s in placed) == 20
    assert record["seeds"] == {"BTCUSDT": tn.null_seed(7, "cand0", "BTCUSDT")}
    assert record["percentile"] == tn.null_percentile(tn.total_return(real.values), null)


def test_null_seeds_follow_the_production_seed_function() -> None:
    cfg = _cfg("W6:e=0.1:6async")
    tapes, scheds = tnp.replicate(cfg, 0, tnp.QUICK_TAG, candidates=2)
    for i, key, base in ((0, "cand0", 11), (1, "cand1", 2**40 + 3)):
        fit = tnp.candidate_fitness(cfg, tapes, scheds[i], "fee")
        real = fit.mtm_daily_returns or tnp.real_series(cfg, tapes, scheds[i], "fee")
        record, _null = tnp.null_record(
            tapes.train,
            scheds[i],
            real,
            seed_base=base,
            key=key,
            closes=tapes.closes,
            buy_hold=(True, []),
            draws=3,
        )
        assert record["seeds"] == {s: tn.null_seed(base, key, s) for s in tnp.SYMBOLS6}
        assert record["draws"] == 3


# --------------------------------------------------------------------------- verdicts (R-gb4-1)


def test_calibration_rule_is_the_bonferroni_one_sided_binomial_test() -> None:
    # R-gb4-2: 24 lines (12 configs x fee/credit), alpha 0.05 / 24 each
    assert tnp.CALIBRATION_LINES == 24 and tnp.CALIBRATION_K == 34
    assert binom.sf(33, 400, 0.05) == 0.0020688826558905244 <= 0.05 / 24 < binom.sf(32, 400, 0.05)
    assert tnp.calibration_verdict({"a": 33, "b": 0}, 400).status == "PASS"
    failed = tnp.calibration_verdict({"a": 34, "b": 0}, 400)
    assert (failed.status, failed.exit_code, failed.failing) == ("FAIL", 1, ("a",))
    assert tnp.calibration_verdict({"a": 0}, 10).exit_code == 2  # quick: never binding


def test_selection_rule_is_the_g_pool_two_look_table() -> None:
    assert tnp.selection_verdict({"a": 3}, 200).status == "PASS"
    assert tnp.selection_verdict({"a": 11}, 200).status == "FAIL"
    assert tnp.selection_verdict({"a": 4}, 200).exit_code == 2
    assert tnp.selection_verdict({"a": 4}, 200, {"a": (27, 800)}).status == "PASS"
    assert tnp.selection_verdict({"a": 10}, 200, {"a": (28, 800)}).status == "FAIL"


def test_quick_stream_is_never_a_verdict_even_at_binding_sample_sizes() -> None:
    # a quick-stream wall run at 400/200 reps printed a binding FAIL before this rule
    failing = {"W3:e=0.3:6sync": {"reps": 400, "null_pass": {"fee": 99, "credit": 86}}}
    kw: dict[str, Any] = {
        "keys": ["W3:e=0.3:6sync"],
        "single_reps": 400,
        "selection_reps": 200,
        "candidates": 167,
        "single": failing,
        "selection": {},
        "topup": {},
    }
    quick = tnp.build_report(quick=True, tag=tnp.QUICK_TAG, **kw)["verdict"]
    assert (quick["status"], quick["exit_code"]) == ("MARGINAL", 2)
    assert quick["lines"]["calibration:fee"]["status"] == "FAIL"  # the reading stays visible
    binding = tnp.build_report(quick=False, tag=tnp.TAG, **kw)["verdict"]
    assert (binding["status"], binding["exit_code"]) == ("FAIL", 1)


def test_quick_smoke_runs_the_chain_on_the_quick_stream(tmp_path: Path) -> None:
    out = tmp_path / "gnull.json"
    code = tnp.main(
        [
            "--quick",
            "--configs",
            "W6:e=0.1:1sym",
            "--single-reps",
            "1",
            "--selection-reps",
            "1",
            "--candidates",
            "3",
            "--workers",
            "1",
            "--out",
            str(out),
        ]
    )
    report = json.loads(out.read_text())
    assert code == 2 and report["quick"] is True and report["seed_tag"] == 0x71A0
    assert report["configs"] == ["W6:e=0.1:1sym"]
    single = report["single"]["W6:e=0.1:1sym"]
    assert single["reps"] == 1 and set(single["null_pass"]) == {"fee", "credit"}
    selection = report["selection"]["W6:e=0.1:1sym"]
    assert selection["reps"] == 1
    assert set(selection["fp"]) == {
        "replacement:fee",
        "replacement:credit",
        "combined:fee",
        "combined:credit",
    }
    assert report["verdict"]["status"] == "MARGINAL"
    rerun = tmp_path / "again.json"
    tnp.main(
        [
            "--quick",
            "--configs",
            "W6:e=0.1:1sym",
            "--single-reps",
            "1",
            "--selection-reps",
            "1",
            "--candidates",
            "3",
            "--workers",
            "1",
            "--out",
            str(rerun),
        ]
    )
    assert rerun.read_text() == out.read_text()  # deterministic, no wall clock in the JSON


def test_non_binding_sizes_are_forced_onto_the_quick_stream(tmp_path: Path) -> None:
    # without --quick, a run at other sizes than 400 / 200 / K = 167 would draw binding
    # replicates 0..n (a peek): it runs on the quick stream, writes no binding file, no verdict
    out = tmp_path / "small.json"
    code = tnp.main(
        [
            "--configs",
            "W6:e=0.1:1sym",
            "--single-reps",
            "1",
            "--selection-reps",
            "1",
            "--candidates",
            "3",
            "--workers",
            "1",
            "--out",
            str(out),
        ]
    )
    report = json.loads(out.read_text())
    assert (code, report["seed_tag"], report["quick"]) == (2, 0x71A0, True)
    assert report["verdict"]["status"] == "MARGINAL"
    assert tnp.run_stream(False, 400, 200, 167) == (False, 0x7141)
    assert tnp.run_stream(False, 400, 200, 166) == (True, 0x71A0)
    assert tnp.run_stream(False, 399, 200, 167) == (True, 0x71A0)
    assert tnp.run_stream(False, 400, 201, 167) == (True, 0x71A0)
    assert tnp.run_stream(True, 400, 200, 167) == (True, 0x71A0)


def test_selection_counts_a_replicate_when_any_train_passer_clears_a_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # any-pick-of-all: only the LAST pick in score order clears the replacement chain on the
    # credit case; the replicate counts for exactly that line, and every pick reached the chain
    cfg = _cfg("W3:e=0.3:1sym")
    seen: list[tuple[str, float | None]] = []

    def fake(
        chrom: Any,
        fit: Any,
        holdout: Any,
        windows: Any,
        closes: Any,
        settings: Any,
        *,
        null_spread: float | None = None,
        timing_null: Any = None,
    ) -> dict[str, Any]:
        case = "credit" if seen and len(seen) >= picks_fee[0] else "fee"
        seen.append((case, null_spread))
        last = case == "credit" and len(seen) == picks_fee[0] + picks_credit[0]
        return {
            "chain_null_replacement": {167: last},
            "chain_null_combined": {167: False},
        }

    real = tnp.selection_replicate(cfg, 0, tnp.QUICK_TAG, 12)
    picks_fee, picks_credit = [real["picks"]["fee"]], [real["picks"]["credit"]]
    assert picks_fee[0] >= 1 and picks_credit[0] >= 1
    monkeypatch.setattr(tnp.gatechain, "evaluate_candidate", fake)
    out = tnp.selection_replicate(cfg, 0, tnp.QUICK_TAG, 12)
    assert out["fp"] == {
        "replacement:fee": False,
        "replacement:credit": True,
        "combined:fee": False,
        "combined:credit": False,
    }
    assert len(seen) == picks_fee[0] + picks_credit[0]
    assert all(spread is not None for _case, spread in seen)  # the q95 rule's own spread
