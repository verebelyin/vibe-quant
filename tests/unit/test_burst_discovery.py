"""Tests for scripts/agents/burst.py — the scratch-DB burst discovery grid."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "burst.py"
_spec = importlib.util.spec_from_file_location("burst", _SCRIPT)
assert _spec is not None and _spec.loader is not None
burst = importlib.util.module_from_spec(_spec)
sys.modules["burst"] = burst
_spec.loader.exec_module(burst)


def _eval(trades: int, ret: float, sharpe: float, error: str | None = None) -> dict[str, Any]:
    return {"trades": trades, "ret": ret, "sharpe": sharpe, "pf": 1.0, "dd": 0.1, "error": error, "genome": "g"}


def test_summarize_counts_only_profitable_genomes_over_the_trade_floor() -> None:
    evals = [
        _eval(49, 0.5, 9.0),  # profitable but under the 50-trade floor
        _eval(50, 0.01, 1.2),
        _eval(400, 0.03, 2.5),
        _eval(1500, -0.6, -4.0),
        _eval(0, 0.0, -1.0, error="boom"),
    ]
    s = burst.summarize(evals)
    assert (s["n"], s["n_err"], s["n_low"], s["n_active"], s["n_win"]) == (5, 1, 1, 3, 2)
    assert s["best"]["sharpe"] == 2.5
    assert s["median_trades"] == 225.0


def test_summarize_has_no_best_without_a_winner() -> None:
    assert burst.summarize([_eval(1500, -0.6, -4.0)])["best"] is None


def test_sample_is_a_deterministic_prefix_of_one_shuffle() -> None:
    small = burst.sample_cells(10, sample_seed=3)
    big = burst.sample_cells(40, sample_seed=3)
    assert big[:10] == small
    keys = [burst.cell_key(c) for c in burst.all_cells(seeds=2)]
    assert len(keys) == len(set(keys))


def test_discovery_argv_uses_the_scratch_db_and_the_cell() -> None:
    cell = {**burst.sample_cells(1, 1)[0], "pop": 6, "gens": 1}
    argv = burst._discovery_argv(cell, 7, Path("/tmp/x/state.db"))
    flag = {argv[i]: argv[i + 1] for i in range(1, len(argv) - 1) if argv[i].startswith("--")}
    assert flag["--db"] == "/tmp/x/state.db"
    assert flag["--symbols"] == cell["symbol"]
    assert (flag["--start-date"], flag["--end-date"]) == (cell["start"], cell["end"])
    assert flag["--leverage"] == str(cell["leverage"])
    assert flag["--indicator-pool"] == ",".join(cell["pair"])
    assert (flag["--population-size"], flag["--max-generations"]) == ("6", "1")


def test_cell_refuses_the_real_state_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VIBE_QUANT_DB", str(burst.REPO_ROOT / "data" / "state" / "vibe_quant.db"))
    with pytest.raises(SystemExit, match="real state DB"):
        burst.run_cell(burst.sample_cells(1, 1)[0], tmp_path / "r.json")


def test_run_skips_cells_already_in_the_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cells = burst.sample_cells(3, 1)
    (tmp_path / "ledger.jsonl").write_text(json.dumps({"key": burst.cell_key(cells[0])}) + "\n")
    launched: list[str] = []

    class _Done:
        returncode = 0

        def poll(self) -> int:
            return 0

    def fake_launch(cell: dict[str, Any]) -> tuple[_Done, Path, float]:
        launched.append(burst.cell_key(cell))
        return _Done(), tmp_path, 0.0

    def fake_finish(cell: dict[str, Any], tmp: Path, rc: int | None, wall: float) -> dict[str, Any]:
        return {"key": burst.cell_key(cell), "n": 1, "n_win": 0, "n_low": 0, "best": None}

    monkeypatch.setattr(burst, "_launch", fake_launch)
    monkeypatch.setattr(burst, "_finish", fake_finish)
    monkeypatch.setattr(burst.time, "sleep", lambda _s: None)
    args = argparse.Namespace(
        out=str(tmp_path), cells=3, parallel=2, budget_s=60, sample_seed=1, seeds=1, pop=6, gens=1,
        tier="nt", pairs="", directions="", symbols="", windows="", leverages="", eval_windows=1,
    )
    assert burst.run(args) == 0
    assert launched == [burst.cell_key(c) for c in cells[1:]]
    rows = burst.load_ledger(tmp_path / "ledger.jsonl")
    assert [r["key"] for r in rows] == [burst.cell_key(c) for c in cells]


def test_fast_tier_cells_get_their_own_ledger_key() -> None:
    cell = burst.sample_cells(1, 1)[0]
    nt, fast = burst.with_budget([cell], 6, 1, "nt")[0], burst.with_budget([cell], 6, 1, "fast")[0]
    assert burst.cell_key(nt) == burst.cell_key(cell)
    assert burst.cell_key(fast) == burst.cell_key(cell) + "|fast"


def test_months_tile_the_range_with_exclusive_ends() -> None:
    assert burst.MONTHS[0][0] == "2026-03-01"
    assert burst.MONTHS[-1][1] == "2026-10-20"
    assert all(a[1] == b[0] for a, b in zip(burst.MONTHS, burst.MONTHS[1:], strict=False))


def test_grid_overrides_and_multi_window_keys() -> None:
    args = argparse.Namespace(
        pairs="KAMA+NATR", directions="both", symbols="ZECUSDT,NEARUSDT",
        windows="2026-03-01:2026-06-01", leverages="10", eval_windows=3,
    )
    cells = burst.all_cells(1, burst.grid_from_args(args))
    assert len(cells) == 2
    assert {c["symbol"] for c in cells} == {"ZECUSDT", "NEARUSDT"}
    assert all(c["pair"] == ["KAMA", "NATR"] and c["eval_windows"] == 3 for c in cells)
    argv = burst._discovery_argv({**cells[0], "pop": 6, "gens": 1}, 1, Path("/tmp/x.db"))
    assert argv[argv.index("--eval-windows") + 1] == "3"
    assert argv[argv.index("--end-date") + 1] == "2026-06-01"
    assert burst.cell_key(cells[0]).endswith("|to2026-06-01|ew3")


def _winner_row(cell: dict[str, Any], chromosome: dict[str, Any] | None = None) -> dict[str, Any]:
    best: dict[str, Any] = {
        "sharpe": 2.0, "ret": 0.05, "trades": 60, "pf": 1.5, "dd": 0.04, "genome": "g",
    }
    if chromosome is not None:
        best["chromosome"] = chromosome
    return {"key": burst.cell_key(cell), "cell": cell, "n": 4, "n_win": 1, "best": best}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_report_counts_a_winner_only_when_its_genome_transfers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import vibe_quant.discovery.buyhold as buyhold

    monkeypatch.setattr(buyhold, "load_daily_closes", lambda symbols, catalog: {})
    cell = {"pair": ["ADX", "MACD"], "direction": "long", "symbol": "ZECUSDT",
            "start": "2026-03-01", "end": "2026-04-01", "leverage": 10, "seed": 1,
            "eval_windows": 1}
    transfers, fails = _winner_row(cell), _winner_row({**cell, "seed": 2})
    _write_rows(tmp_path / "ledger.jsonl", [transfers, fails])
    nxt = {"start": "2026-04-01", "end": "2026-05-01", "ret": 0.05, "trades": 80}
    _write_rows(tmp_path / "transfer.jsonl", [
        {"key": transfers["key"], "symbols": {"NEARUSDT": {"ret": 0.10, "trades": 90}}, "next": nxt},
        {"key": fails["key"], "symbols": {"NEARUSDT": {"ret": -0.10, "trades": 90}}, "next": nxt},
    ])

    burst.main(["report", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    block = out.split("## transfer", 1)[1]
    line = next(ln for ln in block.splitlines() if ln.startswith("ADX+MACD"))
    assert line.split() == ["ADX+MACD", "2", "|", "2", "|", "0.500", "|", "1.000", "|", "0.500"]


def _chromosome(indicator: str = "ADX", threshold: float = 25.0) -> Any:
    from vibe_quant.discovery.operators import ConditionType, StrategyChromosome, StrategyGene

    return StrategyChromosome(
        entry_genes=[
            StrategyGene(
                indicator_type=indicator,
                parameters={"period": 14.0},
                condition=ConditionType.GT,
                threshold=threshold,
            )
        ],
        exit_genes=[
            StrategyGene(
                indicator_type="RSI",
                parameters={"period": 14.0},
                condition=ConditionType.LT,
                threshold=30.0,
            )
        ],
        stop_loss_pct=2.0,
        take_profit_pct=5.0,
    )


def _transfer_cell(**over: Any) -> dict[str, Any]:
    return {
        "pair": ["ADX", "MACD"], "direction": "long", "symbol": "ZECUSDT",
        "start": "2026-03-01", "end": "2026-04-01", "leverage": 10, "seed": 1,
        "eval_windows": 1, **over,
    }


class _StubProc:
    returncode = 0

    def poll(self) -> int:
        return 0


def test_best_genome_round_trips_through_the_ledger() -> None:
    from dataclasses import replace

    from vibe_quant.discovery.genome import chromosome_to_serializable, serializable_to_chromosome

    for original in (_chromosome("ADX", 25.0), _chromosome("MACD", 0.5)):
        evals = [{**_eval(60, 0.1, 2.0), "chromosome": chromosome_to_serializable(original)}]
        best = burst.summarize(evals)["best"]
        assert best is not None
        stored = json.loads(json.dumps(best["chromosome"]))  # exactly what the ledger holds
        reloaded = serializable_to_chromosome(stored)
        assert replace(reloaded, uid=original.uid) == original


def test_next_window_is_the_same_length_after_the_cell_and_clips_to_the_catalog_end() -> None:
    assert burst.next_window(_transfer_cell()) == {"start": "2026-04-01", "end": "2026-05-01"}
    assert burst.next_window(
        _transfer_cell(start="2026-08-01", end="2026-10-01")
    ) == {"start": "2026-10-01", "end": "2026-10-20"}
    assert burst.next_window(_transfer_cell(start="2026-10-01", end="2026-10-20")) is None


def test_transfer_jobs_cover_other_symbols_and_the_next_window() -> None:
    zec = _winner_row(_transfer_cell(), {"entry_genes": []})
    near = _winner_row(
        _transfer_cell(symbol="NEARUSDT", start="2026-05-01", end="2026-06-01"), {"entry_genes": []}
    )
    jobs = burst.transfer_jobs([zec, near], ["NEARUSDT", "SOLUSDT", "ZECUSDT"])
    assert [j["symbols"] for j in jobs] == [
        ["NEARUSDT", "SOLUSDT"],
        ["SOLUSDT", "ZECUSDT"],
    ]
    assert [j["next"] for j in jobs] == [
        {"start": "2026-04-01", "end": "2026-05-01"},
        {"start": "2026-06-01", "end": "2026-07-01"},
    ]


def test_transfer_backtest_forwards_the_cells_settings() -> None:
    from vibe_quant.fast_tier.backtest_fn import FastBacktestFn

    slow = burst._transfer_backtest(
        _transfer_cell(leverage=25, eval_windows=3), "NEARUSDT", "2026-03-01", "2026-06-01"
    )
    assert (slow.symbols, slow.timeframe) == (["NEARUSDT"], "1m")
    assert slow.windows is not None and len(slow.windows) == 3
    assert (slow.liquidation.mode, slow.liquidation.leverage) == ("isolated", 25.0)
    assert slow.sizing.max_margin_pct == 0.5
    assert (slow.min_trades, slow.entry_order, slow.symbol_agg) == (50, "market", "portfolio")

    fast = burst._transfer_backtest(
        _transfer_cell(leverage=10, tier="fast"), "ZECUSDT", "2026-05-01", "2026-05-08"
    )
    assert isinstance(fast, FastBacktestFn)
    assert fast.confirm_on_nt is False
    assert fast.nt_fallback is True
    assert fast.windows is None
    assert fast.liquidation.leverage == 10.0


def test_transfer_one_evaluates_every_other_symbol_and_the_next_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[tuple[str, str, str]] = []
    chromosomes: list[Any] = []

    def fake_fn(cell: dict[str, Any], symbol: str, start: str, end: str) -> Any:
        seen.append((symbol, start, end))

        def run(chromosome: Any) -> dict[str, Any]:
            chromosomes.append(chromosome)
            return {"total_return": 0.02 if symbol == "NEARUSDT" else -0.01,
                    "total_trades": 70, "sharpe_ratio": 1.5}

        return run

    monkeypatch.setattr(burst, "_transfer_backtest", fake_fn)
    from vibe_quant.discovery.genome import chromosome_to_serializable

    chromosome = _chromosome()
    spec = {
        "key": "k",
        "cell": _transfer_cell(),
        "chromosome": chromosome_to_serializable(chromosome),
        "symbols": ["NEARUSDT", "SOLUSDT"],
        "next": {"start": "2026-04-01", "end": "2026-05-01"},
    }
    burst.transfer_one(spec, tmp_path / "r.json")
    row = json.loads((tmp_path / "r.json").read_text())
    assert seen == [
        ("NEARUSDT", "2026-03-01", "2026-04-01"),
        ("SOLUSDT", "2026-03-01", "2026-04-01"),
        ("ZECUSDT", "2026-04-01", "2026-05-01"),
    ]
    assert (row["symbols"]["NEARUSDT"]["ret"], row["symbols"]["SOLUSDT"]["ret"]) == (0.02, -0.01)
    assert row["symbols"]["NEARUSDT"]["trades"] == 70
    assert row["next"] == {"start": "2026-04-01", "end": "2026-05-01", "ret": -0.01,
                           "trades": 70, "sharpe": 1.5, "error": None}
    assert len(chromosomes) == 3
    assert chromosomes[0].entry_genes[0].threshold == 25.0


def test_transfer_one_records_an_eval_error_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_fn(cell: dict[str, Any], symbol: str, start: str, end: str) -> Any:
        def run(chromosome: Any) -> dict[str, Any]:
            raise RuntimeError("no data")
        return run

    monkeypatch.setattr(burst, "_transfer_backtest", fake_fn)
    from vibe_quant.discovery.genome import chromosome_to_serializable

    spec = {"key": "k", "cell": _transfer_cell(), "symbols": ["NEARUSDT"], "next": None,
            "chromosome": chromosome_to_serializable(_chromosome())}
    burst.transfer_one(spec, tmp_path / "r.json")
    row = json.loads((tmp_path / "r.json").read_text())
    assert row == {"key": "k", "symbols": {"NEARUSDT": {
        "ret": 0.0, "trades": 0, "sharpe": 0.0, "error": "RuntimeError: no data"}}, "next": None}
    assert burst.genome_transfers(row) is False


def test_transfer_skips_winners_without_a_chromosome_and_counts_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = _winner_row(_transfer_cell())  # a pre-chromosome ledger row
    new = _winner_row(_transfer_cell(symbol="NEARUSDT"), {"entry_genes": []})
    _write_rows(tmp_path / "ledger.jsonl", [old, new])
    launched: list[str] = []

    def fake_launch(spec: dict[str, Any]) -> tuple[_StubProc, Path, float]:
        launched.append(spec["key"])
        return _StubProc(), tmp_path, 0.0

    def fake_finish(spec: dict[str, Any], tmp: Path, rc: int | None, wall: float) -> dict[str, Any]:
        return {"key": spec["key"], "symbols": {"ZECUSDT": {"ret": 0.1, "trades": 5}},
                "next": {"start": "2026-04-01", "end": "2026-05-01", "ret": 0.1}}

    monkeypatch.setattr(burst, "_launch_transfer", fake_launch)
    monkeypatch.setattr(burst, "_finish_transfer", fake_finish)
    monkeypatch.setattr(burst.time, "sleep", lambda _s: None)
    assert burst.transfer(argparse.Namespace(out=str(tmp_path), parallel=2, budget_s=60)) == 0
    assert launched == [new["key"]]
    assert "skipped 1 without a chromosome" in capsys.readouterr().out
    assert [r["key"] for r in burst.load_ledger(tmp_path / "transfer.jsonl")] == [new["key"]]


def test_transfer_resumes_from_the_existing_transfer_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    row = _winner_row(_transfer_cell(), {"entry_genes": []})
    _write_rows(tmp_path / "ledger.jsonl", [row])
    _write_rows(tmp_path / "transfer.jsonl", [
        {"key": row["key"], "symbols": {"NEARUSDT": {"ret": 0.1, "trades": 5}},
         "next": {"start": "2026-04-01", "end": "2026-05-01", "ret": 0.1}},
    ])
    launched: list[str] = []

    def fake_launch(spec: dict[str, Any]) -> tuple[_StubProc, Path, float]:
        launched.append(spec["key"])
        return _StubProc(), tmp_path, 0.0

    monkeypatch.setattr(burst, "_launch_transfer", fake_launch)
    monkeypatch.setattr(burst.time, "sleep", lambda _s: None)
    assert burst.transfer(argparse.Namespace(out=str(tmp_path), parallel=2, budget_s=60)) == 0
    assert launched == []
    assert "0 to evaluate (1 already in transfer.jsonl)" in capsys.readouterr().out
    assert len(burst.load_ledger(tmp_path / "transfer.jsonl")) == 1


def test_transfer_share_requires_a_strictly_positive_return_everywhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import vibe_quant.discovery.buyhold as buyhold

    monkeypatch.setattr(buyhold, "load_daily_closes", lambda symbols, catalog: {})
    zero_symbol = _winner_row(_transfer_cell(seed=1))
    zero_next = _winner_row(_transfer_cell(seed=2))
    no_others = _winner_row(_transfer_cell(seed=3, symbol="NEARUSDT"))
    _write_rows(tmp_path / "ledger.jsonl", [zero_symbol, zero_next, no_others])
    nxt = {"start": "2026-04-01", "end": "2026-05-01", "ret": 0.1}
    _write_rows(tmp_path / "transfer.jsonl", [
        # other symbols: +0.1, next: exactly 0.0 -> fails on the next window
        {"key": zero_next["key"], "symbols": {"NEARUSDT": {"ret": 0.1}}, "next": {**nxt, "ret": 0.0}},
        # other symbols: exactly 0.0 -> fails on the other symbol
        {"key": zero_symbol["key"], "symbols": {"NEARUSDT": {"ret": 0.0}}, "next": nxt},
        # one-symbol ledger: no other-symbol evidence -> never counts
        {"key": no_others["key"], "symbols": {}, "next": nxt},
    ])
    burst.main(["report", "--out", str(tmp_path)])
    block = capsys.readouterr().out.split("## transfer", 1)[1]
    line = next(ln for ln in block.splitlines() if ln.startswith("ADX+MACD"))
    assert line.split() == ["ADX+MACD", "3", "|", "3", "|", "0.500", "|", "0.667", "|", "0.000"]


def test_report_filters_keys_and_prints_each_sub_window_buy_and_hold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from datetime import date

    import vibe_quant.discovery.buyhold as buyhold

    def day(s: str) -> int:
        return (date.fromisoformat(s) - date(1970, 1, 1)).days

    # close doubles over the first sub-window and halves over the last
    closes = {"ZECUSDT": {day("2026-02-28"): 100.0, day("2026-03-30"): 200.0,
                          day("2026-04-29"): 200.0, day("2026-05-31"): 100.0}}
    monkeypatch.setattr(buyhold, "load_daily_closes", lambda symbols, catalog: closes)
    best = {"sharpe": 2.0, "ret": 0.05, "trades": 60, "pf": 1.5, "dd": 0.04}
    multi = {"pair": ["KAMA", "NATR"], "direction": "both", "symbol": "ZECUSDT", "start": "2026-03-01",
             "end": "2026-06-01", "leverage": 10, "seed": 1, "eval_windows": 3, "tier": "fast"}
    single = {**multi, "end": "2026-04-01", "eval_windows": 1}
    rows = [{"key": burst.cell_key(c), "cell": c, "n": 4, "n_win": 1, "best": best} for c in (multi, single)]
    (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    burst.main(["report", "--out", str(tmp_path), "--filter", "|ew3|fast"])
    out = capsys.readouterr().out
    assert "cells=1 failed=0 distinct_evals=4" in out
    assert "bh=+0.0000 bh_sub=[+1.0000,+0.0000,-0.5000]" in out
