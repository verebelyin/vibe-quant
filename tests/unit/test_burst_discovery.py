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
