"""Tests for scripts/agents/burst.py — the scratch-DB burst discovery grid."""

from __future__ import annotations

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


def test_cell_refuses_the_real_state_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VIBE_QUANT_DB", str(burst.REPO_ROOT / "data" / "state" / "vibe_quant.db"))
    with pytest.raises(SystemExit, match="real state DB"):
        burst.run_cell(burst.sample_cells(1, 1)[0], tmp_path / "r.json")


def _winner_row(cell: dict[str, Any], chromosome: dict[str, Any] | None = None) -> dict[str, Any]:
    best: dict[str, Any] = {
        "sharpe": 2.0, "ret": 0.05, "trades": 60, "pf": 1.5, "dd": 0.04, "genome": "g",
    }
    if chromosome is not None:
        best["chromosome"] = chromosome
    return {"key": burst.cell_key(cell), "cell": cell, "n": 4, "n_win": 1, "best": best}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _pt(ret: float, trades: int = 80, bh: bool = True, **extra: Any) -> dict[str, Any]:
    """One transfer point as transfer_one writes it: metrics + the buy-and-hold verdict."""
    return {"ret": ret, "trades": trades, "bh": {"passed": bh}, **extra}


def test_a_genome_that_loses_to_buy_and_hold_anywhere_never_transfers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Profitable on every test but below buy-and-hold on one (own cell, another coin or
    the next window): not a transfer. Rows from before the buy-and-hold rule never count."""
    import vibe_quant.discovery.buyhold as buyhold

    monkeypatch.setattr(buyhold, "load_daily_closes", lambda symbols, catalog: {})
    rows = [_winner_row(_transfer_cell(seed=i)) for i in range(1, 6)]
    _write_rows(tmp_path / "ledger.jsonl", rows)

    def xfer(row: dict[str, Any], own: bool = True, sym: bool = True, nxt: bool = True) -> dict[str, Any]:
        return {"key": row["key"], "own": _pt(0.3, bh=own), "symbols": {"NEARUSDT": _pt(0.1, bh=sym)},
                "next": _pt(0.05, bh=nxt)}

    old = {"key": rows[4]["key"], "symbols": {"NEARUSDT": {"ret": 0.1, "trades": 80}},
           "next": {"ret": 0.05, "trades": 80}}  # profit everywhere, no buy-and-hold verdict
    _write_rows(tmp_path / "transfer.jsonl", [
        xfer(rows[0]), xfer(rows[1], own=False), xfer(rows[2], sym=False), xfer(rows[3], nxt=False), old,
    ])
    assert [burst.genome_transfers(json.loads(ln)) for ln in
            (tmp_path / "transfer.jsonl").read_text().splitlines()] == [True, False, False, False, False]
    burst.main(["report", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "rows from before that rule: 1, never a transfer" in out
    line = next(ln for ln in out.split("## transfer", 1)[1].splitlines() if ln.startswith("ADX+MACD"))
    assert line.split()[-3] == "0.200"  # 1 of 5 winners


def test_buy_and_hold_verdict_is_discoverys_gate_on_the_whole_window() -> None:
    """Below the coin's own buy-and-hold daily Sharpe fails, above passes, per direction."""
    from array import array

    from vibe_quant.metrics import DailyReturns

    first, n = 20_000, 40
    closes: dict[str, dict[int, float]] = {"ZECUSDT": {first - 1: 100.0}}
    for i in range(n):  # +2% / 0% alternating: daily Sharpe ~ 0.99
        prev = closes["ZECUSDT"][first + i - 1]
        closes["ZECUSDT"][first + i] = prev * (1.02 if i % 2 == 0 else 1.0)
    worse = DailyReturns(first, array("d", [0.01 if i % 2 == 0 else -0.005 for i in range(n)]))
    better = DailyReturns(first, array("d", [0.01 if i % 2 == 0 else 0.005 for i in range(n)]))

    lose = burst._bh_verdict(worse, "long", closes, "ZECUSDT")
    assert lose["passed"] is False
    assert lose["sharpe_ann"] < lose["bh_sharpe_ann"] == lose["bar_ann"]
    assert lose["reason"].startswith("Buy-hold: sharpe")
    assert burst._bh_verdict(better, "long", closes, "ZECUSDT")["passed"] is True
    # a short genome only has to beat -bh; a missing series or close fails closed
    assert burst._bh_verdict(worse, "short", closes, "ZECUSDT")["passed"] is True
    assert burst._bh_verdict(None, "long", closes, "ZECUSDT")["passed"] is False
    assert burst._bh_verdict(better, "long", {}, "ZECUSDT")["passed"] is False


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


def test_binomial_tail_is_exact_at_hand_computed_values() -> None:
    from fractions import Fraction

    # P(X >= 2 | n = 4, p = 1/4) = 1 - (3/4)^4 - 4*(1/4)*(3/4)^3 = 67/256
    assert burst._binom_upper(2, 4, Fraction(1, 4)) == 0.26171875
    assert burst._binom_upper(2, 3, Fraction(1, 2)) == 0.5
    assert burst._binom_upper(4, 3, Fraction(1, 2)) == 0.0


def test_atr_cells_reach_the_discovery_spec_and_the_transfer() -> None:
    """--atr-risk: the cell's argv parses into a discovery spec sizing by ATR under the L x
    balance ceiling, and the transfer re-evaluates with the same sizing."""
    from vibe_quant.discovery.run_spec import DiscoveryRunSpec
    from vibe_quant.sizing_settings import SizingSettings

    base = {**burst.sample_cells(1, 1)[0], "pop": 6, "gens": 1, "leverage": 3}
    atr = {**base, "sizing": "atr", "atr_risk": 0.005}
    argv = burst._discovery_argv(atr, 7, Path("/tmp/x/state.db"))
    assert argv[:3] == ["discovery", "--run-id", "7"]
    spec_argv = argv[3:]  # from_argv takes the argv without --run-id
    expected = SizingSettings(max_position_pct=3.0, risk_per_trade=0.005, max_margin_pct=1.0,
                              sizing_mode="atr")
    assert DiscoveryRunSpec.from_argv(spec_argv).sizing_settings() == expected
    assert burst.cell_key(atr) == burst.cell_key(base) + "|atr0.005"
    assert burst._transfer_backtest(atr, atr["symbol"], atr["start"], atr["end"]).sizing == expected


def test_blind_verdict_reads_the_whole_next_window_and_prints_worst_of_n(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """blind_2026_rsi PASS rule (>= 15 trades, ret > 0, Sharpe >= 1) on ``next_full``;
    a row from before ``next_full`` existed gets no verdict, never the worst-of-N one."""
    import vibe_quant.discovery.buyhold as buyhold

    monkeypatch.setattr(buyhold, "load_daily_closes", lambda symbols, catalog: {})
    rsi_spec = importlib.util.spec_from_file_location(
        "blind_2026_rsi", _SCRIPT.parents[1] / "research" / "blind_2026_rsi.py"
    )
    assert rsi_spec is not None and rsi_spec.loader is not None
    blind = importlib.util.module_from_spec(rsi_spec)
    rsi_spec.loader.exec_module(blind)
    cell = _transfer_cell(start="2026-05-01", end="2026-08-01", eval_windows=3)
    new, old = _winner_row(cell), _winner_row({**cell, "symbol": "SUIUSDT"})
    _write_rows(tmp_path / "ledger.jsonl", [new, old])
    worst = {"ret": -0.02, "trades": 9, "sharpe": -0.5, "dd": 0.1, "error": None}
    full = {"ret": 0.08, "trades": 40, "sharpe": 1.3, "dd": 0.2, "error": None}
    _write_rows(tmp_path / "transfer.jsonl", [
        {"key": new["key"], "symbols": {}, "next": worst, "next_full": full},
        {"key": old["key"], "symbols": {}, "next": full},  # pre-iqj2h row: next only
    ])
    assert blind.main(["--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    lines = {ln.split()[1]: ln for ln in out.splitlines() if ln.startswith(("PASS", "FAIL"))}
    assert lines[new["key"]].startswith("PASS")
    assert "blind ret=+0.0800 trades=40 sharpe=1.3 dd=0.2000" in lines[new["key"]]
    assert "worst-of-N ret=-0.0200 trades=9" in lines[new["key"]]
    assert lines[old["key"]].startswith("FAIL")
    assert "blind no eval (None)" in lines[old["key"]]
    assert "without_blind_eval=1" in out
    assert "PRIMARY pass 1/2" in out
