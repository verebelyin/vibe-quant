"""Opt-in per-symbol worst-of aggregation for discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.__main__ import main
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.discovery.gate_chain import TrialBudget
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from pathlib import Path


def _m(sharpe: float, ret: float, trades: int, dd: float = 0.1, pf: float = 1.5) -> dict[str, Any]:
    return {
        "sharpe_ratio": sharpe, "total_return": ret, "total_trades": trades,
        "max_drawdown": dd, "profit_factor": pf, "skewness": 0.0, "kurtosis": 3.0,
        "trade_returns": (ret,) * 2,
    }  # fmt: skip


def _failer_bt(total_return: float = -0.1) -> dict[str, Any]:
    """Worst-symbol FAILER backtest: symbol C's train return <= 0 fails the gate.

    ``total_return`` is the pooled worst-of headline; varying it changes only the
    behavioural fingerprint, never the per-symbol gate outcome.
    """
    return {
        **_m(-0.5, total_return, 120),  # worst-of headline: C drags every field down
        "symbol_results": {
            "A": _m(4.0, 0.6, 120, dd=0.05, pf=4.0),
            "B": _m(4.0, 0.6, 120, dd=0.05, pf=4.0),
            "C": _m(-0.5, -0.1, 120),  # train return <= 0 -> worst-symbol gate FAIL
        },
    }


class FakeFn(NTBacktestFn):
    """NTBacktestFn whose per-run backtest is a table lookup (no NT)."""

    def __init__(self, table: dict[tuple[str, ...], dict[str, Any]], *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.table = table
        self.calls: list[tuple[str, ...]] = []

    def _run_single(
        self, chromosome: Any, start_date: str, end_date: str, symbols: list[str] | None = None
    ) -> dict[str, Any]:  # type: ignore[override]
        key = tuple(symbols if symbols is not None else self.symbols)
        self.calls.append(key)
        return dict(self.table[key])


def test_worst_mode_takes_min_across_symbols() -> None:
    table = {
        ("A",): _m(2.0, 0.30, 100, dd=0.10, pf=2.0),
        ("B",): _m(0.5, 0.05, 80, dd=0.25, pf=1.1),
    }
    fn = FakeFn(table, ["A", "B"], "4h", "2024-01-01", "2024-06-01", symbol_agg="worst")
    out = fn(object())  # type: ignore[arg-type]
    assert out["sharpe_ratio"] == 0.5
    assert out["total_return"] == 0.05
    assert out["profit_factor"] == 1.1
    assert out["max_drawdown"] == 0.25
    assert out["total_trades"] == 80  # MIN across symbols
    assert len(out["trade_returns"]) == 4  # concatenated


def test_one_symbol_below_min_trades_zeroes_fitness() -> None:
    table = {("A",): _m(2.0, 0.3, 200), ("B",): _m(1.0, 0.1, 10)}
    fn = FakeFn(table, ["A", "B"], "4h", "2024-01-01", "2024-06-01",
                min_trades=50, symbol_agg="worst")  # fmt: skip
    out = fn(object())  # type: ignore[arg-type]
    assert out["total_trades"] < 50  # pipeline hard gate (<min_trades) -> fitness 0


def test_worst_mode_never_exits_across_symbols() -> None:
    # A lone losing symbol no longer means score 0 (soft score) -> run them all.
    table = {
        ("A",): _m(-1.0, -0.2, 100),
        ("B",): _m(1.0, 0.1, 100),
        ("C",): _m(1.0, 0.1, 100),
    }
    fn = FakeFn(table, ["A", "B", "C"], "4h", "2024-01-01", "2024-06-01", symbol_agg="worst")
    out = fn(object())  # type: ignore[arg-type]
    assert fn.calls == [("A",), ("B",), ("C",)]
    assert "early_exit_symbol" not in out
    assert set(out["symbol_results"]) == {"A", "B", "C"}  # type: ignore[arg-type]


def test_portfolio_default_calls_runner_once_with_all_symbols() -> None:
    table = {("A", "B"): _m(1.0, 0.1, 100)}
    fn = FakeFn(table, ["A", "B"], "4h", "2024-01-01", "2024-06-01")
    assert fn.symbol_agg == "portfolio"
    out = fn(object())  # type: ignore[arg-type]
    assert fn.calls == [("A", "B")]
    assert out["sharpe_ratio"] == 1.0


def test_single_symbol_worst_equals_portfolio() -> None:
    table = {("A",): _m(1.0, 0.1, 100)}
    kw: dict[str, Any] = {"min_trades": 20}
    w = FakeFn(table, ["A"], "4h", "2024-01-01", "2024-06-01", symbol_agg="worst", **kw)
    p = FakeFn(table, ["A"], "4h", "2024-01-01", "2024-06-01", **kw)
    assert w(object()) == p(object())  # type: ignore[arg-type]
    wins = [("2024-01-01", "2024-03-01"), ("2024-03-01", "2024-06-01")]
    w2 = FakeFn(table, ["A"], "4h", "2024-01-01", "2024-06-01", windows=wins,
                symbol_agg="worst", **kw)  # fmt: skip
    p2 = FakeFn(table, ["A"], "4h", "2024-01-01", "2024-06-01", windows=wins, **kw)
    assert w2(object()) == p2(object())  # type: ignore[arg-type]


def test_worst_mode_runs_windows_per_symbol() -> None:
    table = {("A",): _m(1.0, 0.1, 100), ("B",): _m(2.0, 0.2, 100)}
    wins = [("2024-01-01", "2024-03-01"), ("2024-03-01", "2024-06-01")]
    fn = FakeFn(table, ["A", "B"], "4h", "2024-01-01", "2024-06-01", windows=wins,
                symbol_agg="worst")  # fmt: skip
    out = fn(object())  # type: ignore[arg-type]
    assert fn.calls == [("A",), ("A",), ("B",), ("B",)]
    assert out["sharpe_ratio"] == 1.0


# --- CLI / API / notes ------------------------------------------------------


@pytest.fixture()
async def client(tmp_path: Path):
    app = create_app()
    state_mgr = StateManager(db_path=tmp_path / "t.db")
    _ = state_mgr.conn
    job_mgr = BacktestJobManager(db_path=tmp_path / "t.db")
    ws_mgr = ConnectionManager()
    await ws_mgr.start()
    app.state.state_manager = state_mgr
    app.state.job_manager = job_mgr
    app.state.ws_manager = ws_mgr
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, job_mgr
    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


_BODY = {
    "symbols": ["BTCUSDT", "ETHUSDT"], "timeframes": ["4h"], "population": 12,
    "generations": 10, "mutation_rate": 0.15, "crossover_rate": 0.7,
    "elite_count": 2, "tournament_size": 3, "convergence_generations": 5,
}  # fmt: skip


async def test_cli_and_api_forward_symbol_agg(client: Any) -> None:
    from unittest.mock import patch

    ac, job_mgr = client
    with (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as popen,
    ):
        popen.return_value.pid = 1
        r = await ac.post("/api/discovery/launch", json={**_BODY, "symbol_agg": "worst"})
        assert r.status_code == 201, r.text
        args = popen.call_args.args[0]
        assert args[args.index("--symbol-agg") + 1] == "worst"

        r = await ac.post("/api/discovery/launch", json=_BODY)
        assert r.status_code == 201, r.text
        assert "--symbol-agg" not in popen.call_args.args[0]

        r = await ac.post("/api/discovery/launch", json={**_BODY, "symbol_agg": "bogus"})
        assert r.status_code == 422


def _create_run(db_path: Path) -> int:
    import uuid

    state = StateManager(db_path)
    sid = state.create_strategy(name=f"__sa_{uuid.uuid4().hex[:8]}__", dsl_config={"type": "discovery"},
                                description="t", strategy_type="discovery")  # fmt: skip
    run_id = state.create_backtest_run(
        strategy_id=sid, run_mode="discovery", symbols=["BTCUSDT", "ETHUSDT"],
        timeframe="1h", start_date="2025-01-01", end_date="2025-04-01", parameters={},
    )  # fmt: skip
    state.close()
    return run_id


def _argv(db: Path, run_id: int, *extra: str) -> list[str]:
    return [
        "prog", "--run-id", str(run_id), "--population-size", "6", "--max-generations", "2",
        "--elite-count", "1", "--symbols", "BTCUSDT,ETHUSDT", "--timeframe", "1h",
        "--start-date", "2025-01-01", "--end-date", "2025-04-01", "--db", str(db), *extra,
    ]  # fmt: skip


def _notes(db: Path, run_id: int) -> dict[str, Any]:
    import json

    state = StateManager(db)
    res = state.get_backtest_result(run_id)
    state.close()
    assert res is not None
    return dict(json.loads(res["notes"]))


def test_notes_record_symbol_agg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "s.db"
    rid = _create_run(db)
    monkeypatch.setattr("sys.argv", _argv(db, rid, "--mock", "--symbol-agg", "worst"))
    assert main() == 0
    assert _notes(db, rid)["symbol_agg"] == "worst"

    rid2 = _create_run(db)
    monkeypatch.setattr("sys.argv", _argv(db, rid2, "--mock"))
    assert main() == 0
    assert _notes(db, rid2)["symbol_agg"] == "portfolio"


def test_holdout_and_full_range_fns_use_same_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vibe_quant.discovery.run as discovery_run
    from vibe_quant.discovery.mock_backtest import mock_backtest as _mock_backtest

    made: list[dict[str, Any]] = []

    def fake_make(**kw: Any) -> Any:
        made.append(kw)
        return _mock_backtest

    factory_modes: list[str] = []

    def fake_ntfn(syms: Any, tf: Any, s: Any, e: Any, *a: Any, **kw: Any) -> Any:
        factory_modes.append(kw.get("symbol_agg", "portfolio"))
        return _mock_backtest

    real_pipeline = discovery_run.DiscoveryPipeline

    def spy_pipeline(*a: Any, **kw: Any) -> Any:
        factory = kw.get("backtest_fn_factory")
        if factory is not None:
            factory("2025-01-01", "2025-02-01")
        return real_pipeline(*a, **kw)

    monkeypatch.setattr(discovery_run, "DiscoveryPipeline", spy_pipeline)
    monkeypatch.setattr(discovery_run, "_make_nt_backtest_fn", fake_make)
    monkeypatch.setattr(discovery_run, "NTBacktestFn", fake_ntfn)
    monkeypatch.setattr(discovery_run, "_check_data_available", lambda syms: True)
    db = tmp_path / "h.db"
    rid = _create_run(db)
    monkeypatch.setattr(
        "sys.argv",
        _argv(
            db, rid, "--symbol-agg", "worst", "--eval-windows", "2", "--cross-window-months", "1"
        ),  # fmt: skip
    )
    main()
    assert len(made) >= 2  # GA fn + holdout (+ full-range / train-return)
    assert all(kw.get("symbol_agg") == "worst" for kw in made), made
    assert factory_modes
    assert set(factory_modes) == {"worst"}


# --- fix round 1: consistency / drift / early-exit scope / notes -----------


def test_symbol_early_exit_disabled_under_soft_score() -> None:
    table = {("A",): _m(-1.0, -0.2, 100), ("B",): _m(1.0, 0.1, 100)}
    fn = FakeFn(table, ["A", "B"], "4h", "2024-01-01", "2024-06-01", symbol_agg="worst")
    out = fn(object())  # type: ignore[arg-type]
    assert fn.calls == [("A",), ("B",)]
    assert out["sharpe_ratio"] == -1.0 and out["total_return"] == -0.2  # real values
    assert out["symbol_metrics"]["B"]["trades"] == 100  # type: ignore[index]
    assert out["trades_sum"] == 200


def test_per_window_early_exit_still_inside_symbol() -> None:
    wins = [("2024-01-01", "2024-03-01"), ("2024-03-01", "2024-06-01")]
    table = {("A",): _m(-1.0, -0.2, 100), ("B",): _m(1.0, 0.1, 100)}
    fn = FakeFn(table, ["A", "B"], "4h", "2024-01-01", "2024-06-01", windows=wins,
                symbol_agg="worst")  # fmt: skip
    fn(_chrom())
    assert fn.calls == [("A",), ("B",), ("B",)]  # A stops after window 0


# --- soft worst-mode GA score ---------------------------- (regression: vibe-quant-ox73t)


def test_soft_score_formula() -> None:
    from vibe_quant.discovery.fitness import soft_worst_score

    assert soft_worst_score([0.0, 0.6, 0.8, 1.0]) == pytest.approx(0.35)
    assert soft_worst_score([0.42]) == 0.42
    assert soft_worst_score([0.0, 0.0, 0.0]) == 0.0
    assert soft_worst_score([]) == 0.0


def _chrom(**kw: Any) -> Any:
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    base: dict[str, Any] = {
        "entry_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0, None)],
        "exit_genes": [StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        "stop_loss_pct": 2.0, "take_profit_pct": 4.0, "direction": Direction.LONG,
    }  # fmt: skip
    base.update(kw)
    return StrategyChromosome(**base)


def _cfg(**kw: Any) -> Any:
    from vibe_quant.discovery.pipeline import DiscoveryConfig

    base: dict[str, Any] = {
        "population_size": 6, "max_generations": 2, "elite_count": 1, "top_k": 5,
        "min_trades": 50, "max_workers": None, "timeframe": "4h",
        "start_date": "2024-01-01", "end_date": "2024-06-01",
        "require_bootstrap_ci": False, "require_dsr": False,
    }  # fmt: skip
    base.update(kw)
    return DiscoveryConfig(**base)


def _eval(table: dict[tuple[str, ...], dict[str, Any]], **kw: Any) -> Any:
    from vibe_quant.discovery.fitness import _evaluate_single

    fn = FakeFn(table, ["A", "B"], "4h", "2024-01-01", "2024-06-01", symbol_agg="worst",
                min_trades=50)  # fmt: skip
    return _evaluate_single(_chrom(**kw), fn, None, 50, "4h")


def test_worst_mode_gives_gradient() -> None:
    # B always loses -> old worst-of fitness is exactly 0 for both genomes
    lose = _m(-0.5, -0.1, 120)
    g1 = _eval({("A",): _m(1.5, 0.3, 120), ("B",): lose})
    g2 = _eval({("A",): _m(2.5, 0.6, 120), ("B",): lose})
    assert g1.adjusted_score > 0 and g2.adjusted_score > 0
    assert g1.adjusted_score != g2.adjusted_score
    assert g2.adjusted_score > g1.adjusted_score
    # reported metrics stay worst-of; per-symbol breakdown recorded
    assert g1.total_return == -0.1 and g1.sharpe_ratio == -0.5
    assert g1.symbol_scores["B"] == 0.0 and g1.symbol_scores["A"] > 0
    assert g1.adjusted_score == pytest.approx(0.5 * 0.0 + 0.5 * (g1.symbol_scores["A"] / 2))


def test_portfolio_mode_unchanged() -> None:
    from vibe_quant.discovery.fitness import _evaluate_single

    bt = _m(1.5, 0.3, 120, dd=0.2, pf=1.6)
    r = _evaluate_single(_chrom(), lambda c: bt, None, 50, "4h")
    assert r.adjusted_score == 0.5256666666666667  # pinned from main
    assert r.raw_score == 0.5256666666666667
    assert r.symbol_scores is None
    # one symbol in worst mode (falls through to portfolio path) is identical
    fn = FakeFn({("A",): bt}, ["A"], "4h", "2024-01-01", "2024-06-01", symbol_agg="worst")
    r1 = _evaluate_single(_chrom(), fn, None, 50, "4h")
    assert r1.adjusted_score == r.adjusted_score and r1.symbol_scores is None


def test_worst_symbol_train_gate_rejects_champion() -> None:
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    fit = _eval({("A",): _m(2.0, 0.4, 120), ("B",): _m(-0.5, -0.1, 120)})
    assert fit.adjusted_score > 0  # selectable by the GA ...
    good = _eval({("A",): _m(2.0, 0.4, 120), ("B",): _m(1.0, 0.2, 120)})
    pipe = DiscoveryPipeline(_cfg(), backtest_fn=lambda c: {})
    out = pipe.gates.screen(
        [(_chrom(), fit), (_chrom(stop_loss_pct=1.0), good)], TrialBudget(100, {})
    )
    assert [f for _, f in out] == [good]  # ... but not a champion
    rej = pipe.gates.rejections
    assert len(rej) == 1 and rej[0]["stage"] == "worst_symbol_train"
    assert any("worst-symbol train return <= 0 (B)" in r for r in rej[0]["reasons"])  # type: ignore[attr-defined]


def test_worst_symbol_train_gate_trades() -> None:
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    # A: 200 trades; B: 60 -> passes eval (>=50) but gate demands min_trades=80
    fit = _eval({("A",): _m(2.0, 0.4, 200), ("B",): _m(1.0, 0.2, 60)})
    pipe = DiscoveryPipeline(_cfg(min_trades=80), backtest_fn=lambda c: {})
    assert pipe.gates.screen([(_chrom(), fit)], TrialBudget(100, {})) == []
    assert any("(B)" in r for r in pipe.gates.rejections[0]["reasons"])  # type: ignore[attr-defined]


def _worst_discovery_run(state: StateManager, **entry_extra: Any) -> int:
    run_id = state.create_backtest_run(
        strategy_id=None, run_mode="discovery", symbols=["BTCUSDT", "ETHUSDT"], timeframe="4h",
        start_date="2024-01-01", end_date="2025-01-01", parameters={},
    )  # fmt: skip
    state.update_backtest_run_status(run_id, "completed")
    entry = {"dsl": {"name": "genome_abc123"}, "sharpe": 1.0, "trades": 60,
             "full_range_sharpe": 1.0, "full_range_trades": 60, **entry_extra}  # fmt: skip
    import json

    state.save_backtest_result(run_id, {"total_trades": 0}, trades=[])
    state.update_result_notes(
        run_id, json.dumps({"symbol_agg": "worst", "top_strategies": [entry]})
    )
    return run_id


@pytest.fixture()
def state(tmp_path: Path):
    mgr = StateManager(tmp_path / "c.db")
    _ = mgr.conn
    yield mgr
    mgr.close()


def test_worst_consistency_uses_trades_sum(state: StateManager) -> None:
    from vibe_quant.validation.consistency import assess_consistency, find_screening_reference

    _worst_discovery_run(state, full_range_trades_sum=130)
    ref = find_screening_reference(state, 999, "genome_abc123")
    assert ref is not None and ref.trades == 130
    report = assess_consistency(ref, 1.0, 130)
    assert not report.is_flagged
    assert any("worst symbol" in n for n in report.to_dict()["notes"])  # type: ignore[attr-defined]


def test_worst_consistency_legacy_without_sum_skips_trades(state: StateManager) -> None:
    from vibe_quant.validation.consistency import assess_consistency, find_screening_reference

    _worst_discovery_run(state)
    ref = find_screening_reference(state, 999, "genome_abc123")
    assert ref is not None
    report = assess_consistency(ref, 1.0, 130)
    assert not any("trade" in f for f in report.flags)
    assert any("trade comparison skipped" in n for n in report.notes)


def test_portfolio_consistency_unchanged(state: StateManager) -> None:
    import json

    from vibe_quant.validation.consistency import assess_consistency, find_screening_reference

    rid = _worst_discovery_run(state)
    state.update_result_notes(rid, json.dumps({"top_strategies": [{
        "dsl": {"name": "genome_abc123"}, "full_range_sharpe": 1.0, "full_range_trades": 60}]}))  # fmt: skip
    ref = find_screening_reference(state, 999, "genome_abc123")
    assert ref is not None and ref.trades == 60
    report = assess_consistency(ref, 1.0, 130)
    assert any("trade-count-divergence" in f for f in report.flags)
    assert report.notes == []


def _drift(
    state: StateManager, notes: dict[str, Any], scr_trades: int, scr_sharpe: float = 1.0
) -> dict[str, Any]:
    import json

    from vibe_quant.screening.replay_drift import check_replay_drift

    did = _worst_discovery_run(state)
    state.update_result_notes(did, json.dumps(notes))
    rid = state.create_backtest_run(
        strategy_id=None, run_mode="screening", symbols=["BTCUSDT", "ETHUSDT"], timeframe="4h",
        start_date="2024-01-01", end_date="2025-01-01",
        parameters={"promote_source": {"discovery_run_id": did, "strategy_index": 0}},
    )  # fmt: skip
    state.conn.execute(
        "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades,"
        " is_pareto_optimal) VALUES (?, '{}', ?, ?, 1)", (rid, scr_sharpe, scr_trades),
    )  # fmt: skip
    state.conn.commit()
    out = check_replay_drift(state, rid)
    assert out is not None
    return dict(out)


def test_worst_replay_drift_uses_trades_sum(state: StateManager) -> None:
    e = {"dsl": {"name": "g"}, "full_range_sharpe": 1.0, "full_range_trades": 60,
         "full_range_trades_sum": 130}  # fmt: skip
    out = _drift(state, {"symbol_agg": "worst", "top_strategies": [e]}, 130)
    assert out["flagged"] is False and out["discovery_trades"] == 130


def test_worst_replay_drift_legacy_skips_trades_with_note(state: StateManager) -> None:
    e = {"dsl": {"name": "g"}, "full_range_sharpe": 1.0, "full_range_trades": 60}
    out = _drift(state, {"symbol_agg": "worst", "top_strategies": [e]}, 130)
    assert out["flagged"] is False
    assert "trade comparison skipped" in str(out["note"])


def test_portfolio_replay_drift_unchanged(state: StateManager) -> None:
    e = {"dsl": {"name": "g"}, "full_range_sharpe": 1.0, "full_range_trades": 60}
    out = _drift(state, {"top_strategies": [e]}, 130)
    assert out["flagged"] is True and "note" not in out


async def test_replay_metrics_note_worst_mode(client: Any, tmp_path: Path) -> None:
    import json

    ac, job_mgr = client
    st = job_mgr._db_path if hasattr(job_mgr, "_db_path") else None  # noqa: SLF001
    assert st is not None
    s = StateManager(st)
    rid = _worst_discovery_run(s)
    s.update_result_notes(rid, json.dumps({"symbol_agg": "worst", "top_strategies": [{
        "dsl": {"name": "g", "strategy_type": "momentum", "entry": {"conditions": [
            {"indicator": "RSI", "params": {"period": 14}, "operator": "<", "value": 30}]},
            "exit": {"conditions": [
            {"indicator": "RSI", "params": {"period": 14}, "operator": ">", "value": 70}]}}}]}))  # fmt: skip
    s.conn.execute("UPDATE backtest_runs SET parameters=? WHERE id=?",
                   (json.dumps({"eval_windows": 3}), rid))  # fmt: skip
    s.conn.commit()
    r = await ac.post(f"/api/discovery/results/{rid}/replay/0")
    assert r.status_code == 201, r.text
    note = r.json()["metrics_note"]
    assert "WORST symbol" in note and "worst-of-3" in note
    assert "Compare against the champion's full_range" not in note  # advice is wrong in worst mode


def test_worst_replay_drift_sharpe_is_one_sided(state: StateManager) -> None:
    e = {"dsl": {"name": "g"}, "full_range_sharpe": 1.0, "full_range_trades": 60,
         "full_range_trades_sum": 130}  # fmt: skip
    notes = {"symbol_agg": "worst", "top_strategies": [e]}
    # portfolio Sharpe above the worst-symbol reference is healthy
    assert _drift(state, notes, 130, scr_sharpe=1.5)["flagged"] is False
    # ... but a collapse below it still flags
    assert _drift(state, notes, 130, scr_sharpe=0.5)["flagged"] is True


def test_portfolio_replay_drift_sharpe_still_two_sided(state: StateManager) -> None:
    e = {"dsl": {"name": "g"}, "full_range_sharpe": 1.0, "full_range_trades": 130}
    assert _drift(state, {"top_strategies": [e]}, 130, scr_sharpe=1.5)["flagged"] is True


def test_symbol_metrics_non_finite_become_none() -> None:
    table = {("A",): _m(float("nan"), 0.1, 100), ("B",): _m(float("inf"), 0.2, 100)}
    fn = FakeFn(table, ["A", "B"], "4h", "2024-01-01", "2024-06-01", symbol_agg="worst")
    out = fn(object())  # type: ignore[arg-type]
    sm = out["symbol_metrics"]
    assert sm["A"]["sharpe"] is None and sm["B"]["sharpe"] is None  # type: ignore[index]
    assert sm["A"]["return"] == 0.1  # type: ignore[index]


# --- review follow-ups: top-K gate ordering / notes pins / penalty pin -------


def _genes(*thresholds: float) -> list[Any]:
    from vibe_quant.discovery.operators import ConditionType, StrategyGene

    return [StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, t, None) for t in thresholds]


def test_topk_ranks_gate_passers_first(monkeypatch: pytest.MonkeyPatch) -> None:
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    # 5 high-soft-score gate-FAILERS (two stellar symbols carry the one loser)
    # + 1 lower-soft-score gate-PASSER (every symbol merely OK).
    ok_sym = _m(0.1, 0.01, 120, dd=0.4, pf=0.8)
    passer_bt: dict[str, Any] = {
        **ok_sym,
        "symbol_results": {"A": dict(ok_sym), "B": dict(ok_sym), "C": dict(ok_sym)},
    }
    failers = [_chrom(stop_loss_pct=sl) for sl in (1.0, 1.4, 1.8, 2.2, 2.6)]
    passer = _chrom(stop_loss_pct=3.0)
    # Distinct headline total_return per failer so the behavioural fingerprint
    # dedupe keeps all five (they are NOT behaviourally identical).
    table = {c.uid: _failer_bt(-0.1 - i * 1e-6) for i, c in enumerate(failers)}
    table[passer.uid] = passer_bt
    pipe = DiscoveryPipeline(
        _cfg(max_generations=1, min_diversity_distance=0.0, gate_pool="top_k"),
        backtest_fn=lambda c: table[c.uid],
    )
    monkeypatch.setattr(
        "vibe_quant.discovery.pipeline.initialize_population",
        lambda n, **kw: [*failers, passer],
    )
    monkeypatch.setattr(DiscoveryPipeline, "_apply_indicator_pool_filter", lambda self: None)
    monkeypatch.setattr(DiscoveryPipeline, "_preflight_aux_data", lambda self: None)
    result = pipe.run()
    top_k_ids = [c.uid for c, _ in pipe._all_scored[:5]]
    assert passer.uid in top_k_ids  # top-K keeps a slot for the gate-passer
    assert [c.uid for c, _ in result.top_strategies] == [passer.uid]
    rejected = [r for r in result.guardrail_rejections if r["stage"] == "worst_symbol_train"]
    assert len(rejected) == 4  # the other 4 top-K slots went to gate-failers


def test_topk_dedupes_identical_gate_failers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical-metric gate-failers share ONE top-K slot (behavioural dedupe).

    Round-1 added the fingerprint skip to
    ``_select_diverse_top_k``: five failers built from ONE identical backtest
    dict are behaviourally identical, so only the first is selected and just one
    ``worst_symbol_train`` rejection is recorded (the structural distance rule
    alone, with min_diversity_distance=0.0, would have kept all five).

    regression: vibe-quant-pqon2
    """
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    loser_bt = _failer_bt()  # ONE shared dict -> identical fingerprints
    ok_sym = _m(0.1, 0.01, 120, dd=0.4, pf=0.8)
    passer_bt: dict[str, Any] = {
        **ok_sym,
        "symbol_results": {"A": dict(ok_sym), "B": dict(ok_sym), "C": dict(ok_sym)},
    }
    failers = [_chrom(stop_loss_pct=sl) for sl in (1.0, 1.4, 1.8, 2.2, 2.6)]
    passer = _chrom(stop_loss_pct=3.0)
    table = {c.uid: loser_bt for c in failers}
    table[passer.uid] = passer_bt
    pipe = DiscoveryPipeline(
        _cfg(max_generations=1, min_diversity_distance=0.0, gate_pool="top_k"),
        backtest_fn=lambda c: table[c.uid],
    )
    monkeypatch.setattr(
        "vibe_quant.discovery.pipeline.initialize_population",
        lambda n, **kw: [*failers, passer],
    )
    monkeypatch.setattr(DiscoveryPipeline, "_apply_indicator_pool_filter", lambda self: None)
    monkeypatch.setattr(DiscoveryPipeline, "_preflight_aux_data", lambda self: None)
    result = pipe.run()
    assert [c.uid for c, _ in result.top_strategies] == [passer.uid]
    rejected = [r for r in result.guardrail_rejections if r["stage"] == "worst_symbol_train"]
    assert len(rejected) == 1  # the other 4 identical failers are deduped away


def test_worst_mode_notes_soft_score_and_symbol_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import statistics

    import vibe_quant.discovery.run as discovery_run
    from vibe_quant.discovery.fitness import compute_fitness_score, soft_worst_score

    table = {
        ("BTCUSDT",): _m(2.0, 0.4, 120, dd=0.1, pf=2.0),
        ("ETHUSDT",): _m(1.0, 0.2, 100, dd=0.2, pf=1.2),
    }

    def fake_make(**kw: Any) -> Any:
        return FakeFn(
            table, kw["symbols"], kw["timeframe"], kw["start_date"], kw["end_date"],
            windows=kw.get("windows"), min_trades=kw.get("min_trades", 0),
            symbol_agg=kw.get("symbol_agg", "portfolio"),
        )  # fmt: skip

    pop = [_chrom(stop_loss_pct=sl) for sl in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)]
    monkeypatch.setattr("vibe_quant.discovery.pipeline.initialize_population", lambda n, **kw: pop)
    monkeypatch.setattr(discovery_run, "_check_data_available", lambda syms: True)
    monkeypatch.setattr(discovery_run, "_make_nt_backtest_fn", fake_make)
    db = tmp_path / "w.db"
    rid = _create_run(db)
    monkeypatch.setattr(
        "sys.argv",
        _argv(
            # This test checks worst-mode notes/soft scores, not the buy-hold
            # gate; FakeFn emits no MTM series, so the gate correctly rejects.
            db,
            rid,
            "--symbol-agg",
            "worst",
            "--max-generations",
            "1",
            "--eval-windows",
            "1",
            "--train-test-split",
            "0",
            "--max-workers",
            "-1",
            "--no-dsr",
            "--no-bootstrap-ci",
            "--no-buy-hold-gate",
        ),  # fmt: skip
    )
    assert main() == 0
    notes = _notes(db, rid)
    assert notes["worst_mode_score"] == "0.5*min+0.5*median"
    entries = notes["top_strategies"]
    assert entries  # the all-green table always yields champions
    expected = {
        "BTCUSDT": round(compute_fitness_score(2.0, 0.1, 2.0, 0.4), 6),
        "ETHUSDT": round(compute_fitness_score(1.0, 0.2, 1.2, 0.2), 6),
    }
    assert entries[0]["symbol_scores"] == expected
    raws = [compute_fitness_score(2.0, 0.1, 2.0, 0.4), compute_fitness_score(1.0, 0.2, 1.2, 0.2)]
    assert entries[0]["score"] == pytest.approx(soft_worst_score(raws))
    assert entries[0]["score"] == pytest.approx(
        0.5 * min(expected.values()) + 0.5 * statistics.median(expected.values()), abs=1e-5
    )


def test_portfolio_pin_nonzero_penalties() -> None:
    from vibe_quant.discovery.fitness import _evaluate_single

    chrom = _chrom(entry_genes=_genes(25.0, 28.0, 31.0), exit_genes=_genes(65.0, 68.0, 71.0))
    bt = _m(2.0, 0.4, 1200, dd=0.1, pf=2.0)  # 6 genes + 1200 trades -> both penalties > 0
    r = _evaluate_single(chrom, lambda c: bt, None, 50, "4h")
    assert r.complexity_penalty > 0
    assert r.overtrade_penalty > 0
    assert r.sl_tp_penalty == 0.0
    assert r.symbol_scores is None
    assert r.adjusted_score == pytest.approx(
        r.raw_score - r.complexity_penalty - r.overtrade_penalty
    )
    assert r.adjusted_score == 0.22833333333333344


# --- symbol_agg="pooled" (vibe-quant-clmmc.5 C-T1) ---------------------------------------

POOL_DAY0 = 19724  # first UTC day index of window 0 in every pooled fake


def _series(values: list[float], first_day: int = POOL_DAY0) -> Any:
    from array import array

    from vibe_quant.metrics import DailyReturns

    return DailyReturns(first_day, array("d", values))


def _pwin(values: list[float], trades: int, first_day: int = POOL_DAY0,
          pnls: tuple[float, ...] | None = None) -> dict[str, Any]:  # fmt: skip
    """One symbol's raw window result as ``_run_single`` returns it in pooled mode."""
    ret = 1.0
    for v in values:
        ret *= 1.0 + v
    return {
        **_m(1.0, ret - 1.0, trades), "mtm_daily_returns": _series(values, first_day),
        "trade_pnls": pnls if pnls is not None else (1.0,) * trades,
        "total_fees": 0.5, "gross_pnl": 2.0, "starting_balance": 1000.0, "days": len(values),
    }  # fmt: skip


class PooledFake(NTBacktestFn):
    """Module-level (picklable) pooled fake: ``table[(symbol, window start)]`` is the raw run.

    ``shift[(chromosome uid, symbol)]`` moves that symbol's series by N days (C8).
    """

    def __init__(self, table: dict[tuple[str, str], dict[str, Any]], symbols: list[str],
                 windows: list[tuple[str, str]] | None = None, min_trades: int = 0,
                 shift: dict[tuple[str, str], int] | None = None) -> None:  # fmt: skip
        super().__init__(symbols, "4h", "2024-01-01", "2024-06-01", windows=windows,
                         min_trades=min_trades, symbol_agg="pooled")  # fmt: skip
        self.table = table
        self.shift = shift or {}
        self.calls: list[tuple[str, str]] = []

    def _run_single(
        self, chromosome: Any, start_date: str, end_date: str, symbols: list[str] | None = None
    ) -> dict[str, Any]:  # type: ignore[override]
        assert symbols is not None and len(symbols) == 1
        self.calls.append((symbols[0], start_date))
        out = dict(self.table[(symbols[0], start_date)])
        days = self.shift.get((getattr(chromosome, "uid", ""), symbols[0]), 0)
        if days:
            s = out["mtm_daily_returns"]
            out["mtm_daily_returns"] = _series(list(s.values), s.first_day + days)
        return out


def _pooled_fn(per_sym: dict[str, dict[str, Any]], **kw: Any) -> PooledFake:
    """Single-window pooled fake over ``per_sym`` (symbol -> raw window result)."""
    return PooledFake({(s, "2024-01-01"): r for s, r in per_sym.items()}, list(per_sym), **kw)


STRONG = [0.0, 0.01, 0.02, -0.005, 0.015, 0.01, -0.002, 0.012, 0.008, 0.011, -0.004, 0.009]


def test_pooled_forces_series_and_splits() -> None:
    fn = _pooled_fn({"B": _pwin(STRONG, 30), "A": _pwin(STRONG, 30)})
    assert fn.collect_daily_returns is True  # every factory / holdout / full-range fn inherits
    assert fn.splits_by_symbol is True
    one = PooledFake({}, ["A"])
    assert one.splits_by_symbol is False


@pytest.mark.parametrize(
    ("n", "min_trades", "trades", "passes"),
    [
        (6, 50, 5, True), (6, 50, 4, False), (6, 50, 0, False),  # C6: floor max(5, 50 // 12) = 5
        (2, 50, 12, True), (2, 50, 11, False),  # floor max(5, 50 // 4) = 12
    ],
)  # fmt: skip
def test_pooled_symbol_trade_floor(n: int, min_trades: int, trades: int, passes: bool) -> None:
    """C6: one thin coin zeroes the score; 0 trades never divides by zero."""
    from vibe_quant.discovery.fitness import _evaluate_single, passes_train_gate

    syms = [f"S{i}" for i in range(n)]
    per_sym = {s: _pwin(STRONG, 60) for s in syms}
    per_sym[syms[-1]] = _pwin(STRONG if trades else [0.0] * len(STRONG), trades)
    fn = _pooled_fn(per_sym, min_trades=min_trades)
    fr = _evaluate_single(_chrom(), fn, None, min_trades, "4h")
    assert fr.error is None
    assert fr.pooled_symbol_stats is not None
    assert fr.pooled_symbol_stats[syms[-1]][1] == trades
    assert (fr.adjusted_score > 0) is passes
    assert passes_train_gate(fr, min_trades) is passes
    assert (fr.mtm_daily_returns is not None) is passes  # keep_series = the train gate


@pytest.mark.parametrize(("total", "passes"), [(50, True), (49, False)])
def test_pooled_total_trade_gate(total: int, passes: bool) -> None:
    from vibe_quant.discovery.fitness import _evaluate_single

    fn = _pooled_fn({"A": _pwin(STRONG, 25), "B": _pwin(STRONG, total - 25)}, min_trades=50)
    fr = _evaluate_single(_chrom(), fn, None, 50, "4h")
    assert fr.total_trades == total
    assert (fr.adjusted_score > 0) is passes


def test_pooled_runs_every_window() -> None:
    """C7: a symbol losing window 1 of 3 still runs windows 2 and 3 (no early exit)."""
    wins = [
        ("2024-01-01", "2024-02-01"),
        ("2024-02-01", "2024-03-01"),
        ("2024-03-01", "2024-04-01"),
    ]
    lose = [0.0, -0.05, -0.02, 0.01, -0.03, 0.0, -0.01, 0.0, -0.02, -0.01]
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for w, (ws, _we) in enumerate(wins):
        for sym in ("A", "B"):
            vals = lose if (sym == "A" and w == 0) else STRONG[:10]
            table[(sym, ws)] = _pwin(vals, 20, first_day=POOL_DAY0 + 10 * w)
    fn = PooledFake(table, ["A", "B"], windows=wins)
    out = fn(_chrom())
    assert fn.calls == [("A", ws) for ws, _ in wins] + [("B", ws) for ws, _ in wins]
    assert out["window_trades"] == (40, 40, 40)
    assert "early_exit" not in out
    assert len(out["mtm_daily_returns"]) == 29  # 30 days chained, day 0 dropped


def test_pooled_misaligned_series_is_error_result() -> None:
    """C8: one day of offset fails the candidate closed with the exact reason, score 0."""
    from vibe_quant.discovery.fitness import _evaluate_single

    chrom = _chrom()
    fn = _pooled_fn({"BTCUSDT": _pwin(STRONG, 30), "ETHUSDT": _pwin(STRONG, 30)},
                    shift={(chrom.uid, "ETHUSDT"): 1})  # fmt: skip
    msg = "PooledSeriesError: pooled series misaligned (ETHUSDT first day 19725 != 19724)"
    assert fn(chrom)["error"] == msg
    fr = _evaluate_single(chrom, fn, None, 10, "4h")
    assert (fr.error, fr.adjusted_score) == (msg, 0.0)


def test_pooled_length_mismatch_is_error_result() -> None:
    fn = _pooled_fn({"BTCUSDT": _pwin(STRONG, 30), "ETHUSDT": _pwin(STRONG[:-1], 30)})
    out = fn(_chrom())
    assert out["error"] == "PooledSeriesError: pooled series misaligned (ETHUSDT days 11 != 12)"


@pytest.mark.slow(reason="the process-pool case spawns a real pool, ~3 s")
@pytest.mark.parametrize("parallel", [False, True], ids=["inline", "process-pool"])
def test_pooled_misaligned_next_to_valid(parallel: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """GATES-11: an aggregate-time error never escapes the split parent; the neighbour scores."""
    from concurrent.futures import ProcessPoolExecutor

    from vibe_quant.discovery import fitness as fitness_mod
    from vibe_quant.discovery.fitness import evaluate_population

    monkeypatch.setattr(fitness_mod, "_parallel_broken", False)
    bad, good = _chrom(), _chrom()
    fn = _pooled_fn({"BTCUSDT": _pwin(STRONG, 30), "ETHUSDT": _pwin(STRONG, 30)},
                    shift={(bad.uid, "ETHUSDT"): 1})  # fmt: skip
    if parallel:
        with ProcessPoolExecutor(max_workers=2) as pool:
            got = evaluate_population([bad, good], fn, None, max_workers=2, executor=pool,
                                      min_trades=10, timeframe="4h")  # fmt: skip
    else:
        got = evaluate_population([bad, good], fn, None, min_trades=10, timeframe="4h")
    assert (
        got[0].error
        == "PooledSeriesError: pooled series misaligned (ETHUSDT first day 19725 != 19724)"
    )
    assert got[0].adjusted_score == 0.0
    assert got[1].error is None and got[1].adjusted_score > 0


def test_pooled_aggregate_data_unavailable_still_fails_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from vibe_quant.discovery.fitness import evaluate_population
    from vibe_quant.errors import DataUnavailableError

    def boom(self: Any, *a: Any) -> Any:
        raise DataUnavailableError("no bars")

    monkeypatch.setattr(NTBacktestFn, "_aggregate_pooled", boom)
    fn = _pooled_fn({"A": _pwin(STRONG, 30), "B": _pwin(STRONG, 30)})
    with pytest.raises(DataUnavailableError, match="no bars"):
        evaluate_population([_chrom()], fn, None, min_trades=10, timeframe="4h")


def test_pooled_sharpe_is_internal_252_day_mtm() -> None:
    """GATES-13: fitness Sharpe = the pooled MTM series (day 0 dropped) in 252-day units."""
    import math
    import statistics

    from vibe_quant.discovery.fitness import _evaluate_single

    other = [0.0, -0.004, 0.006, 0.001, -0.002, 0.003, 0.0, 0.004, -0.001, 0.002, 0.001, -0.003]
    fn = _pooled_fn({"A": _pwin(STRONG, 30), "B": _pwin(other, 30)})
    fr = _evaluate_single(_chrom(), fn, None, 10, "4h")
    pooled = [(a + b) / 2 for a, b in zip(STRONG, other, strict=True)][1:]
    expected = statistics.mean(pooled) / statistics.stdev(pooled) * math.sqrt(252)
    assert fr.sharpe_ratio == pytest.approx(expected, rel=1e-12)
    assert fr.sharpe_ratio == pytest.approx(14.361460812275176, rel=1e-12)


def _noise(seed: int, n: int) -> list[float]:
    import numpy as np

    x = np.random.default_rng(seed).normal(0.0, 0.01, n)
    return [0.0, *(x - x.mean()).tolist()][:n]


def _boot_lower(values: tuple[float, ...]) -> float:
    import numpy as np

    from vibe_quant.overfitting.bootstrap_sharpe import bootstrap_sharpe_ci

    return bootstrap_sharpe_ci(np.array(values), n_bootstrap=2000, seed=7).ci_lower


def test_pooled_bootstrap() -> None:
    """C9 / GATES-12: the bootstrap input is the pooled daily series (synchronised days).

    6 identical histories == one history (to rounding, same verdict); 1 strong + 5 zero-edge
    noise coins dilute it (lower bound drops). Exactly flat coins (0 trades) would leave the
    t-stat unchanged (scale-invariant): the trade floor scores that candidate 0.
    """
    from vibe_quant.discovery.fitness import _evaluate_single

    n = 120
    strong = [0.0] + [0.004 + 0.01 * ((i * 7919) % 13 - 6) / 6 for i in range(1, n)]
    alone = _evaluate_single(_chrom(), _pooled_fn({"A": _pwin(strong, 80), "B": _pwin(strong, 80)}),
                             None, 10, "4h")  # fmt: skip
    syms = [f"S{i}" for i in range(6)]
    six = _evaluate_single(
        _chrom(), _pooled_fn({s: _pwin(strong, 80) for s in syms}), None, 10, "4h"
    )
    assert len(six.trade_returns) == n - 1
    single_lower = _boot_lower(tuple(strong[1:]))
    assert _boot_lower(six.trade_returns) == pytest.approx(single_lower, rel=1e-9)
    assert _boot_lower(alone.trade_returns) == pytest.approx(single_lower, rel=1e-9)
    assert single_lower > 1.0

    mixed = {
        "S0": _pwin(strong, 80),
        **{s: _pwin(_noise(i, n), 80) for i, s in enumerate(syms[1:])},
    }
    diluted = _evaluate_single(_chrom(), _pooled_fn(mixed), None, 10, "4h")
    assert _boot_lower(diluted.trade_returns) < single_lower

    flat = {"S0": _pwin(strong, 80), **{s: _pwin([0.0] * n, 0) for s in syms[1:]}}
    fr = _evaluate_single(_chrom(), _pooled_fn(flat), None, 10, "4h")
    assert fr.adjusted_score == 0.0  # trade floor max(5, 10 // 12) = 5 per coin


def _hadamard_rows() -> list[list[float]]:
    h = [[1.0]]
    for _ in range(3):
        h = [r + r for r in h] + [r + [-v for v in r] for r in h]
    return h  # 8 x 8; rows 1..7 zero-mean and mutually orthogonal


def _k_series(rows: list[list[float]]) -> dict[str, Any]:
    return {f"S{i}": _series([v / 64 for v in r]) for i, r in enumerate(rows)}


def test_pooled_k_eff() -> None:
    """C10: K_eff = N / (1 + (N-1) rho_bar) over non-constant aligned symbol series."""
    from vibe_quant.discovery.backtest_fn import pooled_k_eff

    h = _hadamard_rows()
    z = h[1]
    six = [[a + b for a, b in zip(z, h[i], strict=True)] for i in range(2, 8)]  # pairwise rho 0.5
    assert pooled_k_eff(_k_series(six)) == (1.7142857142857142, None, 0.5)
    anti = [
        [a + b for a, b in zip(z, h[2], strict=True)],
        [-a + b for a, b in zip(z, h[3], strict=True)],
    ]
    assert pooled_k_eff(_k_series(anti)) == (4.0, None, -0.5)  # just above the -1/(N-1) edge
    opposite = [h[1], [-v for v in h[1]]]
    assert pooled_k_eff(_k_series(opposite)) == (
        None,
        "mean correlation -1.0000: K_eff undefined",
        -1.0,
    )
    const = [0.5] * 8
    assert pooled_k_eff(_k_series([six[0], const, six[1]])) == (2 / 1.5, None, 0.5)
    assert pooled_k_eff(_k_series([six[0], const, const])) == (
        None, "fewer than 2 non-constant symbol series", None,
    )  # fmt: skip
    shifted = {
        "A": _series([v / 64 for v in six[0]]),
        "B": _series([v / 64 for v in six[1]], POOL_DAY0 + 1),
    }
    assert pooled_k_eff(shifted) == (None, "symbol series misaligned", None)


def test_pooled_symbol_order_and_shape() -> None:
    """C4 (unit): the pooled boundary sorts symbols once; no per-symbol soft score rides along."""
    a, b = _pwin(STRONG, 30, pnls=(2.0, -1.0)), _pwin(STRONG[::-1], 40, pnls=(3.0, -4.0))
    fwd = PooledFake({("A", "2024-01-01"): a, ("B", "2024-01-01"): b}, ["A", "B"])(_chrom())
    rev = PooledFake({("A", "2024-01-01"): a, ("B", "2024-01-01"): b}, ["B", "A"])(_chrom())
    assert list(fwd["symbol_mtm_daily"]) == list(rev["symbol_mtm_daily"]) == ["A", "B"]
    for key in ("sharpe_ratio", "total_return", "max_drawdown", "profit_factor", "total_trades"):
        assert fwd[key] == rev[key]
    assert list(fwd["mtm_daily_returns"].values) == list(rev["mtm_daily_returns"].values)
    assert fwd["profit_factor"] == 5.0 / 5.0  # trade PnLs of both coins, one pool
    assert fwd["pooled_symbol_stats"] == {
        "A": (a["total_return"], 30),
        "B": (b["total_return"], 40),
    }
    assert "symbol_results" not in fwd and "synthetic_symbols" not in fwd
    assert fwd["trades_sum"] == 70 and fwd["starting_balance"] == 2000.0
    assert fwd["days"] == 12  # shared calendar of the coins, not their sum (fee-rate input)


def test_pooled_drawdown_counts_a_losing_first_day() -> None:
    """Pooled max drawdown is of the daily-close equity INCLUDING day 0 (only Sharpe drops it)."""
    v = [-0.05, 0.02, -0.01, 0.03]
    out = PooledFake({("A", "2024-01-01"): _pwin(v, 10), ("B", "2024-01-01"): _pwin(v, 10)},
                     ["A", "B"])(_chrom())  # fmt: skip
    assert out["max_drawdown"] == 0.050000000000000044  # 1 - 0.95; without day 0 it is 0.01


# --- C-T2: pooled gates (sign floor, DSR / bootstrap input, buy-hold, notes) -----------------


def _pooled_fit(per_sym: dict[str, dict[str, Any]], min_trades: int = 10) -> Any:
    from vibe_quant.discovery.fitness import _evaluate_single

    fr = _evaluate_single(
        _chrom(), _pooled_fn(per_sym, min_trades=min_trades), None, min_trades, "4h"
    )
    assert fr.error is None
    return fr


LOSE = [0.0, -0.01, 0.002, -0.004, -0.003, 0.001, -0.002, -0.001, 0.0, -0.002, 0.001, -0.003]


@pytest.mark.parametrize(
    ("n", "positive", "reasons"),
    [
        (2, 1, ["pooled sign floor: 1/2 < 2"]), (2, 2, []),
        (3, 1, ["pooled sign floor: 1/3 < 2"]), (3, 2, []),
        (4, 2, ["pooled sign floor: 2/4 < 3"]), (4, 3, []),
        (5, 2, ["pooled sign floor: 2/5 < 3"]), (5, 3, []),
        (6, 2, ["pooled sign floor: 2/6 < 4"]), (6, 3, ["pooled sign floor: 3/6 < 4"]),
        (6, 4, []),
    ],
)  # fmt: skip
def test_pooled_sign_floor(n: int, positive: int, reasons: list[str]) -> None:
    """C5: need = (3N + 4) // 5 profitable coins (2, 2, 3, 3, 4 for N = 2..6), each edge."""
    from vibe_quant.discovery.gate_chain import pooled_symbol_gate_reasons

    syms = [f"S{i}" for i in range(n)]
    fr = _pooled_fit({s: _pwin(STRONG if i < positive else LOSE, 30) for i, s in enumerate(syms)})
    assert sum(r > 0 for r, _t in fr.pooled_symbol_stats.values()) == positive
    assert pooled_symbol_gate_reasons(fr) == reasons


def test_pooled_sign_floor_zero_return_is_not_positive() -> None:
    from vibe_quant.discovery.gate_chain import pooled_symbol_gate_reasons

    fr = _pooled_fit({"A": _pwin(STRONG, 30), "B": _pwin([0.0] * len(STRONG), 30)})
    assert fr.pooled_symbol_stats["B"] == (0.0, 30)
    assert pooled_symbol_gate_reasons(fr) == ["pooled sign floor: 1/2 < 2"]


def test_pooled_gate_fails_closed_without_symbol_stats() -> None:
    """Gates fail closed: a pooled-mode fitness with no per-coin stats is rejected at stage
    ``pooled_symbol_train``, never waved through to DSR."""
    from dataclasses import replace

    from vibe_quant.discovery.gate_chain import pooled_symbol_gate_reasons

    fr = replace(
        _pooled_fit({"A": _pwin(STRONG, 30), "B": _pwin(STRONG, 30)}), pooled_symbol_stats=None
    )
    assert pooled_symbol_gate_reasons(fr) == ["pooled symbol stats missing"]
    pipe = _pooled_pipe(require_dsr=True)
    assert pipe.gates.screen([(_chrom(), fr)], TrialBudget(1, {})) == []
    (row,) = pipe.gates.rejections
    assert (row["stage"], row["reasons"]) == (
        "pooled_symbol_train",
        ["pooled symbol stats missing"],
    )


WEAK = [0.0, 0.003, -0.004, 0.002, -0.001, 0.003, -0.003, 0.002, -0.002, 0.001, -0.001, 0.0021]


def _pooled_pipe(**cfg: Any) -> Any:
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    fn = _pooled_fn({"A": _pwin(STRONG, 30), "B": _pwin(STRONG, 30)})
    return DiscoveryPipeline(_cfg(symbol_agg="pooled", min_trades=10, **cfg), backtest_fn=fn)


def test_pooled_screen_dsr_on_pool_no_per_symbol_floor() -> None:
    """C11: DSR tests the pooled chained series; no per-symbol PSR floor (worst mode's)."""
    fr = _pooled_fit({"A": _pwin(STRONG, 30), "B": _pwin(STRONG, 30), "C": _pwin(STRONG, 30),
                      "D": _pwin(WEAK, 30)})  # fmt: skip
    pipe = _pooled_pipe(require_dsr=True)
    chrom = _chrom()
    pipe.gates.screen([(chrom, fr)], TrialBudget(1, {}))
    rec = pipe.gates.dsr_records[chrom.uid]
    assert rec["input"] == "pooled_mtm_daily_returns"
    assert rec["observations"] == len(fr.mtm_daily_returns) == len(STRONG) - 1
    assert "symbol_psr_floor" not in rec and "symbols" not in rec
    reasons = [r for row in pipe.gates.rejections for r in row["reasons"]]  # type: ignore[attr-defined]
    assert not any("per-symbol" in r for r in reasons)


def test_pooled_sign_floor_rejects_before_dsr() -> None:
    """C11: a sign-floor failure is stage ``pooled_symbol_train`` and never reaches DSR."""
    fr = _pooled_fit({"A": _pwin(STRONG, 30), "B": _pwin(LOSE, 30)})
    for pool, dsr in (
        ("top_k", None),
        ("all", {"status": "skipped", "reason": "pooled_symbol_train"}),
    ):
        pipe = _pooled_pipe(require_dsr=True, gate_pool=pool)
        chrom = _chrom()
        assert pipe.gates.screen([(chrom, fr)], TrialBudget(1, {})) == []
        (row,) = pipe.gates.rejections
        assert row["stage"] == "pooled_symbol_train"
        assert row["reasons"] == ["pooled sign floor: 1/2 < 2"]
        assert "dsr" not in row
        assert pipe.gates.dsr_records.get(chrom.uid) == dsr


@pytest.mark.parametrize(
    ("days", "insufficient"), [(12, False), (4, True)], ids=["ci", "too-short"]
)
def test_pooled_bootstrap_reason_counts_days(days: int, insufficient: bool) -> None:
    """Carry-over: the pooled bootstrap input is daily pooled returns, worded as such."""
    vals = [0.0, 0.001, 0.002, 0.0005, 0.001, -0.001, 0.0, 0.0002, -0.0003, 0.001, 0.0, -0.0001]
    fr = _pooled_fit({"A": _pwin(vals[:days], 30), "B": _pwin(vals[:days], 30)})
    pipe = _pooled_pipe(require_bootstrap_ci=True, bootstrap_min_sharpe=50.0)
    assert pipe.gates.screen([(_chrom(), fr)], TrialBudget(1, {})) == []
    (reason,) = [r for r in pipe.gates.rejections[0]["reasons"] if "Bootstrap" in r]  # type: ignore[attr-defined]
    if insufficient:
        assert (
            reason == "Bootstrap CI required but insufficient pooled daily returns (got 3, need ≥5)"
        )
    else:
        assert reason.endswith(", pooled daily returns (n=11))"), reason
        assert "trade" not in reason


def test_portfolio_bootstrap_reason_unchanged() -> None:
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    fit = _eval({("A",): _m(2.0, 0.4, 120), ("B",): _m(1.0, 0.2, 120)})
    pipe = DiscoveryPipeline(_cfg(require_bootstrap_ci=True, bootstrap_min_sharpe=50.0),
                             backtest_fn=lambda c: {})  # fmt: skip
    pipe.gates.screen([(_chrom(), fit)], TrialBudget(1, {}))
    (reason,) = [r for r in pipe.gates.rejections[0]["reasons"] if "Bootstrap" in r]  # type: ignore[attr-defined]
    assert reason == "Bootstrap CI required but insufficient trade returns (got 4, need ≥5)"


def test_pooled_ranks_sign_floor_passers_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """The top-K sort puts sign-floor passers first, like worst mode's gate passers."""
    from vibe_quant.discovery.pipeline import DiscoveryPipeline

    failers = [_chrom(stop_loss_pct=sl) for sl in (1.0, 1.4, 1.8, 2.2, 2.6)]
    passer = _chrom(stop_loss_pct=3.0)
    big = [0.0, 0.05, 0.04, -0.01, 0.06, 0.05, -0.02, 0.04, 0.03, 0.05, -0.01, 0.04]
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for i, c in enumerate(failers):  # A carries a losing B: high pooled score, sign floor 1/2
        table[c.uid] = {"A": _pwin([v * (1 + i / 100) for v in big], 30), "B": _pwin(LOSE, 30)}
    table[passer.uid] = {"A": _pwin(STRONG, 30), "B": _pwin(STRONG, 30)}

    class ByUid(PooledFake):
        def _run_single(
            self, chromosome: Any, start_date: str, end_date: str, symbols: list[str] | None = None
        ) -> dict[str, Any]:  # type: ignore[override]
            assert symbols is not None
            return dict(table[chromosome.uid][symbols[0]])

    fn = ByUid({}, ["A", "B"], min_trades=10)
    pipe = DiscoveryPipeline(
        _cfg(
            symbol_agg="pooled",
            min_trades=10,
            max_generations=1,
            min_diversity_distance=0.0,
            require_buy_hold=False,
            gate_pool="top_k",
        ),
        backtest_fn=fn,
    )
    monkeypatch.setattr(
        "vibe_quant.discovery.pipeline.initialize_population", lambda n, **kw: [*failers, passer]
    )
    monkeypatch.setattr(DiscoveryPipeline, "_apply_indicator_pool_filter", lambda self: None)
    monkeypatch.setattr(DiscoveryPipeline, "_preflight_aux_data", lambda self: None)
    monkeypatch.setattr(DiscoveryPipeline, "_preflight_bars", lambda self: None)
    result = pipe.run()
    assert passer.uid in [c.uid for c, _ in pipe._all_scored[:5]]
    assert [c.uid for c, _ in result.top_strategies] == [passer.uid]
    assert [r["stage"] for r in result.guardrail_rejections] == ["pooled_symbol_train"] * 4


def test_pooled_buy_hold_sorted_equal_weight() -> None:
    """C15: pooled buy-hold is the portfolio benchmark (per_symbol False) over SORTED symbols;
    a reversed launch list gives identical record bytes."""
    import json

    from vibe_quant.discovery.gate_chain import GateChain, GateEvidence

    fr = _pooled_fit({"A": _pwin(STRONG, 30), "B": _pwin(STRONG, 30)})
    closes = {
        "A": {POOL_DAY0 + i: 100.0 * (1.0 + 0.001 * i) for i in range(12)},
        "B": {POOL_DAY0 + i: 50.0 * (1.0 - 0.002 * i) for i in range(12)},
    }
    seen: list[list[str]] = []

    def load(symbols: list[str]) -> dict[str, dict[int, float]]:
        seen.append(list(symbols))
        return closes

    records = []
    for order in (["A", "B"], ["B", "A"]):
        fn = PooledFake({}, order)
        chain = GateChain(_cfg(symbol_agg="pooled", min_trades=10),
                          GateEvidence(fn, True, buy_hold_closes_fn=load))  # fmt: skip
        chrom = _chrom()
        kept = chain.apply_buy_hold([(chrom, fr)])
        records.append(json.dumps(chain.buy_hold_records[chrom.uid]))
        assert len(kept) in (0, 1)
    assert seen == [["A", "B"], ["A", "B"]]
    assert records[0] == records[1]
    assert '"per_symbol"' not in records[0] or '"per_symbol": false' in records[0]


_BASIS = {"sharpe": "pooled daily MTM (252-day)", "max_dd": "daily-close"}


@pytest.mark.parametrize(
    ("per_sym", "block"),
    [
        ({"B": (LOSE, 30), "A": (STRONG, 40)},
         {"symbols": {"A": {"return": 0.08693929208190121, "trades": 40},
                      "B": {"return": -0.020854354288419752, "trades": 30}},
          "positive": 1, "need": 2, "k_eff": 1.830851819849363, "k_eff_reason": None,
          "mean_corr": 0.09238769534311837, "basis": _BASIS}),
        ({"C": (WEAK, 12), "A": (STRONG, 30), "B": (STRONG[::-1], 25)},
         {"symbols": {"A": {"return": 0.08693929208190121, "trades": 30},
                      "B": {"return": 0.08693929208190099, "trades": 25},
                      "C": {"return": 0.0020709293627332315, "trades": 12}},
          "positive": 3, "need": 2, "k_eff": 3.683159335726193, "k_eff_reason": None,
          "mean_corr": -0.09274094241588078, "basis": _BASIS}),
    ],
    ids=["two-coins", "three-coins"],
)  # fmt: skip
def test_pooled_notes_block(per_sym: dict[str, Any], block: dict[str, Any]) -> None:
    """§2.3 notes: the champion entry and every rejection row carry the same ``pooled`` block
    (sorted symbols, sign floor, K_eff, basis labels; pooled max DD is daily-close)."""
    import json

    from vibe_quant.discovery.gate_chain import rejection_record
    from vibe_quant.discovery.record import champion_entry

    fr = _pooled_fit({s: _pwin(v, t) for s, (v, t) in per_sym.items()})
    chrom = _chrom()
    entry = champion_entry(chrom, fr, timeframe="4h", dsr=None, buy_hold=None)
    row = rejection_record(chrom, fr, "holdout", ["Holdout: x"], timeframe="4h")
    assert entry["pooled"] == block
    assert row["pooled"] == block
    assert json.dumps(entry["pooled"]) == json.dumps(block)  # key order is the contract


def test_portfolio_and_worst_rows_have_no_pooled_block() -> None:
    from vibe_quant.discovery.gate_chain import rejection_record
    from vibe_quant.discovery.record import champion_entry

    fit = _eval({("A",): _m(2.0, 0.4, 120), ("B",): _m(1.0, 0.2, 120)})
    assert "pooled" not in champion_entry(_chrom(), fit, timeframe="4h", dsr=None, buy_hold=None)
    assert "pooled" not in rejection_record(_chrom(), fit, "guardrails", [], timeframe="4h")


@pytest.mark.parametrize(
    ("shifted", "coins", "message"),
    [
        ("BTCUSDT", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], "BTCUSDT first day 19725 != 19724"),
        ("BTCUSDT", ["BTCUSDT", "ETHUSDT"], "BTCUSDT first day 19725 != 19724"),
        ("SOLUSDT", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], "SOLUSDT first day 19725 != 19724"),
    ],
    ids=["sorted-first-of-3", "sorted-first-of-2", "last-of-3"],
)  # fmt: skip
def test_pooled_misaligned_names_the_shifted_coin(
    shifted: str, coins: list[str], message: str
) -> None:
    """Carry-over: the message names the coin whose first day differs from the majority (ties:
    the earliest first day is the reference), also when it sorts first."""
    chrom = _chrom()
    fn = _pooled_fn({c: _pwin(STRONG, 30) for c in coins}, shift={(chrom.uid, shifted): 1})
    assert fn(chrom)["error"] == f"PooledSeriesError: pooled series misaligned ({message})"


def _pooled_mock(chrom: Any) -> dict[str, Any]:
    """The mock run plus the per-coin stats a pooled NT run reports (gates fail closed
    without them); both coins profitable with the run's trade count."""
    from vibe_quant.discovery.mock_backtest import mock_backtest

    out = mock_backtest(chrom)
    stat = (abs(float(out.get("total_return", 0.0))) + 0.01, int(out.get("total_trades", 0)))
    return {**out, "pooled_symbol_stats": {"BTCUSDT": stat, "ETHUSDT": stat}}


@pytest.mark.slow(reason="a whole CLI discovery run per mode (mock engine), ~2 s each")
@pytest.mark.parametrize(("agg", "fns"), [("portfolio", 1), ("worst", 2), ("pooled", 2)])
def test_full_range_headline_runs_for_pooled(
    agg: str, fns: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carry-over: a pooled champion always gets the full-range headline run (trades_sum /
    symbol_metrics), even with one eval window and no split; portfolio keeps the fallback."""
    import vibe_quant.discovery.run as discovery_run
    from vibe_quant.discovery.mock_backtest import mock_backtest as _mock_backtest

    made: list[dict[str, Any]] = []

    def fake_make(**kw: Any) -> Any:
        made.append(kw)
        return _pooled_mock if kw["symbol_agg"] == "pooled" else _mock_backtest

    monkeypatch.setattr(discovery_run, "_make_nt_backtest_fn", fake_make)
    monkeypatch.setattr(discovery_run, "_check_data_available", lambda syms: True)
    db = tmp_path / "fr.db"
    rid = _create_run(db)
    monkeypatch.setattr("sys.argv", _argv(
        db, rid, "--symbol-agg", agg, "--eval-windows", "1", "--train-test-split", "0",
        "--no-dsr", "--no-bootstrap-ci", "--no-buy-hold-gate", "--seed", "3",
        "--population-size", "4", "--max-generations", "1",
    ))  # fmt: skip
    assert main() == 0
    assert _notes(db, rid)["top_strategies"], "fixture needs a champion"
    assert len(made) == fns
    assert all(kw["symbol_agg"] == agg for kw in made)


POOLED_NOTE = (
    "reference is pooled (own account per coin); validation is one shared account: "
    "Sharpe and trades not compared"
)


@pytest.mark.parametrize(("val_sharpe", "val_trades"), [(-1.0, 10), (5.0, 999)])
def test_pooled_consistency_note_no_flag(
    state: StateManager, val_sharpe: float, val_trades: int
) -> None:
    """C14 validation side: a pooled discovery reference is labelled, never compared."""
    import json

    from vibe_quant.validation.consistency import assess_consistency, find_screening_reference

    rid = _worst_discovery_run(state, full_range_trades_sum=60)
    state.update_result_notes(rid, json.dumps({"symbol_agg": "pooled", "top_strategies": [{
        "dsl": {"name": "genome_abc123"}, "full_range_sharpe": 2.0, "full_range_trades": 60,
        "full_range_trades_sum": 60}]}))  # fmt: skip
    ref = find_screening_reference(state, 999, "genome_abc123")
    assert ref is not None
    report = assess_consistency(ref, val_sharpe, val_trades)
    assert report.flags == []
    assert report.notes == [POOLED_NOTE]


def test_pooled_screening_run_reference_note_no_flag(state: StateManager) -> None:
    """A pooled screening child (run param ``symbol_agg``) is a reference that is not compared,
    and ``symbol_agg`` never blocks the exact-row parameter match."""
    from vibe_quant.validation.consistency import (
        _NON_STRATEGY_PARAM_KEYS,
        assess_consistency,
        find_screening_reference,
    )

    assert "symbol_agg" in _NON_STRATEGY_PARAM_KEYS
    sid = state.create_strategy(name="genome_p", dsl_config={"name": "genome_p"})
    for params in ({"symbol_agg": "pooled"}, {}):
        rid = state.create_backtest_run(
            strategy_id=sid, run_mode="screening", symbols=["BTCUSDT", "ETHUSDT"],
            timeframe="4h", start_date="2024-01-01", end_date="2025-01-01", parameters=params,
        )  # fmt: skip
        state.update_backtest_run_status(rid, "completed")
        state.conn.execute(
            "INSERT INTO sweep_results (run_id, parameters, sharpe_ratio, total_trades,"
            " is_pareto_optimal) VALUES (?, '{}', 3.0, 80, 1)", (rid,),
        )  # fmt: skip
        state.conn.commit()
        ref = find_screening_reference(
            state, sid, "genome_p", {"symbol_agg": "pooled"} if params else {},
            ("2024-01-01", "2025-01-01"), ["BTCUSDT", "ETHUSDT"],
        )  # fmt: skip
        assert ref is not None and ref.source == f"screening_run:{rid}"  # exact row matched
        report = assess_consistency(ref, -1.0, 10)
        if params:
            assert (report.flags, report.notes) == ([], [POOLED_NOTE])
        else:
            assert report.flags  # the shared-account reference still compares


def test_pooled_replay_drift_uses_trades_sum(state: StateManager) -> None:
    """C14: the exact pooled replay compares Sharpe two-sided and trades with the summed count."""
    e = {"dsl": {"name": "g"}, "full_range_sharpe": 1.0, "full_range_trades": 60,
         "full_range_trades_sum": 130}  # fmt: skip
    notes = {"symbol_agg": "pooled", "top_strategies": [e]}
    out = _drift(state, notes, 130)
    assert out["flagged"] is False and out["discovery_trades"] == 130
    assert _drift(state, notes, 60)["flagged"] is True  # 60 vs the 130 sum
    assert _drift(state, notes, 130, scr_sharpe=1.5)["flagged"] is True  # exact replay: two-sided


def _check_consistency(state: StateManager, val_sharpe: float, val_trades: int) -> Any:
    """The production validation-runner check whose return value lands in ``notes.consistency``."""
    from types import SimpleNamespace

    from vibe_quant.validation.runner import ValidationRunner

    runner = object.__new__(ValidationRunner)
    runner._state = state
    runner._build_strategy_params = lambda cfg: {}
    return runner._check_screening_consistency(
        1, 999, "genome_abc123", SimpleNamespace(sharpe_ratio=val_sharpe, total_trades=val_trades),
        run_config={"start_date": "2024-01-01", "end_date": "2025-01-01", "parameters": {},
                    "symbols": ["BTCUSDT", "ETHUSDT"]},
    )  # fmt: skip


def _set_agg(state: StateManager, rid: int, agg: str) -> None:
    import json

    state.update_result_notes(rid, json.dumps({"symbol_agg": agg, "top_strategies": [{
        "dsl": {"name": "genome_abc123"}, "full_range_sharpe": 2.0, "full_range_trades": 60,
        "full_range_trades_sum": 60}]}))  # fmt: skip


def test_pooled_validation_persists_the_note_unflagged(state: StateManager) -> None:
    """B1: a validation of a pooled champion carries the pooled label in its result notes,
    with no flag, even when it collapses (Sharpe 2.0 -> -1.0 is not compared)."""
    rid = _worst_discovery_run(state)
    _set_agg(state, rid, "pooled")
    assert _check_consistency(state, -1.0, 10) == {
        "flags": [], "screen_sharpe": 2.0, "screen_trades": 60,
        "screen_source": f"discovery_run:{rid}", "val_sharpe": -1.0, "val_trades": 10,
        "notes": [POOLED_NOTE],
    }  # fmt: skip


def test_worst_validation_persists_the_note_unflagged(state: StateManager) -> None:
    """B1: the worst-mode basis note reaches the result notes without a flag too."""
    rid = _worst_discovery_run(state, full_range_trades_sum=60)
    report = _check_consistency(state, 1.0, 60)
    assert report["screen_source"] == f"discovery_run:{rid}"
    assert (report["flags"], report["notes"]) == (
        [],
        ["reference Sharpe = worst symbol (discovery symbol_agg=worst)"],
    )


def test_plain_validation_persists_no_report(state: StateManager) -> None:
    """B1: a consistent run with no note persists nothing (ordinary rows unchanged)."""
    rid = _worst_discovery_run(state)
    _set_agg(state, rid, "portfolio")
    assert _check_consistency(state, 2.0, 60) is None
    assert _check_consistency(state, -1.0, 60)["flags"]  # the flag path still reports


@pytest.mark.parametrize(
    ("agg", "expected"),
    [
        ("pooled", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
        ("portfolio", ["SOLUSDT", "ETHUSDT", "BTCUSDT"]),
        ("worst", ["SOLUSDT", "ETHUSDT", "BTCUSDT"]),
    ],
)  # fmt: skip
def test_buy_hold_notes_symbols_sorted_for_pooled(agg: str, expected: list[str]) -> None:
    """S1: notes ``buy_hold_gate.symbols`` lists the coins the pooled benchmark ran over
    (sorted), whatever the launch order; other modes keep the launch order."""
    from vibe_quant.discovery.pipeline import DiscoveryResult
    from vibe_quant.discovery.record import provenance_notes

    notes = provenance_notes(
        DiscoveryResult([], [], 0, False, None), use_mock=True, eval_windows_count=1,
        eval_windows=None, symbol_agg=agg, split_ratio=0.0, direction=None,
        cross_window_months=[], cross_window_min_sharpe=0.5, wfa_oos_step_days=0,
        wfa_min_consistency=0.75, num_seeds=1, seed=42, bootstrap_min_sharpe=0.0,
        holdout_min_sharpe=0.0, symbols=["SOLUSDT", "ETHUSDT", "BTCUSDT"],
    )  # fmt: skip
    assert notes["buy_hold_gate"]["symbols"] == expected  # type: ignore[index]
