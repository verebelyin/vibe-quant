"""Opt-in per-symbol worst-of aggregation for discovery (vibe-quant-kzowb)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery.__main__ import build_parser, main
from vibe_quant.discovery.backtest_fn import NTBacktestFn
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from pathlib import Path


def _m(sharpe: float, ret: float, trades: int, dd: float = 0.1, pf: float = 1.5) -> dict[str, Any]:
    return {
        "sharpe_ratio": sharpe, "total_return": ret, "total_trades": trades,
        "max_drawdown": dd, "profit_factor": pf, "skewness": 0.0, "kurtosis": 3.0,
        "trade_returns": (ret,) * 2,
    }  # fmt: skip


class FakeFn(NTBacktestFn):
    """NTBacktestFn whose per-run backtest is a table lookup (no NT)."""

    def __init__(self, table: dict[tuple[str, ...], dict[str, Any]], *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.table = table
        self.calls: list[tuple[str, ...]] = []

    def _run_single(self, chromosome: Any, start_date: str, end_date: str,
                    symbols: list[str] | None = None) -> dict[str, Any]:  # type: ignore[override]
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


def test_worst_mode_early_exit_skips_remaining_symbols() -> None:
    table = {
        ("A",): _m(-1.0, -0.2, 100),
        ("B",): _m(1.0, 0.1, 100),
        ("C",): _m(1.0, 0.1, 100),
    }
    fn = FakeFn(table, ["A", "B", "C"], "4h", "2024-01-01", "2024-06-01",
                symbol_agg="worst", symbol_early_exit=True)  # fmt: skip
    out = fn(object())  # type: ignore[arg-type]
    assert fn.calls == [("A",)]
    assert out["early_exit_symbol"] == 0
    assert out["sharpe_ratio"] == -1.0
    assert out["total_return"] == -0.2  # real value kept for a return stop


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


def test_cli_parser_symbol_agg() -> None:
    p = build_parser()
    assert p.parse_args(["--run-id", "1"]).symbol_agg == "portfolio"
    assert p.parse_args(["--run-id", "1", "--symbol-agg", "worst"]).symbol_agg == "worst"
    with pytest.raises(SystemExit):
        p.parse_args(["--run-id", "1", "--symbol-agg", "bogus"])


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
    import vibe_quant.discovery.__main__ as dm
    from vibe_quant.discovery.mock_backtest import mock_backtest as _mock_backtest

    made: list[dict[str, Any]] = []

    def fake_make(**kw: Any) -> Any:
        made.append(kw)
        return _mock_backtest

    factory_modes: list[str] = []

    def fake_ntfn(syms: Any, tf: Any, s: Any, e: Any, *a: Any, **kw: Any) -> Any:
        factory_modes.append(kw.get("symbol_agg", "portfolio"))
        return _mock_backtest

    real_pipeline = dm.DiscoveryPipeline

    def spy_pipeline(*a: Any, **kw: Any) -> Any:
        factory = kw.get("backtest_fn_factory")
        if factory is not None:
            factory("2025-01-01", "2025-02-01")
        return real_pipeline(*a, **kw)

    monkeypatch.setattr(dm, "DiscoveryPipeline", spy_pipeline)
    monkeypatch.setattr(dm, "_make_nt_backtest_fn", fake_make)
    monkeypatch.setattr(dm, "NTBacktestFn", fake_ntfn)
    monkeypatch.setattr(dm, "_check_data_available", lambda syms: True)
    db = tmp_path / "h.db"
    rid = _create_run(db)
    monkeypatch.setattr(
        "sys.argv",
        _argv(db, rid, "--symbol-agg", "worst", "--eval-windows", "2",
              "--cross-window-months", "1"),  # fmt: skip
    )
    main()
    assert len(made) >= 2  # GA fn + holdout (+ full-range / train-return)
    assert all(kw.get("symbol_agg") == "worst" for kw in made), made
    assert factory_modes
    assert set(factory_modes) == {"worst"}


# --- fix round 1: consistency / drift / early-exit scope / notes -----------


def test_symbol_early_exit_only_when_enabled() -> None:
    table = {("A",): _m(-1.0, -0.2, 100), ("B",): _m(1.0, 0.1, 100)}
    on = FakeFn(table, ["A", "B"], "4h", "2024-01-01", "2024-06-01",
                symbol_agg="worst", symbol_early_exit=True)  # fmt: skip
    on(object())  # type: ignore[arg-type]
    assert on.calls == [("A",)]
    off = FakeFn(table, ["A", "B"], "4h", "2024-01-01", "2024-06-01", symbol_agg="worst")
    out = off(object())  # type: ignore[arg-type]
    assert off.calls == [("A",), ("B",)]  # holdout/full-range/etc. compute everything
    assert out["sharpe_ratio"] == -1.0 and out["total_return"] == -0.2  # real values
    assert out["symbol_metrics"]["B"]["trades"] == 100  # type: ignore[index]
    assert out["trades_sum"] == 200


def test_symbol_early_exit_keeps_real_values_for_return_stop() -> None:
    table = {("A",): _m(-0.7, -0.2, 100), ("B",): _m(1.0, 0.1, 100)}
    fn = FakeFn(table, ["A", "B"], "4h", "2024-01-01", "2024-06-01",
                symbol_agg="worst", symbol_early_exit=True)  # fmt: skip
    out = fn(object())  # type: ignore[arg-type]
    assert out["early_exit_symbol"] == 0
    assert out["sharpe_ratio"] == -0.7 and out["total_return"] == -0.2


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
