"""Exact pooled replay (vibe-quant-clmmc.5 C14, user Q12 = exact).

A pooled champion's screening children (replay and screening promote) run the pooled evaluator
itself, so their one sweep row reproduces the discovery full-range headline exactly and the
drift check stays clean. A validation promote stays one shared account: it gets no
``symbol_agg`` run parameter. Real NautilusTrader over the local catalog (BTC + ETH 4h, three
months); skipped without one.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from httpx import AsyncClient

    from vibe_quant.db.state_manager import StateManager

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CATALOG_PATH / "data" / "bar").exists(), reason="no data catalog"
)

SYMBOLS = ["ETHUSDT", "BTCUSDT"]  # launch order is not sorted on purpose
START, END = "2024-01-01", "2024-04-01"
COST_EDGE_KEYS = ("turnover", "gross_edge_bps", "cost_paid_bps", "breakeven_fee_bps")
CELL = {"liquidation_mode": "isolated", "leverage": 5.0, "max_margin_pct": 0.5}


def _chrom() -> Any:
    from vibe_quant.discovery.operators import (
        ConditionType,
        Direction,
        StrategyChromosome,
        StrategyGene,
    )

    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0, None)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0, None)],
        stop_loss_pct=2.0, take_profit_pct=4.0, direction=Direction.LONG,
    )  # fmt: skip


def _headline(chrom: Any, cell: dict[str, Any] | None) -> tuple[dict[str, Any], Any]:
    """The discovery full-range headline, built as run.py builds it (one pooled fn per cell),
    and the pooled evaluation's ``cost_edge``."""
    from vibe_quant.bankroll_grid import BankrollCell
    from vibe_quant.discovery.backtest_fn import NTBacktestFn, full_range_headline
    from vibe_quant.sizing_settings import SizingSettings

    kw: dict[str, Any] = {}
    if cell is not None:
        c = BankrollCell.from_note(cell)
        kw = {"liquidation": c.liquidation, "sizing": c.sizing(SizingSettings())}
    bt = NTBacktestFn(SYMBOLS, "4h", START, END, symbol_agg="pooled", **kw)(chrom)
    assert "error" not in bt
    head = full_range_headline(
        bt, fallback_sharpe=0.0, fallback_trades=0, fallback_max_dd=0.0, fallback_pf=0.0,
        fallback_return=0.0,
    )  # fmt: skip
    return head, bt.get("cost_edge")


def _discovery_run(state: StateManager, entries: list[dict[str, Any]], grid: bool) -> int:
    run_id = state.create_backtest_run(
        strategy_id=None, run_mode="discovery", symbols=SYMBOLS, timeframe="4h",
        start_date=START, end_date=END, parameters={"symbol_agg": "pooled"},
    )  # fmt: skip
    state.update_backtest_run_status(run_id, "completed")
    state.save_backtest_result(run_id, {"total_trades": 0}, trades=[])
    notes: dict[str, Any] = {"type": "discovery", "symbol_agg": "pooled", "top_strategies": entries}
    if grid:
        notes["bankroll_grid"] = {"cells": [CELL]}
    state.update_result_notes(run_id, json.dumps(notes))
    return run_id


def _entry(chrom: Any, head: dict[str, Any], cell: dict[str, Any] | None) -> dict[str, Any]:
    from vibe_quant.discovery.genome import chromosome_to_dsl

    entry: dict[str, Any] = {"dsl": {**chromosome_to_dsl(chrom), "timeframe": "4h"}, **head}
    if cell is not None:
        entry["bankroll"] = dict(CELL)
    return entry


@asynccontextmanager
async def _client(db: Path) -> AsyncIterator[tuple[AsyncClient, StateManager]]:
    from unittest.mock import patch

    from httpx import ASGITransport, AsyncClient

    from vibe_quant.api.app import create_app
    from vibe_quant.api.ws.manager import ConnectionManager
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.jobs.manager import BacktestJobManager

    app = create_app()
    state = StateManager(db_path=db)
    _ = state.conn
    jobs = BacktestJobManager(db_path=db)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws
    with (
        patch.object(BacktestJobManager, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as popen,
    ):
        popen.return_value.pid = 4321
        try:
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                yield client, state
        finally:
            await ws.stop()
            jobs.close()
            state.close()


def _run_child(db: Path, run_id: int) -> int:
    from vibe_quant.screening.__main__ import main

    return main(["run", "--run-id", str(run_id), "--db", str(db)])


def _row(db: Path, run_id: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from vibe_quant.db.state_manager import StateManager

    state = StateManager(db)
    try:
        rows = state.get_sweep_results(run_id)
        run = state.get_backtest_run(run_id)
        assert run is not None
        return rows, dict(run)
    finally:
        state.close()


@pytest.mark.parametrize("grid", [False, True], ids=["no-split", "grid-cell"])
async def test_pooled_children_replay_the_headline_exactly(tmp_path: Path, grid: bool) -> None:
    """C14 through both endpoints: replay and screening promote of champion index 1 write one
    sweep row equal to the full-range headline (Sharpe, summed trades, return, DD, PF) and the
    drift check is clean; a grid champion's child runs at its own (non-default) cell."""
    cell = CELL if grid else None
    chrom = _chrom()
    head, cost_edge = _headline(chrom, cell)
    assert head["full_range_trades"] == head["full_range_trades_sum"] > 0
    other = _entry(_chrom(), {"full_range_sharpe": 9.0, "full_range_trades": 1}, cell)
    db = tmp_path / "state.db"
    async with _client(db) as (client, state):
        did = _discovery_run(state, [other, _entry(chrom, head, cell)], grid)
        r = await client.post(f"/api/discovery/results/{did}/replay/1")
        assert r.status_code == 201, r.text
        replay_id = r.json()["replay_run_id"]
        assert "symbol_agg=pooled" in r.json()["metrics_note"]
        r = await client.post(f"/api/discovery/results/{did}/promote/1?mode=screening")
        assert r.status_code == 201, r.text
        promote_id = r.json()["run_id"]
        r = await client.post(f"/api/discovery/results/{did}/promote/1?mode=validation")
        assert r.status_code == 201, r.text
        validation_params = state.get_backtest_run(r.json()["run_id"])["parameters"]  # type: ignore[index]
    assert "symbol_agg" not in validation_params  # validation stays one shared account

    for child in (replay_id, promote_id):
        assert _run_child(db, child) == 0
        rows, run = _row(db, child)
        assert run["status"] == "completed"
        assert run["parameters"]["symbol_agg"] == "pooled"
        (row,) = rows
        assert row["sharpe_ratio"] == head["full_range_sharpe"]
        assert row["total_trades"] == head["full_range_trades_sum"]
        assert row["total_return"] == head["full_range_return_pct"]
        assert row["max_drawdown"] == head["full_range_max_dd"]
        assert row["profit_factor"] == head["full_range_pf"]
        # R26: the child's row records the pooled evaluation's cost edge, every column set.
        from vibe_quant.metrics import cost_edge_columns

        columns = {k: row[k] for k in COST_EDGE_KEYS}
        assert columns == cost_edge_columns(cost_edge)
        assert None not in columns.values()
    _rows, promoted = _row(db, promote_id)
    drift = promoted["parameters"].get("replay_drift")
    if drift is None:
        from vibe_quant.db.state_manager import StateManager

        st = StateManager(db)
        res = st.get_backtest_result(promote_id)
        st.close()
        drift = json.loads(res["notes"])["replay_drift"] if res else None
    assert drift is not None
    assert drift["flagged"] is False
    assert (drift["sharpe_ratio"], drift["trade_ratio"]) == (1.0, 1.0)


def test_pooled_child_evaluator_error_fails_without_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GATES-13: an evaluator error is a failed child; no numbers are persisted."""
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery.backtest_fn import NTBacktestFn
    from vibe_quant.dsl.translator import translate_dsl_config

    msg = "PooledSeriesError: pooled series misaligned (ETHUSDT first day 19725 != 19724)"
    monkeypatch.setattr(
        NTBacktestFn, "evaluate_dsl", lambda self, dsl: {**NTBacktestFn.error_result(_chrom(), msg)}
    )
    from vibe_quant.discovery.genome import chromosome_to_dsl

    db = tmp_path / "state.db"
    state = StateManager(db)
    dsl = translate_dsl_config(
        {**chromosome_to_dsl(_chrom()), "timeframe": "4h"}, strategy_name="g"
    )
    rid = state.create_backtest_run(
        strategy_id=None, run_mode="screening", symbols=SYMBOLS, timeframe="4h",
        start_date=START, end_date=END, parameters={"dsl_override": dsl, "symbol_agg": "pooled"},
    )  # fmt: skip
    state.close()
    assert _run_child(db, rid) == 1
    rows, run = _row(db, rid)
    assert rows == []
    assert run["status"] == "failed"
    assert msg in str(run["error_message"])


def test_unknown_symbol_agg_child_fails_closed(tmp_path: Path) -> None:
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery.genome import chromosome_to_dsl
    from vibe_quant.dsl.translator import translate_dsl_config

    db = tmp_path / "state.db"
    state = StateManager(db)
    dsl = translate_dsl_config(
        {**chromosome_to_dsl(_chrom()), "timeframe": "4h"}, strategy_name="g"
    )
    rid = state.create_backtest_run(
        strategy_id=None, run_mode="screening", symbols=SYMBOLS, timeframe="4h",
        start_date=START, end_date=END, parameters={"dsl_override": dsl, "symbol_agg": "worst"},
    )  # fmt: skip
    state.close()
    assert _run_child(db, rid) == 1
    rows, run = _row(db, rid)
    assert rows == []
    assert run["error_message"] == (
        "ValueError: run parameter symbol_agg='worst': a screening child runs only 'pooled'"
    )
