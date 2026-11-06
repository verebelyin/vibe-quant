"""Screening run summary exposes the screening venue's real starting balance.

Screening runs have no backtest_results row, so the summary is synthesized from
the best sweep row. The UI derives net PnL / fee drag / fees-per-year from
``starting_balance``; ``None`` left every screening ratio at n/a.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.db.state_manager import StateManager
from vibe_quant.validation.venue import create_venue_config_for_screening

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture()
async def api(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, StateManager]]:
    app = create_app()
    state = StateManager(db_path=tmp_path / "test.db")
    _ = state.conn
    app.state.state_manager = state
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, state
    state.close()


def _screening_run(
    state: StateManager,
    *,
    parameters: dict[str, object],
    sweep: dict[str, object],
) -> int:
    """A screening run (no ``backtest_results`` row) and its one sweep row."""
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters=parameters,
    )
    state.save_sweep_result(
        run_id,
        {"parameters": {"x": 1}, "sharpe_ratio": 1.0, **sweep},
    )
    return run_id


async def test_screening_summary_uses_the_recorded_sweep_balance(
    api: tuple[AsyncClient, StateManager],
) -> None:
    """The summary reports the balance recorded on the best sweep row.

    ``starting_balance`` 10000.0 -> 10000.0, even though the run parameters ask for
    5000 and the venue default is 1000.
    """
    client, state = api
    run_id = _screening_run(
        state,
        parameters={"initial_balance": 5000},
        sweep={"starting_balance": 10000.0, "total_return": 0.05, "total_trades": 10},
    )

    r = await client.get(f"/api/results/runs/{run_id}")

    assert r.status_code == 200
    body = r.json()
    assert body["starting_balance"] == 10000.0
    # net PnL the UI derives is scaled by the balance the run actually used
    assert body["total_return"] * body["starting_balance"] == pytest.approx(0.05 * 10000.0)


async def test_screening_summary_null_sweep_balance_reads_through_the_one_null_rule(
    api: tuple[AsyncClient, StateManager],
) -> None:
    """A NULL recorded balance is 1000, even when the params say 10000.

    Run 836 shape: the parameters ask for 10000 but the sweep row predates the
    column, so the summary reports the default the run actually used — and it does so
    through ``recorded_balance``, not a second NULL rule of its own.
    """
    client, state = api
    run_id = _screening_run(
        state,
        parameters={"initial_balance": 10000},
        sweep={"total_return": 0.05, "total_trades": 10},
    )

    r = await client.get(f"/api/results/runs/{run_id}")

    assert r.status_code == 200
    body = r.json()
    assert body["starting_balance"] == 1000.0
    assert body["starting_balance"] != 10000


async def test_screening_summary_rejects_a_corrupt_recorded_balance(
    api: tuple[AsyncClient, StateManager],
) -> None:
    """The recorded value goes through ``recorded_balance`` (which validates),
    so a corrupt balance is an error rather than a silent default."""
    client, state = api
    run_id = _screening_run(
        state,
        parameters={"initial_balance": 5000},
        sweep={"starting_balance": 1000.5, "total_return": 0.05},
    )

    r = await client.get(f"/api/results/runs/{run_id}")

    assert r.status_code == 500


async def test_screening_summary_carries_screening_venue_balance(
    api: tuple[AsyncClient, StateManager],
) -> None:
    client, state = api
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        # A requested initial_balance is never what the screening venue used.
        parameters={"initial_balance": 5000},
    )
    state.save_sweep_result(
        run_id,
        {
            "parameters": {"x": 1},
            "sharpe_ratio": 1.0,
            "total_return": 0.05,
            "total_trades": 10,
            "total_fees": 12.5,
        },
    )

    r = await client.get(f"/api/results/runs/{run_id}")

    assert r.status_code == 200
    body = r.json()
    venue_balance = float(create_venue_config_for_screening().starting_balance_usdt)
    assert body["starting_balance"] == venue_balance
    assert body["starting_balance"] != 5000
    # net PnL the UI derives from it is the screening venue's, not a guess
    assert body["total_return"] * body["starting_balance"] == pytest.approx(0.05 * venue_balance)


def _one_loss_trade(net_pnl: float) -> dict[str, object]:
    """A single closed trade; the chart endpoints skip open trades."""
    return {
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "leverage": 1,
        "entry_time": "2024-03-01T00:00:00",
        "exit_time": "2024-03-15T00:00:00",
        "entry_price": 50000.0,
        "exit_price": 49000.0,
        "quantity": 1.0,
        "net_pnl": net_pnl,
        "exit_reason": "signal",
    }


def _screening_run_with_trade(
    state: StateManager,
    *,
    sweep: dict[str, object],
    net_pnl: float = -100.0,
) -> int:
    """A screening run with the given best sweep row and one closed loss trade."""
    run_id = _screening_run(state, parameters={"initial_balance": 5000}, sweep=sweep)
    state.save_trades_batch(run_id, [_one_loss_trade(net_pnl)])
    return run_id


async def test_equity_curve_uses_the_recorded_sweep_balance(
    api: tuple[AsyncClient, StateManager],
) -> None:
    """The equity curve starts from the balance recorded on the best sweep row.

    The reviewer's probe: sweep 10000 with one trade net_pnl -100 gives 10000 then
    9900. Before the fix the curve started at the hard-coded 1000 (1000 then 900).
    """
    client, state = api
    run_id = _screening_run_with_trade(state, sweep={"starting_balance": 10000.0})

    r = await client.get(f"/api/results/runs/{run_id}/equity-curve")

    assert r.status_code == 200
    assert r.json() == [
        {"timestamp": "2024-03-01T00:00:00", "equity": 10000.0},
        {"timestamp": "2024-03-15T00:00:00", "equity": 9900.0},
    ]


async def test_drawdown_uses_the_recorded_sweep_balance(
    api: tuple[AsyncClient, StateManager],
) -> None:
    """The drawdown peak starts from the recorded sweep balance.

    Sweep 10000 with net_pnl -100 gives a 0.01 drawdown, not the 0.1 a 1000 base
    produces.
    """
    client, state = api
    run_id = _screening_run_with_trade(state, sweep={"starting_balance": 10000.0})

    r = await client.get(f"/api/results/runs/{run_id}/drawdown")

    assert r.status_code == 200
    assert r.json() == [{"timestamp": "2024-03-15T00:00:00", "drawdown": 0.01}]


async def test_monthly_returns_use_the_recorded_sweep_balance(
    api: tuple[AsyncClient, StateManager],
) -> None:
    """The monthly % is the PnL over the recorded sweep balance.

    Sweep 10000 with net_pnl -100 gives -1.0%, not the -10.0% a 1000 base gives.
    """
    client, state = api
    run_id = _screening_run_with_trade(state, sweep={"starting_balance": 10000.0})

    r = await client.get(f"/api/results/runs/{run_id}/monthly-returns")

    assert r.status_code == 200
    assert r.json() == [{"year": 2024, "month": 3, "return_pct": -1.0}]


async def test_chart_endpoints_read_a_null_sweep_balance_through_the_one_null_rule(
    api: tuple[AsyncClient, StateManager],
) -> None:
    """A NULL sweep balance is 1000 for every chart endpoint too.

    A row written before the column existed reads as the default through
    ``recorded_balance`` — never the requested ``initial_balance`` (5000 here).
    """
    client, state = api
    run_id = _screening_run_with_trade(state, sweep={"total_return": 0.05})

    equity = await client.get(f"/api/results/runs/{run_id}/equity-curve")
    drawdown = await client.get(f"/api/results/runs/{run_id}/drawdown")
    monthly = await client.get(f"/api/results/runs/{run_id}/monthly-returns")

    assert equity.status_code == 200
    assert equity.json() == [
        {"timestamp": "2024-03-01T00:00:00", "equity": 1000.0},
        {"timestamp": "2024-03-15T00:00:00", "equity": 900.0},
    ]
    assert drawdown.status_code == 200
    assert drawdown.json() == [{"timestamp": "2024-03-15T00:00:00", "drawdown": 0.1}]
    assert monthly.status_code == 200
    assert monthly.json() == [{"year": 2024, "month": 3, "return_pct": -10.0}]


async def test_chart_endpoints_prefer_a_result_row_balance_when_one_exists(
    api: tuple[AsyncClient, StateManager],
) -> None:
    """A validation run (with a result row) is unchanged.

    Its own ``backtest_results.starting_balance`` wins over the run's sweep rows,
    and there is no sweep row at all.
    """
    client, state = api
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )
    state.save_backtest_result(run_id, {"total_trades": 1, "starting_balance": 10000.0})
    state.save_trades_batch(run_id, [_one_loss_trade(-100.0)])

    equity = await client.get(f"/api/results/runs/{run_id}/equity-curve")
    drawdown = await client.get(f"/api/results/runs/{run_id}/drawdown")
    monthly = await client.get(f"/api/results/runs/{run_id}/monthly-returns")

    assert equity.status_code == 200
    assert equity.json() == [
        {"timestamp": "2024-03-01T00:00:00", "equity": 10000.0},
        {"timestamp": "2024-03-15T00:00:00", "equity": 9900.0},
    ]
    assert drawdown.status_code == 200
    assert drawdown.json() == [{"timestamp": "2024-03-15T00:00:00", "drawdown": 0.01}]
    assert monthly.status_code == 200
    assert monthly.json() == [{"year": 2024, "month": 3, "return_pct": -1.0}]
