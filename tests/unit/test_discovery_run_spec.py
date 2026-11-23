"""DiscoveryRunSpec outcome tests.

One test per thing a caller or reader depends on: the window plan the run executes
(and the API prechecks), the parameters an API launch stores, the notes a run
persists, the settings helpers that feed the engines (fees, leverage, sizing,
bankroll cells) and the launch wiring at the API edge. The old goldens and the
field-order / dest-reflection pins are gone; literals below are the contract.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import pytest

from vibe_quant.bankroll_grid import BankrollGridError
from vibe_quant.discovery.config_check import DiscoveryConfigError
from vibe_quant.discovery.run_spec import (
    DiscoveryRunSpec,
    build_parser,
)
from vibe_quant.fees import FeeSchedule
from vibe_quant.launch_settings import LaunchSettings
from vibe_quant.liquidation_settings import LiquidationConfigError, LiquidationSettings
from vibe_quant.sizing_settings import SizingConfigError, SizingSettings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from httpx import AsyncClient

    from vibe_quant.db.state_manager import StateManager

API_START = "2024-01-01"
API_END = "2025-01-01"


def _api_spec(**overrides: Any) -> DiscoveryRunSpec:
    """The spec an API request with these fields launches."""
    from vibe_quant.api.schemas.discovery import DiscoveryLaunchRequest

    body: dict[str, Any] = {"symbols": ["BTCUSDT"], "timeframes": ["4h"], **overrides}
    return DiscoveryRunSpec.from_request(
        DiscoveryLaunchRequest(**body), start_date=API_START, end_date=API_END
    )


# --- plan: the windows the run backtests (look-ahead sensitive) ------------


def test_plan_derives_literal_windows() -> None:
    """Split, eval and cross/WFA windows, literally, for one non-default config."""
    spec = DiscoveryRunSpec.from_argv([
        "--start-date", "2025-01-01",
        "--end-date", "2025-07-01",
        "--train-test-split", "0.5",
        "--eval-windows", "2",
        "--cross-window-months", "1,2",
        "--wfa-oos-step-days", "30",
    ])  # fmt: skip
    plan = spec.plan()
    assert (
        plan.train_start, plan.train_end, plan.holdout_start, plan.holdout_end
    ) == ("2025-01-01", "2025-04-01", "2025-04-01", "2025-07-01")
    assert plan.eval_windows == [("2025-01-01", "2025-02-15"), ("2025-02-15", "2025-04-01")]
    assert plan.cross_windows == [
        (1, "2025-02-01", "2025-04-01"),
        (2, "2025-03-01", "2025-04-01"),
    ]
    assert plan.wfa_windows == [
        ("2025-01-01", "2025-01-31"),
        ("2025-01-31", "2025-03-02"),
        ("2025-03-02", "2025-04-01"),
    ]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--cross-window-months", "0"], "cross_window_months must be >= 1, got [0]"),
        (
            ["--cross-window-months", "abc"],
            "cross_window_months: 'abc' is not an integer",
        ),
        (["--wfa-oos-step-days", "-1"], "wfa_oos_step_days must be >= 0, got -1"),
    ],
)
def test_plan_rejects_invalid_windows(argv: list[str], message: str) -> None:
    """A bad window config raises the exact message (the API 422s with it)."""
    with pytest.raises(DiscoveryConfigError) as exc:
        DiscoveryRunSpec.from_argv(argv).plan()
    assert str(exc.value) == message


# --- settings helpers (fees / leverage / sizing / grid are money) ----------


def test_resolved_bootstrap_min_sharpe_uses_the_timeframe_floor() -> None:
    """The floor table: 1m 0.5, 4h/1d 0.0, anything else 1.0; an explicit value wins."""
    for argv, expected in (
        ([], 0.0),  # 4h
        (["--timeframe", "1m"], 0.5),
        (["--timeframe", "1d"], 0.0),
        (["--timeframe", "1h"], 1.0),
        (["--timeframe", "1m", "--bootstrap-min-sharpe", "0.2"], 0.2),
        (["--timeframe", "1h", "--bootstrap-min-sharpe", "0.0"], 0.0),
    ):
        assert DiscoveryRunSpec.from_argv(argv).resolved_bootstrap_min_sharpe() == expected, argv


def test_list_and_count_helpers() -> None:
    """Two unrelated argv rows: the parsed lists/counts main reads, literally."""
    spec = DiscoveryRunSpec.from_argv([
        "--symbols", " BTCUSDT , ETHUSDT ",
        "--indicator-pool", " RSI , KAMA ",
        "--eval-windows", "0",
        "--max-workers", "-1",
        "--cross-window-months=-1,3",
    ])
    assert spec.symbol_list() == ["BTCUSDT", "ETHUSDT"]
    assert spec.indicator_pool_list() == ["RSI", "KAMA"]
    assert spec.eval_window_count() == 1
    assert spec.worker_count() is None
    assert spec.cross_window_month_list() == [-1, 3]

    other = DiscoveryRunSpec.from_argv([
        "--symbols", "",
        "--eval-windows", "3",
        "--max-workers", "0",
    ])
    assert other.symbol_list() == []
    assert other.indicator_pool_list() is None
    assert other.eval_window_count() == 3
    assert other.worker_count() == 0
    assert other.cross_window_month_list() == []


def test_fee_schedule_helper() -> None:
    """The fee schedule main builds, and the tier error main lets escape."""
    default = DiscoveryRunSpec.from_argv([]).fee_schedule()
    assert default.to_notes() == {"fee_tier": "VIP0", "bnb_discount": False}
    other = DiscoveryRunSpec.from_argv(["--fee-tier", "VIP0", "--bnb-discount"]).fee_schedule()
    assert other.to_notes() == {"fee_tier": "VIP0", "bnb_discount": True}

    with pytest.raises(ValueError) as exc:
        DiscoveryRunSpec.from_argv(["--fee-tier", "VIP99"]).fee_schedule()
    assert str(exc.value) == "unknown fee tier 'VIP99'; known tiers: ['VIP0']"


def test_liquidation_settings_helper() -> None:
    """Mode + resolved leverage, and the alias conflict main fails the run with."""
    off = DiscoveryRunSpec.from_argv(["--leverage", "5"]).liquidation_settings()
    assert (off.mode, off.leverage) == ("off", 5.0)
    cross = DiscoveryRunSpec.from_argv(
        ["--liquidation-mode", "cross", "--leverage", "20"]
    ).liquidation_settings()
    assert (cross.mode, cross.leverage) == ("cross", 20.0)
    alias = DiscoveryRunSpec.from_argv(["--liquidation-leverage", "7"]).liquidation_settings()
    assert (alias.mode, alias.leverage) == ("off", 7.0)

    with pytest.raises(LiquidationConfigError) as exc:
        DiscoveryRunSpec.from_argv(
            ["--leverage", "5", "--liquidation-leverage", "7"]
        ).liquidation_settings()
    assert str(exc.value) == "conflicting leverage 5.0 and liquidation_leverage 7.0"


def test_sizing_settings_helper() -> None:
    """Only given sizing keys reach the settings; every default is unchanged."""
    default = DiscoveryRunSpec.from_argv([]).sizing_settings()
    assert default.max_position_pct == 0.5
    assert default.risk_per_trade == 0.02
    assert default.max_margin_pct is None
    assert default.to_params() == {}

    other = DiscoveryRunSpec.from_argv([
        "--max-position-pct", "2",
        "--risk-per-trade", "0.01",
        "--max-margin-pct", "0.3",
    ]).sizing_settings()
    assert other.to_params() == {
        "max_position_pct": 2.0,
        "risk_per_trade": 0.01,
        "max_margin_pct": 0.3,
    }


def test_is_grid_and_bankroll_cells() -> None:
    """A partial grid is refused with today's text; a full one builds every cell."""
    assert DiscoveryRunSpec.from_argv([]).is_grid() is False

    with pytest.raises(BankrollGridError) as exc:
        DiscoveryRunSpec.from_argv(["--bankroll-modes", "isolated"]).is_grid()
    assert str(exc.value) == (
        "give all three of --bankroll-modes, --bankroll-leverages and "
        "--bankroll-max-margin-pcts together"
    )

    with pytest.raises(BankrollGridError) as exc:
        DiscoveryRunSpec.from_argv([
            "--bankroll-modes", "isolated",
            "--bankroll-leverages", "5",
            "--bankroll-max-margin-pcts", "0.5",
            "--leverage", "5",
        ]).is_grid()
    assert str(exc.value).endswith(
        "use either the bankroll grid or --liquidation-mode/--leverage/--max-margin-pct"
    )

    spec = DiscoveryRunSpec.from_argv([
        "--bankroll-modes", "isolated,cross",
        "--bankroll-leverages", "5,10",
        "--bankroll-max-margin-pcts", "0.5",
    ])
    assert spec.is_grid() is True
    cells = spec.bankroll_cells(spec.sizing_settings())
    assert [(c.mode, c.leverage, c.max_margin_pct) for c in cells] == [
        ("isolated", 5.0, 0.5),
        ("isolated", 10.0, 0.5),
        ("cross", 5.0, 0.5),
        ("cross", 10.0, 0.5),
    ]


def test_launch_settings_helper() -> None:
    """The spec builds the run's LaunchSettings; the entry order is the caller's."""
    spec = DiscoveryRunSpec.from_argv([
        "--fee-tier", "VIP0", "--bnb-discount", "--liquidation-mode", "cross",
        "--leverage", "20", "--max-position-pct", "2", "--risk-per-trade", "0.01",
    ])  # fmt: skip
    assert spec.launch_settings("market") == LaunchSettings(
        fee_schedule=FeeSchedule(tier="VIP0", bnb_discount=True),
        liquidation=LiquidationSettings(mode="cross", leverage=20.0),
        sizing=SizingSettings(max_position_pct=2.0, risk_per_trade=0.01),
        entry_order="market",
    )

    # An unrelated argv of defaults is the defaults LaunchSettings.
    assert DiscoveryRunSpec.from_argv(["--no-dsr"]).launch_settings("market") == LaunchSettings()

    # A cap the run's leverage cannot cover is refused with the sizing message.
    with pytest.raises(
        SizingConfigError,
        match=r"max_position_pct \(15\.0\) must not exceed leverage \(10\.0\)",
    ):
        DiscoveryRunSpec.from_argv(["--max-position-pct", "15"]).launch_settings("market")


# --- what the API stores / the CLI persists -------------------------------


def test_to_params_records_the_launch_literally() -> None:
    """The stored parameters (promote/replay inherit them): literal, both inputs."""
    default = _api_spec()
    assert default.to_params() == {
        "population": 50,
        "generations": 20,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "elite_count": 2,
        "tournament_size": 3,
        "convergence_generations": 10,
        "max_workers": 4,
        "eval_windows": 3,
        "symbol_agg": "portfolio",
        "train_test_split": 0.8,
        "gate_pool": "all",
    }
    # The API default never puts symbol_agg / immigrant_fraction on the argv.
    argv = default.to_argv()
    assert "--symbol-agg" not in argv
    assert "--immigrant-fraction" not in argv

    nested = _api_spec(
        cross_window_months=[1, 2],
        wfa_oos_step_days=30,
        num_seeds=3,
        liquidation_mode="isolated",
        leverage=20,
        max_position_pct=2,
        risk_per_trade=0.01,
        max_margin_pct=0.3,
    )
    assert nested.to_params() == {
        "population": 50,
        "generations": 20,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "elite_count": 2,
        "tournament_size": 3,
        "convergence_generations": 10,
        "max_workers": 4,
        "eval_windows": 3,
        "symbol_agg": "portfolio",
        "train_test_split": 0.8,
        "gate_pool": "all",
        "cross_window_months": [1, 2],
        "cross_window_min_sharpe": 0.5,
        "num_seeds": 3,
        "wfa_oos_step_days": 30,
        "wfa_min_consistency": 0.75,
        "liquidation_mode": "isolated",
        "leverage": 20.0,
        "max_position_pct": 2.0,
        "risk_per_trade": 0.01,
        "max_margin_pct": 0.3,
    }


def test_to_argv_round_trips_two_unrelated_launches() -> None:
    """The argv the router spawns carries every value the spec holds.

    A flag dropped here is not an error: the CLI falls back to its default and the run
    silently differs from the stored params.
    """
    loaded = _api_spec(
        cross_window_months=[1, 2],
        wfa_oos_step_days=30,
        num_seeds=3,
        liquidation_mode="isolated",
        leverage=20,
        max_position_pct=2,
        risk_per_trade=0.01,
        max_margin_pct=0.3,
        direction="short",
        indicator_pool=["RSI", "KAMA"],
        seed=7,
        symbol_agg="worst",
        bootstrap_min_sharpe=0.9,
    )
    for spec in (_api_spec(), loaded):
        assert DiscoveryRunSpec.from_argv(spec.to_argv()) == spec


def test_eval_windows_zero_vs_explicit() -> None:
    """0 is not forwarded (CLI default 3 applies); 3 is forwarded and stored."""
    zero = _api_spec(eval_windows=0)
    assert "--eval-windows" not in zero.to_argv()
    assert "eval_windows" not in zero.to_params()
    assert zero.eval_window_count() == 3
    assert (
        zero.to_config(zero.plan(), symbols=["BTCUSDT"], prior_trials_fn=None).eval_windows == 3
    )

    explicit = _api_spec(eval_windows=3)
    argv = explicit.to_argv()
    assert argv[argv.index("--eval-windows") + 1] == "3"
    assert explicit.to_params()["eval_windows"] == 3
    assert (
        explicit.to_config(
            explicit.plan(), symbols=["BTCUSDT"], prior_trials_fn=None
        ).eval_windows
        == 3
    )


def test_to_config_carries_the_plan_and_resolved_symbols() -> None:
    """The config the pipeline runs: split windows, resolved symbols, floor, gates."""
    spec = DiscoveryRunSpec.from_argv([
        "--symbols", "",  # falls back to the run row's resolved symbols
        "--timeframe", "1m",
        "--start-date", "2025-01-01",
        "--end-date", "2025-02-01",
        "--train-test-split", "0.8",
    ])  # fmt: skip
    config = spec.to_config(spec.plan(), symbols=["ETHUSDT"], prior_trials_fn=None)
    assert (config.start_date, config.end_date) == ("2025-01-01", "2025-01-25")
    assert (config.holdout_start_date, config.holdout_end_date) == ("2025-01-25", "2025-02-01")
    assert (config.symbols, config.timeframe, config.eval_windows) == (["ETHUSDT"], "1m", 3)
    assert config.bootstrap_min_sharpe == 0.5  # 1m floor, not the 1.0 default
    assert config.require_dsr is True


def test_cli_holdout_min_trades_reaches_the_holdout_gate() -> None:
    """``--holdout-min-trades 7`` is the floor the gate applies: 6 holdout trades are rejected."""
    from vibe_quant.discovery.gate_chain import HoldoutResult, holdout_min_trades, holdout_reasons

    spec = DiscoveryRunSpec.from_argv([
        "--symbols", "BTCUSDT",
        "--timeframe", "1d",
        "--start-date", "2024-01-01",
        "--end-date", "2025-01-01",
        "--train-test-split", "0.8",
        "--holdout-min-trades", "7",
    ])  # fmt: skip
    config = spec.to_config(spec.plan(), symbols=["BTCUSDT"], prior_trials_fn=None)
    assert config.holdout_min_trades == 7
    assert holdout_min_trades(config) == 7
    six_trades = HoldoutResult(
        sharpe_ratio=0.5, max_drawdown=0.01, profit_factor=1.5, total_trades=6, total_return=0.02
    )
    assert holdout_reasons(six_trades, holdout_min_trades(config), config.holdout_min_sharpe) == [
        "Holdout: 6 trades < 7"
    ]


def test_to_notes_agrees_with_to_params() -> None:
    """Every spec-derived note agrees with the stored params.

    ``to_notes`` is the one notes writer: a field the spec owns is recorded from
    the same value ``to_params`` stores, not re-derived. ``symbols`` is the
    resolved list the run backtested (``--symbols ""`` falls back to the run row).
    """
    from vibe_quant.discovery.pipeline import DiscoveryResult

    result = DiscoveryResult(
        generations=[], top_strategies=[], total_candidates_evaluated=160, converged=True,
        convergence_generation=1,
    )  # fmt: skip

    cases = [
        {},
        {"cross_window_months": [1, 2], "wfa_oos_step_days": 30, "num_seeds": 3},
        {
            "liquidation_mode": "isolated",
            "leverage": 20,
            "max_position_pct": 2.0,
            "risk_per_trade": 0.01,
        },
    ]
    for overrides in cases:
        spec = _api_spec(**overrides)
        notes = spec.to_notes(
            result,
            symbols=["BTCUSDT"],
            use_mock=True,
            seed=7,
            prior_run_ids=[],
            prior_legacy={},
            cells=None,
        )
        params = spec.to_params()
        n = max(1, cast("int", params.get("eval_windows", 3)))
        assert notes["eval_windows"] == (n if n > 1 else None), overrides
        assert notes["symbol_agg"] == params["symbol_agg"], overrides
        assert notes["train_test_split"] == (
            params["train_test_split"] if params["train_test_split"] > 0 else None
        ), overrides
        assert notes["direction"] == params.get("direction"), overrides
        assert notes["cross_window_months"] == params.get("cross_window_months"), overrides
        assert notes["cross_window_min_sharpe"] == (
            params["cross_window_min_sharpe"] if params.get("cross_window_months") else None
        ), overrides
        assert notes["wfa_oos_step_days"] == params.get("wfa_oos_step_days"), overrides
        assert notes["wfa_min_consistency"] == (
            params["wfa_min_consistency"] if params.get("wfa_oos_step_days") else None
        ), overrides
        assert notes["num_seeds"] == params.get("num_seeds"), overrides
        assert notes.get("indicator_pool") == params.get("indicator_pool"), overrides
        sizing_keys = {"max_position_pct", "risk_per_trade", "max_margin_pct"} & set(params)
        assert notes.get("sizing") == ({k: params[k] for k in sizing_keys} or None), overrides
        assert notes.get("leverage") == (
            params["leverage"]
            if params.get("liquidation_mode", "off") == "off" and "leverage" in params
            else None
        ), overrides
        assert notes["buy_hold_gate"]["symbols"] == ["BTCUSDT"], overrides  # type: ignore[index]
        assert notes["mock"] is True and notes["evaluated"] == 160, overrides


# --- the API launch edge --------------------------------------------------


@asynccontextmanager
async def _api_client(db_path: Path) -> AsyncIterator[tuple[AsyncClient, StateManager]]:
    """A real app over a temp DB, with both launch prechecks stubbed."""
    from unittest.mock import patch

    from httpx import ASGITransport, AsyncClient

    from vibe_quant.api import data_precheck, mark_precheck
    from vibe_quant.api.app import create_app
    from vibe_quant.api.ws.manager import ConnectionManager
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.jobs.manager import BacktestJobManager

    app = create_app()
    state = StateManager(db_path=db_path)
    _ = state.conn
    jobs = BacktestJobManager(db_path=db_path)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    with (
        patch.object(BacktestJobManager, "is_process_alive", return_value=True),
        patch("subprocess.Popen") as popen,
        patch.object(data_precheck, "precheck_discovery_data", lambda *a, **k: None),
        patch.object(mark_precheck, "precheck_discovery", lambda *a, **k: None),
    ):
        popen.return_value.pid = 4321
        try:
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                yield client, state
        finally:
            await ws.stop()
            jobs.close()
            state.close()


def _run_count(state: StateManager) -> int:
    return int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])


async def test_empty_pool_422_no_row(tmp_path: Path) -> None:
    """from_request never raises for ``[]``; the router refuses with today's 422."""
    spec = _api_spec(indicator_pool=[])
    assert spec.indicator_pool == ""
    assert "--indicator-pool" in spec.to_argv()
    assert spec.to_params()["indicator_pool"] == []

    async with _api_client(tmp_path / "pool.db") as (client, state):
        r = await client.post(
            "/api/discovery/launch",
            json={
                "symbols": ["BTCUSDT"],
                "timeframes": ["4h"],
                "start_date": API_START,
                "end_date": API_END,
                "indicator_pool": [],
            },
        )
        assert r.status_code == 422, r.text
        assert r.json()["detail"] == (
            "indicator_pool must not be empty; omit it (null) to use the default pool"
        )
        assert _run_count(state) == 0


async def test_default_dates(tmp_path: Path) -> None:
    """Omitted start/end become today-365d / today on the run row."""
    async with _api_client(tmp_path / "dates.db") as (client, state):
        now = datetime.now()
        r = await client.post(
            "/api/discovery/launch", json={"symbols": ["BTCUSDT"], "timeframes": ["4h"]}
        )
        assert r.status_code == 201, r.text
        run = state.get_backtest_run(r.json()["run_id"])
        assert run is not None
        assert run["start_date"] == (now - timedelta(days=365)).strftime("%Y-%m-%d")
        assert run["end_date"] == now.strftime("%Y-%m-%d")


def test_parser_is_the_cli_contract() -> None:
    """A bad flag keeps argparse's SystemExit(2); a good row parses its dest."""
    with pytest.raises(SystemExit) as exc:
        DiscoveryRunSpec.from_argv(["--nope"])
    assert exc.value.code == 2
    assert build_parser().parse_args(["--run-id", "42"]).run_id == 42


# --- power preflight (bead vibe-quant-clmmc.1) ------------------------------


def test_power_preflight_prior_lookup_failure_is_unknown() -> None:
    """A raising prior-trials fn -> prior_n "unknown", bar (a) at N = 1."""

    def _boom(start: str, end: str) -> int | None:
        raise RuntimeError("db gone")

    spec = DiscoveryRunSpec.from_argv(
        ["--run-id", "1", "--start-date", "2023-06-01", "--end-date", "2026-06-01"]
    )
    pre = spec.power_preflight(spec.plan(), _boom)
    assert pre["prior_n"] == "unknown"
    assert pre["bar_a"]["n_trials"] == 1  # type: ignore[index]
    assert pre["bar_b"]["n_trials"] == 1000  # type: ignore[index]
    seen: list[tuple[str, str]] = []

    def _seven(start: str, end: str) -> int | None:
        seen.append((start, end))
        return 7

    pre = spec.power_preflight(spec.plan(), _seven)
    assert seen == [("2023-06-01", "2025-10-24")]  # the gate's train window
    assert pre["prior_n"] == 7
    assert pre["train_days"] == 876
    assert pre["bar_a"]["n_trials"] == 8  # type: ignore[index]
    assert pre["bar_b"]["n_trials"] == 1007  # type: ignore[index]


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (
            ["--population-size", "4", "--max-generations", "1"],
            {"budget": 4, "prior_n": 0, "train_days": 47, "a": 1, "b": 4},
        ),
        (
            ["--population-size", "6", "--max-generations", "3", "--no-dsr"],
            {"budget": 18, "prior_n": 0, "train_days": 47, "a": "DSR off", "b": "DSR off"},
        ),
    ],
    ids=["dsr", "no_dsr"],
)
def test_run_notes_carry_power_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    extra: list[str],
    expected: dict[str, object],
) -> None:
    """A run logs the Power line before it runs and stores the same dict in notes."""
    import json
    import logging

    from tests.unit.test_run_discovery import drive_run_discovery
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import run as discovery_run
    from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult

    db_path = tmp_path / "state.db"
    state = StateManager(db_path)
    run_id = state.create_backtest_run(
        strategy_id=None, run_mode="discovery", symbols=["BTCUSDT"], timeframe="4h",
        start_date="2025-01-01", end_date="2025-03-01", parameters={},
    )  # fmt: skip
    state.close()
    lines_before_run: list[str] = []

    def fake_run(self: object) -> DiscoveryResult:
        lines_before_run.extend(r.getMessage() for r in caplog.records)
        return DiscoveryResult(
            generations=[], top_strategies=[], total_candidates_evaluated=4, converged=True,
            convergence_generation=1,
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    argv = [
        "--run-id", str(run_id), "--elite-count", "1", "--symbols", "BTCUSDT",
        "--timeframe", "4h", "--start-date", "2025-01-01", "--end-date", "2025-03-01",
        "--eval-windows", "1", "--db", str(db_path), "--mock", "--max-workers", "-1", *extra,
    ]  # fmt: skip
    with caplog.at_level(logging.INFO, logger="vibe_quant.discovery.run"):
        outcome = drive_run_discovery(db_path, run_id, argv, discovery_run.MockBacktests())
    assert outcome.exit_code == 0

    state = StateManager(db_path)
    result = state.get_backtest_result(run_id)
    state.close()
    assert result is not None
    pre = json.loads(result["notes"])["power_preflight"]
    assert pre["budget"] == expected["budget"]
    assert pre["prior_n"] == expected["prior_n"]
    assert pre["train_days"] == expected["train_days"]
    assert pre["holdout_days"] == 12
    for key, want in (("bar_a", expected["a"]), ("bar_b", expected["b"])):
        assert (pre[key] if isinstance(want, str) else pre[key]["n_trials"]) == want
    assert any(line == pre["line"] for line in lines_before_run), lines_before_run


@pytest.mark.parametrize(
    ("end_date", "reason"),
    [
        ("2025-01-02", "num_observations must be >= 2, got 1"),
        ("2025-01-03", "no annual Sharpe up to 10000 passes DSR at N=1 over 2 days"),
    ],
    ids=["one-day", "two-day-unbounded"],
)
def test_unavailable_power_preflight_never_stops_a_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    end_date: str,
    reason: str,
) -> None:
    """A preflight that cannot be computed is stored as unavailable: the run completes."""
    import json
    import logging
    import sqlite3

    from tests.unit.test_run_discovery import drive_run_discovery
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import run as discovery_run
    from vibe_quant.discovery.pipeline import DiscoveryPipeline, DiscoveryResult

    db_path = tmp_path / "state.db"
    state = StateManager(db_path)
    run_id = state.create_backtest_run(
        strategy_id=None, run_mode="discovery", symbols=["BTCUSDT"], timeframe="4h",
        start_date="2025-01-01", end_date=end_date, parameters={},
    )  # fmt: skip
    state.close()

    def fake_run(self: object) -> DiscoveryResult:
        return DiscoveryResult(
            generations=[], top_strategies=[], total_candidates_evaluated=4, converged=True,
            convergence_generation=1,
        )  # fmt: skip

    monkeypatch.setattr(DiscoveryPipeline, "run", fake_run)
    argv = [
        "--run-id", str(run_id), "--seed", "7", "--population-size", "4",
        "--max-generations", "1", "--elite-count", "1", "--symbols", "BTCUSDT",
        "--timeframe", "4h", "--start-date", "2025-01-01", "--end-date", end_date,
        "--eval-windows", "1", "--train-test-split", "0", "--db", str(db_path), "--mock",
        "--max-workers", "-1",
    ]  # fmt: skip
    with caplog.at_level(logging.INFO, logger="vibe_quant.discovery"):
        outcome = drive_run_discovery(db_path, run_id, argv, discovery_run.MockBacktests())
    assert outcome.exit_code == 0

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT r.status, json_valid(n.notes), n.notes FROM backtest_runs r "
            "JOIN backtest_results n ON n.run_id = r.id WHERE r.id = ?",
            (run_id,),
        ).fetchone()
    assert row[:2] == ("completed", 1)

    def _no_constant(name: str) -> object:
        raise AssertionError(f"non-JSON constant {name} in notes")

    pre = json.loads(row[2], parse_constant=_no_constant)["power_preflight"]
    assert pre == {"status": "unavailable", "reason": reason}
    assert any(
        r.levelno >= logging.WARNING and f"Power preflight unavailable: {reason}" in r.getMessage()
        for r in caplog.records
    ), caplog.text


async def test_launch_accepts_a_ui_shaped_body(tmp_path: Path) -> None:
    """The stricter shared validation (bool, NaN string, int64) keeps a normal form body a 201."""
    body = {
        "population": 50, "generations": 20, "mutation_rate": 0.1, "crossover_rate": 0.8,
        "elite_count": 2, "tournament_size": 3, "convergence_generations": 10,
        "symbols": ["BTCUSDT", "ETHUSDT"], "timeframes": ["4h"], "indicator_pool": None,
        "start_date": API_START, "end_date": API_END, "direction": "long",
        "eval_windows": 3, "train_test_split": 0.8, "num_seeds": 1, "wfa_oos_step_days": 30,
        "wfa_min_consistency": 0.75, "cross_window_months": [1, 2],
        "cross_window_min_sharpe": 0.5, "immigrant_fraction": 0.15, "entropy_threshold": 0.4,
        "crowding_enabled": True, "seed": 7, "no_bootstrap_ci": True,
        "bootstrap_min_sharpe": 0.0, "bnb_discount": True, "liquidation_mode": "isolated",
        "leverage": 5, "max_position_pct": 0.25, "risk_per_trade": 0.01,
    }  # fmt: skip
    async with _api_client(tmp_path / "ui.db") as (client, state):
        r = await client.post("/api/discovery/launch", json=body)
        assert r.status_code == 201, r.text
        run = state.get_backtest_run(r.json()["run_id"])
        assert run is not None
        assert run["symbols"] == ["BTCUSDT", "ETHUSDT"]


# --- --gate-pool (vibe-quant-clmmc.2) --------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "param", "argv"),
    [
        ({}, "all", []),
        ({"gate_pool": "top_k"}, "top_k", ["--gate-pool", "top_k"]),
        ({"gate_pool": "all"}, "all", []),
    ],
    ids=["absent", "top_k", "all"],
)
def test_gate_pool_api_params_and_argv(body: dict[str, Any], param: str, argv: list[str]) -> None:
    """``all`` is the default (user Q16, 2026-11-23): the API stores the pool always and forwards
    ``--gate-pool`` only for ``top_k``; the CLI reading that argv runs the same pool."""
    spec = _api_spec(**body)

    assert spec.to_params()["gate_pool"] == param
    assert [a for a in spec.to_argv() if "gate-pool" in a or a in ("all", "top_k")] == argv
    again = DiscoveryRunSpec.from_argv(spec.to_argv())
    assert again.to_config(again.plan(), symbols=["BTCUSDT"], prior_trials_fn=None).gate_pool == param


def test_gate_pool_api_refuses_unknown_value() -> None:
    from pydantic import ValidationError

    from vibe_quant.api.schemas.discovery import DiscoveryLaunchRequest

    with pytest.raises(ValidationError, match="gate_pool"):
        DiscoveryLaunchRequest(symbols=["BTCUSDT"], timeframes=["4h"], gate_pool="x")  # type: ignore[arg-type]


async def test_gate_pool_bad_value_is_422_with_no_row(tmp_path: Path) -> None:
    async with _api_client(tmp_path / "gp.db") as (client, state):
        r = await client.post(
            "/api/discovery/launch",
            json={"symbols": ["BTCUSDT"], "timeframes": ["4h"], "gate_pool": "x"},
        )
        assert r.status_code == 422
        assert "gate_pool" in r.text
        assert _run_count(state) == 0


@pytest.mark.parametrize(
    ("argv", "note"),
    [([], "all"), (["--gate-pool", "all"], "all"), (["--gate-pool", "top_k"], "top_k")],
    ids=["omitted", "all", "top_k"],
)
def test_gate_pool_note_always_written(argv: list[str], note: str) -> None:
    """R-gpf-1: both values are written, so a reader never needs to know the launch default."""
    from vibe_quant.discovery.pipeline import DiscoveryResult

    spec = DiscoveryRunSpec.from_argv(argv)
    result = DiscoveryResult(
        generations=[], top_strategies=[], total_candidates_evaluated=1, converged=False,
        convergence_generation=None,
    )  # fmt: skip
    notes = spec.to_notes(
        result, symbols=["BTCUSDT"], use_mock=True, seed=1, prior_run_ids=[], prior_legacy={},
        cells=None,
    )  # fmt: skip
    assert notes["gate_pool"] == note


# --- --null-gate (vibe-quant-clmmc.14) ------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "param"), [({}, None), ({"null_gate": False}, None), ({"null_gate": True}, True)],
    ids=["absent", "false", "true"],
)  # fmt: skip
def test_null_gate_api_params_and_argv(body: dict[str, Any], param: bool | None) -> None:
    """Stored and forwarded only when on, so a default launch keeps params and argv unchanged;
    the forwarded argv parses back to the flag the run reads (``args.null_gate``)."""
    spec = _api_spec(**body)

    assert spec.to_params().get("null_gate") == param
    assert ("null_gate" in spec.to_params()) is (param is not None)
    assert spec.to_argv().count("--null-gate") == (1 if param else 0)
    args = build_parser().parse_args(["--run-id", "0", *spec.to_argv()])
    assert args.null_gate is bool(param)
    assert DiscoveryRunSpec.from_argv(spec.to_argv()).to_argv() == spec.to_argv()


def test_null_gate_api_refuses_a_non_bool() -> None:
    from pydantic import ValidationError

    from vibe_quant.api.schemas.discovery import DiscoveryLaunchRequest

    with pytest.raises(ValidationError, match="null_gate"):
        DiscoveryLaunchRequest(symbols=["BTCUSDT"], timeframes=["4h"], null_gate="yes")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("extra", "detail"),
    [
        ({"symbols": ["BTCUSDT", "ETHUSDT"]},
         "--null-gate needs one symbol or --symbol-agg worst (a shared account couples "
         "symbol sizing)"),
        ({"liquidation_mode": "isolated"},
         "--null-gate needs --liquidation-mode off (liquidated trades have no re-placeable exit)"),
        ({"bankroll_grid": {"modes": ["isolated"], "leverages": [5.0], "max_margin_pcts": [0.5]}},
         "--null-gate cannot run a bankroll grid (its cells run liquidation on)"),
        ({"evaluator": "fast"},
         "--null-gate needs real NT backtests (no --mock, no --evaluator fast)"),
        ({"timeframes": ["2m"]},
         "--null-gate needs a timeframe with a bar tape (1s|5s|1m|5m|15m|1h|4h|1d), got 2m"),
        ({"symbols": ["BTCUSDT", "ETHUSDT"], "symbol_agg": "pooled"},
         "--null-gate cannot judge --symbol-agg pooled (no pooled timing null yet)"),
        ({"symbols": ["BTCUSDT", "ETHUSDT"], "liquidation_mode": "cross"},
         "--null-gate needs one symbol or --symbol-agg worst (a shared account couples "
         "symbol sizing); --null-gate needs --liquidation-mode off (liquidated trades have no "
         "re-placeable exit)"),
    ],
    ids=["portfolio-2-symbols", "liquidation-on", "bankroll-grid", "fast", "no-tape-timeframe",
         "pooled", "two-problems"],
)  # fmt: skip
async def test_null_gate_launch_refusal_is_422_with_no_row(
    tmp_path: Path, extra: dict[str, Any], detail: str
) -> None:
    """GATES-6 / B12: the CLI's own refusal texts, before any run row or job exists; the
    preflight of the same body refuses with the same texts."""
    body = {"symbols": ["BTCUSDT"], "timeframes": ["4h"], "start_date": API_START,
            "end_date": API_END, "null_gate": True, **extra}  # fmt: skip
    async with _api_client(tmp_path / "ng.db") as (client, state):
        r = await client.post("/api/discovery/launch", json=body)
        assert r.status_code == 422, r.text
        assert r.json()["detail"] == detail
        assert _run_count(state) == 0
        pre = await client.post("/api/discovery/preflight", json=body)  # same texts
        assert pre.status_code == 422, pre.text
        assert pre.json()["detail"] == detail


@pytest.mark.parametrize(
    "extra", [{}, {"symbols": ["BTCUSDT", "ETHUSDT"], "symbol_agg": "worst"}], ids=["one", "worst"]
)
async def test_null_gate_launch_forwards_the_flag(tmp_path: Path, extra: dict[str, Any]) -> None:
    """An acceptable ``--null-gate`` launch is a 201 whose subprocess gets the flag."""
    from unittest.mock import patch

    body = {"symbols": ["BTCUSDT"], "timeframes": ["4h"], "start_date": API_START,
            "end_date": API_END, "null_gate": True, **extra}  # fmt: skip
    async with _api_client(tmp_path / "ng.db") as (client, state):
        with patch("subprocess.Popen") as popen:
            popen.return_value.pid = 4321
            r = await client.post("/api/discovery/launch", json=body)
            command = popen.call_args.args[0]
        assert r.status_code == 201, r.text
        assert command.count("--null-gate") == 1
        run = state.get_backtest_run(r.json()["run_id"])
        assert run is not None
        assert run["parameters"]["null_gate"] is True


# --- --symbol-agg pooled (vibe-quant-clmmc.5 C-T2) ------------------------------------------


@pytest.mark.parametrize(
    ("extra", "detail"),
    [
        ({"symbols": ["BTCUSDT"]}, "symbol_agg=pooled needs at least 2 symbols"),
        ({"evaluator": "fast"}, "symbol_agg=pooled needs --evaluator nt"),
        ({"symbols": ["BTCUSDT"], "evaluator": "fast"},
         "symbol_agg=pooled needs at least 2 symbols; symbol_agg=pooled needs --evaluator nt"),
    ],
    ids=["one-symbol", "fast", "both"],
)  # fmt: skip
async def test_pooled_launch_refusal_is_422_with_no_row(
    tmp_path: Path, extra: dict[str, Any], detail: str
) -> None:
    """C2: the API refuses with the CLI's texts before any run row; the preflight too."""
    body = {"symbols": ["BTCUSDT", "ETHUSDT"], "timeframes": ["4h"], "start_date": API_START,
            "end_date": API_END, "symbol_agg": "pooled", **extra}  # fmt: skip
    async with _api_client(tmp_path / "pooled.db") as (client, state):
        r = await client.post("/api/discovery/launch", json=body)
        assert r.status_code == 422, r.text
        assert r.json()["detail"] == detail
        assert _run_count(state) == 0
        pre = await client.post("/api/discovery/preflight", json=body)
        assert pre.status_code == 422, pre.text
        assert pre.json()["detail"] == detail


async def test_pooled_launch_is_accepted_and_forwarded(tmp_path: Path) -> None:
    """CLI/API accept pooled after C-T2: stored param, forwarded flag, argv round trip."""
    from unittest.mock import patch

    body = {"symbols": ["BTCUSDT", "ETHUSDT"], "timeframes": ["4h"], "start_date": API_START,
            "end_date": API_END, "symbol_agg": "pooled"}  # fmt: skip
    spec = _api_spec(symbols=["BTCUSDT", "ETHUSDT"], symbol_agg="pooled")
    assert spec.to_params()["symbol_agg"] == "pooled"
    assert DiscoveryRunSpec.from_argv(spec.to_argv()).symbol_agg == "pooled"
    async with _api_client(tmp_path / "pooled.db") as (client, state):
        with patch("subprocess.Popen") as popen:
            popen.return_value.pid = 4321
            r = await client.post("/api/discovery/launch", json=body)
            command = popen.call_args.args[0]
        assert r.status_code == 201, r.text
        assert command[command.index("--symbol-agg") + 1] == "pooled"
        run = state.get_backtest_run(r.json()["run_id"])
        assert run is not None
        assert run["parameters"]["symbol_agg"] == "pooled"


@pytest.mark.parametrize(
    ("extra", "problem"),
    [
        (["--symbols", "BTCUSDT"], "symbol_agg=pooled needs at least 2 symbols"),
        (["--evaluator", "fast"], "symbol_agg=pooled needs --evaluator nt"),
        (["--mock"], "symbol_agg=pooled cannot run with --mock (the mock has no per-coin accounts)"),
    ],
    ids=["one-symbol", "fast", "mock"],
)  # fmt: skip
def test_pooled_cli_refusal_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   extra: list[str], problem: str) -> None:  # fmt: skip
    """C2: the CLI run refuses with exit 2 before any pipeline exists."""
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import run
    from vibe_quant.jobs.manager import BacktestJobManager

    monkeypatch.setattr(run, "DiscoveryPipeline", lambda *a, **kw: pytest.fail("pipeline built"))
    db = tmp_path / "state.db"
    state = StateManager(db)
    run_id = state.create_backtest_run(None, "discovery", ["BTCUSDT"], "4h", API_START, API_END, {})
    argv = ["--seed", "7", "--start-date", API_START, "--end-date", API_END, "--timeframe", "4h",
            "--symbols", "BTCUSDT,ETHUSDT", "--db", str(db), "--run-id", str(run_id),
            "--symbol-agg", "pooled", *extra]  # fmt: skip
    args = build_parser().parse_args(argv)
    jobs = BacktestJobManager(db)
    try:
        outcome = run.run_discovery(
            args, run.MockBacktests() if args.mock else run.NTBacktests(),
            run.StateRecordStore(state, jobs), run_id=run_id,
            started_at=0.0,
        )  # fmt: skip
    finally:
        state.close()
        jobs.close()
    assert outcome.exit_code == 2
    assert outcome.message == f"Discovery failed: {problem}"


@pytest.mark.parametrize("agg", ["portfolio", "pooled"])
def test_pooled_coverage_refusal_is_todays_message(tmp_path: Path, agg: str) -> None:
    """C13: a pooled run over a coin's pre-listing range (HYPEUSDT before 2025-05-31) fails
    closed with the coverage message a portfolio run gets (real catalog; skipped without)."""
    from vibe_quant.data.catalog import DEFAULT_CATALOG_PATH
    from vibe_quant.db.state_manager import StateManager
    from vibe_quant.discovery import run
    from vibe_quant.jobs.manager import BacktestJobManager

    if not (DEFAULT_CATALOG_PATH / "data" / "bar").exists():
        pytest.skip("no data catalog")
    db = tmp_path / "state.db"
    state = StateManager(db)
    run_id = state.create_backtest_run(
        None, "discovery", ["BTCUSDT", "HYPEUSDT"], "4h", "2025-01-01", "2025-04-01", {}
    )
    args = build_parser().parse_args(
        ["--run-id", str(run_id), "--db", str(db), "--symbols", "BTCUSDT,HYPEUSDT",
         "--timeframe", "4h", "--start-date", "2025-01-01", "--end-date", "2025-04-01",
         "--population-size", "4", "--max-generations", "1", "--seed", "1", "--symbol-agg", agg]
    )  # fmt: skip
    jobs = BacktestJobManager(db)
    try:
        outcome = run.run_discovery(
            args, run.NTBacktests(), run.StateRecordStore(state, jobs), run_id=run_id,
            started_at=0.0,
        )  # fmt: skip
        assert state.get_backtest_run(run_id)["status"] == "failed"  # type: ignore[index]
        assert state.get_backtest_result(run_id) is None
    finally:
        state.close()
        jobs.close()
    assert outcome.exit_code == 1
    assert outcome.message.startswith(
        "Discovery failed: No catalog bars for HYPEUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL in "
        "window 2025-01-01..2025-01-25 (catalog: "
    ), outcome.message
