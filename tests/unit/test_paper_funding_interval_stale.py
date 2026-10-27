"""Paper refuses to launch a funding-avoidance strategy on stale archived intervals.

The filter reads each coin's settlement grid from the archive's latest
``raw_funding_intervals`` row; a coin that changed its interval after that row
would be filtered on the wrong hours. /start and /restore refuse (409) when a
symbol's latest row is older than ``FUNDING_INTERVAL_MAX_AGE_DAYS`` or missing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.routers import paper_trading
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.data.archive import RawDataArchive
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl import aux_data
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

DAY_MS = 86_400_000
NOW_MS = 1_800_000_000_000  # fixed clock: every age below is exact
MAX_AGE_MS = paper_trading.FUNDING_INTERVAL_MAX_AGE_DAYS * DAY_MS

DSL: dict[str, Any] = {
    "name": "funding_filter",
    "timeframe": "1h",
    "indicators": {"rsi_14": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi_14 < 30"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
    "time_filters": {"avoid_around_funding": {"enabled": True}},
}
DSL_OFF = {**DSL, "name": "funding_filter_off", "time_filters": {}}
DSL_EXPLICIT_OFF = {
    **DSL,
    "name": "funding_filter_explicit_off",
    "time_filters": {"avoid_around_funding": {"enabled": False}},
}


class Ctx:
    def __init__(
        self,
        ac: AsyncClient,
        state: StateManager,
        archive_db: Path,
        cfg_dir: Path,
    ) -> None:
        self.ac = ac
        self.state = state
        self.archive_db = archive_db
        self.cfg_dir = cfg_dir
        self.on = state.create_strategy("funding_filter", DSL)
        self.off = state.create_strategy("funding_filter_off", DSL_OFF)
        self.explicit_off = state.create_strategy("funding_filter_explicit_off", DSL_EXPLICIT_OFF)

    def archive(self, symbol: str, age_ms: int) -> None:
        """One interval row for ``symbol`` whose age at NOW_MS is ``age_ms``."""
        archive = RawDataArchive(self.archive_db)
        try:
            archive.insert_funding_intervals(symbol, [(NOW_MS - age_ms, 8)], "binance_vision")
        finally:
            archive.close()

    def runs(self) -> int:
        return len(self.state.list_backtest_runs())

    def start(self, strategy_id: int, symbols: list[str]) -> Any:
        return self.ac.post(
            "/api/paper/start", json={"strategy_id": strategy_id, "symbols": symbols}
        )

    def saved_config(self, trader_id: str, strategy_id: int, symbols: list[str]) -> None:
        self.cfg_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg_dir / "paper_1.json").write_text(
            json.dumps(
                {
                    "trader_id": trader_id,
                    "strategy_id": strategy_id,
                    "symbols": symbols,
                    "binance": {"testnet": True},
                }
            )
        )


@pytest.fixture()
async def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Ctx]:
    db = tmp_path / "api.db"
    monkeypatch.setattr("vibe_quant.db.connection.DEFAULT_DB_PATH", db)
    cfg_dir = tmp_path / "paper_configs"
    monkeypatch.setattr("vibe_quant.api.routers.paper_trading.PAPER_CONFIG_DIR", cfg_dir)
    monkeypatch.setattr("vibe_quant.api.routers.paper_trading._now_ms", lambda: NOW_MS)
    archive_db = tmp_path / "archive" / "raw_data.db"
    archive_db.parent.mkdir()
    monkeypatch.setattr(aux_data, "_archive_path", archive_db)
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "tk")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "ts")
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    app = create_app()
    state = StateManager(db_path=db)
    _ = state.conn
    jobs = BacktestJobManager(db_path=db)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws

    def fake_start_job(
        run_id: int, job_type: str, command: list[str], log_file: str | None = None
    ) -> int:
        jobs.conn.execute(
            """INSERT OR REPLACE INTO background_jobs
               (run_id, pid, job_type, status, started_at, heartbeat_at)
               VALUES (?, 4242, 'paper', 'running', datetime('now'), datetime('now'))""",
            (run_id,),
        )
        jobs.conn.commit()
        return 4242

    monkeypatch.setattr(jobs, "start_job", fake_start_job)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield Ctx(ac, state, archive_db, cfg_dir)
    await ws.stop()
    jobs.close()
    state.close()


def test_the_limit_is_45_days() -> None:
    """Binance Vision publishes whole months, so the newest row is 1-31+ days old."""
    assert paper_trading.FUNDING_INTERVAL_MAX_AGE_DAYS == 45


def test_the_clock_is_epoch_milliseconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.time", lambda: 1_800_000_000.25)

    assert paper_trading._now_ms() == 1_800_000_000_250


async def test_start_refuses_a_stale_interval_and_names_symbol_age_and_refresh(ctx: Ctx) -> None:
    ctx.archive("BTCUSDT", 46 * DAY_MS)

    r = await ctx.start(ctx.on, ["BTCUSDT"])

    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert "BTCUSDT" in detail
    assert "46 days" in detail
    assert "funding-intervals --symbols BTCUSDT" in detail
    assert ctx.runs() == 0


async def test_start_names_only_the_stale_symbol_with_its_own_age(ctx: Ctx) -> None:
    ctx.archive("BTCUSDT", 2 * DAY_MS)
    ctx.archive("ETHUSDT", 100 * DAY_MS)

    r = await ctx.start(ctx.on, ["BTCUSDT", "ETHUSDT"])

    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert "ETHUSDT" in detail
    assert "100 days" in detail
    assert "BTCUSDT" not in detail.replace("funding-intervals --symbols ETHUSDT", "")
    assert ctx.runs() == 0


async def test_start_age_equal_to_the_limit_is_allowed(ctx: Ctx) -> None:
    ctx.archive("BTCUSDT", MAX_AGE_MS)

    r = await ctx.start(ctx.on, ["BTCUSDT"])

    assert r.status_code == 201, r.text


async def test_start_age_one_ms_over_the_limit_is_refused(ctx: Ctx) -> None:
    ctx.archive("BTCUSDT", MAX_AGE_MS + 1)

    r = await ctx.start(ctx.on, ["BTCUSDT"])

    assert r.status_code == 409, r.text
    assert ctx.runs() == 0


async def test_start_age_one_ms_under_the_limit_is_allowed(ctx: Ctx) -> None:
    ctx.archive("BTCUSDT", MAX_AGE_MS - 1)

    r = await ctx.start(ctx.on, ["BTCUSDT"])

    assert r.status_code == 201, r.text


async def test_the_newest_row_decides_not_an_old_one(ctx: Ctx) -> None:
    ctx.archive("BTCUSDT", 200 * DAY_MS)
    ctx.archive("BTCUSDT", 3 * DAY_MS)

    r = await ctx.start(ctx.on, ["BTCUSDT"])

    assert r.status_code == 201, r.text


async def test_start_refuses_a_symbol_without_any_interval_row(ctx: Ctx) -> None:
    ctx.archive("BTCUSDT", DAY_MS)

    r = await ctx.start(ctx.on, ["BTCUSDT", "SOLUSDT"])

    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert "SOLUSDT" in detail
    assert "no funding interval" in detail
    assert ctx.runs() == 0


async def test_start_refuses_when_the_archive_file_is_missing(ctx: Ctx) -> None:
    assert not ctx.archive_db.exists()

    r = await ctx.start(ctx.on, ["BTCUSDT"])

    assert r.status_code == 409, r.text
    assert "no funding interval" in r.json()["detail"]
    assert not ctx.archive_db.exists()  # the check never creates the archive


async def test_start_refuses_when_the_archive_has_no_interval_table(ctx: Ctx) -> None:
    import sqlite3

    conn = sqlite3.connect(ctx.archive_db)
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()

    r = await ctx.start(ctx.on, ["BTCUSDT"])

    assert r.status_code == 409, r.text
    assert ctx.runs() == 0


async def test_a_strategy_without_the_filter_ignores_stale_intervals(ctx: Ctx) -> None:
    ctx.archive("BTCUSDT", 400 * DAY_MS)

    r = await ctx.start(ctx.off, ["BTCUSDT"])

    assert r.status_code == 201, r.text


async def test_a_filter_configured_but_disabled_ignores_stale_intervals(ctx: Ctx) -> None:
    ctx.archive("BTCUSDT", 400 * DAY_MS)

    r = await ctx.start(ctx.explicit_off, ["BTCUSDT"])

    assert r.status_code == 201, r.text


async def test_start_from_a_validation_run_is_checked_too(ctx: Ctx) -> None:
    ctx.archive("ETHUSDT", 60 * DAY_MS)
    vrun = ctx.state.create_backtest_run(
        strategy_id=ctx.on,
        run_mode="validation",
        symbols=["ETHUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )
    ctx.state.update_backtest_run_status(vrun, "completed")
    before = ctx.runs()

    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": vrun})

    assert r.status_code == 409, r.text
    assert "ETHUSDT" in r.json()["detail"]
    assert "60 days" in r.json()["detail"]
    assert ctx.runs() == before


async def test_restore_refuses_a_stale_interval_and_creates_no_run(ctx: Ctx) -> None:
    ctx.archive("SOLUSDT", 90 * DAY_MS)
    ctx.saved_config("PAPER-001", ctx.on, ["SOLUSDT"])

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 409, r.text
    assert "SOLUSDT" in r.json()["detail"]
    assert "90 days" in r.json()["detail"]
    assert ctx.runs() == 0


async def test_restore_with_fresh_intervals_runs(ctx: Ctx) -> None:
    ctx.archive("SOLUSDT", MAX_AGE_MS)
    ctx.saved_config("PAPER-001", ctx.on, ["SOLUSDT"])

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 201, r.text


async def test_the_refusal_is_logged_without_exception_text(
    ctx: Ctx, caplog: pytest.LogCaptureFixture
) -> None:
    """An unreadable archive logs the exception class only (its text may carry a path)."""
    ctx.archive_db.write_text("LEAKMARKER not a database")

    with caplog.at_level("WARNING", logger="vibe_quant.api.routers.paper_trading"):
        r = await ctx.start(ctx.on, ["BTCUSDT"])

    assert r.status_code == 409, r.text
    assert "refus" in caplog.text.lower()
    assert "LEAKMARKER" not in caplog.text
    assert "LEAKMARKER" not in r.text


def _strategy_with_enabled(ctx: Ctx, name: str, enabled: object) -> int:
    dsl = {**DSL, "name": name, "time_filters": {"avoid_around_funding": {"enabled": enabled}}}
    return ctx.state.create_strategy(name, dsl)


@pytest.mark.parametrize("enabled", ["false", "False", "0", "no", "off", 0])
async def test_a_filter_the_dsl_reads_as_off_ignores_stale_intervals(
    ctx: Ctx, enabled: object
) -> None:
    """The DSL parses these as off (the node compiles no filter), so the launch is not refused."""
    sid = _strategy_with_enabled(ctx, "off_form", enabled)

    r = await ctx.start(sid, ["BTCUSDT"])  # no archive at all

    assert r.status_code == 201, r.text


@pytest.mark.parametrize("enabled", ["true", "True", "1", "yes", "on", 1, "maybe", [], {}])
async def test_a_filter_the_dsl_reads_as_on_or_cannot_read_is_checked(
    ctx: Ctx, enabled: object
) -> None:
    """On forms and unreadable values fail closed: no archived interval refuses."""
    sid = _strategy_with_enabled(ctx, "on_form", enabled)

    r = await ctx.start(sid, ["BTCUSDT"])

    assert r.status_code == 409, r.text


def _validation_run(ctx: Ctx, symbols: list[str]) -> int:
    vrun = ctx.state.create_backtest_run(
        strategy_id=ctx.on,
        run_mode="validation",
        symbols=symbols,
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={},
    )
    ctx.state.update_backtest_run_status(vrun, "completed")
    return vrun


def _vrun_config(ctx: Ctx, vrun: object, **extra: object) -> None:
    ctx.cfg_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "trader_id": "PAPER-001",
        "strategy_id": ctx.on,
        "validation_run_id": vrun,
        "binance": {"testnet": True},
        **extra,
    }
    (ctx.cfg_dir / "paper_1.json").write_text(json.dumps(config))


@pytest.mark.parametrize("symbols", ["absent", "empty"])
async def test_restore_without_symbols_checks_the_validation_runs_symbols(
    ctx: Ctx, symbols: str
) -> None:
    """The node trades the run's symbols when the config names none: check those."""
    ctx.archive("BTCUSDT", 200 * DAY_MS)
    vrun = _validation_run(ctx, ["BTCUSDT"])
    _vrun_config(ctx, vrun, **({"symbols": []} if symbols == "empty" else {}))

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 409, r.text
    assert "BTCUSDT" in r.json()["detail"]
    assert "200 days" in r.json()["detail"]
    assert ctx.runs() == 1  # the validation run only


async def test_restore_without_symbols_launches_on_the_validation_runs_fresh_symbols(
    ctx: Ctx,
) -> None:
    ctx.archive("BTCUSDT", MAX_AGE_MS)
    vrun = _validation_run(ctx, ["BTCUSDT"])
    _vrun_config(ctx, vrun)

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 201, r.text


@pytest.mark.parametrize("vrun", [None, 99999, "7"])
async def test_restore_with_no_resolvable_symbols_is_refused_400(ctx: Ctx, vrun: object) -> None:
    """No symbols in the config and none from a validation run: nothing to vouch for."""
    _vrun_config(ctx, vrun)

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-001"})

    assert r.status_code == 400, r.text
    assert "symbols" in r.json()["detail"]
    assert ctx.runs() == 0


async def test_start_from_a_run_with_no_symbols_is_refused_400(ctx: Ctx) -> None:
    vrun = _validation_run(ctx, [])

    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": vrun})

    assert r.status_code == 400, r.text
