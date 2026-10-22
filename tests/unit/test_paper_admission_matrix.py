"""rabmo.10a: every paper start refusal in one matrix (characterization).

Before rabmo.10b moves the admission decision into one pure function, this file
pins the three public entry points that can refuse a paper start, byte for byte:

* ``POST /api/paper/start`` -- HTTP status + ``detail``;
* ``run_with_config`` / ``main`` -- exit code + the exact ``stderr`` lines;
* ``PaperTradingNode._initialize`` -- the ``ConfigurationError`` text.

Each row also pins what the refusal leaves behind: no run row / no launch for
the API, and for the CLI the run row state (untouched for the two consent rows,
persisted as the terminal failure for the no-checkpoint row).

Three order properties rabmo.10b must preserve:

* P-R1: the post-Telegram re-check refuses a start whose kill switch flipped
  during the probe (423, no run row).
* r3 F10: a failed latest read keeps the earlier CLEAN read, so startup refuses
  only when the switch is really unreadable or engaged.
* r3 F11: the CLI's consent checks stay sequential and the checkpoint lookup is
  short-circuited for every start that does not need it.

Production behaviour is unchanged by this file; the mutants that prove the rows
have teeth are listed in the handoff.
"""

from __future__ import annotations

import errno
import json
import os
from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

import vibe_quant.api.routers.paper_trading as paper_router
from tests.unit.test_audit_paper_node import _config
from vibe_quant.alerts.telegram import TelegramUnavailableError
from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.paper import lineage
from vibe_quant.paper.account_lock import acquire_account_lock, acquire_trader_lock
from vibe_quant.paper.admission import (
    Admitted,
    Refusal,
    RefusalCode,
    StartFacts,
    StartIntent,
    check,
)
from vibe_quant.paper.cli import main, run_with_config
from vibe_quant.paper.config import (
    BinanceTestnetConfig,
    ConfigurationError,
    PaperTradingConfig,
    account_id,
)
from vibe_quant.paper.kill_switch_monitor import KillSwitchSnapshot
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from vibe_quant.paper.persistence import StatePersistence

DSL = {
    "name": "test_strategy",
    "timeframe": "1h",
    "indicators": {"rsi_14": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi_14 < 30"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}

#: Makes the missing-row re-seed INSERT fail with a real SQL error (tdjop R3b).
_UNRESEEDABLE_TRIGGER = (
    "CREATE TRIGGER ks_block_reseed BEFORE INSERT ON system_state "
    "BEGIN SELECT missing_sql_function(); END"
)

_KILL_MISSING_503 = "System kill switch state unavailable (system_state row missing); refusing"
_LOCK_UNOPENABLE_503 = "cannot probe the paper account lock (PermissionError); check the lock dir"
_NO_PASSWD_HOME_503 = "cannot resolve the OS user's home dir for the paper lock"

_API_LIVE_400 = "testnet=false trades real funds: set confirm_live=true to proceed"
_API_FRESH_400 = (
    "start_fresh with testnet=false skips the LIVE risk state another trader id left "
    "on this account (peak, daily loss, losing streak, ERROR halt); this trader id's "
    "own saved limits and halt are kept: set confirm_fresh_live=true to proceed"
)
_API_NO_CHECKPOINT_400 = (
    "LIVE start has no saved risk state to carry (no checkpoint on this account and "
    "no earlier live session of this trader on another key): drawdown, daily-loss and "
    "losing-streak limits start from zero: set confirm_fresh_live=true to proceed"
)

_CLI_LIVE_TEXT = (
    "Refusing to trade LIVE: config has binance.testnet=false; "
    "re-run with --live to confirm real-money trading"
)
_CLI_FRESH_TEXT = (
    "Refusing LIVE start_fresh: it skips the risk state another trader id left on this "
    "account (peak, daily loss, losing streak, ERROR halt), so those limits may count from "
    "current equity; this trader id's own saved limits and halt are kept; re-run with "
    "--confirm-fresh-live to confirm"
)
_CLI_NO_CHECKPOINT_TEXT = (
    "Refusing LIVE start: no saved risk state to carry (no checkpoint on this "
    "account and no earlier live session of this trader on another key): "
    "drawdown, daily-loss and losing-streak limits would start from zero; "
    "re-run with --confirm-fresh-live to confirm"
)

_NODE_KILL_ENGAGED = "System kill switch engaged (ops); refusing to start"
_TELEGRAM_503 = "Telegram alerts are configured but not working (test); refusing to start"


# ------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _binance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both venues' credentials, so every config validates without the real env."""
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "tk")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "ts")
    monkeypatch.setenv("BINANCE_API_KEY", "lk")
    monkeypatch.setenv("BINANCE_API_SECRET", "ls")


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp state DB plus the API's tmp config dir (never the real home)."""
    db = tmp_path / "api.db"
    monkeypatch.setattr("vibe_quant.db.connection.DEFAULT_DB_PATH", db)
    monkeypatch.setattr(
        "vibe_quant.api.routers.paper_trading.PAPER_CONFIG_DIR", tmp_path / "paper_configs"
    )
    return db


class Ctx:
    """The API rig: real router over an in-process state DB and job manager."""

    def __init__(
        self,
        app: Any,
        ac: AsyncClient,
        state: StateManager,
        jobs: BacktestJobManager,
        tmp: Path,
    ) -> None:
        self.app = app
        self.ac = ac
        self.state = state
        self.jobs = jobs
        self.tmp = tmp
        self.launched: list[list[str]] = []
        self.strategy_id = state.create_strategy("test_strategy", DSL)
        self.validation_run = state.create_backtest_run(
            strategy_id=self.strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={"rsi_14_period": 21},
        )
        state.update_backtest_run_status(self.validation_run, "completed")

    def activate(self, run_id: int) -> None:
        self.jobs.conn.execute(
            """INSERT OR REPLACE INTO background_jobs
               (run_id, pid, job_type, status, started_at, heartbeat_at)
               VALUES (?, 4242, 'paper', 'running', datetime('now'), datetime('now'))""",
            (run_id,),
        )
        self.jobs.conn.commit()


@pytest.fixture()
async def ctx(tmp_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Ctx]:
    app = create_app()
    state = StateManager(db_path=tmp_db)
    _ = state.conn
    jobs = BacktestJobManager(db_path=tmp_db)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        c = Ctx(app, ac, state, jobs, tmp_path)

        def fake_start_job(
            run_id: int, job_type: str, command: list[str], log_file: str | None = None
        ) -> int:
            c.launched.append(command)
            c.activate(run_id)
            return 4242

        monkeypatch.setattr(jobs, "start_job", fake_start_job)
        yield c
    await ws.stop()
    jobs.close()
    state.close()


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    """A strategy plus a completed validation run, so a node can be built."""
    db = tmp_path / "state.db"
    sm = StateManager(db)
    sid = sm.create_strategy("test_strategy", DSL)
    rid = sm.create_backtest_run(
        strategy_id=sid,
        run_mode="validation",
        symbols=["BTCUSDT"],
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        parameters={"rsi_14_period": 21, "take_profit_risk_reward": 3.0, "leverage": 5},
    )
    sm.update_backtest_run_status(rid, "completed")
    sm.close()
    return {"db": db, "sid": sid, "rid": rid, "tmp": tmp_path}


class _FirstReadMonitor:
    """The kill-switch monitor as the node sees it: one snapshot, no thread."""

    def __init__(self, snap: KillSwitchSnapshot) -> None:
        self._snap = snap
        self.starts = 0
        self.stops = 0

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1

    def snapshot(self) -> KillSwitchSnapshot:
        return self._snap

    async def wait_first_read(self, timeout_s: float) -> KillSwitchSnapshot:
        return self._snap


# ----------------------------------------------------------------- API helpers


def _api_account(api_key: str = "tk", *, testnet: bool = True) -> str:
    return account_id(BinanceTestnetConfig(api_key=api_key, api_secret="ts", testnet=testnet))


def _paper_run_count(state: StateManager) -> int:
    row = state.conn.execute(
        "SELECT COUNT(*) FROM backtest_runs WHERE run_mode = 'paper'"
    ).fetchone()
    assert row is not None
    return int(row[0])


def _assert_api_refused(
    ctx: Ctx,
    response: Any,
    *,
    status: int,
    detail: str,
    runs_before: int = 0,
    launches_before: int = 0,
) -> None:
    """The refusal's full contract: status, detail, no new run row, no new launch."""
    assert response.status_code == status, response.text
    assert response.json()["detail"] == detail
    assert _paper_run_count(ctx.state) == runs_before
    assert len(ctx.launched) == launches_before


# ----------------------------------------------------------- API refusal rows


async def test_api_kill_engaged_423(ctx: Ctx) -> None:
    """Kill engaged -> 423 with the engaged detail, no run row, no launch."""
    ctx.state.set_kill_switch("ops")

    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})

    _assert_api_refused(ctx, r, status=423, detail="System kill switch engaged: ops")


async def test_api_kill_row_missing_and_unrecreatable_503(ctx: Ctx) -> None:
    """A missing row that cannot be re-created is a 503, never "not killed"."""
    _ = ctx.state.conn
    ctx.state.conn.execute("DELETE FROM system_state")
    ctx.state.conn.execute(_UNRESEEDABLE_TRIGGER)
    ctx.state.conn.commit()

    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})

    _assert_api_refused(ctx, r, status=503, detail=_KILL_MISSING_503)


async def test_api_session_active_409(ctx: Ctx) -> None:
    """A second start while the first session runs -> 409 naming its run id."""
    first = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert first.status_code == 201, first.text
    run_id = first.json()["run_id"]

    second = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})

    _assert_api_refused(
        ctx,
        second,
        status=409,
        detail=f"Paper session already running (run_id={run_id}); stop it first",
        runs_before=1,
        launches_before=1,
    )


async def test_api_account_busy_409(ctx: Ctx) -> None:
    """Another process holding the account lock -> 409 naming the holder body."""
    account = _api_account()
    held = acquire_account_lock(account, "PAPER-OTHER")
    try:
        r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    finally:
        held.release()

    _assert_api_refused(
        ctx,
        r,
        status=409,
        detail=(
            f"Paper session already holds account {account} "
            f"({os.getpid()} PAPER-OTHER); stop it first"
        ),
    )


async def test_api_live_without_confirmation_400(ctx: Ctx) -> None:
    """LIVE without confirm_live -> 400 before credentials or any probe."""
    r = await ctx.ac.post(
        "/api/paper/start",
        json={"validation_run_id": ctx.validation_run, "testnet": False},
    )

    _assert_api_refused(ctx, r, status=400, detail=_API_LIVE_400)


async def test_api_live_start_fresh_without_confirm_fresh_live_400(ctx: Ctx) -> None:
    """LIVE start_fresh needs its own consent, even with confirm_live."""
    r = await ctx.ac.post(
        "/api/paper/start",
        json={
            "validation_run_id": ctx.validation_run,
            "testnet": False,
            "confirm_live": True,
            "start_fresh": True,
        },
    )

    _assert_api_refused(ctx, r, status=400, detail=_API_FRESH_400)


async def test_api_live_without_a_checkpoint_400(ctx: Ctx) -> None:
    """LIVE with nothing to carry -> 400, before the Telegram probe and the run row."""
    r = await ctx.ac.post(
        "/api/paper/start",
        json={
            "validation_run_id": ctx.validation_run,
            "testnet": False,
            "confirm_live": True,
        },
    )

    _assert_api_refused(ctx, r, status=400, detail=_API_NO_CHECKPOINT_400)


async def test_api_lock_probe_unopenable_503(ctx: Ctx, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unopenable lock file is a 503 naming the class, no run row, no leak."""
    marker = "EACCES-TEXT-MARKER-14"
    real_open = os.open
    # ``account_lock_holder`` returns early when the lock file does not exist, so
    # the unopenable file has to exist (an unheld file is enough: the probe opens it).
    lock_file = ctx.tmp / "paper_locks" / f"{_api_account().replace(':', '_')}.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text("")

    def _denied(path: object, flags: int, *args: object) -> int:
        if str(path).endswith(".lock"):
            raise PermissionError(errno.EACCES, marker)
        return real_open(path, flags, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", _denied)

    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})

    _assert_api_refused(ctx, r, status=503, detail=_LOCK_UNOPENABLE_503)
    assert marker not in r.text


@pytest.mark.parametrize(
    "getpwuid",
    [
        pytest.param(
            lambda _uid: (_ for _ in ()).throw(KeyError("no such uid")), id="no-passwd-entry"
        ),
        pytest.param(lambda _uid: type("P", (), {"pw_dir": ""})(), id="empty-pw-dir"),
    ],
)
async def test_api_no_passwd_home_503(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch, getpwuid: Any
) -> None:
    """No passwd home -> no lock dir -> 503 with the reason, not a bare 500."""
    monkeypatch.setattr("vibe_quant.paper.account_lock.PAPER_LOCK_DIR", None)
    monkeypatch.setattr("vibe_quant.paper.account_lock.pwd.getpwuid", getpwuid)

    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})

    _assert_api_refused(ctx, r, status=503, detail=_NO_PASSWD_HOME_503)


async def test_api_telegram_unavailable_503(ctx: Ctx, monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured-but-broken Telegram -> 503 with ``str(exc)`` as the detail."""

    async def _refuse(*_args: Any, **_kwargs: Any) -> None:
        raise TelegramUnavailableError(_TELEGRAM_503)

    monkeypatch.setattr(paper_router, "require_working_telegram", _refuse)

    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})

    _assert_api_refused(ctx, r, status=503, detail=_TELEGRAM_503)


async def test_pr1_recheck_refuses_a_kill_flipped_during_the_telegram_await(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P-R1: the probe awaits; a kill that lands in that window still refuses 423.

    The re-check runs before ``create_backtest_run``, so the loser leaves no run
    row and no job -- the same contract T15b must keep when it re-orders calls.
    """

    async def _flip(*_args: Any, **_kwargs: Any) -> None:
        ctx.state.set_kill_switch("flipped during probe")
        return None

    monkeypatch.setattr(paper_router, "require_working_telegram", _flip)

    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})

    _assert_api_refused(
        ctx, r, status=423, detail="System kill switch engaged: flipped during probe"
    )


# ------------------------------------------------------------ CLI helpers


def _write_config(
    path: Path,
    *,
    testnet: bool,
    start_fresh: bool = False,
    db_path: Path | None = None,
    trader_id: str | None = None,
) -> None:
    data: dict[str, object] = {
        "trader_id": trader_id or ("PAPER-001" if testnet else "LIVE-001"),
        "binance": {"testnet": testnet},
        "symbols": ["BTCUSDT"],
        "strategy_id": 1,
        "start_fresh": start_fresh,
    }
    if db_path is not None:
        data["db_path"] = str(db_path)
    path.write_text(json.dumps(data))


def _patch_run(
    monkeypatch: pytest.MonkeyPatch,
    started: list[PaperTradingConfig],
    error: Exception | None = None,
) -> None:
    async def _run(config: PaperTradingConfig) -> None:
        started.append(config)
        if error is not None:
            raise error

    monkeypatch.setattr("vibe_quant.paper.cli.run_paper_trading", _run)


def _paper_run_row(db_path: Path) -> int:
    """A paper run + running job row, as the API leaves them after launching the CLI."""
    state = StateManager(db_path=db_path)
    try:
        strategy_id = state.create_strategy("s", DSL)
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="paper",
            symbols=["BTCUSDT"],
            timeframe="",
            start_date="",
            end_date="",
            parameters={},
        )
        state.update_backtest_run_status(run_id, "running", pid=4242)
        state.conn.execute(
            """INSERT INTO background_jobs
               (run_id, pid, job_type, status, started_at, heartbeat_at)
               VALUES (?, 4242, 'paper', 'running', datetime('now'), datetime('now'))""",
            (run_id,),
        )
        state.conn.commit()
        return run_id
    finally:
        state.close()


def _rows(db_path: Path, run_id: int) -> tuple[tuple[str, str | None], tuple[str, str | None]]:
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        run = conn.execute(
            "SELECT status, error_message FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()
        job = conn.execute(
            "SELECT status, error_message FROM background_jobs WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    return (run[0], run[1]), (job[0], job[1])


def _stderr(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().err.splitlines()


# ----------------------------------------------------------- CLI refusal rows


async def test_cli_live_without_confirmation_leaves_the_run_row_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--live`` missing: exit 1, the exact line, and today's row rule (no write)."""
    db_path = tmp_path / "state.db"
    run_id = _paper_run_row(db_path)
    config_path = tmp_path / "live.json"
    _write_config(config_path, testnet=False, db_path=db_path)
    started: list[PaperTradingConfig] = []
    _patch_run(monkeypatch, started)

    assert await run_with_config(config_path, run_id=run_id) == 1

    assert started == []
    assert _stderr(capsys) == [_CLI_LIVE_TEXT]
    assert _rows(db_path, run_id) == (("running", None), ("running", None))


async def test_cli_live_start_fresh_without_confirm_fresh_live_leaves_the_run_row_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The T3 start_fresh text at the CLI; only the no-checkpoint row persists."""
    db_path = tmp_path / "state.db"
    run_id = _paper_run_row(db_path)
    config_path = tmp_path / "fresh.json"
    _write_config(config_path, testnet=False, start_fresh=True, db_path=db_path)
    started: list[PaperTradingConfig] = []
    _patch_run(monkeypatch, started)

    assert await run_with_config(config_path, run_id=run_id, allow_live=True) == 1

    assert started == []
    assert _stderr(capsys) == [_CLI_FRESH_TEXT]
    assert _rows(db_path, run_id) == (("running", None), ("running", None))


async def test_cli_live_no_checkpoint_fails_the_run_and_job_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The no-checkpoint refusal is the run's AND job's terminal failure."""
    db_path = tmp_path / "state.db"
    run_id = _paper_run_row(db_path)
    config_path = tmp_path / "live.json"
    _write_config(config_path, testnet=False, db_path=db_path)
    started: list[PaperTradingConfig] = []
    _patch_run(monkeypatch, started)

    assert await run_with_config(config_path, run_id=run_id, allow_live=True) == 1

    assert started == []
    assert _stderr(capsys) == [_CLI_NO_CHECKPOINT_TEXT]
    assert _rows(db_path, run_id) == (
        ("failed", _CLI_NO_CHECKPOINT_TEXT),
        ("failed", _CLI_NO_CHECKPOINT_TEXT),
    )
    assert not (tmp_path / "paper_locks").exists(), "a refused start must hold nothing"


async def test_cli_account_busy_exits_one_with_the_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AccountBusyError keeps its curated text through the CLI's allowlisted printer."""
    config_path = tmp_path / "testnet.json"
    _write_config(config_path, testnet=True)
    started: list[PaperTradingConfig] = []
    _patch_run(monkeypatch, started)
    account = _api_account("tk")
    held = acquire_account_lock(account, "PAPER-OTHER")
    try:
        assert await run_with_config(config_path) == 1
    finally:
        held.release()

    assert started == []
    assert _stderr(capsys) == [
        "Paper trading error: AccountBusyError: Paper session already running on account "
        f"{account} (pid {os.getpid()}, trader PAPER-OTHER); stop it first"
    ]


async def test_cli_trader_busy_exits_one_with_the_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """TraderBusyError: one trader id, one session, on any key and either venue."""
    config_path = tmp_path / "testnet.json"
    _write_config(config_path, testnet=True)
    started: list[PaperTradingConfig] = []
    _patch_run(monkeypatch, started)
    account = _api_account("tk")
    held = acquire_trader_lock("PAPER-001", account)
    try:
        assert await run_with_config(config_path) == 1
    finally:
        held.release()

    assert started == []
    assert _stderr(capsys) == [
        "Paper trading error: TraderBusyError: Paper session already running for trader "
        f"PAPER-001 (pid {os.getpid()}, account {account}); stop it first"
    ]


def test_cli_renders_the_node_configuration_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI child prints the node's refusal text verbatim after ``main`` parsing."""
    config_path = tmp_path / "testnet.json"
    _write_config(config_path, testnet=True)
    started: list[PaperTradingConfig] = []
    _patch_run(monkeypatch, started, ConfigurationError(_NODE_KILL_ENGAGED))
    monkeypatch.setattr("sys.argv", ["paper", "start", "--config", str(config_path)])

    assert main() == 1

    # The node refusal happens inside run_paper_trading, so the config reached it.
    assert [c.trader_id for c in started] == ["PAPER-001"]
    assert _stderr(capsys) == [f"Paper trading error: ConfigurationError: {_NODE_KILL_ENGAGED}"]


# ---------------------------------------------------------- node startup rows


async def test_startup_kill_engaged_refuses(seeded: dict[str, Any]) -> None:
    """Kill engaged at startup: the node's exact ConfigurationError text."""
    node = PaperTradingNode(_config(seeded))
    node._kill_monitor = _FirstReadMonitor(  # type: ignore[assignment]
        KillSwitchSnapshot(
            engaged=True, reason="ops", last_ok_mono=1.0, error_class=None, consecutive_failures=0
        )
    )
    try:
        with pytest.raises(ConfigurationError) as exc_info:
            await node._initialize()
        assert str(exc_info.value) == _NODE_KILL_ENGAGED
    finally:
        await node._shutdown()


async def test_startup_kill_unreadable_refuses(seeded: dict[str, Any]) -> None:
    """No clean read at startup: the unreadable text, never "not engaged"."""
    node = PaperTradingNode(_config(seeded))
    node._kill_monitor = _FirstReadMonitor(  # type: ignore[assignment]
        KillSwitchSnapshot(
            engaged=False,
            reason=None,
            last_ok_mono=None,
            error_class="OperationalError",
            consecutive_failures=1,
        )
    )
    try:
        with pytest.raises(ConfigurationError) as exc_info:
            await node._initialize()
        assert (
            str(exc_info.value)
            == "System kill switch state unreadable (OperationalError); refusing to start"
        )
    finally:
        await node._shutdown()


async def test_startup_telegram_unavailable_refuses(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured-but-broken Telegram refuses the node start with ``str(exc)``.

    ``from None`` keeps the probe's own exception (whose ``str()`` holds the token
    URL) out of the raised ConfigurationError's chain.
    """

    async def _refuse() -> None:
        raise TelegramUnavailableError(_TELEGRAM_503)

    monkeypatch.setattr("vibe_quant.paper.node.require_working_telegram", _refuse)
    node = PaperTradingNode(_config(seeded))
    try:
        with pytest.raises(ConfigurationError) as exc_info:
            await node._initialize()
        assert str(exc_info.value) == _TELEGRAM_503
        assert exc_info.value.__cause__ is None
    finally:
        await node._shutdown()


@pytest.mark.parametrize(
    ("engaged", "expected"),
    [
        (True, _NODE_KILL_ENGAGED),
        (False, None),
    ],
    ids=["engaged", "disengaged"],
)
async def test_startup_preserves_prior_clean_read_with_latest_error(
    seeded: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    engaged: bool,
    expected: str | None,
) -> None:
    """r3 F10: a failed latest read keeps the earlier CLEAN timestamp.

    The predicate is ``last_ok_mono is None``, not ``error_class is not None``:
    engaged still refuses with the engaged text; disengaged starts past the
    check (a later step is stubbed so the test proves it got there).
    """
    node = PaperTradingNode(_config(seeded))
    node._kill_monitor = _FirstReadMonitor(  # type: ignore[assignment]
        KillSwitchSnapshot(
            engaged=engaged,
            reason="ops" if engaged else None,
            last_ok_mono=1.0,
            error_class="OperationalError",
            consecutive_failures=1,
        )
    )

    def _past_check() -> Any:
        raise RuntimeError("PAST-CHECK")

    try:
        if expected is not None:
            with pytest.raises(ConfigurationError) as exc_info:
                await node._initialize()
            assert str(exc_info.value) == expected
        else:
            monkeypatch.setattr(node, "_load_strategy", _past_check)
            with pytest.raises(RuntimeError, match="PAST-CHECK"):
                await node._initialize()
    finally:
        await node._shutdown()


# --------------------------------------- r3 F11: sequential consent + lookup


async def test_admission_preserves_checkpoint_lookup_short_circuit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The persistence is opened only for a LIVE start that still needs consent.

    Cases: live without ``--live``, live start_fresh without
    ``--confirm-fresh-live``, live with ``--confirm-fresh-live`` and testnet.
    """
    calls: list[tuple[str, str]] = []

    def _spy(store: StatePersistence, account: str, trader_id: str) -> bool:
        calls.append((account, trader_id))
        raise AssertionError("the checkpoint lookup must be short-circuited")

    monkeypatch.setattr(lineage, "has_prior_state", _spy)
    started: list[PaperTradingConfig] = []
    _patch_run(monkeypatch, started)
    db_path = tmp_path / "state.db"

    live = tmp_path / "live.json"
    _write_config(live, testnet=False, db_path=db_path)
    assert await run_with_config(live) == 1
    assert _stderr(capsys) == [_CLI_LIVE_TEXT]

    fresh = tmp_path / "fresh.json"
    _write_config(fresh, testnet=False, start_fresh=True, db_path=db_path)
    assert await run_with_config(fresh, allow_live=True) == 1
    assert _stderr(capsys) == [_CLI_FRESH_TEXT]

    assert await run_with_config(fresh, allow_live=True, allow_fresh_live=True) == 0

    testnet = tmp_path / "testnet.json"
    _write_config(testnet, testnet=True, db_path=db_path)
    assert await run_with_config(testnet) == 0

    assert calls == []
    assert [(c.binance.testnet, c.start_fresh) for c in started] == [(False, True), (True, False)]


# ----------------------------------------- rabmo.10b: the pure admission decision

#: Start intents used by the table. ``live=False`` is testnet.
_TESTNET = StartIntent(live=False, confirm_live=False, start_fresh=False, confirm_fresh_live=False)
_LIVE_UNCONFIRMED = StartIntent(
    live=True, confirm_live=False, start_fresh=False, confirm_fresh_live=False
)
_LIVE_CONFIRMED = StartIntent(
    live=True, confirm_live=True, start_fresh=False, confirm_fresh_live=False
)
_LIVE_FRESH_UNCONFIRMED = StartIntent(
    live=True, confirm_live=True, start_fresh=True, confirm_fresh_live=False
)
_LIVE_FRESH_CONFIRMED = StartIntent(
    live=True, confirm_live=True, start_fresh=True, confirm_fresh_live=True
)

_CHECK_TABLE: tuple[tuple[str, StartIntent, StartFacts, Refusal | Admitted], ...] = (
    # (a) each refusal alone: one row per RefusalCode, every other fact clean.
    (
        "kill-missing",
        _TESTNET,
        StartFacts(kill_missing=True),
        Refusal(RefusalCode.KILL_MISSING, None),
    ),
    (
        "kill-unreadable",
        _TESTNET,
        StartFacts(kill_unreadable="no read within 10 s"),
        Refusal(RefusalCode.KILL_UNREADABLE, "no read within 10 s"),
    ),
    (
        "kill-engaged",
        _TESTNET,
        StartFacts(kill_engaged="maintenance"),
        Refusal(RefusalCode.KILL_ENGAGED, "maintenance"),
    ),
    (
        "session-active",
        _TESTNET,
        StartFacts(active_run_id=7),
        Refusal(RefusalCode.SESSION_ACTIVE, 7),
    ),
    (
        "live-not-confirmed",
        _LIVE_UNCONFIRMED,
        StartFacts(),
        Refusal(RefusalCode.LIVE_NOT_CONFIRMED, None),
    ),
    (
        "fresh-not-confirmed",
        _LIVE_FRESH_UNCONFIRMED,
        StartFacts(),
        Refusal(RefusalCode.FRESH_NOT_CONFIRMED, None),
    ),
    (
        "lock-dir-unavailable",
        _TESTNET,
        StartFacts(lock_dir_error="cannot resolve the OS user's home dir"),
        Refusal(RefusalCode.LOCK_DIR_UNAVAILABLE, "cannot resolve the OS user's home dir"),
    ),
    (
        "lock-probe-failed",
        _TESTNET,
        StartFacts(lock_probe_error="PermissionError"),
        Refusal(RefusalCode.LOCK_PROBE_FAILED, "PermissionError"),
    ),
    (
        "account-busy",
        _TESTNET,
        StartFacts(account_holder="pid 42"),
        Refusal(RefusalCode.ACCOUNT_BUSY, "pid 42"),
    ),
    (
        "trader-busy",
        _TESTNET,
        StartFacts(trader_holder="pid 43"),
        Refusal(RefusalCode.TRADER_BUSY, "pid 43"),
    ),
    (
        "no-prior-state",
        _LIVE_CONFIRMED,
        StartFacts(has_prior_state=False),
        Refusal(RefusalCode.NO_PRIOR_STATE, None),
    ),
    (
        "telegram-unavailable",
        _TESTNET,
        StartFacts(telegram_error="telegram unavailable"),
        Refusal(RefusalCode.TELEGRAM_UNAVAILABLE, "telegram unavailable"),
    ),
    # (b) presence edges: an empty reason and a zero run id are still refusals;
    # kill_missing=False means "probed, present".
    (
        "engaged-empty-reason-is-a-refusal",
        _TESTNET,
        StartFacts(kill_engaged=""),
        Refusal(RefusalCode.KILL_ENGAGED, ""),
    ),
    (
        "active-run-id-zero-is-a-refusal",
        _TESTNET,
        StartFacts(active_run_id=0),
        Refusal(RefusalCode.SESSION_ACTIVE, 0),
    ),
    (
        "kill-missing-false-is-not-a-refusal",
        _TESTNET,
        StartFacts(kill_missing=False),
        Admitted(forward_fresh_consent=False),
    ),
    # (c) precedence: two applicable facts, the earlier RefusalCode wins.
    (
        "kill-missing-over-engaged",
        _TESTNET,
        StartFacts(kill_missing=True, kill_engaged="maintenance"),
        Refusal(RefusalCode.KILL_MISSING, None),
    ),
    (
        "kill-unreadable-over-engaged",
        _TESTNET,
        StartFacts(kill_unreadable="OperationalError", kill_engaged="maintenance"),
        Refusal(RefusalCode.KILL_UNREADABLE, "OperationalError"),
    ),
    (
        "engaged-over-session-active",
        _TESTNET,
        StartFacts(kill_engaged="maintenance", active_run_id=3),
        Refusal(RefusalCode.KILL_ENGAGED, "maintenance"),
    ),
    (
        "session-active-over-live-not-confirmed",
        _LIVE_UNCONFIRMED,
        StartFacts(active_run_id=3),
        Refusal(RefusalCode.SESSION_ACTIVE, 3),
    ),
    (
        "live-not-confirmed-over-fresh-not-confirmed",
        StartIntent(live=True, confirm_live=False, start_fresh=True, confirm_fresh_live=False),
        StartFacts(),
        Refusal(RefusalCode.LIVE_NOT_CONFIRMED, None),
    ),
    (
        "fresh-not-confirmed-over-lock-dir",
        _LIVE_FRESH_UNCONFIRMED,
        StartFacts(lock_dir_error="cannot resolve the OS user's home dir"),
        Refusal(RefusalCode.FRESH_NOT_CONFIRMED, None),
    ),
    (
        "lock-dir-over-lock-probe",
        _TESTNET,
        StartFacts(lock_dir_error="no lock dir", lock_probe_error="PermissionError"),
        Refusal(RefusalCode.LOCK_DIR_UNAVAILABLE, "no lock dir"),
    ),
    (
        "lock-probe-over-account-busy",
        _TESTNET,
        StartFacts(lock_probe_error="PermissionError", account_holder="pid 42"),
        Refusal(RefusalCode.LOCK_PROBE_FAILED, "PermissionError"),
    ),
    (
        "account-busy-over-trader-busy",
        _TESTNET,
        StartFacts(account_holder="pid 42", trader_holder="pid 43"),
        Refusal(RefusalCode.ACCOUNT_BUSY, "pid 42"),
    ),
    (
        "account-busy-over-no-prior-state",
        _LIVE_CONFIRMED,
        StartFacts(account_holder="pid 42", has_prior_state=False),
        Refusal(RefusalCode.ACCOUNT_BUSY, "pid 42"),
    ),
    (
        "trader-busy-over-no-prior-state",
        _LIVE_CONFIRMED,
        StartFacts(trader_holder="pid 43", has_prior_state=False),
        Refusal(RefusalCode.TRADER_BUSY, "pid 43"),
    ),
    (
        "no-prior-state-over-telegram",
        _LIVE_CONFIRMED,
        StartFacts(has_prior_state=False, telegram_error="telegram unavailable"),
        Refusal(RefusalCode.NO_PRIOR_STATE, None),
    ),
    (
        "all-facts-bad-is-kill-missing",
        _LIVE_UNCONFIRMED,
        StartFacts(
            kill_missing=True,
            kill_unreadable="OperationalError",
            kill_engaged="maintenance",
            active_run_id=3,
            lock_dir_error="no lock dir",
            lock_probe_error="PermissionError",
            account_holder="pid 42",
            trader_holder="pid 43",
            has_prior_state=False,
            telegram_error="telegram unavailable",
        ),
        Refusal(RefusalCode.KILL_MISSING, None),
    ),
    # (d) consent edges: testnet never refuses on consent, and never forwards it.
    (
        "testnet-consent-000",
        StartIntent(live=False, confirm_live=False, start_fresh=False, confirm_fresh_live=False),
        StartFacts(has_prior_state=False),
        Admitted(forward_fresh_consent=False),
    ),
    (
        "testnet-consent-001",
        StartIntent(live=False, confirm_live=False, start_fresh=False, confirm_fresh_live=True),
        StartFacts(has_prior_state=False),
        Admitted(forward_fresh_consent=False),
    ),
    (
        "testnet-consent-010",
        StartIntent(live=False, confirm_live=False, start_fresh=True, confirm_fresh_live=False),
        StartFacts(has_prior_state=False),
        Admitted(forward_fresh_consent=False),
    ),
    (
        "testnet-consent-011",
        StartIntent(live=False, confirm_live=False, start_fresh=True, confirm_fresh_live=True),
        StartFacts(has_prior_state=False),
        Admitted(forward_fresh_consent=False),
    ),
    (
        "testnet-consent-100",
        StartIntent(live=False, confirm_live=True, start_fresh=False, confirm_fresh_live=False),
        StartFacts(has_prior_state=False),
        Admitted(forward_fresh_consent=False),
    ),
    (
        "testnet-consent-101",
        StartIntent(live=False, confirm_live=True, start_fresh=False, confirm_fresh_live=True),
        StartFacts(has_prior_state=False),
        Admitted(forward_fresh_consent=False),
    ),
    (
        "testnet-consent-110",
        StartIntent(live=False, confirm_live=True, start_fresh=True, confirm_fresh_live=False),
        StartFacts(has_prior_state=False),
        Admitted(forward_fresh_consent=False),
    ),
    (
        "testnet-consent-111",
        StartIntent(live=False, confirm_live=True, start_fresh=True, confirm_fresh_live=True),
        StartFacts(has_prior_state=False),
        Admitted(forward_fresh_consent=False),
    ),
    # (d) the full live grid: start_fresh x confirm_fresh_live x has_prior_state,
    # each expected value hand-derived from
    # live and confirm_fresh_live and (start_fresh or has_prior_state is False)
    # and the three consent refusals above.
    (
        "live-fresh-confirmed-prior-none",
        _LIVE_FRESH_CONFIRMED,
        StartFacts(has_prior_state=None),
        Admitted(forward_fresh_consent=True),
    ),
    (
        "live-fresh-confirmed-prior-true",
        _LIVE_FRESH_CONFIRMED,
        StartFacts(has_prior_state=True),
        Admitted(forward_fresh_consent=True),
    ),
    (
        "live-fresh-confirmed-prior-false",
        _LIVE_FRESH_CONFIRMED,
        StartFacts(has_prior_state=False),
        Admitted(forward_fresh_consent=True),
    ),
    (
        "live-fresh-unconfirmed-prior-none",
        _LIVE_FRESH_UNCONFIRMED,
        StartFacts(has_prior_state=None),
        Refusal(RefusalCode.FRESH_NOT_CONFIRMED, None),
    ),
    (
        "live-fresh-unconfirmed-prior-true",
        _LIVE_FRESH_UNCONFIRMED,
        StartFacts(has_prior_state=True),
        Refusal(RefusalCode.FRESH_NOT_CONFIRMED, None),
    ),
    (
        "live-fresh-unconfirmed-prior-false",
        _LIVE_FRESH_UNCONFIRMED,
        StartFacts(has_prior_state=False),
        Refusal(RefusalCode.FRESH_NOT_CONFIRMED, None),
    ),
    (
        "live-confirmed-fresh-consent-prior-none",
        StartIntent(live=True, confirm_live=True, start_fresh=False, confirm_fresh_live=True),
        StartFacts(has_prior_state=None),
        Admitted(forward_fresh_consent=False),
    ),
    (
        "live-confirmed-fresh-consent-prior-true",
        StartIntent(live=True, confirm_live=True, start_fresh=False, confirm_fresh_live=True),
        StartFacts(has_prior_state=True),
        Admitted(forward_fresh_consent=False),
    ),
    (
        "live-confirmed-fresh-consent-prior-false",
        StartIntent(live=True, confirm_live=True, start_fresh=False, confirm_fresh_live=True),
        StartFacts(has_prior_state=False),
        Admitted(forward_fresh_consent=True),
    ),
    (
        "live-confirmed-prior-none",
        _LIVE_CONFIRMED,
        StartFacts(has_prior_state=None),
        Admitted(forward_fresh_consent=False),
    ),
    (
        "live-confirmed-prior-true",
        _LIVE_CONFIRMED,
        StartFacts(has_prior_state=True),
        Admitted(forward_fresh_consent=False),
    ),
    (
        "live-confirmed-prior-false",
        _LIVE_CONFIRMED,
        StartFacts(has_prior_state=False),
        Refusal(RefusalCode.NO_PRIOR_STATE, None),
    ),
    # (e) a second, unrelated detail per gate that carries one, so the row pins
    # the forwarded value and not just the fact's presence.
    (
        "lock-probe-failed-other-class",
        _TESTNET,
        StartFacts(lock_probe_error="NotADirectoryError"),
        Refusal(RefusalCode.LOCK_PROBE_FAILED, "NotADirectoryError"),
    ),
    (
        "account-busy-other-holder",
        _TESTNET,
        StartFacts(account_holder="pid 99"),
        Refusal(RefusalCode.ACCOUNT_BUSY, "pid 99"),
    ),
    (
        "trader-busy-other-holder",
        _TESTNET,
        StartFacts(trader_holder="pid 77"),
        Refusal(RefusalCode.TRADER_BUSY, "pid 77"),
    ),
    (
        "telegram-unavailable-other-reason",
        _TESTNET,
        StartFacts(telegram_error="bot token rejected"),
        Refusal(RefusalCode.TELEGRAM_UNAVAILABLE, "bot token rejected"),
    ),
    # (f) an empty detail is still present (`is not None`), so it refuses and the
    # empty string is what the adapter formats.
    (
        "kill-unreadable-empty-is-a-refusal",
        _TESTNET,
        StartFacts(kill_unreadable=""),
        Refusal(RefusalCode.KILL_UNREADABLE, ""),
    ),
    (
        "lock-dir-unavailable-empty-is-a-refusal",
        _TESTNET,
        StartFacts(lock_dir_error=""),
        Refusal(RefusalCode.LOCK_DIR_UNAVAILABLE, ""),
    ),
    (
        "lock-probe-failed-empty-is-a-refusal",
        _TESTNET,
        StartFacts(lock_probe_error=""),
        Refusal(RefusalCode.LOCK_PROBE_FAILED, ""),
    ),
    (
        "account-busy-empty-is-a-refusal",
        _TESTNET,
        StartFacts(account_holder=""),
        Refusal(RefusalCode.ACCOUNT_BUSY, ""),
    ),
    (
        "trader-busy-empty-is-a-refusal",
        _TESTNET,
        StartFacts(trader_holder=""),
        Refusal(RefusalCode.TRADER_BUSY, ""),
    ),
    (
        "telegram-unavailable-empty-is-a-refusal",
        _TESTNET,
        StartFacts(telegram_error=""),
        Refusal(RefusalCode.TELEGRAM_UNAVAILABLE, ""),
    ),
)


@pytest.mark.parametrize(
    ("intent", "facts", "expected"),
    [
        pytest.param(intent, facts, expected, id=row_id)
        for row_id, intent, facts, expected in _CHECK_TABLE
    ],
)
def test_check_table(intent: StartIntent, facts: StartFacts, expected: Refusal | Admitted) -> None:
    """The contract: every gate's outcome, and the winner when two apply."""
    assert check(intent, facts) == expected
