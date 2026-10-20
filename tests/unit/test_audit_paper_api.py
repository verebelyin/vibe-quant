"""/api/paper + /api/system paper control (vibe-quant-e70tl.4 / .19).

The paper node is simulated by a task that drains the ``paper_commands``
queue exactly like ``PaperTradingNode.process_commands`` does, so these tests
exercise the real API <-> node protocol without spawning a process.
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import functools
import json
import logging
import os
import sqlite3
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.schemas.paper_trading import PaperStartRequest
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.paper.account_lock import acquire_account_lock
from vibe_quant.paper.config import BinanceTestnetConfig, account_id, is_valid_trader_id
from vibe_quant.paper.node import PaperTradingNode
from vibe_quant.paper.persistence import (
    PaperCommandQueue,
    StateCheckpoint,
    StatePersistence,
    StoreWorker,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

DSL = {
    "name": "test_strategy",
    "timeframe": "1h",
    "indicators": {"rsi_14": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi_14 < 30"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "api.db"
    monkeypatch.setattr("vibe_quant.db.connection.DEFAULT_DB_PATH", db)
    monkeypatch.setattr(
        "vibe_quant.api.routers.paper_trading.PAPER_CONFIG_DIR", tmp_path / "paper_configs"
    )
    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "tk")
    monkeypatch.setenv("BINANCE_TESTNET_API_SECRET", "ts")
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    return db


class Ctx:
    def __init__(
        self, ac: AsyncClient, state: StateManager, jobs: BacktestJobManager, tmp: Path
    ) -> None:
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

    def config_for(self, run_id: int) -> dict[str, Any]:
        path = self.tmp / "paper_configs" / f"paper_{run_id}.json"
        data: dict[str, Any] = json.loads(path.read_text())
        return data

    def activate(self, run_id: int) -> None:
        """Mark a paper run's job as running (what start_job would record)."""
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
        c = Ctx(ac, state, jobs, tmp_path)

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


# ------------------------------------------------------------------- /start


def test_start_request_defaults_to_testnet() -> None:
    assert PaperStartRequest().testnet is True
    assert PaperStartRequest().confirm_live is False
    assert PaperStartRequest().start_fresh is False
    assert PaperStartRequest().confirm_fresh_live is False


async def test_start_from_validation_run_writes_runnable_config(ctx: Ctx) -> None:
    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r.status_code == 201, r.text
    body = r.json()
    run_id = body["run_id"]
    assert body["trader_id"] == f"PAPER-{run_id:03d}"
    assert is_valid_trader_id(body["trader_id"])
    assert body["testnet"] is True

    cfg = ctx.config_for(run_id)
    assert cfg["symbols"] == ["BTCUSDT"], "audit: /start wrote symbols: [] (always failed)"
    assert cfg["strategy_id"] == ctx.strategy_id
    assert cfg["validation_run_id"] == ctx.validation_run
    assert cfg["binance"] == {"testnet": True}
    assert cfg["db_path"].endswith("api.db")
    assert cfg["start_fresh"] is False
    assert "api_key" not in json.dumps(cfg)
    assert "--live" not in ctx.launched[0]

    # The CLI loads it into a config that validates (testnet env creds).
    from vibe_quant.paper.cli import load_config_from_json

    loaded = load_config_from_json(ctx.tmp / "paper_configs" / f"paper_{run_id}.json")
    assert loaded.validate() == []
    assert loaded.binance.api_key == "tk"

    run = ctx.state.get_backtest_run(run_id)
    assert run is not None and run["symbols"] == ["BTCUSDT"]


async def test_start_fresh_recorded_in_params_and_config(ctx: Ctx) -> None:
    """R21: start_fresh travels into the run params AND the launched config JSON."""
    from vibe_quant.paper.cli import load_config_from_json

    r = await ctx.ac.post(
        "/api/paper/start",
        json={"validation_run_id": ctx.validation_run, "start_fresh": True},
    )
    assert r.status_code == 201, r.text
    run_id = r.json()["run_id"]
    run = ctx.state.get_backtest_run(run_id)
    assert run is not None
    assert run["parameters"]["start_fresh"] is True
    cfg_path = ctx.tmp / "paper_configs" / f"paper_{run_id}.json"
    assert ctx.config_for(run_id)["start_fresh"] is True
    assert load_config_from_json(cfg_path).start_fresh is True

    # Absent -> False, in all three places: the field is not optional-true.
    ctx.jobs.conn.execute("DELETE FROM background_jobs")
    ctx.jobs.conn.commit()
    r2 = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r2.status_code == 201, r2.text
    run_id2 = r2.json()["run_id"]
    run2 = ctx.state.get_backtest_run(run_id2)
    assert run2 is not None
    assert run2["parameters"]["start_fresh"] is False
    assert ctx.config_for(run_id2)["start_fresh"] is False
    cfg_path2 = ctx.tmp / "paper_configs" / f"paper_{run_id2}.json"
    assert load_config_from_json(cfg_path2).start_fresh is False


_ABSENT = object()


@pytest.mark.parametrize(
    ("start_fresh", "status"),
    [
        (_ABSENT, 201),
        (True, 201),
        ("true", 422),  # a string bool is not a bool
        (1, 422),  # an int 1 is not a bool
        (None, 422),  # explicit null is not the absent default
    ],
)
async def test_start_fresh_body_rows(ctx: Ctx, start_fresh: object, status: int) -> None:
    """R21: strict bool — only JSON true/false or the absent default are accepted."""
    payload: dict[str, object] = {"validation_run_id": ctx.validation_run}
    if start_fresh is not _ABSENT:
        payload["start_fresh"] = start_fresh
    r = await ctx.ac.post("/api/paper/start", json=payload)
    assert r.status_code == status, r.text
    if status != 201:
        assert ctx.launched == []
        return
    run_id = r.json()["run_id"]
    run = ctx.state.get_backtest_run(run_id)
    assert run is not None
    expected = start_fresh is True
    assert run["parameters"]["start_fresh"] is expected
    assert ctx.config_for(run_id)["start_fresh"] is expected


async def test_second_start_while_active_is_409(ctx: Ctx) -> None:
    r1 = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r1.status_code == 201
    r2 = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r2.status_code == 409
    assert len(ctx.launched) == 1


def _api_account(api_key: str = "tk", *, testnet: bool = True) -> str:
    """The account id the API resolves from the ``tmp_db`` fixture's env keys."""
    return account_id(BinanceTestnetConfig(api_key=api_key, api_secret="ts", testnet=testnet))


def _paper_run_count(state: StateManager) -> int:
    row = state.conn.execute(
        "SELECT COUNT(*) FROM backtest_runs WHERE run_mode = 'paper'"
    ).fetchone()
    assert row is not None
    return int(row[0])


def _seed_account_checkpoint(
    db_path: Path, *, api_key: str, testnet: bool, trader_id: str = "PAPER-OTHER"
) -> None:
    """Give the env key's account a saved risk state (any trader's checkpoint)."""
    account = _api_account(api_key, testnet=testnet)
    store = StatePersistence(db_path, trader_id=trader_id)
    try:
        store.save_checkpoint(
            StateCheckpoint(trader_id=trader_id, node_status={"account": account})
        )
    finally:
        store.close()


def _hold_raw_fd(tmp: Path, account: str) -> int:
    """A foreign process's holder: its own open file description on the lock file."""
    directory = tmp / "paper_locks"
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / f"{account.replace(':', '_')}.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.write(fd, b"4242 PAPER-OTHER\n")
    return fd


async def test_start_409_when_another_process_holds_the_account_lock(ctx: Ctx) -> None:
    """rw1g8: a running CLI session on this account refuses a second API start."""
    held = acquire_account_lock(_api_account(), "PAPER-OTHER")
    before = _paper_run_count(ctx.state)
    try:
        r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    finally:
        held.release()

    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert f"Paper session already holds account {_api_account()}" in detail
    assert "PAPER-OTHER" in detail
    assert ctx.launched == []
    assert _paper_run_count(ctx.state) == before
    # The probe left nothing held: the same request succeeds once the holder is gone.
    r2 = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r2.status_code == 201, r2.text


def _missing_passwd_entry(_uid: int) -> None:
    raise KeyError("no such uid")


def _empty_passwd_home(_uid: int) -> SimpleNamespace:
    return SimpleNamespace(pw_dir="")


_OS_HOME_REFUSAL = "cannot resolve the OS user's home dir for the paper lock"


@pytest.mark.parametrize(
    "getpwuid",
    [_missing_passwd_entry, _empty_passwd_home],
    ids=["no-passwd-entry", "empty-pw-dir"],
)
async def test_start_503_when_the_os_home_cannot_be_resolved(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch, getpwuid: Any
) -> None:
    """28j0h: no passwd home -> no lock dir -> 503 with the reason, not a bare 500.

    The raise happens inside the lock probe, before any run row or launch; the real
    home is never touched because no lock path is built.
    """
    monkeypatch.setattr("vibe_quant.paper.account_lock.PAPER_LOCK_DIR", None)
    monkeypatch.setattr("vibe_quant.paper.account_lock.pwd.getpwuid", getpwuid)
    before = _paper_run_count(ctx.state)

    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})

    assert r.status_code == 503, r.text
    assert r.json()["detail"] == _OS_HOME_REFUSAL
    assert ctx.launched == []
    assert _paper_run_count(ctx.state) == before


async def test_restore_503_when_the_os_home_cannot_be_resolved(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """28j0h: /restore refuses with the same 503 reason before the run row."""
    cfg_dir = ctx.tmp / "paper_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "paper_1.json").write_text(
        json.dumps(
            {
                "trader_id": "PAPER-009",
                "strategy_id": ctx.strategy_id,
                "symbols": ["BTCUSDT"],
                "binance": {"testnet": True},
            }
        )
    )
    monkeypatch.setattr("vibe_quant.paper.account_lock.PAPER_LOCK_DIR", None)
    monkeypatch.setattr("vibe_quant.paper.account_lock.pwd.getpwuid", _missing_passwd_entry)
    before = _paper_run_count(ctx.state)

    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-009"})

    assert r.status_code == 503, r.text
    assert r.json()["detail"] == _OS_HOME_REFUSAL
    assert ctx.launched == []
    assert _paper_run_count(ctx.state) == before


async def test_start_503_when_the_lock_probe_cannot_open_the_lock_file(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """65iq3: an OSError opening the lock file (e.g. PermissionError) is a 503 naming
    the problem class, not a bare 500; no run row, no launch, no exception text."""
    marker = "EACCES-TEXT-MARKER-65"
    fd = _hold_raw_fd(ctx.tmp, _api_account())
    os.close(fd)
    real_open = os.open

    def _denied(path: object, flags: int, *args: object) -> int:
        if str(path).endswith(".lock"):
            raise PermissionError(errno.EACCES, marker)
        return real_open(path, flags, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", _denied)
    before = _paper_run_count(ctx.state)
    with caplog.at_level(logging.DEBUG):
        r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})

    assert r.status_code == 503, r.text
    assert r.json()["detail"] == (
        "cannot probe the paper account lock (PermissionError); check the lock dir"
    )
    assert ctx.launched == []
    assert _paper_run_count(ctx.state) == before
    assert marker not in caplog.text
    assert marker not in r.text


async def test_start_409_when_the_holder_body_is_unreadable(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """F1: a held lock with an unreadable body is still a 409 (holder unknown), not a 500."""
    marker = "EIO-TEXT-MARKER-91"

    def _eio(_fd: int, _length: int, _offset: int) -> bytes:
        raise OSError(errno.EIO, marker)

    fd = _hold_raw_fd(ctx.tmp, _api_account())
    monkeypatch.setattr(os, "pread", _eio)
    before = _paper_run_count(ctx.state)
    try:
        with caplog.at_level(logging.DEBUG):
            r = await ctx.ac.post(
                "/api/paper/start", json={"validation_run_id": ctx.validation_run}
            )
    finally:
        os.close(fd)

    assert r.status_code == 409, r.text
    assert r.json()["detail"] == (
        f"Paper session already holds account {_api_account()} (unknown); stop it first"
    )
    assert ctx.launched == []
    assert _paper_run_count(ctx.state) == before
    assert marker not in caplog.text
    assert marker not in r.text


async def test_start_allowed_while_another_account_is_locked(ctx: Ctx) -> None:
    """The lock is per account, not global."""
    held = acquire_account_lock(_api_account("other_key"), "PAPER-OTHER")
    try:
        r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    finally:
        held.release()
    assert r.status_code == 201, r.text


async def test_start_allowed_when_a_dead_holder_left_the_file(ctx: Ctx) -> None:
    """Closing the holder's fd is what SIGKILL does: no stale lock, no cleanup code."""
    fd = _hold_raw_fd(ctx.tmp, _api_account())
    r_busy = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r_busy.status_code == 409, r_busy.text
    os.close(fd)

    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r.status_code == 201, r.text


async def test_live_start_with_the_same_key_is_a_different_account(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """testnet-locked key K does not block a LIVE session on key K."""
    monkeypatch.setenv("BINANCE_API_KEY", "tk")
    monkeypatch.setenv("BINANCE_API_SECRET", "ts")
    # 6ub67: a LIVE account with history needs no extra consent.
    _seed_account_checkpoint(ctx.tmp / "api.db", api_key="tk", testnet=False)
    held = acquire_account_lock(_api_account("tk", testnet=True), "PAPER-OTHER")
    try:
        r = await ctx.ac.post(
            "/api/paper/start",
            json={
                "validation_run_id": ctx.validation_run,
                "testnet": False,
                "confirm_live": True,
            },
        )
    finally:
        held.release()
    assert r.status_code == 201, r.text
    assert ctx.launched[0][-1] == "--live"


async def test_live_requires_confirmation_and_live_env(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = await ctx.ac.post(
        "/api/paper/start", json={"validation_run_id": ctx.validation_run, "testnet": False}
    )
    assert r.status_code == 400 and "confirm_live" in r.text
    r = await ctx.ac.post(
        "/api/paper/start",
        json={"validation_run_id": ctx.validation_run, "testnet": False, "confirm_live": True},
    )
    assert r.status_code == 400 and "BINANCE_API_KEY" in r.text, r.text
    assert ctx.launched == []

    monkeypatch.setenv("BINANCE_API_KEY", "lk")
    monkeypatch.setenv("BINANCE_API_SECRET", "ls")
    # 6ub67: the live account already has history, so confirm_live is the only gate.
    _seed_account_checkpoint(ctx.tmp / "api.db", api_key="lk", testnet=False)
    r = await ctx.ac.post(
        "/api/paper/start",
        json={"validation_run_id": ctx.validation_run, "testnet": False, "confirm_live": True},
    )
    assert r.status_code == 201, r.text
    assert ctx.launched[0][-1] == "--live"
    assert ctx.config_for(r.json()["run_id"])["binance"] == {"testnet": False}


async def test_missing_testnet_env_is_400(ctx: Ctx, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY")
    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r.status_code == 400 and "BINANCE_TESTNET_API_KEY" in r.text


@pytest.mark.parametrize(
    (
        "testnet",
        "start_fresh",
        "confirm_fresh_live",
        "confirm_live",
        "status",
        "detail_marker",
        "fresh_flag",
        "seed_checkpoint",
    ),
    [
        # v4l2n row 3: testnet start_fresh needs no extra confirmation; the flag is ignored.
        (True, True, True, _ABSENT, 201, None, False, False),
        # row 5/6: live + start_fresh without the flag is refused BEFORE any run row.
        (False, True, _ABSENT, True, 400, "confirm_fresh_live", False, False),
        (False, True, False, True, 400, "confirm_fresh_live", False, False),
        # row 7: confirmed -> the CLI gets the flag, and --live stays last.
        (False, True, True, True, 201, None, True, False),
        # row 8: the existing confirm_live check wins over the fresh check.
        (False, True, True, _ABSENT, 400, "confirm_live", False, False),
        # row 9: strict bool -- string/int/null never coerce.
        (False, True, "true", True, 422, None, False, False),
        (False, True, 1, True, 422, None, False, False),
        (False, True, None, True, 422, None, False, False),
        # 6ub67 row 4: live, no start_fresh, no saved risk state, no consent -> refused.
        (False, False, _ABSENT, True, 400, "no saved risk state", False, False),
        (False, False, False, True, 400, "no saved risk state", False, False),
        # 6ub67 row 5: the same account with explicit consent -> flag forwarded.
        (False, False, True, True, 201, None, True, False),
        # row 10: with saved risk state the flag is not needed for a normal start.
        (False, False, True, True, 201, None, False, True),
    ],
    ids=[
        "row3-testnet-fresh",
        "row5-live-fresh-absent",
        "row6-live-fresh-false",
        "row7-live-fresh-confirmed",
        "row8-confirm-live-wins",
        "row9-string",
        "row9-int",
        "row9-null",
        "row4-live-no-checkpoint",
        "row4-live-no-checkpoint-false",
        "row5-live-no-checkpoint-confirmed",
        "row10-live-not-fresh-with-checkpoint",
    ],
)
async def test_live_start_fresh_confirmation_rows(
    ctx: Ctx,
    monkeypatch: pytest.MonkeyPatch,
    testnet: bool,
    start_fresh: bool,
    confirm_fresh_live: object,
    confirm_live: object,
    status: int,
    detail_marker: str | None,
    fresh_flag: bool,
    seed_checkpoint: bool,
) -> None:
    """v4l2n + 6ub67: a LIVE start needs consent when it resets the limits or has none."""
    monkeypatch.setenv("BINANCE_API_KEY", "lk")
    monkeypatch.setenv("BINANCE_API_SECRET", "ls")
    if seed_checkpoint:
        _seed_account_checkpoint(ctx.tmp / "api.db", api_key="lk", testnet=False)

    payload: dict[str, object] = {
        "validation_run_id": ctx.validation_run,
        "testnet": testnet,
        "start_fresh": start_fresh,
    }
    if confirm_fresh_live is not _ABSENT:
        payload["confirm_fresh_live"] = confirm_fresh_live
    if confirm_live is not _ABSENT:
        payload["confirm_live"] = confirm_live

    before = _paper_run_count(ctx.state)
    r = await ctx.ac.post("/api/paper/start", json=payload)

    assert r.status_code == status, r.text
    if status == 201:
        assert len(ctx.launched) == 1
        if fresh_flag:
            assert ctx.launched[0][-2:] == ["--confirm-fresh-live", "--live"]
        else:
            assert "--confirm-fresh-live" not in ctx.launched[0]
        if not testnet:
            assert ctx.launched[0][-1] == "--live"
        run_id = r.json()["run_id"]
        run = ctx.state.get_backtest_run(run_id)
        assert run is not None
        assert run["parameters"]["start_fresh"] is start_fresh
        assert ctx.config_for(run_id)["start_fresh"] is start_fresh
        return

    assert ctx.launched == []
    assert _paper_run_count(ctx.state) == before
    if detail_marker is not None:
        assert detail_marker in json.dumps(r.json()["detail"])


async def test_live_start_accepts_a_legacy_checkpoint_of_the_posted_trader(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """6ub67 + I6: a legacy row without an account key counts for its own trader."""
    monkeypatch.setenv("BINANCE_API_KEY", "lk")
    monkeypatch.setenv("BINANCE_API_SECRET", "ls")
    store = StatePersistence(ctx.tmp / "api.db", trader_id="LIVE-777")
    try:
        store.save_checkpoint(
            StateCheckpoint(trader_id="LIVE-777", node_status={"state": "running"})
        )
    finally:
        store.close()

    r = await ctx.ac.post(
        "/api/paper/start",
        json={
            "validation_run_id": ctx.validation_run,
            "testnet": False,
            "confirm_live": True,
            "trader_id": "LIVE-777",
        },
    )

    assert r.status_code == 201, r.text
    assert ctx.launched[0][-1] == "--live"
    assert "--confirm-fresh-live" not in ctx.launched[0]


@pytest.mark.parametrize(
    ("exc", "marker"),
    [
        (RuntimeError("LOOKUP-MARKER-31"), "LOOKUP-MARKER-31"),
        (sqlite3.OperationalError("database is locked MARKER-58"), "MARKER-58"),
        (OSError(errno.EIO, "disk MARKER-93"), "MARKER-93"),
    ],
    ids=["runtime", "sqlite", "oserror"],
)
async def test_live_start_fails_closed_when_the_checkpoint_lookup_raises(
    ctx: Ctx,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    exc: Exception,
    marker: str,
) -> None:
    """6ub67: a lookup that cannot answer counts as "no checkpoint", class only logged."""

    def _boom(self: StatePersistence, account: str, trader_id: str) -> bool:
        raise exc

    monkeypatch.setenv("BINANCE_API_KEY", "lk")
    monkeypatch.setenv("BINANCE_API_SECRET", "ls")
    monkeypatch.setattr(StatePersistence, "has_account_checkpoint", _boom)
    payload: dict[str, object] = {
        "validation_run_id": ctx.validation_run,
        "testnet": False,
        "confirm_live": True,
    }
    before = _paper_run_count(ctx.state)

    with caplog.at_level(logging.DEBUG):
        refused = await ctx.ac.post("/api/paper/start", json=payload)

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == (
        "LIVE start has no saved risk state for this account (new or rotated "
        "API key, or first live session): drawdown, daily-loss and "
        "losing-streak limits start from zero: set confirm_fresh_live=true "
        "to proceed"
    )
    assert ctx.launched == []
    assert _paper_run_count(ctx.state) == before
    assert type(exc).__name__ in caplog.text
    assert marker not in caplog.text
    assert marker not in refused.text

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        consented = await ctx.ac.post(
            "/api/paper/start", json={**payload, "confirm_fresh_live": True}
        )

    assert consented.status_code == 201, consented.text
    assert ctx.launched[0][-2:] == ["--confirm-fresh-live", "--live"]
    assert _paper_run_count(ctx.state) == before + 1
    assert marker not in caplog.text
    assert marker not in consented.text


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"strategy_id": 1, "api_key": "K", "api_secret": "S", "symbols": ["BTCUSDT"]}, 422),
        ({"strategy_id": 1, "trader_id": "paper_1", "symbols": ["BTCUSDT"]}, 422),
        ({"strategy_id": 1, "symbols": ["BTCUSDT"], "risk_per_trade": 2}, 422),
        ({"strategy_id": 1, "symbols": ["BTCUSDT"], "sizing_method": "kelly"}, 422),
        ({"strategy_id": 1}, 400),
        ({}, 400),
    ],
)
async def test_start_rejects_bad_requests(ctx: Ctx, payload: dict[str, Any], status: int) -> None:
    if "strategy_id" in payload:
        payload = {**payload, "strategy_id": ctx.strategy_id}
    r = await ctx.ac.post("/api/paper/start", json=payload)
    assert r.status_code == status, r.text
    assert ctx.launched == []


async def test_start_without_validation_run_uses_given_symbols(ctx: Ctx) -> None:
    r = await ctx.ac.post(
        "/api/paper/start",
        json={"strategy_id": ctx.strategy_id, "symbols": ["ethusdt"], "risk_per_trade": 0.01},
    )
    assert r.status_code == 201, r.text
    cfg = ctx.config_for(r.json()["run_id"])
    assert cfg["symbols"] == ["ETHUSDT"]
    assert cfg["validation_run_id"] is None
    assert cfg["sizing"] == {"risk_per_trade": "0.01"}


# ------------------------------------------------------------- control


async def _fake_node(db: Path, trader_id: str, handler: Any, stop: asyncio.Event) -> None:
    q = PaperCommandQueue(db)
    try:
        while not stop.is_set():
            for cmd in q.claim_pending(trader_id):
                ok, result, error = handler(cmd)
                if ok:
                    q.complete(cmd.id, result)
                else:
                    q.fail(cmd.id, error, result)
            await asyncio.sleep(0.02)
    finally:
        q.close()


async def _real_node_loop(node: PaperTradingNode, stop: asyncio.Event) -> None:
    """Drain the queue through the real ``PaperTradingNode.process_commands``."""
    try:
        while not stop.is_set():
            await node.process_commands()
            await asyncio.sleep(0.02)
    finally:
        if node._commands is not None:
            node._commands.close()


async def _start(ctx: Ctx) -> tuple[int, str]:
    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.validation_run})
    assert r.status_code == 201
    return r.json()["run_id"], r.json()["trader_id"]


async def test_close_all_returns_node_errors_as_502(ctx: Ctx, tmp_db: Path) -> None:
    _run_id, trader_id = await _start(ctx)
    seen: list[str] = []

    def handler(cmd: Any) -> tuple[bool, dict[str, Any], str]:
        seen.append(cmd.command)
        return (
            False,
            {"targeted_positions": ["P-1"], "still_open": ["P-1"], "errors": ["venue down"]},
            "venue down; positions still open after 10s: ['P-1']",
        )

    stop = asyncio.Event()
    task = asyncio.create_task(_fake_node(tmp_db, trader_id, handler, stop))
    try:
        r = await ctx.ac.post("/api/paper/close-all-positions")
    finally:
        stop.set()
        await task
    assert seen == ["close_all"]
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert "venue down" in detail["error"]
    assert detail["result"]["still_open"] == ["P-1"]


async def test_close_all_success(ctx: Ctx, tmp_db: Path) -> None:
    _run_id, trader_id = await _start(ctx)

    def handler(cmd: Any) -> tuple[bool, dict[str, Any], None]:
        return True, {"targeted_positions": ["P-1", "P-2"], "still_open": [], "errors": []}, None

    stop = asyncio.Event()
    task = asyncio.create_task(_fake_node(tmp_db, trader_id, handler, stop))
    try:
        r = await ctx.ac.post("/api/paper/close-all-positions")
    finally:
        stop.set()
        await task
    assert r.status_code == 200, r.text
    assert r.json()["targeted_positions"] == ["P-1", "P-2"]
    assert r.json()["still_open"] == []


async def test_raised_command_error_never_reaches_api_detail(
    ctx: Ctx, tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node's command-failure detail must be class + fixed text, never ``str(exc)``.

    The exception text of a failing command can hold a signed request URL (or an
    apiKey), and this detail is the HTTP body the operator gets back.
    """
    _run_id, trader_id = await _start(ctx)

    async def _boom(_cmd: Any) -> Any:
        raise RuntimeError("signature=ZZLEAK9")

    node = PaperTradingNode.__new__(PaperTradingNode)
    node._config = SimpleNamespace(trader_id=trader_id)  # type: ignore[assignment]
    node._commands = PaperCommandQueue(tmp_db)
    node._cmd_worker = StoreWorker("vq-test-cmd")
    monkeypatch.setattr(node, "_execute_command", _boom)

    stop = asyncio.Event()
    task = asyncio.create_task(_real_node_loop(node, stop))
    try:
        r = await ctx.ac.post("/api/paper/close-all-positions")
    finally:
        stop.set()
        await task
    assert r.status_code == 502, r.text
    detail = r.json()["detail"]
    assert detail["error"] == "RuntimeError: command failed; see the paper node log"
    assert "ZZLEAK9" not in r.text


async def test_halt_and_pause_commands(ctx: Ctx, tmp_db: Path) -> None:
    _run_id, trader_id = await _start(ctx)
    seen: list[str] = []

    def handler(cmd: Any) -> tuple[bool, dict[str, Any], None]:
        seen.append(cmd.command)
        return True, {"state": "halted" if cmd.command == "halt" else "paused"}, None

    stop = asyncio.Event()
    task = asyncio.create_task(_fake_node(tmp_db, trader_id, handler, stop))
    try:
        r1 = await ctx.ac.post("/api/paper/halt")
        r2 = await ctx.ac.post("/api/paper/halt?mode=pause")
    finally:
        stop.set()
        await task
    assert seen == ["halt", "pause"]
    assert r1.json()["state"] == "halted"
    assert r2.json()["state"] == "paused"


async def test_unacknowledged_command_times_out_and_expires(
    ctx: Ctx, tmp_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vibe_quant.api.routers.paper_trading.COMMAND_TIMEOUT_SECS", 0.3)
    _run_id, trader_id = await _start(ctx)
    r = await ctx.ac.post("/api/paper/halt")
    assert r.status_code == 504
    assert "cancelled" in r.json()["detail"]
    q = PaperCommandQueue(tmp_db)
    try:
        assert q.claim_pending(trader_id) == [], "expired command must not run later"
    finally:
        q.close()


async def test_resume_refused_while_killed(ctx: Ctx) -> None:
    await _start(ctx)
    ctx.state.set_kill_switch("ops")
    r = await ctx.ac.post("/api/paper/resume")
    assert r.status_code == 423


async def test_kill_queues_kill_for_active_session(ctx: Ctx, tmp_db: Path) -> None:
    _run_id, trader_id = await _start(ctx)
    r = await ctx.ac.post("/api/system/kill", json={"reason": "divergence"})
    assert r.status_code == 200
    q = PaperCommandQueue(tmp_db)
    try:
        cmds = q.claim_pending(trader_id)
    finally:
        q.close()
    assert [c.command for c in cmds] == ["kill"]
    assert "divergence" in cmds[0].payload["message"]


# ----------------------------------------------------------- read endpoints


async def test_status_reports_node_state_from_checkpoint(ctx: Ctx, tmp_db: Path) -> None:
    run_id, trader_id = await _start(ctx)
    r = await ctx.ac.get("/api/paper/status")
    assert r.json()["state"] == "starting"
    p = StatePersistence(tmp_db, trader_id)
    p.save_checkpoint(
        StateCheckpoint(
            trader_id=trader_id,
            node_status={
                "state": "halted",
                "halt_reason": "max_drawdown",
                "error_message": "drawdown 6.00% >= limit 5.00%",
                "risk": {"drawdown_pct": "0.06"},
            },
        )
    )
    p.close()
    body = (await ctx.ac.get("/api/paper/status")).json()
    assert body["state"] == "halted"
    assert body["halt_reason"] == "max_drawdown"
    assert body["run_id"] == run_id and body["trader_id"] == trader_id
    assert body["pnl_metrics"] == {"drawdown_pct": "0.06"}


async def test_status_exposes_snapshot_errors(ctx: Ctx, tmp_db: Path) -> None:
    """A checkpoint whose position snapshot failed must surface the error list."""
    _run_id, trader_id = await _start(ctx)
    p = StatePersistence(tmp_db, trader_id)
    p.save_checkpoint(
        StateCheckpoint(
            trader_id=trader_id,
            node_status={
                "state": "running",
                "snapshot_errors": ["positions: RuntimeError", "order O-2: KeyError"],
            },
        )
    )
    p.close()
    body = (await ctx.ac.get("/api/paper/status")).json()
    assert body["snapshot_errors"] == ["positions: RuntimeError", "order O-2: KeyError"]


async def test_status_snapshot_errors_empty_when_clean(ctx: Ctx, tmp_db: Path) -> None:
    """Idle, pre-checkpoint and clean-checkpoint responses all carry an empty list."""
    idle = (await ctx.ac.get("/api/paper/status")).json()
    assert idle["state"] == "idle"
    assert idle["snapshot_errors"] == []
    _run_id, trader_id = await _start(ctx)
    assert (await ctx.ac.get("/api/paper/status")).json()["snapshot_errors"] == []
    p = StatePersistence(tmp_db, trader_id)
    p.save_checkpoint(StateCheckpoint(trader_id=trader_id, node_status={"state": "running"}))
    p.close()
    assert (await ctx.ac.get("/api/paper/status")).json()["snapshot_errors"] == []


async def test_status_snapshot_errors_malformed_is_empty(ctx: Ctx, tmp_db: Path) -> None:
    """node_status is untyped JSON: a non-list value must not leak through."""
    _run_id, trader_id = await _start(ctx)
    p = StatePersistence(tmp_db, trader_id)
    p.save_checkpoint(
        StateCheckpoint(
            trader_id=trader_id,
            node_status={"state": "running", "snapshot_errors": "boom"},
        )
    )
    p.close()
    assert (await ctx.ac.get("/api/paper/status")).json()["snapshot_errors"] == []


async def test_positions_and_orders_map_nautilus_to_dict(ctx: Ctx, tmp_db: Path) -> None:
    """Audit: UI showed symbol '' and entry 0 for real Position/Order.to_dict() data."""
    p = StatePersistence(tmp_db, "PAPER-042")
    p.save_checkpoint(
        StateCheckpoint(
            trader_id="PAPER-042",
            positions={
                "BTCUSDT-PERP.BINANCE-S-000": {
                    "position_id": "BTCUSDT-PERP.BINANCE-S-000",
                    "instrument_id": "BTCUSDT-PERP.BINANCE",
                    "side": "LONG",
                    "quantity": "0.010",
                    "avg_px_open": 60000.0,
                    "unrealized_pnl": -12.5,
                    "leverage": 5,
                }
            },
            orders={
                "O-1": {
                    "instrument_id": "BTCUSDT-PERP.BINANCE",
                    "type": "STOP_MARKET",
                    "side": "SELL",
                    "quantity": "0.010",
                    "trigger_price": "59000.0",
                    "status": "ACCEPTED",
                    "is_reduce_only": True,
                }
            },
        )
    )
    p.close()
    pos = (await ctx.ac.get("/api/paper/positions?trader_id=PAPER-042")).json()
    assert pos == [
        {
            "symbol": "BTCUSDT-PERP.BINANCE",
            "direction": "LONG",
            "quantity": 0.01,
            "entry_price": 60000.0,
            "unrealized_pnl": -12.5,
            "leverage": 5.0,
        }
    ]
    orders = (await ctx.ac.get("/api/paper/orders?trader_id=PAPER-042")).json()
    assert orders == [
        {
            "order_id": "O-1",
            "symbol": "BTCUSDT-PERP.BINANCE",
            "side": "SELL",
            "quantity": 0.01,
            "price": 59000.0,
            "status": "ACCEPTED",
        }
    ]


async def test_restore_forces_start_fresh_false(ctx: Ctx) -> None:
    """R22/K4: a prior config's start_fresh:true never re-injects into a restore."""
    cfg_dir = ctx.tmp / "paper_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "paper_1.json").write_text(
        json.dumps(
            {
                "trader_id": "PAPER-777",
                "strategy_id": ctx.strategy_id,
                "symbols": ["BTCUSDT"],
                "binance": {"testnet": True},
                "start_fresh": True,
            }
        )
    )
    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-777"})
    assert r.status_code == 201, r.text
    run_id = r.json()["run_id"]
    run = ctx.state.get_backtest_run(run_id)
    assert run is not None
    assert run["parameters"]["start_fresh"] is False
    assert ctx.config_for(run_id)["start_fresh"] is False


async def test_restore_409_when_the_account_lock_is_held(ctx: Ctx) -> None:
    """rw1g8: /restore honors the same per-account lock as /start."""
    cfg_dir = ctx.tmp / "paper_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "paper_1.json").write_text(
        json.dumps(
            {
                "trader_id": "PAPER-777",
                "strategy_id": ctx.strategy_id,
                "symbols": ["BTCUSDT"],
                "binance": {"testnet": True},
            }
        )
    )
    held = acquire_account_lock(_api_account(), "PAPER-OTHER")
    before = _paper_run_count(ctx.state)
    try:
        r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "PAPER-777"})
    finally:
        held.release()

    assert r.status_code == 409, r.text
    assert f"Paper session already holds account {_api_account()}" in r.json()["detail"]
    assert ctx.launched == []
    assert _paper_run_count(ctx.state) == before


async def test_restore_after_key_rotation_persists_the_curated_refusal(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """6ub67 F1: a /restore whose real child refuses leaves a curated terminal failure.

    The saved trader traded LIVE on another key; the env key resolves to an
    account with no checkpoint, so the REAL CLI child refuses. The run and the job
    must end ``failed`` carrying the curated text (not stay ``running``, not be
    reduced to "Process terminated unexpectedly" by a later sync).
    """
    from pathlib import Path as _Path

    monkeypatch.setenv("BINANCE_API_KEY", "rotated-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "rotated-secret")
    monkeypatch.setenv("PYTHONPATH", str(_Path(__file__).resolve().parents[2]))
    # Undo the fixture's fake launcher: this test spawns the real CLI.
    monkeypatch.setattr(
        ctx.jobs, "start_job", functools.partial(BacktestJobManager.start_job, ctx.jobs)
    )
    _seed_account_checkpoint(ctx.tmp / "api.db", api_key="old-key", testnet=False)
    cfg_dir = ctx.tmp / "paper_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "paper_1.json").write_text(
        json.dumps(
            {
                "trader_id": "LIVE-001",
                "strategy_id": ctx.strategy_id,
                "symbols": ["BTCUSDT"],
                "binance": {"testnet": False},
            }
        )
    )

    r = await ctx.ac.post(
        "/api/paper/restore", json={"trader_id": "LIVE-001", "confirm_live": True}
    )
    assert r.status_code == 201, r.text
    run_id = r.json()["run_id"]

    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        run = ctx.state.get_backtest_run(run_id)
        assert run is not None
        if run["status"] != "running":
            break
        await asyncio.sleep(0.25)

    curated = (
        "Refusing LIVE start: no saved risk state for this account (new or rotated "
        "API key, or first live session): drawdown, daily-loss and losing-streak "
        "limits would start from zero; re-run with --confirm-fresh-live to confirm"
    )
    run = ctx.state.get_backtest_run(run_id)
    assert run is not None
    assert (run["status"], run["error_message"]) == ("failed", curated)
    job = ctx.jobs.conn.execute(
        "SELECT status, error_message FROM background_jobs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert (job["status"], job["error_message"]) == ("failed", curated)
    # A later sync keeps the curated text (it only acts on a still-running job).
    ctx.jobs.sync_job_status(run_id)
    run = ctx.state.get_backtest_run(run_id)
    assert run is not None and run["error_message"] == curated


async def test_restore_live_session_requires_confirmation(ctx: Ctx) -> None:
    cfg_dir = ctx.tmp / "paper_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "paper_1.json").write_text(
        json.dumps(
            {
                "trader_id": "LIVE-001",
                "strategy_id": ctx.strategy_id,
                "symbols": ["BTCUSDT"],
                "binance": {"testnet": False},
            }
        )
    )
    r = await ctx.ac.post("/api/paper/restore", json={"trader_id": "LIVE-001"})
    assert r.status_code == 400 and "confirm_live" in r.text
    assert ctx.launched == []


# ------------------------------------------------------------------- /stop


def _register_paper_job(ctx: Ctx, pid: int, identity: str | None) -> None:
    ctx.jobs.conn.execute(
        """INSERT OR REPLACE INTO background_jobs
           (run_id, pid, pid_start_time, job_type, status, started_at, heartbeat_at)
           VALUES (?, ?, ?, 'paper', 'running', datetime('now'), datetime('now'))""",
        (ctx.validation_run + 1000, pid, identity),
    )
    ctx.jobs.conn.commit()


async def test_stop_never_signals_a_recycled_pid(ctx: Ctx) -> None:
    """Stored PID now owned by an unrelated process -> /stop leaves it alone."""
    import subprocess

    victim = subprocess.Popen(["sleep", "30"])
    try:
        _register_paper_job(ctx, victim.pid, identity="not-this-process")
        r = await ctx.ac.post("/api/paper/stop")
        assert r.status_code == 200, r.text
        await asyncio.sleep(0.2)
        assert victim.poll() is None, "unrelated process was killed via a recycled PID"
    finally:
        victim.kill()
        victim.wait()


async def test_stop_terminates_the_owned_node_process(ctx: Ctx) -> None:
    import subprocess

    from vibe_quant.jobs.manager import process_start_time

    node = subprocess.Popen(["sleep", "30"])
    try:
        _register_paper_job(ctx, node.pid, identity=process_start_time(node.pid))
        r = await ctx.ac.post("/api/paper/stop")
        assert r.status_code == 200, r.text
        assert node.wait(timeout=5) == -15  # SIGTERM
    finally:
        if node.poll() is None:
            node.kill()
            node.wait()
