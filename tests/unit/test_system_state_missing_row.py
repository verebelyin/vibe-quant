"""A missing ``system_state`` row fails closed (bead vibe-quant-lickf, task F2).

``init_schema`` seeds the singleton row with
``INSERT OR IGNORE INTO system_state (id, kill_switch) VALUES (1, 0)``, so a
missing row id=1 means someone deleted it or the DB is broken. Before this
change every reader got the defensive defaults dict (``kill_switch: False``),
so a deleted row silently disabled the global kill switch everywhere. The paper
node and the paper API now read the row with ``require_row=True`` and treat a
missing row as an unreadable switch; ``/api/system/status`` keeps returning the
defaults (unchanged).

Contract rows:

* L1 ``get_system_state(require_row=True)`` with no row raises
      ``SystemStateMissingError`` (a ``RuntimeError``) with the exact message;
      the default ``require_row=False`` read still returns the defaults dict and
      a present row reads live.
* L2 ``PaperTradingNode._kill_switch`` counts a missing row as a failed read
      (engaged for that check, streak +1, class-only log); three in a row halt
      ERROR with the same fail-closed message as s40h9.
* L3 startup and resume refuse with the existing unreadable-read texts and the
      exception class ``SystemStateMissingError``.
* L4 ``/api/paper/start|restore|resume`` refuse with HTTP 503 and the exact
      detail, creating no run row and no job.
* L5 a present row is unchanged everywhere.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from tests.unit.test_audit_paper_node import _config, _FakeTradingNode
from tests.unit.test_paper_kill_switch_fail_closed import _RecordGuard
from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager, SystemStateMissingError
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.paper.config import ConfigurationError
from vibe_quant.paper.guard import HaltReason
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path
    from typing import Any

DSL = {
    "name": "test_strategy",
    "timeframe": "1h",
    "indicators": {"rsi_14": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi_14 < 30"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}

_MISSING = "system_state row id=1 is missing"
_REFUSAL = "System kill switch state unavailable (system_state row missing); refusing"


def _drop_state_row(state: StateManager) -> None:
    """Delete the singleton row through ``state``'s own connection.

    ``init_schema`` re-inserts the row on every fresh connection, so the delete
    must happen after the connection that will serve the read is open (and is
    asserted to have taken effect).
    """
    _ = state.conn  # open + init schema (re-inserts the row) before deleting it
    state.conn.execute("DELETE FROM system_state")
    state.conn.commit()
    count = state.conn.execute("SELECT COUNT(*) FROM system_state").fetchone()[0]
    assert count == 0, "the row must really be gone for the read under test"


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    """A strategy plus a completed validation run (same shape as the s40h9 tests)."""
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


@pytest.fixture()
def fake_tradingnode(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTradingNode]:
    """Redirect NT's TradingNode to the recording fake (copy of the audit fixture)."""
    import nautilus_trader.live.node as live_node

    _FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", _FakeTradingNode)
    return _FakeTradingNode


# --------------------------------------------------------------- L1


def test_require_row_raises_when_row_missing(tmp_path: Path) -> None:
    """L1: ``require_row=True`` with no row raises the dedicated error, whose
    message is the exact one the paper API/ node texts are built around."""
    sm = StateManager(tmp_path / "state.db")
    try:
        _drop_state_row(sm)
        with pytest.raises(SystemStateMissingError) as exc_info:
            sm.get_system_state(require_row=True)
        assert isinstance(exc_info.value, RuntimeError)
        assert str(exc_info.value) == _MISSING
    finally:
        sm.close()


def test_default_read_keeps_defaults_when_row_missing(tmp_path: Path) -> None:
    """L1: the default read (``require_row=False``, what /api/system/status
    calls) still answers with the defaults dict instead of raising."""
    sm = StateManager(tmp_path / "state.db")
    try:
        _drop_state_row(sm)
        assert sm.get_system_state() == {
            "kill_switch": False,
            "reason": None,
            "killed_at": None,
            "killed_by": None,
            "updated_at": None,
        }
    finally:
        sm.close()


def test_require_row_reads_live_row_when_present(tmp_path: Path) -> None:
    """L1: with the row present ``require_row=True`` is a plain read (and must
    not be a synonym for 'raise')."""
    sm = StateManager(tmp_path / "state.db")
    try:
        assert sm.get_system_state(require_row=True)["kill_switch"] is False
        sm.set_kill_switch("ops", "tester")
        state = sm.get_system_state(require_row=True)
        assert state["kill_switch"] is True
        assert state["reason"] == "ops"
        assert state["killed_by"] == "tester"
    finally:
        sm.close()


# --------------------------------------------------------------- L2


async def test_node_treats_missing_row_as_failed_read(
    seeded: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """L2: a missing row is a failed read - engaged, streak +1, class-only log."""
    guard = _RecordGuard()
    node = PaperTradingNode(_config(seeded))
    node._guard = guard  # type: ignore[assignment]
    _drop_state_row(node._state_manager)
    try:
        with caplog.at_level(logging.ERROR):
            engaged, reason = node._kill_switch()
        assert engaged is True
        assert reason == "kill switch state unreadable (SystemStateMissingError)"
        assert node._kill_read_failures == 1
        assert node._kill_read_error == "SystemStateMissingError"
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert errors[0].getMessage() == (
            "could not read system kill switch (SystemStateMissingError); 1 consecutive failure(s)"
        )
        assert errors[0].exc_info is None
        assert _MISSING not in caplog.text, "the log stays class-only"
        assert guard.calls == []
    finally:
        await node._shutdown()


async def test_node_three_missing_row_reads_halt_error(seeded: dict[str, Any]) -> None:
    """L2: three consecutive missing-row reads halt ERROR with the s40h9 text."""
    guard = _RecordGuard()
    node = PaperTradingNode(_config(seeded))
    node._guard = guard  # type: ignore[assignment]
    _drop_state_row(node._state_manager)
    try:
        for _ in range(3):
            node._check_kill_switch()
        assert guard.calls == [("halt", HaltReason.ERROR)]
        assert guard.messages == [
            "system kill switch unreadable for 3 consecutive reads"
            " (SystemStateMissingError); failing closed"
        ]
    finally:
        await node._shutdown()


# --------------------------------------------------------------- L3


async def test_startup_refuses_when_row_missing(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """L3: a missing row at _initialize refuses like any unreadable read, before
    any TradingNode is built."""
    node = PaperTradingNode(_config(seeded))
    _drop_state_row(node._state_manager)
    try:
        with pytest.raises(ConfigurationError) as exc_info:
            await node._initialize()
        assert str(exc_info.value) == (
            "System kill switch state unreadable (SystemStateMissingError); refusing to start"
        )
        assert fake_tradingnode.instances == []
    finally:
        await node._shutdown()


async def test_resume_refused_when_row_missing(seeded: dict[str, Any]) -> None:
    """L3: resume with a missing row is refused with the unreadable-read text and
    never reaches the guard."""
    guard = _RecordGuard()
    node = PaperTradingNode(_config(seeded))
    node._guard = guard  # type: ignore[assignment]
    _drop_state_row(node._state_manager)
    try:
        assert node.resume() == (
            False,
            "system kill switch state unreadable (SystemStateMissingError);"
            " cannot resume until it reads cleanly",
        )
        assert guard.calls == []
    finally:
        await node._shutdown()


# --------------------------------------------------------------- L4 (API)


class _ApiCtx:
    def __init__(
        self, ac: AsyncClient, state: StateManager, jobs: BacktestJobManager, tmp: Path
    ) -> None:
        self.ac = ac
        self.state = state
        self.jobs = jobs
        self.tmp = tmp
        self.launched: list[list[str]] = []
        self.strategy_id = state.create_strategy("test_strategy", DSL)
        self.run_id = state.create_backtest_run(
            strategy_id=self.strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters={"rsi_14_period": 21},
        )
        state.update_backtest_run_status(self.run_id, "completed")

    def run_count(self) -> int:
        row = self.state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()
        return int(row[0])


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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


@pytest.fixture()
async def api(
    api_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[_ApiCtx]:
    app = create_app()
    state = StateManager(db_path=api_env)
    _ = state.conn  # init schema
    jobs = BacktestJobManager(db_path=api_env)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        c = _ApiCtx(ac, state, jobs, tmp_path)

        def fake_start_job(
            run_id: int, job_type: str, command: list[str], log_file: str | None = None
        ) -> int:
            c.launched.append(command)
            return 4242

        monkeypatch.setattr(jobs, "start_job", fake_start_job)
        yield c
    await ws.stop()
    jobs.close()
    state.close()


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/paper/start", {"validation_run_id": 1}),
        ("/api/paper/restore", {"trader_id": "PAPER-001"}),
        ("/api/paper/resume", None),
    ],
)
async def test_paper_control_refused_when_row_missing(
    api: _ApiCtx, path: str, body: dict[str, Any] | None
) -> None:
    """L4: start/restore/resume all fail closed with 503 - no run row, no job.

    ``validation_run_id`` is patched to this fixture's run because it is created
    per-test; the refusal must fire before any run lookup anyway.
    """
    body = dict(body or {})
    if "validation_run_id" in body:
        body["validation_run_id"] = api.run_id
    _drop_state_row(api.state)
    before = api.run_count()

    r = await api.ac.post(path, json=body)

    assert r.status_code == 503, r.text
    assert r.json()["detail"] == _REFUSAL
    assert api.run_count() == before, "the refusal must not create a backtest run row"
    assert api.launched == [], "the refusal must not start a job"


# --------------------------------------------------------------- L5


async def test_present_row_unchanged(api: _ApiCtx) -> None:
    """L5: with the row present the paper API still starts (and the default
    status read still answers) - the missing-row path must not touch it."""
    r = await api.ac.post("/api/paper/start", json={"validation_run_id": api.run_id})
    assert r.status_code == 201, r.text
    assert len(api.launched) == 1
    status = await api.ac.get("/api/system/status")
    assert status.status_code == 200
    assert status.json()["kill_switch"] is False


async def test_node_read_unchanged_when_row_present(seeded: dict[str, Any]) -> None:
    """L5: a present, unkilled row is a clean read for the node (no failure)."""
    guard = _RecordGuard()
    node = PaperTradingNode(_config(seeded))
    node._guard = guard  # type: ignore[assignment]
    try:
        assert node._kill_switch() == (False, None)
        assert node._kill_read_failures == 0
        assert node._kill_read_error is None
        assert guard.calls == []
    finally:
        await node._shutdown()
