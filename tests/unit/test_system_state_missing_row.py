"""A missing ``system_state`` row is re-created ENGAGED (bead vibe-quant-tdjop, task T1).

This file used to pin the previous fail-closed read (a missing row raised
``SystemStateMissingError`` from ``require_row=True`` reads, and the node/API
texts said "state unavailable"/"unreadable"). Task T1 replaced that rule (design
20261016-safety-bankroll, "Existing assertions this contract changes"): a deleted
kill-switch record is now re-created ENGAGED on the spot, so every reader sees an
engaged switch instead of a failed read.

The row-level coverage of the new rule lives in
``tests/unit/test_kill_switch_row.py`` (rows R1-R8); this file keeps the
end-to-end coverage: what the paper NODE and the paper API do on a missing row.

Contract rows:

* L1 a read on a deleted row (cached conn, ``require_row`` either way) re-seeds
      it engaged and returns the reseed reason; a present row reads live.
* L2 ``PaperTradingNode._kill_switch`` reads the re-seeded row as engaged
      (no failure streak, no class-only ERROR log) and ``_check_kill_switch``
      halts KILL_SWITCH once with the reseed reason.
* L3 startup refuses with the engaged text (never the "unreadable" one) and
      resume is refused because the switch is engaged.
* L4 ``/api/paper/start|restore|resume`` refuse with HTTP 423 and the engaged
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
from vibe_quant.db.schema import SYSTEM_STATE_RESEEDED_REASON
from vibe_quant.db.state_manager import StateManager
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

_REASON = SYSTEM_STATE_RESEEDED_REASON
#: API refusal detail (``_refuse_if_killed``).
_ENGAGED = f"System kill switch engaged: {_REASON}"
#: Startup refusal text (``PaperTradingNode._initialize``).
_ENGAGED_START = f"System kill switch engaged ({_REASON}); refusing to start"


def _drop_state_row(state: StateManager) -> None:
    """Delete the singleton row through ``state``'s own connection.

    The delete must happen after the connection that will serve the read is open
    (and is asserted to have taken effect), so the read under test really sees a
    missing row.
    """
    _ = state.conn  # open + init schema before deleting the row
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


def test_cached_conn_read_reseeds_engaged(tmp_path: Path) -> None:
    """L1: the read on a deleted row re-creates it engaged and returns the
    reseed reason - it never raises and never answers "not killed"."""
    sm = StateManager(tmp_path / "state.db")
    try:
        _drop_state_row(sm)
        state = sm.get_system_state(require_row=True)
        row = sm.conn.execute(
            "SELECT id, kill_switch, reason, killed_at, killed_by FROM system_state"
        ).fetchone()
        assert row is not None, "the read must have re-created the row"
        again = sm.get_system_state(require_row=True)
    finally:
        sm.close()
    assert state["kill_switch"] is True
    assert state["reason"] == _REASON
    assert (row[0], row[1], row[2], row[4]) == (1, 1, _REASON, "system")
    assert row[3] is not None
    assert again["kill_switch"] is True, "the row stays engaged after the read"


def test_default_read_reseeds_engaged_never_defaults(tmp_path: Path) -> None:
    """L1: the no-arg read (what /api/system/status calls) also re-creates the
    row engaged - the old disengaged defaults dict is gone."""
    sm = StateManager(tmp_path / "state.db")
    try:
        _drop_state_row(sm)
        state = sm.get_system_state(require_row=False)
    finally:
        sm.close()
    assert state["kill_switch"] is True
    assert state["reason"] == _REASON


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


async def test_node_reads_reseeded_engaged(
    seeded: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """L2: a deleted row is a clean engaged read for the node - engaged with the
    reseed reason, no failure streak, no ERROR log, no guard call."""
    guard = _RecordGuard()
    node = PaperTradingNode(_config(seeded))
    node._guard = guard  # type: ignore[assignment]
    _drop_state_row(node._state_manager)
    try:
        with caplog.at_level(logging.ERROR):
            engaged, reason = node._kill_switch()
        assert engaged is True
        assert reason == _REASON
        assert node._kill_read_failures == 0, "a re-seeded engaged row is not a failed read"
        assert node._kill_read_error is None
        assert guard.calls == []
        assert [r for r in caplog.records if r.levelno == logging.ERROR] == []
        assert any(r.levelno == logging.CRITICAL for r in caplog.records), (
            "the re-seed is recorded at CRITICAL"
        )
    finally:
        await node._shutdown()


async def test_node_missing_row_halts_kill_switch_once(seeded: dict[str, Any]) -> None:
    """L2: repeated checks on the re-seeded row halt KILL_SWITCH exactly once
    (the latch), with the reseed reason - never the 'unreadable' ERROR halt."""
    guard = _RecordGuard()
    node = PaperTradingNode(_config(seeded))
    node._guard = guard  # type: ignore[assignment]
    _drop_state_row(node._state_manager)
    try:
        for _ in range(3):
            node._check_kill_switch()
        assert guard.calls == [("halt", HaltReason.KILL_SWITCH)]
        assert guard.messages == [f"system kill switch: {_REASON}"]
    finally:
        await node._shutdown()


# --------------------------------------------------------------- L3


async def test_startup_refuses_when_row_missing(
    seeded: dict[str, Any], fake_tradingnode: type[_FakeTradingNode]
) -> None:
    """L3: a deleted row at _initialize refuses like any engaged switch (the
    reason names the re-seed), before any TradingNode is built."""
    node = PaperTradingNode(_config(seeded))
    _drop_state_row(node._state_manager)
    try:
        with pytest.raises(ConfigurationError) as exc_info:
            await node._initialize()
        assert str(exc_info.value) == _ENGAGED_START
        assert fake_tradingnode.instances == []
    finally:
        await node._shutdown()


async def test_resume_refused_when_row_missing(seeded: dict[str, Any]) -> None:
    """L3: resume with a deleted row is refused because the row is engaged, and
    the refusal reaches the guard as ``kill_switch_engaged=True``."""
    guard = _RecordGuard()
    node = PaperTradingNode(_config(seeded))
    node._guard = guard  # type: ignore[assignment]
    _drop_state_row(node._state_manager)
    try:
        assert node.resume() == (
            False,
            "system kill switch is engaged; unlock it before resuming",
        )
        assert guard.calls == [("resume", True)]
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
    """L4: start/restore/resume all refuse with 423 and the re-seed reason - no
    run row, no job.

    ``validation_run_id`` is patched to this fixture's run because it is created
    per-test; the refusal must fire before any run lookup anyway.
    """
    body = dict(body or {})
    if "validation_run_id" in body:
        body["validation_run_id"] = api.run_id
    _drop_state_row(api.state)
    before = api.run_count()

    r = await api.ac.post(path, json=body)

    assert r.status_code == 423, r.text
    assert r.json()["detail"] == _ENGAGED
    assert api.run_count() == before, "the refusal must not create a backtest run row"
    assert api.launched == [], "the refusal must not start a job"


# --------------------------------------------------------------- L5


async def test_present_row_unchanged(api: _ApiCtx) -> None:
    """L5: with the row present the paper API still starts (and the status read
    still answers) - the missing-row path must not touch it."""
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
