"""Telegram start gate (bead vibe-quant-u2r9r, user decisions u2r9r=A, D3=A).

A *configured* Telegram that does not work must refuse the start on every path
(node and API) **before** any trading node exists; an *unconfigured* Telegram
starts as today. The probe is one silent ``sendChatAction`` ("typing…"), which --
unlike ``getMe`` -- also fails on a wrong chat id and on a bot the user blocked.

Contract rows T1..T21 of the design file (data/swarm/20261017-power-paper).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

import vibe_quant.api.routers.paper_trading as paper_router
from tests.unit.test_audit_paper_node import DSL, _config, _FakeTradingNode
from vibe_quant.alerts import telegram as telegram_mod
from vibe_quant.alerts.telegram import (
    ENV_TELEGRAM_BOT_TOKEN,
    ENV_TELEGRAM_CHAT_ID,
    FAIL_MSG,
    HALF_MSG,
    TelegramConfig,
    TelegramUnavailableError,
    require_working_telegram,
    verify_telegram,
)
from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.paper.config import BinanceTestnetConfig, ConfigurationError
from vibe_quant.paper.node import PaperTradingNode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

TOKEN = "123:ZZTOKEN9"
CHAT = "42"
LEAK = "ZZLEAK9"

_401_BODY = {"ok": False, "error_code": 401, "description": "Unauthorized"}


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def seeded(tmp_path: Path) -> dict[str, Any]:
    """A strategy plus a completed validation run (shape shared with the node tests)."""
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
    """Redirect NT's ``TradingNode`` to the recording fake (copy of the audit fixture)."""
    import nautilus_trader.live.node as live_node

    _FakeTradingNode.instances = []
    monkeypatch.setattr(live_node, "TradingNode", _FakeTradingNode)
    return _FakeTradingNode


class _Api:
    """Minimal /api/paper harness (shape copied from ``test_audit_paper_api``)."""

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

    def paper_runs(self) -> int:
        return self.state.conn.execute(
            "SELECT COUNT(*) FROM backtest_runs WHERE run_mode = 'paper'"
        ).fetchone()[0]

    def activate(self, run_id: int) -> None:
        self.jobs.conn.execute(
            """INSERT OR REPLACE INTO background_jobs
               (run_id, pid, job_type, status, started_at, heartbeat_at)
               VALUES (?, 4242, 'paper', 'running', datetime('now'), datetime('now'))""",
            (run_id,),
        )
        self.jobs.conn.commit()


@pytest.fixture()
async def api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[_Api]:
    db = tmp_path / "api.db"
    monkeypatch.setattr("vibe_quant.db.connection.DEFAULT_DB_PATH", db)
    monkeypatch.setattr(
        "vibe_quant.api.routers.paper_trading.PAPER_CONFIG_DIR", tmp_path / "paper_configs"
    )
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
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        ctx = _Api(ac, state, jobs, tmp_path)

        def fake_start_job(
            run_id: int, job_type: str, command: list[str], log_file: str | None = None
        ) -> int:
            ctx.launched.append(command)
            ctx.activate(run_id)
            return 4242

        monkeypatch.setattr(jobs, "start_job", fake_start_job)
        yield ctx
    await ws.stop()
    jobs.close()
    state.close()


# ------------------------------------------------------------------- helpers


def _configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_TELEGRAM_BOT_TOKEN, TOKEN)
    monkeypatch.setenv(ENV_TELEGRAM_CHAT_ID, CHAT)


def _install_probe(
    monkeypatch: pytest.MonkeyPatch, handler: Any
) -> list[httpx.AsyncClient]:
    """Patch the probe-client seam with a ``MockTransport``; return the recorded clients."""
    clients: list[httpx.AsyncClient] = []

    def _make(timeout_s: float) -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), timeout=timeout_s + 1.0
        )
        clients.append(client)
        return client

    monkeypatch.setattr(telegram_mod, "_make_probe_client", _make)
    return clients


def _ok(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": True})


def _resp(status: int, body: Any) -> Any:
    def _handler(_request: httpx.Request) -> httpx.Response:
        if isinstance(body, dict):
            return httpx.Response(status, json=body)
        return httpx.Response(status, text=str(body))

    return _handler


def _expired(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(401, json=_401_BODY)


def _transport_error(_request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError(f"boom {LEAK} https://api.telegram.org/bot{TOKEN}/x")


def _reason_of(exc: BaseException) -> str:
    """The reason substring of a refusal message (``...not working ({reason}); refusing``)."""
    match = re.search(r"not working \((.*)\); refusing", str(exc), re.DOTALL)
    assert match, str(exc)
    return match.group(1)


def _probe_cfg() -> TelegramConfig:
    return TelegramConfig(bot_token=TOKEN, chat_id=CHAT)


# ------------------------------------------------------------------ T6 / T14


async def test_expired_token_refuses_node_start_before_trading_node(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T6/T14: an expired token refuses ``node.start()`` before any trading node exists."""
    _configured(monkeypatch)
    _install_probe(monkeypatch, _expired)
    persisted: list[int] = []

    class _SpyPersistence:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            persisted.append(1)

    monkeypatch.setattr("vibe_quant.paper.node.StatePersistence", _SpyPersistence)
    node = PaperTradingNode(_config(seeded))
    built: list[int] = []
    monkeypatch.setattr(node, "_create_live_trading_node", lambda: built.append(1))

    with caplog.at_level(logging.INFO), pytest.raises(ConfigurationError) as excinfo:
        await node.start()

    exc = excinfo.value
    expected = FAIL_MSG.format(reason="HTTP 401 Unauthorized")
    assert str(exc) == expected
    assert built == [], "the gate must run before the trading node is built"
    assert persisted == [], "the gate must run before StatePersistence is constructed"
    assert fake_tradingnode.instances == []
    assert TOKEN not in str(exc)
    assert TOKEN not in repr(exc)
    assert TOKEN not in caplog.text
    assert TOKEN not in (node.status.error_message or "")

    log_file = seeded["tmp"] / "logs" / "PAPER-007.jsonl"
    text = log_file.read_text()
    assert TOKEN not in text
    log_error = [
        line["data"]
        for line in (json.loads(raw) for raw in text.splitlines())
        if line.get("data", {}).get("action") == "error"
    ]
    assert log_error == [{"action": "error", "error": expected, "passed": False}]


# --------------------------------------------------------- class-only logging


async def test_refusal_logs_no_exception_chain(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The refusal is logged class-only: no ``exc_info`` and no chained-exception text.

    The probe's httpx error is CHAINED to an inner marker error here, so a
    ``logger.exception`` (or an un-suppressed ``raise ... from exc``) anywhere on
    the refusal path would print both markers (cc-sh-p1-1).
    """
    _configured(monkeypatch)

    def _chained(_request: httpx.Request) -> httpx.Response:
        try:
            raise ValueError(f"CHAIN_MARKER {LEAK}")
        except ValueError as cause:
            raise httpx.ConnectError(f"CHAIN_MARKER {TOKEN}") from cause

    _install_probe(monkeypatch, _chained)
    node = PaperTradingNode(_config(seeded))
    try:
        with caplog.at_level(logging.DEBUG), pytest.raises(ConfigurationError) as excinfo:
            await node._initialize()
    finally:
        await node._shutdown()

    exc = excinfo.value
    assert exc.__cause__ is None and exc.__suppress_context__
    assert "CHAIN_MARKER" not in caplog.text
    assert TOKEN not in caplog.text and LEAK not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


# ------------------------------------------------------- T7..T10 (the probe)


@pytest.mark.parametrize(
    ("status", "body", "reason"),
    [
        (401, _401_BODY, "HTTP 401 Unauthorized"),
        (400, {"description": "Bad Request: chat not found"}, "HTTP 400 Bad Request: chat not found"),
        (
            403,
            {"description": "Forbidden: bot was blocked by the user"},
            "HTTP 403 Forbidden: bot was blocked by the user",
        ),
        (200, {"ok": False}, "HTTP 200 ok=false"),
        (200, "<html>", "HTTP 200 invalid body"),
        (500, {"description": "Internal"}, "HTTP 500 Internal"),
        (201, {"ok": True, "description": "Created"}, "HTTP 201 Created"),
        (201, {"ok": True}, "HTTP 201 invalid body"),
        (204, {"ok": True}, "HTTP 204 invalid body"),
    ],
)
async def test_probe_http_rows(
    status: int, body: Any, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7-T10: the failure reason is ``HTTP <status> <description>``, no retry."""
    clients = _install_probe(monkeypatch, _resp(status, body))

    with pytest.raises(TelegramUnavailableError) as excinfo:
        await verify_telegram(_probe_cfg())

    assert _reason_of(excinfo.value) == reason
    assert len(clients) == 1


@pytest.mark.parametrize("token", [TOKEN + " ", " " + TOKEN, "123:ZZQQK EN99ABCDE"], ids=["trailing", "leading", "inner"])
async def test_whitespace_token_still_redacted(
    token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token with whitespace is redacted raw, stripped and per long part."""
    long_part = max(token.split(), key=len)
    description = f"echo [{token}] [{token.strip()}] [{long_part}] tail"
    _install_probe(monkeypatch, _resp(401, {"description": description}))
    cfg = TelegramConfig(bot_token=token, chat_id=CHAT)

    with pytest.raises(TelegramUnavailableError) as excinfo:
        await verify_telegram(cfg)

    reason = _reason_of(excinfo.value)
    assert long_part not in str(excinfo.value)
    assert token.strip() not in str(excinfo.value)
    assert reason == "HTTP 401 echo [<redacted>] [<redacted>] [<redacted>] tail"


# ------------------------------------------------------------ T11 / T13 probe


async def test_transport_error_class_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """T11: an httpx transport error is reported by class name only (its str has the URL)."""
    _install_probe(monkeypatch, _transport_error)

    with pytest.raises(TelegramUnavailableError) as excinfo:
        await verify_telegram(_probe_cfg())

    assert _reason_of(excinfo.value) == "ConnectError"
    assert LEAK not in str(excinfo.value)
    assert TOKEN not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__


# -------------------------------------------------------------------- T12


async def test_probe_timeout_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """T12: ``asyncio.timeout`` governs (httpx's own timeout is 1 s higher)."""

    async def _slow(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, json={"ok": True})

    _install_probe(monkeypatch, _slow)

    start = time.monotonic()
    with pytest.raises(TelegramUnavailableError) as excinfo:
        await verify_telegram(_probe_cfg(), timeout_s=0.05)
    elapsed = time.monotonic() - start

    assert _reason_of(excinfo.value) == "TimeoutError"
    assert elapsed < 1.0, f"asyncio.timeout(0.05) must fire well before httpx's {0.05 + 1.0}s"


# -------------------------------------------------------------------- T19


async def test_description_redacted_and_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """T19: the literal token and the ``/bot<token>`` form are redacted; reason cut to 200."""
    description = f"see /bot{TOKEN}/x and {TOKEN} " + "a" * 500
    _install_probe(monkeypatch, _resp(401, {"description": description}))

    with pytest.raises(TelegramUnavailableError) as excinfo:
        await verify_telegram(_probe_cfg())

    reason = _reason_of(excinfo.value)
    assert TOKEN not in str(excinfo.value)
    assert reason.count("<redacted>") == 2
    assert len(reason) == 200


# -------------------------------------------------------------------- T21


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(_ok, id="success"),
        pytest.param(_expired, id="401"),
        pytest.param(_transport_error, id="connect-error"),
    ],
)
async def test_probe_client_closed(handler: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """T21: the probe client is closed on success AND on every failure."""
    clients = _install_probe(monkeypatch, handler)

    with contextlib.suppress(TelegramUnavailableError):
        await verify_telegram(_probe_cfg())

    assert len(clients) == 1
    assert clients[0].is_closed is True


async def test_probe_client_closed_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """T21 (timeout input): the client is closed when ``asyncio.timeout`` fires."""

    async def _slow(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(10)
        return httpx.Response(200, json={"ok": True})

    clients = _install_probe(monkeypatch, _slow)

    with pytest.raises(TelegramUnavailableError):
        await verify_telegram(_probe_cfg(), timeout_s=0.05)

    assert len(clients) == 1
    assert clients[0].is_closed is True


# -------------------------------------------------------------------- T1


@pytest.mark.parametrize("env", ["unset", "empty"])
async def test_unconfigured_starts_as_today(
    env: str,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    api: _Api,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T1: neither var set (or both ``""``) -> the node starts, the API returns 201."""
    if env == "empty":
        monkeypatch.setenv(ENV_TELEGRAM_BOT_TOKEN, "")
        monkeypatch.setenv(ENV_TELEGRAM_CHAT_ID, "")
    clients = _install_probe(monkeypatch, _ok)
    node = PaperTradingNode(_config(seeded))
    with caplog.at_level(logging.INFO):
        await node._initialize()
    try:
        assert node._journal.telegram is None
        assert any(
            "Telegram alerts disabled (TELEGRAM_BOT_TOKEN/CHAT_ID not set)" in r.getMessage()
            for r in caplog.records
        )
        assert fake_tradingnode.instances, "the session really started"
    finally:
        await node._shutdown()
    assert clients == [], "an unconfigured Telegram must not probe"

    assert await require_working_telegram() is None
    assert clients == [], "an unconfigured Telegram must not probe"

    res = await api.ac.post("/api/paper/start", json={"validation_run_id": api.validation_run})
    assert res.status_code == 201, res.text
    assert clients == []


async def test_space_only_token_is_treated_as_set(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hostile probe: a token of ``" "`` counts as set (only ``""`` is unset), so it refuses."""
    monkeypatch.setenv(ENV_TELEGRAM_BOT_TOKEN, " ")
    monkeypatch.setenv(ENV_TELEGRAM_CHAT_ID, CHAT)
    clients = _install_probe(monkeypatch, _resp(404, {"description": "Not Found"}))

    node = PaperTradingNode(_config(seeded))
    try:
        with pytest.raises(ConfigurationError) as excinfo:
            await node._initialize()
    finally:
        await node._shutdown()

    assert str(excinfo.value) == FAIL_MSG.format(reason="HTTP 404 Not Found")
    assert len(clients) == 1


# ------------------------------------------------------------- T2 / T3


@pytest.mark.parametrize(
    ("env", "var"),
    [
        ("token-only", ENV_TELEGRAM_CHAT_ID),
        ("chat-only", ENV_TELEGRAM_BOT_TOKEN),
    ],
)
async def test_half_configured_refuses(
    env: str,
    var: str,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    api: _Api,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2/T3: one var set -> HALF_MSG on both paths, 0 HTTP requests, no run row."""
    if env == "token-only":
        monkeypatch.setenv(ENV_TELEGRAM_BOT_TOKEN, TOKEN)
    else:
        monkeypatch.setenv(ENV_TELEGRAM_CHAT_ID, CHAT)
    clients = _install_probe(monkeypatch, _ok)
    expected = HALF_MSG.format(VAR=var)

    node = PaperTradingNode(_config(seeded))
    built: list[int] = []
    monkeypatch.setattr(node, "_create_live_trading_node", lambda: built.append(1))
    try:
        with pytest.raises(ConfigurationError) as excinfo:
            await node._initialize()
        assert str(excinfo.value) == expected
        assert built == []
    finally:
        await node._shutdown()

    runs_before = api.paper_runs()
    res = await api.ac.post("/api/paper/start", json={"validation_run_id": api.validation_run})
    assert res.status_code == 503, res.text
    assert res.json()["detail"] == expected
    assert api.paper_runs() == runs_before
    assert api.launched == []
    assert clients == [], "a half-configured Telegram must not probe"
    assert fake_tradingnode.instances == []


# -------------------------------------------------------------------- T5


async def test_working_telegram_starts(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T5: a working bot gives one POST to ``sendChatAction`` with the typing payload."""
    _configured(monkeypatch)
    seen: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True, "result": True})

    _install_probe(monkeypatch, _handler)
    node = PaperTradingNode(_config(seeded))
    try:
        await node._initialize()
        assert node._journal.telegram is not None
        assert node._journal.telegram.config.chat_id == CHAT
        assert node._journal.telegram.config.bot_token == TOKEN
    finally:
        await node._shutdown()

    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert str(seen[0].url) == f"https://api.telegram.org/bot{TOKEN}/sendChatAction"
    assert json.loads(seen[0].content) == {"chat_id": CHAT, "action": "typing"}


# -------------------------------------------------------------------- T15


async def test_live_refuses_too(
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T15: a live config refuses exactly like a testnet one."""
    _configured(monkeypatch)
    _install_probe(monkeypatch, _expired)
    cfg = _config(seeded, binance=BinanceTestnetConfig("key", "secret", testnet=False))

    node = PaperTradingNode(cfg)
    try:
        with pytest.raises(ConfigurationError) as excinfo:
            await node._initialize()
        assert str(excinfo.value) == FAIL_MSG.format(reason="HTTP 401 Unauthorized")
    finally:
        await node._shutdown()
    assert fake_tradingnode.instances == []


# -------------------------------------------------------------------- T13


@pytest.mark.parametrize("handler", [_expired, _transport_error], ids=["401", "connect-error"])
async def test_no_token_in_any_sink(
    handler: Any,
    seeded: dict[str, Any],
    fake_tradingnode: type[_FakeTradingNode],
    api: _Api,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T13: the token never reaches a message, a log, the event log, a status or the API JSON."""
    _configured(monkeypatch)
    _install_probe(monkeypatch, handler)
    node = PaperTradingNode(_config(seeded))
    try:
        with caplog.at_level(logging.INFO), pytest.raises(ConfigurationError) as excinfo:
            await node._initialize()
        exc = excinfo.value
        assert exc.__cause__ is None
        assert exc.__suppress_context__
        log_text = (seeded["tmp"] / "logs" / "PAPER-007.jsonl").read_text()
        status = node.status
        status.error_message = str(exc)
        for sink in (str(exc), repr(exc), caplog.text, log_text, status.error_message or ""):
            assert TOKEN not in sink and LEAK not in sink
    finally:
        await node._shutdown()

    runs_before = api.paper_runs()
    res = await api.ac.post("/api/paper/start", json={"validation_run_id": api.validation_run})
    assert res.status_code == 503, res.text
    assert api.paper_runs() == runs_before
    assert TOKEN not in res.text and LEAK not in res.text
    assert TOKEN not in json.dumps(api.state.get_backtest_run(api.validation_run))


@pytest.mark.parametrize("status", [201, 204])
async def test_non_200_2xx_probe_refuses_via_api(
    status: int, api: _Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2xx other than 200 with ok=true is refused: 503, 0 runs, 0 jobs."""
    _configured(monkeypatch)
    _install_probe(monkeypatch, _resp(status, {"ok": True}))

    res = await api.ac.post("/api/paper/start", json={"validation_run_id": api.validation_run})

    assert res.status_code == 503, res.text
    assert res.json()["detail"] == FAIL_MSG.format(reason=f"HTTP {status} invalid body")
    assert api.paper_runs() == 0
    assert api.launched == []


# -------------------------------------------------------------------- T16


async def test_restore_precheck(api: _Api, monkeypatch: pytest.MonkeyPatch) -> None:
    """T16: /restore under a broken Telegram -> 503 and no job launched."""
    cfg_dir = api.tmp / "paper_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "paper_1.json").write_text(
        json.dumps(
            {
                "trader_id": "PAPER-042",
                "strategy_id": api.strategy_id,
                "validation_run_id": api.validation_run,
                "binance": {"testnet": True},
                "symbols": ["BTCUSDT"],
            }
        )
    )
    _configured(monkeypatch)
    _install_probe(monkeypatch, _expired)

    res = await api.ac.post("/api/paper/restore", json={"trader_id": "PAPER-042"})

    assert res.status_code == 503, res.text
    assert res.json()["detail"] == FAIL_MSG.format(reason="HTTP 401 Unauthorized")
    assert api.launched == []
    assert api.paper_runs() == 0


# -------------------------------------------------------------------- T17


async def test_api_precedence(api: _Api, monkeypatch: pytest.MonkeyPatch) -> None:
    """T17: kill switch -> 423, missing creds -> 400, both BEFORE the Telegram probe."""
    calls: list[int] = []

    async def _spy(*_args: Any, **_kwargs: Any) -> None:
        calls.append(1)

    monkeypatch.setattr(paper_router, "require_working_telegram", _spy)
    body = {"validation_run_id": api.validation_run}

    monkeypatch.delenv("BINANCE_TESTNET_API_KEY")
    res = await api.ac.post("/api/paper/start", json=body)
    assert res.status_code == 400 and "BINANCE_TESTNET_API_KEY" in res.text
    assert calls == [], "the credentials check runs before the probe"
    assert api.paper_runs() == 0

    monkeypatch.setenv("BINANCE_TESTNET_API_KEY", "tk")

    async def _refuse(*_args: Any, **_kwargs: Any) -> None:
        calls.append(1)
        raise TelegramUnavailableError(FAIL_MSG.format(reason="HTTP 401 Unauthorized"))

    monkeypatch.setattr(paper_router, "require_working_telegram", _refuse)
    res = await api.ac.post("/api/paper/start", json=body)
    assert res.status_code == 503
    assert calls == [1]
    assert api.paper_runs() == 0, "the probe runs before create_backtest_run"

    api.state.set_kill_switch("precedence test")
    res = await api.ac.post("/api/paper/start", json=body)
    assert res.status_code == 423, res.text
    assert calls == [1], "the kill switch is read before the probe"
    assert api.paper_runs() == 0


# -------------------------------------------------------------------- T20


async def _run_concurrently(
    api: _Api, monkeypatch: pytest.MonkeyPatch, posts: list[Any]
) -> list[httpx.Response]:
    """Fire every call whose probe blocks until all of them are inside the window."""
    gate = asyncio.Event()
    arrived = asyncio.Event()
    count = 0

    async def _blocked(*_args: Any, **_kwargs: Any) -> None:
        nonlocal count
        count += 1
        if count == len(posts):
            arrived.set()
        await gate.wait()

    monkeypatch.setattr(paper_router, "require_working_telegram", _blocked)
    tasks = [asyncio.create_task(post()) for post in posts]
    await asyncio.wait_for(arrived.wait(), 5)
    gate.set()
    return list(await asyncio.gather(*tasks))


async def test_concurrent_start_one_wins(api: _Api, monkeypatch: pytest.MonkeyPatch) -> None:
    """T20: the probe await must not let two /start requests both create a run."""
    r1, r2 = await _run_concurrently(
        api,
        monkeypatch,
        [
            lambda: api.ac.post(
                "/api/paper/start", json={"validation_run_id": api.validation_run}
            )
            for _ in range(2)
        ],
    )

    codes = sorted([r1.status_code, r2.status_code])
    assert codes == [201, 409], (r1.status_code, r2.status_code, r1.text, r2.text)
    winner = r1 if r1.status_code == 201 else r2
    loser = r2 if r1.status_code == 201 else r1
    assert loser.json()["detail"] == (
        f"Paper session already running (run_id={winner.json()['run_id']}); stop it first"
    )
    assert api.paper_runs() == 1
    assert len(api.launched) == 1


async def test_concurrent_restore_one_wins(api: _Api, monkeypatch: pytest.MonkeyPatch) -> None:
    """T20: same race fix for two concurrent /restore of one trader_id."""
    cfg_dir = api.tmp / "paper_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "paper_1.json").write_text(
        json.dumps(
            {
                "trader_id": "PAPER-042",
                "strategy_id": api.strategy_id,
                "validation_run_id": api.validation_run,
                "binance": {"testnet": True},
                "symbols": ["BTCUSDT"],
            }
        )
    )
    r1, r2 = await _run_concurrently(
        api,
        monkeypatch,
        [
            lambda: api.ac.post("/api/paper/restore", json={"trader_id": "PAPER-042"})
            for _ in range(2)
        ],
    )

    codes = sorted([r1.status_code, r2.status_code])
    assert codes == [201, 409], (r1.status_code, r2.status_code, r1.text, r2.text)
    winner = r1 if r1.status_code == 201 else r2
    loser = r2 if r1.status_code == 201 else r1
    assert loser.json()["detail"] == (
        f"Paper session already running (run_id={winner.json()['run_id']}); stop it first"
    )
    assert api.paper_runs() == 1
    assert len(api.launched) == 1
