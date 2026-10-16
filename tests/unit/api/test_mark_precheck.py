"""MARK-coverage precheck on mode-on launches (bead vibe-quant-592lt, contract M).

A mode-on screening/validation/discovery launch without MARK 1m coverage must
answer 422 BEFORE any ``backtest_runs`` row exists, instead of failing the run
later inside ``apply_liquidation -> require_mark_coverage``. The window checked
is the one the subprocess itself scans: the request dates for screening and
discovery, and the 1m-clamped window for validation (so a request end past the
data is not a false 422).

Nothing here reads the real catalog: ``mark_precheck.DEFAULT_CATALOG_PATH`` is
patched to a per-test tmp catalog (temp parquet files only -- the worktree's
``data/catalog`` is a symlink into the real datasets and must never be written).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator  # noqa: TCH003
from pathlib import Path  # noqa: TCH003
from typing import Any  # noqa: TCH003
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from vibe_quant.api import mark_precheck
from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.data import mark_bars
from vibe_quant.data.catalog import get_bar_type
from vibe_quant.data.mark_bars import mark_bar_type
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.validation.extraction import date_to_ns
from vibe_quant.validation.runner import ValidationRunner

_MINUTE_NS = 60 * 1_000_000_000
_PRICE_SCALE = 10**16
_FRONTEND_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "frontend"
    / "src"
    / "components"
    / "backtest"
    / "__fixtures__"
    / "legacy_ui_strategy.json"
)

# Canonical (not frontend-format) 1m DSL: resolves to fill_timeframe "1m" with no
# 5s detail data, i.e. the window validation clamps to the 1m coverage.
_DSL_1M: dict[str, object] = {
    "name": "mark_precheck_probe",
    "timeframe": "1m",
    "indicators": {"rsi": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi < 30"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 3.0},
}


def _ns(value: str) -> int:
    ns = date_to_ns(value)
    assert ns is not None
    return ns


def _encode_price(price: float) -> bytes:
    """NT fixed-point price: 128-bit little-endian int, low 8 bytes first."""
    raw = int(round(price * _PRICE_SCALE))
    return (raw & (2**64 - 1)).to_bytes(8, "little") + (
        (raw >> 64) & (2**64 - 1)
    ).to_bytes(8, "little")


def _write_bars(catalog: Path, bar_type: str, ts_ns: list[int], base: float = 100.0) -> None:
    """Write a catalog parquet holding one bar per ``ts`` (open time)."""
    table = pa.table(
        {
            "open": pa.array([_encode_price(base) for _ in ts_ns], type=pa.binary(16)),
            "high": pa.array([_encode_price(base + 2.0) for _ in ts_ns], type=pa.binary(16)),
            "low": pa.array([_encode_price(base - 2.0) for _ in ts_ns], type=pa.binary(16)),
            "close": pa.array([_encode_price(base + 0.5) for _ in ts_ns], type=pa.binary(16)),
            "volume": pa.array([_encode_price(1.0) for _ in ts_ns], type=pa.binary(16)),
            "ts_event": pa.array(ts_ns, type=pa.uint64()),
            "ts_init": pa.array(ts_ns, type=pa.uint64()),
        }
    )
    bar_dir = catalog / "data" / "bar" / bar_type
    bar_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, bar_dir / "part.parquet")


def _bars_every_minute(start_ns: int, end_ns: int) -> list[int]:
    """One MARK bar per minute of ``[start_ns, end_ns)`` (last opens at end - 60 s)."""
    return list(range(start_ns, end_ns, _MINUTE_NS))


@pytest.fixture(autouse=True)
def _catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the precheck at a tmp catalog so no test can reach the real one."""
    catalog = tmp_path / "catalog"
    monkeypatch.setattr(mark_precheck, "DEFAULT_CATALOG_PATH", catalog)
    return catalog


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
async def client(
    tmp_db: Path,
) -> AsyncIterator[tuple[AsyncClient, StateManager, BacktestJobManager]]:
    app = create_app()
    state_mgr = StateManager(db_path=tmp_db)
    _ = state_mgr.conn

    job_mgr = BacktestJobManager(db_path=tmp_db)
    ws_mgr = ConnectionManager()
    await ws_mgr.start()

    app.state.state_manager = state_mgr
    app.state.job_manager = job_mgr
    app.state.ws_manager = ws_mgr

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, state_mgr, job_mgr

    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


class _PopenPatch:
    """Patch subprocess.Popen + liveness so a launch creates a real run row."""

    def __init__(self, job_mgr: BacktestJobManager, pid: int = 12345) -> None:
        self._alive = patch.object(job_mgr, "is_process_alive", return_value=True)
        self._popen = patch("subprocess.Popen")
        self._pid = pid

    def __enter__(self) -> Any:
        self._alive.start()
        mock_popen = self._popen.start()
        mock_popen.return_value.pid = self._pid
        return mock_popen

    def __exit__(self, *exc: object) -> None:
        self._popen.stop()
        self._alive.stop()


def _launch_body(
    state: StateManager,
    name: str,
    *,
    mode: str | None = None,
    dsl: dict[str, object] | None = None,
    symbols: list[str] | None = None,
    start: str = "2025-01-01",
    end: str = "2025-01-02",
) -> dict[str, object]:
    strategy_id = state.create_strategy(
        name=name, dsl_config=_DSL_1M if dsl is None else dsl
    )
    body: dict[str, object] = {
        "strategy_id": strategy_id,
        "symbols": ["BTCUSDT"] if symbols is None else symbols,
        "timeframe": "1m",
        "start_date": start,
        "end_date": end,
        "parameters": {"initial_balance": 1000},
    }
    if mode is not None:
        body["liquidation_mode"] = mode
    return body


_DISCOVERY_BODY: dict[str, object] = {
    "symbols": ["BTCUSDT"],
    "timeframes": ["4h"],
    "population": 12,
    "generations": 10,
    "start_date": "2025-01-01",
    "end_date": "2025-01-02",
}


# A span long enough for the default holdout split (reaches the MARK step).
_DISCOVERY_WIDE: dict[str, object] = {
    **_DISCOVERY_BODY,
    "start_date": "2024-01-01",
    "end_date": "2025-01-01",
}


def _run_count(state: StateManager) -> int:
    return int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])


def _assert_not_mark_422(text: str) -> None:
    assert "needs MARK 1m bars" not in text


# --- M1/M2/M3/M4: the windows and symbols each checker uses ---


def test_screening_window_is_half_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1: end exclusive -- the last required bar opens at end - 60 s."""
    start, end = "2025-01-01", "2025-01-02"
    start_ns, end_ns = _ns(start), _ns(end)
    bars = _bars_every_minute(start_ns, end_ns)

    covered = tmp_path / "covered"
    _write_bars(covered, mark_bar_type("BTCUSDT"), bars)
    monkeypatch.setattr(mark_precheck, "DEFAULT_CATALOG_PATH", covered)
    mark_precheck.precheck_screening("isolated", ["BTCUSDT"], start, end)

    short = tmp_path / "short"
    _write_bars(short, mark_bar_type("BTCUSDT"), bars[:-1])  # last bar = end - 120 s
    monkeypatch.setattr(mark_precheck, "DEFAULT_CATALOG_PATH", short)
    with pytest.raises(HTTPException) as exc:
        mark_precheck.precheck_screening("isolated", ["BTCUSDT"], start, end)
    assert exc.value.status_code == 422
    assert "BTCUSDT: MARK coverage ends" in exc.value.detail


def test_validation_uses_clamped_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M2: the checked window is the runner's clamped one, not the request dates."""
    symbol = "BTCUSDT"
    request = ("2026-03-01", "2026-10-02")  # request end is past the 1m data
    # The last 1m LAST bar opens 2026-03-16T23:59, so validation clamps its end
    # to the whole UTC day 2026-03-17.
    one_m = [_ns("2026-03-01"), _ns("2026-03-16T23:59")]

    covered = tmp_path / "covered"
    _write_bars(covered, str(get_bar_type(symbol, "1m")), one_m)
    _write_bars(
        covered, mark_bar_type(symbol), _bars_every_minute(_ns("2026-03-01"), _ns("2026-03-17"))
    )
    monkeypatch.setattr(mark_precheck, "DEFAULT_CATALOG_PATH", covered)
    # MARK ends at 2026-03-16 23:59 = clamped end - 60 s: covered.
    mark_precheck.precheck_validation("isolated", [symbol], *request, _DSL_1M, "s", {})

    short = tmp_path / "short"
    _write_bars(short, str(get_bar_type(symbol, "1m")), one_m)
    _write_bars(
        short, mark_bar_type(symbol), _bars_every_minute(_ns("2026-03-01"), _ns("2026-03-15"))
    )
    monkeypatch.setattr(mark_precheck, "DEFAULT_CATALOG_PATH", short)
    with pytest.raises(HTTPException) as exc:
        mark_precheck.precheck_validation("isolated", [symbol], *request, _DSL_1M, "s", {})
    assert exc.value.status_code == 422
    # "missing 2026-03" is the CLAMPED end's month; the request's 2026-10 is not used.
    assert "missing 2026-03) before requested end" in exc.value.detail


def test_discovery_checks_full_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M3: every stage's sub-window lies in [start, end), so the span is checked."""
    start, end = "2025-01-01", "2025-01-02"
    start_ns, end_ns = _ns(start), _ns(end)
    bars = _bars_every_minute(start_ns, end_ns)

    covered = tmp_path / "covered"
    _write_bars(covered, mark_bar_type("BTCUSDT"), bars)
    monkeypatch.setattr(mark_precheck, "DEFAULT_CATALOG_PATH", covered)
    mark_precheck.precheck_discovery("isolated", ["BTCUSDT"], start, end)

    short = tmp_path / "short"
    _write_bars(short, mark_bar_type("BTCUSDT"), bars[:-60])  # last hour missing
    monkeypatch.setattr(mark_precheck, "DEFAULT_CATALOG_PATH", short)
    with pytest.raises(HTTPException) as exc:
        mark_precheck.precheck_discovery("isolated", ["BTCUSDT"], start, end)
    assert exc.value.status_code == 422
    assert "missing 2025-01) before requested end" in exc.value.detail


def test_untraded_symbol_without_mark_is_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M4: ALL requested symbols are required, in request order (first missing wins)."""
    start, end = "2025-01-01", "2025-01-02"
    bars = _bars_every_minute(_ns(start), _ns(end))
    _write_bars(tmp_path / "catalog", mark_bar_type("BTCUSDT"), bars)
    with pytest.raises(HTTPException) as exc:
        mark_precheck.precheck_screening("isolated", ["BTCUSDT", "ETHUSDT"], start, end)
    assert exc.value.status_code == 422
    assert "ETHUSDT: no MARK price bars available" in exc.value.detail


def test_frontend_format_fixture_requires_mark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M14: the frontend-format DSL parses exactly as the runner parses it, then 422s."""
    dsl = json.loads(_FRONTEND_FIXTURE.read_text())
    # Proof the fixture is accepted by the runner's own parser (a skipped
    # precheck would also 201/no-op, so this is what makes the 422 a MARK one).
    parsed = ValidationRunner.__new__(ValidationRunner)._validate_dsl(dsl, strategy_name="ui")
    assert parsed.timeframe == "4h"

    start, end = "2025-01-01", "2025-01-02"
    _write_bars(
        tmp_path / "catalog",
        str(get_bar_type("BTCUSDT", "1m")),
        _bars_every_minute(_ns(start), _ns(end)),
    )
    with pytest.raises(HTTPException) as exc:
        mark_precheck.precheck_validation(
            "isolated", ["BTCUSDT"], start, end, dsl, "legacy_ui", {}
        )
    assert exc.value.status_code == 422
    assert "BTCUSDT: no MARK price bars available" in exc.value.detail


def test_unparseable_dates_skip_the_check() -> None:
    """A date the converters cannot parse is not a MARK problem (and never a TypeError)."""
    mark_precheck.precheck_screening("isolated", ["BTCUSDT"], "not-a-date", "also-not")
    mark_precheck.precheck_discovery("isolated", ["BTCUSDT"], "not-a-date", "also-not")


# --- M5/M6/M7/M9/M11: launch endpoints ---


def test_mode_off_is_a_noop_at_module_level() -> None:
    """M5: each checker no-ops on mode off (the routers also skip the call)."""
    with patch.object(mark_bars, "load_mark_series", side_effect=AssertionError("loaded MARK")):
        mark_precheck.precheck_screening("off", ["BTCUSDT"], "2025-01-01", "2025-01-02")
        mark_precheck.precheck_discovery("off", ["BTCUSDT"], "2025-01-01", "2025-01-02")
        mark_precheck.precheck_validation(
            "off", ["BTCUSDT"], "2025-01-01", "2025-01-02", _DSL_1M, "s", {}
        )


async def test_mode_off_never_loads_mark(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """M5: mode off opens no MARK file, leaves the stored params unchanged, 201."""
    ac, state, job_mgr = client
    body = _launch_body(state, "off_mode")

    with (
        patch.object(mark_precheck, "precheck_screening") as precheck,
        patch.object(mark_bars, "load_mark_series", side_effect=AssertionError("loaded MARK")),
        _PopenPatch(job_mgr) as popen,
    ):
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["id"])
    assert run is not None
    assert run["parameters"] == {"initial_balance": 1000}
    assert popen.call_count == 1
    assert precheck.call_count == 0


async def test_detail_reuses_mark_text(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """M6: the 422 detail is the MARK rule's own failure text plus the download hint."""
    ac, state, job_mgr = client
    body = _launch_body(state, "mark_text", mode="isolated")
    start_ns, end_ns = _ns("2025-01-01"), _ns("2025-01-02")
    with pytest.raises(mark_bars.MarkDataMissing) as missing:
        mark_bars.check_mark_coverage(
            mark_precheck.DEFAULT_CATALOG_PATH, ["BTCUSDT"], start_ns, end_ns
        )
    expected = (
        "liquidation_mode='isolated' needs MARK 1m bars: "
        f"{missing.value} (python -m vibe_quant.data download-mark "
        "--start YYYY-MM-DD --catalog)"
    )

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == expected
    assert "BTCUSDT: no MARK price bars available for 2025-01-01T00:00Z..2025-01-02T00:00Z" in (
        r.json()["detail"]
    )
    assert popen.call_count == 0


async def test_422_leaves_no_run_row(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """M7: all three launch endpoints 422 before any run row or job exists."""
    ac, state, job_mgr = client
    _write_bars(
        mark_precheck.DEFAULT_CATALOG_PATH,
        str(get_bar_type("BTCUSDT", "1m")),
        _bars_every_minute(_ns("2025-01-01"), _ns("2025-01-02")),
    )
    cases = [
        ("/api/backtest/screening", _launch_body(state, "no_row_screen", mode="isolated")),
        ("/api/backtest/validation", _launch_body(state, "no_row_valid", mode="isolated")),
        (
            "/api/discovery/launch",
            {
                **_DISCOVERY_BODY,
                "liquidation_mode": "isolated",
                "start_date": "2024-01-01",
                "end_date": "2025-01-02",
            },
        ),
    ]
    before = _run_count(state)

    for path, body in cases:
        with (
            _PopenPatch(job_mgr) as popen,
            patch.object(job_mgr, "start_job") as start_job,
        ):
            r = await ac.post(path, json=body)
        assert r.status_code == 422, f"{path}: {r.text}"
        assert "needs MARK 1m bars" in r.text, path
        assert start_job.call_count == 0, path
        assert popen.call_count == 0, path
        assert _run_count(state) == before, path


async def test_discovery_default_end_after_mark_is_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """M3b: the default end_date is today, so a stale MARK archive fails every stage."""
    ac, state, job_mgr = client
    _write_bars(
        mark_precheck.DEFAULT_CATALOG_PATH,
        mark_bar_type("BTCUSDT"),
        _bars_every_minute(_ns("2025-01-01"), _ns("2025-01-02")),
    )
    body = {
        "symbols": ["BTCUSDT"],
        "timeframes": ["4h"],
        "population": 12,
        "generations": 10,
        "liquidation_mode": "isolated",
    }

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json=body)

    assert r.status_code == 422, r.text
    assert "BTCUSDT: MARK coverage ends" in r.json()["detail"]
    assert popen.call_count == 0


async def test_validation_clamp_error_is_422_only_when_mode_on(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """M2b: a clamping failure is a 422 in mode-on; mode-off is untouched (201)."""
    ac, state, job_mgr = client
    body = _launch_body(state, "clamp_off", mode="off")
    with _PopenPatch(job_mgr):
        r_off = await ac.post("/api/backtest/validation", json=body)
    assert r_off.status_code == 201, r_off.text

    on = _launch_body(state, "clamp_on", mode="isolated")
    with _PopenPatch(job_mgr):
        r_on = await ac.post("/api/backtest/validation", json=on)
    assert r_on.status_code == 422, r_on.text
    assert "No 1m bars for BTCUSDT in the catalog" in r_on.json()["detail"]
    _assert_not_mark_422(r_on.text)


async def test_validation_unparseable_dsl_skips_precheck(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """M2c: a bad DSL is not a MARK problem -- precheck skipped, run fails later."""
    ac, state, job_mgr = client
    body = _launch_body(state, "bad_dsl", mode="isolated", dsl={})

    with (
        caplog.at_level("INFO", logger="vibe_quant.api.mark_precheck"),
        _PopenPatch(job_mgr) as popen,
    ):
        r = await ac.post("/api/backtest/validation", json=body)

    assert r.status_code == 201, r.text
    assert popen.call_count == 1
    assert "MARK precheck skipped: strategy DSL does not parse" in caplog.text


async def test_loader_runtime_error_is_not_swallowed(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """M11: a parquet-churn RuntimeError is a 500, not a 201/422."""
    ac, state, job_mgr = client
    body = _launch_body(state, "churn", mode="isolated")

    with patch.object(mark_bars, "load_mark_series", side_effect=RuntimeError("parquet churn")):
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 500, r.text
    assert _run_count(state) == 0
    assert job_mgr.list_active_jobs() == []


@pytest.mark.parametrize("endpoint", ["screening", "validation", "discovery"])
async def test_precheck_does_not_block_loop(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    tmp_path: Path,
    endpoint: str,
) -> None:
    """M9b: a GET completes while the loader is HELD inside the precheck (3 endpoints).

    The loader signals entry and then blocks on a threading.Event; the GET must
    finish before the loader does. An inline (no ``to_thread``) precheck blocks
    the whole loop in the loader, so the loader has already finished (timed out)
    by the time the GET is served.
    """
    ac, state, _job_mgr = client
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def _held_load(*args: object, **kwargs: object) -> None:
        entered.set()
        release.wait(3)  # bounded so an inline mutant fails instead of hanging
        finished.set()

    if endpoint == "validation":
        _write_bars(
            tmp_path / "catalog",
            str(get_bar_type("BTCUSDT", "1m")),
            [_ns("2025-01-01"), _ns("2025-01-01T23:59")],
        )
    if endpoint == "discovery":
        url, body = "/api/discovery/launch", {**_DISCOVERY_WIDE, "liquidation_mode": "isolated"}
    else:
        url, body = f"/api/backtest/{endpoint}", _launch_body(state, "loop", mode="isolated")

    with patch.object(mark_bars, "load_mark_series", _held_load):
        post = asyncio.create_task(ac.post(url, json=body))
        assert await asyncio.to_thread(entered.wait, 5), "loader never entered the precheck"
        r_get = await ac.get("/api/backtest/jobs")
        held_during_get = not finished.is_set()
        release.set()
        r_post = await post

    assert r_get.status_code == 200, r_get.text
    assert held_during_get, "GET was only served after the loader finished: loop was blocked"
    assert r_post.status_code == 422, r_post.text
    assert "needs MARK 1m bars" in r_post.text


async def test_concurrent_prechecks_serialize_decode(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """M17: concurrent API coverage checks never overlap (cold decode ~1.1 GB peak each)."""
    ac, state, _job_mgr = client
    active = 0
    max_active = 0
    guard = threading.Lock()

    def _counting_check(*args: object, **kwargs: object) -> None:
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.15)
        with guard:
            active -= 1

    bodies = [_launch_body(state, f"ser{i}", mode="isolated") for i in range(4)]
    with patch.object(mark_precheck, "check_mark_coverage", _counting_check), _PopenPatch(_job_mgr):
        rs = await asyncio.gather(
            *(ac.post("/api/backtest/screening", json=b) for b in bodies)
        )
        assert max_active == 1, f"{max_active} coverage checks ran concurrently"
        assert all(r.status_code == 201 for r in rs), [r.text for r in rs]

    # mode-off never loads
    calls: list[object] = []
    with (
        patch.object(mark_precheck, "check_mark_coverage", lambda *a, **k: calls.append(a)),
        _PopenPatch(_job_mgr),
    ):
        r = await ac.post("/api/backtest/screening", json=_launch_body(state, "off"))
    assert r.status_code == 201, r.text
    assert calls == []


async def test_discovery_admission_rechecked_after_mark_await(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """M16: 4 running + 2 concurrent mode-on launches -> one 201, one 409, 5 jobs."""
    ac, state, job_mgr = client
    for i in range(4):
        run_id = state.create_backtest_run(
            strategy_id=None,
            run_mode="discovery",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2025-03-07",
            end_date="2026-03-07",
            parameters={"population": 12, "generations": 10},
        )
        job_mgr.conn.execute(
            """INSERT INTO background_jobs
               (run_id, pid, job_type, status, started_at, heartbeat_at)
               VALUES (?, ?, 'discovery', 'running', datetime('now'), datetime('now'))""",
            (run_id, 99900 + i),
        )
        job_mgr.conn.commit()
    assert _run_count(state) == 4

    # Both launches pass the first admission check, then meet inside the precheck.
    barrier = threading.Barrier(2, timeout=5)

    def _meet(*args: object, **kwargs: object) -> None:
        barrier.wait()

    body = {**_DISCOVERY_WIDE, "liquidation_mode": "isolated"}
    with _PopenPatch(job_mgr) as popen, patch.object(mark_precheck, "precheck_discovery", _meet):
        rs = await asyncio.gather(
            ac.post("/api/discovery/launch", json=body),
            ac.post("/api/discovery/launch", json=body),
        )

    assert sorted(r.status_code for r in rs) == [201, 409], [r.text for r in rs]
    assert popen.call_count == 1
    assert len(job_mgr.list_active_jobs()) == 5
    assert _run_count(state) == 5  # the loser created no row


async def test_precheck_runs_after_existing_checks(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """M8: pydantic -> 404 -> screening entry_order -> pool -> config -> MARK."""
    ac, state, job_mgr = client

    with _PopenPatch(job_mgr):
        r = await ac.post("/api/backtest/screening", json={"strategy_id": 1})
    assert r.status_code == 422
    _assert_not_mark_422(r.text)

    body = _launch_body(state, "order_404", mode="isolated")
    body["strategy_id"] = 99999
    with _PopenPatch(job_mgr):
        r = await ac.post("/api/backtest/screening", json=body)
    assert r.status_code == 404, r.text
    _assert_not_mark_422(r.text)

    maker = _launch_body(state, "order_maker", mode="isolated")
    maker["entry_order"] = "post_only"
    with _PopenPatch(job_mgr):
        r = await ac.post("/api/backtest/screening", json=maker)
    assert r.status_code == 422, r.text
    assert "is validation-only" in r.json()["detail"]

    pool = {**_DISCOVERY_BODY, "liquidation_mode": "isolated", "indicator_pool": ["NOT_A_POOL"]}
    with _PopenPatch(job_mgr):
        r = await ac.post("/api/discovery/launch", json=pool)
    assert r.status_code == 422, r.text
    assert "not available for GA discovery" in r.json()["detail"]

    config = {
        **_DISCOVERY_BODY,
        "liquidation_mode": "isolated",
        # a real split, then a WFA window longer than the train range
        "start_date": "2024-01-01",
        "end_date": "2026-01-01",
        "wfa_oos_step_days": 3650,
    }
    with _PopenPatch(job_mgr):
        r = await ac.post("/api/discovery/launch", json=config)
    assert r.status_code == 422, r.text
    assert "WFA:" in r.json()["detail"]
    _assert_not_mark_422(r.text)

    final = _launch_body(state, "order_mark", mode="isolated")
    with _PopenPatch(job_mgr):
        r = await ac.post("/api/backtest/screening", json=final)
    assert r.status_code == 422, r.text
    assert "needs MARK 1m bars" in r.json()["detail"]
    assert _run_count(state) == 0
