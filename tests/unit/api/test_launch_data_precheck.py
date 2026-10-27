"""Launch-time data prechecks: fill ticks + context-indicator aux data (vibe-quant-b5xfj).

A discovery/screening launch whose fill ticks or context-indicator aux data do
not cover the run window must answer 422 BEFORE any ``backtest_runs`` row exists,
instead of 201 + a run that dies inside the subprocess:

* ``MissingBarDataError: fill ticks for BNBUSDT 1d do not cover the run window``
  (discovery 1d, run 923);
* ``AuxDataUnavailableError: BTCUSDT daily bars end before the requested window``
  (discovery 4h, run 924).

Nothing here reads real data: the data layer is faked at
``vibe_quant.data.fill_ticks.resolve_fill_ticks`` and
``vibe_quant.dsl.aux_data.preflight``, and ``data_precheck.DEFAULT_CATALOG_PATH``
is a per-test tmp path (the worktree's ``data/catalog`` is a symlink into the real
datasets). ``@pytest.mark.real_data_precheck`` opts out of the suite-wide stub in
``tests/unit/conftest.py``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator  # noqa: TCH003
from pathlib import Path  # noqa: TCH003
from typing import Any  # noqa: TCH003

import pytest
from httpx import ASGITransport, AsyncClient

from tests.unit.api.test_liquidation_launch import _PopenPatch
from vibe_quant.api import data_precheck
from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.data import fill_ticks as fill_ticks_mod
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl import aux_data
from vibe_quant.dsl.aux_data import AuxDataUnavailableError
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.screening.nt_runner import MissingBarDataError

pytestmark = pytest.mark.real_data_precheck

_EM_DASH_MSG = (
    "BTCUSDT daily bars end before the requested window "
    "(last close 2026-10-08 00:00, window end 2026-10-15) — download BTCUSDT 1d data"
)
# The literal the module appends to a fill-tick 422 (pinned here on purpose: a
# test that imported the constant could not see it change).
_FILL_HINT = "(python -m vibe_quant.data ingest --symbols SYMBOL --start YYYY-MM-DD)"

_DISCOVERY: dict[str, object] = {
    "symbols": ["BTCUSDT"],
    "timeframes": ["4h"],
    "population": 12,
    "generations": 10,
    "start_date": "2024-01-01",
    "end_date": "2025-01-01",
}

_DSL_4H: dict[str, object] = {
    "name": "data_precheck_4h",
    "timeframe": "4h",
    "indicators": {"rsi": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi < 30"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 3.0},
}

_DSL_FUNDING: dict[str, object] = {
    **_DSL_4H,
    "name": "data_precheck_funding",
    "indicators": {"funding": {"type": "FUNDING", "period": 8}},
}

# EMA does not declare std_dev: the launch's existing indicator-param guard 422s it.
_DSL_UNUSED_PARAM: dict[str, object] = {
    **_DSL_4H,
    "name": "data_precheck_unused_param",
    "indicators": {"ema": {"type": "EMA", "period": 20, "std_dev": 2}},
}

# The stored strategy of the dsl_override cases must NOT itself need ticks (1m),
# so only the override can drive the fill-tick check (S3).
_DSL_1M: dict[str, object] = {**_DSL_4H, "name": "data_precheck_1m", "timeframe": "1m"}

# 1h primary: screening_fill_mode's tick timeframe is 1h, never the 4h default (SF-b).
_DSL_1H: dict[str, object] = {**_DSL_4H, "name": "data_precheck_1h", "timeframe": "1h"}

# A 1d strategy that also loads 1h bars: the FINEST loaded timeframe (1h) is the
# tick timeframe, so an ignored additional_timeframes list would resolve 1d (SF-b).
_DSL_1D_PLUS_1H: dict[str, object] = {
    **_DSL_4H,
    "name": "data_precheck_1d_plus_1h",
    "timeframe": "1d",
    "additional_timeframes": ["1h"],
}

# No indicators: StrategyDSL.model_validate rejects it, so the CLI never reaches
# the data layer and neither may the precheck (S3).
_INVALID_OVERRIDE: dict[str, object] = {"name": "bad_override", "timeframe": "4h"}


@pytest.fixture(autouse=True)
def _tmp_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the precheck at an empty tmp catalog so no test can reach the real one."""
    catalog = tmp_path / "catalog"
    monkeypatch.setattr(data_precheck, "DEFAULT_CATALOG_PATH", catalog)
    return catalog


@pytest.fixture()
async def client(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, StateManager, BacktestJobManager]]:
    app = create_app()
    state_mgr = StateManager(db_path=tmp_path / "test.db")
    _ = state_mgr.conn

    job_mgr = BacktestJobManager(db_path=tmp_path / "test.db")
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


def _run_count(state: StateManager) -> int:
    return int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])


def _insert_running_discovery(state: StateManager, job_mgr: BacktestJobManager, pid: int) -> int:
    """A running discovery job, as a concurrent launch leaves it behind (M16)."""
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
        (run_id, pid),
    )
    job_mgr.conn.commit()
    return run_id


def _spy_ok(calls: list[tuple[object, ...]]) -> Any:
    """A no-op data-layer stand-in that records its args."""

    def _record(*args: object) -> None:
        calls.append(args)

    return _record


def _screen_body(
    state: StateManager, dsl: dict[str, object], *, name: str, **extra: object
) -> dict[str, object]:
    body: dict[str, object] = {
        "strategy_id": state.create_strategy(name=name, dsl_config=dsl),
        "symbols": ["BTCUSDT"],
        "timeframe": "4h",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "parameters": {"initial_balance": 1000},
    }
    body.update(extra)
    return body


# --- D1: discovery fill ticks ---


@pytest.mark.parametrize("symbol", ["BTCUSDT", "BNBUSDT"])
async def test_discovery_fill_tick_gap_is_422_without_a_run_row(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    _tmp_catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
) -> None:
    """D1: a fill-tick gap is a 422 with the resolver's text + the download hint."""
    ac, state, job_mgr = client
    resolved: list[tuple[object, ...]] = []

    def _resolve(*args: object) -> object:
        resolved.append(args)
        raise MissingBarDataError(
            f"fill ticks for {symbol} 1d do not cover the run window: "
            "last tick 1791417719999000000 < end-tf"
        )

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _resolve)
    monkeypatch.setattr(aux_data, "preflight", _spy_ok([]))

    body = {**_DISCOVERY, "symbols": [symbol], "timeframes": ["1d"]}
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json=body)

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == (
        f"fill ticks for {symbol} 1d do not cover the run window: "
        f"last tick 1791417719999000000 < end-tf {_FILL_HINT}"
    )
    # The window is the request's span and the timeframe is the 1d tick set.
    assert resolved == [
        ([symbol], "1d", "2024-01-01", "2025-01-01", _tmp_catalog)
    ]
    assert popen.call_count == 0
    assert job_mgr.list_active_jobs() == []
    assert _run_count(state) == 0


# --- D2: discovery context-indicator aux data ---


async def test_discovery_aux_gap_is_422_without_a_run_row(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D2: a context pool with no aux data is a 422 carrying the preflight's text."""
    ac, state, job_mgr = client
    preflighted: list[tuple[object, ...]] = []

    def _preflight(*args: object) -> None:
        preflighted.append(args)
        raise AuxDataUnavailableError(_EM_DASH_MSG)

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _spy_ok([]))
    monkeypatch.setattr(aux_data, "preflight", _preflight)

    body = {
        **_DISCOVERY,
        "indicator_pool": ["BTC_TREND"],
        "start_date": "2025-01-01",
        "end_date": "2026-10-15",
    }
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json=body)

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == _EM_DASH_MSG
    assert preflighted == [(["BTC_TREND"], ["BTCUSDT"], "2025-01-01", "2026-10-15")]
    assert popen.call_count == 0
    assert job_mgr.list_active_jobs() == []
    assert _run_count(state) == 0


# --- D3/D4/D5: discovery data that is present, unused timeframes, default pool ---


async def test_discovery_with_covered_data_launches_201(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D3: both prechecks pass -> the launch is unchanged (201, one run, one job)."""
    ac, state, job_mgr = client
    resolved: list[tuple[object, ...]] = []
    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _spy_ok(resolved))
    monkeypatch.setattr(aux_data, "preflight", _spy_ok([]))

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json=_DISCOVERY)

    assert r.status_code == 201, r.text
    assert resolved == [(["BTCUSDT"], "4h", "2024-01-01", "2025-01-01", data_precheck.DEFAULT_CATALOG_PATH)]
    assert popen.call_count == 1
    assert _run_count(state) == 1


async def test_discovery_1m_never_resolves_fill_ticks(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D4: a 1m discovery fills on its own bars, so no tick set is resolved."""
    ac, state, job_mgr = client

    def _resolve(*args: object) -> object:
        pytest.fail(f"a 1m launch resolved fill ticks: {args!r}")

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _resolve)
    monkeypatch.setattr(aux_data, "preflight", _spy_ok([]))

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json={**_DISCOVERY, "timeframes": ["1m"]})

    assert r.status_code == 201, r.text
    assert popen.call_count == 1
    assert _run_count(state) == 1


async def test_discovery_default_pool_preflights_no_context_indicator(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D5: pool None -> the default pool has no context spec, so aux is a no-op."""
    ac, state, job_mgr = client
    seen: list[list[str]] = []
    real_preflight = aux_data.preflight

    def _preflight(indicator_types: list[str], symbols: list[str], start: str, end: str) -> None:
        seen.append(list(indicator_types))
        real_preflight(indicator_types, symbols, start, end)

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _spy_ok([]))
    monkeypatch.setattr(aux_data, "preflight", _preflight)

    with _PopenPatch(job_mgr):
        r = await ac.post("/api/discovery/launch", json=_DISCOVERY)

    assert r.status_code == 201, r.text
    assert len(seen) == 1, seen
    assert seen[0], "the default pool is not empty"
    assert "RSI" in seen[0]
    for context in ("FUNDING", "BTC_TREND", "BTC_ROC", "OWN_TREND"):
        assert context not in seen[0]


# --- S1/S2: screening ---


async def test_screening_fill_tick_gap_is_422_without_a_run_row(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    _tmp_catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S1: the stored DSL's timeframe drives the tick check over the request window."""
    ac, state, job_mgr = client
    resolved: list[tuple[object, ...]] = []

    def _resolve(*args: object) -> object:
        resolved.append(args)
        raise MissingBarDataError(
            "fill ticks for BTCUSDT 4h do not cover the run window: no ticks in the window"
        )

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _resolve)
    monkeypatch.setattr(aux_data, "preflight", _spy_ok([]))

    body = _screen_body(state, _DSL_4H, name="screen_ticks")
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == (
        "fill ticks for BTCUSDT 4h do not cover the run window: no ticks in the window "
        f"{_FILL_HINT}"
    )
    assert resolved == [(["BTCUSDT"], "4h", "2025-01-01", "2025-02-01", _tmp_catalog)]
    assert popen.call_count == 0
    assert job_mgr.list_active_jobs() == []
    assert _run_count(state) == 0


async def test_screening_aux_gap_is_422_without_a_run_row(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S2: the DSL's context indicator is preflighted with the request window."""
    ac, state, job_mgr = client
    preflighted: list[tuple[object, ...]] = []

    def _preflight(*args: object) -> None:
        preflighted.append(args)
        raise AuxDataUnavailableError(_EM_DASH_MSG)

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _spy_ok([]))
    monkeypatch.setattr(aux_data, "preflight", _preflight)

    body = _screen_body(state, _DSL_FUNDING, name="screen_aux")
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == _EM_DASH_MSG
    assert preflighted == [(["FUNDING"], ["BTCUSDT"], "2025-01-01", "2025-02-01")]
    assert popen.call_count == 0
    assert _run_count(state) == 0


# --- O1: ordering ---


async def test_existing_422s_win_over_the_data_precheck(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O1: pool validation, config and the indicator-param guard run BEFORE the data precheck."""
    ac, state, job_mgr = client
    resolved: list[tuple[object, ...]] = []
    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _spy_ok(resolved))
    monkeypatch.setattr(aux_data, "preflight", _spy_ok([]))

    pool = {**_DISCOVERY, "indicator_pool": ["NOT_A_POOL"]}
    with _PopenPatch(job_mgr):
        r_pool = await ac.post("/api/discovery/launch", json=pool)
    assert r_pool.status_code == 422, r_pool.text
    assert "not available for GA discovery" in r_pool.json()["detail"]
    assert resolved == []

    maker = _screen_body(state, _DSL_4H, name="order_maker", entry_order="post_only")
    with _PopenPatch(job_mgr):
        r_maker = await ac.post("/api/backtest/screening", json=maker)
    assert r_maker.status_code == 422, r_maker.text
    assert "is validation-only" in r_maker.json()["detail"]
    assert resolved == []

    unused = _screen_body(state, _DSL_UNUSED_PARAM, name="order_unused")
    with _PopenPatch(job_mgr):
        r_unused = await ac.post("/api/backtest/screening", json=unused)
    assert r_unused.status_code == 422, r_unused.text
    assert r_unused.json()["detail"] == (
        "Indicator 'EMA' does not use param 'std_dev'. Valid params: ['period']"
    )
    assert resolved == []

    # And the data precheck itself still fires on the same request shapes.
    def _raise(*args: object) -> object:
        resolved.append(args)
        raise MissingBarDataError("fill ticks for BTCUSDT 4h do not cover the run window: no ticks")

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _raise)
    with _PopenPatch(job_mgr):
        r_data = await ac.post("/api/backtest/screening", json=_screen_body(state, _DSL_4H, name="order_ok"))
    assert r_data.status_code == 422, r_data.text
    assert "fill ticks for BTCUSDT 4h" in r_data.json()["detail"]
    assert _run_count(state) == 0


async def test_data_precheck_runs_before_the_mark_precheck(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O1: with both a tick gap and a MARK gap, the data 422 is the one returned."""
    ac, state, job_mgr = client
    from vibe_quant.api import mark_precheck

    mark_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(mark_precheck, "precheck_discovery", _spy_ok(mark_calls))
    monkeypatch.setattr(mark_precheck, "precheck_screening", _spy_ok([]))

    def _resolve(*args: object) -> object:
        raise MissingBarDataError("fill ticks for BTCUSDT 1d do not cover the run window: no ticks")

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _resolve)
    monkeypatch.setattr(aux_data, "preflight", _spy_ok([]))

    body = {
        **_DISCOVERY,
        "timeframes": ["1d"],
        "liquidation_mode": "isolated",
        "leverage": 20,
    }
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json=body)

    assert r.status_code == 422, r.text
    assert "fill ticks for BTCUSDT 1d" in r.json()["detail"]
    assert "needs MARK 1m bars" not in r.json()["detail"]
    assert mark_calls == []
    assert popen.call_count == 0
    assert _run_count(state) == 0


async def test_mode_off_launch_rechecks_capacity_after_the_data_precheck(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O1/R4: a full capacity reached inside the data precheck still wins (409, no row).

    A mode-off launch has no MARK await, so its capacity re-check after the data
    precheck is what keeps M16: 4 running jobs at admission, the 5th taken by a
    concurrent launch *during* the precheck, and this launch must get the
    existing capacity 409 and create no run row of its own.
    """
    ac, state, job_mgr = client
    for i in range(4):
        _insert_running_discovery(state, job_mgr, 99900 + i)
    assert _run_count(state) == 4

    def _fill_up(*args: object) -> None:
        _insert_running_discovery(state, job_mgr, 99999)

    monkeypatch.setattr(data_precheck, "precheck_discovery_data", _fill_up)

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json=_DISCOVERY)

    assert r.status_code == 409, r.text
    assert "5 discovery jobs already running" in r.json()["detail"]
    assert popen.call_count == 0
    assert len(job_mgr.list_active_jobs()) == 5  # the 4 + the race's; no job of this launch
    assert _run_count(state) == 5  # the four + the race's: this launch created none


# --- S3: the screening precheck reads the EFFECTIVE DSL (parameters.dsl_override) ---


async def test_screening_dsl_override_drives_the_tick_check(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    _tmp_catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3: a launch dsl_override is the DSL the CLI runs, so the check must use it.

    Stored 1m (which needs no ticks at all) + override 4h (which does): pre-fix the
    stored DSL won, the check found nothing to resolve and the launch answered 201
    with a run row, while the subprocess would then fail on 4h fill ticks.
    """
    ac, state, job_mgr = client
    resolved: list[tuple[object, ...]] = []

    def _resolve(*args: object) -> object:
        resolved.append(args)
        raise MissingBarDataError(
            "fill ticks for BTCUSDT 4h do not cover the run window: no ticks in the window"
        )

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _resolve)
    monkeypatch.setattr(aux_data, "preflight", _spy_ok([]))

    body = _screen_body(state, _DSL_1M, name="screen_override_ticks")
    body["parameters"] = {"initial_balance": 1000, "dsl_override": _DSL_4H}
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == (
        "fill ticks for BTCUSDT 4h do not cover the run window: no ticks in the window "
        f"{_FILL_HINT}"
    )
    # The override's 4h DSL, not the stored 1m one, selected the tick set.
    assert resolved == [(["BTCUSDT"], "4h", "2025-01-01", "2025-02-01", _tmp_catalog)]
    assert popen.call_count == 0
    assert _run_count(state) == 0


async def test_screening_dsl_override_1m_skips_the_tick_check(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3: stored 4h + override 1m -> no tick set (the override fills on its own bars).

    Pre-fix the stored 4h DSL resolved a tick set the run would never ask for.
    """
    ac, state, job_mgr = client
    resolved: list[tuple[object, ...]] = []
    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _spy_ok(resolved))
    monkeypatch.setattr(aux_data, "preflight", _spy_ok([]))

    body = _screen_body(state, _DSL_4H, name="screen_override_1m")
    body["parameters"] = {"initial_balance": 1000, "dsl_override": _DSL_1M}
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 201, r.text
    assert resolved == []
    assert popen.call_count == 1
    assert _run_count(state) == 1


async def test_screening_dsl_override_context_indicator_drives_the_aux_check(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3: the override's context indicator is preflighted, not the stored DSL's RSI."""
    ac, state, job_mgr = client
    preflighted: list[tuple[object, ...]] = []

    def _preflight(*args: object) -> None:
        preflighted.append(args)
        raise AuxDataUnavailableError(_EM_DASH_MSG)

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _spy_ok([]))
    monkeypatch.setattr(aux_data, "preflight", _preflight)

    body = _screen_body(state, _DSL_4H, name="screen_override_aux")
    body["parameters"] = {"initial_balance": 1000, "dsl_override": _DSL_FUNDING}
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == _EM_DASH_MSG
    assert preflighted == [(["FUNDING"], ["BTCUSDT"], "2025-01-01", "2025-02-01")]
    assert popen.call_count == 0
    assert _run_count(state) == 0


async def test_archive_refresh_recovers_taker_launch(
    client: tuple[AsyncClient, StateManager, BacktestJobManager], tmp_path: Path
) -> None:
    """A 422 "download 1m data" must clear on the next launch once the archive is refreshed."""
    import sqlite3

    ac, state, job_mgr = client
    minute_ns = 60 * 10**9
    day0 = 1_704_067_200 * 10**9  # 2024-01-01T00:00:00Z
    path = tmp_path / "refresh.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE raw_klines(symbol TEXT, interval TEXT, open_time INTEGER, "
        "volume REAL, taker_buy_volume REAL)"
    )
    insert = "INSERT INTO raw_klines VALUES ('ZECUSDT', '1m', ?, 1.0, 1.0)"
    db.executemany(insert, [((day0 + i * minute_ns) // 10**6,) for i in range(-1441, 1440)])
    db.commit()
    db.close()
    aux_data.configure(path)
    dsl: dict[str, object] = {
        "name": "taker_refresh",
        "timeframe": "1m",
        "indicators": {"x": {"type": "TAKER_IMBALANCE", "period": 5}},
        "entry_conditions": {"long": ["x > 0.1"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 4.0},
    }
    body = {
        "strategy_id": state.create_strategy(name="taker_refresh", dsl_config=dsl),
        "symbols": ["ZECUSDT"],
        "timeframe": "1m",
        "start_date": "2024-01-01",
        "end_date": "2024-01-03",
        "parameters": {"dsl_override": dsl},
    }
    try:
        with _PopenPatch(job_mgr) as popen:
            before = await ac.post("/api/backtest/screening", json=body)
            assert before.status_code == 422, before.text
            assert popen.call_count == 0

            db = sqlite3.connect(path)
            db.executemany(insert, [((day0 + i * minute_ns) // 10**6,) for i in range(1440, 2880)])
            db.commit()
            db.close()

            after = await ac.post("/api/backtest/screening", json=body)
            assert after.status_code == 201, after.text
            assert popen.call_count == 1
        assert _run_count(state) == 1
    finally:
        aux_data.configure(None, None)


async def test_screening_invalid_dsl_override_skips_the_data_precheck(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3: an override the CLI cannot parse is a strategy problem, never a data 422."""
    ac, state, job_mgr = client

    def _unexpected(*args: object) -> object:
        pytest.fail(f"the data precheck ran on an unparseable override: {args!r}")

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _unexpected)
    monkeypatch.setattr(aux_data, "preflight", _unexpected)

    body = _screen_body(state, _DSL_4H, name="screen_bad_override")
    body["parameters"] = {"initial_balance": 1000, "dsl_override": _INVALID_OVERRIDE}
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 201, r.text
    assert popen.call_count == 1
    assert _run_count(state) == 1


async def test_screening_untranslatable_stored_dsl_skips_the_data_precheck(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S4: risk.trailing_stop_pct makes translate_dsl_config raise ValueError: launch 201."""
    ac, state, job_mgr = client

    def _unexpected(*args: object) -> object:
        pytest.fail(f"the data precheck ran on an untranslatable DSL: {args!r}")

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _unexpected)
    monkeypatch.setattr(aux_data, "preflight", _unexpected)

    # Frontend DslConfig shape (a `general` key) so translate_dsl_config really runs.
    dsl: dict[str, object] = {
        "general": {"timeframe": "4h"},
        "indicators": [{"type": "RSI", "name": "rsi", "params": {"period": 14}}],
        "entry_conditions": {"long": ["rsi < 30"]},
        "risk": {
            "stop_loss": {"type": "fixed_pct", "percent": 2.0},
            "take_profit": {"type": "fixed_pct", "percent": 3.0},
            "trailing_stop_pct": 2,
        },
    }
    body = _screen_body(state, dsl, name="screen_trailing_pct")
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 201, r.text
    assert popen.call_count == 1
    assert _run_count(state) == 1


@pytest.mark.parametrize("exc", [KeyError("indicators"), TypeError("not a mapping")])
async def test_screening_translate_error_skips_the_data_precheck(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
) -> None:
    """S4: a KeyError/TypeError out of translate_dsl_config is a strategy problem, not a 500."""
    ac, state, job_mgr = client

    def _unexpected(*args: object) -> object:
        pytest.fail(f"the data precheck ran on an untranslatable DSL: {args!r}")

    def _boom(*args: object, **kwargs: object) -> object:
        raise exc

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _unexpected)
    monkeypatch.setattr(aux_data, "preflight", _unexpected)
    monkeypatch.setattr("vibe_quant.dsl.translator.translate_dsl_config", _boom)

    body = _screen_body(state, _DSL_4H, name="screen_translate_boom")
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 201, r.text
    assert popen.call_count == 1
    assert _run_count(state) == 1


async def test_screening_empty_dsl_override_falls_back_to_the_stored_dsl(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    _tmp_catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3: an EMPTY dsl_override is not an override — the CLI tests truthiness."""
    ac, state, job_mgr = client
    resolved: list[tuple[object, ...]] = []

    def _resolve(*args: object) -> object:
        resolved.append(args)
        raise MissingBarDataError("fill ticks for BTCUSDT 4h do not cover the run window: no ticks")

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _resolve)
    monkeypatch.setattr(aux_data, "preflight", _spy_ok([]))

    body = _screen_body(state, _DSL_4H, name="screen_empty_override")
    body["parameters"] = {"initial_balance": 1000, "dsl_override": {}}
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 422, r.text
    assert resolved == [(["BTCUSDT"], "4h", "2025-01-01", "2025-02-01", _tmp_catalog)]
    assert popen.call_count == 0
    assert _run_count(state) == 0


# --- SF(a)/O2: the discovery precheck leaves the event loop ---


async def test_discovery_data_precheck_runs_off_the_event_loop(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SF(a): a launch must ``await asyncio.to_thread`` the precheck, never run it inline.

    A parquet read on the loop would stall every other request; pinning both the
    offload and the worker thread identity kills an inline call (which the round-1
    tests could not see).
    """
    ac, state, job_mgr = client
    loop_thread = threading.get_ident()
    worker_threads: list[int] = []
    offloaded: list[object] = []
    real_to_thread = asyncio.to_thread

    async def _spy_to_thread(func: object, /, *args: object, **kwargs: object) -> object:
        offloaded.append(func)
        return await real_to_thread(func, *args, **kwargs)

    def _precheck(*args: object) -> None:
        worker_threads.append(threading.get_ident())

    monkeypatch.setattr(asyncio, "to_thread", _spy_to_thread)
    monkeypatch.setattr(data_precheck, "precheck_discovery_data", _precheck)

    with _PopenPatch(job_mgr):
        r = await ac.post("/api/discovery/launch", json=_DISCOVERY)

    assert r.status_code == 201, r.text
    assert offloaded.count(_precheck) == 1, offloaded
    assert len(worker_threads) == 1, worker_threads
    assert worker_threads[0] != loop_thread


# --- SF(b): the screening tick timeframe is the DSL's finest LOADED timeframe ---


async def test_screening_1h_dsl_resolves_1h_ticks_not_4h(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    _tmp_catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SF(b): a 1h strategy's tick set is its own 1h, not the 4h the request names."""
    ac, state, job_mgr = client
    resolved: list[tuple[object, ...]] = []

    def _resolve(*args: object) -> object:
        resolved.append(args)
        raise MissingBarDataError("fill ticks for BTCUSDT 1h do not cover the run window: no ticks")

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _resolve)
    monkeypatch.setattr(aux_data, "preflight", _spy_ok([]))

    body = _screen_body(state, _DSL_1H, name="screen_1h")
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 422, r.text
    assert "fill ticks for BTCUSDT 1h" in r.json()["detail"]
    assert resolved == [(["BTCUSDT"], "1h", "2025-01-01", "2025-02-01", _tmp_catalog)]
    assert popen.call_count == 0
    assert _run_count(state) == 0


async def test_screening_additional_timeframe_drives_the_tick_timeframe(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    _tmp_catalog: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SF(b): a 1d strategy that also loads 1h bars resolves the finest loaded tf (1h)."""
    ac, state, job_mgr = client
    resolved: list[tuple[object, ...]] = []

    def _resolve(*args: object) -> object:
        resolved.append(args)
        raise MissingBarDataError("fill ticks for BTCUSDT 1h do not cover the run window: no ticks")

    monkeypatch.setattr(fill_ticks_mod, "resolve_fill_ticks", _resolve)
    monkeypatch.setattr(aux_data, "preflight", _spy_ok([]))

    body = _screen_body(state, _DSL_1D_PLUS_1H, name="screen_1d_plus_1h")
    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 422, r.text
    assert "fill ticks for BTCUSDT 1h" in r.json()["detail"]
    assert resolved == [(["BTCUSDT"], "1h", "2025-01-01", "2025-02-01", _tmp_catalog)]
    assert popen.call_count == 0
    assert _run_count(state) == 0
