"""Launch-time indicator-pool validation (bead vibe-quant-l42df).

A pool the GA genome cannot use must be refused with the pipeline's EXACT
message BEFORE any backtest_runs row or job exists, through the ONE validator
``operators.validate_indicator_pool`` that ``_apply_indicator_pool_filter``
also calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator  # noqa: TCH003
from pathlib import Path  # noqa: TCH003

import pytest
from httpx import ASGITransport, AsyncClient

from tests.unit.api.test_liquidation_launch import _PopenPatch
from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager


@pytest.fixture(autouse=True)
def _pristine_ga_pool() -> Iterator[None]:
    """Rebuild the lazily-cached GA pool from the registry for each test.

    Any test in the same worker that runs ``DiscoveryPipeline.run()`` filters
    the module-global ``operators.INDICATOR_POOL`` IN PLACE (the default pool
    drops the context specs), and ``_ensure_pool`` never re-populates a
    non-empty pool. Rebuild it so this file sees the real GA pool, then
    restore the previous state.
    """
    from vibe_quant.discovery import operators as ops

    caches = [
        ops.INDICATOR_POOL,
        ops._INDICATOR_NAMES,
        ops.THRESHOLD_RANGES,
        ops._INT_PARAMS,
        ops._PARAM_DEFAULTS,
    ]
    saved = [(c, c.copy()) for c in caches]
    for c in caches:
        c.clear()
    ops._ensure_pool()
    yield
    for c, snap in saved:
        c.clear()
        if isinstance(c, dict):
            c.update(snap)  # type: ignore[arg-type]
        else:
            c.extend(snap)  # type: ignore[arg-type]


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


_BODY: dict[str, object] = {
    "symbols": ["BTCUSDT"],
    "timeframes": ["4h"],
    "population": 12,
    "generations": 10,
}


def _run_count(state: StateManager) -> int:
    return int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])


def _available_pool_names() -> list[str]:
    from vibe_quant.discovery.operators import INDICATOR_POOL, _ensure_pool

    _ensure_pool()
    return sorted(INDICATOR_POOL.keys())


def _unknown_pool_message(unknown: list[str]) -> str:
    """The pipeline's exact unknown-names message, spelled out independently."""
    return (
        "indicator_pool contains names not available for GA discovery: "
        f"{sorted(unknown)}. Available: {_available_pool_names()}. "
        "(Indicators with threshold_range=None, e.g. price-vs-MA plugins, "
        "are not yet supported by the genome — see bd-9c1g.)"
    )


@pytest.mark.parametrize(
    ("pool", "unknown"),
    [
        (["STOCH", "CCI", "ROC", "ATR"], ["ATR"]),
        (["NOPE"], ["NOPE"]),
    ],
    ids=["one-unknown", "all-unknown"],
)
async def test_launch_pool_the_ga_cannot_use_is_422_before_any_run(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    pool: list[str],
    unknown: list[str],
) -> None:
    """P1: unknown names -> 422 with the validator's exact text, no run, no job."""
    ac, state, job_mgr = client
    before = _run_count(state)

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json={**_BODY, "indicator_pool": pool})

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == _unknown_pool_message(unknown)
    assert _run_count(state) == before
    assert popen.call_count == 0


async def test_launch_empty_pool_is_422_before_any_run(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """P3: an empty pool is refused; omitting it (null) selects the default."""
    ac, state, job_mgr = client
    before = _run_count(state)

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json={**_BODY, "indicator_pool": []})

    assert r.status_code == 422, r.text
    assert (
        r.json()["detail"]
        == "indicator_pool must not be empty; omit it (null) to use the default pool"
    )
    assert _run_count(state) == before
    assert popen.call_count == 0


async def test_launch_context_indicator_pool_is_accepted(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
) -> None:
    """P2: FUNDING is a context indicator in the pool, excluded only from the default."""
    ac, state, job_mgr = client

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post(
            "/api/discovery/launch", json={**_BODY, "indicator_pool": ["FUNDING", "RSI"]}
        )

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["run_id"])
    assert run is not None
    assert run["parameters"]["indicator_pool"] == ["FUNDING", "RSI"]
    argv = popen.call_args.args[0]
    assert argv[argv.index("--indicator-pool") + 1] == "FUNDING,RSI"


@pytest.mark.parametrize(
    "body_extra", [{}, {"indicator_pool": None}], ids=["omitted", "null"]
)
async def test_launch_default_pool_unchanged(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    body_extra: dict[str, object],
) -> None:
    """P4: null/omitted -> the default pool, no stored key, no CLI flag."""
    ac, state, job_mgr = client

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json={**_BODY, **body_extra})

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["run_id"])
    assert run is not None
    assert "indicator_pool" not in run["parameters"]
    assert "--indicator-pool" not in popen.call_args.args[0]


async def test_launch_validates_through_the_shared_validator(
    client: tuple[AsyncClient, StateManager, BacktestJobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P5: the launch path validates via operators.validate_indicator_pool."""
    from vibe_quant.discovery import operators

    seen: list[list[str]] = []
    real = operators.validate_indicator_pool

    def spy(names: object) -> None:
        seen.append(list(names))  # type: ignore[call-overload]
        real(names)  # type: ignore[arg-type]

    monkeypatch.setattr(operators, "validate_indicator_pool", spy)
    ac, _state, job_mgr = client

    with _PopenPatch(job_mgr):
        r = await ac.post("/api/discovery/launch", json={**_BODY, "indicator_pool": ["RSI"]})

    assert r.status_code == 201, r.text
    assert seen == [["RSI"]]
