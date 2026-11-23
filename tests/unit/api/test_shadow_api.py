"""The /api/shadow router: enrol, list (+ unseen badge, kill switch), days, update admission.

Design arch-c7-2 §2 (API signatures), contract rows R17 (API half) and R39 (pre-check half;
the CLI's "skipped: ..." race completion is T5's). The engine seam ``resolve_frozen`` is
faked as in ``test_shadow_tracker.py``; the store, catalog/archive history checks and the
update lock are real.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from tests.unit.test_shadow_tracker import DSL, NOW, PREREG, World, build_data, install
from vibe_quant.api.app import create_app
from vibe_quant.api.routers import shadow as shadow_router
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.data import catalog as catalog_mod
from vibe_quant.data.catalog import CatalogManager
from vibe_quant.db.state_manager import StateManager, shadow_update_lock
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.shadow import tracker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_EPOCH = date(1970, 1, 1)


def _body(**over: object) -> dict[str, object]:
    return {
        "name": "rsi-btc",
        "source_kind": "strategy",
        "dsl_json": json.dumps(DSL),
        "symbols": ["BTCUSDT"],
        "parameters": {},
        "pre_registration": PREREG,
        "max_dd_limit": 0.5,
        **over,
    }


class _Jobs:
    """Records start_job calls instead of spawning (T5 owns the CLI it would run)."""

    def __init__(self, real: BacktestJobManager, *, refuse: str | None = None) -> None:
        self.real = real
        self.calls: list[tuple[int, str, list[str]]] = []
        self.refuse = refuse

    @property
    def db_path(self) -> Path:
        return self.real.db_path

    def start_job(self, run_id: int, job_type: str, command: list[str], **kw: object) -> int:
        if self.refuse is not None:
            raise ValueError(self.refuse)
        self.calls.append((run_id, job_type, command))
        return 4242


@pytest.fixture()
async def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict[str, object]]:
    install(monkeypatch, World())
    cat, arch = build_data(tmp_path)
    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", cat)
    monkeypatch.setattr(tracker, "DEFAULT_ARCHIVE_PATH", arch)
    monkeypatch.setattr(shadow_router, "_utcnow", lambda: NOW)
    db = tmp_path / "state.db"
    app = create_app()
    sm = StateManager(db_path=db)
    _ = sm.conn
    real_jobs = BacktestJobManager(db_path=db)
    jobs = _Jobs(real_jobs)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = sm
    app.state.job_manager = jobs
    app.state.catalog_manager = CatalogManager()
    app.state.ws_manager = ws
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {"client": client, "sm": sm, "jobs": jobs, "db": db}
    await ws.stop()
    real_jobs.close()
    sm.close()


# --- enrol -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("over", "drop", "needle"),
    [
        ({}, "pre_registration", "pre_registration"),  # missing -> pydantic 422
        ({"pre_registration": " " + "x" * 39 + " "}, None, "pre_registration"),  # 39 stripped
        ({"max_dd_limit": 0}, None, "max_dd_limit"),
        ({"max_dd_limit": 1.5}, None, "max_dd_limit"),
        ({"warmup_days": 29}, None, "warmup_days"),
        ({"warmup_days": 731}, None, "warmup_days"),
        ({"source_kind": "paper"}, None, "source_kind"),
        ({"dsl_json": "not json"}, None, "dsl_json"),  # review S1: was a 500
        ({"dsl_json": "[]"}, None, "dsl_json"),
        ({"name": "   "}, None, "name"),
    ],
)
async def test_enrol_refuses_missing_prereg(
    ctx: dict[str, object], over: dict[str, object], drop: str | None, needle: str
) -> None:
    client, sm = ctx["client"], ctx["sm"]
    assert isinstance(client, AsyncClient) and isinstance(sm, StateManager)
    body = _body(**over)
    if drop is not None:
        del body[drop]
    r = await client.post("/api/shadow/enrol", json=body)
    assert r.status_code == 422
    assert needle in json.dumps(r.json()["detail"])
    assert sm.list_shadow_strategies() == []


async def test_enrol_refuses_history_text(ctx: dict[str, object]) -> None:
    """A tracker refusal (warm-up starts before the 1m data) comes back as 422 text."""
    client, sm = ctx["client"], ctx["sm"]
    assert isinstance(client, AsyncClient) and isinstance(sm, StateManager)
    r = await client.post("/api/shadow/enrol", json=_body(backdate_days=1500, warmup_days=730))
    assert r.status_code == 422
    assert r.json()["detail"] == (
        "1m data for BTCUSDT starts 2022-01-01, after warm-up start 2020-10-15"
    )
    assert sm.list_shadow_strategies() == []


async def test_enrol_creates_rows(ctx: dict[str, object]) -> None:
    client, sm = ctx["client"], ctx["sm"]
    assert isinstance(client, AsyncClient) and isinstance(sm, StateManager)
    r1 = await client.post("/api/shadow/enrol", json=_body())
    r2 = await client.post(
        "/api/shadow/enrol",
        json=_body(name="rsi-btc-test", backdate_days=10, warmup_days=30, max_dd_limit=1),
    )
    r3 = await client.post("/api/shadow/enrol", json=_body(name="rsi-btc-2", strategy_id=7))
    assert [r.status_code for r in (r1, r2, r3)] == [201, 201, 201]
    b1, b2, b3 = r1.json(), r2.json(), r3.json()
    assert (b1["name"], b1["slot"], b1["is_test"], b1["state"]) == ("rsi-btc", 1, False, "tracking")
    assert (b1["first_day"], b1["run_start"]) == ("2026-11-23", "2026-05-27")
    assert (b2["slot"], b2["is_test"], b2["family"]) == (None, True, None)
    assert (b2["first_day"], b2["run_start"]) == ("2026-11-13", "2026-10-14")
    assert (b3["slot"], b3["strategy_id"]) == (2, 7)
    stored = {r["id"]: r for r in sm.list_shadow_strategies()}
    assert sorted(stored) == sorted([b1["id"], b2["id"], b3["id"]])
    row = stored[b1["id"]]
    assert row["pre_registration"] == PREREG
    assert row["symbols"] == ["BTCUSDT"]
    assert row["max_dd_limit"] == 0.5
    assert row["enrolled_at"] == "2026-11-22T18:00:00.000000+00:00"
    assert stored[b2["id"]]["max_dd_limit"] == 1.0
    assert stored[b2["id"]]["is_test"] == 1


# --- list, days, seen ----------------------------------------------------------------------


async def test_list_fields_and_seen(ctx: dict[str, object]) -> None:
    client, sm = ctx["client"], ctx["sm"]
    assert isinstance(client, AsyncClient) and isinstance(sm, StateManager)
    a = (await client.post("/api/shadow/enrol", json=_body())).json()["id"]
    b = (await client.post("/api/shadow/enrol", json=_body(name="rsi-b"))).json()["id"]
    stats = {
        "days": 3,
        "total_return": 0.0303,
        "sharpe_ann": 1.25,
        "se_ann": 0.5,
        "ub95": 2.07,
        "max_dd": 0.0,
        "blocks": 0,
        "e_value": 1.0,
    }
    d0 = (date(2026, 11, 23) - _EPOCH).days
    sm.commit_shadow_update(
        a,
        new_days=[(d0, 0.01), (d0 + 1, 0.02), (d0 + 2, -0.0)],
        fields={"stats": stats, "committed_through": "2026-11-25", "state": "stalled"},
        events=[
            {
                "kind": "state_change",
                "from_state": "tracking",
                "to_state": "stalled",
                "message": "g",
            },
            {"kind": "note", "message": "a note never counts"},
        ],
    )
    sm.commit_shadow_update(
        b,
        new_days=[],
        fields={"state": "retired", "state_reason": "inputs changed: venue"},
        events=[
            {
                "kind": "state_change",
                "from_state": "tracking",
                "to_state": "retired",
                "message": "r",
            }
        ],
    )

    body = (await client.get("/api/shadow")).json()
    assert body["unseen_changes"] == 2
    assert body["kill_switch_engaged"] is False
    rows = {r["id"]: r for r in body["strategies"]}
    assert set(rows) == {a, b}
    ra, rb = rows[a], rows[b]
    assert (ra["state"], ra["committed_through"], ra["stats"]) == ("stalled", "2026-11-25", stats)
    assert (ra["slot"], ra["first_day"], ra["pre_registration"]) == (1, "2026-11-23", PREREG)
    assert (rb["state"], rb["state_reason"], rb["slot"], rb["stats"]) == (
        "retired",
        "inputs changed: venue",
        2,
        None,
    )
    assert "dsl_json" not in ra and "frozen_identity" not in ra

    days = (await client.get(f"/api/shadow/{a}/days")).json()
    assert days == [
        {"day": "2026-11-23", "ret": 0.01},
        {"day": "2026-11-24", "ret": 0.02},
        {"day": "2026-11-25", "ret": -0.0},
    ]
    assert (await client.get(f"/api/shadow/{b}/days")).json() == []
    assert (await client.get("/api/shadow/999/days")).status_code == 404

    r = await client.post("/api/shadow/events/seen")
    assert (r.status_code, r.json()) == (200, {"marked": 2})
    assert (await client.get("/api/shadow")).json()["unseen_changes"] == 0
    assert (await client.post("/api/shadow/events/seen")).json() == {"marked": 0}
    seen = sm.conn.execute(
        "SELECT kind, seen_at IS NOT NULL FROM shadow_events ORDER BY id"
    ).fetchall()
    assert [tuple(s) for s in seen] == [("state_change", 1), ("note", 0), ("state_change", 1)]


# --- update admission (R39) + kill switch ---------------------------------------------------


async def test_update_admission(ctx: dict[str, object]) -> None:
    client, jobs, db = ctx["client"], ctx["jobs"], ctx["db"]
    assert isinstance(client, AsyncClient) and isinstance(jobs, _Jobs) and isinstance(db, Path)
    with shadow_update_lock(db):
        r = await client.post("/api/shadow/update")
    assert (r.status_code, r.json()["detail"]) == (409, "shadow update already running")
    assert jobs.calls == []

    r = await client.post("/api/shadow/update")
    assert r.status_code == 202
    body = r.json()
    assert len(jobs.calls) == 1
    job_id, job_type, command = jobs.calls[0]
    assert body == {
        "status": "started",
        "job_id": job_id,
        "pid": 4242,
        "kill_switch_engaged": False,
    }
    assert job_id < 0  # never a backtest run id
    assert job_type == "shadow_update"
    assert command == [
        sys.executable,
        "-m",
        "vibe_quant.shadow",
        "update",
        "--db",
        str(db),
        "--job-id",
        str(job_id),
    ]

    jobs.refuse = f"Run {job_id} already has an active job (pid=4242)"
    r = await client.post("/api/shadow/update")
    assert (r.status_code, r.json()["detail"]) == (409, jobs.refuse)


async def test_kill_switch_engaged(ctx: dict[str, object]) -> None:
    """Informational only: read like paper does; a missing row reads ENGAGED."""
    client, sm = ctx["client"], ctx["sm"]
    assert isinstance(client, AsyncClient) and isinstance(sm, StateManager)
    assert (await client.get("/api/shadow")).json()["kill_switch_engaged"] is False
    sm.set_kill_switch("test halt")
    assert (await client.get("/api/shadow")).json()["kill_switch_engaged"] is True
    assert (await client.post("/api/shadow/update")).json()["kill_switch_engaged"] is True
    sm.clear_kill_switch()
    assert (await client.post("/api/shadow/update")).json()["kill_switch_engaged"] is False
    sm.conn.execute("DELETE FROM system_state")
    sm.conn.commit()
    assert (await client.get("/api/shadow")).json()["kill_switch_engaged"] is True
