"""Bankroll report API: launch, read, cancel (design 2gyk1.3 T4: C26 API part, C27, C28).

``POST /api/results/runs/{id}/bankroll-report`` admits one report (one at a time) and
spawns its job through the real ``BacktestJobManager.start_job``; the job command is
replaced by a harmless sleeper so no engine runs. ``GET`` reconciles first, then returns
the latest report with its freshness.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.routers import results as results_router
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.bankroll.report import default_workers, dsl_hasher
from vibe_quant.bankroll.snapshot import report_root
from vibe_quant.db import run_lifecycle
from vibe_quant.db.state_manager import BANKROLL_JOB_ID_BASE, StateManager
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

NAME = "bankroll_api_rsi"
RSI_DSL: dict[str, object] = {
    "name": NAME,
    "timeframe": "4h",
    "indicators": {"rsi": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi < 45"], "short": ["rsi > 60"]},
    "exit_conditions": {"long": ["rsi > 55"], "short": ["rsi < 40"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 3.0},
}
PARAMS: dict[str, object] = {"risk_per_trade": 0.02, "max_position_pct": 0.5, "leverage": 10}
SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]
DEFAULT_REQUEST: dict[str, object] = {
    "user_cap": 1.0,
    "dd_limit": 0.3,
    "target_vol": 0.2,
    "multiples": [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 10.0],
    "liquidation_mode": None,
    "workers": default_workers(),
    "margin_sweep": False,
    "sweep_modes": ["isolated", "cross"],
    "sweep_leverages": [3, 5, 10],
    "sweep_margin_limits": [0.5, 1.0],
}
SWEEP_DEFAULTS = {
    k: DEFAULT_REQUEST[k]
    for k in ("margin_sweep", "sweep_modes", "sweep_leverages", "sweep_margin_limits")
}


class Api:
    def __init__(self, client: AsyncClient, sm: StateManager, jobs: BacktestJobManager) -> None:
        self.client = client
        self.sm = sm
        self.jobs = jobs
        self.commands: list[list[str]] = []
        self.command = SLEEPER
        self.strategies: dict[str, int] = {}

    def parent(
        self,
        *,
        name: str = NAME,
        dsl: dict[str, object] = RSI_DSL,
        mode: str = "validation",
        status: str = "completed",
        identity: bool = True,
        execution_time: float | None = 80.0,
    ) -> int:
        if name not in self.strategies:
            self.strategies[name] = self.sm.create_strategy(name, {**dsl, "name": name})
        run_id = self.sm.create_backtest_run(
            self.strategies[name], mode, ["BTCUSDT"], "4h", "2025-03-01", "2025-06-01",
            dict(PARAMS),
        )  # fmt: skip
        notes: dict[str, object] = {}
        if identity:
            notes["validation_input"] = {
                "v": 1,
                "strategy_name": name,
                "dsl_sha256": dsl_hasher(name)({**dsl, "name": name}),
            }
        self.sm.save_backtest_result(
            run_id,
            {"total_return": 0.25, "execution_time_seconds": execution_time,
             "notes": json.dumps(notes)},
        )  # fmt: skip
        self.sm.update_backtest_run_status(run_id, status)
        return run_id

    async def post(self, run_id: int, body: object = None, *, content: str | None = None) -> Any:
        url = f"/api/results/runs/{run_id}/bankroll-report"
        if content is not None:
            return await self.client.post(
                url, content=content, headers={"content-type": "application/json"}
            )
        return await self.client.post(url, json={} if body is None else body)

    async def get(self, run_id: int) -> Any:
        return await self.client.get(f"/api/results/runs/{run_id}/bankroll-report")

    def reports(self) -> list[dict[str, Any]]:
        rows = self.sm.conn.execute("SELECT * FROM bankroll_reports ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def job(self, job_id: int) -> dict[str, Any] | None:
        row = self.sm.conn.execute(
            "SELECT * FROM background_jobs WHERE run_id = ?", (job_id,)
        ).fetchone()
        return None if row is None else dict(row)

    def kill_all(self) -> None:
        for row in self.sm.conn.execute(
            "SELECT run_id FROM background_jobs WHERE status = 'running'"
        ).fetchall():
            self.jobs.kill_job(int(row[0]), force=True)


@pytest.fixture()
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "scratch"
    base.mkdir()
    monkeypatch.setattr(results_router, "default_scratch_base", lambda: base)
    return base


@pytest.fixture()
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scratch: Path) -> AsyncIterator[Api]:
    app = create_app()
    db = tmp_path / "state.db"
    sm = StateManager(db_path=db)
    _ = sm.conn
    jobs = BacktestJobManager(db_path=db)
    app.state.state_manager = sm
    app.state.job_manager = jobs
    app.state.ws_manager = ConnectionManager()  # the job-kill route broadcasts
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        world = Api(client, sm, jobs)

        def command(report_id: int, job_id: int, db_path: Path) -> list[str]:
            world.commands.append(["report", str(report_id), str(job_id), str(db_path)])
            return list(world.command)

        monkeypatch.setattr(results_router, "_bankroll_command", command)
        try:
            yield world
        finally:
            world.kill_all()
            jobs.close()
            sm.close()


# --- C28: strict request domain ------------------------------------------------------

REJECTED: list[tuple[str, object]] = [
    ("user_cap bool", {"user_cap": True}),
    ("user_cap numeric string", {"user_cap": "0.5"}),
    ("user_cap NaN string", {"user_cap": "NaN"}),
    ("user_cap inf string", {"user_cap": "inf"}),
    ("user_cap zero", {"user_cap": 0}),
    ("user_cap negative", {"user_cap": -0.5}),
    ("user_cap above max", {"user_cap": 10.0000001}),
    ("dd_limit one", {"dd_limit": 1.0}),
    ("dd_limit zero", {"dd_limit": 0.0}),
    ("dd_limit bool", {"dd_limit": False}),
    ("target_vol above max", {"target_vol": 5.0000001}),
    ("target_vol zero", {"target_vol": 0}),
    ("multiples empty", {"multiples": []}),
    ("multiples duplicates", {"multiples": [0.5, 0.5]}),
    ("multiples int/float duplicates", {"multiples": [1, 1.0]}),
    ("multiples ten entries", {"multiples": [0.1 * i for i in range(1, 11)]}),
    ("multiples zero", {"multiples": [0, 1]}),
    ("multiples negative", {"multiples": [-1]}),
    ("multiples above max", {"multiples": [10.0000001]}),
    ("multiples bool", {"multiples": [True]}),
    ("multiples string", {"multiples": ["2"]}),
    ("multiples not a list", {"multiples": 2}),
    ("workers zero", {"workers": 0}),
    ("workers five", {"workers": 5}),
    ("workers float", {"workers": 2.0}),
    ("workers bool", {"workers": True}),
    ("liquidation_mode off", {"liquidation_mode": "off"}),
    ("extra field", {"user_capp": 1.0}),
    ("margin_sweep string", {"margin_sweep": "true"}),
    ("margin_sweep int", {"margin_sweep": 1}),
]
# The margin grid is checked whether the sweep is on or off.
GRID_REJECTED: list[tuple[str, dict[str, object], str]] = [
    ("sweep_leverages float", {"sweep_leverages": [3.0]}, "int_type"),
    ("sweep_leverages bool", {"sweep_leverages": [True]}, "int_type"),
    ("sweep_leverages empty", {"sweep_leverages": []}, "bankroll grid leverages must not be empty"),
    ("sweep_leverages zero", {"sweep_leverages": [0]}, "bankroll leverage"),
    ("sweep_modes off", {"sweep_modes": ["off"]}, "literal_error"),
    ("sweep_modes empty", {"sweep_modes": []}, "bankroll grid modes must not be empty"),
    ("sweep_margin_limits zero", {"sweep_margin_limits": [0]}, "bankroll max_margin_pct"),
    ("sweep_margin_limits above one", {"sweep_margin_limits": [1.0000001]}, "bankroll max_margin_pct"),
    ("sweep_margin_limits bool", {"sweep_margin_limits": [True]}, "float_type"),
    ("18 cells",
     {"sweep_leverages": [3, 5, 10], "sweep_margin_limits": [0.25, 0.5, 1.0]},
     "bankroll grid has 18 cells; at most 12 are allowed"),
    ("extra sweep field", {"sweep_cushions": [0.5]}, "extra_forbidden"),
]  # fmt: skip
for _case, _body, _ in GRID_REJECTED:
    for _on in (True, False):
        REJECTED.append((f"{_case} sweep {_on}", {**_body, "margin_sweep": _on}))


@pytest.mark.parametrize(("case", "body"), REJECTED, ids=[c for c, _ in REJECTED])
async def test_request_domain(api: Api, case: str, body: object) -> None:
    run_id = api.parent()
    resp = await api.post(run_id, body)
    assert resp.status_code == 422, (case, resp.text)
    assert api.reports() == []
    assert api.commands == []


@pytest.mark.parametrize(
    ("case", "body", "text"), GRID_REJECTED, ids=[c for c, _, _ in GRID_REJECTED]
)
async def test_request_domain_grid_text(
    api: Api, case: str, body: dict[str, object], text: str
) -> None:
    resp = await api.post(api.parent(), body)
    assert resp.status_code == 422, (case, resp.text)
    assert text in resp.text, (case, resp.text)


@pytest.mark.parametrize(
    "content",
    ['{"user_cap": NaN}', '{"user_cap": Infinity}', '{"dd_limit": -Infinity}',
     '{"user_cap": 1e309}', '{"multiples": [NaN]}', '{"multiples": [1e309]}'],
)  # fmt: skip
async def test_request_domain_non_finite_literals(api: Api, content: str) -> None:
    run_id = api.parent()
    resp = await api.post(run_id, content=content)
    assert resp.status_code == 422, resp.text
    assert api.reports() == []


async def test_request_domain_defaults(api: Api) -> None:
    run_id = api.parent()
    resp = await api.post(run_id)
    assert resp.status_code == 202, resp.text
    (row,) = api.reports()
    assert json.loads(row["request"]) == DEFAULT_REQUEST
    assert row["run_id"] == run_id and row["status"] == "running"


ACCEPTED: list[tuple[str, dict[str, object], dict[str, object]]] = [
    ("upper edges",
     {"user_cap": 10, "dd_limit": 0.999, "target_vol": 5, "multiples": [10], "workers": 4,
      "liquidation_mode": "isolated"},
     {"user_cap": 10.0, "dd_limit": 0.999, "target_vol": 5.0, "multiples": [10.0],
      "workers": 4, "liquidation_mode": "isolated", **SWEEP_DEFAULTS}),
    ("lower edges",
     {"user_cap": 1e-9, "dd_limit": 1e-9, "target_vol": 1e-9, "workers": 1,
      "multiples": [1e-9, 0.5, 1, 2, 3, 4, 5, 6, 7], "liquidation_mode": "cross"},
     {"user_cap": 1e-9, "dd_limit": 1e-9, "target_vol": 1e-9, "workers": 1,
      "multiples": [1e-9, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "liquidation_mode": "cross",
      **SWEEP_DEFAULTS}),
    ("sweep on, 12 cells, mode edge",
     {"margin_sweep": True, "sweep_modes": ["cross", "isolated", "cross"],
      "sweep_leverages": [10, 3, 5, 3], "sweep_margin_limits": [1, 0.5], "workers": 1},
     {**DEFAULT_REQUEST, "workers": 1, "margin_sweep": True,
      "sweep_modes": ["cross", "isolated", "cross"], "sweep_leverages": [10, 3, 5, 3],
      "sweep_margin_limits": [1.0, 0.5]}),
    ("sweep off, one cell, limit edges",
     {"sweep_modes": ["cross"], "sweep_leverages": [1], "sweep_margin_limits": [1.0]},
     {**DEFAULT_REQUEST, "sweep_modes": ["cross"], "sweep_leverages": [1],
      "sweep_margin_limits": [1.0]}),
]  # fmt: skip


@pytest.mark.parametrize(("case", "body", "stored"), ACCEPTED, ids=[c for c, _, _ in ACCEPTED])
async def test_request_domain_edges_accepted(
    api: Api, case: str, body: dict[str, object], stored: dict[str, object]
) -> None:
    run_id = api.parent()
    resp = await api.post(run_id, body)
    assert resp.status_code == 202, (case, resp.text)
    (row,) = api.reports()
    assert json.loads(row["request"]) == stored


# --- C27: launch errors --------------------------------------------------------------


async def test_launch_errors(api: Api) -> None:
    missing = await api.post(999_999)
    assert missing.status_code == 404, missing.text

    huge = await api.post(2**70)
    assert huge.status_code in (404, 422), huge.text

    screening = await api.post(api.parent(mode="screening"))
    assert screening.status_code == 422
    assert screening.json()["detail"].startswith("not_validation:")

    pending = await api.post(api.parent(status="running"))
    assert pending.status_code == 422
    assert pending.json()["detail"].startswith("not_completed:")

    unverified = await api.post(api.parent(identity=False))
    assert unverified.status_code == 422
    detail = unverified.json()["detail"]
    assert detail.startswith("inputs_unverified:") and "re-validate first" in detail

    edited = api.parent(name="edited_rsi")
    sid = api.strategies["edited_rsi"]
    api.sm.update_strategy(
        sid,
        dsl_config={
            **RSI_DSL,
            "name": "edited_rsi",
            "stop_loss": {"type": "fixed_pct", "percent": 2.5},
        },
    )
    stale = await api.post(edited)
    assert stale.status_code == 422
    detail = stale.json()["detail"]
    assert detail.startswith("stale_parent:") and "re-validate first" in detail

    assert api.reports() == [] and api.commands == []

    first = await api.post(api.parent())
    assert first.status_code == 202, first.text
    running = await api.post(api.parent(name="second_rsi"))
    assert running.status_code == 409
    assert running.json()["detail"].startswith("report_running:")
    assert len(api.reports()) == 1 and len(api.commands) == 1


# --- C26: lifecycle through the API --------------------------------------------------


async def test_lifecycle_api(api: Api, scratch: Path) -> None:
    run_id = api.parent()
    resp = await api.post(run_id, {"workers": 2})
    assert resp.status_code == 202, resp.text
    body = resp.json()
    report_id = body["report_id"]
    job_id = body["job_id"]
    assert job_id == BANKROLL_JOB_ID_BASE - report_id
    assert body["estimated_seconds"] == 400  # 80 s x (1 baseline + ceil(8 cells / 2))
    assert api.commands == [["report", str(report_id), str(job_id), str(api.jobs.db_path)]]
    # The job runs without a run-mode entry (negative id, like data jobs).
    job = api.job(job_id)
    assert job is not None and job["job_type"] == "bankroll_report"
    assert job["status"] == "running" and api.jobs.is_process_alive(int(job["pid"]))

    got = await api.get(run_id)
    assert got.status_code == 200, got.text
    out = got.json()
    assert (out["id"], out["status"], out["freshness"], out["report"]) == (
        report_id, "running", "fresh", None,
    )  # fmt: skip
    assert out["job_id"] == job_id and out["stale_reasons"] == []
    assert "inputs" not in out

    # A live report's scratch root survives a reconcile; a dead one's goes.
    live_root = report_root(scratch, report_id)
    live_root.mkdir(parents=True)
    orphan = report_root(scratch, report_id + 7)
    orphan.mkdir(parents=True)

    # Cancel = the existing job kill; the next GET reconciles the report to failed.
    assert (await api.client.delete(f"/api/backtest/jobs/{job_id}")).status_code == 204
    cancelled = (await api.get(run_id)).json()
    assert (cancelled["status"], cancelled["error_message"]) == ("failed", "cancelled")
    assert not live_root.exists() and not orphan.exists()

    # A retry after the failure is admitted.
    again = await api.post(run_id, {"multiples": [0.5, 2], "workers": 4})
    assert again.status_code == 202, again.text
    assert again.json()["report_id"] == report_id + 1


async def test_lifecycle_api_estimate_unknown(api: Api) -> None:
    resp = await api.post(api.parent(execution_time=None), {"multiples": [0.5, 2], "workers": 4})
    assert resp.status_code == 202
    assert resp.json()["estimated_seconds"] is None


async def test_lifecycle_api_estimate_small(api: Api) -> None:
    resp = await api.post(api.parent(execution_time=10.0), {"multiples": [0.5, 2], "workers": 2})
    assert resp.status_code == 202
    assert resp.json()["estimated_seconds"] == 30  # 10 s x (1 + ceil(3 cells / 2))


NINE = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]


@pytest.mark.parametrize(
    ("sweep", "seconds"),
    [
        ({"margin_sweep": True}, 700),  # 100 s x (1 + ceil(9/4) + ceil(12/4))
        ({"margin_sweep": True, "sweep_leverages": [3], "sweep_margin_limits": [0.5]}, 500),
        ({"margin_sweep": True, "sweep_modes": ["cross"], "sweep_margin_limits": [0.5]}, 500),
        ({"margin_sweep": True, "sweep_modes": ["cross"], "sweep_leverages": [3, 3, 5, 10, 20],
          "sweep_margin_limits": [0.5]}, 500),  # dedup: 4 cells, one wave
        ({"margin_sweep": True, "sweep_modes": ["cross"], "sweep_leverages": [3, 5, 10, 20],
          "sweep_margin_limits": [0.25, 0.5]}, 600),  # 8 cells, two waves
        ({"margin_sweep": False}, 400),
        ({}, 400),
    ],
)  # fmt: skip
async def test_estimate_counts_sweep(api: Api, sweep: dict[str, object], seconds: int) -> None:
    run_id = api.parent(execution_time=100.0)
    resp = await api.post(run_id, {"multiples": NINE, "workers": 4, **sweep})
    assert resp.status_code == 202, resp.text
    assert resp.json()["estimated_seconds"] == seconds


async def test_lifecycle_api_dead_child(api: Api) -> None:
    """A job process that died before its first heartbeat fails the report on GET."""
    run_id = api.parent()
    api.command = [sys.executable, "-c", "pass"]
    resp = await api.post(run_id)
    assert resp.status_code == 202
    job = api.job(resp.json()["job_id"])
    assert job is not None
    pid = int(job["pid"])
    os.waitpid(pid, 0)  # the child exited without a heartbeat or a terminal write
    out = (await api.get(run_id)).json()
    assert out["status"] == "failed"
    assert out["error_message"].startswith("job failed without publishing: process not running")
    assert (await api.post(run_id)).status_code == 202


async def test_lifecycle_api_spawn_failure(api: Api) -> None:
    run_id = api.parent()
    api.command = [str(Path("/nonexistent/vq-bankroll-python"))]
    resp = await api.post(run_id)
    assert resp.status_code == 500, resp.text
    (row,) = api.reports()
    assert row["status"] == "failed"
    assert row["error_message"].startswith("launch failed: FileNotFoundError")
    assert api.job(int(row["job_id"])) is None
    api.command = SLEEPER
    assert (await api.post(run_id)).status_code == 202


async def test_lifecycle_api_job_id_taken(api: Api) -> None:
    """A data job already holding the next report's job id refuses with 409."""
    run_id = api.parent()
    held = BANKROLL_JOB_ID_BASE - 1  # the first report id is 1
    run_lifecycle.open_attempt(api.sm.conn, held, "data_update", None, exists=False)
    api.sm.conn.execute("UPDATE background_jobs SET status = 'completed' WHERE run_id = ?", (held,))
    api.sm.conn.commit()
    before = api.job(held)
    resp = await api.post(run_id)
    assert resp.status_code == 409
    assert resp.json()["detail"].startswith("job_id_taken:")
    assert api.job(held) == before
    assert api.reports() == [] and api.commands == []


async def test_lifecycle_api_open_job_holds_slot(api: Api) -> None:
    """An open bankroll job row (its report cascaded away) still holds the slot."""
    run_id = api.parent()
    first = await api.post(run_id)
    assert first.status_code == 202
    api.sm.conn.execute("DELETE FROM backtest_runs WHERE id = ?", (run_id,))
    api.sm.conn.commit()
    assert api.reports() == []
    other = await api.post(api.parent(name="other_rsi"))
    assert other.status_code == 409
    assert other.json()["detail"].startswith("job_open:")


# --- GET: freshness with the strategy's own DSL hasher ---------------------------------


def _complete(sm: StateManager, report_id: int, recommendation: dict[str, object]) -> None:
    sm.conn.execute(
        "UPDATE bankroll_reports SET status = 'completed', report = ?, "
        "completed_at = datetime('now') WHERE id = ?",
        (json.dumps({"v": 1, "recommendation": recommendation, "rows": []}), report_id),
    )
    sm.conn.commit()


@pytest.mark.parametrize(("name", "k"), [("fresh_rsi_a", 0.5), ("fresh_rsi_b", 2.0)])
async def test_get_freshness(api: Api, name: str, k: float) -> None:
    run_id = api.parent(name=name)
    resp = await api.post(run_id)
    report_id = resp.json()["report_id"]
    _complete(api.sm, report_id, {"k": k, "ceiling": 1.0})

    fresh = (await api.get(run_id)).json()
    assert (fresh["status"], fresh["freshness"], fresh["stale_reasons"]) == (
        "completed", "fresh", [],
    )  # fmt: skip
    assert fresh["report"]["recommendation"] == {"k": k, "ceiling": 1.0}

    api.sm.update_strategy(api.strategies[name], dsl_config={
        **RSI_DSL, "name": name, "take_profit": {"type": "fixed_pct", "percent": 4.0},
    })  # fmt: skip
    stale = (await api.get(run_id)).json()
    assert (stale["freshness"], stale["stale_reasons"]) == ("stale", ["dsl_changed"])
    assert stale["report"]["recommendation"] is None


async def test_get_missing_and_empty(api: Api) -> None:
    assert (await api.get(999_999)).status_code == 404
    assert (await api.get(2**70)).status_code == 404
    empty = await api.get(api.parent())
    assert empty.status_code == 200 and empty.json() is None


FRONTEND_DSL: dict[str, object] = {
    "general": {"timeframe": "4h"},
    "indicators": [{"type": "RSI", "name": "rsi", "params": {"period": 14}}],
    "conditions": {"long_entry": [{"left": "rsi", "operator": "<", "right": 30}]},
    "risk": {
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    },
}


@pytest.mark.parametrize("name", ["frontend_rsi_a", "frontend_rsi_b"])
async def test_frontend_dsl_hashes_with_the_strategy_name(api: Api, name: str) -> None:
    """A frontend-format DSL translates with the strategy's name; admission and GET use it.

    A rename changes the translated DSL, so the parent turns stale (re-validate first).
    """
    run_id = api.parent(name=name, dsl=FRONTEND_DSL)
    resp = await api.post(run_id)
    assert resp.status_code == 202, resp.text
    report_id = resp.json()["report_id"]
    _complete(api.sm, report_id, {"k": 1.0, "ceiling": 1.0})
    fresh = (await api.get(run_id)).json()
    assert (fresh["freshness"], fresh["report"]["recommendation"]) == (
        "fresh",
        {"k": 1.0, "ceiling": 1.0},
    )

    api.sm.update_strategy(api.strategies[name], name=f"{name}_renamed")
    stale = (await api.get(run_id)).json()
    assert (stale["freshness"], stale["stale_reasons"]) == ("stale", ["dsl_changed"])
    assert stale["report"]["recommendation"] is None


async def test_get_terminates_a_stale_heartbeat_worker_and_fails_the_report(api: Api) -> None:
    """C26 via the API: a live worker whose heartbeat is an hour old is SIGTERMed and the report fails."""
    run_id = api.parent()
    response = await api.post(run_id)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    pid = int(api.job(job_id)["pid"])
    api.sm.conn.execute(
        "UPDATE background_jobs SET heartbeat_at = datetime('now','-1 hour') WHERE run_id=?",
        (job_id,),
    )
    api.sm.conn.commit()
    out = (await api.get(run_id)).json()
    for _ in range(50):
        try:
            if os.waitpid(pid, os.WNOHANG)[0]:
                break
        except ChildProcessError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("stale worker was not terminated")
    assert out["status"] == "failed"
    assert "heartbeat stale" in out["error_message"]


async def test_get_fails_a_report_whose_worker_pid_was_reused(api: Api) -> None:
    """C26 via the API: a pid whose start time no longer matches is treated as dead (pid reuse)."""
    run_id = api.parent()
    response = await api.post(run_id)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    api.sm.conn.execute(
        "UPDATE background_jobs SET pid_start_time = 'bogus', heartbeat_at = datetime('now') WHERE run_id=?",
        (job_id,),
    )
    api.sm.conn.commit()
    out = (await api.get(run_id)).json()
    assert out["status"] == "failed"
    assert "pid reused" in out["error_message"]
