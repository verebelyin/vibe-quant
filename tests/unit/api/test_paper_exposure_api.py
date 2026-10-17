"""/api/paper/start refuses a cap above the run's leverage (D-S5, task P1).

Contract rows P9 (422 before any credentials check or run row) and P10 (a default
launch is unchanged). The endpoint calls the one shared rule
:func:`vibe_quant.paper.config.check_paper_exposure` after the validation-run and
symbol checks and before ``_require_credentials`` / ``create_backtest_run``, so a
refusal costs no run row and no job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager

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


class Ctx:
    def __init__(
        self,
        ac: AsyncClient,
        state: StateManager,
        jobs: BacktestJobManager,
        tmp: Path,
    ) -> None:
        self.ac = ac
        self.state = state
        self.jobs = jobs
        self.tmp = tmp
        self.launched: list[list[str]] = []
        self.strategy_id = state.create_strategy("test_strategy", DSL)
        self.leverage_2 = self._run({"leverage": 2})
        self.default_run = self._run({})

    def _run(self, parameters: dict[str, Any]) -> int:
        rid = self.state.create_backtest_run(
            strategy_id=self.strategy_id,
            run_mode="validation",
            symbols=["BTCUSDT"],
            timeframe="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            parameters=dict(parameters),
        )
        self.state.update_backtest_run_status(rid, "completed")
        return rid

    def run_count(self) -> int:
        row = self.state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()
        return int(row[0])


@pytest.fixture()
async def ctx(api_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Ctx]:
    app = create_app()
    state = StateManager(db_path=api_env)
    _ = state.conn
    jobs = BacktestJobManager(db_path=api_env)
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
            return 4242

        monkeypatch.setattr(jobs, "start_job", fake_start_job)
        yield c
    await ws.stop()
    jobs.close()
    state.close()


async def test_start_422_before_row_and_credentials(
    ctx: Ctx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P9: with NO credentials set the refusal is still 422 (checked first); no row, no job."""
    monkeypatch.delenv("BINANCE_TESTNET_API_KEY")
    monkeypatch.delenv("BINANCE_TESTNET_API_SECRET")
    before = ctx.run_count()

    r = await ctx.ac.post(
        "/api/paper/start",
        json={"validation_run_id": ctx.leverage_2, "max_position_pct": 3.0},
    )

    assert r.status_code == 422, r.text
    assert r.json()["detail"] == (
        "paper refuses to start: max_position_pct (3.0) must not exceed leverage (2.0) "
        f"(max_position_pct from paper sizing override; leverage from validation run "
        f"{ctx.leverage_2})"
    )
    assert ctx.run_count() == before, "the refusal must not create a backtest run row"
    assert ctx.launched == [], "the refusal must not start a job"


async def test_start_default_unchanged(ctx: Ctx) -> None:
    """P10: a default launch (cap 0.5 <= leverage 10) still returns 201 and starts."""
    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.default_run})
    assert r.status_code == 201, r.text
    assert len(ctx.launched) == 1


def _launched_config(ctx: Ctx) -> dict[str, Any]:
    import json
    from pathlib import Path

    command = ctx.launched[-1]
    path = next(Path(part) for part in command if part.endswith(".json"))
    data: dict[str, Any] = json.loads(path.read_text())
    return data


async def test_start_forwards_max_margin_pct_as_a_number(ctx: Ctx) -> None:
    """The cushion override reaches the node's config file as a JSON number; absent stays absent."""
    r = await ctx.ac.post(
        "/api/paper/start", json={"validation_run_id": ctx.default_run, "max_margin_pct": 0.3}
    )
    assert r.status_code == 201, r.text
    assert _launched_config(ctx)["sizing"]["max_margin_pct"] == 0.3

    r = await ctx.ac.post("/api/paper/start", json={"validation_run_id": ctx.default_run})
    assert r.status_code == 201, r.text
    assert "max_margin_pct" not in _launched_config(ctx).get("sizing", {})


@pytest.mark.parametrize("bad", [0, -0.1, 1.01, True])
async def test_start_rejects_bad_max_margin_pct(ctx: Ctx, bad: object) -> None:
    before = ctx.run_count()
    r = await ctx.ac.post(
        "/api/paper/start", json={"validation_run_id": ctx.default_run, "max_margin_pct": bad}
    )
    assert r.status_code == 422, r.text
    assert ctx.run_count() == before and ctx.launched == []


async def test_start_accepts_max_margin_pct_of_exactly_one(ctx: Ctx) -> None:
    r = await ctx.ac.post(
        "/api/paper/start", json={"validation_run_id": ctx.default_run, "max_margin_pct": 1}
    )
    assert r.status_code == 201, r.text
