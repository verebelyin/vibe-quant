"""``entry_order`` / maker params on the backtest API.

Validation accepts ``entry_order`` / ``maker_offset_ticks`` /
``maker_entry_timeout_bars`` and stores them in the run parameters only when
post_only. Screening refuses a post-only entry both as a top-level field and
nested in ``parameters``. A post-only validation run created through the API
must make the paper resolve step raise (risk-officer contract).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.validation.runner import ValidationRunner

if TYPE_CHECKING:
    from vibe_quant.validation.results import ValidationResult

_VALID_DSL: dict = {
    "name": "entry_order_contract",
    "description": "test strategy",
    "version": 1,
    "timeframe": "1h",
    "additional_timeframes": [],
    "indicators": {"rsi": {"type": "RSI", "period": 14, "source": "close"}},
    "entry_conditions": {"long": ["rsi < 30"], "short": ["rsi > 70"]},
    "exit_conditions": {"long": ["rsi > 50"], "short": ["rsi < 50"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 4.0},
}


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
async def client(tmp_db: Path):
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
        yield ac, state_mgr, job_mgr, tmp_db

    await ws_mgr.stop()
    job_mgr.close()
    state_mgr.close()


_STRATEGY_SEQ = itertools.count(1)


def _body(
    state: StateManager,
    *,
    parameters: dict | None = None,
    dsl: dict | None = None,
    strategy_name: str = "entry_order_api",
    **extra: object,
) -> dict:
    body: dict = {
        "strategy_id": state.create_strategy(
            name=f"{strategy_name}-{next(_STRATEGY_SEQ)}", dsl_config=dsl or {}
        ),
        "symbols": ["BTCUSDT"],
        "timeframe": "1m",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "parameters": parameters if parameters is not None else {"initial_balance": 1000},
    }
    body.update(extra)
    return body


def _launch_patches(job_mgr):
    return (
        patch.object(job_mgr, "is_process_alive", return_value=True),
        patch("subprocess.Popen"),
    )


async def test_api_rejects_post_only_for_screening(client) -> None:
    ac, state, _job_mgr, _db = client

    top_level = await ac.post(
        "/api/backtest/screening", json=_body(state, entry_order="post_only")
    )
    assert top_level.status_code == 422
    assert "validation-only" in top_level.json()["detail"]

    nested = await ac.post(
        "/api/backtest/screening",
        json=_body(state, parameters={"initial_balance": 1000, "entry_order": "post_only"}),
    )
    assert nested.status_code == 422
    assert "validation-only" in nested.json()["detail"]


async def test_validation_launch_post_only_params_stored(client) -> None:
    ac, state, job_mgr, _db = client

    body = _body(state, entry_order="post_only", maker_offset_ticks=2, maker_entry_timeout_bars=5)
    with patch.object(job_mgr, "is_process_alive", return_value=True), patch(
        "subprocess.Popen"
    ) as mock_popen:
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/backtest/validation", json=body)

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["id"])
    assert run is not None
    assert run["parameters"]["entry_order"] == "post_only"
    assert run["parameters"]["maker_offset_ticks"] == 2
    assert run["parameters"]["maker_entry_timeout_bars"] == 5


async def test_default_launch_params_unchanged(client) -> None:
    ac, state, job_mgr, _db = client

    body = _body(state, parameters={"initial_balance": 1000, "leverage": 5})
    with patch.object(job_mgr, "is_process_alive", return_value=True), patch(
        "subprocess.Popen"
    ) as mock_popen:
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/backtest/validation", json=body)

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["id"])
    assert run is not None
    assert run["parameters"] == {"initial_balance": 1000, "leverage": 5}


async def test_post_only_validation_run_refused_by_paper(
    client, tmp_path: Path
) -> None:
    """Risk-officer contract: API-created post-only run -> paper resolve raises."""
    ac, state, job_mgr, db_path = client
    sid = state.create_strategy(name="entry_order_contract", dsl_config=_VALID_DSL)
    body = {
        "strategy_id": sid,
        "symbols": ["BTCUSDT"],
        "timeframe": "1m",
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "parameters": {"initial_balance": 1000},
        "entry_order": "post_only",
        "maker_offset_ticks": 2,
        "maker_entry_timeout_bars": 5,
    }
    with patch.object(job_mgr, "is_process_alive", return_value=True), patch(
        "subprocess.Popen"
    ) as mock_popen:
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/backtest/validation", json=body)
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    run = state.get_backtest_run(rid)
    assert run is not None
    assert run["parameters"]["entry_order"] == "post_only"  # as the API stored it
    state.update_backtest_run_status(rid, "completed")

    from vibe_quant.dsl.compiler import StrategyCompiler
    from vibe_quant.dsl.parser import validate_strategy_dict
    from vibe_quant.paper.config import (
        BinanceTestnetConfig,
        ConfigurationError,
        PaperTradingConfig,
    )
    from vibe_quant.paper.node import PaperTradingNode

    cfg = PaperTradingConfig(
        trader_id="PAPER-007",
        binance=BinanceTestnetConfig("key", "secret", testnet=True),
        strategy_id=sid,
        validation_run_id=rid,
        db_path=db_path,
        logs_path=tmp_path / "logs",
    )
    node = PaperTradingNode(cfg)
    dsl = validate_strategy_dict(_VALID_DSL)
    module = StrategyCompiler().compile_to_module(dsl)
    with pytest.raises(ConfigurationError, match="entry_order") as excinfo:
        node._resolve_run_config(module, dsl)
    assert str(rid) in str(excinfo.value)



# ---------------------------------------------------------------------------
# API-stored params through the REAL ValidationRunner._run_backtest (review F4)
# ---------------------------------------------------------------------------


async def _launch_and_run(
    client_parts: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **launch: object
) -> tuple[int, ValidationResult, list[str]]:
    """Launch a validation run through the API, then execute it with the real runner
    on a synthetic temp catalog. Returns (run_id, result, entry_order of every
    engine's strategy, in run order)."""
    import vibe_quant.data.catalog as catalog_mod
    from tests.unit.test_validation_maker_entries import (
        SYNTHETIC_DSL,
        write_synthetic_catalog,
    )

    ac, state, job_mgr, db = client_parts
    write_synthetic_catalog(tmp_path / "catalog")
    monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", tmp_path / "catalog")
    sid = state.create_strategy(name="maker_real_run_api", dsl_config=SYNTHETIC_DSL)
    body = {
        "strategy_id": sid,
        "symbols": ["BTCUSDT"],
        "timeframe": "1h",
        "start_date": "2024-01-01",
        "end_date": "2024-01-05",
        "parameters": {},
        **launch,
    }
    with patch.object(job_mgr, "is_process_alive", return_value=True), patch(
        "subprocess.Popen"
    ) as mock_popen:
        mock_popen.return_value.pid = 12345
        r = await ac.post("/api/backtest/validation", json=body)
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]

    # Which entry_order did each real engine's strategy actually get?
    seen: list[str] = []
    original = ValidationRunner._extract_results

    def spy(self: ValidationRunner, **kw: Any) -> ValidationResult:
        seen.extend(s.config.entry_order for s in kw["engine"].trader.strategies())
        return original(self, **kw)

    monkeypatch.setattr(ValidationRunner, "_extract_results", spy)
    runner = ValidationRunner(db_path=db, logs_path=tmp_path / "logs")
    try:
        result = runner.run(run_id=run_id)
    finally:
        runner.close()
    return run_id, result, seen


def _event_counts(tmp_path: Path, run_id: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in (tmp_path / "logs" / f"{run_id}.jsonl").read_text().splitlines():
        kind = json.loads(line)["event"]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


async def test_api_post_only_run_completes_with_twin_and_one_result(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """API-stored post_only params -> real engines: maker run, market twin, one result row."""
    run_id, result, seen = await _launch_and_run(
        client,
        tmp_path,
        monkeypatch,
        entry_order="post_only",
        maker_offset_ticks=1,
        maker_entry_timeout_bars=2,
    )
    _ac, state, _job_mgr, _db = client

    # the primary engine got post_only, the twin engine got market
    assert seen == ["post_only", "market"]

    run = state.get_backtest_run(run_id)
    assert run is not None and run["status"] == "completed", run
    assert result.total_trades > 0
    assert state.conn.execute(
        "SELECT COUNT(*) FROM backtest_results WHERE run_id = ?", (run_id,)
    ).fetchone()[0] == 1

    maker = result.notes["maker_entries"]
    assert isinstance(maker, dict)
    assert maker["submitted"] >= result.total_trades  # >= : some entries may never fill
    assert maker["filled"] >= 1
    assert 0.0 < maker["fill_rate"] <= 1.0
    twin = maker["market_twin"]
    assert set(twin) == {"sharpe_ratio", "total_return", "total_trades", "total_fees"}
    assert twin["total_trades"] > 0
    assert twin["total_fees"] != result.total_fees  # a different (market) run, not a copy

    # persisted notes carry the same maker block
    notes_raw = state.get_result_notes_raw(run_id)
    assert notes_raw is not None
    assert json.loads(notes_raw)["maker_entries"]["market_twin"] == twin

    # trades: only the post_only run's, with a maker entry leg (2 bp) vs taker exit (5 bp)
    stored = state.conn.execute(
        "SELECT entry_price, quantity, entry_fee, exit_fee FROM trades WHERE run_id = ?", (run_id,)
    ).fetchall()
    assert len(stored) == result.total_trades
    for entry_price, quantity, entry_fee, _exit_fee in stored:
        assert entry_fee == pytest.approx(entry_price * quantity * 0.0002, rel=1e-4)

    # the twin logged no trade events: one POSITION_OPEN per stored trade, not two
    assert _event_counts(tmp_path, run_id)["POSITION_OPEN"] == result.total_trades


async def test_api_market_run_has_no_twin_and_no_maker_notes(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id, result, seen = await _launch_and_run(client, tmp_path, monkeypatch)
    _ac, state, _job_mgr, _db = client

    assert seen == ["market"]  # a single engine, no twin
    run = state.get_backtest_run(run_id)
    assert run is not None and run["status"] == "completed", run
    assert "maker_entries" not in result.notes
    notes_raw = state.get_result_notes_raw(run_id)
    assert notes_raw is not None and "maker_entries" not in json.loads(notes_raw)
    assert state.conn.execute(
        "SELECT COUNT(*) FROM backtest_results WHERE run_id = ?", (run_id,)
    ).fetchone()[0] == 1
    assert _event_counts(tmp_path, run_id)["POSITION_OPEN"] == result.total_trades
