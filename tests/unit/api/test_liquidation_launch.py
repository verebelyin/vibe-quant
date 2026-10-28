"""Liquidation launch plumbing on the API (bead vibe-quant-2gyk1.1.10, task D1).

Contract rows E1-E8, RP1-RP3 and the two optional ``SweepResultResponse``
fields: ``liquidation_mode`` is a top-level request field, stored into the run
parameters only when on; a nested ``parameters.liquidation_mode`` is refused;
discovery forwards ``--liquidation-mode`` when on and ``--leverage`` when it is
not the default (the ``--liquidation-leverage`` spelling is never sent again);
and promote/replay inherit the parent's authoritative ``notes.liquidation`` /
``notes.leverage``.

C8-C12/R1/R3/A2/A3 (vibe-quant-6wq6w) add the one-leverage contract: ``leverage``
is the field (``liquidation_leverage`` a deprecated alias), an explicit ``null``
is a 422, disagreeing aliases are a 422, and an off-mode leverage is inherited.

The mode/leverage DOMAIN is tabled once at ``tests/unit/test_liquidation_settings.py``;
the rows here are the WIRING (422 + nothing launches) and the stored-parameter shapes, so
each route keeps one bad value. Promote and replay call the same
``record.child_params`` helper, so the notes-presence matrix runs through ONE route and the
replay route keeps one end-to-end wiring test.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator  # noqa: TCH003
from itertools import count
from pathlib import Path  # noqa: TCH003
from typing import Any  # noqa: TCH003
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api import mark_precheck
from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.validation.extraction import date_to_ns

_STRATEGY_SEQ = count()


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture()
async def client(
    tmp_db: Path,
) -> AsyncIterator[tuple[AsyncClient, StateManager, BacktestJobManager, Path]]:
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


@pytest.fixture(autouse=True)
def mark_precheck_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    """Record the MARK-coverage precheck (vibe-quant-592lt) calls of a launch.

    Every launch here would otherwise open the real ``data/catalog`` (a symlink
    into the datasets, and stale MARK data would 422 a mode-on launch). The
    recorded args are ``(catalog_path, symbols, start_ns, end_ns)``: a mode-on
    launch must check the request's symbols, a mode-off launch must not check at
    all.
    """
    calls: list[tuple[object, ...]] = []

    def _record(*args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(mark_precheck, "check_mark_coverage", _record)
    return calls


def _launch_body(
    state: StateManager,
    *,
    mode: str | None = None,
    parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    strategy_id = state.create_strategy(
        name=f"liq_api_{next(_STRATEGY_SEQ)}", dsl_config={}
    )
    body: dict[str, object] = {
        "strategy_id": strategy_id,
        "symbols": ["BTCUSDT"],
        "timeframe": "1m",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "parameters": parameters if parameters is not None else {"initial_balance": 1000},
    }
    if mode is not None:
        body["liquidation_mode"] = mode
    return body


def _run_count(state: StateManager) -> int:
    return int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])


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


# --- E1/E2: screening & validation storage ---


@pytest.mark.parametrize("endpoint", ["screening", "validation"])
async def test_mode_off_stored_params_unchanged(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
    mark_precheck_calls: list[tuple[object, ...]],
) -> None:
    """E1: field absent or "off" -> stored parameters identical to today."""
    ac, state, job_mgr, _db = client

    for mode in (None, "off"):
        with _PopenPatch(job_mgr) as popen:
            r = await ac.post(f"/api/backtest/{endpoint}", json=_launch_body(state, mode=mode))

        assert r.status_code == 201, r.text
        run = state.get_backtest_run(r.json()["id"])
        assert run is not None
        assert run["parameters"] == {"initial_balance": 1000}
        assert "liquidation_mode" not in run["parameters"]
        assert popen.call_count == 1
    # mode off: no MARK file is opened at all (M5)
    assert mark_precheck_calls == []


@pytest.mark.parametrize("endpoint", ["screening", "validation"])
async def test_mode_on_stored_in_params_leverage_untouched(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    endpoint: str,
    mark_precheck_calls: list[tuple[object, ...]],
) -> None:
    """E2: "cross" -> parameters.liquidation_mode="cross"; leverage stays as sent."""
    ac, state, job_mgr, _db = client
    body = _launch_body(state, mode="cross", parameters={"initial_balance": 1000, "leverage": 20})

    with _PopenPatch(job_mgr):
        r = await ac.post(f"/api/backtest/{endpoint}", json=body)

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["id"])
    assert run is not None
    assert run["parameters"]["liquidation_mode"] == "cross"
    assert run["parameters"]["leverage"] == 20
    if endpoint == "screening":
        # mode on: checked with the request's symbols and window (M4/M1)
        assert mark_precheck_calls == [
            (
                mark_precheck.DEFAULT_CATALOG_PATH,
                ["BTCUSDT"],
                date_to_ns("2025-01-01"),
                date_to_ns("2025-02-01"),
            )
        ]
    else:
        # this fixture's strategy DSL is {} -> unparseable, so the validation
        # precheck skips itself and the run fails later as today (M2c)
        assert mark_precheck_calls == []


async def test_uppercase_mode_is_422_with_no_run_row(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
) -> None:
    """E3: "ISOLATED" is not a valid mode -> 422, no run row."""
    ac, state, job_mgr, _db = client
    body = _launch_body(state, mode="ISOLATED")
    before = _run_count(state)

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/validation", json=body)

    assert r.status_code == 422, r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


async def test_params_key_rejected(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
) -> None:
    """E4: a nested parameters.liquidation_mode is refused, no run row.

    The mode is a top-level field; the nested key used to be silently ignored.
    """
    ac, state, job_mgr, _db = client
    body = _launch_body(
        state, parameters={"initial_balance": 1000, "liquidation_mode": "isolated"}
    )
    before = _run_count(state)

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/validation", json=body)

    assert r.status_code == 422, r.text
    assert "use the top-level liquidation_mode field" in r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


async def test_mode_on_leverage_over_max_is_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    mark_precheck_calls: list[tuple[object, ...]],
) -> None:
    """E5: mode on with parameters.leverage=200 -> 422 (MAX_LEVERAGE), no run row."""
    ac, state, job_mgr, _db = client
    body = _launch_body(
        state, mode="isolated", parameters={"initial_balance": 1000, "leverage": 200}
    )
    before = _run_count(state)

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/backtest/screening", json=body)

    assert r.status_code == 422, r.text
    assert _run_count(state) == before
    assert popen.call_count == 0
    # a pydantic 422 wins over the MARK precheck (M8)
    assert mark_precheck_calls == []


# --- E6/E7/E8: discovery launch ---


_DISCOVERY_BODY: dict[str, object] = {
    "symbols": ["BTCUSDT"],
    "timeframes": ["4h"],
    "population": 12,
    "generations": 10,
}


async def test_discovery_default_leaves_params_and_argv_unchanged(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    mark_precheck_calls: list[tuple[object, ...]],
) -> None:
    """C6: no leverage field -> no leverage/liquidation params keys, no CLI flags."""
    ac, state, job_mgr, _db = client

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json=dict(_DISCOVERY_BODY))

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["run_id"])
    assert run is not None
    assert "liquidation_mode" not in run["parameters"]
    assert "leverage" not in run["parameters"]
    argv = popen.call_args.args[0]
    assert "--liquidation-mode" not in argv
    assert "--leverage" not in argv
    assert "--liquidation-leverage" not in argv
    assert mark_precheck_calls == []


@pytest.mark.parametrize("field", ["leverage", "liquidation_leverage"])
async def test_discovery_off_leverage_params_and_argv(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    field: str,
    mark_precheck_calls: list[tuple[object, ...]],
) -> None:
    """C8: mode off with an explicit leverage -> params {"leverage": 50.0}, --leverage 50.0."""
    ac, state, job_mgr, _db = client

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json={**_DISCOVERY_BODY, field: 50})

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["run_id"])
    assert run is not None
    assert run["parameters"]["leverage"] == 50.0
    assert "liquidation_mode" not in run["parameters"]
    argv = popen.call_args.args[0]
    assert argv[argv.index("--leverage") + 1] == "50.0"
    assert "--liquidation-mode" not in argv
    assert "--liquidation-leverage" not in argv
    assert mark_precheck_calls == []


async def test_discovery_on_flags_and_params(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    mark_precheck_calls: list[tuple[object, ...]],
) -> None:
    """C9: isolated 20 -> params {"liquidation_mode","leverage"}, argv --leverage (no alias)."""
    ac, state, job_mgr, _db = client

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post(
            "/api/discovery/launch",
            json={**_DISCOVERY_BODY, "liquidation_mode": "isolated", "leverage": 20},
        )

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["run_id"])
    assert run is not None
    assert run["parameters"]["liquidation_mode"] == "isolated"
    assert run["parameters"]["leverage"] == 20.0

    argv = popen.call_args.args[0]
    assert argv[argv.index("--liquidation-mode") + 1] == "isolated"
    assert argv[argv.index("--leverage") + 1] == "20.0"
    assert "--liquidation-leverage" not in argv
    # mode on: MARK coverage is checked for the request's symbols (M3/M4)
    assert [call[1] for call in mark_precheck_calls] == [["BTCUSDT"]]


async def test_discovery_bad_leverage_is_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
) -> None:
    """E8/A3: a fractional leverage -> 422, no run row (the domain is tabled at the owner)."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post(
            "/api/discovery/launch",
            json={**_DISCOVERY_BODY, "liquidation_mode": "isolated", "leverage": 2.5},
        )

    assert r.status_code == 422, r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


async def test_discovery_off_bad_leverage_is_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
) -> None:
    """A3: the domain is checked in off mode too; a string is not a number."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json={**_DISCOVERY_BODY, "leverage": "20"})

    assert r.status_code == 422, r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


async def test_discovery_null_leverage_is_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
) -> None:
    """R1: an explicitly sent null is a 422, never a silent default of 10."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post("/api/discovery/launch", json={**_DISCOVERY_BODY, "leverage": None})

    assert r.status_code == 422, r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


async def test_discovery_alias_conflict_is_422(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
) -> None:
    """A2: leverage 3 with liquidation_leverage 20 -> 422."""
    ac, state, job_mgr, _db = client
    before = _run_count(state)

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post(
            "/api/discovery/launch",
            json={**_DISCOVERY_BODY, "leverage": 3, "liquidation_leverage": 20},
        )

    assert r.status_code == 422, r.text
    assert "conflicting leverage 3.0 and liquidation_leverage 20.0" in r.text
    assert _run_count(state) == before
    assert popen.call_count == 0


async def test_discovery_alias_equal_is_accepted(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
) -> None:
    """A2: both given but equal (3 vs 3.0) is fine."""
    ac, state, job_mgr, _db = client

    with _PopenPatch(job_mgr) as popen:
        r = await ac.post(
            "/api/discovery/launch",
            json={**_DISCOVERY_BODY, "leverage": 3, "liquidation_leverage": 3.0},
        )

    assert r.status_code == 201, r.text
    run = state.get_backtest_run(r.json()["run_id"])
    assert run is not None
    assert run["parameters"]["leverage"] == 3.0
    argv = popen.call_args.args[0]
    assert argv[argv.index("--leverage") + 1] == "3.0"


# --- RP1/RP2/RP3: promote & replay inheritance ---

_GENOME_DSL: dict[str, object] = {
    "name": "ga_liq_1",
    "strategy_type": "momentum",
    "entry_conditions": {"long": ["rsi < 30"]},
    "exit_conditions": {"long": ["rsi > 70"]},
}


def _create_discovery_run(state: StateManager, *, notes_extra: dict[str, object] | None = None) -> int:
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="discovery",
        symbols=["BTCUSDT"],
        timeframe="4h",
        start_date="2024-01-01",
        end_date="2025-01-01",
        parameters={"population": 20, "generations": 10},
    )
    state.update_backtest_run_status(run_id, "completed")
    payload: dict[str, object] = {"top_strategies": [{"dsl": dict(_GENOME_DSL), "score": 1.5, "trades": 42}]}
    if notes_extra:
        payload.update(notes_extra)
    state.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)",
        (run_id, json.dumps(payload)),
    )
    state.conn.commit()
    return run_id


_ABSENT = object()  # the notes key is not written at all (vs an explicit null)


def _presence_notes(liquidation: object, leverage: object) -> dict[str, object]:
    notes: dict[str, object] = {}
    if liquidation is not _ABSENT:
        notes["liquidation"] = liquidation
    if leverage is not _ABSENT:
        notes["leverage"] = leverage
    return notes


_ON_20: dict[str, object] = {"mode": "isolated", "leverage": 20}
_ON_NO_LEV: dict[str, object] = {"mode": "isolated"}
_ON_NULL_LEV: dict[str, object] = {"mode": "isolated", "leverage": None}
_ISO = "isolated"

# The ONE inheritance table (RP1/RP2/RP3, C10-C12): (notes.liquidation, notes.leverage,
# expected child params; None -> 409). A null anywhere fails closed, a nested leverage that
# is missing falls back to notes.leverage (else 10), two present-but-different leverages
# fail closed, and a default-leverage parent adds nothing. Both routes call the same
# ``record.child_params`` helper, so the rows run through ONE route; the replay path
# (the other route) has its own wiring test below.
_PRESENCE_MATRIX: list[tuple[object, object, dict[str, object] | None]] = [
    (_ABSENT, _ABSENT, {}),
    (_ABSENT, None, None),
    (_ABSENT, 10.0, {}),
    (_ABSENT, 3, {"leverage": 3.0}),
    (_ON_20, _ABSENT, {"liquidation_mode": _ISO, "leverage": 20.0}),
    (_ON_20, 20.0, {"liquidation_mode": _ISO, "leverage": 20.0}),
    (_ON_20, None, None),
    (_ON_20, 3, None),
    (_ON_NO_LEV, _ABSENT, {"liquidation_mode": _ISO, "leverage": 10.0}),
    (_ON_NO_LEV, None, None),
    (_ON_NO_LEV, 3, {"liquidation_mode": _ISO, "leverage": 3.0}),
    (_ON_NULL_LEV, _ABSENT, None),
    (_ON_NULL_LEV, None, None),
    (_ON_NULL_LEV, 3, None),
]


@pytest.mark.parametrize(("liquidation", "leverage", "expected"), _PRESENCE_MATRIX)
async def test_child_notes_presence_matrix(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    liquidation: object,
    leverage: object,
    expected: dict[str, object] | None,
) -> None:
    """B1/S1: notes.leverage / notes.liquidation absent vs null vs set."""
    ac, state, job_mgr, _db = client
    run_id = _create_discovery_run(state, notes_extra=_presence_notes(liquidation, leverage))

    with _PopenPatch(job_mgr, pid=4242):
        r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")

    if expected is None:
        assert r.status_code == 409, r.text
        assert f"discovery run {run_id} has an unreadable notes.liquidation" in r.json()["detail"]
        return
    assert r.status_code == 201, r.text
    child = state.get_backtest_run(r.json()["run_id"])
    assert child is not None
    inherited = {k: v for k, v in child["parameters"].items() if k in {"liquidation_mode", "leverage"}}
    assert inherited == expected


async def test_replay_child_params_set_the_screening_venue_leverage(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3 (venue half): a replay of an off/3 champion builds its screening runner at 3.

    The replay child stores {"leverage": 3.0}; the screening path that consumes child
    params (create_screening_pipeline -> LiquidationSettings.from_params) must hand
    exactly that to the NTScreeningRunner, i.e. the discovery runner's venue leverage.
    """
    from decimal import Decimal

    from tests.unit.test_fee_schedule_plumbing import _minimal_dsl
    from vibe_quant.screening import pipeline as pipeline_mod

    ac, state, job_mgr, _db = client
    run_id = _create_discovery_run(state, notes_extra={"leverage": 3.0})
    with _PopenPatch(job_mgr, pid=4242):
        r = await ac.post(f"/api/discovery/results/{run_id}/replay/0")
    assert r.status_code == 201, r.text
    child = state.get_backtest_run(r.json()["replay_run_id"])
    assert child is not None
    assert child["parameters"]["leverage"] == 3.0

    monkeypatch.setattr(pipeline_mod, "resolve_parent_fill_ticks", lambda *a, **k: None)
    pipeline = pipeline_mod.create_screening_pipeline(
        _minimal_dsl(), symbols=["BTCUSDT"], parameters=child["parameters"]
    )
    assert pipeline._runner._leverage == 3.0
    assert pipeline._runner._venue_config().default_leverage == Decimal("3")


@pytest.mark.parametrize(
    "malformed",
    ["junk", {"leverage": 20}],
    ids=["not-an-object", "missing-mode"],
)
async def test_child_malformed_liquidation_notes_is_409(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
    malformed: object,
) -> None:
    """RP3: an unreadable parent notes.liquidation fails closed (409)."""
    ac, state, job_mgr, _db = client
    run_id = _create_discovery_run(state, notes_extra={"liquidation": malformed})

    with _PopenPatch(job_mgr, pid=4242):
        r = await ac.post(f"/api/discovery/results/{run_id}/promote/0")

    assert r.status_code == 409, r.text
    assert f"discovery run {run_id} has an unreadable notes.liquidation" in r.json()["detail"]


# --- SweepResultResponse optional fields ---


async def test_sweep_response_exposes_optional_liquidation_fields(
    client: tuple[AsyncClient, StateManager, BacktestJobManager, Path],
) -> None:
    """The sweep response carries liquidation_count/forfeited; NULL rows -> None."""
    ac, state, _job_mgr, _db = client
    run_id = state.create_backtest_run(
        strategy_id=None,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="1m",
        start_date="2025-01-01",
        end_date="2025-02-01",
        parameters={"liquidation_mode": "isolated", "leverage": 20.0},
    )
    state.save_sweep_result(
        run_id,
        {"parameters": {"leverage": 20.0}, "liquidation_count": 3, "liquidation_forfeited": 1.5},
    )
    state.save_sweep_result(run_id, {"parameters": {"leverage": 10.0}})

    r = await ac.get(f"/api/results/runs/{run_id}/sweeps")

    assert r.status_code == 200, r.text
    rows = {row["parameters"]["leverage"]: row for row in r.json()}
    assert rows[20.0]["liquidation_count"] == 3
    assert rows[20.0]["liquidation_forfeited"] == 1.5
    assert rows[10.0]["liquidation_count"] is None
    assert rows[10.0]["liquidation_forfeited"] is None
