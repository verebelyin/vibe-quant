"""Discovery notes readers: literal outcome rows for the persisted-field contract.

Each row is the outcome MEASURED at the public edge (``DiscoveryRecord``, the promote /
replay / export endpoints, the seed loader, the prior-trial scan): the value a child run
inherits (fees, leverage, sizing, a grid champion's cell), the 409/422 a corrupt stored
field fails closed with, and the DSR trial count the next run is charged. The rabmo.8
goldens of whole API responses are gone; what remains is one row per field a reader owns.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.routers import discovery as router
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import record
from vibe_quant.discovery.campaign import extract_top_chromosomes
from vibe_quant.discovery.genome import chromosome_to_serializable
from vibe_quant.discovery.operators import (
    ConditionType,
    Direction,
    StrategyChromosome,
    StrategyGene,
)
from vibe_quant.discovery.record import (
    ChampionBankrollInvalid,
    DiscoveryRecord,
    RecordUnreadable,
)
from vibe_quant.discovery.seed import load_seed_chromosomes
from vibe_quant.jobs.manager import BacktestJobManager
from vibe_quant.screening.replay_drift import _load_discovery_metrics

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

# The app's regime classifier reads local market data; the stub below replaces it.
ROUTER_REGIME_SIGN = "vibe_quant.discovery.gate_chain.window_regime_sign"

# --- adapter block: the names a reader move (rabmo.8 T10/T11) flips --------------------------


def fee_params(state: StateManager, run_id: int) -> tuple[dict[str, object], str]:
    return DiscoveryRecord.load(state, run_id).fee_params()


def liquidation_params(
    state: StateManager, run_id: int, strategy_index: int | None = None
) -> dict[str, object]:
    return DiscoveryRecord.load(state, run_id).liquidation_params(strategy_index=strategy_index)


def sizing_params(
    state: StateManager, run_id: int, strategy_index: int | None = None
) -> dict[str, float]:
    return DiscoveryRecord.load(state, run_id).sizing_params(strategy_index=strategy_index)


def outcome(fn: Callable[[], object]) -> tuple[object, ...]:
    """``("ok", value)`` or ``("err", http status, detail)`` for a reader call.

    The record raises its own exceptions; the router maps them to the same 409/422
    status and byte-identical detail, pinned here by the literal rows.
    """
    try:
        return ("ok", fn())
    except RecordUnreadable as exc:
        return ("err", 409, exc.detail)
    except ChampionBankrollInvalid as exc:
        return ("err", 422, exc.detail)
    except HTTPException as exc:
        return ("err", exc.status_code, exc.detail)


def prior_trials(
    state: StateManager, run_id: int, timeframe: str, window: tuple[str, str]
) -> tuple[int | None, list[int], int, int]:
    return record.prior_discovery_trials(state, run_id, timeframe, window)


# --- shared fixtures: a temp DB and a real app over it -----------------------------------------

RUN = 7


def insert_discovery_run(
    state: StateManager,
    run_id: int,
    *,
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    parameters: dict[str, object],
    notes: str | None,
    job: bool,
) -> None:
    """A completed discovery run with an explicit id, its result row and a finished job row."""
    state.conn.execute(
        "INSERT INTO backtest_runs (id, strategy_id, run_mode, symbols, timeframe, start_date, "
        "end_date, parameters, status) VALUES (?, NULL, 'discovery', ?, ?, ?, ?, ?, 'completed')",
        (
            run_id,
            json.dumps(symbols),
            timeframe,
            start_date,
            end_date,
            json.dumps(parameters),
        ),
    )
    state.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)", (run_id, notes)
    )
    if job:
        state.conn.execute(
            "INSERT INTO background_jobs (run_id, pid, job_type, status, started_at, "
            "completed_at) VALUES (?, 1, 'discovery', 'completed', '2026-01-01 00:00:00', "
            "'2026-01-01 01:00:00')",
            (run_id,),
        )
    state.conn.commit()


def row_counts(state: StateManager) -> list[int]:
    return [
        int(state.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # noqa: S608
        for table in ("strategies", "backtest_runs")
    ]


def body_of(response: Any) -> object:
    try:
        return response.json()
    except ValueError:
        return {"text": response.text}


def _regime_sign_stub() -> Any:
    """Deterministic stand-in for the catalog-reading regime classifier.

    One regime for every window: no fixture below expects the opposing-regime gate to fire.
    """

    def sign(symbol: str, start_date: str, end_date: str) -> int:
        return 1

    return sign


@asynccontextmanager
async def api_client(
    db_path: Path,
) -> AsyncIterator[tuple[AsyncClient, StateManager, BacktestJobManager]]:
    """The real app on a temp DB; job launches return pid 4242 (no subprocess)."""
    app = create_app()
    state = StateManager(db_path=db_path)
    _ = state.conn
    jobs = BacktestJobManager(db_path=db_path)
    ws = ConnectionManager()
    await ws.start()
    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    try:
        with (
            patch.object(jobs, "start_job", return_value=4242),
            patch(ROUTER_REGIME_SIGN, side_effect=_regime_sign_stub()),
        ):
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                yield ac, state, jobs
    finally:
        await ws.stop()
        jobs.close()
        state.close()


@pytest.fixture()
def state(tmp_path: Path) -> Any:
    mgr = StateManager(db_path=tmp_path / "record.db")
    _ = mgr.conn
    yield mgr
    mgr.close()


def put(
    state: StateManager,
    notes: object,
    *,
    run_id: int = RUN,
    timeframe: str = "4h",
    symbols: list[str] | None = None,
    parameters: dict[str, object] | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2025-01-01",
    result_row: bool = True,
    job: bool = False,
) -> None:
    """A completed discovery run whose notes are ``notes`` (str stored verbatim, else JSON)."""
    text = notes if notes is None or isinstance(notes, str) else json.dumps(notes)
    insert_discovery_run(
        state,
        run_id,
        symbols=symbols or ["BTCUSDT"],
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters or {},
        notes=text,
        job=job,
    )
    if not result_row:
        state.conn.execute("DELETE FROM backtest_results WHERE run_id = ?", (run_id,))
        state.conn.commit()


def with_champions(extra: dict[str, object], *champions: object) -> dict[str, object]:
    return {**extra, "top_strategies": list(champions)}


def err(status: int, detail: str) -> tuple[object, ...]:
    return ("err", status, detail)


def unreadable(what: str, why: str) -> tuple[object, ...]:
    return err(409, f"discovery run {RUN} has an unreadable notes.{what}: {why}")


def no_bankroll(index: object) -> tuple[object, ...]:
    return err(422, f"discovery run {RUN} strategy {index} has no valid bankroll setup")


# A champion DSL as stored in a real notes era: the input the promote/replay readers expect.
_GENOME_DSL: dict[str, object] = {
    "name": "genome_6509e60a4ea9",
    "timeframe": "4h",
    "indicators": {
        "price_position_entry_0": {"type": "PRICE_POSITION", "period": 98},
        "rsi_exit_0": {"type": "RSI", "period": 38},
    },
    "entry_conditions": {"short": ["price_position_entry_0 crosses_below 0.3668"]},
    "exit_conditions": {"short": ["rsi_exit_0 >= 25.05"]},
    "stop_loss": {"type": "fixed_pct", "percent": 0.81},
    "take_profit": {"type": "fixed_pct", "percent": 7.27},
}


# --- R21: fee schedule ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("notes", "expected"),
    [
        ({}, ("ok", ({}, "legacy_default_vip0"))),
        ({"fee_schedule": None}, ("ok", ({}, "legacy_default_vip0"))),
        ({"fee_schedule": []}, unreadable("fee_schedule", "expected an object, got []")),
        (
            {"fee_schedule": {"tier": "VIP0", "bnb_discount": True}},
            ("ok", ({"bnb_discount": True}, "discovery_notes")),
        ),
    ],
    ids=["absent", "null", "list", "bnb_discount"],
)
def test_fee_rows(state: StateManager, notes: dict[str, object], expected: tuple[object, ...]) -> None:
    put(state, notes)
    assert outcome(lambda: fee_params(state, RUN)) == expected


# --- R22: liquidation ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("notes", "expected"),
    [
        ({}, ("ok", {})),
        ({"liquidation": None}, ("ok", {})),
        (
            {"leverage": None},
            unreadable("liquidation", "leverage must be a whole number in [1, 125], got None"),
        ),
        ({"leverage": 5}, ("ok", {"leverage": 5.0})),
        ({"liquidation": []}, unreadable("liquidation", "expected an object, got []")),
        (
            {"liquidation": {"mode": "isolated", "leverage": 20}},
            ("ok", {"liquidation_mode": "isolated", "leverage": 20.0}),
        ),
        (
            {"liquidation": {"mode": "isolated"}},
            ("ok", {"liquidation_mode": "isolated", "leverage": 10.0}),
        ),
        (
            {"liquidation": {"mode": "isolated", "leverage": 20}, "leverage": 5},
            unreadable("liquidation", "conflicting leverage 20 and liquidation_leverage 5"),
        ),
        (
            {"leverage": 200},
            unreadable("liquidation", "leverage must be a whole number in [1, 125], got 200"),
        ),
    ],
    ids=[
        "absent",
        "null",
        "top_leverage_null",
        "top_leverage_5",
        "list",
        "isolated_20",
        "isolated_default_leverage",
        "conflicting",
        "top_leverage_200",
    ],
)
def test_liquidation_rows(
    state: StateManager, notes: dict[str, object], expected: tuple[object, ...]
) -> None:
    put(state, notes)
    assert outcome(lambda: liquidation_params(state, RUN)) == expected


# --- R23 / R23b: sizing ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("notes", "expected"),
    [
        ({}, ("ok", {})),
        ({"sizing": None}, unreadable("sizing", "expected an object, got None")),
        ({"sizing": []}, unreadable("sizing", "expected an object, got []")),
        (
            {"sizing": {"max_position_pct": 0}},
            unreadable("sizing", "max_position_pct must be a number in (0, 125], got 0"),
        ),
        (
            {"sizing": {"max_position_pct": 20}},
            unreadable("sizing", "max_position_pct (20.0) must not exceed leverage (10.0)"),
        ),
        ({"sizing": {"risk_per_trade": 0.01}}, ("ok", {"risk_per_trade": 0.01})),
    ],
    ids=["absent", "null", "list", "pct_zero", "pct_over_default_leverage", "risk_per_trade"],
)
def test_sizing_rows(
    state: StateManager, notes: dict[str, object], expected: tuple[object, ...]
) -> None:
    put(state, notes)
    assert outcome(lambda: sizing_params(state, RUN)) == expected


def test_sizing_path_propagates_liquidation_409_text(state: StateManager) -> None:
    """R23b: sizing resolves the leverage through the liquidation reader; its 409 text wins."""
    put(
        state,
        {
            "sizing": {"risk_per_trade": 0.01},
            "liquidation": {"mode": "isolated", "leverage": 20},
            "leverage": 5,
        },
    )
    assert outcome(lambda: sizing_params(state, RUN)) == unreadable(
        "liquidation", "conflicting leverage 20 and liquidation_leverage 5"
    )


# --- R24: bankroll grid rows -----------------------------------------------------------------

_CELL = {"liquidation_mode": "isolated", "leverage": 5.0, "max_margin_pct": 0.5}


def grid(*champions: object, **extra: object) -> dict[str, object]:
    return {"bankroll_grid": {"n_cells": 1}, "top_strategies": list(champions), **extra}


@pytest.mark.parametrize(
    ("notes", "index", "expected"),
    [
        (grid({"bankroll": _CELL}), None, no_bankroll(None)),
        ({"bankroll_grid": None}, 0, no_bankroll(0)),  # top_strategies absent
        (grid({"bankroll": _CELL}), 3, no_bankroll(3)),  # index 3 of 1 champion
        (grid({"bankroll": _CELL}), -1, no_bankroll(-1)),
        (grid("x"), 0, no_bankroll(0)),  # champion not a dict
        (grid({"bankroll": {"liquidation_mode": "isolated"}}), 0, no_bankroll(0)),
    ],
    ids=[
        "index_none",
        "no_top_strategies",
        "index_out_of_range",
        "index_negative",
        "champion_not_dict",
        "bankroll_incomplete",
    ],
)
def test_grid_rows(
    state: StateManager, notes: dict[str, object], index: int | None, expected: tuple[object, ...]
) -> None:
    put(state, notes)
    assert outcome(lambda: liquidation_params(state, RUN, index)) == expected
    assert outcome(lambda: sizing_params(state, RUN, index)) == expected


def test_grid_valid_cell_and_null_sizing(state: StateManager) -> None:
    put(state, grid({"bankroll": _CELL}))
    assert outcome(lambda: liquidation_params(state, RUN, 0)) == (
        "ok",
        {"liquidation_mode": "isolated", "leverage": 5.0},
    )
    assert outcome(lambda: sizing_params(state, RUN, 0)) == ("ok", {"max_margin_pct": 0.5})

    state.conn.execute("DELETE FROM backtest_runs WHERE id = ?", (RUN,))
    state.conn.execute("DELETE FROM backtest_results WHERE run_id = ?", (RUN,))
    put(state, grid({"bankroll": _CELL}, sizing=None))
    assert outcome(lambda: sizing_params(state, RUN, 0)) == unreadable(
        "sizing", "expected an object, got None"
    )


# --- R25: promote precedence -------------------------------------------------------------------


def set_notes(state: StateManager, notes: object) -> None:
    state.conn.execute(
        "UPDATE backtest_results SET notes = ? WHERE run_id = ?", (json.dumps(notes), RUN)
    )
    state.conn.commit()


async def test_promote_precedence_no_rows(tmp_path: Path) -> None:
    """404 index -> 409 regime gate -> 409 fee -> 409 liquidation -> 409 sizing; first wins."""
    champion: dict[str, object] = {"dsl": _GENOME_DSL, "chromosome": {"direction": "short"}}
    faults: dict[str, object] = {"fee_schedule": [], "liquidation": [], "sizing": None}
    async with api_client(tmp_path / "p.db") as (ac, state, _jobs):
        put(state, {**faults, "top_strategies": [champion]}, timeframe="1m")
        start = row_counts(state)

        async def promote(index: int) -> tuple[int, str]:
            response = await ac.post(f"/api/discovery/results/{RUN}/promote/{index}")
            assert row_counts(state) == start  # no strategies / backtest_runs row on any error
            body = body_of(response)
            assert isinstance(body, dict)
            return response.status_code, str(body.get("detail"))

        assert await promote(5) == (404, "Strategy index out of range")
        status, detail = await promote(0)
        assert (status, detail.startswith("1m short champions must pass")) == (409, True)

        champion["chromosome"] = {"direction": "long"}
        set_notes(state, {**faults, "top_strategies": [champion]})
        assert await promote(0) == (
            409,
            f"discovery run {RUN} has an unreadable notes.fee_schedule: expected an object, got []",
        )
        del faults["fee_schedule"]
        set_notes(state, {**faults, "top_strategies": [champion]})
        assert await promote(0) == (
            409,
            f"discovery run {RUN} has an unreadable notes.liquidation: expected an object, got []",
        )
        del faults["liquidation"]
        set_notes(state, {**faults, "top_strategies": [champion]})
        assert await promote(0) == (
            409,
            f"discovery run {RUN} has an unreadable notes.sizing: expected an object, got None",
        )
        del faults["sizing"]
        set_notes(state, {**faults, "top_strategies": [champion]})
        response = await ac.post(f"/api/discovery/results/{RUN}/promote/0")
        assert response.status_code == 201
        assert row_counts(state) == [start[0] + 1, start[1] + 1]


# --- R26: holdout dates ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("holdout", "mode", "vrange", "expected"),
    [
        (["2025-01-01", "2025-03-01"], "validation", "holdout", ("2025-01-01", "2025-03-01", "holdout", ["2025-01-01", "2025-03-01"])),
        (["2025-01-01", "2025-03-01"], "validation", "full", ("2024-01-01", "2025-01-01", "full", ["2025-01-01", "2025-03-01"])),
        (["2025-01-01", "2025-03-01"], "screening", "holdout", ("2024-01-01", "2025-01-01", "full", ["2025-01-01", "2025-03-01"])),
        ("ABSENT", "validation", "holdout", ("2024-01-01", "2025-01-01", "full", None)),
        (["a", "b", "c"], "validation", "holdout", ("2024-01-01", "2025-01-01", "full", None)),
    ],
    ids=["holdout", "full_range", "screening", "absent", "three"],
)
async def test_holdout_dates_rows(
    tmp_path: Path,
    holdout: object,
    mode: str,
    vrange: str,
    expected: tuple[str, str, str, list[str] | None],
) -> None:
    notes: dict[str, object] = {"top_strategies": [{"dsl": _GENOME_DSL}]}
    if holdout != "ABSENT":
        notes["holdout_dates"] = holdout
    async with api_client(tmp_path / "h.db") as (ac, state, _jobs):
        put(state, notes)
        response = await ac.post(
            f"/api/discovery/results/{RUN}/promote/0?mode={mode}&validation_range={vrange}"
        )
        body = body_of(response)
        assert response.status_code == 201
        assert isinstance(body, dict)
        run = state.get_backtest_run(body["run_id"])
        assert run is not None
        source = run["parameters"]["promote_source"]
        assert (body["start_date"], body["end_date"], body["date_range"], source["holdout_dates"]) == expected
        assert (run["start_date"], run["end_date"]) == (expected[0], expected[1])


# --- R28: jobs seed ----------------------------------------------------------------------------


async def test_jobs_seed_rows(tmp_path: Path) -> None:
    async with api_client(tmp_path / "s.db") as (ac, state, _jobs):
        for run_id, seed in enumerate([7, True, "7", "ABSENT"], start=1):
            notes: dict[str, object] = {} if seed == "ABSENT" else {"seed": seed}
            put(state, notes, run_id=run_id, job=True)
        response = await ac.get("/api/discovery/jobs")
        seeds = {e["run_id"]: e["seed"] for e in response.json()}
        assert seeds == {1: 7, 2: None, 3: None, 4: None}


# --- R29: results shape ------------------------------------------------------------------------


async def test_results_shape_rows(tmp_path: Path) -> None:
    async with api_client(tmp_path / "r.db") as (ac, state, _jobs):
        put(state, {"top_strategies": {}, "reason": 5, "guardrail_rejections": {}})
        response = await ac.get(f"/api/discovery/results/{RUN}")
        assert response.status_code == 200
        assert response.json() == {"strategies": [], "reason": None, "guardrail_rejections": None}

        put(state, {"top_strategies": [], "reason": "why", "guardrail_rejections": []}, run_id=8)
        response = await ac.get("/api/discovery/results/8")
        assert response.json() == {"strategies": [], "reason": "why", "guardrail_rejections": []}

        response = await ac.get("/api/discovery/results/999")  # no row at all
        assert response.json() == {"strategies": [], "reason": None, "guardrail_rejections": None}


# --- R30: /results/latest ----------------------------------------------------------------------


async def test_latest_rows(tmp_path: Path) -> None:
    has_trades = {"top_strategies": [{"trades": 3, "sharpe": 1.0}]}
    zero_trade = {"top_strategies": [{"trades": 0, "sharpe": 0.1}]}
    async with api_client(tmp_path / "l.db") as (ac, state, _jobs):
        put(state, has_trades, run_id=1)
        put(state, zero_trade, run_id=2)
        put(state, "{not json", run_id=3)  # newest
        response = await ac.get("/api/discovery/results/latest")
        assert response.json()["strategies"] == has_trades["top_strategies"]

        state.conn.execute("DELETE FROM backtest_results WHERE run_id = 1")
        state.conn.commit()
        response = await ac.get("/api/discovery/results/latest")  # fallback: first valid, 0 trades
        assert response.json()["strategies"] == zero_trade["top_strategies"]

        put(state, {"top_strategies": ["not a dict"]}, run_id=4)  # newest: skipped
        response = await ac.get("/api/discovery/results/latest")
        assert response.status_code == 200
        assert response.json()["strategies"] == zero_trade["top_strategies"]

        odd_trades = {"top_strategies": [{"trades": "x"}]}
        put(state, odd_trades, run_id=5)  # newest: no trades pass, still the first valid row
        response = await ac.get("/api/discovery/results/latest")
        assert response.json()["strategies"] == odd_trades["top_strategies"]


# --- R31: seed loader --------------------------------------------------------------------------


def _long_chromosome() -> StrategyChromosome:
    return StrategyChromosome(
        entry_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.LT, 30.0)],
        exit_genes=[StrategyGene("RSI", {"period": 14.0}, ConditionType.GT, 70.0)],
        stop_loss_pct=2.0,
        take_profit_pct=3.0,
        direction=Direction.LONG,
    )


def test_seed_loader_rows(state: StateManager, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    assert load_seed_chromosomes(state, 99) is None  # no row

    put(state, "", run_id=1)
    put(state, None, run_id=2)
    assert load_seed_chromosomes(state, 1) is None
    assert load_seed_chromosomes(state, 2) is None
    assert not caplog.records

    put(state, "{not json", run_id=3)
    assert load_seed_chromosomes(state, 3) is None
    assert [r.getMessage() for r in caplog.records] == ["Failed to load seed chromosomes from run 3"]
    assert caplog.records[0].levelno == logging.WARNING

    put(state, {"top_strategies": [{"score": 1.0}, {"chromosome": {}}, {"chromosome": None}]}, run_id=4)
    assert load_seed_chromosomes(state, 4) is None  # entries without a usable chromosome skipped

    caplog.clear()
    both = _long_chromosome()
    both.direction = Direction.BOTH
    blob = chromosome_to_serializable(both)
    for key in [k for k in blob if k.startswith("short_")]:
        del blob[key]  # a pre-inydc both blob has no short-side keys at all
    put(state, {"top_strategies": [{"chromosome": blob}]}, run_id=5)
    loaded = load_seed_chromosomes(state, 5)
    assert loaded is not None
    assert [c.direction for c in loaded] == [Direction.LONG]  # legacy both loaded as long
    assert [r.getMessage() for r in caplog.records] == [
        "Warm-start: loaded 1 legacy direction=both seed chromosome(s) as long"
    ]


@pytest.mark.parametrize("notes", ["[]", {"top_strategies": [None]}, {"top_strategies": 5}])
def test_seed_loader_malformed_notes_load_nothing(
    state: StateManager, caplog: pytest.LogCaptureFixture, notes: object
) -> None:
    """Non-object notes or non-object entries: no seeds and a warning, never a crash."""
    caplog.set_level(logging.WARNING)
    put(state, notes)
    assert load_seed_chromosomes(state, RUN) is None
    assert [r.getMessage() for r in caplog.records] == [
        f"Failed to load seed chromosomes from run {RUN}"
    ]


# --- R32: campaign extract ----------------------------------------------------------------------


def test_campaign_extract_rows(state: StateManager, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    assert extract_top_chromosomes(99, state) == []  # no row

    put(state, "", run_id=1)
    assert extract_top_chromosomes(1, state) == []
    assert not caplog.records

    put(state, "{not json", run_id=2)
    assert extract_top_chromosomes(2, state) == []
    assert [r.getMessage() for r in caplog.records] == ["malformed notes JSON for run_id=2"]

    put(state, {"top_strategies": 5}, run_id=3)
    assert extract_top_chromosomes(3, state) == []
    put(state, {"top_strategies": None}, run_id=4)
    assert extract_top_chromosomes(4, state) == []

    put(state, {"top_strategies": [{"i": i} for i in range(5)]}, run_id=5)
    assert extract_top_chromosomes(5, state, top_k=3) == [{"i": 0}, {"i": 1}, {"i": 2}]

    caplog.clear()
    put(state, "[]", run_id=6)
    put(state, {"top_strategies": [{"i": 0}, None]}, run_id=7)
    assert extract_top_chromosomes(6, state) == []
    assert extract_top_chromosomes(7, state) == []
    assert [r.getMessage() for r in caplog.records] == [
        "malformed notes JSON for run_id=6",
        "malformed notes JSON for run_id=7",
    ]


# --- R33: replay-drift metrics reader -----------------------------------------------------------


def test_load_discovery_metrics_rows(state: StateManager) -> None:
    assert _load_discovery_metrics(state, 99, 0) is None  # no row
    put(state, None, run_id=1)
    assert _load_discovery_metrics(state, 1, 0) is None  # non-str notes
    put(state, "{not json", run_id=2)
    assert _load_discovery_metrics(state, 2, 0) is None
    put(state, "[]", run_id=3)
    assert _load_discovery_metrics(state, 3, 0) is None  # non-dict
    put(state, {"top_strategies": {}}, run_id=4)
    assert _load_discovery_metrics(state, 4, 0) is None  # non-list
    put(state, {"top_strategies": [{"sharpe": 1.0, "trades": 5}]}, run_id=5)
    assert _load_discovery_metrics(state, 5, 1) is None  # index out of range
    assert _load_discovery_metrics(state, 5, -1) is None
    put(state, {"top_strategies": ["x"]}, run_id=6)
    assert _load_discovery_metrics(state, 6, 0) is None  # non-dict entry

    put(
        state,
        {"top_strategies": [{"sharpe": 1.0, "trades": 5, "full_range_sharpe": 2.0, "full_range_trades": 9}]},
        run_id=7,
    )
    assert _load_discovery_metrics(state, 7, 0) == (2.0, 9, None)  # full_range_* preferred
    put(state, {"top_strategies": [{"sharpe": 1.0, "trades": 5}]}, run_id=8)
    assert _load_discovery_metrics(state, 8, 0) == (1.0, 5, None)  # pre-rewru fallback


def test_load_discovery_metrics_worst_multi_symbol_notes(state: StateManager) -> None:
    worst = "discovery reference Sharpe = worst symbol (symbol_agg=worst)."
    two = ["BTCUSDT", "ETHUSDT"]
    put(
        state,
        {"symbol_agg": "worst", "top_strategies": [{"sharpe": 1.0, "trades": 2, "full_range_trades_sum": 40}]},
        run_id=1,
        symbols=two,
    )
    assert _load_discovery_metrics(state, 1, 0) == (1.0, 40, worst)
    put(state, {"symbol_agg": "worst", "top_strategies": [{"sharpe": 1.0, "trades": 2}]}, run_id=2, symbols=two)
    assert _load_discovery_metrics(state, 2, 0) == (
        1.0,
        None,
        worst
        + " trade comparison skipped: stored trades are the MIN per symbol "
        "(no trades_sum persisted) while the replay is a portfolio run.",
    )
    put(state, {"symbol_agg": "worst", "top_strategies": [{"sharpe": 1.0, "trades": 2}]}, run_id=3)
    assert _load_discovery_metrics(state, 3, 0) == (1.0, 2, None)  # one symbol: no note


# --- R35: replay-drift merge --------------------------------------------------------------------

_DRIFT = {"drift": 0.5}


@pytest.mark.parametrize(
    ("notes", "expected"),
    [
        ("{not json", {"replay_drift": _DRIFT}),
        ("[]", {"replay_drift": _DRIFT}),
        ('{"a": 1}', {"a": 1, "replay_drift": _DRIFT}),
    ],
    ids=["bad_json", "list", "dict"],
)
def test_merge_replay_drift_rows(state: StateManager, notes: str, expected: dict[str, object]) -> None:
    put(state, notes)
    state.merge_replay_drift_payload(RUN, _DRIFT)
    assert state.get_result_notes_raw(RUN) == json.dumps(expected)


def test_merge_replay_drift_without_notes_goes_to_parameters(state: StateManager) -> None:
    put(state, "", parameters={"a": 1})  # result row, empty notes
    state.merge_replay_drift_payload(RUN, _DRIFT)
    run = state.get_backtest_run(RUN)
    assert run is not None
    assert run["parameters"] == {"a": 1, "replay_drift": _DRIFT}
    assert state.get_result_notes_raw(RUN) == ""

    put(state, "", run_id=8, parameters={"b": 2}, result_row=False)  # no result row at all
    state.merge_replay_drift_payload(8, _DRIFT)
    run = state.get_backtest_run(8)
    assert run is not None
    assert run["parameters"] == {"b": 2, "replay_drift": _DRIFT}


# --- R36: prior trials ---------------------------------------------------------------------------


_WINDOW = ("2024-01-01", "2025-01-01")
_EFF = "cumulative_effective_same_tf_overlapping_train"


def _prior_notes(**extra: object) -> dict[str, object]:
    return {"evaluated": 100, "dsr_trials": {"n_rule": _EFF, "run_n_eff": 40}, **extra}


def test_prior_trials_effective_and_legacy_rows(state: StateManager) -> None:
    put(state, _prior_notes(), run_id=1)  # charged run_n_eff
    put(state, {"evaluated": 50}, run_id=2)  # legacy: tallied, never charged
    put(state, _prior_notes(mock=True), run_id=3)  # mock skipped
    put(state, {"dsr_trials": {"n_rule": _EFF, "run_n_eff": 5}}, run_id=4)  # evaluated 0 skipped
    put(state, _prior_notes(), run_id=5, start_date="2020-01-01", end_date="2020-06-01")  # disjoint
    put(state, _prior_notes(), run_id=6, timeframe="1h")  # other timeframe
    assert prior_trials(state, 10, "4h", _WINDOW) == (40, [1], 1, 50)
    assert prior_trials(state, 1, "4h", _WINDOW) == (0, [], 0, 0)  # only earlier runs count


@pytest.mark.parametrize(
    ("run_n_eff", "charged"),
    [(1, 1), (100, 100), (0, 100), (101, 100), (None, 100)],
    ids=["lower_edge", "upper_edge", "zero", "over_evaluated", "null"],
)
def test_prior_trials_invalid_run_n_eff_charges_evaluated(
    state: StateManager, caplog: pytest.LogCaptureFixture, run_n_eff: object, charged: int
) -> None:
    caplog.set_level(logging.WARNING)
    put(state, {"evaluated": 100, "dsr_trials": {"n_rule": _EFF, "run_n_eff": run_n_eff}}, run_id=1)
    assert prior_trials(state, 10, "4h", _WINDOW) == (charged, [1], 0, 0)
    valid = charged == run_n_eff
    messages = [r.getMessage() for r in caplog.records]
    if valid:
        assert messages == []
    else:
        assert messages == [
            f"prior trial count: run 1 has an invalid run_n_eff {run_n_eff!r}; charging evaluated 100"
        ]


def test_prior_trials_garbled_rows_are_skipped_with_warning(
    state: StateManager, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    put(state, "{not json", run_id=1)
    put(state, {"evaluated": "many", "dsr_trials": {"n_rule": _EFF, "run_n_eff": 3}}, run_id=2)
    put(state, "[]", run_id=3)  # non-dict: skipped silently
    put(state, _prior_notes(), run_id=4)
    assert prior_trials(state, 10, "4h", _WINDOW) == (40, [4], 0, 0)
    assert [r.getMessage() for r in caplog.records] == [
        "prior trial count: run 1 garbled notes; skipped",
        "prior trial count: run 2 garbled evaluated count; skipped",
    ]


# --- R38d: a null champion is a 409, not a crash ------------------------------------------------


class _FakeState:
    """Just enough StateManager for the export endpoint: one result row."""

    def __init__(self, notes: str) -> None:
        self._notes = notes

    def get_backtest_result(self, run_id: int) -> dict[str, object]:
        return {"run_id": run_id, "notes": self._notes}


async def test_export_null_champion_is_a_409() -> None:
    """``top_strategies: [null]``, index 0: in range but not an object -> 409, not a 404."""
    fake = _FakeState(json.dumps({"top_strategies": [None]}))
    with pytest.raises(HTTPException) as null_info:
        await router.export_discovered_strategy(RUN, 0, fake)  # type: ignore[arg-type]
    assert null_info.value.status_code == 409
    assert null_info.value.detail == f"Stored strategy 0 of run {RUN} is not an object"
    with pytest.raises(HTTPException) as excinfo:  # the bounds check itself still 404s
        await router.export_discovered_strategy(RUN, 1, fake)  # type: ignore[arg-type]
    assert excinfo.value.status_code == 404


async def test_promote_and_replay_null_champion_are_409s(state: StateManager) -> None:
    put(state, {"top_strategies": [None]})
    for endpoint in (router.promote_discovered_strategy, router.replay_discovered_strategy):
        with pytest.raises(HTTPException) as excinfo:
            await endpoint(RUN, 0, state, MagicMock(), MagicMock())
        assert excinfo.value.status_code == 409
    assert row_counts(state) == [0, 1]  # no strategy, no new run


def test_wfa_entry_persists_consistency_and_window_dates() -> None:
    """The rolling-window summary the badge reads: consistency, counts, per-window dates."""
    from vibe_quant.discovery.gate_chain import HoldoutResult, WFARollingResult

    def window(sharpe: float, ret: float) -> HoldoutResult:
        return HoldoutResult(
            sharpe_ratio=sharpe, max_drawdown=0.04, profit_factor=1.3, total_trades=25,
            total_return=ret,
        )  # fmt: skip

    entry = record.wfa_entry(
        WFARollingResult(
            oos_windows=[window(0.8, 0.02), window(-0.3, -0.01)],
            window_dates=[("2026-01-01", "2026-02-01")],  # second window has no dates
            windows_profitable=1,
            windows_sharpe_positive=1,
            total_windows=2,
            consistency=0.5,
            sharpe_consistency=0.5,
            passed=False,
        )
    )
    assert entry["consistency"] == 0.5
    assert (entry["windows_profitable"], entry["total_windows"], entry["passed"]) == (1, 2, False)
    assert entry["windows"] == [
        {"dates": ("2026-01-01", "2026-02-01"), "sharpe": 0.8, "return_pct": 0.02, "trades": 25},
        {"dates": None, "sharpe": -0.3, "return_pct": -0.01, "trades": 25},
    ]


def test_no_champion_notes_persist_best_raw_score() -> None:
    notes = record.no_champion_notes(
        {"seed": 7}, best_raw_score=-0.25, guardrail_rejections=[]
    )
    assert notes["best_raw_score"] == -0.25
    assert notes["outcome"] == "no_viable_strategies"
