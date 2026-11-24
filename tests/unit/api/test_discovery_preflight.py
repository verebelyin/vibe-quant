"""POST /api/discovery/preflight: the power line a launch form shows before launch.

The body is the launch body (same validators) plus the preflight-only ``no_dsr``.
The catalog's 1m ranges are faked at ``CatalogManager.get_bar_date_range``; the
state DB is a tmp file, so the prior-trial lookup sees only rows a test inserts.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator  # noqa: TCH003
from datetime import UTC, datetime
from pathlib import Path  # noqa: TCH003
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.db.state_manager import StateManager

_RANGES = {
    "BTCUSDT": datetime(2020, 1, 1, 0, 0, 59, 999000, tzinfo=UTC),
    "HYPEUSDT": datetime(2025, 5, 30, 10, 30, 59, 999000, tzinfo=UTC),
}


@pytest.fixture()
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[AsyncClient, StateManager]]:
    from vibe_quant.api import data_precheck
    from vibe_quant.data.catalog import CatalogManager

    monkeypatch.setattr(data_precheck, "DEFAULT_CATALOG_PATH", tmp_path / "catalog")

    def _range(self: object, symbol: str, interval: str) -> tuple[datetime, datetime] | None:
        first = _RANGES.get(symbol)
        return None if first is None else (first, datetime(2026, 11, 10, tzinfo=UTC))

    monkeypatch.setattr(CatalogManager, "get_bar_date_range", _range)
    app = create_app()
    state = StateManager(db_path=tmp_path / "test.db")
    _ = state.conn
    app.state.state_manager = state
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, state
    state.close()


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "symbols": ["BTCUSDT"],
        "timeframes": ["4h"],
        "start_date": "2023-06-01",
        "end_date": "2026-06-01",
        "population": 50,
        "generations": 20,
    }
    body.update(overrides)
    return body


def _seed_prior_run(state: StateManager, run_n_eff: int) -> None:
    """One earlier non-mock 4h discovery run with an effective-N trial record."""
    from vibe_quant.discovery.neff import N_RULE_EFFECTIVE

    run_id = state.create_backtest_run(
        strategy_id=None, run_mode="discovery", symbols=["BTCUSDT"], timeframe="4h",
        start_date="2024-01-01", end_date="2025-01-01", parameters={},
    )  # fmt: skip
    notes = {
        "evaluated": run_n_eff * 2,
        "dsr_trials": {"n_rule": N_RULE_EFFECTIVE, "run_n_eff": run_n_eff},
    }
    state.conn.execute(
        "INSERT INTO backtest_results (run_id, notes) VALUES (?, ?)", (run_id, json.dumps(notes))
    )
    state.conn.commit()


async def test_preflight_reports_bars_holdout_and_full_history(
    client: tuple[AsyncClient, StateManager],
) -> None:
    ac, state = client
    _seed_prior_run(state, 37)
    r = await ac.post("/api/discovery/preflight", json=_body())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["train_days"] == 876
    assert data["holdout_days"] == 220
    assert data["budget"] == 1000
    assert data["prior_n"] == 37
    assert data["bar_a"]["n_trials"] == 38
    assert data["bar_b"]["n_trials"] == 1037
    assert data["full_history_start"] == "2020-01-01"
    assert data["sharpe_basis"] == "sqrt252"
    assert data["line"].startswith("Power: DSR bar ")
    assert data["line"].count("(252-day basis)") == 1

    r = await ac.post(
        "/api/discovery/preflight",
        json=_body(symbols=["BTCUSDT", "HYPEUSDT"], population=20, generations=15),
    )
    assert r.status_code == 200, r.text
    assert r.json()["full_history_start"] == "2025-05-31"
    assert r.json()["bar_b"]["n_trials"] == 337


async def test_no_dsr_replaces_the_bars(client: tuple[AsyncClient, StateManager]) -> None:
    ac, _ = client
    r = await ac.post("/api/discovery/preflight", json=_body(no_dsr=True))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["bar_a"] == "DSR off"
    assert data["bar_b"] == "DSR off"
    assert data["warning"] is None
    assert data["line"].startswith("Power: DSR off; holdout 220 days")


async def test_prior_lookup_failure_is_unknown(
    client: tuple[AsyncClient, StateManager], monkeypatch: pytest.MonkeyPatch
) -> None:
    ac, state = client

    def _boom(*args: object) -> object:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(state, "list_prior_discovery_notes", _boom)
    r = await ac.post("/api/discovery/preflight", json=_body())
    assert r.status_code == 200, r.text
    assert r.json()["prior_n"] == "unknown"
    assert r.json()["bar_a"]["n_trials"] == 1


@pytest.mark.parametrize(
    ("dates", "both"),
    [
        (("2025-01-01", "2025-01-21"), ("2025-01-01", "2025-01-21")),  # 20-day range
        (("2025-03-01", "2025-03-01"), ("2025-03-01", "2025-03-01")),  # end == start
        (("2025-03-02", "2025-03-01"), ("2025-03-02", "2025-03-01")),  # end < start
    ],
    ids=["20-day", "end-equals-start", "end-before-start"],
)
async def test_short_or_inverted_range_is_a_plain_422(
    client: tuple[AsyncClient, StateManager],
    dates: tuple[str, str],
    both: tuple[str, str],
) -> None:
    ac, _ = client
    r = await ac.post(
        "/api/discovery/preflight", json=_body(start_date=dates[0], end_date=dates[1])
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str)
    assert both[0] in detail and both[1] in detail
    assert "Traceback" not in detail


async def test_thirty_day_train_range_is_accepted(
    client: tuple[AsyncClient, StateManager],
) -> None:
    """Train exactly 30 days (split 0) is the edge that passes."""
    ac, _ = client
    r = await ac.post(
        "/api/discovery/preflight",
        json=_body(start_date="2025-01-01", end_date="2025-01-31", train_test_split=0),
    )
    assert r.status_code == 200, r.text
    assert r.json()["train_days"] == 30
    r = await ac.post(
        "/api/discovery/preflight",
        json=_body(start_date="2025-01-01", end_date="2025-01-30", train_test_split=0),
    )
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == (
        "train range 2025-01-01 to 2025-01-30 is 29 days; the power preflight needs at least 30"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"population": 0},
        {"population": True},
        {"population": "abc"},
        {"generations": -1},
        {"generations": 10**30},
        {"generations": True},
        {"no_dsr": "yes"},
        {"wfa_oos_step_days": 10**400},
        {"cross_window_months": [10**400]},
        {"num_seeds": 10**400},
        {"cross_window_min_sharpe_365": "NaN"},
        {"cross_window_min_sharpe_365": "-Infinity"},
        {"immigrant_fraction": "-Infinity"},
        {"symbols": []},
    ],
    ids=[
        "pop-0", "pop-bool", "pop-str", "gen-neg", "gen-huge", "gen-bool", "no-dsr-str",
        "wfa-huge", "cross-window-huge", "seeds-huge", "nan-string", "neg-inf-sharpe-string",
        "neg-inf-string", "no-symbols",
    ],
)  # fmt: skip
async def test_bad_input_is_a_422(
    client: tuple[AsyncClient, StateManager], overrides: dict[str, Any]
) -> None:
    ac, _ = client
    r = await ac.post("/api/discovery/preflight", json=_body(**overrides))
    assert r.status_code == 422, r.text


@pytest.mark.parametrize(
    "overrides",
    [{"wfa_oos_step_days": 10**9}, {"cross_window_months": [2**62]}],
    ids=["wfa-step", "cross-window"],
)
async def test_calendar_overflow_is_a_plain_422(
    client: tuple[AsyncClient, StateManager], overrides: dict[str, Any]
) -> None:
    """A representable window size past the calendar is a plain 422, never a 500."""
    ac, _ = client
    r = await ac.post("/api/discovery/preflight", json=_body(**overrides))
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str)
    assert detail.startswith("range 2023-06-01 to 2026-06-01: "), detail


async def test_non_finite_number_is_a_422(client: tuple[AsyncClient, StateManager]) -> None:
    ac, _ = client
    raw = json.dumps(_body()).replace('"population": 50', '"mutation_rate": NaN, "population": 50')
    r = await ac.post(
        "/api/discovery/preflight", content=raw, headers={"content-type": "application/json"}
    )
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400"])
async def test_non_finite_sharpe_limit_literal_is_a_422(
    client: tuple[AsyncClient, StateManager], token: str
) -> None:
    ac, _ = client
    raw = json.dumps(_body()).replace(
        '"population": 50', f'"cross_window_min_sharpe_365": {token}, "population": 50'
    )
    r = await ac.post(
        "/api/discovery/preflight", content=raw, headers={"content-type": "application/json"}
    )
    assert r.status_code == 422, r.text


async def test_unknown_symbol_is_named(client: tuple[AsyncClient, StateManager]) -> None:
    """An unknown coin fails the launch validators; a known coin with no bars fails coverage."""
    ac, _ = client
    r = await ac.post("/api/discovery/preflight", json=_body(symbols=["BTCUSDT", "NOPEUSDT"]))
    assert r.status_code == 422, r.text
    assert "NOPEUSDT" in str(r.json()["detail"])
    r = await ac.post("/api/discovery/preflight", json=_body(symbols=["BTCUSDT", "SOLUSDT"]))
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "SOLUSDT: no 1m bars in the catalog"


# --- GET /api/discovery/full-history-start: coverage, independent of the range ---


async def test_full_history_start_ignores_the_range(
    client: tuple[AsyncClient, StateManager],
) -> None:
    """A range the preflight refuses still gets the coverage start (B5)."""
    ac, _ = client
    r = await ac.post(
        "/api/discovery/preflight",
        json=_body(symbols=["BTCUSDT", "HYPEUSDT"], start_date="2026-10-21", end_date="2026-11-10"),
    )
    assert r.status_code == 422, r.text
    r = await ac.get("/api/discovery/full-history-start", params={"symbols": "BTCUSDT,HYPEUSDT"})
    assert r.status_code == 200, r.text
    assert r.json() == {"full_history_start": "2025-05-31"}
    r = await ac.get("/api/discovery/full-history-start", params={"symbols": "BTCUSDT"})
    assert r.json() == {"full_history_start": "2020-01-01"}


async def test_full_history_start_clamps_to_mark(
    client: tuple[AsyncClient, StateManager], monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibe_quant.data import mark_bars

    class _Series:
        ts = [int(datetime(2020, 3, 4, 0, 1, tzinfo=UTC).timestamp()) * 1_000_000_000]

    monkeypatch.setattr(mark_bars, "load_mark_series", lambda path, symbol: _Series())
    ac, _ = client
    r = await ac.get(
        "/api/discovery/full-history-start", params={"symbols": "BTCUSDT", "mark": "true"}
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"full_history_start": "2020-03-05"}


@pytest.mark.parametrize(
    ("symbols", "detail"),
    [
        ("BTCUSDT,SOLUSDT", "SOLUSDT: no 1m bars in the catalog"),
        ("", "symbols must name at least one symbol"),
        (" , ", "symbols must name at least one symbol"),
    ],
    ids=["uncovered", "empty", "blank-items"],
)
async def test_full_history_start_refuses_plainly(
    client: tuple[AsyncClient, StateManager], symbols: str, detail: str
) -> None:
    ac, _ = client
    r = await ac.get("/api/discovery/full-history-start", params={"symbols": symbols})
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == detail


# --- --fitness v2 refusals reach the preflight too (vibe-quant-clmmc.3 T4) -----------------


@pytest.mark.parametrize(
    ("extra", "detail"),
    [
        (
            {"fitness": "v2", "evaluator": "fast"},
            "--fitness v2 needs --evaluator nt (the fast-tier confirm compares every fitness "
            "field)",
        ),
        (
            {"fitness": "v2", "null_gate": True},
            "--fitness v2 cannot run with --null-gate (the timing-null re-run compares every "
            "fitness field)",
        ),
    ],
    ids=["fast", "null-gate"],
)  # fmt: skip
async def test_fitness_v2_preflight_refusal_is_422(
    client: tuple[AsyncClient, StateManager], extra: dict[str, Any], detail: str
) -> None:
    """R26: the preflight refuses with the launch's own texts (one shared rule, one message)."""
    ac, _ = client
    r = await ac.post("/api/discovery/preflight", json=_body(**extra))
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == detail


async def test_fitness_v2_preflight_of_a_clean_launch_reports_normally(
    client: tuple[AsyncClient, StateManager],
) -> None:
    """A clean v2 body is not refused: the power preflight is reported exactly like v1's."""
    ac, _ = client
    v1 = await ac.post("/api/discovery/preflight", json=_body())
    v2 = await ac.post("/api/discovery/preflight", json=_body(fitness="v2"))
    assert v2.status_code == 200, v2.text
    assert v2.json()["line"] == v1.json()["line"]
    assert v2.json()["budget"] == v1.json()["budget"]


async def test_stop_cap_preflight_refusal_is_422(client: tuple[AsyncClient, StateManager]) -> None:
    """R24: the preflight refuses with the launch's own stop-cap text, off mode only."""
    ac, _ = client
    r = await ac.post("/api/discovery/preflight", json=_body(leverage=107))
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == (
        "--leverage 107 with liquidation off liquidates a BTCUSDT short 0.53% from entry; a 95% "
        "stop cap (0.50%) leaves no room above the smallest stop discovery draws (0.5%). Turn "
        "liquidation on (--liquidation-mode isolated or cross) or lower --leverage."
    )
    on = await ac.post(
        "/api/discovery/preflight", json=_body(leverage=107, liquidation_mode="isolated")
    )
    assert "smallest stop" not in on.text  # past the cap check (this fake catalog has no bars)
