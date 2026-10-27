"""Unknown top-level strategy params fail closed in screening and at the launch API.

Beads vibe-quant-fahwt (screening read only ``parameters.sweep``; every other top-level
key was silently ignored) and vibe-quant-47984 (the launch API accepted the same key and
created a run row). The rule is validation's (bead vibe-quant-nxmxd.5): the run's own
settings (:data:`vibe_quant.launch_settings.RUN_LEVEL_KEYS`) are split off, every other
key must name a compiled strategy-config field (dot notation and aliases resolved) or the
run fails closed -- screening in the run itself, the API with a 422 before any
``backtest_runs`` row exists.

The screening entry point also consumes/writes its own metadata keys
(:data:`vibe_quant.launch_settings.SCREENING_ONLY_KEYS`); a stored screening row
legitimately carries them and re-running it must not fail.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import AsyncIterator  # noqa: TCH003
from pathlib import Path  # noqa: TCH003
from typing import Any  # noqa: TCH003
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.api.ws.manager import ConnectionManager
from vibe_quant.db.state_manager import StateManager
from vibe_quant.jobs.manager import BacktestJobManager


def _dsl(name: str = "unkparam_probe", timeframe: str = "1m") -> dict[str, object]:
    return {
        "name": name,
        "timeframe": timeframe,
        "indicators": {"rsi": {"type": "RSI", "period": 14}},
        "entry_conditions": {"long": ["rsi < 30"]},
        "exit_conditions": {"long": ["rsi > 70"]},
        "stop_loss": {"type": "fixed_pct", "percent": 2.0},
        "take_profit": {"type": "fixed_pct", "percent": 3.0},
    }


# --- screening CLI (fahwt) -------------------------------------------------


def _seed_screening_run(
    db: Path, parameters: dict[str, object], *, dsl: dict[str, object] | None = None
) -> int:
    state = StateManager(db)
    strategy_id = None
    if "dsl_override" not in parameters:
        strategy_id = state.create_strategy(name="unkparam_probe", dsl_config=dsl or _dsl())
    run_id = state.create_backtest_run(
        strategy_id=strategy_id,
        run_mode="screening",
        symbols=["BTCUSDT"],
        timeframe="1m",
        start_date="2025-01-01",
        end_date="2025-02-01",
        parameters=parameters,
    )
    state.close()
    return run_id


def _run_cli(
    db: Path, run_id: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, MagicMock]:
    """Run ``screening run`` with the pipeline factory replaced by a recorder."""
    from vibe_quant.screening import __main__ as cli
    from vibe_quant.screening import pipeline as pipeline_mod

    factory = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(pipeline_mod, "create_screening_pipeline", factory)
    code = cli.cmd_run(argparse.Namespace(run_id=run_id, db=str(db)))
    return code, factory


def _run_row(db: Path, run_id: int) -> dict[str, Any]:
    state = StateManager(db)
    row = state.get_backtest_run(run_id)
    state.close()
    assert row is not None
    return dict(row)


def _run_cli_real_pipeline(
    db: Path, run_id: int, monkeypatch: pytest.MonkeyPatch
) -> int:
    """``screening run`` end to end: CLI resolution, real pipeline, real DB rows.

    Only the ENGINE is replaced (``use_mock=True`` -> ``_run_mock_backtest``): the
    factory, parameter grid, ranking and ``save_results`` are the production ones,
    so the assertion is about what a real run writes into ``sweep_results``.
    """
    from vibe_quant.screening import __main__ as cli
    from vibe_quant.screening import pipeline as pipeline_mod

    real_factory = pipeline_mod.create_screening_pipeline

    def factory_with_mock_engine(dsl: Any, **kwargs: Any) -> Any:
        return real_factory(dsl, use_mock=True, max_workers=1, **kwargs)

    monkeypatch.setattr(pipeline_mod, "create_screening_pipeline", factory_with_mock_engine)
    return cli.cmd_run(argparse.Namespace(run_id=run_id, db=str(db)))


def _stored_parameters_json(db: Path, run_id: int) -> list[str]:
    """The ``sweep_results.parameters`` text of a run, in row (insertion) order."""
    state = StateManager(db)
    try:
        rows = state.conn.execute(
            "SELECT parameters FROM sweep_results WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
    finally:
        state.close()
    return [str(row[0]) for row in rows]


def _sorted_stored_parameters(db: Path, run_id: int) -> list[dict[str, Any]]:
    """Stored parameters in a ranking-independent order (rows are saved by Sharpe)."""
    rows = [json.loads(raw) for raw in _stored_parameters_json(db, run_id)]
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


# --- recorded top-level overrides in sweep_results.parameters (na00j) --------


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [
        # A dot-notation override resolves to the compiled config field.
        ({"rsi.period": 10}, {"rsi_period": 10}),
        # Two unrelated overrides, one string and one float, stay typed.
        (
            {"stop_loss_type": "atr", "take_profit.percent": 3.5},
            {"stop_loss_type": "atr", "take_profit_percent": 3.5},
        ),
    ],
)
def test_screening_row_records_top_level_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parameters: dict[str, object],
    expected: dict[str, object],
) -> None:
    """A stored row must show every value it ran with, not only the swept grid point."""
    db = tmp_path / "s.db"
    run_id = _seed_screening_run(db, parameters)

    code = _run_cli_real_pipeline(db, run_id, monkeypatch)

    assert code == 0, _run_row(db, run_id)["error_message"]
    rows = _sorted_stored_parameters(db, run_id)
    assert rows == [expected]


@pytest.mark.parametrize(
    ("sweep", "expected"),
    [
        (
            {"rsi.period": [12, 14]},
            [
                {"rsi.period": 12, "stop_loss_type": "fixed_pct"},
                {"rsi.period": 14, "stop_loss_type": "fixed_pct"},
            ],
        ),
        # The underscore spelling of the same field records as given, too.
        (
            {"rsi_period": [12, 14]},
            [
                {"rsi_period": 12, "stop_loss_type": "fixed_pct"},
                {"rsi_period": 14, "stop_loss_type": "fixed_pct"},
            ],
        ),
    ],
)
def test_screening_sweep_rows_record_each_value_and_the_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sweep: dict[str, list[int]],
    expected: list[dict[str, object]],
) -> None:
    """Every combination's row carries its own grid value AND the fixed override."""
    db = tmp_path / "s.db"
    run_id = _seed_screening_run(
        db, {"sweep": sweep, "stop_loss_type": "fixed_pct"}
    )

    code = _run_cli_real_pipeline(db, run_id, monkeypatch)

    assert code == 0, _run_row(db, run_id)["error_message"]
    rows = _sorted_stored_parameters(db, run_id)
    assert rows == sorted(expected, key=lambda row: json.dumps(row, sort_keys=True))


@pytest.mark.parametrize(
    ("parameters", "expected_json"),
    [
        ({"sweep": {"rsi_period": [12, 14]}}, ['{"rsi_period": 12}', '{"rsi_period": 14}']),
        ({}, ["{}"]),
    ],
)
def test_screening_row_parameters_are_unchanged_without_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parameters: dict[str, object],
    expected_json: list[str],
) -> None:
    """A run with no top-level override stores byte-identical parameters to today."""
    db = tmp_path / "s.db"
    run_id = _seed_screening_run(db, parameters)

    code = _run_cli_real_pipeline(db, run_id, monkeypatch)

    assert code == 0, _run_row(db, run_id)["error_message"]
    # Exact JSON text, not a parsed equality: today's rows must stay byte-identical.
    assert sorted(_stored_parameters_json(db, run_id)) == sorted(expected_json)


def test_screening_unknown_top_level_param_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rsi_perid`` (a typo, never in ``sweep``) used to be silently ignored."""
    db = tmp_path / "s.db"
    run_id = _seed_screening_run(db, {"rsi_perid": 14})

    code, factory = _run_cli(db, run_id, monkeypatch)

    assert code == 1
    assert factory.call_count == 0
    row = _run_row(db, run_id)
    assert row["status"] == "failed"
    assert "rsi_perid" in str(row["error_message"])


def _run_cli_configs(
    db: Path, run_id: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, list[Any]]:
    """Run ``screening run`` through the REAL factory/grid/runner and collect each
    combination's compiled strategy config (engine execution replaced by a recorder).
    """
    import importlib

    from vibe_quant.screening import __main__ as cli
    from vibe_quant.screening import pipeline as pipeline_mod

    real_factory = pipeline_mod.create_screening_pipeline
    configs: list[Any] = []

    def capture(dsl: Any, **kwargs: Any) -> MagicMock:
        pipeline = real_factory(dsl, max_workers=1, **kwargs)
        for combination in pipeline._param_grid:
            runner: Any = pipeline._runner
            spec, _ = runner._build_engine_spec(combination)
            assert spec is not None
            strategy = spec.strategy_configs[0]
            module, name = strategy.config_path.split(":")
            configs.append(getattr(importlib.import_module(module), name)(**strategy.config))
        return MagicMock()

    monkeypatch.setattr(pipeline_mod, "create_screening_pipeline", capture)
    code = cli.cmd_run(argparse.Namespace(run_id=run_id, db=str(db)))
    return code, configs


@pytest.mark.parametrize(
    ("parameters", "field", "expected"),
    [
        ({"rsi.period": 10}, "rsi_period", 10),
        ({"stop_loss.percent": 3.5}, "stop_loss_percent", 3.5),
        # The sweep is numeric-only: strings and booleans must arrive typed, as in validation.
        ({"stop_loss_type": "atr"}, "stop_loss_type", "atr"),
        ({"take_profit_type": "atr"}, "take_profit_type", "atr"),
        ({"log_events": False}, "log_events", False),
        # A null/empty sweep container must not swallow the override.
        ({"sweep": None, "rsi.period": 10}, "rsi_period", 10),
        ({"sweep": [], "rsi.period": 10}, "rsi_period", 10),
        ({"sweep": {}, "rsi.period": 10}, "rsi_period", 10),
    ],
)
def test_screening_valid_top_level_param_reaches_the_compiled_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parameters: dict[str, object],
    field: str,
    expected: object,
) -> None:
    """Validation forwards these to its config with their type; screening does the same."""
    db = tmp_path / "s.db"
    run_id = _seed_screening_run(db, parameters)

    code, configs = _run_cli_configs(db, run_id, monkeypatch)

    assert code == 0, _run_row(db, run_id)["error_message"]
    assert len(configs) == 1
    actual = getattr(configs[0], field)
    assert actual == expected
    assert type(actual) is type(expected)
    assert _run_row(db, run_id)["status"] == "completed"


def test_screening_override_applies_to_every_sweep_combination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fixed override and a sweep over ANOTHER field compose: 2 combos, both use 10."""
    db = tmp_path / "s.db"
    run_id = _seed_screening_run(
        db, {"rsi.period": 10, "log_events": False, "sweep": {"stop_loss.percent": [2.0, 3.0]}}
    )

    code, configs = _run_cli_configs(db, run_id, monkeypatch)

    assert code == 0, _run_row(db, run_id)["error_message"]
    assert sorted(c.stop_loss_percent for c in configs) == [2.0, 3.0]
    assert [c.rsi_period for c in configs] == [10, 10]
    assert [c.log_events for c in configs] == [False, False]


@pytest.mark.parametrize("top", ["rsi.period", "rsi_period"])
@pytest.mark.parametrize("swept", ["rsi.period", "rsi_period"])
def test_screening_override_that_is_also_a_sweep_key_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, top: str, swept: str
) -> None:
    """No implicit precedence: the same field twice, in any spelling, fails closed."""
    db = tmp_path / "s.db"
    run_id = _seed_screening_run(db, {top: 10, "sweep": {swept: [14]}})

    code, factory = _run_cli(db, run_id, monkeypatch)

    assert code == 1
    assert factory.call_count == 0
    row = _run_row(db, run_id)
    assert row["status"] == "failed"
    assert f"'{top}'" in str(row["error_message"])
    assert f"'{swept}'" in str(row["error_message"])


@pytest.mark.parametrize("sweep", [[1, 2], "rsi_period", 5, True])
def test_screening_malformed_sweep_container_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sweep: object
) -> None:
    """A sweep that is not a mapping is refused, never ignored next to an override."""
    db = tmp_path / "s.db"
    run_id = _seed_screening_run(db, {"rsi.period": 10, "sweep": sweep})

    code, factory = _run_cli(db, run_id, monkeypatch)

    assert code == 1
    assert factory.call_count == 0
    row = _run_row(db, run_id)
    assert row["status"] == "failed"
    assert "parameters.sweep" in str(row["error_message"])


def test_runner_refuses_override_that_is_also_a_sweep_key() -> None:
    """The runner checks too, so a caller bypassing the CLI cannot get a silent precedence."""
    from vibe_quant.screening.nt_runner import NTScreeningRunner, UnknownStrategyParamError

    dsl = {**_dsl(), "sweep": {"rsi.period": [14]}}
    with pytest.raises(UnknownStrategyParamError, match=r"rsi\.period"):
        NTScreeningRunner(
            dsl, ["BTCUSDT"], "2025-01-01", "2025-02-01", strategy_overrides={"rsi_period": 10}
        )
    with pytest.raises(UnknownStrategyParamError, match="rsi_perid"):
        NTScreeningRunner(
            _dsl(), ["BTCUSDT"], "2025-01-01", "2025-02-01", strategy_overrides={"rsi_perid": 10}
        )


def test_screening_run_level_keys_and_sweep_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run's own settings are accepted and never forwarded into the strategy config."""
    db = tmp_path / "s.db"
    run_id = _seed_screening_run(
        db,
        {
            "sweep": {"rsi_period": [10, 14]},
            "leverage": 2,
            "initial_balance": 2000,
            "max_position_pct": 0.4,
            "fee_tier": "VIP0",
            "bnb_discount": True,
            "overfitting_filters": {"min_trades": 5},
            "latency_preset": "cloud",
        },
    )

    code, factory = _run_cli(db, run_id, monkeypatch)

    assert code == 0
    assert factory.call_count == 1
    assert factory.call_args.args[0].sweep == {"rsi_period": [10, 14]}


def test_screening_only_metadata_keys_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A discovery replay row carries dsl_override/replay_note/compiler_version/replay_drift."""
    db = tmp_path / "s.db"
    run_id = _seed_screening_run(
        db,
        {
            "dsl_override": _dsl("unkparam_override"),
            "replay_note": "discovery replay",
            "compiler_version": "abc123",
            "replay_drift": {"sharpe_delta": 0.01},
        },
    )

    code, factory = _run_cli(db, run_id, monkeypatch)

    assert code == 0
    assert factory.call_count == 1
    dsl = factory.call_args.args[0]
    assert dsl.name == "unkparam_override"
    assert dsl.sweep == {}


@pytest.mark.parametrize("key", ["purpose", "note", "of_filters"])
def test_screening_free_form_param_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    """Free-form row keys are not a loophole (user ruling 2026-10-27)."""
    db = tmp_path / "s.db"
    run_id = _seed_screening_run(db, {key: "free form"})

    code, factory = _run_cli(db, run_id, monkeypatch)

    assert code == 1
    assert factory.call_count == 0
    row = _run_row(db, run_id)
    assert row["status"] == "failed"
    assert key in str(row["error_message"])


# --- launch API (47984) ----------------------------------------------------


@pytest.fixture()
async def client(tmp_path: Path) -> AsyncIterator[tuple[Any, ...]]:
    app = create_app()
    state = StateManager(db_path=tmp_path / "test.db")
    _ = state.conn  # init schema
    jobs = BacktestJobManager(db_path=tmp_path / "test.db")
    ws = ConnectionManager()
    await ws.start()
    start_job = MagicMock(return_value=4242)
    jobs.start_job = start_job  # type: ignore[method-assign]

    app.state.state_manager = state
    app.state.job_manager = jobs
    app.state.ws_manager = ws

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, state, start_job

    await ws.stop()
    jobs.close()
    state.close()


def _body(
    state: StateManager,
    parameters: dict[str, object],
    *,
    name: str,
    dsl: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "strategy_id": state.create_strategy(name=name, dsl_config=dsl or _dsl()),
        "symbols": ["BTCUSDT"],
        "timeframe": "1m",
        "start_date": "2025-01-01",
        "end_date": "2025-02-01",
        "parameters": parameters,
    }


def _run_count(state: StateManager) -> int:
    return int(state.conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0])


@pytest.mark.parametrize("endpoint", ["screening", "validation"])
@pytest.mark.parametrize("key", ["rsi_perid", "purpose"])
async def test_launch_unknown_param_is_422_before_any_run(
    client: tuple[Any, ...], endpoint: str, key: str
) -> None:
    ac, state, start_job = client
    before = _run_count(state)

    r = await ac.post(
        f"/api/backtest/{endpoint}",
        json=_body(state, {key: 14 if key == "rsi_perid" else "free form"}, name=f"{endpoint}_{key}"),
    )

    assert r.status_code == 422, r.text
    assert key in r.json()["detail"]
    assert _run_count(state) == before
    assert start_job.call_count == 0


@pytest.mark.parametrize("endpoint", ["screening", "validation"])
async def test_launch_run_level_keys_still_launch(
    client: tuple[Any, ...], endpoint: str
) -> None:
    ac, state, start_job = client

    r = await ac.post(
        "/api/backtest/" + endpoint,
        json=_body(
            state,
            {"sweep": {"rsi_period": [10, 14]}, "initial_balance": 2000, "leverage": 2},
            name=f"runlevel_{endpoint}",
        ),
    )

    assert r.status_code == 201, r.text
    assert start_job.call_count == 1


async def test_launch_screening_checks_the_effective_dsl_override(
    client: tuple[Any, ...]
) -> None:
    """``parameters.dsl_override`` is the DSL the screening subprocess runs (gotcha 6062).

    The stored DSL has no ``rsi``; only the override does. Four cases pin that the
    override decides which keys resolve.
    """
    ac, state, start_job = client
    stored_no_rsi = {
        **_dsl("unkparam_stored_ema"),
        "indicators": {"ema": {"type": "EMA", "period": 20}},
        "entry_conditions": {"long": ["ema > 100"]},
        "exit_conditions": {"long": ["ema < 100"]},
    }

    ok = await ac.post(
        "/api/backtest/screening",
        json=_body(
            state,
            {"dsl_override": _dsl("unkparam_override_rsi"), "rsi.period": 10},
            name="override_rsi_ok",
            dsl=stored_no_rsi,
        ),
    )
    assert ok.status_code == 201, ok.text

    # The override has no ``ema``: resolved against the STORED DSL this would pass,
    # against the effective one it must fail.
    bad = await ac.post(
        "/api/backtest/screening",
        json=_body(
            state,
            {"dsl_override": _dsl("unkparam_override_rsi"), "ema.period": 30},
            name="override_rsi_bad",
            dsl=stored_no_rsi,
        ),
    )
    assert bad.status_code == 422, bad.text
    assert "ema.period" in bad.json()["detail"]


async def test_launch_validation_rejects_dsl_override(client: tuple[Any, ...]) -> None:
    """Validation never reads ``dsl_override``, so a row carrying one fails closed."""
    ac, state, start_job = client
    before = _run_count(state)

    r = await ac.post(
        "/api/backtest/validation",
        json=_body(state, {"dsl_override": _dsl("unkparam_val_override")}, name="val_override"),
    )

    assert r.status_code == 422, r.text
    assert "dsl_override" in r.json()["detail"]
    assert _run_count(state) == before
    assert start_job.call_count == 0


@pytest.mark.parametrize(
    "parameters",
    [
        {"rsi.period": 10, "sweep": {"rsi.period": [14]}},
        {"rsi.period": 10, "sweep": {"rsi_period": [14]}},
        {"rsi_period": 10, "sweep": {"rsi.period": [14]}},
        {"rsi_period": 10, "sweep": {"rsi_period": [14]}},
        {"rsi_period": 10, "sweep": [1, 2]},
    ],
)
async def test_launch_screening_override_sweep_overlap_is_422_before_any_run(
    client: tuple[Any, ...], parameters: dict[str, object]
) -> None:
    ac, state, start_job = client
    before = _run_count(state)

    r = await ac.post("/api/backtest/screening", json=_body(state, parameters, name="overlap"))

    assert r.status_code == 422, r.text
    assert "sweep" in r.json()["detail"]
    assert _run_count(state) == before
    assert start_job.call_count == 0


@pytest.mark.parametrize("sweep", [None, [], {}])
async def test_launch_screening_empty_sweep_with_override_launches(
    client: tuple[Any, ...], sweep: object
) -> None:
    ac, state, start_job = client

    r = await ac.post(
        "/api/backtest/screening",
        json=_body(state, {"sweep": sweep, "rsi.period": 10}, name="empty_sweep"),
    )

    assert r.status_code == 201, r.text
    assert start_job.call_count == 1
