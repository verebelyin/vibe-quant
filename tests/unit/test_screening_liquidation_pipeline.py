"""Screening pipeline + CLI + sweep columns for opt-in liquidation (B1).

Contract rows: E10, D1, I1, I2, I7, I8, X1.

The one parse rule is :class:`LiquidationSettings` (bead vibe-quant-2gyk1.1.8):
``create_screening_pipeline`` derives it from the run's parameters (the single
source, no screening CLI flag) and passes it to ``NTScreeningRunner`` and to
``ScreeningPipeline``. A mode-on integrity failure
(:class:`~vibe_quant.errors.LiquidationIntegrityError`, a
``DataUnavailableError``) is re-raised by the sweep and fails the run; an
ordinary strategy error with the mode on still yields a -999 row and a
completed run (I7).
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

from httpx import ASGITransport, AsyncClient

from vibe_quant.api.app import create_app
from vibe_quant.db.state_manager import StateManager
from vibe_quant.dsl import parse_strategy_string
from vibe_quant.errors import LiquidationIntegrityError
from vibe_quant.liquidation_settings import LiquidationSettings
from vibe_quant.screening.pipeline import (
    FAILED_BACKTEST_SHARPE,
    ScreeningPipeline,
    create_screening_pipeline,
)
from vibe_quant.screening.types import BacktestMetrics, ScreeningResult

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_STRATEGY: dict[str, object] = {
    "name": "liq_probe",
    "timeframe": "4h",
    "indicators": {"rsi": {"type": "RSI", "period": 14}},
    "entry_conditions": {"long": ["rsi < 45"]},
    "exit_conditions": {"long": ["rsi > 55"]},
    "stop_loss": {"type": "fixed_pct", "percent": 2.0},
    "take_profit": {"type": "fixed_pct", "percent": 3.0},
}

_MINIMAL_YAML = """
name: liq_probe_min
timeframe: 5m
indicators:
  rsi:
    type: RSI
    period: 14
entry_conditions:
  long:
    - rsi < 30
stop_loss:
  type: fixed_pct
  percent: 2.0
take_profit:
  type: fixed_pct
  percent: 3.0
"""


def _dsl() -> Any:
    return parse_strategy_string(_MINIMAL_YAML)


# ---------------------------------------------------------------------------
# Picklable fake runners: they stand in for NTScreeningRunner in the pool.
# ---------------------------------------------------------------------------


class _MarkMissingRunner:
    """Raises the mode-on integrity error the runner wraps a MarkDataMissing in."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, params: dict[str, float | int]) -> BacktestMetrics:
        from vibe_quant.data.mark_bars import MarkDataMissing

        raise LiquidationIntegrityError.wrap(
            MarkDataMissing("no MARK bars for BTCUSDT-PERP"), "isolated"
        )


class _ExitTimingRunner:
    """A mode-on runner whose metrics carry the exit-timing counts (E8 transport)."""

    NOTES: dict[str, Any] = {
        "refined": 2,
        "kept_bar_close": 1,
        "by_symbol": {"BTCUSDT": {"refined": 2, "kept_bar_close": 1}},
    }

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, params: dict[str, float | int]) -> BacktestMetrics:
        return BacktestMetrics(parameters=params, sharpe_ratio=1.0, exit_timing=dict(self.NOTES))


class _PlainBoomRunner:
    """An ordinary strategy failure OUTSIDE the liquidation path (I7/I8)."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, params: dict[str, float | int]) -> BacktestMetrics:
        raise ValueError("ordinary strategy failure")


def _install_fake_runner(
    monkeypatch: pytest.MonkeyPatch, runner_cls: type, workers: int
) -> None:
    """Route the real factory onto ``runner_cls`` and force ``workers``."""
    from vibe_quant.screening import pipeline as pipeline_mod

    real_factory = pipeline_mod.create_screening_pipeline
    monkeypatch.setattr(pipeline_mod, "resolve_parent_fill_ticks", lambda *a, **k: None)
    monkeypatch.setattr(
        "vibe_quant.screening.nt_runner.NTScreeningRunner", runner_cls
    )

    def fake_factory(dsl: Any, **kwargs: Any) -> ScreeningPipeline:
        kwargs["max_workers"] = workers
        return real_factory(dsl, **kwargs)

    monkeypatch.setattr(pipeline_mod, "create_screening_pipeline", fake_factory)


def _make_run(db: Path, parameters: dict[str, object]) -> int:
    state = StateManager(db)
    try:
        strategy_id = state.create_strategy(name="liq_probe", dsl_config=_STRATEGY)
        return state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2024-03-01",
            parameters=parameters,
        )
    finally:
        state.close()


def _final_run(db: Path, run_id: int) -> dict[str, Any]:
    state = StateManager(db)
    try:
        run = state.get_backtest_run(run_id)
        rows = state.get_sweep_results(run_id)
    finally:
        state.close()
    assert run is not None
    return {"run": run, "rows": rows}


def _run_cli(db: Path, run_id: int) -> int:
    from vibe_quant.screening.__main__ import cmd_run

    return cmd_run(argparse.Namespace(run_id=run_id, db=str(db)))


# ---------------------------------------------------------------------------
# I1 / I2: a mode-on integrity failure fails the run (sequential and pooled)
# ---------------------------------------------------------------------------


def test_mode_on_mark_missing_fails_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I1: mode on, MarkDataMissing -> failed run, prefix message, 0 sweep rows, exit 1."""
    db = tmp_path / "i1.db"
    run_id = _make_run(db, {"liquidation_mode": "isolated"})
    _install_fake_runner(monkeypatch, _MarkMissingRunner, workers=1)

    assert _run_cli(db, run_id) == 1

    final = _final_run(db, run_id)
    assert final["run"]["status"] == "failed"
    assert final["run"]["error_message"].startswith(
        "LiquidationIntegrityError: liquidation_mode='isolated': MarkDataMissing: "
    )
    assert final["rows"] == []


def test_mode_on_mark_missing_fails_run_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I2: same as I1 but the integrity error travels through the worker pool."""
    db = tmp_path / "i2.db"
    run_id = _make_run(db, {"liquidation_mode": "isolated"})
    _install_fake_runner(monkeypatch, _MarkMissingRunner, workers=2)

    assert _run_cli(db, run_id) == 1

    final = _final_run(db, run_id)
    assert final["run"]["status"] == "failed"
    assert final["run"]["error_message"].startswith(
        "LiquidationIntegrityError: liquidation_mode='isolated': MarkDataMissing: "
    )
    assert final["rows"] == []


# ---------------------------------------------------------------------------
# E10: an invalid stored mode fails the run before any engine exists
# ---------------------------------------------------------------------------


def test_invalid_mode_fails_run_with_zero_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E10: DB params mode 'x' -> failed run, config-error message, 0 rows, exit 1."""
    db = tmp_path / "e10.db"
    run_id = _make_run(db, {"liquidation_mode": "x"})

    assert _run_cli(db, run_id) == 1

    final = _final_run(db, run_id)
    assert final["run"]["status"] == "failed"
    assert final["run"]["error_message"].startswith(
        "LiquidationConfigError: liquidation_mode must be one of "
    )
    assert final["rows"] == []


# ---------------------------------------------------------------------------
# I7 / I8: ordinary strategy errors keep the -999 sentinel
# ---------------------------------------------------------------------------


def test_mode_on_ordinary_error_is_999_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I7: mode on, an error outside the liquidation path -> -999 row, completed run."""
    db = tmp_path / "i7.db"
    run_id = _make_run(db, {"liquidation_mode": "isolated"})
    _install_fake_runner(monkeypatch, _PlainBoomRunner, workers=1)

    assert _run_cli(db, run_id) == 0

    final = _final_run(db, run_id)
    assert final["run"]["status"] == "completed"
    assert len(final["rows"]) == 1
    assert final["rows"][0]["sharpe_ratio"] == -999.0


# ---------------------------------------------------------------------------
# D1: sweep columns are written only when the mode is on
# ---------------------------------------------------------------------------


def test_save_results_writes_liquidation_only_when_on(tmp_path: Path) -> None:
    """D1: on -> int count (0 allowed) + float forfeited; off -> both NULL."""
    db = tmp_path / "d1.db"
    state = StateManager(db)
    try:
        strategy_id = state.create_strategy(name="liq_probe", dsl_config=_STRATEGY)

        def new_run(parameters: dict[str, object]) -> int:
            return state.create_backtest_run(
                strategy_id=strategy_id,
                run_mode="screening",
                symbols=["BTCUSDT"],
                timeframe="4h",
                start_date="2024-01-01",
                end_date="2024-03-01",
                parameters=parameters,
            )

        hit_run = new_run({"liquidation_mode": "isolated", "leverage": 50.0})
        miss_run = new_run({"liquidation_mode": "cross", "leverage": 20.0})
        off_run = new_run({})

        on = ScreeningPipeline(
            dsl=_dsl(), liquidation=LiquidationSettings(mode="isolated", leverage=50.0)
        )
        on.save_results(
            ScreeningResult(
                "liq_probe", 1, 1, 0.1,
                [BacktestMetrics(parameters={"a": 1}, sharpe_ratio=1.0,
                                 liquidation_count=3, liquidation_forfeited=12.5)],
                [0],
            ),
            state,
            hit_run,
        )
        on.save_results(
            ScreeningResult(
                "liq_probe", 1, 1, 0.1,
                [BacktestMetrics(parameters={"a": 1}, sharpe_ratio=1.0)],
                [0],
            ),
            state,
            miss_run,
        )
        ScreeningPipeline(dsl=_dsl()).save_results(
            ScreeningResult(
                "liq_probe", 1, 1, 0.1,
                [BacktestMetrics(parameters={"a": 1}, sharpe_ratio=1.0,
                                 liquidation_count=3, liquidation_forfeited=12.5)],
                [0],
            ),
            state,
            off_run,
        )

        def cols(run_id: int) -> tuple[Any, Any]:
            row = state.conn.execute(
                "SELECT liquidation_count, liquidation_forfeited "
                "FROM sweep_results WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            return (row[0], row[1])

        assert cols(hit_run) == (3, 12.5)
        assert cols(miss_run) == (0, 0.0)
        assert cols(off_run) == (None, None)
    finally:
        state.close()


def test_save_results_leaves_liquidation_null_for_sentinel_row(tmp_path: Path) -> None:
    """D1 (row 4): a crashed/stalled combo (-999) stores NULLs, never 0 hits.

    A sentinel row never ran the engine, so liquidation_count=0 would be a
    fake measurement: the UI must read "-", not "0 hits". The batch therefore
    MIXES a sentinel row and a real 0-hit row, the case that would otherwise
    raise a sqlite3.ProgrammingError in save_sweep_results_batch.
    """
    db = tmp_path / "d1_sentinel.db"
    state = StateManager(db)
    try:
        strategy_id = state.create_strategy(name="liq_probe", dsl_config=_STRATEGY)
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2024-03-01",
            parameters={"liquidation_mode": "isolated", "leverage": 50.0},
        )

        ScreeningPipeline(
            dsl=_dsl(), liquidation=LiquidationSettings(mode="isolated", leverage=50.0)
        ).save_results(
            ScreeningResult(
                "liq_probe", 2, 1, 0.1,
                [
                    BacktestMetrics(
                        parameters={"a": 1},
                        sharpe_ratio=FAILED_BACKTEST_SHARPE,
                        liquidation_count=3,
                        liquidation_forfeited=12.5,
                    ),
                    BacktestMetrics(
                        parameters={"a": 2},
                        sharpe_ratio=1.0,
                        liquidation_count=0,
                        liquidation_forfeited=0.0,
                    ),
                ],
                [1],
            ),
            state,
            run_id,
        )

        cols_by_sharpe = {
            row[0]: (row[1], row[2])
            for row in state.conn.execute(
                "SELECT sharpe_ratio, liquidation_count, liquidation_forfeited "
                "FROM sweep_results WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        }
        assert cols_by_sharpe[FAILED_BACKTEST_SHARPE] == (None, None)
        assert cols_by_sharpe[1.0] == (0, 0.0)
    finally:
        state.close()


# ---------------------------------------------------------------------------
# Plumbing: the factory derives settings once and hands them to both sinks
# ---------------------------------------------------------------------------


def test_factory_passes_liquidation_to_runner_and_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibe_quant.screening import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "resolve_parent_fill_ticks", lambda *a, **k: None)
    on = create_screening_pipeline(
        _dsl(),
        symbols=["BTCUSDT"],
        parameters={"liquidation_mode": "isolated", "leverage": 20.0},
    )
    assert on._runner._liquidation_mode == "isolated"
    assert on._runner._leverage == 20.0
    assert on._liquidation.on
    assert on._liquidation.leverage == 20.0

    off = create_screening_pipeline(_dsl(), symbols=["BTCUSDT"])
    assert off._runner._liquidation_mode == "off"
    assert not off._liquidation.on
    assert off._liquidation.to_params() == {}


# ---------------------------------------------------------------------------
# E1/E3: sweep_results.exit_timing is written only when the mode is on and
# round-trips from the DB JSON through the API unchanged
# ---------------------------------------------------------------------------


def _capture_batch(monkeypatch: pytest.MonkeyPatch) -> dict[int, list[dict[str, Any]]]:
    """Record the exact row dicts ``save_results`` hands to the batch writer."""
    captured: dict[int, list[dict[str, Any]]] = {}
    real = StateManager.save_sweep_results_batch

    def spy(self: StateManager, run_id: int, results: Any) -> None:
        captured[run_id] = [dict(r) for r in results]
        return real(self, run_id, results)

    monkeypatch.setattr(StateManager, "save_sweep_results_batch", spy)
    return captured


async def test_save_results_writes_exit_timing_only_when_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E1: mode on -> the row carries the notes JSON; mode off -> no key, API null.

    Mode off never sets ``metrics.exit_timing`` (None) and never adds the key, so
    the batch insert leaves ``sweep_results.exit_timing`` NULL and the API returns
    ``exit_timing: null``.
    """
    db = tmp_path / "e1.db"
    state = StateManager(db)
    captured = _capture_batch(monkeypatch)
    notes = {"refined": 0, "kept_bar_close": 0, "by_symbol": {}}
    try:
        strategy_id = state.create_strategy(name="liq_probe", dsl_config=_STRATEGY)

        def new_run(parameters: dict[str, object]) -> int:
            return state.create_backtest_run(
                strategy_id=strategy_id,
                run_mode="screening",
                symbols=["BTCUSDT"],
                timeframe="4h",
                start_date="2024-01-01",
                end_date="2024-03-01",
                parameters=parameters,
            )

        on_run = new_run({"liquidation_mode": "isolated", "leverage": 50.0})
        off_run = new_run({})

        ScreeningPipeline(
            dsl=_dsl(), liquidation=LiquidationSettings(mode="isolated", leverage=50.0)
        ).save_results(
            ScreeningResult(
                "liq_probe", 1, 1, 0.1,
                [BacktestMetrics(parameters={"a": 1}, sharpe_ratio=1.0, exit_timing=notes)],
                [0],
            ),
            state,
            on_run,
        )
        ScreeningPipeline(dsl=_dsl()).save_results(
            ScreeningResult(
                "liq_probe", 1, 1, 0.1,
                [BacktestMetrics(parameters={"a": 1}, sharpe_ratio=1.0)],
                [0],
            ),
            state,
            off_run,
        )

        off_column = state.conn.execute(
            "SELECT exit_timing FROM sweep_results WHERE run_id = ?", (off_run,)
        ).fetchone()[0]

        app = create_app()
        app.state.state_manager = state
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            api_off = (await client.get(f"/api/results/runs/{off_run}/sweeps")).json()
    finally:
        state.close()

    # mode on: key present, JSON text, single deterministic key set
    assert captured[on_run][0]["exit_timing"] == json.dumps(notes, sort_keys=True)
    # mode off: the key is absent, so the batch column stays NULL
    assert "exit_timing" not in captured[off_run][0]
    assert off_column is None
    assert api_off[0]["exit_timing"] is None


async def test_exit_timing_round_trips_db_and_api(tmp_path: Path) -> None:
    """E3: the DB JSON decodes to the notes dict and the API returns the same object."""
    db = tmp_path / "e3.db"
    state = StateManager(db)
    notes = {
        "refined": 18,
        "kept_bar_close": 2,
        "by_symbol": {
            "BTCUSDT": {"refined": 12, "kept_bar_close": 1},
            "ETHUSDT": {"refined": 6, "kept_bar_close": 1},
        },
    }
    try:
        strategy_id = state.create_strategy(name="liq_probe", dsl_config=_STRATEGY)
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2024-03-01",
            parameters={"liquidation_mode": "isolated", "leverage": 50.0},
        )
        ScreeningPipeline(
            dsl=_dsl(), liquidation=LiquidationSettings(mode="isolated", leverage=50.0)
        ).save_results(
            ScreeningResult(
                "liq_probe", 1, 1, 0.1,
                [BacktestMetrics(parameters={"a": 1}, sharpe_ratio=1.0, exit_timing=notes)],
                [0],
            ),
            state,
            run_id,
        )

        rows = state.get_sweep_results(run_id)

        app = create_app()
        app.state.state_manager = state
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            api = (await client.get(f"/api/results/runs/{run_id}/sweeps")).json()
    finally:
        state.close()

    assert rows[0]["exit_timing"] == notes
    assert api[0]["exit_timing"] == notes


def test_save_results_leaves_exit_timing_null_for_sentinel_row(tmp_path: Path) -> None:
    """E4: a -999 sentinel row stores NULL exit_timing (it never ran the engine), while
    the real row stores its JSON -- both rows set the key, so one batch has one key set."""
    db = tmp_path / "e4.db"
    state = StateManager(db)
    notes = {
        "refined": 1,
        "kept_bar_close": 0,
        "by_symbol": {"BTCUSDT": {"refined": 1, "kept_bar_close": 0}},
    }
    try:
        strategy_id = state.create_strategy(name="liq_probe", dsl_config=_STRATEGY)
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2024-03-01",
            parameters={"liquidation_mode": "isolated", "leverage": 50.0},
        )
        ScreeningPipeline(
            dsl=_dsl(), liquidation=LiquidationSettings(mode="isolated", leverage=50.0)
        ).save_results(
            ScreeningResult(
                "liq_probe", 2, 1, 0.1,
                [
                    BacktestMetrics(
                        parameters={"a": 1},
                        sharpe_ratio=FAILED_BACKTEST_SHARPE,
                        exit_timing=None,
                    ),
                    BacktestMetrics(parameters={"a": 2}, sharpe_ratio=1.0, exit_timing=notes),
                ],
                [1],
            ),
            state,
            run_id,
        )

        by_sharpe = {
            row[0]: row[1]
            for row in state.conn.execute(
                "SELECT sharpe_ratio, exit_timing FROM sweep_results WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        }
    finally:
        state.close()

    assert by_sharpe[FAILED_BACKTEST_SHARPE] is None
    assert json.loads(by_sharpe[1.0]) == notes


def test_parallel_pipeline_keeps_exit_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E8: metrics are pickled back from the worker processes (runner attributes are
    not), so a mode-on PARALLEL sweep persists the exit-timing counts."""
    db = tmp_path / "e8.db"
    run_id = _make_run(db, {"liquidation_mode": "isolated"})
    _install_fake_runner(monkeypatch, _ExitTimingRunner, workers=2)

    assert _run_cli(db, run_id) == 0

    final = _final_run(db, run_id)
    assert final["run"]["status"] == "completed"
    assert len(final["rows"]) == 1
    assert final["rows"][0]["exit_timing"] == _ExitTimingRunner.NOTES
