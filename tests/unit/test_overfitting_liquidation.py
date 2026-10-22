"""Overfitting runners + auto_screen liquidation plumbing (T5 task B3).

Covers contract rows E13 (auto_screen stays off, no liquidation-package import,
no params key), E14 (WFA/CV runners read the screening run's ``parameters`` into
a :class:`LiquidationSettings` and forward it to ``NTScreeningRunner``) and I11
(the overfitting WFA filter lets a :class:`LiquidationIntegrityError` propagate
instead of swallowing it as ``passed_wfa=False``; mode off keeps that swallow).
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import date
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.errors import DataUnavailableError, LiquidationIntegrityError
from vibe_quant.overfitting.types import FilterConfig
from vibe_quant.overfitting.wfa import WFAConfig

if TYPE_CHECKING:
    from pathlib import Path

    import pytest as _pytest


# ---------------------------------------------------------------------------
# Runner plumbing (E14)
# ---------------------------------------------------------------------------


def _seed_run(tmp_path: Path, *, parameters: dict[str, object]) -> Path:
    """Minimal state DB with a strategy + run + its stored ``parameters``.

    Also carries the recorded-balance tables the WFA/CV runners read through
    ``launch_settings.parent_effective_balance`` (R11): run 42 has no sweep rows
    and one recorded ``backtest_results.starting_balance`` of 1000.0, so these
    reruns run at 1000 — exactly the default these tests were written against.
    """
    db = tmp_path / "state.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dsl_config TEXT NOT NULL
        );
        CREATE TABLE backtest_runs (
            id INTEGER PRIMARY KEY,
            strategy_id INTEGER,
            symbols TEXT NOT NULL,
            timeframe TEXT,
            parameters JSON NOT NULL
        );
        CREATE TABLE sweep_results (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            starting_balance REAL
        );
        CREATE TABLE backtest_results (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            starting_balance REAL
        );
        """
    )
    dsl = {"name": "stub", "timeframe": "4h", "indicators": {}}
    conn.execute(
        "INSERT INTO strategies (id, name, dsl_config) VALUES (?,?,?)",
        (1, "stub", json.dumps(dsl)),
    )
    conn.execute(
        "INSERT INTO backtest_runs (id, strategy_id, symbols, timeframe, parameters) "
        "VALUES (?,?,?,?,?)",
        (42, 1, json.dumps(["BTCUSDT"]), "4h", json.dumps(parameters)),
    )
    conn.execute(
        "INSERT INTO backtest_results (run_id, starting_balance) VALUES (?,?)", (42, 1000.0)
    )
    conn.commit()
    conn.close()
    return db


class _NoTicks:
    """``ParentFillTicks`` stand-in: no tick set for any window."""

    def get(self, _start: str, _end: str) -> None:
        return None


class _CaptureRunner:
    """``NTScreeningRunner`` stand-in capturing the kwargs it was built with."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def __call__(self, _params: dict[str, object]) -> object:
        return type("M", (), {"sharpe_ratio": 1.0, "total_return": 0.1})()


@pytest.fixture
def capture_runner(monkeypatch: _pytest.MonkeyPatch) -> list[dict[str, object]]:
    import vibe_quant.screening.nt_runner as ntr
    import vibe_quant.screening.pipeline as sp

    captured: list[dict[str, object]] = []

    def _build(**kwargs: object) -> _CaptureRunner:
        runner = _CaptureRunner(**kwargs)
        captured.append(runner.kwargs)
        return runner

    monkeypatch.setattr(ntr, "NTScreeningRunner", _build)
    monkeypatch.setattr(sp, "ParentFillTicks", lambda *a, **k: _NoTicks())
    return captured


def test_wfa_runner_reads_liquidation_settings(
    tmp_path: Path, capture_runner: list[dict[str, object]]
) -> None:
    """E14: WFA builds from_params(screening run parameters) and forwards it."""
    from vibe_quant.overfitting.nt_runner import NTWFARunner

    db = _seed_run(tmp_path, parameters={"liquidation_mode": "isolated", "leverage": 20})
    runner = NTWFARunner(run_id=42, db_path=db)
    runner.backtest("x", date(2024, 2, 1), date(2024, 3, 1), {})

    assert capture_runner[0]["liquidation_mode"] == "isolated"
    assert capture_runner[0]["leverage"] == 20.0


def test_wfa_runner_mode_off_defaults(
    tmp_path: Path, capture_runner: list[dict[str, object]]
) -> None:
    """E14/P1: a run row without the key stays off at the default leverage."""
    from vibe_quant.overfitting.nt_runner import NTWFARunner

    db = _seed_run(tmp_path, parameters={})
    runner = NTWFARunner(run_id=42, db_path=db)
    runner.backtest("x", date(2024, 2, 1), date(2024, 3, 1), {})

    assert capture_runner[0]["liquidation_mode"] == "off"
    assert capture_runner[0]["leverage"] == 10.0


def test_cv_runner_reads_liquidation_settings(
    tmp_path: Path, capture_runner: list[dict[str, object]], monkeypatch: _pytest.MonkeyPatch
) -> None:
    """E14: the CV runner reads the run parameters the same way."""
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner

    db = _seed_run(tmp_path, parameters={"liquidation_mode": "cross", "leverage": 50})
    monkeypatch.setattr(NTPurgedKFoldRunner, "_load_bar_timestamps", lambda _self: None)
    runner = NTPurgedKFoldRunner(run_id=42, db_path=db, catalog_path=tmp_path / "catalog")
    runner._backtest("2024-02-01", "2024-03-01")  # noqa: SLF001

    assert capture_runner[0]["liquidation_mode"] == "cross"
    assert capture_runner[0]["leverage"] == 50.0


def test_cv_runner_mode_off_defaults(
    tmp_path: Path, capture_runner: list[dict[str, object]], monkeypatch: _pytest.MonkeyPatch
) -> None:
    """E14/P1: CV off keeps the runner's defaults byte-identical."""
    from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner

    db = _seed_run(tmp_path, parameters={})
    monkeypatch.setattr(NTPurgedKFoldRunner, "_load_bar_timestamps", lambda _self: None)
    runner = NTPurgedKFoldRunner(run_id=42, db_path=db, catalog_path=tmp_path / "catalog")
    runner._backtest("2024-02-01", "2024-03-01")  # noqa: SLF001

    assert capture_runner[0]["liquidation_mode"] == "off"
    assert capture_runner[0]["leverage"] == 10.0


# ---------------------------------------------------------------------------
# I11: the WFA filter must not swallow a liquidation integrity failure
# ---------------------------------------------------------------------------


def _seed_pipeline_run(
    tmp_path: Path, parameters: dict[str, object] | None = None
) -> tuple[Path, int]:
    """State DB with one screening run and two sweep candidates for the pipeline."""
    db = tmp_path / "t.db"
    sm = StateManager(db)
    sid = sm.create_strategy("s", {"timeframe": "4h", "indicators": {}})
    rid = sm.create_backtest_run(
        sid, "screening", ["BTCUSDT"], "4h", "2022-01-01", "2024-12-31",
        parameters or {"liquidation_mode": "isolated", "leverage": 20},
    )
    sm.save_sweep_results_batch(
        rid,
        [
            {"parameters": {"rsi_period": 14}, "sharpe_ratio": 2.0, "total_return": 0.3,
             "total_trades": 100},
            {"parameters": {"rsi_period": 2}, "sharpe_ratio": 1.9, "total_return": 0.3,
             "total_trades": 100},
        ],
    )
    sm.close()
    return db, rid


def _wfa_only_config() -> FilterConfig:
    return FilterConfig(
        enable_dsr=False,
        enable_wfa=True,
        enable_purged_kfold=False,
        wfa_config=WFAConfig(
            in_sample_days=30, out_of_sample_days=10, step_days=5, min_windows=1
        ),
    )


class _RaisingRunner:
    """BacktestRunner whose every call raises ``exc`` (no real NT backtest)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def optimize(self, *_args: object, **_kwargs: object) -> object:
        raise self._exc

    def backtest(self, *_args: object, **_kwargs: object) -> object:
        raise self._exc


def test_wfa_integrity_error_propagates(tmp_path: Path) -> None:
    """I11: mode on -> the WFA filter re-raises; the CLI then exits non-zero."""
    from vibe_quant.overfitting.pipeline import OverfittingPipeline

    exc = LiquidationIntegrityError(
        "liquidation_mode='isolated': MarkDataMissing: BTCUSDT"
    )
    db, rid = _seed_pipeline_run(tmp_path)
    pipeline = OverfittingPipeline(db, wfa_runner=_RaisingRunner(exc))
    try:
        with pytest.raises(LiquidationIntegrityError, match="MarkDataMissing"):
            pipeline.run(rid, config=_wfa_only_config())
    finally:
        pipeline.close()


def test_wfa_value_error_still_marked_passed_wfa_false(tmp_path: Path) -> None:
    """I11 (mode off unchanged): a plain ValueError is still swallowed to False."""
    from vibe_quant.overfitting.pipeline import OverfittingPipeline

    db, rid = _seed_pipeline_run(tmp_path)
    pipeline = OverfittingPipeline(db, wfa_runner=_RaisingRunner(ValueError("boom")))
    try:
        result = pipeline.run(rid, config=_wfa_only_config())
    finally:
        pipeline.close()

    assert result.passed_wfa == 0
    assert result.candidates
    assert all(c.passed_wfa is False for c in result.candidates)


def test_wfa_data_unavailable_error_propagates_mode_off(tmp_path: Path) -> None:
    """A missing-data error fails the run loudly with no passed_wfa=False row."""
    from vibe_quant.overfitting.pipeline import OverfittingPipeline

    exc = DataUnavailableError("no 1m bars for BTCUSDT 2022-01-01..2022-06-01")
    db, rid = _seed_pipeline_run(tmp_path, parameters={"liquidation_mode": "off"})
    pipeline = OverfittingPipeline(db, wfa_runner=_RaisingRunner(exc))
    try:
        with pytest.raises(DataUnavailableError, match="no 1m bars") as caught:
            pipeline.run(rid, config=_wfa_only_config())
    finally:
        pipeline.close()

    # Identity, not just a message prefix: a re-wrap that truncates the
    # BTCUSDT/window diagnostics must fail this test.
    assert caught.value is exc

    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT passed_walk_forward FROM sweep_results").fetchall()
    conn.close()
    assert rows
    assert all(row[0] is None for row in rows)


def test_wfa_aux_data_unavailable_error_propagates(tmp_path: Path) -> None:
    """An AuxDataUnavailableError (a DataUnavailableError subclass) propagates."""
    from vibe_quant.dsl.aux_data import AuxDataUnavailableError
    from vibe_quant.overfitting.pipeline import OverfittingPipeline

    exc = AuxDataUnavailableError("aux series missing for BTCUSDT")
    db, rid = _seed_pipeline_run(tmp_path, parameters={"liquidation_mode": "off"})
    pipeline = OverfittingPipeline(db, wfa_runner=_RaisingRunner(exc))
    try:
        with pytest.raises(AuxDataUnavailableError, match="aux series missing") as caught:
            pipeline.run(rid, config=_wfa_only_config())
    finally:
        pipeline.close()

    # Identity, not just a message prefix (the subclass must not be re-wrapped).
    assert caught.value is exc


@pytest.mark.parametrize(("is_robust", "expected"), [(True, 2), (False, 0)])
def test_wfa_result_is_robust_maps_to_passed_wfa(
    tmp_path: Path,
    monkeypatch: _pytest.MonkeyPatch,
    is_robust: bool,
    expected: int,
) -> None:
    """Row 5: passed_wfa follows wfa_result.is_robust, unchanged."""
    from vibe_quant.overfitting import wfa as wfa_mod
    from vibe_quant.overfitting.pipeline import OverfittingPipeline

    db, rid = _seed_pipeline_run(tmp_path, parameters={"liquidation_mode": "off"})

    def _fake_run(
        self: wfa_mod.WalkForwardAnalysis,
        strategy_id: str,
        data_start: date,
        data_end: date,
        param_grid: dict[str, object],
    ) -> wfa_mod.WFAResult:
        return wfa_mod.WFAResult(
            windows=(),
            aggregated_oos_sharpe=0.0,
            aggregated_oos_return=0.0,
            efficiency=0.0,
            is_vs_oos_degradation=0.0,
            consistency_ratio=0.0,
            is_robust=is_robust,
            config=self.config,
        )

    monkeypatch.setattr(wfa_mod.WalkForwardAnalysis, "run", _fake_run)
    pipeline = OverfittingPipeline(db, wfa_runner=_RaisingRunner(ValueError("unused")))
    try:
        result = pipeline.run(rid, config=_wfa_only_config())
    finally:
        pipeline.close()

    assert result.passed_wfa == expected
    assert all(c.passed_wfa is is_robust for c in result.candidates)


def test_cli_run_real_wfa_propagates_integrity_error(
    tmp_path: Path, monkeypatch: _pytest.MonkeyPatch
) -> None:
    """I11: the ``run --real-wfa`` CLI path must let the error escape main().

    Drives the real entrypoint: monkeypatched NTWFARunner raises, and the
    exception must surface out of ``main()`` (non-zero) rather than be
    swallowed by a catch-all around ``cmd_run``.
    """
    import vibe_quant.overfitting.__main__ as cli
    import vibe_quant.overfitting.nt_runner as ntr

    exc = LiquidationIntegrityError(
        "liquidation_mode='isolated': MarkDataMissing: BTCUSDT"
    )
    db, rid = _seed_pipeline_run(tmp_path)
    monkeypatch.setattr(ntr, "NTWFARunner", lambda *a, **k: _RaisingRunner(exc))

    with pytest.raises(LiquidationIntegrityError, match="liquidation_mode='isolated'"):
        cli.main(
            [
                "run",
                "--run-id", str(rid),
                "--db", str(db),
                "--filters", "wfa",
                "--real-wfa",
                "--wfa-is-days", "30",
                "--wfa-oos-days", "10",
                "--wfa-step-days", "5",
                "--wfa-min-windows", "1",
            ]
        )


# ---------------------------------------------------------------------------
# E13: auto_screen stays off
# ---------------------------------------------------------------------------


class _AutoScreenCapture:
    def __init__(self, captured: dict[str, object]) -> None:
        self._captured = captured

    def __call__(self, **kwargs: object) -> object:
        self._captured["runner_kwargs"] = kwargs
        captured = self._captured

        def _run(params: dict[str, object]) -> object:
            captured["params"] = params
            return None

        return _run


def test_auto_screen_runner_stays_off_and_params_have_no_key(
    monkeypatch: _pytest.MonkeyPatch,
) -> None:
    """E13: auto_screen passes mode off explicitly and no liquidation param key."""
    import vibe_quant.research.auto_screen as auto
    import vibe_quant.screening.nt_runner as ntr
    import vibe_quant.screening.pipeline as sp

    captured: dict[str, object] = {}
    monkeypatch.setattr(ntr, "NTScreeningRunner", _AutoScreenCapture(captured))
    monkeypatch.setattr(sp, "resolve_parent_fill_ticks", lambda *a, **k: None)

    auto._run_single_metrics({"name": "x"}, "2024-01-01", "2024-03-01")  # noqa: SLF001

    assert captured["runner_kwargs"]["liquidation_mode"] == "off"
    assert "liquidation_mode" not in captured["params"]


def test_auto_screen_imports_no_liquidation_package() -> None:
    """E13/X1: importing auto_screen must not pull in the liquidation package."""
    code = (
        "import sys, vibe_quant.research.auto_screen; "
        "bad = sorted(m for m in sys.modules "
        "if m == 'vibe_quant.liquidation' or m.startswith('vibe_quant.liquidation.')); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stdout + out.stderr
