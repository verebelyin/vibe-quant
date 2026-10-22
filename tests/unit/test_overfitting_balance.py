"""Overfitting reruns use the parent's RECORDED balance, never its parameters.

Bead vibe-quant-zpbmo, task Z2. Binding contract rows: R11
(:func:`vibe_quant.launch_settings.parent_effective_balance`: the sweep rows'
single distinct value, else ``backtest_results.starting_balance``, both through
``recorded_balance``; a conflict or no record at all raises) and R11b (a legacy
parent whose ``parameters`` say ``{"initial_balance": 10000}`` while every sweep
row is NULL reruns at 1000, on both the WFA and the CV runner).
"""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.launch_settings import parent_effective_balance
from vibe_quant.overfitting.nt_cv_runner import NTPurgedKFoldRunner
from vibe_quant.overfitting.nt_runner import NTWFARunner

if TYPE_CHECKING:
    from pathlib import Path

_RUN_ID = 1


def _seed(
    tmp_path: Path,
    name: str,
    *,
    parameters: dict[str, object] | None = None,
    sweep: list[dict[str, Any]] | None = None,
    result: dict[str, Any] | None = None,
) -> tuple[Path, int]:
    """A state DB with one screening run plus the rows the case needs."""
    db = tmp_path / f"{name}.db"
    state = StateManager(db)
    try:
        strategy_id = state.create_strategy("stub", {"timeframe": "4h", "indicators": {}})
        run_id = state.create_backtest_run(
            strategy_id=strategy_id,
            run_mode="screening",
            symbols=["BTCUSDT"],
            timeframe="4h",
            start_date="2024-01-01",
            end_date="2024-03-01",
            parameters=parameters or {},
        )
        if sweep is not None:
            state.save_sweep_results_batch(run_id, sweep)
        if result is not None:
            state.save_backtest_result(run_id, result)
    finally:
        state.close()
    return db, run_id


def _effective(db: Path, run_id: int) -> int:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return parent_effective_balance(conn, run_id)
    finally:
        conn.close()


def _sweep_row(starting_balance: object = None, sharpe: float = 1.0) -> dict[str, Any]:
    row: dict[str, Any] = {"parameters": {"rsi.period": 14}, "sharpe_ratio": sharpe}
    if starting_balance is not None:
        row["starting_balance"] = starting_balance
    return row


# --------------------------------------------------------------- R11: the rule


def test_single_sweep_value_is_the_effective_balance(tmp_path: Path) -> None:
    """R11: every sweep row at 10000.0 -> the rerun runs at 10000."""
    db, run_id = _seed(tmp_path, "sweep_value", sweep=[_sweep_row(10000.0), _sweep_row(10000.0, 2.0)])
    assert _effective(db, run_id) == 10000


def test_null_sweep_rows_mean_the_default(tmp_path: Path) -> None:
    """R11/R11b: NULL sweep rows (a legacy run) record no balance -> 1000."""
    db, run_id = _seed(
        tmp_path, "sweep_null", parameters={"initial_balance": 10000}, sweep=[_sweep_row(), _sweep_row()]
    )
    assert _effective(db, run_id) == 1000


def test_conflicting_sweep_values_raise(tmp_path: Path) -> None:
    """R11: two distinct recorded values are an inconsistent run -> ValueError."""
    db, run_id = _seed(tmp_path, "sweep_conflict", sweep=[_sweep_row(10000.0), _sweep_row(5000.0)])
    with pytest.raises(ValueError, match="conflicting sweep starting_balance"):
        _effective(db, run_id)


def test_backtest_result_is_the_fallback(tmp_path: Path) -> None:
    """R11: no sweep rows at all -> ``backtest_results.starting_balance``, same rule."""
    db, run_id = _seed(tmp_path, "result", result={"starting_balance": 10000.0})
    assert _effective(db, run_id) == 10000


def test_null_backtest_result_means_the_default(tmp_path: Path) -> None:
    """R11: a validation parent with a NULL recorded balance -> 1000."""
    db, run_id = _seed(tmp_path, "result_null", result={})
    assert _effective(db, run_id) == 1000


def test_no_recorded_balance_at_all_raises(tmp_path: Path) -> None:
    """R11: neither sweep rows nor a backtest result -> the rerun fails closed."""
    db, run_id = _seed(tmp_path, "nothing")
    with pytest.raises(ValueError, match=r"has no recorded starting balance"):
        _effective(db, run_id)


# ------------------------------------------------- R11b: the legacy regression


class _Capture:
    """``NTScreeningRunner`` stand-in capturing the kwargs it was built with."""

    def __init__(self, **kwargs: object) -> None:
        captures.append(kwargs)

    def __call__(self, _params: dict[str, object]) -> object:
        return type("M", (), {"sharpe_ratio": 1.0, "total_return": 0.1})()


captures: list[dict[str, object]] = []


@pytest.fixture(autouse=True)
def _clear_captures() -> None:
    captures.clear()


def _install_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    import vibe_quant.screening.nt_runner as ntr
    import vibe_quant.screening.pipeline as sp

    monkeypatch.setattr(ntr, "NTScreeningRunner", _Capture)
    monkeypatch.setattr(sp, "ParentFillTicks", lambda *a, **k: _NoTicks())


class _NoTicks:
    def get(self, _start: str, _end: str) -> None:
        return None


def test_legacy_wfa_rerun_uses_1000_not_the_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R11b: params say 10000, sweep rows are NULL -> the WFA rerun builds 1000."""
    _install_capture(monkeypatch)
    db, run_id = _seed(
        tmp_path, "legacy_wfa", parameters={"initial_balance": 10000}, sweep=[_sweep_row(), _sweep_row()]
    )
    NTWFARunner(run_id=run_id, db_path=db).backtest("x", date(2024, 2, 1), date(2024, 3, 1), {})
    assert [c["starting_balance"] for c in captures] == [1000]


def test_legacy_cv_rerun_uses_1000_not_the_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R11b: same legacy parent -> the CV rerun builds 1000 on every fold backtest."""
    _install_capture(monkeypatch)
    monkeypatch.setattr(NTPurgedKFoldRunner, "_load_bar_timestamps", lambda _self: None)
    db, run_id = _seed(
        tmp_path, "legacy_cv", parameters={"initial_balance": 10000}, sweep=[_sweep_row(), _sweep_row()]
    )
    runner = NTPurgedKFoldRunner(run_id=run_id, db_path=db, catalog_path=tmp_path / "catalog")
    runner._backtest("2024-02-01", "2024-03-01")  # noqa: SLF001
    assert [c["starting_balance"] for c in captures] == [1000]


def test_wfa_rerun_follows_a_recorded_10000(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R11: a run whose sweep rows recorded 10000 reruns at 10000, not at 1000."""
    _install_capture(monkeypatch)
    db, run_id = _seed(tmp_path, "recorded_wfa", sweep=[_sweep_row(10000.0), _sweep_row(10000.0, 2.0)])
    NTWFARunner(run_id=run_id, db_path=db).backtest("x", date(2024, 2, 1), date(2024, 3, 1), {})
    assert [c["starting_balance"] for c in captures] == [10000]


def test_cv_rerun_follows_a_recorded_10000(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R11: same for the CV runner."""
    _install_capture(monkeypatch)
    monkeypatch.setattr(NTPurgedKFoldRunner, "_load_bar_timestamps", lambda _self: None)
    db, run_id = _seed(tmp_path, "recorded_cv", sweep=[_sweep_row(10000.0), _sweep_row(10000.0, 2.0)])
    runner = NTPurgedKFoldRunner(run_id=run_id, db_path=db, catalog_path=tmp_path / "catalog")
    runner._backtest("2024-02-01", "2024-03-01")  # noqa: SLF001
    assert [c["starting_balance"] for c in captures] == [10000]
