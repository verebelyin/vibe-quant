"""The discovery engine adapters and the state-backed record store.

Edge tests of the ``BacktestAdapter`` / ``RecordStore`` seams (bead
vibe-quant-rabmo.7, step T14): ``NTBacktests``/``MockBacktests`` build every cell
function from THIS module's globals (so a patched ``run.NTBacktestFn`` reaches
the train, holdout and factory functions), and ``StateRecordStore`` writes the
same rows ``main()`` writes today.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from vibe_quant.db.state_manager import StateManager
from vibe_quant.discovery import run
from vibe_quant.jobs.manager import BacktestJobManager

if TYPE_CHECKING:
    from pathlib import Path


def _cell_kwargs(**over: object) -> dict[str, object]:
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.sizing_settings import SizingSettings

    kwargs: dict[str, object] = {
        "symbols": ["BTCUSDT"],
        "timeframe": "4h",
        "symbol_agg": "portfolio",
        "fee_schedule": FeeSchedule(),
        "entry_order": "market",
        "liquidation": LiquidationSettings(),
        "sizing": SizingSettings(),
        "train_start": "2024-01-01",
        "train_end": "2024-06-01",
        "eval_windows": None,
        "min_trades": 0,
        "holdout": ("2024-06-01", "2024-08-01"),
        "needs_factory": True,
        "evaluator": "nt",
    }
    kwargs.update(over)
    return kwargs


def _nt_fn_kwargs() -> dict[str, object]:
    from vibe_quant.fees import FeeSchedule
    from vibe_quant.liquidation_settings import LiquidationSettings
    from vibe_quant.sizing_settings import SizingSettings

    return {
        "symbols": ["BTCUSDT"],
        "timeframe": "4h",
        "start_date": "2024-01-01",
        "end_date": "2024-06-01",
        "symbol_agg": "portfolio",
        "fee_schedule": FeeSchedule(),
        "entry_order": "market",
        "liquidation": LiquidationSettings(),
        "sizing": SizingSettings(),
    }


class _FakeNT:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs


def test_nt_backtestfn_patch_reaches_every_cell_fn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run.NTBacktestFn patched to a fake is what train, holdout and factory build."""
    monkeypatch.setattr(run, "NTBacktestFn", _FakeNT)

    adapter = run.NTBacktests()
    fns = adapter.cell_fns(**_cell_kwargs())

    assert isinstance(fns.backtest_fn, _FakeNT)
    assert isinstance(fns.holdout_backtest_fn, _FakeNT)
    assert fns.backtest_fn_factory is not None
    assert isinstance(fns.backtest_fn_factory("2024-01-01", "2024-03-01"), _FakeNT)
    assert isinstance(adapter.nt_fn(**_nt_fn_kwargs()), _FakeNT)
    assert adapter.mock is False


@pytest.mark.parametrize(("max_position_pct", "risk_per_trade"), [(0.15, 0.01), (0.4, 0.03)])
def test_nt_backtests_forward_the_run_sizing(
    monkeypatch: pytest.MonkeyPatch, max_position_pct: float, risk_per_trade: float
) -> None:
    """Every engine fn the NT adapter builds carries the run's sizing, not the defaults."""
    from vibe_quant.sizing_settings import SizingSettings

    monkeypatch.setattr(run, "NTBacktestFn", _FakeNT)
    sizing = SizingSettings(max_position_pct=max_position_pct, risk_per_trade=risk_per_trade)
    adapter = run.NTBacktests()
    fns = adapter.cell_fns(**_cell_kwargs(sizing=sizing))
    assert fns.backtest_fn_factory is not None
    nt_kwargs = {**_nt_fn_kwargs(), "sizing": sizing}

    built = [
        fns.backtest_fn,
        fns.holdout_backtest_fn,
        fns.backtest_fn_factory("2024-01-01", "2024-03-01"),
        adapter.nt_fn(**nt_kwargs),
    ]

    monkeypatch.setattr("vibe_quant.fast_tier.backtest_fn.FastBacktestFn", _FakeNT)
    built.append(adapter.cell_fns(**_cell_kwargs(sizing=sizing, evaluator="fast")).backtest_fn)

    for fn in built:
        assert isinstance(fn, _FakeNT)
        got = fn.kwargs["sizing"]
        assert (got.max_position_pct, got.risk_per_trade) == (max_position_pct, risk_per_trade)


def test_mock_backtests_hand_back_the_mock_backtest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mock adapter's cell fns are the module's _mock_backtest, not the engine."""
    sentinel = object()
    monkeypatch.setattr(run, "_mock_backtest", sentinel)

    adapter = run.MockBacktests()
    fns = adapter.cell_fns(**_cell_kwargs())

    assert adapter.mock is True
    assert fns.backtest_fn is sentinel
    assert fns.holdout_backtest_fn is sentinel
    assert fns.backtest_fn_factory is not None
    assert fns.backtest_fn_factory("2024-01-01", "2024-03-01") is sentinel
    with pytest.raises(RuntimeError, match="mock adapter runs no NT backtest"):
        adapter.nt_fn(**_nt_fn_kwargs())


def _seeded_run(db: Path) -> tuple[StateManager, BacktestJobManager, int]:
    state = StateManager(db)
    jobs = BacktestJobManager(db)
    run_id = state.create_backtest_run(
        None, "discovery", ["BTCUSDT"], "4h", "2024-01-01", "2024-06-01", {}
    )
    jobs.conn.execute(
        """INSERT INTO background_jobs (run_id, pid, job_type, status)
           VALUES (?, ?, 'discovery', 'running')""",
        (run_id, os.getpid()),
    )
    jobs.conn.commit()
    return state, jobs, run_id


def _job_row(state: StateManager, run_id: int) -> tuple[str, str | None]:
    row = state.conn.execute(
        "SELECT status, error_message FROM background_jobs WHERE run_id = ?", (run_id,)
    ).fetchone()
    return (row[0], row[1])


def test_mark_completed_records_both_rows(tmp_path: Path) -> None:
    state, jobs, run_id = _seeded_run(tmp_path / "state.db")
    try:
        run.StateRecordStore(state, jobs).mark_completed(run_id)

        row = state.get_backtest_run(run_id)
        assert row is not None
        assert row["status"] == "completed"
        assert _job_row(state, run_id) == ("completed", None)
    finally:
        state.close()
        jobs.close()


def test_mark_failed_records_the_error_on_both_rows(tmp_path: Path) -> None:
    state, jobs, run_id = _seeded_run(tmp_path / "state.db")
    try:
        run.StateRecordStore(state, jobs).mark_failed(run_id, "boom")

        row = state.get_backtest_run(run_id)
        assert row is not None
        assert row["status"] == "failed"
        assert row["error_message"] == "boom"
        assert _job_row(state, run_id) == ("failed", "boom")
    finally:
        state.close()
        jobs.close()


def test_mark_job_failed_on_an_unknown_run_touches_only_the_job(tmp_path: Path) -> None:
    """The run-not-found branch: no run row exists, so only the job is recorded."""
    db = tmp_path / "state.db"
    state = StateManager(db)
    jobs = BacktestJobManager(db)
    missing_run_id = 4242
    jobs.conn.execute(
        """INSERT INTO background_jobs (run_id, pid, job_type, status)
           VALUES (?, ?, 'discovery', 'running')""",
        (missing_run_id, os.getpid()),
    )
    jobs.conn.commit()
    try:
        store = run.StateRecordStore(state, jobs)
        assert store.get_run(missing_run_id) is None

        store.mark_job_failed(missing_run_id, "jobboom")

        assert store.get_run(missing_run_id) is None
        assert _job_row(state, missing_run_id) == ("failed", "jobboom")
    finally:
        state.close()
        jobs.close()


@pytest.mark.parametrize(
    ("metrics", "expected_sharpe", "expected_return"),
    [
        ({"sharpe_ratio": 1.25, "total_return": 0.10, "notes": "{}"}, 1.25, 0.10),
        ({"sharpe_ratio": -0.75, "total_return": -0.30, "notes": "{}"}, -0.75, -0.30),
    ],
)
def test_get_run_and_save_result_round_trip(
    tmp_path: Path,
    metrics: dict[str, object],
    expected_sharpe: float,
    expected_return: float,
) -> None:
    state, jobs, run_id = _seeded_run(tmp_path / "state.db")
    try:
        store = run.StateRecordStore(state, jobs)

        row = store.get_run(run_id)
        assert row is not None
        assert row["id"] == run_id

        store.save_result(run_id, metrics)

        stored = state.get_backtest_result(run_id)
        assert stored is not None
        assert stored["sharpe_ratio"] == expected_sharpe
        assert stored["total_return"] == expected_return
    finally:
        state.close()
        jobs.close()
